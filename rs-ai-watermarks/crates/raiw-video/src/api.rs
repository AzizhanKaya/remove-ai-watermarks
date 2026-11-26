//! The high-level video API (port of `video.py`).

use std::path::{Path, PathBuf};

use raiw_audio::{AudioRuntime, IdentifyOptions, Scheme, WatermarkStatus, identify_audio};
use raiw_core::constants::VIDEO_EXTENSIONS;
use raiw_core::fs::{suffix_lower, with_stem_suffix};
use raiw_metadata::{AiMetadata, get_ai_metadata, strip_and_verify, unreadable_reason};
use raiw_visible::eraser::{Backend, FillBackends};

use crate::audio::{
    VideoAudioOptions, preflight_video_audio_workload, remove_video_audio_all_streams, stream_inventory,
    verify_stream_packets,
};
use crate::decode::iter_decoded_bgr_frames;
use crate::encode::atomic_video_output;
use crate::error::{Result, VideoError};
use crate::invisible::{
    FrameFeed, InvisibleMode, InvisibleOptions, NeuralBackends, VaePixel, regenerate_video_candidate,
    regenerate_video_frames, require_profile_backend, resolve_video_invisible_profile,
    validate_video_profile,
};
use crate::probe::probe_video_encode_profile;
use crate::probe::{probe_video_geometry, probe_video_timestamps};
use crate::provenance::{
    VIDEO_VISIBLE_MARKS, invisible_watermark_hints, is_bytedance_video_provenance, markers_claim_ai,
    non_contradicted_marks, platform_from_video_metadata, video_has_invisible_provenance, visible_platform,
};
use crate::report::{
    AudioWatermarkState, BatchMode, RegenerationMetrics, StreamAction, VideoAllResult, VideoAudioStatus,
    VideoBatchItem, VideoBatchResult, VideoEraseResult, VideoInvisibleResult, VideoMetadataReport,
    VideoMetadataResult, VideoProvenanceReport, VideoVisibleResult,
};
use crate::seedance::{SEEDANCE_EXPERIMENTAL_PROFILE, SeedanceQuality, regenerate_seedance_experimental};
use crate::temporal::{detect_ai_bumper_start, track_video_regions};
use crate::tools::video_tools_available;
use crate::visible::clean::CleanOptions;
use crate::visible::{
    EncodeCleanOptions, FrameLocalization, FrameRegions, MaskStyle, Region, Scans, VideoScan,
    encode_clean_video, for_each_clean_frame, merge_selected_visible_marks, probe_empty_video_scan,
    scan_video_marks, select_stable_visible_marks,
};
use crate::wan::{WanOptions, WanWindowing, profile_strength, regenerate_wan_experimental};
use raiw_core::constants::WAN_EXPERIMENTAL_PROFILES;
use raiw_cv::Mat;

const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];

/// Everything a video operation may need: fill backends, the audio runtime, neural seams.
#[derive(Debug, Default)]
pub struct VideoRuntime {
    pub fills: FillBackends,
    pub audio: AudioRuntime,
    pub neural: NeuralBackends,
}

impl VideoRuntime {
    /// Classical fills, discovered audio tools, and the neural backends on `auto`.
    pub fn discover() -> Self {
        Self::discover_on("auto").unwrap_or_else(|_| Self::classical())
    }

    /// Classical fills and audio tools without any neural backend.
    pub fn classical() -> Self {
        Self {
            fills: FillBackends::classical(),
            audio: AudioRuntime::discover(),
            neural: NeuralBackends::default(),
        }
    }

    /// The runtime for a `--device` value (`auto`, `cuda`, `mps` or `cpu`). With the `gpu`
    /// feature the VAE and message models load on that device on first use (`auto` falls back
    /// to the CPU backend); without it the neural seams stay empty.
    pub fn discover_on(device: &str) -> Result<Self> {
        if !matches!(device, "auto" | "cuda" | "mps" | "cpu") {
            return Err(VideoError::value("device must be auto, cuda, mps, or cpu"));
        }
        #[cfg(feature = "gpu")]
        let neural = crate::neural::backends(device)?;
        #[cfg(not(feature = "gpu"))]
        let neural = NeuralBackends::default();
        Ok(Self {
            neural,
            ..Self::classical()
        })
    }
}

fn expected_marks() -> String {
    let (last, head) = VIDEO_VISIBLE_MARKS
        .split_last()
        .map_or(("", &[][..]), |(l, h)| (*l, h));
    let mut parts = vec!["auto"];
    parts.extend(head.iter().copied());
    format!("{}, or {last}", parts.join(", "))
}

/// `_require_video_runtime`.
pub fn require_video_runtime() -> Result<()> {
    if video_tools_available() {
        Ok(())
    } else {
        Err(VideoError::runtime(
            "Video pixel processing requires ffmpeg and ffprobe on PATH",
        ))
    }
}

/// Whether `path` is a container the regeneration profiles accept.
pub fn is_regenerable(path: &Path) -> bool {
    VIDEO_EXTENSIONS.contains(&suffix_lower(path).as_str())
}

fn sorted_extensions() -> String {
    let mut exts: Vec<&str> = VIDEO_EXTENSIONS.to_vec();
    exts.sort_unstable();
    exts.join(", ")
}

/// `_video_source`: an existing file with a supported extension, matching signature and readable body.
pub fn video_source(source: &Path) -> Result<PathBuf> {
    if !source.exists() {
        return Err(VideoError::NotFound(format!(
            "Video does not exist: {}",
            source.display()
        )));
    }
    if !source.is_file() {
        return Err(VideoError::value(format!(
            "Video source must be a file: {}",
            source.display()
        )));
    }
    let suffix = suffix_lower(source);
    if !VIDEO_EXTENSIONS.contains(&suffix.as_str()) {
        let shown = source
            .extension()
            .map_or_else(|| "<none>".to_owned(), |e| format!(".{}", e.to_string_lossy()));
        return Err(VideoError::value(format!(
            "Unsupported video format {shown}; expected one of: {}",
            sorted_extensions()
        )));
    }
    let mut head = [0u8; 12];
    let read = {
        use std::io::Read;
        let mut file = std::fs::File::open(source)?;
        let mut total = 0;
        while total < head.len() {
            let n = file.read(&mut head[total..])?;
            if n == 0 {
                break;
            }
            total += n;
        }
        total
    };
    let head = &head[..read];
    let matches = match suffix.as_str() {
        ".mp4" | ".mov" | ".m4v" => head.len() >= 8 && &head[4..8] == b"ftyp",
        ".webm" | ".mkv" => head.starts_with(&EBML_MAGIC),
        ".avi" => head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"AVI ",
        ".flv" => head.starts_with(b"FLV"),
        _ => false,
    };
    if !matches {
        return Err(VideoError::value(format!(
            "Video content does not match its {suffix} extension: {}",
            source.display()
        )));
    }
    if let Some(reason) = unreadable_reason(source) {
        return Err(VideoError::value(format!(
            "Cannot read video {}: {reason}",
            source.display()
        )));
    }
    Ok(source.to_path_buf())
}

/// `_video_output`: a distinct same-container output, defaulting to `<stem>_clean`.
pub fn video_output(source: &Path, output: Option<&Path>, operation: &str) -> Result<PathBuf> {
    let path = output.map_or_else(|| with_stem_suffix(source, "_clean"), Path::to_path_buf);
    if suffix_lower(&path) != suffix_lower(source) {
        let source_suffix = source
            .extension()
            .map_or_else(String::new, |e| format!(".{}", e.to_string_lossy()));
        let shown = path
            .extension()
            .map_or_else(|| "<none>".to_owned(), |e| format!(".{}", e.to_string_lossy()));
        return Err(VideoError::value(format!(
            "Video output container must match the source ({source_suffix}); {operation} does not change containers to {shown}"
        )));
    }
    if raiw_core::resolve_lenient(&path) == raiw_core::resolve_lenient(source) {
        return Err(VideoError::value(format!(
            "Video {operation} requires a distinct output path"
        )));
    }
    Ok(path)
}

/// `_parse_mark_selector`: candidate marks and whether several were named explicitly.
pub fn parse_mark_selector(mark: &str) -> Result<(Vec<&'static str>, bool)> {
    let unsupported = || {
        VideoError::value(format!(
            "Unsupported visible video mark; expected {}",
            expected_marks()
        ))
    };
    let lookup = |name: &str| VIDEO_VISIBLE_MARKS.iter().copied().find(|m| *m == name);
    if mark == "auto" {
        return Ok((VIDEO_VISIBLE_MARKS.to_vec(), false));
    }
    if mark.contains(',') {
        let mut parts: Vec<&'static str> = Vec::new();
        for part in mark.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let found = lookup(part).ok_or_else(unsupported)?;
            if !parts.contains(&found) {
                parts.push(found);
            }
        }
        if parts.is_empty() {
            return Err(unsupported());
        }
        let multi = parts.len() > 1;
        return Ok((parts, multi));
    }
    Ok((vec![lookup(mark).ok_or_else(unsupported)?], false))
}

fn scan_or_probe(source: &Path, active: &[&str], collect_timestamps: bool) -> Result<(Scans, VideoScan)> {
    if active.is_empty() {
        return Ok((Vec::new(), probe_empty_video_scan(source, collect_timestamps)?));
    }
    let scans = scan_video_marks(source, active, collect_timestamps)?;
    let base = scans
        .first()
        .map(|(_, s)| s.clone())
        .ok_or_else(|| VideoError::runtime("empty scan"))?;
    Ok((scans, base))
}

/// `inspect_video_metadata`.
pub fn inspect_video_metadata(source: &Path) -> Result<VideoMetadataReport> {
    let source = video_source(source)?;
    let markers = get_ai_metadata(&source);
    Ok(VideoMetadataReport {
        has_ai_metadata: !markers.is_empty(),
        markers,
        source,
    })
}

/// `_recommend_video_recipe`.
pub fn recommend_video_recipe(
    markers: &AiMetadata,
    selected_mark: Option<&str>,
    claims_ai: bool,
    hints: &[String],
    audio_statuses: &[WatermarkStatus],
) -> Option<String> {
    let recipe = |s: &str| Some(s.to_owned());
    if is_bytedance_video_provenance(markers, selected_mark) {
        return if selected_mark.is_none() {
            recipe(
                "remove-ai-watermarks video metadata input.mp4 --remove -o clean.mp4 (try lossless C2PA strip first; add video all \
                 --invisible --profile auto only for consumer CapCut pixel marks)",
            )
        } else {
            recipe(
                "remove-ai-watermarks video all input.mp4 -o clean.mp4 (strips visible mark and C2PA; add --invisible --profile auto \
                 only for consumer CapCut pixel marks)",
            )
        };
    }
    let platform = if claims_ai {
        platform_from_video_metadata(markers)
    } else {
        None
    };
    let google = hints.iter().any(|h| h.to_lowercase().contains("synthid"))
        || platform
            .as_deref()
            .is_some_and(|p| p.to_lowercase().contains("google"))
        || selected_mark == Some("veo");
    if google {
        if audio_statuses.iter().any(|s| *s != WatermarkStatus::Detected) {
            return recipe(
                "remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile auto --audio-invisible --audio-method dac \
                 --audio-allow-unverified",
            );
        }
        if !audio_statuses.is_empty() {
            return recipe(
                "remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile auto --audio-invisible",
            );
        }
        return recipe("remove-ai-watermarks video all input.mp4 -o clean.mp4 --invisible --profile auto");
    }
    if audio_statuses.contains(&WatermarkStatus::Detected) {
        return recipe("remove-ai-watermarks video all input.mp4 -o clean.mp4 --audio-invisible");
    }
    if selected_mark.is_some() || claims_ai {
        return recipe("remove-ai-watermarks video all input.mp4 -o clean.mp4");
    }
    None
}

/// Options for [`identify_video`].
#[derive(Clone, Copy, Debug)]
pub struct IdentifyVideoOptions {
    pub check_visible: bool,
    pub check_audio: bool,
    pub audio_scheme: Scheme,
}

impl Default for IdentifyVideoOptions {
    fn default() -> Self {
        Self {
            check_visible: true,
            check_audio: false,
            audio_scheme: Scheme::Auto,
        }
    }
}

/// `identify_video`: locally readable provenance, stable visible marks, and optional audio marks.
pub fn identify_video(
    runtime: &VideoRuntime,
    source: &Path,
    options: &IdentifyVideoOptions,
) -> Result<VideoProvenanceReport> {
    let source = video_source(source)?;
    let markers = get_ai_metadata(&source);
    let mut selected_marks: Vec<String> = Vec::new();
    let mut detected_frames = 0;
    let mut total_frames = None;
    if options.check_visible {
        require_video_runtime()?;
        let active = non_contradicted_marks(&VIDEO_VISIBLE_MARKS, &markers);
        let (scans, base) = scan_or_probe(&source, &active, false)?;
        let total = base.detections.len();
        total_frames = Some(total);
        let selected = select_stable_visible_marks(&scans, &markers, &VIDEO_VISIBLE_MARKS, false);
        if !selected.is_empty() {
            selected_marks = selected.iter().map(|s| s.mark.clone()).collect();
            detected_frames = (0..total)
                .filter(|&i| {
                    selected
                        .iter()
                        .any(|s| s.regions.get(i).is_some_and(Option::is_some))
                })
                .count();
        }
    }
    let mut audio_reports = Vec::new();
    if options.check_audio {
        let count = stream_inventory(&source)?
            .iter()
            .filter(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("audio"))
            .count();
        for ordinal in 0..count {
            let request = IdentifyOptions {
                check_invisible: true,
                stream: ordinal,
                scheme: options.audio_scheme,
                metadata_markers: None,
            };
            audio_reports.push(identify_audio(&runtime.audio, &source, &request)?);
        }
    }
    let statuses: Vec<WatermarkStatus> = audio_reports.iter().map(|r| r.watermark_status).collect();
    let selected_mark = selected_marks.first().cloned();
    let claims_ai = markers_claim_ai(&markers);
    let has_signal = claims_ai || selected_mark.is_some() || statuses.contains(&WatermarkStatus::Detected);
    let hints = invisible_watermark_hints(&markers);
    let mut caveats = vec![
        "No public local decoder can verify proprietary pixel watermarks such as video SynthID.".to_owned(),
    ];
    if !hints.is_empty() {
        caveats.push(
            "Invisible watermark hints come from metadata declarations or provider policy, not a pixel detection. They do not confirm a \
             surviving watermark or verified removal on this file."
                .to_owned(),
        );
    }
    if !options.check_visible {
        caveats.push("Visible video-mark detection was skipped.".to_owned());
    }
    if options.check_audio {
        caveats.push(
            "Audio stream inspection covers installed compatible public decoders only; proprietary audio marks such as SynthID are not \
             checked locally."
                .to_owned(),
        );
    }
    if !has_signal && (markers.contains_key("issuer") || markers.contains_key("actions")) {
        caveats.push(format!(
            "A C2PA manifest signed by {} records {} and no AI claim, as a platform re-encode does (YouTube re-signs every upload); \
             that is unknown, not proof that the video is clean.",
            markers.get("issuer").map_or("an unregistered signer", String::as_str),
            markers.get("actions").map_or("no actions", String::as_str)
        ));
    } else if !has_signal && !markers.is_empty() {
        caveats.push(
            "Metadata does not establish AI origin; this is unknown, not proof that the video is clean."
                .to_owned(),
        );
    } else if !has_signal {
        caveats.push(
            "No supported signal was found; absence is unknown, not proof that the video is clean."
                .to_owned(),
        );
    }
    let recipe = recommend_video_recipe(&markers, selected_mark.as_deref(), claims_ai, &hints, &statuses);
    let platform = match &selected_mark {
        Some(mark) => visible_platform(mark).map(str::to_owned),
        None if claims_ai => platform_from_video_metadata(&markers),
        None => None,
    };
    Ok(VideoProvenanceReport {
        source,
        is_ai_generated: has_signal.then_some(true),
        confidence: if has_signal { "high" } else { "unknown" },
        platform,
        visible_mark: selected_mark,
        visible_detected_frames: detected_frames,
        total_frames,
        has_ai_metadata: !markers.is_empty(),
        metadata_markers: markers,
        caveats,
        invisible_watermark_hints: hints,
        audio_streams: audio_reports,
        recommended_recipe: recipe,
        visible_marks: selected_marks,
    })
}

/// `remove_video_metadata`: lossless stream-copy metadata removal with verification.
pub fn remove_video_metadata(
    source: &Path,
    output: Option<&Path>,
    keep_standard: bool,
    detected: Option<AiMetadata>,
) -> Result<VideoMetadataResult> {
    let source = video_source(source)?;
    let output = video_output(&source, output, "metadata removal")?;
    let detected = detected.unwrap_or_else(|| get_ai_metadata(&source));
    let outcome = strip_and_verify(&source, Some(&output), keep_standard)
        .map_err(|e| VideoError::runtime(e.to_string()))?;
    Ok(VideoMetadataResult {
        source,
        output: outcome.output,
        detected,
        remaining: outcome.remaining,
    })
}

/// Whether to trim a trailing AI outro bumper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrimBumper {
    #[default]
    Off,
    On,
    Auto,
}

/// Options for [`remove_video_visible`].
#[derive(Clone, Debug)]
pub struct VisibleVideoOptions {
    pub mark: String,
    pub backend: Backend,
    pub strip_metadata: bool,
    pub temporal_consistency: bool,
    pub trim_bumper: TrimBumper,
    pub metadata_markers: Option<AiMetadata>,
}

impl Default for VisibleVideoOptions {
    fn default() -> Self {
        Self {
            mark: "auto".into(),
            backend: Backend::Cv2,
            strip_metadata: true,
            temporal_consistency: true,
            trim_bumper: TrimBumper::Off,
            metadata_markers: None,
        }
    }
}

fn clean_options<'a>(
    merged_pads: Option<&'a [f64]>,
    merged_styles: Option<&'a [MaskStyle]>,
    padding: f64,
    style: MaskStyle,
    backend: Backend,
    temporal: bool,
) -> CleanOptions<'a> {
    CleanOptions {
        backend,
        padding_fraction: padding,
        mask_style: style,
        slot_padding_fractions: merged_pads,
        slot_mask_styles: merged_styles,
        temporal_consistency: temporal,
        ..CleanOptions::default()
    }
}

/// `remove_video_visible`: remove stable provider marks (and an optional trailing AI bumper).
pub fn remove_video_visible(
    runtime: &VideoRuntime,
    source: &Path,
    output: Option<&Path>,
    options: &VisibleVideoOptions,
) -> Result<VideoVisibleResult> {
    require_video_runtime()?;
    let (candidates, explicit_multi) = parse_mark_selector(&options.mark)?;
    let source = video_source(source)?;
    let output = video_output(&source, output, "visible watermark removal")?;
    let markers = options
        .metadata_markers
        .clone()
        .unwrap_or_else(|| get_ai_metadata(&source));
    let active = non_contradicted_marks(&candidates, &markers);
    let (scans, base) = scan_or_probe(&source, &active, true)?;
    let total = base.detections.len();
    let selected = select_stable_visible_marks(&scans, &markers, &candidates, explicit_multi);
    let trim = match options.trim_bumper {
        TrimBumper::On => true,
        TrimBumper::Auto => !markers.is_empty() || !selected.is_empty(),
        TrimBumper::Off => false,
    };
    let bumper = if trim {
        detect_ai_bumper_start(iter_decoded_bgr_frames(&source, None)?, base.fps)?
    } else {
        None
    };
    let trimmed = bumper.map_or(0, |cut| total - cut);
    let Some(merged) = merge_selected_visible_marks(&selected, total).or_else(|| {
        bumper.map(|_| crate::visible::MergedMarks {
            marks: Vec::new(),
            primary_mark: options.mark.clone(),
            scan: base.clone(),
            regions: vec![Vec::new(); total],
            detected_frames: 0,
            padding_fraction: 0.12,
            mask_style: MaskStyle::Box,
            slot_padding_fractions: None,
            slot_mask_styles: None,
        })
    }) else {
        return Ok(VideoVisibleResult {
            source,
            output: None,
            mark: options.mark.clone(),
            total_frames: total,
            detected_frames: 0,
            removed_frames: 0,
            remaining_metadata: if options.strip_metadata {
                markers
            } else {
                AiMetadata::new()
            },
            marks: Vec::new(),
            trimmed_bumper_frames: 0,
        });
    };
    let clean = clean_options(
        merged.slot_padding_fractions.as_deref(),
        merged.slot_mask_styles.as_deref(),
        merged.padding_fraction,
        merged.mask_style,
        options.backend,
        options.temporal_consistency,
    );
    let encode = EncodeCleanOptions {
        clean,
        strip_metadata: options.strip_metadata,
        trim_end_frame: bumper,
        crf: 14,
    };
    let removed = encode_clean_video(
        &source,
        &output,
        &merged.scan,
        &merged.regions,
        &runtime.fills,
        &encode,
    )?;
    let remaining = if options.strip_metadata {
        get_ai_metadata(&output)
    } else {
        AiMetadata::new()
    };
    Ok(VideoVisibleResult {
        source,
        output: Some(output),
        mark: merged.primary_mark,
        total_frames: total,
        detected_frames: merged.detected_frames,
        removed_frames: removed,
        remaining_metadata: remaining,
        marks: merged.marks,
        trimmed_bumper_frames: trimmed,
    })
}

/// Options for [`remove_video_region`].
#[derive(Clone, Debug)]
pub struct EraseVideoOptions {
    pub regions: Vec<Region>,
    pub frame_range: Option<(usize, usize)>,
    pub backend: Backend,
    pub dilate: i32,
    pub strip_metadata: bool,
    pub temporal_consistency: bool,
    pub track: bool,
}

/// `remove_video_region`: erase user boxes across all or a slice of frames.
pub fn remove_video_region(
    runtime: &VideoRuntime,
    source: &Path,
    output: Option<&Path>,
    options: &EraseVideoOptions,
) -> Result<VideoEraseResult> {
    if options.regions.is_empty() {
        return Err(VideoError::value("At least one (x, y, w, h) region is required"));
    }
    if options.dilate < 0 {
        return Err(VideoError::value("dilate must be nonnegative"));
    }
    let source = video_source(source)?;
    let output = video_output(&source, output, "video region erase")?;
    let backend = runtime.fills.resolve(options.backend);
    require_video_runtime()?;
    for r in &options.regions {
        if r.w <= 0 || r.h <= 0 {
            return Err(VideoError::value("Region width and height must be positive"));
        }
    }
    let geometry = probe_video_geometry(&source)?
        .ok_or_else(|| VideoError::runtime(format!("OpenCV could not decode video: {}", source.display())))?;
    let (width, height, fps) = (geometry.width, geometry.height, geometry.fps);
    if width == 0 || height == 0 || fps <= 0.0 {
        return Err(VideoError::runtime(format!(
            "Video metadata is missing geometry or frame rate: {}",
            source.display()
        )));
    }
    let scan_probe = probe_empty_video_scan(&source, true)?;
    let total = scan_probe.detections.len();
    let mut clipped = Vec::new();
    for r in &options.regions {
        let (x0, y0) = (r.x.max(0), r.y.max(0));
        let (x1, y1) = ((r.x + r.w).min(width as i32), (r.y + r.h).min(height as i32));
        if x1 <= x0 || y1 <= y0 {
            return Err(VideoError::value(format!(
                "Region ({}, {}, {}, {}) lies completely outside {width}x{height} video frame",
                r.x, r.y, r.w, r.h
            )));
        }
        clipped.push(Region::new(x0, y0, x1 - x0, y1 - y0));
    }
    let range = match options.frame_range {
        None => None,
        Some((start, end)) => {
            if end < start || start >= total {
                return Err(VideoError::value(format!(
                    "Invalid frame_range ({start}, {end}) for {total}-frame video"
                )));
            }
            Some((start, end.min(total - 1)))
        }
    };
    let per_frame: Vec<Option<Vec<Region>>> = if options.track {
        let (start, end) = range.map_or((0, None), |(s, e)| (s, Some(e)));
        let max_frames = end.map(|e| e + 1);
        let mut boxes = track_video_regions(
            iter_decoded_bgr_frames(&source, max_frames)?,
            &clipped,
            start,
            end,
        )?;
        if boxes.len() < total {
            boxes.resize(total, None);
        }
        if boxes.len() != total {
            return Err(VideoError::runtime(
                "Tracked frame count did not match probed video frame count",
            ));
        }
        boxes
    } else {
        (0..total)
            .map(|i| {
                range
                    .is_none_or(|(s, e)| s <= i && i <= e)
                    .then(|| clipped.clone())
            })
            .collect()
    };
    let detections: Vec<FrameLocalization> = per_frame
        .iter()
        .enumerate()
        .map(|(i, r)| FrameLocalization {
            frame_index: i,
            confidence: if r.as_ref().is_some_and(|v| !v.is_empty()) {
                1.0
            } else {
                0.0
            },
            region: r.as_ref().and_then(|v| v.first().copied()),
        })
        .collect();
    let regions: Vec<FrameRegions> = per_frame
        .into_iter()
        .map(|r| r.map_or_else(Vec::new, |v| v.into_iter().map(Some).collect()))
        .collect();
    let scan = VideoScan {
        width,
        height,
        fps,
        detections,
        timestamps: scan_probe.timestamps,
    };
    let clean = CleanOptions {
        backend: backend.into(),
        temporal_consistency: options.temporal_consistency,
        dilate_px: Some(options.dilate),
        allow_unmasked_donors: options.track && range.is_none(),
        ..CleanOptions::default()
    };
    let encode = EncodeCleanOptions {
        clean,
        strip_metadata: options.strip_metadata,
        trim_end_frame: None,
        crf: 14,
    };
    let erased = encode_clean_video(&source, &output, &scan, &regions, &runtime.fills, &encode)?;
    let remaining = if options.strip_metadata {
        get_ai_metadata(&output)
    } else {
        AiMetadata::new()
    };
    Ok(VideoEraseResult {
        source,
        output,
        regions: clipped,
        total_frames: total,
        erased_frames: erased,
        frame_range: range,
        backend: backend.as_str().to_owned(),
        remaining_metadata: remaining,
        tracked: options.track,
    })
}

/// `remove_video_invisible`: regenerate pixels through the resolved profile.
pub fn remove_video_invisible(
    runtime: &VideoRuntime,
    source: &Path,
    output: Option<&Path>,
    options: &InvisibleOptions,
) -> Result<VideoInvisibleResult> {
    validate_video_profile(options, InvisibleMode::On)?;
    let source = video_source(source)?;
    let markers = get_ai_metadata(&source);
    let profile = resolve_video_invisible_profile(&runtime.neural, options, &source, &markers, None)?;
    run_invisible_profile::<u8>(runtime, &source, output, options, &profile, None)
}

/// The regeneration stage of `remove_video_invisible` for a resolved profile; `frames` streams
/// already-cleaned native frames (`_frame_stream`) instead of decoding the source again.
fn run_invisible_profile<T: VaePixel>(
    runtime: &VideoRuntime,
    source: &Path,
    output: Option<&Path>,
    options: &InvisibleOptions,
    profile: &str,
    frames: Option<&mut dyn Iterator<Item = Result<Mat<T>>>>,
) -> Result<VideoInvisibleResult> {
    require_profile_backend(&runtime.neural, profile)?;
    let legacy = options.profile.ends_with("-experimental");
    if !is_regenerable(source) {
        let what = match profile {
            "seedance-experimental" => "Seedance processing",
            p if WAN_EXPERIMENTAL_PROFILES.contains(&p) => "Wan 2.1 processing",
            _ => "Video SynthID regeneration",
        };
        return Err(VideoError::value(format!(
            "{what} requires one of: {}",
            sorted_extensions()
        )));
    }
    require_video_runtime()?;
    let quality = SeedanceQuality::parse(&options.quality)?;
    let vae = || {
        runtime
            .neural
            .vae
            .as_ref()
            .ok_or_else(|| VideoError::Unavailable("Video regeneration needs the VAE runtime".into()))
    };
    if profile == "seedance-experimental" {
        let output = video_output(source, output, "Experimental Seedance processing")?;
        let metrics =
            regenerate_seedance_experimental(vae()?.as_ref(), source, &output, &options.device, quality)?;
        return Ok(VideoInvisibleResult {
            remaining_metadata: get_ai_metadata(&output),
            source: source.to_path_buf(),
            output,
            noise_std: 0.0,
            metrics,
            profile: SEEDANCE_EXPERIMENTAL_PROFILE.into(),
            quality: Some(quality.as_str().into()),
            experimental: Some(legacy),
            strength: None,
        });
    }
    if WAN_EXPERIMENTAL_PROFILES.contains(&profile) {
        let wan = runtime
            .neural
            .wan_for(profile)
            .ok_or_else(|| VideoError::Unavailable("The Wan 2.1 profile needs the Wan runtime".into()))?;
        let output = video_output(source, output, "Experimental Wan 2.1 processing")?;
        let wan_options = WanOptions {
            profile: profile.to_owned(),
            device: options.device.clone(),
            quality,
            seed: options.seed,
            windowing: WanWindowing::default(),
            inference_steps: options.wan_steps,
        };
        let metrics = match frames {
            Some(stream) => {
                regenerate_wan_experimental(wan.as_ref(), source, &output, &wan_options, Some(stream))?
            }
            None if probe_video_encode_profile(source)?.high_depth() => {
                regenerate_wan_experimental::<u16>(wan.as_ref(), source, &output, &wan_options, None)?
            }
            None => regenerate_wan_experimental::<u8>(wan.as_ref(), source, &output, &wan_options, None)?,
        };
        let strength = profile_strength(profile)?;
        return Ok(VideoInvisibleResult {
            remaining_metadata: get_ai_metadata(&output),
            source: source.to_path_buf(),
            output,
            noise_std: strength,
            metrics,
            profile: profile.to_owned(),
            quality: Some(quality.as_str().into()),
            experimental: Some(legacy),
            strength: Some(strength),
        });
    }
    let output = video_output(source, output, "SynthID removal")?;
    let metrics: RegenerationMetrics = match frames {
        Some(stream) => regenerate_video_frames(
            vae()?.as_ref(),
            source,
            &output,
            options,
            None,
            None,
            FrameFeed::Stream(stream),
        )?,
        None => regenerate_video_candidate(vae()?.as_ref(), source, &output, options, None, None)?,
    };
    Ok(VideoInvisibleResult {
        remaining_metadata: get_ai_metadata(&output),
        source: source.to_path_buf(),
        output,
        noise_std: options.noise_std,
        metrics,
        profile: "synthid".into(),
        quality: None,
        experimental: None,
        strength: None,
    })
}

/// Options for [`remove_video_all`].
#[derive(Clone, Debug)]
pub struct AllVideoOptions {
    pub visible: VisibleVideoOptions,
    pub include_invisible: InvisibleMode,
    pub invisible: InvisibleOptions,
    pub include_audio_invisible: bool,
    pub audio: VideoAudioOptions,
    pub max_duration_seconds: Option<f64>,
    pub max_frame_pixels: Option<usize>,
    pub require_gpu: bool,
}

impl Default for AllVideoOptions {
    fn default() -> Self {
        Self {
            visible: VisibleVideoOptions::default(),
            include_invisible: InvisibleMode::Off,
            invisible: InvisibleOptions::default(),
            include_audio_invisible: false,
            audio: VideoAudioOptions::default(),
            max_duration_seconds: None,
            max_frame_pixels: None,
            require_gpu: false,
        }
    }
}

fn copy_published(from: &Path, to: &Path) -> Result<()> {
    std::fs::copy(from, to)?;
    Ok(())
}

/// `remove_video_all`: visible removal plus verified metadata stripping, with optional invisible
/// and audio stages, published atomically; clean inputs are byte-copied.
pub fn remove_video_all(
    runtime: &VideoRuntime,
    source: &Path,
    output: Option<&Path>,
    options: &AllVideoOptions,
) -> Result<VideoAllResult> {
    validate_video_profile(&options.invisible, options.include_invisible)?;
    if options.max_duration_seconds.is_some_and(|d| d <= 0.0) {
        return Err(VideoError::value("max_duration_seconds must be positive"));
    }
    if options.max_frame_pixels == Some(0) {
        return Err(VideoError::value("max_frame_pixels must be positive"));
    }
    let source = video_source(source)?;
    let output = video_output(&source, output, "complete cleaning")?;
    if options.max_frame_pixels.is_some() || options.max_duration_seconds.is_some() {
        let g = probe_video_geometry(&source)?
            .ok_or_else(|| VideoError::value(format!("Could not open video: {}", source.display())))?;
        if let Some(limit) = options.max_frame_pixels
            && g.width * g.height > limit
        {
            return Err(VideoError::value(format!(
                "Video frame area {}x{} ({} px) exceeds max_frame_pixels={limit}",
                g.width,
                g.height,
                g.width * g.height
            )));
        }
        if let Some(limit) = options.max_duration_seconds {
            let ts = probe_video_timestamps(&source)?;
            let fps = g.fps.max(1e-6);
            let estimate = if ts.len() >= 2 {
                (ts[ts.len() - 1] - ts[0]) + 1.0 / fps
            } else {
                ts.len() as f64 / fps
            };
            if estimate > limit {
                return Err(VideoError::value(format!(
                    "Video duration {estimate:.2}s exceeds max_duration_seconds={limit:.2}s"
                )));
            }
        }
    }
    let explicit_invisible = options.include_invisible == InvisibleMode::On;
    if explicit_invisible && !is_regenerable(&source) {
        return Err(VideoError::value(format!(
            "Video invisible regeneration requires one of: {}",
            sorted_extensions()
        )));
    }
    let include_audio = options.include_audio_invisible || options.audio.vendor.is_some();
    let mut preflight = Vec::new();
    if include_audio {
        if !matches!(
            suffix_lower(&source).as_str(),
            ".mp4" | ".mov" | ".m4v" | ".mkv" | ".webm"
        ) {
            return Err(VideoError::value(
                "Video audio watermark processing requires MP4, MOV, M4V, MKV, or WebM input",
            ));
        }
        preflight = preflight_video_audio_workload(&runtime.audio, &source, &options.audio)?;
    }
    require_video_runtime()?;
    let detected = options
        .visible
        .metadata_markers
        .clone()
        .unwrap_or_else(|| get_ai_metadata(&source));
    let final_stage = atomic_video_output(&output)?;
    let work = tempfile::Builder::new()
        .prefix(&format!(".{}-video-all-", file_stem(&source)))
        .tempdir_in(parent_dir(&source))?;
    let suffix = source
        .extension()
        .map_or_else(String::new, |e| format!(".{}", e.to_string_lossy()));
    let stage_target = if include_audio {
        work.path().join(format!("visual-final{suffix}"))
    } else {
        final_stage.path().to_path_buf()
    };

    let stage = VisualStage {
        runtime,
        source: &source,
        stage_target: &stage_target,
        work: work.path(),
        suffix: &suffix,
        detected: &detected,
        options,
    };
    let (visible, mut remaining, invisible_profile) =
        if explicit_invisible && options.visible.trim_bumper == TrimBumper::Off {
            stage.fused()?
        } else {
            stage.sequential()?
        };

    let mut audio_status = None;
    let mut processed_streams = Vec::new();
    let mut scope = Vec::new();
    let mut audio_quality = None;
    if include_audio {
        let streams = stream_inventory(&stage_target)?;
        let has_audio = streams
            .iter()
            .any(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("audio"));
        let mut published = false;
        let mut state = AudioWatermarkState::Unverified;
        if has_audio {
            let mut reusable = None;
            if !preflight.is_empty() {
                let src: Vec<usize> = preflight.iter().map(|r| r.stream.index).collect();
                let dst: Vec<usize> = streams
                    .iter()
                    .filter(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("audio"))
                    .filter_map(|s| {
                        s.get("index")
                            .and_then(serde_json::Value::as_u64)
                            .map(|v| v as usize)
                    })
                    .collect();
                if src.len() == dst.len()
                    && src
                        .iter()
                        .zip(&dst)
                        .all(|(a, b)| verify_stream_packets(&source, &stage_target, *a, Some(*b)).is_ok())
                {
                    reusable = Some(preflight.clone());
                }
            }
            let mut audio_options = options.audio.clone();
            if options.audio.vendor.is_none() && visible.output.is_some() && visible.mark == "veo" {
                audio_options.vendor = Some(raiw_audio::Vendor::Google);
            }
            audio_options.metadata_markers = Some(detected.clone());
            let result = remove_video_audio_all_streams(
                &runtime.audio,
                &stage_target,
                final_stage.path(),
                &audio_options,
                reusable,
            )?;
            let mut seen: Vec<String> = Vec::new();
            for s in &result.streams {
                for d in s
                    .before
                    .detections
                    .iter()
                    .chain(s.after.iter().flat_map(|a| a.detections.iter()))
                {
                    let name = d.scheme.as_str().to_owned();
                    if !seen.contains(&name) {
                        seen.push(name);
                    }
                }
            }
            seen.sort();
            scope = seen;
            audio_quality = result.quality_passed;
            if result.skipped || result.output.is_none() {
                let all_negative = !result.streams.is_empty()
                    && result
                        .streams
                        .iter()
                        .all(|s| s.before.watermark_status == WatermarkStatus::NotDetected);
                state = if all_negative {
                    AudioWatermarkState::NotDetected
                } else {
                    AudioWatermarkState::Unverified
                };
            } else {
                published = true;
                remaining = get_ai_metadata(final_stage.path());
                let unprocessed_detected = result
                    .streams
                    .iter()
                    .any(|s| s.skipped && s.before.watermark_status == WatermarkStatus::Detected);
                let unprocessed_unverified = result
                    .streams
                    .iter()
                    .any(|s| s.skipped && s.before.watermark_status != WatermarkStatus::NotDetected);
                state = if result.verified_removed == Some(false) || unprocessed_detected {
                    AudioWatermarkState::Detected
                } else if result.verified_removed == Some(true) && !unprocessed_unverified {
                    AudioWatermarkState::NotDetected
                } else {
                    AudioWatermarkState::Unverified
                };
                audio_status = Some(VideoAudioStatus {
                    stream_action: StreamAction::Processed,
                    watermark_status: state,
                });
                processed_streams = result.processed_ordinals;
            }
        }
        if !published {
            copy_published(&stage_target, final_stage.path())?;
            audio_status = Some(VideoAudioStatus {
                stream_action: if has_audio {
                    StreamAction::CopiedIfPresent
                } else {
                    StreamAction::Absent
                },
                watermark_status: state,
            });
        }
    }
    if !remaining.is_empty() {
        let mut keys: Vec<&String> = remaining.keys().collect();
        keys.sort();
        return Err(VideoError::runtime(format!(
            "{} AI metadata marker(s) survived the complete pipeline: {}",
            remaining.len(),
            raiw_core::pyjson::py_repr(&serde_json::json!(keys))
        )));
    }
    final_stage.commit()?;
    Ok(VideoAllResult {
        source,
        output,
        visible_mark: visible.output.as_ref().map(|_| visible.mark.clone()),
        total_frames: visible.total_frames,
        visible_detected_frames: visible.detected_frames,
        visible_removed_frames: visible.removed_frames,
        detected_metadata: detected,
        remaining_metadata: remaining,
        invisible_removed: invisible_profile.is_some(),
        invisible_profile,
        audio_status,
        audio_processed_streams: processed_streams,
        audio_verification_scope: scope,
        audio_quality_passed: audio_quality,
        visible_marks: if visible.output.is_some() {
            visible.marks.clone()
        } else {
            Vec::new()
        },
        trimmed_bumper_frames: visible.trimmed_bumper_frames,
    })
}

/// The visual (visible + invisible) stage of [`remove_video_all`], before any audio work.
struct VisualStage<'a> {
    runtime: &'a VideoRuntime,
    source: &'a Path,
    stage_target: &'a Path,
    work: &'a Path,
    suffix: &'a str,
    detected: &'a AiMetadata,
    options: &'a AllVideoOptions,
}

type VisualOutcome = (VideoVisibleResult, AiMetadata, Option<String>);

impl VisualStage<'_> {
    fn resolve_profile(&self, current: &Path, visible_mark: Option<&str>) -> Result<String> {
        let profile = resolve_video_invisible_profile(
            &self.runtime.neural,
            &self.options.invisible,
            current,
            self.detected,
            visible_mark,
        )?;
        if self.options.require_gpu && profile == "synthid" {
            return Err(VideoError::runtime(
                "require_gpu=True refuses 512p 2D CPU fallback; run on a CUDA/MPS GPU or disable require_gpu",
            ));
        }
        Ok(profile)
    }

    fn invisible_options(&self, profile: &str) -> InvisibleOptions {
        InvisibleOptions {
            profile: profile.to_owned(),
            ..self.options.invisible.clone()
        }
    }

    /// Visible removal to a file, then (explicit or provenance-triggered) regeneration of it.
    fn sequential(&self) -> Result<VisualOutcome> {
        let (runtime, source, options) = (self.runtime, self.source, self.options);
        let visible_output = if options.include_invisible == InvisibleMode::Off {
            self.stage_target.to_path_buf()
        } else {
            self.work.join(format!("visible{}", self.suffix))
        };
        let visible_options = VisibleVideoOptions {
            strip_metadata: true,
            metadata_markers: Some(self.detected.clone()),
            ..options.visible.clone()
        };
        let visible = remove_video_visible(runtime, source, Some(&visible_output), &visible_options)?;
        let current = visible.output.clone().unwrap_or_else(|| source.to_path_buf());
        let visible_mark = visible.output.as_ref().map(|_| visible.mark.clone());
        let mut run_invisible = options.include_invisible == InvisibleMode::On;
        if options.include_invisible == InvisibleMode::Auto
            && video_has_invisible_provenance(self.detected, visible_mark.as_deref(), None)
            && is_regenerable(&current)
        {
            if runtime.neural.vae.is_some() {
                run_invisible = true;
            } else {
                tracing::warn!(
                    "AI video provenance detected on {}; skipping invisible regeneration (neural runtime not built)",
                    source.display()
                );
            }
        }
        if run_invisible {
            let profile = self.resolve_profile(&current, visible_mark.as_deref())?;
            let regenerated = run_invisible_profile::<u8>(
                runtime,
                &current,
                Some(self.stage_target),
                &self.invisible_options(&profile),
                &profile,
                None,
            )?;
            return Ok((visible, regenerated.remaining_metadata, Some(profile)));
        }
        if visible.output.is_none() {
            let result =
                remove_video_metadata(source, Some(self.stage_target), true, Some(self.detected.clone()))?;
            return Ok((visible, result.remaining, None));
        }
        if visible.output.as_deref() != Some(self.stage_target) {
            std::fs::rename(&visible_output, self.stage_target)
                .or_else(|_| copy_published(&visible_output, self.stage_target))?;
        }
        let remaining = visible.remaining_metadata.clone();
        Ok((visible, remaining, None))
    }

    /// `video all --invisible`: scan once and stream cleaned frames straight into regeneration;
    /// Seedance regenerates a lossless cleaned intermediate instead.
    fn fused(&self) -> Result<VisualOutcome> {
        let (runtime, source, options) = (self.runtime, self.source, self.options);
        let (candidates, explicit_multi) = parse_mark_selector(&options.visible.mark)?;
        let active = non_contradicted_marks(&candidates, self.detected);
        let (scans, base) = scan_or_probe(source, &active, true)?;
        let total = base.detections.len();
        let selected = select_stable_visible_marks(&scans, self.detected, &candidates, explicit_multi);
        let profile = self.resolve_profile(source, selected.first().map(|s| s.mark.as_str()))?;
        let invisible = self.invisible_options(&profile);
        let Some(merged) = merge_selected_visible_marks(&selected, total) else {
            let result = run_invisible_profile::<u8>(
                runtime,
                source,
                Some(self.stage_target),
                &invisible,
                &profile,
                None,
            )?;
            let visible = VideoVisibleResult {
                source: source.to_path_buf(),
                output: None,
                mark: options.visible.mark.clone(),
                total_frames: total,
                detected_frames: 0,
                removed_frames: 0,
                remaining_metadata: AiMetadata::new(),
                marks: Vec::new(),
                trimmed_bumper_frames: 0,
            };
            return Ok((visible, result.remaining_metadata, Some(profile)));
        };
        let clean = clean_options(
            merged.slot_padding_fractions.as_deref(),
            merged.slot_mask_styles.as_deref(),
            merged.padding_fraction,
            merged.mask_style,
            options.visible.backend,
            options.visible.temporal_consistency,
        );
        let (result, removed) = if profile == "seedance-experimental" {
            let lossless = self.work.join(format!("visible-lossless{}", self.suffix));
            let encode = EncodeCleanOptions {
                clean,
                strip_metadata: true,
                trim_end_frame: None,
                crf: 0,
            };
            let removed = encode_clean_video(
                source,
                &lossless,
                &merged.scan,
                &merged.regions,
                &runtime.fills,
                &encode,
            )?;
            let result = run_invisible_profile::<u8>(
                runtime,
                &lossless,
                Some(self.stage_target),
                &invisible,
                &profile,
                None,
            )?;
            (result, removed)
        } else if probe_video_encode_profile(source)?.high_depth() {
            self.stream_clean::<u16>(&merged, &clean, &invisible, &profile)?
        } else {
            self.stream_clean::<u8>(&merged, &clean, &invisible, &profile)?
        };
        let visible = VideoVisibleResult {
            source: source.to_path_buf(),
            output: Some(self.stage_target.to_path_buf()),
            mark: merged.primary_mark,
            total_frames: total,
            detected_frames: merged.detected_frames,
            removed_frames: if removed > 0 {
                removed
            } else {
                merged.detected_frames
            },
            remaining_metadata: result.remaining_metadata.clone(),
            marks: merged.marks,
            trimmed_bumper_frames: 0,
        };
        Ok((visible, result.remaining_metadata, Some(profile)))
    }

    /// `_fused_clean_stream`: a cleaner thread feeds frames to regeneration through a bounded channel.
    fn stream_clean<T: VaePixel>(
        &self,
        merged: &crate::visible::MergedMarks,
        clean: &CleanOptions<'_>,
        invisible: &InvisibleOptions,
        profile: &str,
    ) -> Result<(VideoInvisibleResult, usize)> {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<Result<Mat<T>>>(4);
        std::thread::scope(|scope| {
            let cleaner = scope.spawn(move || {
                let mut removed = 0usize;
                let outcome = for_each_clean_frame::<T>(
                    self.source,
                    &merged.scan,
                    &merged.regions,
                    &self.runtime.fills,
                    clean,
                    |frame, was_removed| {
                        removed += usize::from(was_removed);
                        sender
                            .send(Ok(frame))
                            .map_err(|_| VideoError::runtime("regeneration stopped reading cleaned frames"))
                    },
                );
                if let Err(error) = outcome {
                    let _ = sender.send(Err(error));
                }
                removed
            });
            let mut frames = receiver.into_iter();
            let result = run_invisible_profile::<T>(
                self.runtime,
                self.source,
                Some(self.stage_target),
                invisible,
                profile,
                Some(&mut frames),
            );
            // Unblock the cleaner when regeneration stopped early.
            drop(frames);
            let removed = cleaner
                .join()
                .map_err(|_| VideoError::runtime("the visible-cleaning thread panicked"))?;
            result.map(|r| (r, removed))
        })
    }
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn parent_dir(path: &Path) -> PathBuf {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

/// Options for [`remove_video_batch`].
#[derive(Clone, Debug)]
pub struct BatchVideoOptions {
    pub mode: BatchMode,
    pub all: AllVideoOptions,
}

/// `remove_video_batch`: process every supported top-level video sequentially.
pub fn remove_video_batch(
    runtime: &VideoRuntime,
    directory: &Path,
    output_directory: Option<&Path>,
    options: &BatchVideoOptions,
) -> Result<VideoBatchResult> {
    let all = &options.all;
    validate_video_profile(&all.invisible, all.include_invisible)?;
    if !directory.exists() {
        return Err(VideoError::NotFound(format!(
            "Video directory does not exist: {}",
            directory.display()
        )));
    }
    if !directory.is_dir() {
        return Err(VideoError::value(format!(
            "Video batch source must be a directory: {}",
            directory.display()
        )));
    }
    if all.visible.mark != "auto" && !VIDEO_VISIBLE_MARKS.contains(&all.visible.mark.as_str()) {
        return Err(VideoError::value(format!(
            "Unsupported visible video mark; expected {}",
            expected_marks()
        )));
    }
    let include_audio = all.include_audio_invisible || all.audio.vendor.is_some();
    if all.include_invisible != InvisibleMode::Off && options.mode != BatchMode::All {
        return Err(VideoError::value(
            "The invisible video stage is available only in all mode",
        ));
    }
    if include_audio && options.mode != BatchMode::All {
        return Err(VideoError::value(
            "The audio watermark stage is available only in all mode",
        ));
    }
    if options.mode != BatchMode::Metadata {
        require_video_runtime()?;
    }
    let output_directory = output_directory.map_or_else(
        || raiw_core::fs::with_name_suffix(directory, "_clean"),
        Path::to_path_buf,
    );
    if raiw_core::resolve_lenient(&output_directory) == raiw_core::resolve_lenient(directory) {
        return Err(VideoError::value(
            "Video batch output directory must differ from the source directory",
        ));
    }
    let mut sources: Vec<PathBuf> = std::fs::read_dir(directory)?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file() && VIDEO_EXTENSIONS.contains(&suffix_lower(p).as_str()))
        .collect();
    sources.sort_by_key(|p| {
        p.file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    });
    if !sources.is_empty() {
        std::fs::create_dir_all(&output_directory)?;
    }
    let mut items = Vec::new();
    for source in sources {
        let item_output = output_directory.join(source.file_name().unwrap_or_default());
        let outcome = batch_item(runtime, &source, &item_output, options);
        items.push(
            outcome.unwrap_or_else(|error| {
                VideoBatchItem::failed(source.clone(), options.mode, error.to_string())
            }),
        );
    }
    Ok(VideoBatchResult {
        directory: directory.to_path_buf(),
        output_directory,
        items,
    })
}

fn batch_item(
    runtime: &VideoRuntime,
    source: &Path,
    output: &Path,
    options: &BatchVideoOptions,
) -> Result<VideoBatchItem> {
    let base = |changed: bool, out: Option<PathBuf>| VideoBatchItem {
        source: source.to_path_buf(),
        output: out,
        mode: options.mode,
        changed,
        visible_mark: None,
        invisible_removed: false,
        error: None,
        audio_status: None,
        audio_verification_scope: Vec::new(),
        audio_quality_passed: None,
        visible_marks: Vec::new(),
    };
    match options.mode {
        BatchMode::All => {
            let result = remove_video_all(runtime, source, Some(output), &options.all)?;
            if !result.remaining_metadata.is_empty() {
                return Err(VideoError::runtime(format!(
                    "{} AI metadata marker(s) survived the complete pipeline",
                    result.remaining_metadata.len()
                )));
            }
            let changed = result.visible_mark.is_some()
                || !result.detected_metadata.is_empty()
                || result.invisible_removed
                || result.audio().stream_action == StreamAction::Processed;
            Ok(VideoBatchItem {
                visible_mark: result.visible_mark.clone(),
                invisible_removed: result.invisible_removed,
                audio_status: result.audio_status,
                audio_verification_scope: result.audio_verification_scope.clone(),
                audio_quality_passed: result.audio_quality_passed,
                visible_marks: result.visible_marks.clone(),
                ..base(changed, Some(result.output.clone()))
            })
        }
        BatchMode::Visible => {
            let visible_options = VisibleVideoOptions {
                strip_metadata: false,
                ..options.all.visible.clone()
            };
            let result = remove_video_visible(runtime, source, Some(output), &visible_options)?;
            if result.output.is_none() {
                let staged = atomic_video_output(output)?;
                copy_published(source, staged.path())?;
                staged.commit()?;
            }
            let wrote = result.output.is_some();
            Ok(VideoBatchItem {
                visible_mark: wrote.then(|| result.mark.clone()),
                visible_marks: if wrote { result.marks.clone() } else { Vec::new() },
                ..base(wrote, Some(output.to_path_buf()))
            })
        }
        BatchMode::Metadata => {
            let result = remove_video_metadata(source, Some(output), true, None)?;
            if !result.remaining.is_empty() {
                return Err(VideoError::runtime(format!(
                    "{} AI metadata marker(s) survived metadata removal",
                    result.remaining.len()
                )));
            }
            Ok(base(!result.detected.is_empty(), Some(result.output)))
        }
    }
}

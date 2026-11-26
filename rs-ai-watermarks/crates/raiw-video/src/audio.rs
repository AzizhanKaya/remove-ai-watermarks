//! Watermarked-audio processing inside videos (port of `video_audio.py`).
//!
//! One continuous audio track is replaced while every other stream is stream-copied; publication
//! requires identical packet payloads and PTS/DTS within one time-base tick for unchanged streams.

use std::path::{Path, PathBuf};
use std::process::Command;

use raiw_audio::io::AudioStreamInfo;
use raiw_audio::options::{Device, Method, Profile, Scheme, Vendor};
use raiw_audio::policy::{
    Permissions, ProcessingArgs, adaptive_auto_profiles, enforce_publication_gates, require_profile_runtime,
    resolve_audio_profiles, should_auto_scrub, validate_audio_arguments, validate_stream_for_profile,
};
use raiw_audio::quality::{AudioFidelity, audio_quality_passed, measure_fidelity};
use raiw_audio::report::{missing_decoder_reason, verification_outcome};
use raiw_audio::{
    AudioError, AudioProvenanceReport, AudioRuntime, IdentifyOptions, ProfileRequest, WatermarkStatus,
    identify_audio, remove_audio_profile,
};
use raiw_core::fs::suffix_lower;
use raiw_core::{StagedSibling, sha256_file};
use raiw_metadata::AiMetadata;
use serde::Serialize;
use serde_json::Value;

use crate::error::{Result, VideoError};
use crate::process::{head_chars, run_captured, run_with_stderr_file};
use crate::tools::{Tool, require_video_tool};

const CAVEAT_TRACK: &str =
    "Only the selected audio track is processed; other tracks and video marks remain unchanged.";
const CAVEAT_METADATA: &str = "Metadata is not intentionally stripped; container conversion can change it. Verification covers the named decoder.";
const PCM_CAVEAT: &str =
    "The selected track is stored as uncompressed float PCM, which can increase file size.";

/// An exact rational for clocks parsed from ffprobe decimals (`Fraction(str)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Exact {
    pub num: i128,
    pub den: i128,
}

impl Ord for Exact {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.num * other.den).cmp(&(other.num * self.den))
    }
}

impl PartialOrd for Exact {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    (a, b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

impl Exact {
    pub fn new(num: i128, den: i128) -> Self {
        let sign = if den < 0 { -1 } else { 1 };
        let g = gcd(num, den);
        Self {
            num: sign * num / g,
            den: sign * den / g,
        }
    }

    /// `Fraction("1.5")`, `Fraction("1/48000")`, `Fraction("-3")`.
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        if let Some((n, d)) = text.split_once('/') {
            let d: i128 = d.trim().parse().ok()?;
            let n: i128 = n.trim().parse().ok()?;
            return (d != 0).then(|| Self::new(n, d));
        }
        let (negative, body) = text.strip_prefix('-').map_or((false, text), |rest| (true, rest));
        let (int, frac) = body.split_once('.').unwrap_or((body, ""));
        if int.is_empty() && frac.is_empty() {
            return None;
        }
        let digits = format!("{int}{frac}");
        let value: i128 = if digits.is_empty() {
            0
        } else {
            digits.parse().ok()?
        };
        let den = 10i128.checked_pow(frac.len() as u32)?;
        Some(Self::new(if negative { -value } else { value }, den))
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.num * other.den - other.num * self.den, self.den * other.den)
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.num * other.den + other.num * self.den, self.den * other.den)
    }

    fn abs(self) -> Self {
        Self::new(self.num.abs(), self.den)
    }

    fn mul_int(self, value: i128) -> Self {
        Self::new(self.num * value, self.den)
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

/// `VideoAudioRemovalResult`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoAudioRemovalResult {
    pub source: PathBuf,
    pub output: Option<PathBuf>,
    pub before: AudioProvenanceReport,
    pub after: Option<AudioProvenanceReport>,
    pub frames_processed: usize,
    pub verified_removed: Option<bool>,
    pub skipped: bool,
    pub fidelity: Option<AudioFidelity>,
    pub unchanged_streams: Vec<usize>,
    #[serde(serialize_with = "serialize_fraction")]
    pub source_audio_start: Option<Exact>,
    #[serde(serialize_with = "serialize_fraction")]
    pub output_audio_start: Option<Exact>,
    pub method: String,
    pub bandwidth_kbps: Option<f64>,
    pub vocoder_device: Option<String>,
    pub caveats: Vec<String>,
    pub quality_passed: Option<bool>,
}

#[allow(clippy::ref_option)]
fn serialize_fraction<S: serde::Serializer>(
    value: &Option<Exact>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    match value {
        Some(v) if v.den == 1 => serializer.serialize_str(&v.num.to_string()),
        Some(v) => serializer.serialize_str(&format!("{}/{}", v.num, v.den)),
        None => serializer.serialize_none(),
    }
}

/// `VideoAudioMultiStreamResult`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoAudioMultiStreamResult {
    pub source: PathBuf,
    pub output: Option<PathBuf>,
    pub streams: Vec<VideoAudioRemovalResult>,
    pub processed_ordinals: Vec<usize>,
    pub unchanged_streams: Vec<usize>,
    pub verified_removed: Option<bool>,
    pub skipped: bool,
    pub caveats: Vec<String>,
    pub quality_passed: Option<bool>,
}

/// One `ffprobe -show_entries stream=...` row.
pub type StreamRow = serde_json::Map<String, Value>;

fn row_index(row: &StreamRow) -> Option<usize> {
    row.get("index").and_then(Value::as_u64).map(|v| v as usize)
}

fn row_str<'a>(row: &'a StreamRow, key: &str) -> Option<&'a str> {
    row.get(key).and_then(Value::as_str)
}

/// `stream_inventory` (`_streams`): every stream; fails without a video stream.
pub fn stream_inventory(source: &Path) -> Result<Vec<StreamRow>> {
    let ffprobe = require_video_tool(Tool::Ffprobe)?;
    let mut command = Command::new(ffprobe);
    command
        .args(["-v", "error", "-show_entries"])
        .arg("stream=index,codec_type,codec_name,sample_rate,channels,channel_layout,duration,time_base:stream_disposition:stream_tags=language")
        .args(["-of", "json"])
        .arg(source);
    let output = run_captured(&mut command)?;
    if !output.status.success() {
        return Err(VideoError::value(format!(
            "Cannot inspect video streams: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let payload: Value = serde_json::from_slice(&output.stdout)?;
    let streams: Vec<StreamRow> = payload
        .get("streams")
        .and_then(Value::as_array)
        .map(|s| s.iter().filter_map(|v| v.as_object().cloned()).collect())
        .unwrap_or_default();
    if !streams.iter().any(|s| row_str(s, "codec_type") == Some("video")) {
        return Err(VideoError::value("Source has no video stream"));
    }
    Ok(streams)
}

fn lines(mut command: Command) -> Result<Vec<String>> {
    let mut stderr = tempfile::tempfile()?;
    let output = command
        .stdin(std::process::Stdio::null())
        .stderr(stderr.try_clone()?)
        .output()?;
    if !output.status.success() {
        let text = crate::process::read_bounded_stderr(&mut stderr).unwrap_or_default();
        let program = command.get_program().to_string_lossy().into_owned();
        return Err(VideoError::runtime(format!(
            "{program} failed: {}",
            head_chars(&text, 500)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|l| l.trim().to_owned())
        .collect())
}

fn compact_fields(line: &str) -> Vec<(String, String)> {
    line.split('|')
        .filter_map(|part| part.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned())))
        .collect()
}

/// `_timeline`: the first decoded audio start and the total sample count, requiring continuity.
fn timeline(source: &Path, index: usize, rate: u32, tick: Exact) -> Result<(Exact, i128)> {
    let mut command = Command::new(require_video_tool(Tool::Ffprobe)?);
    command
        .args([
            "-v",
            "error",
            "-select_streams",
            &index.to_string(),
            "-show_frames",
        ])
        .args([
            "-show_entries",
            "frame=best_effort_timestamp_time,nb_samples",
            "-of",
            "compact=p=0",
        ])
        .arg(source);
    let mut first: Option<Exact> = None;
    let mut count: i128 = 0;
    let slack = tick.max(Exact::new(1, i128::from(rate)));
    for line in lines(command)? {
        let fields = compact_fields(&line);
        if fields.is_empty() {
            continue;
        }
        let get = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str());
        let unavailable = || VideoError::value("Audio frame timing is unavailable");
        let start = get("best_effort_timestamp_time")
            .and_then(Exact::parse)
            .ok_or_else(unavailable)?;
        let samples: i128 = get("nb_samples")
            .and_then(|v| v.parse().ok())
            .ok_or_else(unavailable)?;
        if samples <= 0 {
            return Err(VideoError::value("Audio frame has no samples"));
        }
        let origin = *first.get_or_insert(start);
        if start.sub(origin.add(Exact::new(count, i128::from(rate)))).abs() > slack {
            return Err(VideoError::value(
                "Discontinuous audio timestamps require a different processing path",
            ));
        }
        count += samples;
    }
    let first = first.ok_or_else(|| VideoError::value("Audio track has no decoded frames"))?;
    Ok((first, count))
}

type PacketRow = (Exact, [Exact; 3], [String; 2]);

/// `_packets`: framemd5 rows (time base, DTS/PTS/duration in seconds, size and payload hash).
fn packets(source: &Path, index: usize) -> Result<Vec<PacketRow>> {
    let mut command = Command::new(require_video_tool(Tool::Ffmpeg)?);
    command
        .args(["-v", "error", "-nostdin", "-copyts", "-i"])
        .arg(source);
    command.args([
        "-map",
        &format!("0:{index}"),
        "-c",
        "copy",
        "-copytb",
        "1",
        "-f",
        "framemd5",
        "-hash",
        "sha256",
        "-",
    ]);
    let mut tick: Option<Exact> = None;
    let mut rows = Vec::new();
    for line in lines(command)? {
        if let Some(rest) = line.strip_prefix("#tb 0:") {
            tick = Exact::parse(rest);
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tb = tick.ok_or_else(|| VideoError::runtime("Packet verifier did not report a time base"))?;
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 6 {
            continue;
        }
        let value = |i: usize| {
            fields[i]
                .parse::<i128>()
                .map(|v| tb.mul_int(v))
                .map_err(|_| VideoError::runtime("Unreadable packet row"))
        };
        rows.push((
            tb,
            [value(1)?, value(2)?, value(3)?],
            [fields[4].to_owned(), fields[5].to_owned()],
        ));
    }
    Ok(rows)
}

/// `verify_stream_packets` (`_verify_packets`): identical payloads, timing within one tick.
pub fn verify_stream_packets(
    source: &Path,
    output: &Path,
    index: usize,
    output_index: Option<usize>,
) -> Result<()> {
    let before = packets(source, index)?;
    let after = packets(output, output_index.unwrap_or(index))?;
    if before.len() != after.len() {
        return Err(VideoError::runtime(format!(
            "Stream {index} encoded packets changed during audio processing"
        )));
    }
    for (a, b) in before.iter().zip(&after) {
        if a.2 != b.2 {
            return Err(VideoError::runtime(format!(
                "Stream {index} encoded packets changed during audio processing"
            )));
        }
        let tolerances = [b.0, b.0, a.0.max(b.0)];
        if a.1
            .iter()
            .zip(&b.1)
            .zip(tolerances)
            .any(|((x, y), t)| x.sub(*y).abs() > t)
        {
            return Err(VideoError::runtime(format!(
                "Stream {index} packet timing changed during audio processing"
            )));
        }
    }
    Ok(())
}

fn packet_trimming(source: &Path, index: usize) -> Result<Vec<(i64, i64)>> {
    let mut command = Command::new(require_video_tool(Tool::Ffprobe)?);
    command
        .args([
            "-v",
            "error",
            "-select_streams",
            &index.to_string(),
            "-show_packets",
            "-show_entries",
        ])
        .arg("packet=stream_index:packet_side_data=side_data_type,skip_samples,discard_padding")
        .args(["-of", "compact=p=0"])
        .arg(source);
    let mut out = Vec::new();
    for line in lines(command)? {
        if line.is_empty() {
            continue;
        }
        let mut skip = 0;
        let mut discard = 0;
        for part in line.split('|').filter(|p| p.contains('=')) {
            let tail = part.rsplit(':').next().unwrap_or(part);
            if let Some((k, v)) = tail.split_once('=') {
                match k {
                    "skip_samples" => skip = v.parse().unwrap_or(0),
                    "discard_padding" => discard = v.parse().unwrap_or(0),
                    _ => {}
                }
            }
        }
        out.push((skip, discard));
    }
    Ok(out)
}

fn verify_audio_trimming(source: &Path, output: &Path, index: usize) -> Result<()> {
    if packet_trimming(source, index)? != packet_trimming(output, index)? {
        return Err(VideoError::runtime(format!(
            "Stream {index} audio priming or tail trimming changed during audio processing"
        )));
    }
    Ok(())
}

/// Options shared by the single- and multi-stream audio removals.
#[derive(Clone, Debug)]
pub struct VideoAudioOptions {
    pub force: bool,
    pub verify: bool,
    pub method: Method,
    pub bandwidth_kbps: f64,
    pub scheme: Scheme,
    pub vocoder_device: Device,
    pub allow_unverified: bool,
    pub allow_positive: bool,
    pub allow_quality_override: bool,
    pub vendor: Option<Vendor>,
    pub streams: Option<Vec<usize>>,
    pub metadata_markers: Option<AiMetadata>,
}

impl Default for VideoAudioOptions {
    fn default() -> Self {
        Self {
            force: false,
            verify: true,
            method: Method::Auto,
            bandwidth_kbps: 24.0,
            scheme: Scheme::Auto,
            vocoder_device: Device::Auto,
            allow_unverified: false,
            allow_positive: false,
            allow_quality_override: false,
            vendor: None,
            streams: None,
            metadata_markers: None,
        }
    }
}

impl VideoAudioOptions {
    fn args(&self) -> ProcessingArgs {
        ProcessingArgs {
            method: self.method,
            scheme: self.scheme,
            verify: self.verify,
            bandwidth_kbps: self.bandwidth_kbps,
            vocoder_device: self.vocoder_device,
            vendor: self.vendor,
        }
    }

    fn permissions(&self) -> Permissions {
        Permissions {
            force: self.force,
            allow_unverified: self.allow_unverified || self.vendor.is_some(),
            allow_positive: self.allow_positive,
            allow_quality_override: self.allow_quality_override,
        }
    }
}

fn audio_error(error: AudioError) -> VideoError {
    VideoError::Audio(error)
}

fn identify(
    runtime: &AudioRuntime,
    source: &Path,
    stream: usize,
    options: &VideoAudioOptions,
) -> Result<AudioProvenanceReport> {
    let request = IdentifyOptions {
        check_invisible: options.verify,
        stream,
        scheme: options.scheme,
        metadata_markers: None,
    };
    identify_audio(runtime, source, &request).map_err(audio_error)
}

fn skipped_result(
    source: &Path,
    before: AudioProvenanceReport,
    first: Profile,
    options: &VideoAudioOptions,
    caveats: &[String],
) -> VideoAudioRemovalResult {
    VideoAudioRemovalResult {
        source: source.to_path_buf(),
        output: None,
        before,
        after: None,
        frames_processed: 0,
        verified_removed: None,
        skipped: true,
        fidelity: None,
        unchanged_streams: Vec::new(),
        source_audio_start: None,
        output_audio_start: None,
        method: first.as_str().to_owned(),
        bandwidth_kbps: matches!(first, Profile::Encodec | Profile::Combined)
            .then_some(options.bandwidth_kbps),
        vocoder_device: (first != Profile::Spectral).then(|| options.vocoder_device.as_str().to_owned()),
        caveats: caveats.to_vec(),
        quality_passed: None,
    }
}

fn ensure_distinct(source: &Path, output: &Path) -> Result<()> {
    let same = raiw_core::resolve_lenient(output) == source
        || (output.exists() && std::fs::canonicalize(output).ok() == std::fs::canonicalize(source).ok());
    if same {
        return Err(VideoError::value("Video output must differ from the source"));
    }
    Ok(())
}

fn require_contiguous(streams: &[StreamRow]) -> Result<()> {
    if streams.iter().enumerate().any(|(i, s)| row_index(s) != Some(i)) {
        return Err(VideoError::value("Video stream indices must be contiguous"));
    }
    Ok(())
}

/// `remove_video_audio`: process one audio track, copying every other stream.
pub fn remove_video_audio(
    runtime: &AudioRuntime,
    source: &Path,
    output: &Path,
    stream: usize,
    options: &VideoAudioOptions,
    before: Option<AudioProvenanceReport>,
) -> Result<VideoAudioRemovalResult> {
    let source = std::fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    let out_ext = suffix_lower(output);
    if !matches!(out_ext.as_str(), ".mkv" | ".mov" | ".mp4" | ".m4v" | ".webm") {
        return Err(VideoError::value(
            "Audio-only video output must be MKV, MOV, MP4, M4V, or WebM",
        ));
    }
    let lossy = matches!(out_ext.as_str(), ".mp4" | ".m4v" | ".webm");
    let codec_label = if out_ext == ".webm" { "Opus" } else { "AAC" };
    let caveats: Vec<String> = vec![
        CAVEAT_TRACK.to_owned(),
        CAVEAT_METADATA.to_owned(),
        if lossy {
            format!(
                "The selected track uses lossy {codec_label} (160k mono / 256k stereo+); verification and fidelity cover the delivered decoded audio."
            )
        } else {
            PCM_CAVEAT.to_owned()
        },
    ];
    ensure_distinct(&source, output)?;
    let streams = stream_inventory(&source)?;
    require_contiguous(&streams)?;
    let tools = runtime
        .tools
        .as_ref()
        .ok_or_else(|| VideoError::runtime("Audio processing requires ffmpeg and ffprobe"))?;
    let info = raiw_audio::io::probe_audio(tools, &source, stream).map_err(audio_error)?;
    let selected = streams
        .iter()
        .find(|s| row_index(s) == Some(info.index))
        .ok_or_else(|| VideoError::value("Audio stream not found"))?;
    let tick = row_str(selected, "time_base")
        .and_then(Exact::parse)
        .ok_or_else(|| VideoError::value("Audio frame timing is unavailable"))?;
    let (start, count) = timeline(&source, info.index, info.sample_rate, tick)?;
    let permissions = options.permissions();
    let mut profiles = resolve_audio_profiles(runtime, &options.args(), permissions).map_err(audio_error)?;
    let mut before = match before {
        Some(report) => report,
        None => identify(runtime, &source, stream, options)?,
    };
    if let Some(markers) = &options.metadata_markers
        && !markers.is_empty()
        && before.metadata_markers.is_empty()
    {
        before.metadata_markers.clone_from(markers);
    }
    if options.method == Method::Auto {
        profiles = adaptive_auto_profiles(runtime, options.scheme, &before, options.vendor);
    }
    let mut allow_unverified = permissions.allow_unverified;
    if should_auto_scrub(runtime, &before, options.method, options.vendor) {
        allow_unverified = true;
    }
    let unverified = options.force || allow_unverified;
    let Some(&first) = profiles.first() else {
        return Err(VideoError::runtime("No audio processing profile was attempted"));
    };
    if !(unverified || options.allow_positive) && before.watermark_status != WatermarkStatus::Detected {
        return Ok(skipped_result(&source, before, first, options, &caveats));
    }
    if !options.verify
        || before.watermark_status != WatermarkStatus::Detected
        || (!unverified && missing_decoder_reason(&before).is_some())
    {
        profiles.truncate(1);
    }
    let work = tempfile::Builder::new().prefix("raiw-video-audio-").tempdir()?;
    let mut last_error: Option<VideoError> = None;
    for (attempt, profile) in profiles.iter().copied().enumerate() {
        let is_last = attempt + 1 == profiles.len();
        let attempt_permissions = Permissions {
            force: options.force && is_last,
            allow_unverified: allow_unverified && is_last,
            allow_positive: options.allow_positive && is_last,
            allow_quality_override: options.allow_quality_override && is_last,
        };
        let candidate = work.path().join(format!("processed-{attempt}.wav"));
        match attempt_profile(
            runtime,
            &source,
            output,
            &streams,
            &info,
            (start, count),
            profile,
            attempt_permissions,
            options,
            &before,
            &candidate,
            lossy,
            &out_ext,
        ) {
            Ok(mut result) => {
                result.caveats.clone_from(&caveats);
                return Ok(result);
            }
            Err(VideoError::Audio(AudioError::Verification(error))) => {
                last_error = Some(VideoError::Audio(AudioError::Verification(error)));
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_error.unwrap_or_else(|| VideoError::runtime("No audio processing profile was attempted")))
}

#[allow(clippy::too_many_arguments)]
fn attempt_profile(
    runtime: &AudioRuntime,
    source: &Path,
    output: &Path,
    streams: &[StreamRow],
    info: &AudioStreamInfo,
    (start, count): (Exact, i128),
    profile: Profile,
    permissions: Permissions,
    options: &VideoAudioOptions,
    before: &AudioProvenanceReport,
    candidate: &Path,
    lossy: bool,
    out_ext: &str,
) -> Result<VideoAudioRemovalResult> {
    let ordinal = streams
        .iter()
        .filter(|s| row_str(s, "codec_type") == Some("audio"))
        .position(|s| row_index(s) == Some(info.index))
        .unwrap_or(0);
    let request = ProfileRequest {
        profile,
        verify: options.verify,
        stream: ordinal,
        bandwidth_kbps: options.bandwidth_kbps,
        scheme: options.scheme,
        vocoder_device: options.vocoder_device,
        permissions,
        preserve_hf_hz: None,
    };
    let processed =
        remove_audio_profile(runtime, source, candidate, &request, Some(before)).map_err(audio_error)?;
    if processed.frames_processed as i128 != count {
        return Err(VideoError::runtime(
            "Audio timeline and decoded sample-frame counts disagree",
        ));
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let staged = StagedSibling::new(output)?;
    let mut command = Command::new(require_video_tool(Tool::Ffmpeg)?);
    command
        .args(["-v", "error", "-nostdin", "-y", "-copyts", "-i"])
        .arg(source);
    command
        .args([
            "-itsoffset".to_owned(),
            format!("{:.9}", start.to_f64()),
            "-i".to_owned(),
        ])
        .arg(candidate);
    for s in streams {
        let index = row_index(s).unwrap_or(0);
        command.args([
            "-map".to_owned(),
            if index == info.index {
                "1:a:0".to_owned()
            } else {
                format!("0:{index}")
            },
        ]);
    }
    for s in streams {
        let index = row_index(s).unwrap_or(0);
        let flags: Vec<String> = s
            .get("disposition")
            .and_then(Value::as_object)
            .map(|d| {
                d.iter()
                    .filter(|(_, v)| v.as_i64().unwrap_or(0) != 0)
                    .map(|(k, _)| k.clone())
                    .collect()
            })
            .unwrap_or_default();
        let flags = if flags.is_empty() {
            "0".to_owned()
        } else {
            flags.join("+")
        };
        command.args([
            format!("-disposition:{index}"),
            flags,
            format!("-map_metadata:s:{index}"),
            format!("0:s:{index}"),
        ]);
    }
    let codec = if out_ext == ".webm" {
        "libopus"
    } else if lossy {
        "aac"
    } else {
        "pcm_f32le"
    };
    command.args([
        "-c".to_owned(),
        "copy".to_owned(),
        format!("-c:{}", info.index),
        codec.to_owned(),
    ]);
    command.args([
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-avoid_negative_ts",
        "disabled",
    ]);
    if lossy {
        command.args([
            format!("-b:{}", info.index),
            if info.channels == 1 {
                "160k".into()
            } else {
                "256k".into()
            },
        ]);
    }
    command.arg(staged.path());
    let (status, stderr) = run_with_stderr_file(&mut command)?;
    if !status.success() {
        return Err(VideoError::runtime(format!(
            "ffmpeg failed to mux processed audio: {}",
            head_chars(&stderr, 500)
        )));
    }
    let delivered = stream_inventory(staged.path())?;
    let layout = |rows: &[StreamRow]| {
        rows.iter()
            .map(|s| (row_index(s), row_str(s, "codec_type").map(str::to_owned)))
            .collect::<Vec<_>>()
    };
    if layout(streams) != layout(&delivered) {
        return Err(VideoError::runtime(
            "Audio processing changed the video stream layout",
        ));
    }
    let language = |s: &StreamRow| {
        s.get("tags")
            .and_then(|t| t.get("language"))
            .and_then(Value::as_str)
            .unwrap_or("und")
            .to_owned()
    };
    for (a, b) in streams.iter().zip(&delivered) {
        if a.get("disposition") != b.get("disposition") || language(a) != language(b) {
            return Err(VideoError::runtime(
                "Audio processing changed stream language or disposition",
            ));
        }
    }
    let selected = &delivered[info.index];
    let output_tick = row_str(selected, "time_base")
        .and_then(Exact::parse)
        .ok_or_else(|| VideoError::runtime("Delivered audio has no time base"))?;
    let delivered_rate: u32 = row_str(selected, "sample_rate")
        .and_then(|r| r.parse().ok())
        .ok_or_else(|| VideoError::runtime("Delivered audio has no sample rate"))?;
    let (delivered_start, delivered_count) =
        timeline(staged.path(), info.index, delivered_rate, output_tick)?;
    let bad_count = if codec == "libopus" {
        let expected = (count as f64 * f64::from(delivered_rate) / f64::from(info.sample_rate))
            .round_ties_even() as i128;
        (delivered_count - expected).abs() > i128::from((delivered_rate / 100).max(2))
    } else {
        count != delivered_count
    };
    if bad_count || start.sub(delivered_start).abs() > output_tick {
        return Err(VideoError::runtime("Mux changed the selected audio timeline"));
    }
    let after = identify(runtime, staged.path(), ordinal, options)?;
    let verified = verification_outcome(before, &after);
    let tools = runtime
        .tools
        .as_ref()
        .ok_or_else(|| VideoError::runtime("Audio processing requires ffmpeg and ffprobe"))?;
    let fidelity = measure_fidelity(tools, source, info, staged.path(), ordinal).map_err(audio_error)?;
    let quality_ok = audio_quality_passed(&fidelity, lossy);
    enforce_publication_gates(before, &after, verified, quality_ok, permissions).map_err(audio_error)?;
    let unchanged: Vec<usize> = streams
        .iter()
        .filter_map(row_index)
        .filter(|i| *i != info.index)
        .collect();
    for &index in &unchanged {
        verify_stream_packets(source, staged.path(), index, None)?;
        if row_str(&streams[index], "codec_type") == Some("audio") {
            verify_audio_trimming(source, staged.path(), index)?;
        }
    }
    if sha256_file(source)? != before.sha256 {
        return Err(VideoError::runtime(
            "Video source changed during audio processing",
        ));
    }
    let output = staged.commit()?;
    let mut after = after;
    after.source.clone_from(&output);
    Ok(VideoAudioRemovalResult {
        source: source.to_path_buf(),
        output: Some(output),
        before: before.clone(),
        after: Some(after),
        frames_processed: count as usize,
        verified_removed: verified,
        skipped: false,
        fidelity: Some(fidelity),
        unchanged_streams: unchanged,
        source_audio_start: Some(start),
        output_audio_start: Some(delivered_start),
        method: processed.method.as_str().to_owned(),
        bandwidth_kbps: processed.bandwidth_kbps,
        vocoder_device: processed.vocoder_device.map(|d| d.as_str().to_owned()),
        caveats: Vec::new(),
        quality_passed: Some(quality_ok),
    })
}

/// `_resolve_selected_ordinals`.
fn selected_ordinals(streams: Option<&[usize]>, count: usize) -> Result<Vec<usize>> {
    let mut ordinals: Vec<usize> = streams.map_or_else(|| (0..count).collect(), <[usize]>::to_vec);
    ordinals.sort_unstable();
    ordinals.dedup();
    if let Some(bad) = ordinals.iter().find(|o| **o >= count) {
        return Err(VideoError::value(format!(
            "Audio stream ordinal {bad} is out of range for {count} track(s)"
        )));
    }
    Ok(ordinals)
}

fn wants_processing(
    runtime: &AudioRuntime,
    report: &AudioProvenanceReport,
    options: &VideoAudioOptions,
    unverified: bool,
) -> bool {
    unverified
        || options.allow_positive
        || report.watermark_status == WatermarkStatus::Detected
        || should_auto_scrub(runtime, report, options.method, options.vendor)
}

fn target_profile(
    runtime: &AudioRuntime,
    report: &AudioProvenanceReport,
    options: &VideoAudioOptions,
) -> Result<Profile> {
    if options.method == Method::Auto {
        adaptive_auto_profiles(runtime, options.scheme, report, options.vendor)
            .first()
            .copied()
            .ok_or_else(|| VideoError::runtime("No audio processing profile was attempted"))
    } else {
        Profile::try_from_method(options.method).ok_or_else(|| VideoError::value("Unsupported audio method"))
    }
}

trait ProfileFromMethod: Sized {
    fn try_from_method(method: Method) -> Option<Self>;
}

impl ProfileFromMethod for Profile {
    fn try_from_method(method: Method) -> Option<Self> {
        Some(match method {
            Method::Spectral => Self::Spectral,
            Method::Encodec => Self::Encodec,
            Method::Combined => Self::Combined,
            Method::Dac => Self::Dac,
            Method::Snac => Self::Snac,
            Method::Auto => return None,
        })
    }
}

/// `preflight_video_audio_workload`: validate arguments and runtimes before any pixel stage.
pub fn preflight_video_audio_workload(
    runtime: &AudioRuntime,
    source: &Path,
    options: &VideoAudioOptions,
) -> Result<Vec<AudioProvenanceReport>> {
    let source = std::fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    let permissions = options.permissions();
    validate_audio_arguments(&options.args(), permissions).map_err(audio_error)?;
    let streams = stream_inventory(&source)?;
    require_contiguous(&streams)?;
    let audio_streams: Vec<&StreamRow> = streams
        .iter()
        .filter(|s| row_str(s, "codec_type") == Some("audio"))
        .collect();
    if audio_streams.is_empty() {
        return Ok(Vec::new());
    }
    let selected = selected_ordinals(options.streams.as_deref(), audio_streams.len())?;
    resolve_audio_profiles(runtime, &options.args(), permissions).map_err(audio_error)?;
    if let Some(profile) = Profile::try_from_method(options.method) {
        require_profile_runtime(runtime, profile).map_err(audio_error)?;
    }
    if !audio_streams
        .iter()
        .any(|s| s.contains_key("sample_rate") && s.contains_key("channels"))
    {
        return Ok(Vec::new());
    }
    let unverified = permissions.unverified();
    let tools = runtime
        .tools
        .as_ref()
        .ok_or_else(|| VideoError::runtime("Audio processing requires ffmpeg and ffprobe"))?;
    if unverified && !options.verify {
        for &ordinal in &selected {
            let info = raiw_audio::io::probe_audio(tools, &source, ordinal).map_err(audio_error)?;
            let markers = options
                .metadata_markers
                .clone()
                .unwrap_or_else(|| raiw_metadata::get_ai_metadata(&source));
            let synthetic = synthetic_report(&source, info.clone(), markers);
            let profile = target_profile(runtime, &synthetic, options)?;
            require_profile_runtime(runtime, profile).map_err(audio_error)?;
            validate_stream_for_profile(&info, profile).map_err(audio_error)?;
        }
        return Ok(Vec::new());
    }
    let mut inspected = Vec::new();
    for ordinal in 0..audio_streams.len() {
        let mut report = match identify(runtime, &source, ordinal, options) {
            Ok(report) => report,
            Err(error) => {
                if unverified && selected.contains(&ordinal) {
                    let info = raiw_audio::io::probe_audio(tools, &source, ordinal).map_err(audio_error)?;
                    let markers = options
                        .metadata_markers
                        .clone()
                        .unwrap_or_else(|| raiw_metadata::get_ai_metadata(&source));
                    let profile = target_profile(
                        runtime,
                        &synthetic_report(&source, info.clone(), markers),
                        options,
                    )?;
                    require_profile_runtime(runtime, profile).map_err(audio_error)?;
                    validate_stream_for_profile(&info, profile).map_err(audio_error)?;
                    continue;
                }
                return Err(error);
            }
        };
        if let Some(markers) = &options.metadata_markers
            && !markers.is_empty()
            && report.metadata_markers.is_empty()
        {
            report.metadata_markers.clone_from(markers);
        }
        if selected.contains(&ordinal) && wants_processing(runtime, &report, options, unverified) {
            let profile = target_profile(runtime, &report, options)?;
            require_profile_runtime(runtime, profile).map_err(audio_error)?;
            validate_stream_for_profile(&report.stream, profile).map_err(audio_error)?;
        }
        inspected.push(report);
    }
    Ok(if inspected.len() == audio_streams.len() {
        inspected
    } else {
        Vec::new()
    })
}

fn synthetic_report(source: &Path, info: AudioStreamInfo, markers: AiMetadata) -> AudioProvenanceReport {
    AudioProvenanceReport {
        source: source.to_path_buf(),
        sha256: String::new(),
        stream: info,
        metadata_markers: markers,
        watermark_status: WatermarkStatus::Unverified,
        detections: Vec::new(),
        caveats: Vec::new(),
        recommended_recipe: None,
    }
}

/// `remove_video_audio_all_streams`: clean every source-positive track in one staged transaction.
pub fn remove_video_audio_all_streams(
    runtime: &AudioRuntime,
    source: &Path,
    output: &Path,
    options: &VideoAudioOptions,
    befores: Option<Vec<AudioProvenanceReport>>,
) -> Result<VideoAudioMultiStreamResult> {
    let source = std::fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    let ext = suffix_lower(output);
    if !matches!(ext.as_str(), ".mkv" | ".mov" | ".mp4" | ".m4v" | ".webm") {
        return Err(VideoError::value(
            "Audio-only video output must be MKV, MOV, MP4, M4V, or WebM",
        ));
    }
    ensure_distinct(&source, output)?;
    let permissions = options.permissions();
    let profiles = resolve_audio_profiles(runtime, &options.args(), permissions).map_err(audio_error)?;
    let streams = stream_inventory(&source)?;
    require_contiguous(&streams)?;
    let audio_indices: Vec<usize> = streams
        .iter()
        .filter(|s| row_str(s, "codec_type") == Some("audio"))
        .filter_map(row_index)
        .collect();
    if audio_indices.is_empty() {
        return Err(VideoError::value("Source has no audio stream"));
    }
    let selected = selected_ordinals(options.streams.as_deref(), audio_indices.len())?;
    let lossy = matches!(ext.as_str(), ".mp4" | ".m4v" | ".webm");
    let caveats: Vec<String> = vec![
        "All source-positive audio tracks are processed; clean tracks and video marks remain unchanged."
            .to_owned(),
        CAVEAT_METADATA.to_owned(),
        if lossy {
            "Processed tracks use lossy AAC with a target bitrate of 192 kbps; verification and fidelity cover the delivered decoded audio.".to_owned()
        } else {
            "Processed tracks are stored as uncompressed float PCM, which can increase file size.".to_owned()
        },
    ];
    let source_sha = sha256_file(&source)?;
    let mut befores: Vec<AudioProvenanceReport> = match befores {
        Some(reports) if reports.len() == audio_indices.len() => reports
            .into_iter()
            .map(|mut r| {
                r.source.clone_from(&source);
                r.sha256.clone_from(&source_sha);
                r
            })
            .collect(),
        _ => (0..audio_indices.len())
            .map(|o| identify(runtime, &source, o, options))
            .collect::<Result<_>>()?,
    };
    if let Some(markers) = &options.metadata_markers
        && !markers.is_empty()
    {
        for report in &mut befores {
            if report.metadata_markers.is_empty() {
                report.metadata_markers.clone_from(markers);
            }
        }
    }
    let unverified = permissions.unverified();
    let to_process: Vec<usize> = befores
        .iter()
        .enumerate()
        .filter(|(o, r)| {
            selected.contains(o)
                && (unverified
                    || options.force
                    || options.allow_positive
                    || wants_processing(runtime, r, options, false))
        })
        .map(|(o, _)| o)
        .collect();
    let first = profiles.first().copied().unwrap_or(Profile::Spectral);
    if to_process.is_empty() {
        return Ok(VideoAudioMultiStreamResult {
            source: source.clone(),
            output: None,
            streams: befores
                .into_iter()
                .map(|r| skipped_result(&source, r, first, options, &caveats))
                .collect(),
            processed_ordinals: Vec::new(),
            unchanged_streams: (0..streams.len()).collect(),
            verified_removed: None,
            skipped: true,
            caveats,
            quality_passed: None,
        });
    }
    let processed_global: Vec<usize> = to_process.iter().map(|o| audio_indices[*o]).collect();
    let unchanged: Vec<usize> = streams
        .iter()
        .filter_map(row_index)
        .filter(|i| !processed_global.contains(i))
        .collect();
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let staged = StagedSibling::new(output)?;
    let mut processed: Vec<(usize, VideoAudioRemovalResult)> = Vec::new();
    let single = to_process.len() == 1;
    {
        let work = tempfile::Builder::new()
            .prefix("raiw-video-audio-multi-")
            .tempdir()?;
        let mut current = source.clone();
        for (step, &ordinal) in to_process.iter().enumerate() {
            let is_last = step + 1 == to_process.len();
            let target = if is_last {
                staged.path().to_path_buf()
            } else {
                work.path().join(format!("step-{step}{ext}"))
            };
            let mut before = befores[ordinal].clone();
            if step > 0 {
                before.source.clone_from(&current);
                before.sha256 = sha256_file(&current)?;
            }
            let result = remove_video_audio(runtime, &current, &target, ordinal, options, Some(before))?;
            if result.skipped || result.output.is_none() {
                return Err(VideoError::runtime(if single {
                    "Selected audio stream was unexpectedly skipped during multi-stream processing".to_owned()
                } else {
                    format!(
                        "Selected audio stream {ordinal} was unexpectedly skipped during multi-stream processing"
                    )
                }));
            }
            processed.push((ordinal, result));
            current = target;
        }
        if !single {
            for &index in &unchanged {
                verify_stream_packets(&source, staged.path(), index, None)?;
                if row_str(&streams[index], "codec_type") == Some("audio") {
                    verify_audio_trimming(&source, staged.path(), index)?;
                }
            }
        }
    }
    let final_output = output.to_path_buf();
    let tools = runtime
        .tools
        .as_ref()
        .ok_or_else(|| VideoError::runtime("Audio processing requires ffmpeg and ffprobe"))?;
    let last_processed = *to_process.last().unwrap_or(&0);
    let mut items = Vec::new();
    for (ordinal, before) in befores.iter().enumerate() {
        let mut clean_before = before.clone();
        clean_before.source.clone_from(&source);
        let Some((_, prev)) = processed.iter().find(|(o, _)| *o == ordinal) else {
            let mut item = skipped_result(&source, clean_before, first, options, &caveats);
            item.output = Some(final_output.clone());
            item.unchanged_streams.clone_from(&unchanged);
            item.bandwidth_kbps = None;
            item.vocoder_device = None;
            items.push(item);
            continue;
        };
        let mut item = prev.clone();
        item.source.clone_from(&source);
        item.output = Some(final_output.clone());
        item.before = clean_before;
        item.unchanged_streams.clone_from(&unchanged);
        item.caveats.clone_from(&caveats);
        if !(ordinal == last_processed && single) {
            let mut after = if ordinal == last_processed && prev.after.is_some() {
                prev.after.clone().unwrap_or_else(|| before.clone())
            } else {
                identify(runtime, staged.path(), ordinal, options)?
            };
            let verified = verification_outcome(before, &after);
            let info = raiw_audio::io::probe_audio(tools, &source, ordinal).map_err(audio_error)?;
            let fidelity =
                measure_fidelity(tools, &source, &info, staged.path(), ordinal).map_err(audio_error)?;
            let quality_ok = audio_quality_passed(&fidelity, lossy);
            let gate = Permissions {
                allow_unverified: permissions.allow_unverified
                    || should_auto_scrub(runtime, before, options.method, options.vendor),
                ..permissions
            };
            enforce_publication_gates(before, &after, verified, quality_ok, gate).map_err(audio_error)?;
            after.source.clone_from(&final_output);
            item.after = Some(after);
            item.verified_removed = verified;
            item.fidelity = Some(fidelity);
            item.quality_passed = Some(quality_ok);
        } else if let Some(after) = &mut item.after {
            after.source.clone_from(&final_output);
        }
        items.push(item);
    }
    if sha256_file(&source)? != source_sha {
        return Err(VideoError::runtime(
            "Video source changed during multi-stream audio processing",
        ));
    }
    staged.commit()?;
    let active: Vec<&VideoAudioRemovalResult> = to_process.iter().map(|o| &items[*o]).collect();
    let inactive: Vec<&VideoAudioRemovalResult> = items
        .iter()
        .enumerate()
        .filter(|(o, _)| !to_process.contains(o))
        .map(|(_, i)| i)
        .collect();
    let verified = if active.iter().any(|r| r.verified_removed == Some(false))
        || inactive
            .iter()
            .any(|i| i.before.watermark_status == WatermarkStatus::Detected)
    {
        Some(false)
    } else if active.iter().all(|r| r.verified_removed == Some(true))
        && inactive
            .iter()
            .all(|i| i.before.watermark_status == WatermarkStatus::NotDetected)
    {
        Some(true)
    } else {
        None
    };
    let quality = (!active.is_empty()).then(|| active.iter().all(|r| r.quality_passed == Some(true)));
    Ok(VideoAudioMultiStreamResult {
        source,
        output: Some(final_output),
        streams: items,
        processed_ordinals: to_process,
        unchanged_streams: unchanged,
        verified_removed: verified,
        skipped: false,
        caveats,
        quality_passed: quality,
    })
}

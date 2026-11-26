//! Cleaned-frame encoding with timing preservation and selective GOP stream-copy
//! (`encode_clean_video`, `_try_selective_gop_clean_encode`).

use std::path::Path;

use raiw_visible::eraser::FillBackends;

use super::clean::{CleanOptions, for_each_clean_frame};
use super::scan::VideoScan;
use super::select::FrameRegions;
use crate::decode::frame_bytes;
use crate::encode::{
    MuxOptions, RawVideoOptions, StagedVideoOutput, command_from, container_format_args, mp4_timescale_args,
    mux_encoded_video, raw_video_command,
};
use crate::error::{Result, VideoError};
use crate::nut::{NutPixelFormat, TimestampedNutWriter};
use crate::probe::{
    Rational, VideoEncodeProfile, count_decoded_video_frames, probe_video_encode_profile,
    probe_video_keyframes,
};
use crate::process::{RawVideoEncoder, run_with_stderr_file};
use crate::temporal::VideoPixel;
use crate::tools::{Tool, require_video_tool};

const HDR_TRANSFERS: [&str; 2] = ["smpte2084", "arib-std-b67"];

/// Options for [`encode_clean_video`].
#[derive(Clone, Copy, Debug)]
pub struct EncodeCleanOptions<'a> {
    pub clean: CleanOptions<'a>,
    pub strip_metadata: bool,
    pub trim_end_frame: Option<usize>,
    pub crf: u32,
}

/// `_timestamp_time_base`: the source time base, else microseconds.
pub fn timestamp_time_base(profile_time_base: Option<&str>) -> Rational {
    profile_time_base.and_then(Rational::parse).unwrap_or(Rational {
        num: 1,
        den: 1_000_000,
    })
}

/// `_timestamps_are_variable`: valid, strictly increasing, non-uniform intervals.
pub fn timestamps_are_variable(scan: &VideoScan, time_base: Rational) -> bool {
    if scan.timestamps.len() != scan.detections.len() || scan.timestamps.len() < 3 {
        return false;
    }
    let tb = time_base.to_f64();
    let ticks: Vec<i64> = scan
        .timestamps
        .iter()
        .map(|t| crate::py::round_int(t / tb))
        .collect();
    let intervals: Vec<i64> = ticks.windows(2).map(|w| w[1] - w[0]).collect();
    let first = intervals[0];
    intervals.iter().all(|i| *i > 0) && intervals.iter().any(|i| *i != first)
}

/// `_source_has_bitstream_ai_sei`: in-band AI SEI/OBU payloads a stream copy would preserve.
fn source_has_bitstream_ai_sei(source: &Path) -> bool {
    match raiw_core::fs::suffix_lower(source).as_str() {
        ".mp4" | ".mov" | ".m4v" => raiw_container::isobmff::scan_video_sei(source, true).is_ok_and(|s| {
            !s.kling_tag_offsets.is_empty() || !s.ai_sei_spans.is_empty() || !s.av1_obu_patches.is_empty()
        }),
        ".mkv" | ".webm" => !raiw_container::ebml::scan_av1_ai_obus(source, true).is_empty(),
        _ => false,
    }
}

fn ffmpeg_ok(args: &[String]) -> Result<bool> {
    let mut command = command_from(args)?;
    Ok(run_with_stderr_file(&mut command)?.0.success())
}

struct GopRequest<'a> {
    source: &'a Path,
    temporary_output: &'a Path,
    scan: &'a VideoScan,
    regions: &'a [FrameRegions],
    options: &'a EncodeCleanOptions<'a>,
    trim_end_seconds: Option<f64>,
    profile: &'a VideoEncodeProfile,
}

fn video_copy_segment(
    ffmpeg: &str,
    input_args: &[String],
    output: &Path,
    timescale: &[String],
) -> Vec<String> {
    let mut cmd = vec![ffmpeg.to_owned(), "-y".into(), "-loglevel".into(), "error".into()];
    cmd.extend_from_slice(input_args);
    cmd.extend(["-map".into(), "0:v:0".into(), "-c:v".into(), "copy".into()]);
    cmd.extend_from_slice(timescale);
    cmd.push(output.to_string_lossy().into_owned());
    cmd
}

/// `_try_selective_gop_clean_encode`: copy untouched H.264 GOPs when verified frame-exact.
fn try_selective_gop(request: &GopRequest<'_>, fills: &FillBackends) -> Result<Option<usize>> {
    let GopRequest {
        source,
        temporary_output,
        scan,
        regions,
        options,
        trim_end_seconds,
        profile,
    } = *request;
    let total = regions.len();
    let suffix = raiw_core::fs::suffix_lower(temporary_output);
    if total < 15
        || profile.pixel_format != "yuv420p"
        || !matches!(suffix.as_str(), ".mp4" | ".mov" | ".m4v" | ".mkv")
    {
        return Ok(None);
    }
    let dirty: Vec<usize> = (0..total)
        .filter(|&i| regions[i].iter().any(Option::is_some))
        .collect();
    let radius = if options.clean.temporal_consistency { 12 } else { 0 };
    match (dirty.first(), dirty.last()) {
        (None, _) | (_, None) if options.trim_end_frame.is_none() => return Ok(None),
        (Some(&first), Some(&last)) => {
            let early_start = first.saturating_sub(radius);
            let early_end = (total - 1).min(last + radius);
            if early_start < 30 && (options.trim_end_frame.is_some() || early_end + 30 >= total) {
                return Ok(None);
            }
        }
        _ => {}
    }
    if options.strip_metadata && source_has_bitstream_ai_sei(source) {
        return Ok(None);
    }
    let Some(spec) = probe_video_keyframes(source)? else {
        return Ok(None);
    };
    if !matches!(spec.start_pts, None | Some(0)) {
        return Ok(None);
    }
    let ffmpeg = require_video_tool(Tool::Ffmpeg)?.to_string_lossy().into_owned();
    let isobmff = matches!(suffix.as_str(), ".mp4" | ".mov" | ".m4v");
    let policy = if options.strip_metadata { "-1" } else { "0" };
    if dirty.is_empty()
        && let (Some(trim), Some(seconds)) = (options.trim_end_frame, trim_end_seconds)
        && spec.has_b_frames == 0
        && spec.keyframes.iter().any(|(i, _)| *i == trim)
    {
        let mut cmd = vec![
            ffmpeg.clone(),
            "-y".into(),
            "-loglevel".into(),
            "error".into(),
            "-i".into(),
        ];
        cmd.push(source.to_string_lossy().into_owned());
        cmd.extend([
            "-frames:v".into(),
            trim.to_string(),
            "-to".into(),
            format!("{seconds:.6}"),
        ]);
        cmd.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c",
                "copy",
                "-map_metadata",
                policy,
                "-map_chapters",
                policy,
            ]
            .map(String::from),
        );
        if isobmff {
            cmd.extend(["-movflags".into(), "+faststart".into()]);
        }
        cmd.extend(container_format_args(temporary_output));
        cmd.push(temporary_output.to_string_lossy().into_owned());
        if ffmpeg_ok(&cmd)? && count_decoded_video_frames(temporary_output)? == Some(trim as u64) {
            tracing::info!("Selective GOP stream-copy: pure keyframe bumper trim ({trim} frames)");
            return Ok(Some(0));
        }
        let _ = std::fs::remove_file(temporary_output);
    }
    if spec.codec_name != "h264" || spec.has_b_frames > 0 {
        return Ok(None);
    }
    if options.trim_end_frame.is_none() && spec.packet_count != total {
        return Ok(None);
    }
    let prefix_limit = dirty.first().map_or(total - 1, |f| f.saturating_sub(radius));
    let prefix_end = spec
        .keyframes
        .iter()
        .filter(|(i, _)| (30..=prefix_limit).contains(i))
        .map(|(i, _)| *i)
        .next_back()
        .unwrap_or(0);
    let mut suffix_start: Option<(usize, f64)> = None;
    if let Some(&last) = dirty.last()
        && options.trim_end_frame.is_none()
    {
        let active_end = (total - 1).min(last + radius);
        suffix_start = spec
            .keyframes
            .iter()
            .copied()
            .find(|(i, _)| active_end < *i && *i + 30 <= total);
    }
    if prefix_end < 30 && suffix_start.is_none() {
        return Ok(None);
    }
    let mid_start = prefix_end;
    let mid_end = suffix_start.map_or(total, |(i, _)| i);
    let h264_profile = match spec.profile_name.as_deref() {
        Some("Constrained Baseline" | "Baseline") => "baseline",
        Some("Main") => "main",
        _ => "high",
    };
    let seg_ext = if suffix == ".mkv" { ".mkv" } else { ".mp4" };
    let timescale = mp4_timescale_args(Path::new(&format!("seg{seg_ext}")), profile.time_base.as_deref());
    let parent = temporary_output.parent().unwrap_or_else(|| Path::new("."));
    let dir = tempfile::Builder::new().prefix(".raiw-gop-").tempdir_in(parent)?;
    let mut segments = Vec::new();
    if prefix_end >= 30 {
        let segment = dir.path().join(format!("seg_prefix{seg_ext}"));
        let input = vec![
            "-i".into(),
            source.to_string_lossy().into_owned(),
            "-frames:v".into(),
            prefix_end.to_string(),
        ];
        if !ffmpeg_ok(&video_copy_segment(&ffmpeg, &input, &segment, &timescale))? {
            return Ok(None);
        }
        segments.push(segment);
    }
    let mut removed = 0usize;
    if mid_end > mid_start {
        let segment = dir.path().join(format!("seg_mid{seg_ext}"));
        let mut raw = RawVideoOptions::new(scan.width, scan.height, scan.fps, options.crf, profile);
        raw.h264_profile = (seg_ext == ".mp4").then_some(h264_profile);
        let mut encoder = RawVideoEncoder::start(command_from(&raw_video_command(&segment, &raw)?)?)?;
        let mut written = 0usize;
        let clean = CleanOptions {
            start_frame: mid_start,
            stop_frame: Some(mid_end),
            ..options.clean
        };
        let outcome =
            for_each_clean_frame::<u8>(source, scan, regions, fills, &clean, |frame, was_removed| {
                if was_removed {
                    removed += 1;
                }
                written += 1;
                encoder.write_frame(&frame_bytes(&frame))
            });
        if outcome.is_err() {
            encoder.abort();
            return Ok(None);
        }
        if encoder.finish(&segment, "selective GOP middle encode").is_err() || written != mid_end - mid_start
        {
            return Ok(None);
        }
        segments.push(segment);
    }
    if let Some((_, seconds)) = suffix_start {
        let segment = dir.path().join(format!("seg_suffix{seg_ext}"));
        let input = vec![
            "-ss".into(),
            format!("{seconds:.6}"),
            "-i".into(),
            source.to_string_lossy().into_owned(),
        ];
        if !ffmpeg_ok(&video_copy_segment(&ffmpeg, &input, &segment, &timescale))? {
            return Ok(None);
        }
        segments.push(segment);
    }
    let list = dir.path().join("concat.txt");
    let mut lines = String::new();
    for segment in &segments {
        use std::fmt::Write as _;
        let _ = writeln!(
            lines,
            "file '{}'",
            segment.to_string_lossy().replace('\'', "'\\''")
        );
    }
    std::fs::write(&list, lines)?;
    let mux = MuxOptions {
        strip_metadata: options.strip_metadata,
        copy_input_timestamps: false,
        trim_end_seconds,
        video_input_args: &["-f", "concat", "-safe", "0"],
    };
    if mux_encoded_video(&list, source, temporary_output, &mux).is_ok()
        && count_decoded_video_frames(temporary_output)? == Some(total as u64)
    {
        tracing::info!(
            "Selective GOP stream-copy succeeded: re-encoded [{mid_start}, {mid_end}) of {total} frames"
        );
        return Ok(Some(removed));
    }
    let _ = std::fs::remove_file(temporary_output);
    Ok(None)
}

/// `encode_clean_video`: decode again, fill accepted regions, and atomically encode with audio.
pub fn encode_clean_video(
    source: &Path,
    output: &Path,
    scan: &VideoScan,
    regions: &[FrameRegions],
    fills: &FillBackends,
    options: &EncodeCleanOptions<'_>,
) -> Result<usize> {
    if regions.len() != scan.detections.len() {
        return Err(VideoError::value(
            "Temporal localization count does not match the scanned frame count",
        ));
    }
    let mut active_regions = regions;
    let mut active_scan = scan.clone();
    let mut trim_end_seconds = None;
    if let Some(trim) = options.trim_end_frame {
        active_regions = &regions[..trim.min(regions.len())];
        if trim < scan.detections.len() {
            trim_end_seconds = if scan.timestamps.len() > trim {
                Some(scan.timestamps[trim])
            } else if scan.fps > 0.0 {
                Some(trim as f64 / scan.fps)
            } else {
                None
            };
            active_scan.detections.truncate(trim);
            active_scan.timestamps.truncate(trim);
        }
    }
    let staged = StagedVideoOutput::new(output)?;
    let mut profile = probe_video_encode_profile(source)?;
    let high_depth = profile.high_depth();
    if (profile.component_depth.unwrap_or(0) > 8
        || profile
            .color_transfer
            .as_deref()
            .is_some_and(|t| HDR_TRANSFERS.contains(&t)))
        && !high_depth
    {
        return Err(VideoError::runtime(format!(
            "Visible video removal currently supports SDR 8-bit input only; refusing to silently reduce {} / {} to 8-bit SDR",
            profile
                .source_pixel_format
                .as_deref()
                .unwrap_or("unknown pixel format"),
            profile.color_transfer.as_deref().unwrap_or("unknown transfer")
        )));
    }
    let time_base = timestamp_time_base(profile.time_base.as_deref());
    let preserve_start = !matches!(profile.start_pts, None | Some(0));
    let timestamped = preserve_start || timestamps_are_variable(&active_scan, time_base);
    if timestamped && profile.time_base.is_none() {
        profile.time_base = Some(time_base.to_string());
    }
    if !timestamped && !high_depth {
        let request = GopRequest {
            source,
            temporary_output: staged.final_output.path(),
            scan: &active_scan,
            regions: active_regions,
            options,
            trim_end_seconds,
            profile: &profile,
        };
        if let Some(removed) = try_selective_gop(&request, fills)? {
            staged.commit()?;
            return Ok(removed);
        }
    }
    let pix_fmt = if high_depth { "bgr48le" } else { "bgr24" };
    let mut raw = RawVideoOptions::new(
        active_scan.width,
        active_scan.height,
        active_scan.fps,
        options.crf,
        &profile,
    );
    raw.timestamped_input = timestamped;
    raw.copy_input_timestamps = preserve_start;
    raw.input_pix_fmt = pix_fmt;
    let encoded = staged.encoded_video.path().to_path_buf();
    let mut encoder = RawVideoEncoder::start(command_from(&raw_video_command(&encoded, &raw)?)?)?;
    let removed = if high_depth {
        stream_frames::<u16>(
            source,
            &active_scan,
            active_regions,
            fills,
            options,
            &mut encoder,
            timestamped,
            time_base,
            &profile,
        )
    } else {
        stream_frames::<u8>(
            source,
            &active_scan,
            active_regions,
            fills,
            options,
            &mut encoder,
            timestamped,
            time_base,
            &profile,
        )
    };
    let removed = match removed {
        Ok(removed) => removed,
        Err(error) => {
            encoder.abort();
            return Err(error);
        }
    };
    encoder.finish(&encoded, "visible-watermark encode")?;
    let mux = MuxOptions {
        strip_metadata: options.strip_metadata,
        copy_input_timestamps: preserve_start,
        trim_end_seconds,
        video_input_args: &[],
    };
    mux_encoded_video(&encoded, source, staged.final_output.path(), &mux)?;
    staged.commit()?;
    Ok(removed)
}

#[allow(clippy::too_many_arguments)]
fn stream_frames<T: VideoPixel>(
    source: &Path,
    scan: &VideoScan,
    regions: &[FrameRegions],
    fills: &FillBackends,
    options: &EncodeCleanOptions<'_>,
    encoder: &mut RawVideoEncoder,
    timestamped: bool,
    time_base: Rational,
    profile: &VideoEncodeProfile,
) -> Result<usize> {
    let mut removed = 0usize;
    if timestamped {
        let pix = if T::BYTES == 1 {
            NutPixelFormat::Bgr24
        } else {
            NutPixelFormat::Bgr48Le
        };
        let pipe = encoder.stdin()?;
        let mut writer = TimestampedNutWriter::new(
            pipe,
            scan.width,
            scan.height,
            time_base,
            profile.start_pts.unwrap_or(0),
            pix,
        )?;
        let mut index = 0usize;
        for_each_clean_frame::<T>(
            source,
            scan,
            regions,
            fills,
            &options.clean,
            |frame, was_removed| {
                if was_removed {
                    removed += 1;
                }
                let ts = scan
                    .timestamps
                    .get(index)
                    .copied()
                    .ok_or_else(|| VideoError::runtime("missing frame timestamp"))?;
                index += 1;
                writer.write(&frame_bytes(&frame), ts)
            },
        )?;
        writer.finish()?;
    } else {
        for_each_clean_frame::<T>(
            source,
            scan,
            regions,
            fills,
            &options.clean,
            |frame, was_removed| {
                if was_removed {
                    removed += 1;
                }
                encoder.write_frame(&frame_bytes(&frame))
            },
        )?;
    }
    Ok(removed)
}

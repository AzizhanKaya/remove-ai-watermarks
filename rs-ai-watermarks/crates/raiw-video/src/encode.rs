//! ffmpeg encode and mux commands (the encoding half of `video_encoding.py`).
//!
//! Frames are encoded by a single-input ffmpeg reading stdin; a second, finite stream-copy mux
//! adds the source audio and applies the metadata/chapter policy. Both write sibling temporaries
//! and only the finished mux is published.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use raiw_core::StagedSibling;
use raiw_core::constants::VIDEO_ENCODERS;
use raiw_core::fs::suffix_lower;
use serde::Serialize;
use serde_json::Value;

use crate::error::{Result, VideoError};
use crate::probe::{Rational, VideoEncodeProfile};
use crate::process::{head_chars, run_with_stderr_file};
use crate::py::format_g;
use crate::tools::{Tool, require_video_tool};

const ISOBMFF: [&str; 3] = [".mp4", ".mov", ".m4v"];

/// The resolved video encoder family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HwEncoder {
    Software,
    Nvenc,
    VideoToolbox,
}

type EncoderCache = Mutex<Vec<(PathBuf, HashSet<String>)>>;

/// `_available_ffmpeg_encoders`: video encoder names, cached per ffmpeg path.
fn available_ffmpeg_encoders(ffmpeg: &Path) -> HashSet<String> {
    static CACHE: OnceLock<EncoderCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(guard) = cache.lock()
        && let Some((_, names)) = guard.iter().find(|(path, _)| path == ffmpeg)
    {
        return names.clone();
    }
    let mut names = HashSet::new();
    if let Ok(output) = Command::new(ffmpeg).args(["-hide_banner", "-encoders"]).output()
        && output.status.success()
    {
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let stripped = line.trim();
            if stripped.len() > 8 && stripped.starts_with('V') {
                let mut parts = stripped.split_whitespace();
                if let (Some(_), Some(name)) = (parts.next(), parts.next()) {
                    names.insert(name.to_owned());
                }
            }
        }
    }
    if let Ok(mut guard) = cache.lock() {
        guard.push((ffmpeg.to_path_buf(), names.clone()));
    }
    names
}

/// `_resolve_hw_codec`.
fn resolve_hw_codec(encoder: &str, ffmpeg: &Path, high_depth: bool) -> (HwEncoder, &'static str) {
    if encoder == "software" {
        return (HwEncoder::Software, "");
    }
    let available = available_ffmpeg_encoders(ffmpeg);
    let vt = if high_depth {
        "hevc_videotoolbox"
    } else {
        "h264_videotoolbox"
    };
    let nv = if high_depth { "hevc_nvenc" } else { "h264_nvenc" };
    match encoder {
        "nvenc" | "auto" if available.contains(nv) => (HwEncoder::Nvenc, nv),
        "videotoolbox" | "auto" if available.contains(vt) => (HwEncoder::VideoToolbox, vt),
        _ => (HwEncoder::Software, ""),
    }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// `_video_codec_args`.
pub fn video_codec_args(
    suffix: &str,
    crf: u32,
    profile: &VideoEncodeProfile,
    encoder: &str,
    ffmpeg: &Path,
    h264_profile: Option<&str>,
) -> Vec<String> {
    let high_depth = profile.high_depth();
    let isobmff = ISOBMFF.contains(&suffix);
    if suffix == ".webm" {
        let mut args = strings(&[
            "-c:v",
            "libvpx-vp9",
            "-row-mt",
            "1",
            "-deadline",
            "good",
            "-cpu-used",
            "2",
        ]);
        if crf == 0 {
            args.extend(strings(&["-lossless", "1"]));
        } else {
            args.extend([
                "-crf".to_owned(),
                crf.to_string(),
                "-b:v".to_owned(),
                "0".to_owned(),
            ]);
        }
        if high_depth {
            args.extend(strings(&["-profile:v", "2"]));
        }
        return args;
    }
    let (mode, codec) = if encoder != "software"
        && crf > 0
        && matches!(profile.pixel_format.as_str(), "yuv420p" | "yuv420p10le")
    {
        resolve_hw_codec(encoder, ffmpeg, high_depth)
    } else {
        (HwEncoder::Software, "")
    };
    match mode {
        HwEncoder::VideoToolbox => {
            let quality = (100.0 - f64::from(crf) * 2.0).round_ties_even().clamp(35.0, 92.0) as i64;
            let mut args = vec![
                "-c:v".into(),
                codec.into(),
                "-allow_sw".into(),
                "1".into(),
                "-q:v".into(),
                quality.to_string(),
            ];
            if high_depth && isobmff {
                args.extend(strings(&["-tag:v", "hvc1"]));
            }
            return args;
        }
        HwEncoder::Nvenc => {
            let mut args = vec![
                "-c:v".into(),
                codec.into(),
                "-preset".into(),
                "p4".into(),
                "-rc".into(),
                "vbr".into(),
                "-cq".into(),
                crf.to_string(),
                "-b:v".into(),
                "0".into(),
            ];
            if high_depth && isobmff {
                args.extend(strings(&["-tag:v", "hvc1"]));
            }
            return args;
        }
        HwEncoder::Software => {}
    }
    if high_depth {
        let mut args = vec![
            "-c:v".into(),
            "libx265".into(),
            "-preset".into(),
            "medium".into(),
            "-crf".into(),
            crf.to_string(),
        ];
        if isobmff {
            args.extend(strings(&["-tag:v", "hvc1"]));
        }
        let mut params = vec!["log-level=error".to_owned()];
        if crf == 0 {
            params.push("lossless=1".into());
        }
        if let Some(p) = profile
            .color_primaries
            .as_deref()
            .filter(|p| matches!(*p, "bt2020" | "bt709"))
        {
            params.push(format!("colorprim={p}"));
        }
        if let Some(t) = profile
            .color_transfer
            .as_deref()
            .filter(|t| matches!(*t, "smpte2084" | "arib-std-b67" | "bt709" | "bt2020-10"))
        {
            params.push(format!("transfer={t}"));
        }
        if let Some(m) = profile
            .color_space
            .as_deref()
            .filter(|m| matches!(*m, "bt2020nc" | "bt709"))
        {
            params.push(format!("colormatrix={m}"));
        }
        if let Some(range) = profile.color_range.as_deref() {
            params.push(format!(
                "range={}",
                if range == "pc" { "full" } else { "limited" }
            ));
        }
        args.extend(["-x265-params".to_owned(), params.join(":")]);
        return args;
    }
    let mut args = vec![
        "-c:v".into(),
        "libx264".into(),
        "-preset".into(),
        "medium".into(),
        "-crf".into(),
        crf.to_string(),
    ];
    if let Some(h264) = h264_profile {
        args.extend(["-profile:v".to_owned(), h264.to_owned()]);
    }
    let mut params = Vec::new();
    if profile.color_primaries.as_deref() == Some("bt709") {
        params.push("colorprim=bt709".to_owned());
    }
    if profile.color_transfer.as_deref() == Some("bt709") {
        params.push("transfer=bt709".to_owned());
    }
    if profile.color_space.as_deref() == Some("bt709") {
        params.push("colormatrix=bt709".to_owned());
    }
    if let Some(range) = profile.color_range.as_deref() {
        params.push(format!(
            "range={}",
            if range == "pc" { "full" } else { "limited" }
        ));
    }
    if !params.is_empty() {
        args.extend(["-x264-params".to_owned(), params.join(":")]);
    }
    args
}

/// `_profile_args`: generic output options for source properties ffmpeg understands.
pub fn profile_args(profile: &VideoEncodeProfile) -> Vec<String> {
    let mut args = vec!["-pix_fmt".to_owned(), profile.pixel_format.clone()];
    for (option, value) in [
        ("-color_range", &profile.color_range),
        ("-colorspace", &profile.color_space),
        ("-color_trc", &profile.color_transfer),
        ("-color_primaries", &profile.color_primaries),
    ] {
        if let Some(value) = value {
            args.extend([option.to_owned(), value.clone()]);
        }
    }
    if let Some(time_base) = &profile.time_base {
        args.extend(["-enc_time_base:v".to_owned(), time_base.clone()]);
    }
    args
}

/// `container_format_args`: `-f mp4` where extension detection is restrictive.
pub fn container_format_args(path: &Path) -> Vec<String> {
    if suffix_lower(path) == ".m4v" {
        strings(&["-f", "mp4"])
    } else {
        Vec::new()
    }
}

/// `mp4_timescale_args`: the ISOBMFF track timescale when the time base has unit numerator.
pub fn mp4_timescale_args(path: &Path, time_base: Option<&str>) -> Vec<String> {
    if ISOBMFF.contains(&suffix_lower(path).as_str())
        && let Some((num, den)) = time_base.and_then(|tb| tb.split_once('/'))
        && num.trim().parse::<i64>() == Ok(1)
        && let Ok(den) = den.trim().parse::<i64>()
    {
        return vec!["-video_track_timescale".to_owned(), den.to_string()];
    }
    Vec::new()
}

/// Options for [`raw_video_command`].
#[derive(Clone, Debug)]
pub struct RawVideoOptions<'a> {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub crf: u32,
    pub profile: &'a VideoEncodeProfile,
    pub timestamped_input: bool,
    pub copy_input_timestamps: bool,
    pub input_pix_fmt: &'a str,
    pub encoder: &'a str,
    pub h264_profile: Option<&'a str>,
}

impl<'a> RawVideoOptions<'a> {
    pub fn new(width: usize, height: usize, fps: f64, crf: u32, profile: &'a VideoEncodeProfile) -> Self {
        Self {
            width,
            height,
            fps,
            crf,
            profile,
            timestamped_input: false,
            copy_input_timestamps: false,
            input_pix_fmt: "bgr24",
            encoder: "software",
            h264_profile: None,
        }
    }
}

/// `raw_video_command`: ffmpeg arguments (program first) for BGR/BGR48LE frames on stdin.
pub fn raw_video_command(output: &Path, options: &RawVideoOptions<'_>) -> Result<Vec<String>> {
    if !VIDEO_ENCODERS.contains(&options.encoder) {
        return Err(VideoError::value(
            "encoder must be software, auto, videotoolbox, or nvenc",
        ));
    }
    let ffmpeg = require_video_tool(Tool::Ffmpeg)?;
    let mut command = vec![
        ffmpeg.to_string_lossy().into_owned(),
        "-y".into(),
        "-loglevel".into(),
        "error".into(),
    ];
    if options.copy_input_timestamps {
        command.push("-copyts".into());
    }
    if options.timestamped_input {
        command.extend(strings(&["-f", "nut", "-i", "pipe:0"]));
    } else {
        command.extend([
            "-f".into(),
            "rawvideo".into(),
            "-pix_fmt".into(),
            options.input_pix_fmt.to_owned(),
            "-s:v".into(),
            format!("{}x{}", options.width, options.height),
            "-r".into(),
            format_g(options.fps, 12),
            "-i".into(),
            "pipe:0".into(),
        ]);
    }
    command.extend(strings(&["-map", "0:v:0"]));
    command.extend(video_codec_args(
        &suffix_lower(output),
        options.crf,
        options.profile,
        options.encoder,
        &ffmpeg,
        options.h264_profile,
    ));
    command.extend(profile_args(options.profile));
    command.extend(strings(&["-map_metadata", "-1", "-map_chapters", "-1"]));
    command.extend(crate::decode::thread_args());
    command.extend(strings(&["-fps_mode", "passthrough"]));
    if options.copy_input_timestamps {
        command.extend(strings(&["-avoid_negative_ts", "disabled"]));
    }
    command.extend(mp4_timescale_args(output, options.profile.time_base.as_deref()));
    command.extend(container_format_args(output));
    command.push(output.to_string_lossy().into_owned());
    Ok(command)
}

/// A `Command` from an argument vector whose first entry is the program.
pub fn command_from(args: &[String]) -> Result<Command> {
    let (program, rest) = args
        .split_first()
        .ok_or_else(|| VideoError::runtime("empty ffmpeg command"))?;
    let mut command = Command::new(program);
    command.args(rest);
    Ok(command)
}

/// Options for [`mux_encoded_video`].
#[derive(Clone, Debug, Default)]
pub struct MuxOptions<'a> {
    pub strip_metadata: bool,
    pub copy_input_timestamps: bool,
    pub trim_end_seconds: Option<f64>,
    pub video_input_args: &'a [&'a str],
}

/// `mux_encoded_video`: copy encoded video and every source audio stream into the container.
pub fn mux_encoded_video(
    encoded_video: &Path,
    source: &Path,
    output: &Path,
    options: &MuxOptions<'_>,
) -> Result<()> {
    let ffmpeg = require_video_tool(Tool::Ffmpeg)?;
    let mut command = Command::new(ffmpeg);
    command.args(["-y", "-loglevel", "error"]);
    if options.copy_input_timestamps {
        command.arg("-copyts");
    }
    command
        .args(options.video_input_args)
        .arg("-i")
        .arg(encoded_video);
    if let Some(end) = options.trim_end_seconds {
        command.args(["-to".to_owned(), format!("{end:.6}")]);
    }
    let policy = if options.strip_metadata { "-1" } else { "1" };
    command
        .arg("-i")
        .arg(source)
        .args(["-map", "0:v:0", "-map", "1:a?", "-c", "copy"]);
    command.args(["-map_metadata", policy, "-map_chapters", policy]);
    if options.copy_input_timestamps {
        command.args(["-avoid_negative_ts", "disabled"]);
    }
    if ISOBMFF.contains(&suffix_lower(output).as_str()) {
        command.args(["-movflags", "+faststart"]);
    }
    command.args(container_format_args(output)).arg(output);
    let (status, stderr) = run_with_stderr_file(&mut command)?;
    if !status.success() {
        return Err(VideoError::runtime(format!(
            "ffmpeg failed to mux {}: {}",
            output.display(),
            head_chars(&stderr, 500)
        )));
    }
    Ok(())
}

/// `atomic_video_output`: a sibling temporary published over `output` on commit.
pub fn atomic_video_output(output: &Path) -> Result<StagedSibling> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    Ok(StagedSibling::new(output)?)
}

/// `staged_video_output`: video-only and final temporaries; commit publishes the final one.
#[derive(Debug)]
pub struct StagedVideoOutput {
    pub encoded_video: StagedSibling,
    pub final_output: StagedSibling,
}

impl StagedVideoOutput {
    pub fn new(output: &Path) -> Result<Self> {
        let final_output = atomic_video_output(output)?;
        let encoded_video = StagedSibling::new(raiw_core::fs::with_stem_suffix(output, "-video"))?;
        Ok(Self {
            encoded_video,
            final_output,
        })
    }

    pub fn commit(self) -> Result<PathBuf> {
        Ok(self.final_output.commit()?)
    }
}

/// `VideoEndRepair`: durations for the separate decode-order and presentation-order ends.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VideoEndRepair {
    pub last_packet_index: usize,
    pub last_packet_duration: i64,
    pub final_frame_pts: i64,
    pub final_frame_duration: i64,
    pub movie_timescale: i64,
}

impl VideoEndRepair {
    pub fn bitstream_filter(&self) -> String {
        format!(
            "setts=pts=PTS:dts=DTS:duration='if(eq(N,{}),{},if(eq(PTS,{}),{},DURATION))'",
            self.last_packet_index,
            self.last_packet_duration,
            self.final_frame_pts,
            self.final_frame_duration
        )
    }
}

/// Round-half-even of `num / den` (`round(Fraction)`).
fn round_rational(num: i128, den: i128) -> i128 {
    let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
    let floor = num.div_euclid(den);
    let twice = 2 * num.rem_euclid(den);
    match twice.cmp(&den) {
        std::cmp::Ordering::Less => floor,
        std::cmp::Ordering::Greater => floor + 1,
        std::cmp::Ordering::Equal => floor + floor.rem_euclid(2),
    }
}

/// `video_end_repair`: fit both packet ends to the original stream window without moving PTS/DTS.
pub fn video_end_repair(source_clock: &Value, encoded_clock: &Value) -> Result<VideoEndRepair> {
    let unreadable = || VideoError::value("Could not read the terminal video packet clocks");
    let int = |v: &Value| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok()));
    let source = source_clock
        .get("streams")
        .and_then(|s| s.get(0))
        .ok_or_else(unreadable)?;
    let encoded = encoded_clock
        .get("streams")
        .and_then(|s| s.get(0))
        .ok_or_else(unreadable)?;
    let source_base = source
        .get("time_base")
        .and_then(Value::as_str)
        .and_then(Rational::parse)
        .ok_or_else(unreadable)?;
    let encoded_base = encoded
        .get("time_base")
        .and_then(Value::as_str)
        .and_then(Rational::parse)
        .ok_or_else(unreadable)?;
    let packets = encoded_clock
        .get("packets")
        .and_then(Value::as_array)
        .filter(|p| !p.is_empty())
        .ok_or_else(unreadable)?;
    let duration_ts = source.get("duration_ts").and_then(int).ok_or_else(unreadable)?;
    let start_pts = source.get("start_pts").and_then(int).ok_or_else(unreadable)?;
    let pts: Vec<i64> = packets
        .iter()
        .map(|p| p.get("pts").and_then(int))
        .collect::<Option<_>>()
        .ok_or_else(unreadable)?;
    let dts: Vec<i64> = packets
        .iter()
        .map(|p| p.get("dts").and_then(int))
        .collect::<Option<_>>()
        .ok_or_else(unreadable)?;
    if encoded_base.num == 0 {
        return Err(unreadable());
    }
    let (sn, sd) = (i128::from(source_base.num), i128::from(source_base.den));
    let (en, ed) = (i128::from(encoded_base.num), i128::from(encoded_base.den));
    let end_pts = round_rational(
        (i128::from(start_pts) + i128::from(duration_ts)) * sn * ed,
        sd * en,
    );
    let final_pts = *pts.iter().max().ok_or_else(unreadable)?;
    let presentation = end_pts - i128::from(final_pts);
    let mut decode = round_rational(i128::from(duration_ts) * sn * ed, sd * en)
        - (i128::from(dts[dts.len() - 1]) - i128::from(dts[0]));
    if pts[pts.len() - 1] == final_pts {
        decode = decode.max(presentation);
    }
    if source_base.num <= 0
        || encoded_base.num <= 0
        || duration_ts * source_base.num <= 0
        || presentation.min(decode) <= 0
    {
        return Err(VideoError::value(
            "Encoded packet clocks do not fit the source video window",
        ));
    }
    Ok(VideoEndRepair {
        last_packet_index: packets.len() - 1,
        last_packet_duration: decode as i64,
        final_frame_pts: final_pts,
        final_frame_duration: presentation as i64,
        movie_timescale: encoded_base.den,
    })
}

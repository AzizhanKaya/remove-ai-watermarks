//! ffprobe queries (the probing half of `video_encoding.py`, plus the capture geometry OpenCV
//! reports and the stream inventory `video_audio.py` reads).

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use serde::Serialize;
use serde_json::Value;

use crate::error::{Result, VideoError};
use crate::process::run_captured;
use crate::tools::{Tool, find_video_tool, require_video_tool};

pub const HIGH_BIT_DEPTH_PIXEL_FORMATS: [&str; 3] = ["yuv420p10le", "yuv422p10le", "yuv444p10le"];
const PIXEL_FORMATS: [&str; 6] = [
    "yuv420p",
    "yuv422p",
    "yuv444p",
    "yuv420p10le",
    "yuv422p10le",
    "yuv444p10le",
];
const PIXEL_FORMAT_ALIASES: [(&str, &str); 7] = [
    ("yuvj420p", "yuv420p"),
    ("yuvj422p", "yuv422p"),
    ("yuvj444p", "yuv444p"),
    ("p010le", "yuv420p10le"),
    ("yuv420p10be", "yuv420p10le"),
    ("yuv422p10be", "yuv422p10le"),
    ("yuv444p10be", "yuv444p10le"),
];
const COLOR_RANGES: [&str; 2] = ["tv", "pc"];
const COLOR_SPACES: [&str; 7] = [
    "bt709",
    "fcc",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "bt2020nc",
    "bt2020c",
];
const COLOR_TRANSFERS: [&str; 16] = [
    "bt709",
    "gamma22",
    "gamma28",
    "smpte170m",
    "smpte240m",
    "linear",
    "log",
    "log_sqrt",
    "iec61966-2-4",
    "bt1361e",
    "iec61966-2-1",
    "bt2020-10",
    "bt2020-12",
    "smpte2084",
    "smpte428",
    "arib-std-b67",
];
const COLOR_PRIMARIES: [&str; 12] = [
    "bt709",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "film",
    "bt2020",
    "smpte428",
    "smpte431",
    "smpte432",
    "jedec-p22",
    "ebu3213",
];

/// Whether a normalized pixel format takes the 16-bit BGR48 path.
pub fn is_high_bit_depth(pixel_format: &str) -> bool {
    HIGH_BIT_DEPTH_PIXEL_FORMATS.contains(&pixel_format)
}

/// `_run_ffprobe`: one query with shared logging; `None` when ffprobe is absent or fails.
pub fn run_ffprobe(
    source: &Path,
    arguments: &[&str],
    purpose: &str,
    output_format: &str,
) -> Result<Option<String>> {
    let Some(ffprobe) = find_video_tool(Tool::Ffprobe) else {
        tracing::warn!(
            "ffprobe is unavailable; cannot inspect {purpose} of {}",
            source.display()
        );
        return Ok(None);
    };
    let mut command = Command::new(ffprobe);
    command
        .args(["-v", "error"])
        .args(arguments)
        .args(["-of", output_format])
        .arg(source);
    let output = run_captured(&mut command)?;
    if !output.status.success() {
        tracing::warn!("ffprobe could not inspect {purpose} for {}", source.display());
        return Ok(None);
    }
    Ok(Some(String::from_utf8_lossy(&output.stdout).into_owned()))
}

/// A reduced positive rational, as Python's `Fraction`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

impl Rational {
    pub fn new(num: i64, den: i64) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let sign = if den < 0 { -1 } else { 1 };
        let g = gcd(num.unsigned_abs(), den.unsigned_abs()).max(1) as i64;
        Some(Self {
            num: sign * num / g,
            den: sign * den / g,
        })
    }

    /// Parse `a/b` or an integer, as `Fraction(str)` does for ffprobe output.
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        match text.split_once('/') {
            Some((num, den)) => Self::new(num.trim().parse().ok()?, den.trim().parse().ok()?),
            None => Self::new(text.parse().ok()?, 1),
        }
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn is_positive(self) -> bool {
        self.num > 0
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.num, self.den)
    }
}

pub(crate) fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `VideoEncodeProfile`: source properties the raw-frame encoder can preserve.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VideoEncodeProfile {
    pub pixel_format: String,
    pub color_range: Option<String>,
    pub color_space: Option<String>,
    pub color_transfer: Option<String>,
    pub color_primaries: Option<String>,
    pub time_base: Option<String>,
    pub start_pts: Option<i64>,
    pub source_pixel_format: Option<String>,
    pub component_depth: Option<u32>,
}

impl Default for VideoEncodeProfile {
    fn default() -> Self {
        Self {
            pixel_format: "yuv420p".to_owned(),
            color_range: None,
            color_space: None,
            color_transfer: None,
            color_primaries: None,
            time_base: None,
            start_pts: None,
            source_pixel_format: None,
            component_depth: None,
        }
    }
}

impl VideoEncodeProfile {
    pub fn high_depth(&self) -> bool {
        is_high_bit_depth(&self.pixel_format)
    }
}

fn known_value(value: Option<&Value>, allowed: &[&str]) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|text| allowed.contains(text))
        .map(str::to_owned)
}

fn normalized_time_base(value: Option<&Value>) -> Option<String> {
    let fraction = Rational::parse(value?.as_str()?)?;
    fraction.is_positive().then(|| fraction.to_string())
}

/// `ffmpeg -pix_fmts` component depths, read once per process.
fn pixel_format_depths() -> &'static HashMap<String, u32> {
    static DEPTHS: OnceLock<HashMap<String, u32>> = OnceLock::new();
    DEPTHS.get_or_init(|| {
        let mut table = HashMap::new();
        let Some(ffmpeg) = find_video_tool(Tool::Ffmpeg) else {
            return table;
        };
        let Ok(output) = Command::new(ffmpeg).args(["-hide_banner", "-pix_fmts"]).output() else {
            return table;
        };
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 5 || fields[0].len() != 5 {
                continue;
            }
            let depth = fields[4]
                .split('-')
                .filter_map(|part| part.parse::<u32>().ok())
                .max();
            if let Some(depth) = depth {
                table.insert(fields[1].to_owned(), depth);
            }
        }
        table
    })
}

/// `_pixel_component_depth`: the largest component depth ffprobe or the format table reports.
fn pixel_component_depth(pixel_format: Option<&str>, raw_bits: Option<&Value>) -> Option<u32> {
    let mut depths = Vec::new();
    if let Some(bits) = raw_bits.and_then(Value::as_str)
        && !bits.is_empty()
        && bits.bytes().all(|b| b.is_ascii_digit())
        && let Ok(bits) = bits.parse()
    {
        depths.push(bits);
    }
    if let Some(format) = pixel_format
        && let Some(depth) = pixel_format_depths().get(format)
    {
        depths.push(*depth);
    }
    depths.into_iter().max()
}

/// `probe_video_encode_profile`.
pub fn probe_video_encode_profile(source: &Path) -> Result<VideoEncodeProfile> {
    let raw = run_ffprobe(
        source,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt,bits_per_raw_sample,color_range,color_space,color_transfer,color_primaries,time_base,start_pts",
        ],
        "video profile",
        "json",
    )?;
    let Some(raw) = raw else {
        return Ok(VideoEncodeProfile::default());
    };
    let payload: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
    let Some(stream) = payload.get("streams").and_then(|s| s.get(0)) else {
        tracing::warn!(
            "ffprobe returned no usable video profile for {}; using yuv420p",
            source.display()
        );
        return Ok(VideoEncodeProfile::default());
    };
    let raw_pixel_format = stream.get("pix_fmt").and_then(Value::as_str);
    let aliased = raw_pixel_format.map(|format| {
        PIXEL_FORMAT_ALIASES
            .iter()
            .find(|(from, _)| *from == format)
            .map_or(format, |(_, to)| *to)
    });
    let pixel_format = aliased
        .filter(|format| PIXEL_FORMATS.contains(format))
        .unwrap_or("yuv420p");
    let time_base = normalized_time_base(stream.get("time_base"));
    let start_pts = stream
        .get("start_pts")
        .and_then(Value::as_i64)
        .filter(|_| time_base.is_some());
    Ok(VideoEncodeProfile {
        pixel_format: pixel_format.to_owned(),
        color_range: known_value(stream.get("color_range"), &COLOR_RANGES),
        color_space: known_value(stream.get("color_space"), &COLOR_SPACES),
        color_transfer: known_value(stream.get("color_transfer"), &COLOR_TRANSFERS),
        color_primaries: known_value(stream.get("color_primaries"), &COLOR_PRIMARIES),
        time_base,
        start_pts,
        source_pixel_format: raw_pixel_format.map(str::to_owned),
        component_depth: pixel_component_depth(raw_pixel_format, stream.get("bits_per_raw_sample")),
    })
}

/// Python's `float(str)` on ffprobe text.
pub(crate) fn py_float(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<f64>().ok()
}

/// `probe_video_duration`: the first video stream duration.
pub fn probe_video_duration(path: &Path) -> Result<f64> {
    let value = run_ffprobe(
        path,
        &["-select_streams", "v:0", "-show_entries", "stream=duration"],
        "video duration",
        "csv=p=0",
    )?
    .ok_or_else(|| VideoError::value(format!("Could not read video duration: {}", path.display())))?;
    let duration = py_float(&value)
        .ok_or_else(|| VideoError::value(format!("could not convert string to float: '{}'", value.trim())))?;
    if !duration.is_finite() || duration <= 0.0 {
        return Err(VideoError::value(format!(
            "Invalid video duration: {}",
            path.display()
        )));
    }
    Ok(duration)
}

/// `VideoKeyframeSpec`: keyframe packet layout for selective GOP stream-copy.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VideoKeyframeSpec {
    pub codec_name: String,
    pub profile_name: Option<String>,
    pub start_pts: Option<i64>,
    pub has_b_frames: i64,
    pub packet_count: usize,
    pub keyframes: Vec<(usize, f64)>,
}

/// `probe_video_keyframes`.
pub fn probe_video_keyframes(source: &Path) -> Result<Option<VideoKeyframeSpec>> {
    let raw = run_ffprobe(
        source,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,start_pts,has_b_frames:packet=pts_time,flags",
        ],
        "video keyframes",
        "json",
    )?;
    let Some(raw) = raw else { return Ok(None) };
    let Ok(payload) = serde_json::from_str::<Value>(&raw) else {
        return Ok(None);
    };
    let Some(stream) = payload.get("streams").and_then(|s| s.get(0)) else {
        return Ok(None);
    };
    let packets = payload
        .get("packets")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let Some(codec_name) = stream.get("codec_name").and_then(Value::as_str) else {
        return Ok(None);
    };
    if packets.is_empty() {
        return Ok(None);
    }
    let mut keyframes = Vec::new();
    for (index, packet) in packets.iter().enumerate() {
        let flags = packet.get("flags").and_then(Value::as_str).unwrap_or("");
        let Some(pts) = packet.get("pts_time").filter(|v| !v.is_null()) else {
            continue;
        };
        if !flags.contains('K') {
            continue;
        }
        let Some(value) = pts.as_str().and_then(py_float).or_else(|| pts.as_f64()) else {
            return Ok(None);
        };
        if !value.is_finite() {
            return Ok(None);
        }
        keyframes.push((index, value));
    }
    if keyframes.is_empty() {
        return Ok(None);
    }
    Ok(Some(VideoKeyframeSpec {
        codec_name: codec_name.to_owned(),
        profile_name: stream.get("profile").and_then(Value::as_str).map(str::to_owned),
        start_pts: stream.get("start_pts").and_then(Value::as_i64),
        has_b_frames: stream.get("has_b_frames").and_then(Value::as_i64).unwrap_or(0),
        packet_count: packets.len(),
        keyframes,
    }))
}

/// `count_decoded_video_frames`: exact decodable frames via `-count_frames`.
pub fn count_decoded_video_frames(source: &Path) -> Result<Option<u64>> {
    let raw = run_ffprobe(
        source,
        &[
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
        ],
        "decoded frame count",
        "csv=p=0",
    )?;
    let Some(raw) = raw else { return Ok(None) };
    let token = raw.trim().lines().next().unwrap_or("").trim();
    Ok((!token.is_empty() && token.bytes().all(|b| b.is_ascii_digit()))
        .then(|| token.parse().ok())
        .flatten())
}

/// `_probe_frame_timestamps`: decoded-frame `best_effort_timestamp_time` values.
pub fn probe_frame_timestamps(source: &Path) -> Result<Vec<f64>> {
    let raw = run_ffprobe(
        source,
        &[
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
        ],
        "video timestamps",
        "csv=p=0",
    )?;
    let Some(raw) = raw else { return Ok(Vec::new()) };
    let mut timestamps = Vec::new();
    for line in raw.lines().filter(|line| !line.is_empty()) {
        let Some(value) = py_float(line.split(',').next().unwrap_or("")) else {
            tracing::warn!(
                "ffprobe returned unusable frame timestamps for {}",
                source.display()
            );
            return Ok(Vec::new());
        };
        timestamps.push(value);
    }
    if timestamps.is_empty() || !timestamps.iter().all(|t| t.is_finite()) {
        tracing::warn!(
            "ffprobe returned no finite frame timestamps for {}",
            source.display()
        );
        return Ok(Vec::new());
    }
    Ok(timestamps)
}

/// `probe_video_timestamps`: display PTS from packets, falling back to decoded frames when
/// packets carry discard flags, missing PTS, or duplicate timestamps.
pub fn probe_video_timestamps(source: &Path) -> Result<Vec<f64>> {
    let raw = run_ffprobe(
        source,
        &[
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,flags",
        ],
        "video packet timestamps",
        "csv=p=0",
    )?;
    let Some(raw) = raw else { return Ok(Vec::new()) };
    let mut values = Vec::new();
    let mut fallback = false;
    for line in raw.lines().filter(|line| !line.is_empty()) {
        let mut parts = line.split(',');
        let Some(value) = py_float(parts.next().unwrap_or("")).filter(|v| v.is_finite()) else {
            fallback = true;
            break;
        };
        let flags = parts.next().map_or("", str::trim);
        if !flags.is_empty()
            && !flags.contains(' ')
            && flags.chars().all(|c| "KDCBT_".contains(c))
            && flags.contains('D')
        {
            fallback = true;
            break;
        }
        values.push(value);
    }
    let duplicates = values.len() > 1 && {
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        sorted.windows(2).any(|w| w[0] == w[1])
    };
    if fallback || values.is_empty() || duplicates {
        return probe_frame_timestamps(source);
    }
    values.sort_by(f64::total_cmp);
    Ok(values)
}

/// The capture properties OpenCV reports for the first video stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoGeometry {
    /// Display width (after rotation side data).
    pub width: usize,
    pub height: usize,
    /// `CAP_PROP_FPS`: the average frame rate, else the stream rate.
    pub fps: f64,
    /// `CAP_PROP_FRAME_COUNT` from the container (`nb_frames`), 0 when unknown.
    pub frame_count: u64,
    /// The stream start, in seconds (OpenCV's timestamp origin).
    pub start_time: f64,
}

fn rate(value: Option<&Value>) -> f64 {
    value
        .and_then(Value::as_str)
        .and_then(Rational::parse)
        .filter(|r| r.num > 0)
        .map_or(0.0, Rational::to_f64)
}

/// Probe the decoded geometry and nominal rate of the first video stream.
pub fn probe_video_geometry(source: &Path) -> Result<Option<VideoGeometry>> {
    let raw = run_ffprobe(
        source,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,start_time:stream_side_data=rotation",
        ],
        "video geometry",
        "json",
    )?;
    let Some(raw) = raw else { return Ok(None) };
    let Ok(payload) = serde_json::from_str::<Value>(&raw) else {
        return Ok(None);
    };
    let Some(stream) = payload.get("streams").and_then(|s| s.get(0)) else {
        return Ok(None);
    };
    let (Some(width), Some(height)) = (
        stream.get("width").and_then(Value::as_u64),
        stream.get("height").and_then(Value::as_u64),
    ) else {
        return Ok(None);
    };
    let rotation = stream
        .get("side_data_list")
        .and_then(Value::as_array)
        .and_then(|list| {
            list.iter()
                .find_map(|item| item.get("rotation").and_then(Value::as_i64))
        })
        .unwrap_or(0);
    let (width, height) = if rotation.rem_euclid(180) == 90 {
        (height, width)
    } else {
        (width, height)
    };
    let average = rate(stream.get("avg_frame_rate"));
    let fps = if average > 0.0 {
        average
    } else {
        rate(stream.get("r_frame_rate"))
    };
    let frame_count = stream
        .get("nb_frames")
        .and_then(Value::as_str)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let start_time = stream
        .get("start_time")
        .and_then(Value::as_str)
        .and_then(py_float)
        .unwrap_or(0.0);
    Ok(Some(VideoGeometry {
        width: width as usize,
        height: height as usize,
        fps,
        frame_count,
        start_time,
    }))
}

/// One `ffprobe -show_streams` entry (`video_audio.stream_inventory`).
pub type StreamEntry = serde_json::Map<String, Value>;

/// `stream_inventory`: every stream with its index, codec type and properties.
pub fn stream_inventory(source: &Path) -> Result<Vec<StreamEntry>> {
    let ffprobe = require_video_tool(Tool::Ffprobe)?;
    let mut command = Command::new(ffprobe);
    command
        .args(["-v", "error", "-show_streams", "-of", "json"])
        .arg(source);
    let output = run_captured(&mut command)?;
    if !output.status.success() {
        return Err(VideoError::runtime(format!(
            "ffprobe could not list streams of {}: {}",
            source.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let payload: Value = serde_json::from_slice(&output.stdout)?;
    Ok(payload
        .get("streams")
        .and_then(Value::as_array)
        .map(|streams| streams.iter().filter_map(|s| s.as_object().cloned()).collect())
        .unwrap_or_default())
}

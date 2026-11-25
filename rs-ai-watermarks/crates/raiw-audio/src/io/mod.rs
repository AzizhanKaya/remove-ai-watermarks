//! Bounded audio decoding and atomic output through ffmpeg (`_internal/audio_io.py`).

mod decode;
mod encode;
mod fraction;
mod probe;
mod process;

use std::path::{Path, PathBuf};

use serde::Serialize;

pub use decode::{AudioReader, read_all};
pub use encode::{OutputFormat, codec_args, write_audio};
pub use fraction::Fraction;
pub use probe::probe_audio;

use crate::error::{AudioError, Result};

/// Default decode block (`read_audio(..., chunk_frames=65536)`).
pub const DEFAULT_CHUNK_FRAMES: usize = 65_536;
const MAX_DECODE_BYTES: usize = 16 * 1024 * 1024;

/// Native geometry and timing of one audio stream, without resampling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioStreamInfo {
    pub index: usize,
    pub sample_rate: u32,
    pub channels: usize,
    pub codec: String,
    pub start_time: Option<Fraction>,
    pub duration: Option<Fraction>,
    pub format_name: String,
    pub channel_layout: Option<String>,
    pub sample_fmt: Option<String>,
    pub bits_per_raw_sample: Option<u32>,
}

/// Located `ffmpeg` and `ffprobe` executables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegTools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl FfmpegTools {
    /// Search `PATH` (after the `RAIW_FFMPEG` / `RAIW_FFPROBE` overrides) for both tools.
    pub fn discover() -> Result<Self> {
        Ok(Self {
            ffmpeg: require_tool("ffmpeg")?,
            ffprobe: require_tool("ffprobe")?,
        })
    }

    pub fn new(ffmpeg: impl Into<PathBuf>, ffprobe: impl Into<PathBuf>) -> Self {
        Self {
            ffmpeg: ffmpeg.into(),
            ffprobe: ffprobe.into(),
        }
    }

    /// `audio_io.is_available()`: both external tools exist.
    pub fn is_available() -> bool {
        Self::discover().is_ok()
    }
}

/// `require_tool(name)`.
pub fn require_tool(name: &str) -> Result<PathBuf> {
    let variable = format!("RAIW_{}", name.to_uppercase());
    if let Some(path) = std::env::var_os(&variable)
        .map(PathBuf::from)
        .filter(|p| p.is_file())
    {
        return Ok(path);
    }
    find_on_path(name).ok_or_else(|| AudioError::failed(format!("{name} is required for audio processing")))
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths).find_map(|dir| {
        let candidate = dir.join(name);
        if is_executable(&candidate) {
            return Some(candidate);
        }
        let exe = dir.join(format!("{name}.exe"));
        is_executable(&exe).then_some(exe)
    })
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

/// Decode block size capped to the 16 MiB budget.
fn bounded_chunk(info: &AudioStreamInfo, chunk_frames: usize) -> Result<usize> {
    if chunk_frames == 0 {
        return Err(AudioError::invalid("Audio chunk size must be positive"));
    }
    let frame_bytes = 4 * info.channels.max(1);
    if frame_bytes > MAX_DECODE_BYTES {
        return Err(AudioError::invalid(
            "Audio channel layout exceeds the decoding memory budget",
        ));
    }
    Ok(chunk_frames.min(MAX_DECODE_BYTES / frame_bytes))
}

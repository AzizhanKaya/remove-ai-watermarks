//! FFmpeg/ffprobe resolution (port of `video_tools.py`).
//!
//! A complete system pair is preferred; a lone system tool still serves base installs. The Rust
//! build ships no offline bundle, so a missing tool is reported instead of downloaded, and `PATH`
//! is never mutated.

use std::fmt;
use std::path::PathBuf;

use crate::error::{Result, VideoError};

/// One of the two FFmpeg executables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tool {
    Ffmpeg,
    Ffprobe,
}

impl Tool {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ffmpeg => "ffmpeg",
            Self::Ffprobe => "ffprobe",
        }
    }
}

impl fmt::Display for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// `find_video_tool`: the system executable, or `None`.
pub fn find_video_tool(tool: Tool) -> Option<PathBuf> {
    which::which(tool.name()).ok()
}

/// `require_video_tool`: resolve a tool or explain what supplies it.
pub fn require_video_tool(tool: Tool) -> Result<PathBuf> {
    find_video_tool(tool).ok_or_else(|| {
        VideoError::runtime(format!(
            "Video processing requires {tool}; install FFmpeg so ffmpeg and ffprobe are on PATH"
        ))
    })
}

/// Whether both executables resolve.
pub fn video_tools_available() -> bool {
    find_video_tool(Tool::Ffmpeg).is_some() && find_video_tool(Tool::Ffprobe).is_some()
}

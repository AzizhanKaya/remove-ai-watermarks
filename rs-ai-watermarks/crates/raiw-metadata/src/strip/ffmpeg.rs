//! `_strip_with_ffmpeg`: lossless stream-copy removal of container tags and chapters.

use std::path::{Path, PathBuf};
use std::process::Command;

use raiw_container::ebml::{attachment_ordinals, scrub_av1_ai_obus};
use raiw_core::StagedSibling;
use raiw_core::fs::suffix_lower;

use super::StripError;

/// The ffmpeg executable (`require_video_tool("ffmpeg")`).
pub fn ffmpeg() -> Result<PathBuf, StripError> {
    crate::audio_creation::which_on_path("ffmpeg").ok_or(StripError::FfmpegMissing)
}

/// The argument list before the output path.
pub fn ffmpeg_args(source: &Path) -> Vec<String> {
    let (total, c2pa) = attachment_ordinals(source);
    let mut args: Vec<String> = ["-y", "-loglevel", "error", "-i"].map(str::to_owned).to_vec();
    args.push(source.to_string_lossy().into_owned());
    args.extend(["-map".to_owned(), "0".to_owned()]);
    for ordinal in &c2pa {
        args.extend(["-map".to_owned(), format!("-0:t:{ordinal}")]);
    }
    args.extend(["-map_metadata".to_owned(), "-1".to_owned()]);
    if total > c2pa.len() {
        args.extend(["-map_metadata:s:t".to_owned(), "0:s:t".to_owned()]);
    }
    args.extend(["-map_chapters", "-1", "-c", "copy"].map(str::to_owned));
    args
}

/// Strip through ffmpeg into a staged sibling, then publish atomically.
pub fn strip_with_ffmpeg(source: &Path, output: &Path) -> Result<PathBuf, StripError> {
    let ffmpeg = ffmpeg()?;
    let args = ffmpeg_args(source);
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let stage = StagedSibling::new(output)?;
    let result = Command::new(&ffmpeg).args(&args).arg(stage.path()).output()?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let trimmed: String = stderr.trim().chars().take(300).collect();
        return Err(StripError::Ffmpeg(format!(
            "ffmpeg failed to strip metadata from {}: {trimmed}",
            source.display()
        )));
    }
    if matches!(suffix_lower(stage.path()).as_str(), ".mkv" | ".webm") {
        scrub_av1_ai_obus(stage.path())?;
    }
    Ok(stage.commit()?)
}

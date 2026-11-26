//! `remove_ai_metadata`: content-routed, lossless where the container allows it.

pub mod ffmpeg;
pub mod jpeg;
pub mod png_lossless;
pub mod png_writer;
pub mod resave;

use std::io;
use std::path::{Path, PathBuf};

use raiw_container::isobmff::{self, MediaStripOptions, STREAM_SCAN_BYTES, is_isobmff};
use raiw_container::tiff::strip_tiff_metadata_lossless;
use raiw_core::fs::suffix_lower;

use crate::audio_creation::audio_creation_tool;
use crate::hooks::Hooks;

/// Why a strip could not complete; Python raises `OSError`, `RuntimeError` or `ValueError`.
#[derive(Debug, thiserror::Error)]
pub enum StripError {
    #[error(transparent)]
    Io(#[from] io::Error),
    /// `ValueError`: a container the strip refuses (BigTIFF, an unwalkable TIFF).
    #[error("{0}")]
    Value(String),
    #[error("ffmpeg not found")]
    FfmpegMissing,
    /// `RuntimeError` from the ffmpeg stream copy.
    #[error("{0}")]
    Ffmpeg(String),
    /// The re-save could not decode the pixels.
    #[error("image decode failed")]
    Decode,
    #[error("{0}")]
    Unsupported(String),
}

const ISOBMFF_EXTS: &[&str] = &[".avif", ".heif", ".heic", ".jxl", ".mp4", ".mov", ".m4v", ".m4a"];
const STREAMING_ISOBMFF_EXTS: &[&str] = &[".mp4", ".mov", ".m4v", ".m4a"];
pub const FFMPEG_STRIP_EXTS: &[&str] = &[
    ".webm", ".mkv", ".mka", ".avi", ".flv", ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aac",
];

/// Write `bytes` to `output`, creating its directory.
pub fn write_output(output: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, bytes)
}

fn copy_through(source: &Path, output: &Path) -> io::Result<()> {
    if output != source {
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(source, output)?;
    }
    Ok(())
}

/// `_sniff_image_format`: the lossy-vs-lossless raster format from magic bytes.
pub fn sniff_image_format(head: &[u8]) -> Option<&'static str> {
    if head.starts_with(b"\xff\xd8") {
        Some("JPEG")
    } else if head.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("PNG")
    } else if head.starts_with(b"RIFF") && head.get(8..12) == Some(b"WEBP") {
        Some("WEBP")
    } else {
        None
    }
}

/// `remove_ai_metadata(source, output, keep_standard)`: the path written.
pub fn remove_ai_metadata(
    source: &Path,
    output: Option<&Path>,
    keep_standard: bool,
) -> Result<PathBuf, StripError> {
    let output = output.unwrap_or(source);
    let head = crate::scan::read_head(source, 26)?;
    let suffix = suffix_lower(source);
    if STREAMING_ISOBMFF_EXTS.contains(&suffix.as_str()) && is_isobmff(&head) {
        let options = MediaStripOptions {
            max_box_scan: STREAM_SCAN_BYTES,
            strip_audio_comments: audio_creation_tool(source).is_some(),
        };
        isobmff::strip_media_file(source, output, options, &Hooks)?;
        return Ok(output.to_path_buf());
    }
    if ISOBMFF_EXTS.contains(&suffix.as_str()) || is_isobmff(&head) {
        isobmff::strip_image_file(source, output, STREAM_SCAN_BYTES, &Hooks)?;
        return Ok(output.to_path_buf());
    }
    if FFMPEG_STRIP_EXTS.contains(&suffix.as_str()) {
        return ffmpeg::strip_with_ffmpeg(source, output);
    }
    let true_fmt = sniff_image_format(&head);
    if matches!(head.get(..6), Some(b"GIF87a" | b"GIF89a")) {
        let data = std::fs::read(source)?;
        match raiw_container::gif::strip_metadata(&data, keep_standard) {
            Some(stripped) => write_output(output, &stripped)?,
            None => copy_through(source, output)?,
        }
        return Ok(output.to_path_buf());
    }
    if head.starts_with(b"II*\x00") || head.starts_with(b"MM\x00*") {
        if strip_tiff_metadata_lossless(source, output, keep_standard, &Hooks)? {
            return Ok(output.to_path_buf());
        }
        return Err(StripError::Value(format!(
            "could not walk TIFF/DNG structure losslessly: {}",
            file_name(source)
        )));
    }
    if head.starts_with(b"II+\x00") || head.starts_with(b"MM\x00+") {
        return Err(StripError::Value(format!(
            "metadata removal does not support BigTIFF containers: {}",
            file_name(source)
        )));
    }
    if keep_standard && true_fmt == Some("JPEG") && jpeg::strip_jpeg_lossless(source, output)? {
        return Ok(output.to_path_buf());
    }
    if true_fmt == Some("PNG")
        && raiw_container::png::bit_depth(&head) > 8
        && png_lossless::strip_png_lossless(source, output, keep_standard)?
    {
        return Ok(output.to_path_buf());
    }
    let probe = crate::pil::probe(source);
    let decodable = probe.loaded().is_some_and(|loaded| loaded.error.is_none());
    if !decodable {
        copy_through(source, output)?;
        return Ok(output.to_path_buf());
    }
    resave::resave(source, output, keep_standard, true_fmt)?;
    Ok(output.to_path_buf())
}

/// `Path.name`.
pub fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

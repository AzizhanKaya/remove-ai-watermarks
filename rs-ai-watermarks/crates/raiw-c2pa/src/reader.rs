//! The validating manifest reader (c2pa-rs, the line c2pa-python 0.38 wraps).
//!
//! No trust anchors or allowlist are installed; only the bundled signing EKUs are kept,
//! because c2pa-rs clears its default EKUs when trust settings are applied.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use c2pa::{Context, Reader};
use serde_json::json;

const ALLOWED_EKUS: &str = concat!(
    "1.3.6.1.5.5.7.3.4\n",
    "1.3.6.1.5.5.7.3.36\n",
    "1.3.6.1.5.5.7.3.8\n",
    "1.3.6.1.5.5.7.3.9\n",
    "1.3.6.1.4.1.311.76.59.1.9\n",
    "1.3.6.1.4.1.62558.2.1",
);

/// An unexpected reader failure (not "no manifest" and not "unsupported container").
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("C2PA reader failed to open {path}: {source}")]
    Open { path: String, source: Box<c2pa::Error> },
    #[error("C2PA reader could not read {path}: {source}")]
    Io { path: String, source: std::io::Error },
}

/// The MIME type Python's `mimetypes.guess_type` gives a path, or `""` for auto-detection.
pub fn mime_from_path(path: &Path) -> &'static str {
    match raiw_core::fs::suffix_lower(path).as_str() {
        ".png" => "image/png",
        ".jpg" | ".jpeg" | ".jpe" | ".jfif" => "image/jpeg",
        ".gif" => "image/gif",
        ".webp" => "image/webp",
        ".heic" => "image/heic",
        ".heif" => "image/heif",
        ".avif" => "image/avif",
        ".jxl" => "image/jxl",
        ".tif" | ".tiff" => "image/tiff",
        ".dng" => "image/dng",
        ".bmp" => "image/bmp",
        ".svg" => "image/svg+xml",
        ".pdf" => "application/pdf",
        ".mp4" | ".m4v" => "video/mp4",
        ".mov" => "video/quicktime",
        ".3gp" => "video/3gpp",
        ".m4a" => "audio/mp4",
        ".mp3" => "audio/mpeg",
        ".wav" => "audio/x-wav",
        ".flac" => "audio/x-flac",
        ".ogg" | ".oga" | ".opus" => "audio/ogg",
        ".aac" => "audio/aac",
        ".mkv" => "video/x-matroska",
        ".mka" => "audio/x-matroska",
        ".webm" => "video/webm",
        ".avi" => "video/x-msvideo",
        ".flv" => "video/x-flv",
        ".txt" => "text/plain",
        _ => "",
    }
}

fn context() -> Result<Context, c2pa::Error> {
    Context::new().with_settings(json!({"trust": {"trust_config": ALLOWED_EKUS}}))
}

/// The manifest store as JSON, `Ok(None)` when the file has none or the container is unsupported.
///
/// Any other failure is an `Err`: Python's `strict=True` raises it, the tolerant path
/// treats it as no manifest.
pub fn read_manifest_store_json(path: &Path) -> Result<Option<String>, ReadError> {
    let display = path.display().to_string();
    let open = |source: c2pa::Error| ReadError::Open {
        path: display.clone(),
        source: Box::new(source),
    };
    let file = File::open(path).map_err(|source| ReadError::Io {
        path: display.clone(),
        source,
    })?;
    let reader = Reader::from_context(context().map_err(open)?)
        .with_stream(mime_from_path(path), BufReader::new(file));
    match reader {
        Ok(reader) => Ok(Some(reader.json())),
        Err(c2pa::Error::JumbfNotFound | c2pa::Error::UnsupportedType) => Ok(None),
        Err(source) => Err(open(source)),
    }
}

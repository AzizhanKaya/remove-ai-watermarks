//! `strip_and_verify` / `strip_written_output`: strip, then re-scan what was written.

use std::path::{Path, PathBuf};

use crate::detect::{AiMetadata, get_ai_metadata};
use crate::readability::unreadable_reason;
use crate::strip::{StripError, file_name, remove_ai_metadata};

#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    /// `ValueError`: the source is not a readable container; nothing was written.
    #[error("cannot read {name}: {reason}")]
    Unreadable { name: String, reason: String },
    #[error(transparent)]
    Strip(#[from] StripError),
    /// `OSError` from the raster normalization.
    #[error("Failed to normalize image after incomplete metadata stripping: {0}")]
    Normalize(String),
}

/// The published path and the markers that survived (empty for a real strip).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StripOutcome {
    pub output: PathBuf,
    pub remaining: AiMetadata,
}

/// `strip_and_verify(source, output, keep_standard=...)`.
pub fn strip_and_verify(
    source: &Path,
    output: Option<&Path>,
    keep_standard: bool,
) -> Result<StripOutcome, VerifyError> {
    if let Some(reason) = unreadable_reason(source) {
        return Err(VerifyError::Unreadable {
            name: file_name(source),
            reason,
        });
    }
    strip_and_rescan(source, output, keep_standard)
}

/// `strip_written_output(path)`: strip in place, skipping the readability probe.
pub fn strip_written_output(path: &Path, keep_standard: bool) -> Result<AiMetadata, VerifyError> {
    Ok(strip_and_rescan(path, Some(path), keep_standard)?.remaining)
}

fn strip_and_rescan(
    source: &Path,
    output: Option<&Path>,
    keep_standard: bool,
) -> Result<StripOutcome, VerifyError> {
    let out = remove_ai_metadata(source, output, keep_standard)?;
    let remaining = get_ai_metadata(&out);
    if remaining.is_empty() {
        return Ok(StripOutcome {
            output: out,
            remaining,
        });
    }
    let Ok(decoded) = image::open(&out) else {
        return Ok(StripOutcome {
            output: out,
            remaining,
        });
    };
    decoded
        .save(&out)
        .map_err(|_| VerifyError::Normalize(out.display().to_string()))?;
    let remaining = get_ai_metadata(&out);
    Ok(StripOutcome {
        output: out,
        remaining,
    })
}

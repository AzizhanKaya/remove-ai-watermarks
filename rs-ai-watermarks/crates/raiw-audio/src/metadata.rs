//! Seam to the container-metadata subsystem (`metadata.py`, `_internal/c2pa.py`), wired by a later wave.

use std::path::{Path, PathBuf};

use crate::error::{AudioError, Result};
use crate::report::Markers;

/// AI-provenance metadata services the audio pipeline consumes.
pub trait MetadataSeam: Send + Sync {
    /// `metadata.get_ai_metadata(path)`.
    fn ai_metadata(&self, path: &Path) -> Result<Markers>;

    /// `metadata.unreadable_reason(path)`: why a structured container cannot be read.
    fn unreadable_reason(&self, path: &Path) -> Option<String>;

    /// `c2pa.markers_claim_ai(markers)`: metadata asserts AI origin rather than neutral transport.
    fn markers_claim_ai(&self, markers: &Markers) -> bool;

    /// `metadata.strip_and_verify(source, output, keep_standard=...)` -> `(written, remaining)`.
    fn strip_and_verify(
        &self,
        source: &Path,
        output: &Path,
        keep_standard: bool,
    ) -> Result<(PathBuf, Markers)>;
}

/// A metadata stage that finds nothing and cannot strip (until `raiw-metadata` is wired).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoMetadata;

impl MetadataSeam for NoMetadata {
    fn ai_metadata(&self, _path: &Path) -> Result<Markers> {
        Ok(Markers::new())
    }

    fn unreadable_reason(&self, _path: &Path) -> Option<String> {
        None
    }

    fn markers_claim_ai(&self, markers: &Markers) -> bool {
        !markers.is_empty()
    }

    fn strip_and_verify(
        &self,
        _source: &Path,
        _output: &Path,
        _keep_standard: bool,
    ) -> Result<(PathBuf, Markers)> {
        Err(AudioError::unavailable(
            "Audio metadata removal requires the metadata subsystem",
        ))
    }
}

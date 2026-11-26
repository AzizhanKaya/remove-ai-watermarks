//! Error type for weight resolution, bundle validation, decoding, and ViT execution.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ClassifyError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Python raises `ValueError` for every malformed or inconsistent bundle.
    #[error("{0}")]
    InvalidBundle(String),
    /// Python raises `RuntimeError` for weight resolution and ViT output failures.
    #[error("{0}")]
    Runtime(String),
    #[error("cannot decode image {path}: {reason}")]
    Decode { path: PathBuf, reason: String },
    #[error("unsupported embedded ViT graph: {0}")]
    UnsupportedVit(String),
    #[error(transparent)]
    Cv(#[from] raiw_cv::CvError),
}

pub type Result<T, E = ClassifyError> = std::result::Result<T, E>;

impl ClassifyError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn bundle(message: impl Into<String>) -> Self {
        Self::InvalidBundle(message.into())
    }
}

/// Fail with `InvalidBundle(message)` unless `ok`.
pub(crate) fn ensure(ok: bool, message: impl FnOnce() -> String) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(ClassifyError::InvalidBundle(message()))
    }
}

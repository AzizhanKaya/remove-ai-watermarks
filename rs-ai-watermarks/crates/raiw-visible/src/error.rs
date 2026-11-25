//! The crate's error type.

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VisibleError {
    #[error(transparent)]
    Cv(#[from] raiw_cv::CvError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error("{op}: unsupported image: {reason}")]
    Unsupported { op: &'static str, reason: String },
    #[error("{0}")]
    Sensitivity(String),
    #[error("unknown mark {0:?}")]
    UnknownMark(String),
    #[error("failed to decode embedded asset: {0}")]
    Asset(&'static str),
    #[error("{backend} backend unavailable: {reason}")]
    BackendUnavailable { backend: &'static str, reason: String },
    #[error("{backend} inference failed: {reason}")]
    Inference { backend: &'static str, reason: String },
    #[error("could not read image: {}", .0.display())]
    Unreadable(PathBuf),
    #[error("failed to write output (is the destination writable?): {}", .0.display())]
    WriteFailed(PathBuf),
    #[error("metadata strip left AI metadata behind: {0:?}")]
    MetadataStripIncomplete(Vec<String>),
}

pub type Result<T, E = VisibleError> = std::result::Result<T, E>;

pub(crate) fn unsupported(op: &'static str, reason: impl Into<String>) -> VisibleError {
    VisibleError::Unsupported { op, reason: reason.into() }
}

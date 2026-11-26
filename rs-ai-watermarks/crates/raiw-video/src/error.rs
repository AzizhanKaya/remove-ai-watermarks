//! Library errors. Variants mirror the Python exception classes so the CLI keeps their messages.

use std::io;

/// Errors raised by the video pipeline.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    /// Python `FileNotFoundError`.
    #[error("{0}")]
    NotFound(String),
    /// Python `ValueError`: invalid arguments or unreadable input.
    #[error("{0}")]
    Value(String),
    /// Python `RuntimeError`: a processing, tool, or verification failure.
    #[error("{0}")]
    Runtime(String),
    /// A backend this build does not compile in (the `gpu` seam).
    #[error("{0}")]
    Unavailable(String),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Cv(#[from] raiw_cv::CvError),
    #[error(transparent)]
    Visible(#[from] raiw_visible::VisibleError),
    #[error(transparent)]
    Audio(#[from] raiw_audio::AudioError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Python `VideoMessageVerificationIncomplete`.
    #[error(transparent)]
    MessageVerification(Box<crate::messages::MessageVerificationIncomplete>),
    #[cfg(feature = "gpu")]
    #[error(transparent)]
    Gpu(#[from] raiw_gpu::GpuError),
}

impl VideoError {
    pub fn value(message: impl Into<String>) -> Self {
        Self::Value(message.into())
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }
}

pub type Result<T, E = VideoError> = std::result::Result<T, E>;

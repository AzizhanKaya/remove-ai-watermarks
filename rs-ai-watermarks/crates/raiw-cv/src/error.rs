//! Error type shared by every kernel.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CvError {
    #[error("buffer of {len} elements does not match {width}x{height}x{channels}")]
    BufferSize { len: usize, width: usize, height: usize, channels: usize },
    #[error("{op}: unsupported channel count {channels}")]
    Channels { op: &'static str, channels: usize },
    #[error("{op}: empty input")]
    Empty { op: &'static str },
    #[error("{op}: input shapes differ")]
    ShapeMismatch { op: &'static str },
    #[error("region {x},{y} {width}x{height} exceeds image {image_width}x{image_height}")]
    RoiOutOfBounds { x: usize, y: usize, width: usize, height: usize, image_width: usize, image_height: usize },
    #[error("{op}: {reason}")]
    InvalidArgument { op: &'static str, reason: String },
    #[error(transparent)]
    Image(#[from] image::ImageError),
}

pub type Result<T, E = CvError> = std::result::Result<T, E>;

pub(crate) fn invalid(op: &'static str, reason: impl Into<String>) -> CvError {
    CvError::InvalidArgument { op, reason: reason.into() }
}

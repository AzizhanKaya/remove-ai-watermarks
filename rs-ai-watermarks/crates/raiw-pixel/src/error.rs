//! Error types for the pixel decoders.

use thiserror::Error;

/// Why the DWT-DCT decoder refused an input (the Python decoder raises in the same cases).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DwtDctError {
    #[error("image too small, should be larger than 256x256")]
    TooSmall,
    #[error("watermark lengths must be positive")]
    InvalidLengths,
    #[error("BGR input needs 3 or 4 channels, got {0}")]
    Channels(usize),
    #[error("the 4-aligned plane is empty ({width}x{height})")]
    EmptyPlane { width: usize, height: usize },
}

/// Why a TrustMark decode could not produce a verdict.
#[derive(Debug, Error)]
pub enum TrustMarkError {
    #[error("unsupported TrustMark variant {0:?}; expected one of P, Q, C")]
    UnsupportedVariant(String),
    #[error("image has no pixels")]
    EmptyImage,
    #[error("TrustMark model failed: {0}")]
    Model(String),
    #[error("TrustMark decoder returned {got} bits, expected {expected}")]
    BitCount { got: usize, expected: usize },
    #[error("BCH decoder state out of range")]
    Bch,
    #[error("image buffer does not match {width}x{height}x{channels}")]
    Buffer { width: usize, height: usize, channels: usize },
}

/// Why the Microsoft InvisMark disruption refused to rewrite pixels.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InvisMarkError {
    #[error("RGB input needs 3 channels, got {0}")]
    Channels(usize),
    #[error("a {width}x{height} image has no 4x4 LL block")]
    NoBlocks { width: usize, height: usize },
}

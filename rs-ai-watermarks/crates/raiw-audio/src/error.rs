//! Error kinds mirroring the Python exception classes the CLI and fallbacks distinguish.

use crate::clock::ClockError;
use crate::dsp::{DesignError, FilterError};
use crate::frames::FramesError;
use crate::report::AudioProvenanceReport;

/// `AudioVerificationIncomplete`: the candidate retained a compatible mark or could not be fully checked.
#[derive(Debug, Clone, PartialEq)]
pub struct VerificationIncomplete {
    pub before: AudioProvenanceReport,
    pub after: AudioProvenanceReport,
    pub reason: Option<String>,
}

impl VerificationIncomplete {
    pub const MESSAGE: &'static str =
        "Audio watermark verification did not confirm removal; candidate was not published";
}

impl std::fmt::Display for VerificationIncomplete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.reason {
            None => f.write_str(Self::MESSAGE),
            Some(reason) => write!(f, "{}. {reason}", Self::MESSAGE),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// Python `ValueError`: rejected arguments or unreadable input.
    #[error("{0}")]
    Invalid(String),
    /// Python `RuntimeError`: a processing step failed or broke an invariant.
    #[error("{0}")]
    Failed(String),
    /// Python `ImportError`: an optional backend is not installed.
    #[error("{0}")]
    Unavailable(String),
    /// Python `FileNotFoundError`.
    #[error("{0}")]
    NotFound(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Verification(Box<VerificationIncomplete>),
}

impl AudioError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(message.into())
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::Unavailable(message.into())
    }

    /// Python `isinstance(exc, RuntimeError)` (verification failures subclass it).
    pub const fn is_runtime(&self) -> bool {
        matches!(self, Self::Failed(_) | Self::Verification(_))
    }
}

impl From<FramesError> for AudioError {
    fn from(err: FramesError) -> Self {
        Self::Invalid(err.to_string())
    }
}

impl From<ClockError> for AudioError {
    fn from(err: ClockError) -> Self {
        Self::Invalid(err.to_string())
    }
}

impl From<FilterError> for AudioError {
    fn from(err: FilterError) -> Self {
        Self::Invalid(err.to_string())
    }
}

impl From<DesignError> for AudioError {
    fn from(err: DesignError) -> Self {
        Self::Invalid(err.to_string())
    }
}

pub type Result<T, E = AudioError> = std::result::Result<T, E>;

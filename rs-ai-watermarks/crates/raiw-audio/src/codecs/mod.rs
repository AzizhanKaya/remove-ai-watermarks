//! Neural codec resynthesis: models behind [`NeuralCodec`], geometry and gain staging ported from Python.

pub mod encodec;
pub mod raw;
pub mod stream;
pub mod surround;

use std::fmt;
use std::sync::Arc;

use serde::{Serialize, Serializer};

/// The pinned codec families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodecKind {
    /// `facebook/encodec_24khz`, multichannel batches at 24 kHz.
    Encodec,
    /// `descript/dac_44khz`, three fixed quantizers on the 44.1 kHz hop-512 grid.
    Dac,
    /// `hubertsiuzdak/snac_44khz`, strides 8/4/2 on the 44.1 kHz hop-3072 grid.
    Snac,
}

impl CodecKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Encodec => "EnCodec",
            Self::Dac => "DAC",
            Self::Snac => "SNAC",
        }
    }

    pub const fn model_rate(self) -> u32 {
        match self {
            Self::Encodec => 24_000,
            Self::Dac | Self::Snac => 44_100,
        }
    }
}

/// An inference device after resolving `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolvedDevice {
    Cpu,
    Mps,
    Cuda,
}

impl ResolvedDevice {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Mps => "mps",
            Self::Cuda => "cuda",
        }
    }
}

impl fmt::Display for ResolvedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for ResolvedDevice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

/// Per-call codec settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodecRequest {
    pub device: ResolvedDevice,
    /// EnCodec target bandwidth in kbps (6, 12 or 24); ignored by DAC and SNAC.
    pub bandwidth_kbps: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct CodecError(pub String);

/// One neural codec forward pass (encode, quantize, decode) on a mono stem at the model rate.
///
/// The output may carry trailing padding; callers trim to their declared grid and reject
/// short or nonfinite results. Implementations cache models per device.
pub trait NeuralCodec: Send + Sync {
    fn kind(&self) -> CodecKind;
    fn reconstruct(&self, stem: &[f32], request: &CodecRequest) -> Result<Vec<f32>, CodecError>;
}

/// The installed codecs; a missing entry is Python's `is_available() == False`.
#[derive(Clone, Default)]
pub struct CodecSet {
    pub encodec: Option<Arc<dyn NeuralCodec>>,
    pub dac: Option<Arc<dyn NeuralCodec>>,
    pub snac: Option<Arc<dyn NeuralCodec>>,
}

impl fmt::Debug for CodecSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodecSet")
            .field("encodec", &self.encodec.is_some())
            .field("dac", &self.dac.is_some())
            .field("snac", &self.snac.is_some())
            .finish()
    }
}

impl CodecSet {
    pub fn get(&self, kind: CodecKind) -> Option<&Arc<dyn NeuralCodec>> {
        match kind {
            CodecKind::Encodec => self.encodec.as_ref(),
            CodecKind::Dac => self.dac.as_ref(),
            CodecKind::Snac => self.snac.as_ref(),
        }
    }
}

/// How stereo pairs enter a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoMode {
    /// Each channel separately (`"independent"` / `"lr"`).
    Independent,
    /// Raw mid/side stems `(L + R) / 2`, `(L - R) / 2`.
    MidSide,
}

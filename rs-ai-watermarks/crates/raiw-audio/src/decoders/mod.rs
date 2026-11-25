//! Watermark decoders: neural forward passes behind traits, receiver logic ported from Python.
//!
//! Each `*Network` trait is the seam for a burn model in `raiw-gpu` (the `gpu` feature); the
//! readers here turn raw network outputs into the public readings, thresholds and framing rules.

pub mod audioseal;
pub mod perth;
pub mod silentcipher;
pub mod wavmark;

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use serde::{Serialize, Serializer};

pub use audioseal::{AudioSealNetwork, AudioSealOutput, AudioSealReading};
pub use perth::{PerthNetwork, PerthReading};
pub use silentcipher::{SilentCipherNetwork, SilentCipherReading};
pub use wavmark::{WavMarkNetwork, WavMarkReading, WavMarkSignatureReading};

/// Failures of a decoder check, split like the Python exception classes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecoderError {
    /// `ValueError`: the input cannot be read by this receiver.
    #[error("{0}")]
    Invalid(String),
    /// `RuntimeError`: the model or its runtime failed; the check stays unverified.
    #[error("{0}")]
    Failed(String),
    /// SilentCipher `ModelCheckError`: invalid model numerics, never retried on another device.
    #[error("{0}")]
    ModelCheck(String),
}

/// One named public decoder family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Decoder {
    AudioSeal,
    WavMark,
    Perth,
    SilentCipher,
}

impl Decoder {
    pub const NAMED: [Self; 4] = [Self::AudioSeal, Self::WavMark, Self::Perth, Self::SilentCipher];
    /// Decoders scanned by `scheme="auto"` (experimental SilentCipher excluded).
    pub const AUTO: [Self; 3] = [Self::AudioSeal, Self::WavMark, Self::Perth];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AudioSeal => "audioseal",
            Self::WavMark => "wavmark",
            Self::Perth => "perth",
            Self::SilentCipher => "silentcipher",
        }
    }

    /// The optional extra that installs this decoder.
    pub const fn extra(self) -> &'static str {
        match self {
            Self::AudioSeal => "audio-detect",
            Self::WavMark => "audio-wavmark",
            Self::Perth => "audio-perth",
            Self::SilentCipher => "audio-silentcipher",
        }
    }
}

impl fmt::Display for Decoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Decoder {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl FromStr for Decoder {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::NAMED
            .into_iter()
            .find(|d| d.as_str() == value)
            .ok_or_else(|| format!("unknown audio decoder {value}"))
    }
}

/// A decoder reading of any family; serializes as that dataclass's fields.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ModelReading {
    AudioSeal(AudioSealReading),
    WavMark(WavMarkReading),
    WavMarkSignature(WavMarkSignatureReading),
    Perth(PerthReading),
    SilentCipher(SilentCipherReading),
}

impl ModelReading {
    pub fn detected(&self) -> bool {
        match self {
            Self::AudioSeal(r) => r.detected(),
            Self::WavMark(r) => r.detected(),
            Self::WavMarkSignature(r) => r.detected(),
            Self::Perth(r) => r.detected(),
            Self::SilentCipher(r) => r.detected(),
        }
    }
}

/// The installed decoders; a missing entry is Python's `is_available() == False`.
#[derive(Clone, Default)]
pub struct DecoderSet {
    pub audioseal: Option<Arc<dyn AudioSealNetwork>>,
    pub wavmark: Option<Arc<dyn WavMarkNetwork>>,
    pub perth: Option<Arc<dyn PerthNetwork>>,
    pub silentcipher: Option<Arc<dyn SilentCipherNetwork>>,
}

impl fmt::Debug for DecoderSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecoderSet")
            .field("audioseal", &self.audioseal.is_some())
            .field("wavmark", &self.wavmark.is_some())
            .field("perth", &self.perth.is_some())
            .field("silentcipher", &self.silentcipher.is_some())
            .finish()
    }
}

impl DecoderSet {
    pub fn is_available(&self, decoder: Decoder) -> bool {
        match decoder {
            Decoder::AudioSeal => self.audioseal.is_some(),
            Decoder::WavMark => self.wavmark.is_some(),
            Decoder::Perth => self.perth.is_some(),
            Decoder::SilentCipher => self.silentcipher.is_some(),
        }
    }
}

/// `float(x.mean())` of a float32 array: float32 pairwise accumulation, float32 division.
pub(crate) fn mean_f32(values: &[f32]) -> f64 {
    let total = values
        .chunks(8192)
        .fold(0.0f32, |acc, chunk| acc + pairwise_f32(chunk));
    f64::from(total / values.len() as f32)
}

fn pairwise_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0f32, |acc, &v| acc + v);
    }
    if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let whole = n - n % 8;
        for block in a[8..whole].chunks_exact(8) {
            for (slot, &v) in r.iter_mut().zip(block) {
                *slot += v;
            }
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[whole..] {
            res += v;
        }
        return res;
    }
    let mut half = n / 2;
    half -= half % 8;
    pairwise_f32(&a[..half]) + pairwise_f32(&a[half..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_names_roundtrip() {
        for decoder in Decoder::NAMED {
            assert_eq!(decoder.as_str().parse::<Decoder>().unwrap(), decoder);
        }
    }

    #[test]
    fn float32_mean_uses_float32_accumulation() {
        assert_eq!(mean_f32(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }
}

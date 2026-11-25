//! Pinned public Perth v1 receiver (`_internal/perth.py`), separate from production Perth deployments.

use serde::Serialize;

use super::DecoderError;

pub const SAMPLE_RATE: u32 = 16_000;
pub const MIN_READ_SAMPLES: usize = 16_000;
pub const PACKAGE_VERSION: &str = "1.0.1";
pub const CHECKPOINT_SHA256: &str = "a15bce457ebc53ce5e6c9c3f11df78cf7ee2bf9cdab0a798902135b4c4027670";

/// The implicit Perth decoder on 16 kHz mono input, including its internal polyphase resampling
/// to the model clock and magnitude processing; returns the raw presence prediction.
pub trait PerthNetwork: Send + Sync {
    fn presence(&self, samples: &[f32]) -> Result<Vec<f32>, DecoderError>;
}

/// Implicit watermark presence, without a user-selected payload.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerthReading {
    pub confidence: f64,
}

impl PerthReading {
    pub fn detected(&self) -> bool {
        self.confidence > 0.5
    }
}

/// `perth.read(model, samples)`.
pub fn read(network: &dyn PerthNetwork, samples: &[f32]) -> Result<PerthReading, DecoderError> {
    if samples.len() < MIN_READ_SAMPLES || !samples.iter().all(|v| v.is_finite()) {
        return Err(DecoderError::Invalid(
            "Perth requires at least one second of finite mono samples at 16 kHz".into(),
        ));
    }
    let raw = network.presence(samples)?;
    match raw.as_slice() {
        [value] if value.is_finite() => Ok(PerthReading {
            confidence: f64::from(value.clamp(0.0, 1.0)),
        }),
        _ => Err(DecoderError::Failed(
            "Perth returned an invalid presence confidence".into(),
        )),
    }
}

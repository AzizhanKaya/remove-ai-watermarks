//! Pinned open AudioSeal receiver (`_internal/audioseal.py`): compatible-model presence and payload bits.

use serde::Serialize;

use super::{DecoderError, mean_f32};

pub const HF_REPO: &str = "facebook/audioseal";
pub const HF_REVISION: &str = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba";
pub const DETECTOR_SHA256: &str = "8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9";
pub const SAMPLE_RATE: u32 = 16_000;
pub const N_BITS: usize = 16;
pub const DETECTION_THRESHOLD: f32 = 0.5;
pub const MESSAGE_THRESHOLD: f32 = 0.5;
/// Native source clocks with a calibrated native-clock receiver.
pub const NATIVE_SAMPLE_RATES: [u32; 2] = [24_000, 48_000];

/// Raw detector outputs: per-sample watermark probabilities (class 1) and the message logits.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AudioSealOutput {
    pub probabilities: Vec<f32>,
    pub message: Vec<f32>,
}

/// The AudioSeal detector forward pass on one mono array at the clock it is given.
pub trait AudioSealNetwork: Send + Sync {
    fn detect(&self, samples: &[f32]) -> Result<AudioSealOutput, DecoderError>;
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioSealReading {
    pub mean_detect_prob: f64,
    pub frac_above_threshold: f64,
    pub decoded_bits: Vec<u8>,
}

impl AudioSealReading {
    /// The detector's own document decision rule at the default threshold.
    pub fn detected(&self) -> bool {
        self.frac_above_threshold >= f64::from(DETECTION_THRESHOLD)
    }

    pub fn label(&self) -> String {
        self.decoded_bits.iter().map(|b| char::from(b'0' + b)).collect()
    }
}

/// `audioseal.read(detector, samples)`.
pub fn read(network: &dyn AudioSealNetwork, samples: &[f32]) -> Result<AudioSealReading, DecoderError> {
    if samples.is_empty() || !samples.iter().all(|v| v.is_finite()) {
        return Err(DecoderError::Invalid(
            "AudioSeal samples must be nonempty finite mono audio".into(),
        ));
    }
    let output = network.detect(samples)?;
    let finite = output
        .probabilities
        .iter()
        .chain(&output.message)
        .all(|v| v.is_finite());
    if !finite || output.probabilities.is_empty() {
        return Err(DecoderError::Failed(
            "AudioSeal returned nonfinite detector values".into(),
        ));
    }
    let above = output
        .probabilities
        .iter()
        .filter(|&&p| p > DETECTION_THRESHOLD)
        .count();
    Ok(AudioSealReading {
        mean_detect_prob: mean_f32(&output.probabilities),
        frac_above_threshold: above as f64 / output.probabilities.len() as f64,
        decoded_bits: output
            .message
            .iter()
            .map(|&v| u8::from(v > MESSAGE_THRESHOLD))
            .collect(),
    })
}

/// Half-open bounds of fixed windows; a short remainder is its own final window.
pub fn window_bounds(
    total: usize,
    window_seconds: f64,
    sample_rate: u32,
) -> Result<Vec<(usize, usize)>, DecoderError> {
    if window_seconds <= 0.0 || sample_rate == 0 {
        return Err(DecoderError::Invalid(
            "window_s and sample_rate must be positive".into(),
        ));
    }
    let size = (window_seconds * f64::from(sample_rate)) as usize;
    if size == 0 {
        return Err(DecoderError::Invalid(
            "window_s is shorter than one sample".into(),
        ));
    }
    Ok((0..total)
        .step_by(size)
        .map(|start| (start, (start + size).min(total)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(AudioSealOutput);

    impl AudioSealNetwork for Fixed {
        fn detect(&self, _samples: &[f32]) -> Result<AudioSealOutput, DecoderError> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn thresholds_follow_the_reference() {
        let net = Fixed(AudioSealOutput {
            probabilities: vec![0.9, 0.5, 0.2, 0.7],
            message: vec![0.6, 0.5, -1.0],
        });
        let reading = read(&net, &[0.1; 4]).unwrap();
        assert_eq!(reading.frac_above_threshold, 0.5);
        assert!(reading.detected());
        assert_eq!(reading.label(), "100");
        assert!((reading.mean_detect_prob - 0.575).abs() < 1e-7);
    }

    #[test]
    fn nonfinite_outputs_are_failures() {
        let net = Fixed(AudioSealOutput {
            probabilities: vec![f32::NAN],
            message: vec![],
        });
        assert!(matches!(read(&net, &[0.1]), Err(DecoderError::Failed(_))));
        assert!(matches!(read(&net, &[]), Err(DecoderError::Invalid(_))));
    }

    #[test]
    fn windows_keep_the_partial_tail() {
        assert_eq!(
            window_bounds(35_000, 1.0, 16_000).unwrap(),
            vec![(0, 16_000), (16_000, 32_000), (32_000, 35_000)]
        );
    }
}

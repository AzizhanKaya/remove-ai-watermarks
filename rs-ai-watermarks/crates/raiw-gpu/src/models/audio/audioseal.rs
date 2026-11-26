//! AudioSeal 16-bit detector (`facebook/audioseal`, `detector_base.pth`).
//!
//! Mirrors `_internal/audioseal.py::read` on top of `audioseal==0.2.1`: a moshi SEANet
//! encoder with a 2-layer LSTM, a frame-to-sample transposed convolution, and a 1x1 head
//! whose first two channels are softmaxed into per-sample detection probabilities and the
//! other sixteen averaged and sigmoided into message bit probabilities.

use std::path::Path;

use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::s;

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub, HubFile};
use crate::io::{check_mono, mono_tensor, read_finite};
use crate::layers::{
    Conv1d, Conv1dSpec, ConvTranspose1d, ConvTranspose1dSpec, PadKind, SeanetConfig, SeanetEncoder,
    SeanetLayout,
};
use crate::weights::StateDict;

pub const HF_REPO: &str = "facebook/audioseal";
pub const HF_REVISION: &str = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba";
pub const DETECTOR_FILENAME: &str = "detector_base.pth";
pub const DETECTOR_SHA256: &str = "8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9";
pub const SAMPLE_RATE: u32 = 16_000;
pub const N_BITS: usize = 16;
pub const DETECTION_THRESHOLD: f32 = 0.5;
pub const MESSAGE_THRESHOLD: f32 = 0.5;

const OUTPUT_DIM: usize = 32;

/// The pinned checkpoint, digest-verified on every resolve.
pub const DETECTOR_FILE: HubFile<'static> =
    HubFile::new(HF_REPO, HF_REVISION, DETECTOR_FILENAME).with_sha256(DETECTOR_SHA256);

/// `audioseal_detector_16bits` card hyper-parameters.
pub fn detector_config() -> SeanetConfig {
    SeanetConfig {
        channels: 1,
        dimension: 128,
        n_filters: 32,
        n_residual_layers: 1,
        ratios: vec![8, 5, 4, 2],
        kernel_size: 7,
        last_kernel_size: 7,
        residual_kernel_size: 3,
        dilation_base: 2,
        compress: 2,
        lstm_layers: 2,
        causal: false,
        elu_alpha: 1.0,
        pad: PadKind::Zeros,
        true_skip: true,
        layout: SeanetLayout::Moshi,
    }
}

/// Raw detector outputs on one mono 16 kHz buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Per-sample probability that the sample is watermarked (`result[0, 1, :]`).
    pub probabilities: Vec<f32>,
    /// Per-bit probability that the decoded message bit is 1.
    pub message: [f32; N_BITS],
}

impl Detection {
    pub fn mean_detect_prob(&self) -> f64 {
        self.probabilities.iter().map(|&p| f64::from(p)).sum::<f64>() / self.probabilities.len().max(1) as f64
    }

    pub fn frac_above_threshold(&self) -> f64 {
        let above = self
            .probabilities
            .iter()
            .filter(|&&p| p > DETECTION_THRESHOLD)
            .count();
        above as f64 / self.probabilities.len().max(1) as f64
    }

    pub fn bits(&self) -> [u8; N_BITS] {
        self.message.map(|p| u8::from(p > MESSAGE_THRESHOLD))
    }

    /// The detector's own document decision rule at the default threshold.
    pub fn detected(&self) -> bool {
        self.frac_above_threshold() >= f64::from(DETECTION_THRESHOLD)
    }

    pub fn label(&self) -> String {
        self.bits().iter().map(|bit| char::from(b'0' + bit)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct AudioSealDetector {
    encoder: SeanetEncoder,
    reverse: ConvTranspose1d,
    head: Conv1d,
    device: ComputeDevice,
}

impl AudioSealDetector {
    /// Resolve the pinned checkpoint through the hub cache (downloading if allowed) and load it.
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        Self::from_checkpoint(&hub.resolve(&DETECTOR_FILE, fetch)?, device)
    }

    /// Load from a local `detector_base.pth` without digest checks.
    pub fn from_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_pytorch(path, Some("model"), &[])?;
        state.fold_weight_norm()?;
        let burn = device.burn();
        let config = detector_config();
        let encoder = SeanetEncoder::load(&mut state, "detector.0", &config, burn)?;
        let hop = config.hop_length();
        let reverse_spec = ConvTranspose1dSpec::new(config.dimension, OUTPUT_DIM, hop, hop);
        let reverse =
            ConvTranspose1d::load(&mut state, "detector.0.reverse_convolution", reverse_spec, burn)?;
        let head = Conv1d::load(
            &mut state,
            "detector.1",
            Conv1dSpec::new(OUTPUT_DIM, 2 + N_BITS, 1),
            burn,
        )?;
        state.finish()?;
        Ok(Self {
            encoder,
            reverse,
            head,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// Detection probabilities and message bits for mono samples at 16 kHz.
    pub fn detect(&self, samples: &[f32], sample_rate: u32) -> Result<Detection> {
        if sample_rate != SAMPLE_RATE {
            return Err(GpuError::Input(format!(
                "AudioSeal expects {SAMPLE_RATE} Hz input, got {sample_rate} Hz"
            )));
        }
        check_mono(samples, "AudioSeal")?;
        let frames = samples.len();
        let x = mono_tensor(samples, self.device.burn());
        let latent = self.encoder.forward(x)?;
        let upsampled = self.reverse.forward(latent).slice(s![.., .., 0..frames]);
        let result = self.head.forward(upsampled);
        let detect = softmax(result.clone().slice(s![.., 0..2, ..]), 1).slice(s![.., 1..2, ..]);
        let message: Tensor<3> = sigmoid(result.slice(s![.., 2.., ..]).mean_dim(2));
        let probabilities = read_finite(detect)?;
        let message = read_finite(message)?;
        let message: [f32; N_BITS] = message.try_into().map_err(|values: Vec<f32>| {
            GpuError::Readback(format!("expected {N_BITS} message bits, got {}", values.len()))
        })?;
        Ok(Detection {
            probabilities,
            message,
        })
    }

    /// One detection per fixed window, oldest first; a trailing partial window is kept.
    pub fn detect_intervals(&self, samples: &[f32], window_s: f64) -> Result<Vec<Detection>> {
        window_bounds(samples.len(), window_s, SAMPLE_RATE)?
            .into_iter()
            .map(|(start, end)| self.detect(&samples[start..end], SAMPLE_RATE))
            .collect()
    }
}

/// Half-open sample bounds of fixed-length windows (`audioseal.window_bounds`).
pub fn window_bounds(total_samples: usize, window_s: f64, sample_rate: u32) -> Result<Vec<(usize, usize)>> {
    if !(window_s > 0.0 && window_s.is_finite()) {
        return Err(GpuError::Input("window_s must be positive".to_owned()));
    }
    if sample_rate == 0 {
        return Err(GpuError::Input("sample_rate must be positive".to_owned()));
    }
    let size = (window_s * f64::from(sample_rate)) as usize;
    if size == 0 {
        return Err(GpuError::Input("window_s is shorter than one sample".to_owned()));
    }
    Ok((0..total_samples)
        .step_by(size)
        .map(|start| (start, (start + size).min(total_samples)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_bounds_keep_partial_tail() {
        assert_eq!(
            window_bounds(40_000, 1.0, 16_000).unwrap(),
            vec![(0, 16_000), (16_000, 32_000), (32_000, 40_000)]
        );
        assert!(window_bounds(10, 0.0, 16_000).is_err());
        assert!(window_bounds(10, 1e-6, 16_000).is_err());
    }

    #[test]
    fn decision_rule_matches_python() {
        let reading = Detection {
            probabilities: vec![0.9, 0.6, 0.1, 0.4],
            message: [0.7; N_BITS],
        };
        assert!(reading.detected());
        assert!((reading.mean_detect_prob() - 0.5).abs() < 1e-6);
        assert_eq!(reading.label(), "1".repeat(N_BITS));
    }
}

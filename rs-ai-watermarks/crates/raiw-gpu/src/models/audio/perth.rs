//! Perth v1 implicit watermark decoder (`resemble-perth==1.0.1`).
//!
//! Mirrors `_internal/perth.py::read`: 16 -> 32 kHz polyphase resampling, a 2048/320 STFT
//! normalized to dB, then three multi-rate conv branches (0.75x, 1x, 1.25x time) whose
//! masked-mean watermark scores are combined by softmax attention into one confidence.
//! The checkpoint ships inside the PyPI wheel, so it is extracted once into the model cache.

use std::path::Path;

use burn::tensor::activation::{leaky_relu, softmax};
use burn::tensor::{Device, Tensor, TensorData};

use crate::device::ComputeDevice;
use crate::dsp::{Stft, resample_poly};
use crate::error::{GpuError, Result};
use crate::hub::{ArchiveMember, Fetch, Hub};
use crate::io::read_finite;
use crate::layers::{Conv1d, Conv1dSpec};
use crate::weights::StateDict;

pub const PACKAGE_VERSION: &str = "1.0.1";
pub const CHECKPOINT_SHA256: &str = "a15bce457ebc53ce5e6c9c3f11df78cf7ee2bf9cdab0a798902135b4c4027670";
pub const SAMPLE_RATE: u32 = 16_000;
pub const MIN_READ_SAMPLES: usize = 16_000;

const MODEL_RATE: u32 = 32_000;
const N_FFT: usize = 2048;
const HOP: usize = 320;
const HIDDEN: usize = 256;
const SUBBAND: usize = 128;
const MAGNITUDE_MIN: f64 = 1e-9;
const HEADROOM_DB: f64 = 15.0;
const MASK_FRACTION: f32 = 0.05;
const BRANCHES: [(&str, f64); 3] = [("slow", 1.25), ("normal", 1.0), ("fast", 0.75)];

/// The pinned checkpoint inside the `resemble_perth-1.0.1` wheel.
pub const CHECKPOINT: ArchiveMember<'static> = ArchiveMember {
    url: "https://files.pythonhosted.org/packages/77/cc/73226dd776f8e9c2975f64f4efc22988fb37e5b185ba5cccd6f2e7196954/resemble_perth-1.0.1-py3-none-any.whl",
    archive_sha256: "65e9c37531b1a128a4a56226b75dece4521683cf3611b0b2a5ffe234f00c9342",
    member: "perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar",
    sha256: CHECKPOINT_SHA256,
    cache_path: "perth/1.0.1/perth_net_250000.pth.tar",
};

/// Implicit watermark presence confidence in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerthReading {
    pub confidence: f32,
    /// The decoder output before clipping.
    pub raw: f32,
}

impl PerthReading {
    pub fn detected(&self) -> bool {
        self.confidence > 0.5
    }
}

#[derive(Debug, Clone)]
struct Branch {
    layers: Vec<Conv1d>,
    rate: f64,
}

impl Branch {
    fn load(state: &mut StateDict, name: &str, rate: f64, device: &Device) -> Result<Self> {
        let prefix = format!("decoder.{name}_layers");
        let mut layers = vec![Conv1d::load(
            state,
            &format!("{prefix}.0.conv"),
            Conv1dSpec::new(SUBBAND, HIDDEN, 1),
            device,
        )?];
        for i in 1..=5 {
            let spec = Conv1dSpec::new(HIDDEN, HIDDEN, 7).padding(3);
            layers.push(Conv1d::load(state, &format!("{prefix}.{i}.conv"), spec, device)?);
        }
        layers.push(Conv1d::load(
            state,
            &format!("{prefix}.6.conv"),
            Conv1dSpec::new(HIDDEN, 2, 1),
            device,
        )?);
        Ok(Self { layers, rate })
    }

    /// `(attention, watermark)` masked means over time.
    fn forward(&self, x: Tensor<3>, mask: Tensor<3>) -> (Tensor<2>, Tensor<2>) {
        let last = self.layers.len() - 1;
        let out = self.layers.iter().enumerate().fold(x, |h, (i, conv)| {
            let h = conv.forward(h);
            if i == last { h } else { leaky_relu(h, 0.01) }
        });
        let denominator = mask.clone().sum_dim(2);
        let pooled = (out * mask).sum_dim(2) / denominator;
        let [batch, _, _] = pooled.dims();
        (
            pooled.clone().narrow(1, 0, 1).reshape([batch, 1]),
            pooled.narrow(1, 1, 1).reshape([batch, 1]),
        )
    }
}

#[derive(Debug, Clone)]
pub struct PerthDetector {
    branches: Vec<Branch>,
    stft: Stft,
    device: ComputeDevice,
}

impl PerthDetector {
    /// Extract the pinned checkpoint from the wheel into the model cache (downloading if allowed) and load it.
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        Self::from_checkpoint(&hub.resolve_archive_member(&CHECKPOINT, fetch)?, device)
    }

    /// Load from a local `perth_net_250000.pth.tar` without digest checks.
    pub fn from_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_pytorch(path, Some("model"), &[])?;
        state.discard(|name| name.starts_with("encoder.") || name.starts_with("ap."));
        let burn = device.burn();
        let branches = BRANCHES
            .iter()
            .map(|&(name, rate)| Branch::load(&mut state, name, rate, burn))
            .collect::<Result<Vec<_>>>()?;
        state.finish()?;
        Ok(Self {
            branches,
            stft: Stft::new(N_FFT, HOP),
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// Presence confidence for at least one second of finite mono 16 kHz samples.
    pub fn read(&self, samples: &[f32], sample_rate: u32) -> Result<PerthReading> {
        if sample_rate != SAMPLE_RATE {
            return Err(GpuError::Input(format!(
                "Perth expects {SAMPLE_RATE} Hz input, got {sample_rate} Hz"
            )));
        }
        if samples.len() < MIN_READ_SAMPLES || !samples.iter().all(|x| x.is_finite()) {
            return Err(GpuError::Input(
                "Perth requires at least one second of finite mono samples at 16 kHz".to_owned(),
            ));
        }
        let native = resample_poly(samples, MODEL_RATE as usize, SAMPLE_RATE as usize);
        let (magnitudes, frames) = self.magnitudes(&native)?;
        let mask = frame_mask(&magnitudes, frames);
        let subband = &magnitudes[..SUBBAND * frames];
        let burn = self.device.burn();
        let mut attention = Vec::with_capacity(self.branches.len());
        let mut marks = Vec::with_capacity(self.branches.len());
        for branch in &self.branches {
            let steps = ((frames as f64) * branch.rate) as usize;
            let resized = if steps == frames {
                subband.to_vec()
            } else {
                lerp_rows(subband, SUBBAND, frames, steps)
            };
            let x: Tensor<3> = Tensor::from_data(TensorData::new(resized, [1, SUBBAND, steps]), burn);
            let m: Tensor<3> = Tensor::from_data(TensorData::new(nearest(&mask, steps), [1, 1, steps]), burn);
            let (attn, mark) = branch.forward(x, m);
            attention.push(attn);
            marks.push(mark);
        }
        let weights = softmax(Tensor::cat(attention, 1), 1);
        let combined = (Tensor::cat(marks, 1) * weights).sum_dim(1);
        let raw = read_finite(combined)?
            .first()
            .copied()
            .ok_or(GpuError::NonFinite)?;
        Ok(PerthReading {
            confidence: raw.clamp(0.0, 1.0),
            raw,
        })
    }

    /// Normalized dB magnitudes, row-major `[bin][frame]`.
    fn magnitudes(&self, signal: &[f32]) -> Result<(Vec<f32>, usize)> {
        let spec = self.stft.forward(signal)?;
        let min_db = 20.0 * MAGNITUDE_MIN.log10();
        let mut out = vec![0.0f32; spec.bins * spec.frames];
        for frame in 0..spec.frames {
            for bin in 0..spec.bins {
                let magnitude = (spec.at(frame, bin).norm() as f32).max(MAGNITUDE_MIN as f32);
                let db = 20.0 * f64::from(magnitude.log10());
                out[bin * spec.frames + frame] = ((db - min_db) / (-min_db + HEADROOM_DB)) as f32;
            }
        }
        Ok((out, spec.frames))
    }
}

/// `magmask`: frames whose summed magnitude exceeds 5% of the loudest frame.
fn frame_mask(magnitudes: &[f32], frames: usize) -> Vec<f32> {
    let sums: Vec<f32> = (0..frames)
        .map(|t| magnitudes.iter().skip(t).step_by(frames).sum())
        .collect();
    let threshold = sums.iter().copied().fold(f32::NEG_INFINITY, f32::max) * MASK_FRACTION;
    sums.iter()
        .map(|&s| if s > threshold { 1.0 } else { 0.0 })
        .collect()
}

/// `F.interpolate(mode="linear", align_corners=True)` per row.
fn lerp_rows(rows: &[f32], channels: usize, len: usize, size: usize) -> Vec<f32> {
    let scale = if size > 1 {
        (len - 1) as f32 / (size - 1) as f32
    } else {
        0.0
    };
    let mut out = Vec::with_capacity(channels * size);
    for row in rows.chunks_exact(len).take(channels) {
        for i in 0..size {
            let src = scale * i as f32;
            let i0 = (src as usize).min(len - 1);
            let i1 = if i0 < len - 1 { i0 + 1 } else { i0 };
            let lambda = src - i0 as f32;
            out.push((1.0 - lambda) * row[i0] + lambda * row[i1]);
        }
    }
    out
}

/// `F.interpolate(mode="nearest")` of a 1-D mask.
fn nearest(mask: &[f32], size: usize) -> Vec<f32> {
    let scale = mask.len() as f32 / size as f32;
    (0..size)
        .map(|i| mask[((i as f32 * scale).floor() as usize).min(mask.len() - 1)])
        .collect()
}

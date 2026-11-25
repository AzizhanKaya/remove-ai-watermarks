//! WavMark 16 kHz decoder (`M4869/WavMark`, `wavmark==0.0.3`).
//!
//! Mirrors `_internal/wavmark.py::read`: one-second windows every 50 ms, STFT (1000/400),
//! the 8-block invertible Hinet run in reverse on device, ISTFT, and a 16000->32 linear head;
//! windows whose first 16 bits equal the synchronization pattern vote on the payload.

use std::path::Path;

use burn::tensor::activation::{leaky_relu, sigmoid};
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, TensorData};
use realfft::num_complex::Complex;

use crate::device::ComputeDevice;
use crate::dsp::{Spectrogram, Stft};
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub, HubFile};
use crate::io::read_finite;
use crate::weights::StateDict;

pub const MODEL_REPOSITORY: &str = "M4869/WavMark";
pub const MODEL_REVISION: &str = "0ad3c7b74f641bddb61f6b85cdf2de0d93a5bfef";
pub const MODEL_FILENAME: &str = "step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl";
pub const MODEL_SHA256: &str = "7a3d873d115c2c644685fb4cf109bdb299718296b88c871d3cf1c8e4cbc13144";
pub const SAMPLE_RATE: u32 = 16_000;
pub const WINDOW: usize = 16_000;
pub const STEP: usize = 800;
pub const NUM_BITS: usize = 32;
pub const SYNC_PATTERN: [u8; 16] = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0];

const N_FFT: usize = 1000;
const HOP: usize = 400;
const BLOCKS: usize = 8;
const DENSE: usize = 32;
const CLAMP: f64 = 2.0;
const BATCH: usize = 64;

pub const MODEL_FILE: HubFile<'static> =
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, MODEL_FILENAME).with_sha256(MODEL_SHA256);

/// A payload accepted by the exact synchronization-pattern check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavMarkReading {
    pub payload: Option<[u8; 16]>,
    pub start_samples: Vec<usize>,
}

impl WavMarkReading {
    pub fn detected(&self) -> bool {
        self.payload.is_some()
    }
}

/// `ResidualDenseBlock_out`: five 3x3 convs with dense concatenation and LeakyReLU.
#[derive(Debug, Clone)]
struct DenseBlock {
    convs: Vec<(Tensor<4>, Tensor<1>)>,
}

impl DenseBlock {
    fn load(state: &mut StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let convs = (0..5)
            .map(|i| {
                let input = 2 + i * DENSE;
                let output = if i == 4 { 2 } else { DENSE };
                let name = format!("{prefix}.conv{}", i + 1);
                Ok((
                    state.tensor(&format!("{name}.weight"), [output, input, 3, 3], device)?,
                    state.tensor(&format!("{name}.bias"), [output], device)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { convs })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let conv = |features: &[Tensor<4>], (weight, bias): &(Tensor<4>, Tensor<1>)| {
            let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
            conv2d(
                Tensor::cat(features.to_vec(), 1),
                weight.clone(),
                Some(bias.clone()),
                options,
            )
        };
        let mut features = vec![x];
        let Some((last, hidden)) = self.convs.split_last() else {
            return features.remove(0);
        };
        for layer in hidden {
            let out = leaky_relu(conv(&features, layer), 0.01);
            features.push(out);
        }
        conv(&features, last)
    }
}

#[derive(Debug, Clone)]
struct InvBlock {
    r: DenseBlock,
    y: DenseBlock,
    f: DenseBlock,
}

impl InvBlock {
    /// Reverse pass: recover `(x1, x2)` from the block output.
    fn reverse(&self, x1: Tensor<4>, x2: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let s1 = self.r.forward(x1.clone());
        let t1 = self.y.forward(x1.clone());
        let scale = sigmoid(s1).sub_scalar(0.5).mul_scalar(CLAMP * 2.0).exp();
        let y2 = (x2 - t1) / scale;
        let t2 = self.f.forward(y2.clone());
        (x1 - t2, y2)
    }
}

#[derive(Debug, Clone)]
pub struct WavMarkDecoder {
    blocks: Vec<InvBlock>,
    fc_back_w: Tensor<2>,
    fc_back_b: Tensor<1>,
    stft: Stft,
    device: ComputeDevice,
}

impl WavMarkDecoder {
    /// Resolve the pinned checkpoint through the hub cache (downloading if allowed) and load it.
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        Self::from_checkpoint(&hub.resolve(&MODEL_FILE, fetch)?, device)
    }

    /// Load from a local checkpoint without digest checks.
    pub fn from_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_pytorch(path, None, &[])?;
        let burn = device.burn();
        let blocks = (0..BLOCKS)
            .map(|i| {
                let prefix = format!("hinet.inv_blocks.{i}");
                Ok(InvBlock {
                    r: DenseBlock::load(&mut state, &format!("{prefix}.r"), burn)?,
                    y: DenseBlock::load(&mut state, &format!("{prefix}.y"), burn)?,
                    f: DenseBlock::load(&mut state, &format!("{prefix}.f"), burn)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let fc_back_w: Tensor<2> = state.tensor("watermark_fc_back.weight", [NUM_BITS, WINDOW], burn)?;
        let fc_back_b = state.tensor("watermark_fc_back.bias", [NUM_BITS], burn)?;
        state.discard(|name| name.starts_with("watermark_fc."));
        state.finish()?;
        Ok(Self {
            blocks,
            fc_back_w: fc_back_w.transpose(),
            fc_back_b,
            stft: Stft::new(N_FFT, HOP),
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// `Model.decode` on one-second windows: raw clamped bit scores per window.
    pub fn decode(&self, windows: &[&[f32]]) -> Result<Vec<[f32; NUM_BITS]>> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        if windows.iter().any(|w| w.len() != WINDOW) {
            return Err(GpuError::Input(format!(
                "WavMark windows must hold exactly {WINDOW} samples"
            )));
        }
        let spectra = windows
            .iter()
            .map(|w| self.stft.forward(w))
            .collect::<Result<Vec<_>>>()?;
        let (frames, bins) = (spectra[0].frames, spectra[0].bins);
        let batch = windows.len();
        let mut planes = vec![0.0f32; batch * 2 * frames * bins];
        for (b, spec) in spectra.iter().enumerate() {
            for (i, value) in spec.values.iter().enumerate() {
                planes[(b * 2) * frames * bins + i] = value.re as f32;
                planes[(b * 2 + 1) * frames * bins + i] = value.im as f32;
            }
        }
        let burn = self.device.burn();
        let x: Tensor<4> = Tensor::from_data(TensorData::new(planes, [batch, 2, frames, bins]), burn);
        let (_, message) = self
            .blocks
            .iter()
            .rev()
            .fold((x.clone(), x), |(x1, x2), block| block.reverse(x1, x2));
        let message = read_finite(message)?;
        let mut expanded = Vec::with_capacity(batch * WINDOW);
        for b in 0..batch {
            let plane = |c: usize| &message[(b * 2 + c) * frames * bins..(b * 2 + c + 1) * frames * bins];
            let values = plane(0)
                .iter()
                .zip(plane(1))
                .map(|(&re, &im)| Complex::new(f64::from(re), f64::from(im)))
                .collect();
            expanded.extend(self.stft.inverse(&Spectrogram { frames, bins, values })?);
        }
        let restored: Tensor<2> = Tensor::from_data(TensorData::new(expanded, [batch, WINDOW]), burn);
        let raw = (restored.matmul(self.fc_back_w.clone()) + self.fc_back_b.clone().unsqueeze_dim(0))
            .clamp(-1.0, 1.0);
        let raw = read_finite(raw)?;
        raw.chunks_exact(NUM_BITS)
            .map(|row| {
                <[f32; NUM_BITS]>::try_from(row)
                    .map_err(|_| GpuError::Readback("WavMark row width".to_owned()))
            })
            .collect()
    }

    /// Slide one-second windows every 50 ms (always covering the final window) and vote.
    pub fn read(&self, samples: &[f32], sample_rate: u32) -> Result<WavMarkReading> {
        if sample_rate != SAMPLE_RATE {
            return Err(GpuError::Input(format!(
                "WavMark expects {SAMPLE_RATE} Hz input, got {sample_rate} Hz"
            )));
        }
        if samples.len() < WINDOW || !samples.iter().all(|x| x.is_finite()) {
            return Err(GpuError::Input(format!(
                "WavMark requires at least {WINDOW} finite mono samples at 16 kHz"
            )));
        }
        let end = samples.len() - WINDOW;
        let mut starts: Vec<usize> = (0..=end).step_by(STEP).collect();
        if starts.last() != Some(&end) {
            starts.push(end);
        }
        let mut locations = Vec::new();
        let mut votes = [0usize; 16];
        for chunk in starts.chunks(BATCH) {
            let windows: Vec<&[f32]> = chunk.iter().map(|&s| &samples[s..s + WINDOW]).collect();
            for (&start, raw) in chunk.iter().zip(self.decode(&windows)?) {
                let bits = raw.map(|v| u8::from(v >= 0.5));
                if bits[..16] == SYNC_PATTERN {
                    locations.push(start);
                    for (vote, &bit) in votes.iter_mut().zip(&bits[16..]) {
                        *vote += usize::from(bit);
                    }
                }
            }
        }
        if locations.is_empty() {
            return Ok(WavMarkReading {
                payload: None,
                start_samples: Vec::new(),
            });
        }
        let payload = votes.map(|v| u8::from(2 * v >= locations.len()));
        Ok(WavMarkReading {
            payload: Some(payload),
            start_samples: locations,
        })
    }
}

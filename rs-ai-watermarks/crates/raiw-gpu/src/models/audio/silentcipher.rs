//! SilentCipher 44.1 kHz decoder (`sony/silentcipher`, `dec_m_0.ckpt`).
//!
//! Mirrors `_internal/silentcipher.py` + `_silentcipher_network.py`: power normalization,
//! a 4096-point STFT magnitude per phase shift (host FFT), ten gated 3x3 conv layers with
//! training-mode batch normalization (per-input statistics), a 1024-band linear head, and
//! the project's majority-vote receiver over 21-symbol message periods.

use std::path::Path;

use burn::tensor::activation::sigmoid;
use burn::tensor::{Device, Tensor, TensorData};

use crate::device::ComputeDevice;
use crate::dsp::Stft;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub, HubFile};
use crate::io::read_finite;
use crate::layers::{Conv2d, Conv2dSpec};
use crate::weights::StateDict;

pub const MODEL_REPOSITORY: &str = "sony/silentcipher";
pub const MODEL_REVISION: &str = "a1c4d021905e0dc5b24be5f68db5fc4dba410ee1";
pub const DECODER_FILENAME: &str = "44_1_khz/73999_iteration/dec_m_0.ckpt";
pub const DECODER_SHA256: &str = "829540c058270d29788f05294894d45bf436e44add0e1099422242ebb94b7088";
pub const SAMPLE_RATE: u32 = 44_100;
pub const WINDOW_SECONDS: usize = 3;
pub const MIN_READ_SAMPLES: usize = SAMPLE_RATE as usize * WINDOW_SECONDS;
pub const PERIOD: usize = 21;
pub const MIN_CYCLES: usize = 3;
pub const THRESHOLD: f64 = 0.9;
pub const HOP_LENGTH: usize = 2048;
pub const N_FFT: usize = 4096;
pub const PHASE_STEP: usize = 10;
pub const MESSAGE_DIM: usize = 5;
pub const MESSAGE_BAND: usize = 1024;

const TARGET_POWER: f64 = 0.002_837_200_844_477_648;
const HIDDEN: usize = 128;
const LAYERS: usize = 10;
const BN_EPS: f64 = 1e-5;

pub const DECODER_FILE: HubFile<'static> =
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, DECODER_FILENAME).with_sha256(DECODER_SHA256);

/// Raw decoder scores `[MESSAGE_DIM][frames]` for one phase shift.
#[derive(Debug, Clone, PartialEq)]
pub struct Scores {
    pub frames: usize,
    /// Row-major `MESSAGE_DIM x frames`.
    pub values: Vec<f32>,
}

impl Scores {
    pub fn row(&self, symbol: usize) -> &[f32] {
        &self.values[symbol * self.frames..(symbol + 1) * self.frames]
    }

    /// Winning symbol per frame (first maximum, as `np.argmax`).
    pub fn argmax(&self) -> Vec<u8> {
        (0..self.frames)
            .map(|t| {
                let mut best = 0;
                for symbol in 1..MESSAGE_DIM {
                    if self.values[symbol * self.frames + t] > self.values[best * self.frames + t] {
                        best = symbol;
                    }
                }
                best as u8
            })
            .collect()
    }
}

/// Accepted payload or a complete negative under the receiver profile.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub payload: Option<[u8; 5]>,
    pub symbol_consistency: f64,
    pub phase_offset: Option<usize>,
    pub phases_checked: usize,
}

impl Reading {
    pub fn detected(&self) -> bool {
        self.payload.is_some()
    }
}

#[derive(Debug, Clone)]
struct GatedLayer {
    conv: Conv2d,
    gate: Conv2d,
    bn_w: Tensor<4>,
    bn_b: Tensor<4>,
}

impl GatedLayer {
    fn load(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<Self> {
        let bn_w: Tensor<1> = state.tensor(&format!("{prefix}.bn.weight"), [output], device)?;
        let bn_b: Tensor<1> = state.tensor(&format!("{prefix}.bn.bias"), [output], device)?;
        for buffer in ["running_mean", "running_var", "num_batches_tracked"] {
            state.take_any(&format!("{prefix}.bn.{buffer}"))?;
        }
        let spec = Conv2dSpec::new(input, output, 3).padding(1);
        Ok(Self {
            conv: Conv2d::load(state, &format!("{prefix}.conv"), spec, device)?,
            gate: Conv2d::load(state, &format!("{prefix}.gate"), spec, device)?,
            bn_w: bn_w.reshape([1, output, 1, 1]),
            bn_b: bn_b.reshape([1, output, 1, 1]),
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let h = self.conv.forward(x.clone()) * sigmoid(self.gate.forward(x));
        let mean = h.clone().mean_dim(3).mean_dim(2);
        let centered = h - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(3).mean_dim(2);
        centered / var.add_scalar(BN_EPS).sqrt() * self.bn_w.clone() + self.bn_b.clone()
    }
}

#[derive(Debug, Clone)]
pub struct SilentCipherDecoder {
    layers: Vec<GatedLayer>,
    /// The cheap 1024-band head runs on the host, next to the score decoding.
    linear_w: Vec<f32>,
    linear_b: f32,
    stft: Stft,
    device: ComputeDevice,
}

impl SilentCipherDecoder {
    /// Resolve the pinned decoder through the hub cache (downloading if allowed) and load it.
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        Self::from_checkpoint(&hub.resolve(&DECODER_FILE, fetch)?, device)
    }

    /// Load from a local `dec_m_0.ckpt` without digest checks.
    pub fn from_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_pytorch(path, None, &[("^module\\.", "")])?;
        let burn = device.burn();
        let layers = (0..LAYERS)
            .map(|index| {
                let input = if index == 0 { 1 } else { HIDDEN };
                let output = if index == LAYERS - 1 { MESSAGE_DIM } else { HIDDEN };
                GatedLayer::load(
                    &mut state,
                    &format!("main.{}", 2 * index + 1),
                    input,
                    output,
                    burn,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let linear_w = state.take("linear.weight", &[1, MESSAGE_BAND])?.data;
        let linear_b = state.take("linear.bias", &[1])?.data[0];
        state.finish()?;
        Ok(Self {
            layers,
            linear_w,
            linear_b,
            stft: Stft::new(N_FFT, HOP_LENGTH),
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// The receiver: normalize power, then try phase shifts until a payload passes the threshold.
    pub fn read(&self, samples: &[f32], sample_rate: u32) -> Result<Reading> {
        if sample_rate != SAMPLE_RATE {
            return Err(GpuError::Input(format!(
                "SilentCipher expects {SAMPLE_RATE} Hz input, got {sample_rate} Hz"
            )));
        }
        if samples.len() < MIN_READ_SAMPLES || !samples.iter().all(|x| x.is_finite()) {
            return Err(GpuError::Input(
                "SilentCipher requires at least three seconds of finite mono audio".to_owned(),
            ));
        }
        let normalized = normalize_power(samples)?;
        let mut best = 0.0f64;
        let mut checked = 0;
        for shift in (0..HOP_LENGTH).step_by(PHASE_STEP) {
            let scores = self.scores(&normalized[shift..])?;
            let (payload, consistency) = decode_sequence(&scores)?;
            checked += 1;
            best = best.max(consistency);
            if let Some(payload) = payload.filter(|_| consistency >= THRESHOLD) {
                return Ok(Reading {
                    payload: Some(payload),
                    symbol_consistency: consistency,
                    phase_offset: Some(shift),
                    phases_checked: checked,
                });
            }
        }
        Ok(Reading {
            payload: None,
            symbol_consistency: best,
            phase_offset: None,
            phases_checked: checked,
        })
    }

    /// Decoder scores for an already power-normalized signal (one phase shift).
    pub fn scores(&self, normalized: &[f32]) -> Result<Scores> {
        let (magnitudes, frames) = self.magnitude(normalized)?;
        let burn = self.device.burn();
        let x: Tensor<4> = Tensor::from_data(TensorData::new(magnitudes, [1, 1, MESSAGE_BAND, frames]), burn);
        // Flush per layer: fusing the whole 10-layer graph corrupts tail tiles on RADV/wgpu
        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(hidden);
            burn.flush().map_err(|err| GpuError::Readback(err.to_string()))?;
        }
        let hidden = read_finite(hidden)?;
        let mut values = vec![0.0f32; MESSAGE_DIM * frames];
        for (symbol, plane) in hidden.chunks_exact(MESSAGE_BAND * frames).enumerate() {
            let out = &mut values[symbol * frames..(symbol + 1) * frames];
            for (band, row) in plane.chunks_exact(frames).enumerate() {
                let weight = self.linear_w[band];
                for (acc, &h) in out.iter_mut().zip(row) {
                    *acc += weight * h;
                }
            }
            for acc in out.iter_mut() {
                *acc += self.linear_b;
            }
        }
        Ok(Scores { frames, values })
    }

    /// `|STFT|` of the zero-padded signal, bands `0..1024`, row-major `[band][frame]`.
    fn magnitude(&self, signal: &[f32]) -> Result<(Vec<f32>, usize)> {
        let mut padded = signal.to_vec();
        padded.resize(signal.len() + (N_FFT - signal.len() % N_FFT), 0.0);
        let spec = self.stft.forward(&padded)?;
        let frames = spec.frames;
        let mut out = vec![0.0f32; MESSAGE_BAND * frames];
        for frame in 0..frames {
            for band in 0..MESSAGE_BAND {
                out[band * frames + frame] = spec.at(frame, band).norm() as f32;
            }
        }
        Ok((out, frames))
    }
}

/// Scale samples to the decoder's training power (`0.0028372 / mean(x^2)`).
pub fn normalize_power(samples: &[f32]) -> Result<Vec<f32>> {
    let power =
        samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / samples.len().max(1) as f64;
    if !power.is_finite() || power <= 0.0 {
        return Err(GpuError::Input(
            "SilentCipher cannot normalize silent or invalid input power".to_owned(),
        ));
    }
    let scale = (TARGET_POWER as f32 / power as f32).sqrt();
    Ok(samples.iter().map(|&x| x * scale).collect())
}

/// `_decode_sequence`: majority symbols per period position, consistency, and the 5-byte payload.
pub fn decode_sequence(scores: &Scores) -> Result<(Option<[u8; 5]>, f64)> {
    let length = scores.frames / PERIOD * PERIOD;
    if length < PERIOD * MIN_CYCLES {
        return Err(GpuError::Input(
            "SilentCipher returned too few complete message repetitions".to_owned(),
        ));
    }
    let predicted = &scores.argmax()[..length];
    let symbols: Vec<u8> = (0..PERIOD)
        .map(|position| {
            let mut counts = [0usize; MESSAGE_DIM];
            for row in predicted.chunks_exact(PERIOD) {
                counts[usize::from(row[position])] += 1;
            }
            let best = counts.iter().copied().max().unwrap_or(0);
            counts.iter().position(|&c| c == best).unwrap_or(0) as u8
        })
        .collect();
    let agree = predicted
        .chunks_exact(PERIOD)
        .flat_map(|row| row.iter().zip(&symbols))
        .filter(|(a, b)| a == b)
        .count();
    let consistency = agree as f64 / length as f64;
    let zeros: Vec<usize> = symbols
        .iter()
        .enumerate()
        .filter(|(_, s)| **s == 0)
        .map(|(i, _)| i)
        .collect();
    let [end] = zeros.as_slice() else {
        return Ok((None, consistency));
    };
    let letters: Vec<u8> = symbols[end + 1..]
        .iter()
        .chain(&symbols[..*end])
        .map(|s| s - 1)
        .collect();
    if letters.len() != 20 || letters.iter().any(|&l| l > 3) {
        return Ok((None, consistency));
    }
    let mut payload = [0u8; 5];
    for (byte, chunk) in payload.iter_mut().zip(letters.chunks_exact(4)) {
        *byte = chunk.iter().fold(0u8, |acc, &l| (acc << 2) | l);
    }
    Ok((Some(payload), consistency))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "GPU layer bisection; needs the cached checkpoint"]
    fn layers_match_cpu_on_gpu() {
        let Ok(gpu) = ComputeDevice::select(crate::DeviceRequest::Cuda) else {
            return;
        };
        let Ok(path) = Hub::from_env().resolve(&DECODER_FILE, Fetch::CacheOnly) else {
            return;
        };
        let cpu = ComputeDevice::cpu();
        let (a, b) = (
            SilentCipherDecoder::from_checkpoint(&path, &cpu).unwrap(),
            SilentCipherDecoder::from_checkpoint(&path, &gpu).unwrap(),
        );
        let samples: Vec<f32> = (0..141_120)
            .map(|i| ((i as f32) * 0.031).sin() * 0.1 + ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.01)
            .collect();
        let (mags, frames) = a.magnitude(&normalize_power(&samples).unwrap()).unwrap();
        let mut xa: Tensor<4> = Tensor::from_data(
            TensorData::new(mags.clone(), [1, 1, MESSAGE_BAND, frames]),
            cpu.burn(),
        );
        let mut xb: Tensor<4> =
            Tensor::from_data(TensorData::new(mags, [1, 1, MESSAGE_BAND, frames]), gpu.burn());
        for (i, (la, lb)) in a.layers.iter().zip(&b.layers).enumerate() {
            xa = la.forward(xa);
            xb = lb.forward(xb);
            let (va, vb) = (read_finite(xa.clone()).unwrap(), read_finite(xb.clone()).unwrap());
            let diff = va
                .iter()
                .zip(&vb)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            eprintln!("frames {frames} layer {i}: max diff {diff:.2e}");
        }
    }

    fn scores_from_symbols(symbols: &[u8], cycles: usize) -> Scores {
        let frames = symbols.len() * cycles;
        let mut values = vec![0.0f32; MESSAGE_DIM * frames];
        for t in 0..frames {
            values[usize::from(symbols[t % symbols.len()]) * frames + t] = 1.0;
        }
        Scores { frames, values }
    }

    #[test]
    fn decodes_rotated_payload() {
        let payload = [0x12u8, 0x34, 0x56, 0x78, 0x9a];
        let letters: Vec<u8> = payload
            .iter()
            .flat_map(|b| (0..4).rev().map(move |k| (b >> (2 * k)) & 3))
            .collect();
        let mut symbols: Vec<u8> = vec![0];
        symbols.extend(letters.iter().map(|l| l + 1));
        symbols.rotate_left(7);
        let (decoded, consistency) = decode_sequence(&scores_from_symbols(&symbols, 3)).unwrap();
        assert_eq!(decoded, Some(payload));
        assert_eq!(consistency, 1.0);
    }

    #[test]
    fn rejects_short_and_ambiguous_sequences() {
        let symbols = [1u8; PERIOD];
        assert!(decode_sequence(&scores_from_symbols(&symbols, 2)).is_err());
        let (decoded, _) = decode_sequence(&scores_from_symbols(&symbols, 3)).unwrap();
        assert_eq!(decoded, None);
    }
}

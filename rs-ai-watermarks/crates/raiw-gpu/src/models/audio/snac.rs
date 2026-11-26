//! SNAC 44.1 kHz multi-scale codec (`hubertsiuzdak/snac_44khz`).
//!
//! Mirrors `_internal/_snac_network.py` + `audio_snac.py`: `reconstruct(x, n_quantizers=3)`
//! with fused weight norm, windowed local attention with rotary embeddings, and the
//! decoder noise blocks seeded exactly like torch's CPU generator.

use std::path::Path;

use burn::tensor::activation::{softmax, tanh};
use burn::tensor::{Device, Int, Tensor, TensorData, s};

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub, HubFile};
use crate::io::{check_mono, mono_tensor, read_finite, read_i64};
use crate::layers::residual::{ResidualUnit, UnitNaming};
use crate::layers::vq::{CodeScore, VectorQuantizer, VqSpec, residual_quantize};
use crate::layers::{Conv1d, Conv1dSpec, ConvTranspose1d, ConvTranspose1dSpec, Snake1d};
use crate::rng::torch_randn;
use crate::weights::StateDict;

pub const MODEL_REPOSITORY: &str = "hubertsiuzdak/snac_44khz";
pub const MODEL_REVISION: &str = "873ebef9718b89660340c6f55a2b515e98cfa1d9";
pub const CHECKPOINT_FILENAME: &str = "pytorch_model.bin";
pub const CHECKPOINT_SHA256: &str = "b0a676cbdc8d1cc53186f6d777bc956fb7932ceacdc657a4c3741646e9e7ead0";
pub const CONFIG_SHA256: &str = "4c98973dd6b4278f6352cedbadc9922a2fa3e4b97619e1769e304e2836c099df";
pub const EXPECTED_STATE_KEY_COUNT: usize = 286;
pub const SAMPLE_RATE: u32 = 44_100;
pub const HOP_LENGTH: usize = 3072;
pub const FIXED_QUANTIZERS: usize = 3;

pub const CHECKPOINT_FILE: HubFile<'static> =
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, CHECKPOINT_FILENAME).with_sha256(CHECKPOINT_SHA256);
pub const CONFIG_FILE: HubFile<'static> =
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, "config.json").with_sha256(CONFIG_SHA256);

const ENCODER_DIM: usize = 64;
const ENCODER_RATES: [usize; 4] = [2, 3, 8, 8];
const DECODER_DIM: usize = 1536;
const DECODER_RATES: [usize; 4] = [8, 8, 3, 2];
const WINDOW: usize = 32;
const HEAD_DIM: usize = 64;
const CODEBOOK_SIZE: usize = 4096;
const CODEBOOK_DIM: usize = 8;
const VQ_STRIDES: [usize; 4] = [8, 4, 2, 1];
const NOISE_SEED: u64 = 0;
const LAYER_NORM_EPS: f64 = 1e-5;

/// Windowed multi-head self-attention with rotary position embeddings, pre-norm, residual.
#[derive(Debug, Clone)]
struct LocalAttention {
    norm_weight: Tensor<1>,
    norm_bias: Tensor<1>,
    to_qkv: Tensor<2>,
    to_out: Tensor<2>,
    cos: Tensor<4>,
    sin: Tensor<4>,
    dim: usize,
    heads: usize,
}

impl LocalAttention {
    fn load(state: &mut StateDict, prefix: &str, dim: usize, device: &Device) -> Result<Self> {
        let norm_weight = state.tensor(&format!("{prefix}.norm.weight"), [dim], device)?;
        let norm_bias = state.tensor(&format!("{prefix}.norm.bias"), [dim], device)?;
        let to_qkv: Tensor<2> = state.tensor(&format!("{prefix}.to_qkv.weight"), [3 * dim, dim], device)?;
        let to_out: Tensor<2> = state.tensor(&format!("{prefix}.to_out.weight"), [dim, dim], device)?;
        let inv_freq = state.take(&format!("{prefix}.rel_pos.inv_freq"), &[HEAD_DIM / 2])?;
        let (cos, sin) = rotary_tables(&inv_freq.data, device);
        Ok(Self {
            norm_weight,
            norm_bias,
            to_qkv: to_qkv.transpose(),
            to_out: to_out.transpose(),
            cos,
            sin,
            dim,
            heads: dim / HEAD_DIM,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, channels, steps] = x.dims();
        let pad = (WINDOW - steps % WINDOW) % WINDOW;
        let normed = self.layer_norm(x.clone().swap_dims(1, 2));
        let padded = if pad > 0 {
            normed.pad([(0, pad), (0, 0)], 0.0)
        } else {
            normed
        };
        let padded_steps = steps + pad;
        let windows = padded_steps / WINDOW;
        let qkv = padded.matmul(self.to_qkv.clone().unsqueeze_dim(0));
        let split = |index: usize| {
            qkv.clone()
                .narrow(2, index * self.dim, self.dim)
                .reshape([batch, windows, WINDOW, self.heads, HEAD_DIM])
                .permute([0, 3, 1, 2, 4])
                .reshape([batch * self.heads, windows, WINDOW, HEAD_DIM])
        };
        let (q, k, v) = (self.rotate(split(0)), self.rotate(split(1)), split(2));
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let weights = softmax(q.matmul(k.swap_dims(2, 3)).mul_scalar(scale), 3);
        let out = weights
            .matmul(v)
            .reshape([batch, self.heads, windows, WINDOW, HEAD_DIM])
            .permute([0, 2, 3, 1, 4])
            .reshape([batch, padded_steps, channels]);
        let out = if pad > 0 {
            out.slice(s![.., 0..steps, ..])
        } else {
            out
        };
        x + out.matmul(self.to_out.clone().unsqueeze_dim(0)).swap_dims(1, 2)
    }

    fn rotate(&self, x: Tensor<4>) -> Tensor<4> {
        let half = HEAD_DIM / 2;
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.clone().narrow(3, half, half);
        let rotated = Tensor::cat(vec![x2.neg(), x1], 3);
        x * self.cos.clone() + rotated * self.sin.clone()
    }

    fn layer_norm(&self, x: Tensor<3>) -> Tensor<3> {
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(2);
        let normed = centered / var.add_scalar(LAYER_NORM_EPS).sqrt();
        normed * self.norm_weight.clone().unsqueeze_dims::<3>(&[0, 1])
            + self.norm_bias.clone().unsqueeze_dims::<3>(&[0, 1])
    }
}

/// `cos`/`sin` of `pos * inv_freq` duplicated across both halves, shaped `[1, 1, WINDOW, HEAD_DIM]`.
fn rotary_tables(inv_freq: &[f32], device: &Device) -> (Tensor<4>, Tensor<4>) {
    let mut cos = Vec::with_capacity(WINDOW * HEAD_DIM);
    let mut sin = Vec::with_capacity(WINDOW * HEAD_DIM);
    for pos in 0..WINDOW {
        let angles: Vec<f32> = inv_freq.iter().map(|&f| pos as f32 * f).collect();
        for &angle in angles.iter().chain(&angles) {
            cos.push(f64::from(angle).cos() as f32);
            sin.push(f64::from(angle).sin() as f32);
        }
    }
    let table =
        |values: Vec<f32>| Tensor::from_data(TensorData::new(values, [1, 1, WINDOW, HEAD_DIM]), device);
    (table(cos), table(sin))
}

#[derive(Debug, Clone)]
struct EncoderBlock {
    units: [ResidualUnit; 3],
    snake: Snake1d,
    conv: Conv1d,
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    snake: Snake1d,
    conv_t: ConvTranspose1d,
    noise: Conv1d,
    units: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn forward(&self, x: Tensor<3>) -> Result<Tensor<3>> {
        let h = self.conv_t.forward(self.snake.forward(x));
        let [batch, _, steps] = h.dims();
        let noise = torch_randn(batch * steps, NOISE_SEED)
            .ok_or_else(|| GpuError::Input("SNAC noise block needs at least 16 frames".to_owned()))?;
        let noise: Tensor<3> = Tensor::from_data(TensorData::new(noise, [batch, 1, steps]), &h.device());
        let h = h.clone() + noise * self.noise.forward(h);
        Ok(self.units.iter().fold(h, |h, unit| unit.forward(h)))
    }
}

#[derive(Debug, Clone)]
pub struct SnacCodec {
    encoder_in: Conv1d,
    encoder_blocks: Vec<EncoderBlock>,
    encoder_attention: LocalAttention,
    encoder_out: Conv1d,
    quantizers: Vec<VectorQuantizer>,
    decoder_in: Conv1d,
    decoder_proj: Conv1d,
    decoder_attention: LocalAttention,
    decoder_blocks: Vec<DecoderBlock>,
    decoder_snake: Snake1d,
    decoder_out: Conv1d,
    device: ComputeDevice,
}

impl SnacCodec {
    /// Load the pinned checkpoint from the hub cache only (Python's `local_files_only=True`).
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::CacheOnly, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        let config = hub.resolve(&CONFIG_FILE, fetch)?;
        let checkpoint = hub.resolve(&CHECKPOINT_FILE, fetch)?;
        if config.parent() != checkpoint.parent() {
            return Err(GpuError::Config(
                "pinned SNAC artifacts must share one snapshot directory".to_owned(),
            ));
        }
        Self::from_checkpoint(&checkpoint, device)
    }

    /// Load from a local `pytorch_model.bin` without digest checks.
    pub fn from_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_pytorch(path, None, &[])?;
        if state.len() != EXPECTED_STATE_KEY_COUNT {
            return Err(GpuError::Config(format!(
                "SNAC checkpoint has {} tensors, expected {EXPECTED_STATE_KEY_COUNT}",
                state.len()
            )));
        }
        state.fold_weight_norm()?;
        let burn = device.burn();
        let encoder_in = Conv1d::load(
            &mut state,
            "encoder.block.0",
            Conv1dSpec::new(1, ENCODER_DIM, 7).padding(3),
            burn,
        )?;
        let mut encoder_blocks = Vec::with_capacity(ENCODER_RATES.len());
        let mut dim = ENCODER_DIM;
        for (index, &stride) in ENCODER_RATES.iter().enumerate() {
            let prefix = format!("encoder.block.{}.block", index + 1);
            let down = Conv1dSpec::new(dim, dim * 2, 2 * stride)
                .stride(stride)
                .padding(stride.div_ceil(2));
            encoder_blocks.push(EncoderBlock {
                units: sequential_units(&mut state, &prefix, 0, dim, burn)?,
                snake: Snake1d::load(&mut state, &format!("{prefix}.3"), dim, burn)?,
                conv: Conv1d::load(&mut state, &format!("{prefix}.4"), down, burn)?,
            });
            dim *= 2;
        }
        let latent = dim;
        let encoder_attention = LocalAttention::load(&mut state, "encoder.block.5", latent, burn)?;
        let depthwise = Conv1dSpec::new(latent, latent, 7).padding(3).groups(latent);
        let encoder_out = Conv1d::load(&mut state, "encoder.block.6", depthwise, burn)?;

        let quantizers = VQ_STRIDES
            .iter()
            .enumerate()
            .map(|(index, &stride)| {
                let spec = VqSpec {
                    input_dim: latent,
                    codebook_size: CODEBOOK_SIZE,
                    codebook_dim: CODEBOOK_DIM,
                    stride,
                    score: CodeScore::Cosine,
                };
                VectorQuantizer::load(&mut state, &format!("quantizer.quantizers.{index}"), spec, burn)
            })
            .collect::<Result<Vec<_>>>()?;

        let decoder_in = Conv1d::load(&mut state, "decoder.model.0", depthwise, burn)?;
        let decoder_proj = Conv1d::load(
            &mut state,
            "decoder.model.1",
            Conv1dSpec::new(latent, DECODER_DIM, 1),
            burn,
        )?;
        let decoder_attention = LocalAttention::load(&mut state, "decoder.model.2", DECODER_DIM, burn)?;
        let mut decoder_blocks = Vec::with_capacity(DECODER_RATES.len());
        let mut dim = DECODER_DIM;
        for (index, &stride) in DECODER_RATES.iter().enumerate() {
            let prefix = format!("decoder.model.{}.block", index + 3);
            let out = dim / 2;
            let up = ConvTranspose1dSpec::new(dim, out, 2 * stride, stride)
                .padding(stride.div_ceil(2))
                .output_padding(stride % 2);
            decoder_blocks.push(DecoderBlock {
                snake: Snake1d::load(&mut state, &format!("{prefix}.0"), dim, burn)?,
                conv_t: ConvTranspose1d::load(&mut state, &format!("{prefix}.1"), up, burn)?,
                noise: Conv1d::load(
                    &mut state,
                    &format!("{prefix}.2.linear"),
                    Conv1dSpec::new(out, out, 1).no_bias(),
                    burn,
                )?,
                units: sequential_units(&mut state, &prefix, 3, out, burn)?,
            });
            dim = out;
        }
        let decoder_snake = Snake1d::load(&mut state, "decoder.model.7", dim, burn)?;
        let decoder_out = Conv1d::load(
            &mut state,
            "decoder.model.8",
            Conv1dSpec::new(dim, 1, 7).padding(3),
            burn,
        )?;
        state.finish()?;
        Ok(Self {
            encoder_in,
            encoder_blocks,
            encoder_attention,
            encoder_out,
            quantizers,
            decoder_in,
            decoder_proj,
            decoder_attention,
            decoder_blocks,
            decoder_snake,
            decoder_out,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// `SNACModel.reconstruct`: pad to the hop, encode, quantize with `n_quantizers`, decode, trim.
    pub fn reconstruct(&self, samples: &[f32], n_quantizers: usize) -> Result<Vec<f32>> {
        let (_, quantized) = self.quantize(samples, n_quantizers)?;
        let decoded = self.decode_latent(quantized)?;
        read_finite(decoded.slice(s![.., .., 0..samples.len()]))
    }

    /// Multi-scale codebook indices (coarsest first) of the hop-padded input.
    pub fn encode(&self, samples: &[f32], n_quantizers: usize) -> Result<Vec<Vec<u32>>> {
        let (codes, _) = self.quantize(samples, n_quantizers)?;
        codes
            .into_iter()
            .map(|code| Ok(read_i64(code)?.into_iter().map(|c| c as u32).collect()))
            .collect()
    }

    fn quantize(&self, samples: &[f32], n_quantizers: usize) -> Result<(Vec<Tensor<2, Int>>, Tensor<3>)> {
        check_mono(samples, "SNAC")?;
        if n_quantizers == 0 || n_quantizers > VQ_STRIDES.len() {
            return Err(GpuError::Input(format!(
                "SNAC n_quantizers must be between 1 and {}",
                VQ_STRIDES.len()
            )));
        }
        let pad = (HOP_LENGTH - samples.len() % HOP_LENGTH) % HOP_LENGTH;
        let mut padded = samples.to_vec();
        padded.resize(samples.len() + pad, 0.0);
        let latent = self.encode_latent(mono_tensor(&padded, self.device.burn()));
        Ok(residual_quantize(&self.quantizers[..n_quantizers], &latent))
    }

    fn encode_latent(&self, x: Tensor<3>) -> Tensor<3> {
        let mut h = self.encoder_in.forward(x);
        for block in &self.encoder_blocks {
            h = block.units.iter().fold(h, |h, unit| unit.forward(h));
            h = block.conv.forward(block.snake.forward(h));
        }
        self.encoder_out.forward(self.encoder_attention.forward(h))
    }

    fn decode_latent(&self, z: Tensor<3>) -> Result<Tensor<3>> {
        let mut h = self
            .decoder_attention
            .forward(self.decoder_proj.forward(self.decoder_in.forward(z)));
        for block in &self.decoder_blocks {
            h = block.forward(h)?;
        }
        Ok(tanh(self.decoder_out.forward(self.decoder_snake.forward(h))))
    }
}

/// Three depthwise residual units (dilations 1, 3, 9) at `<prefix>.<first>..<first + 2>`.
fn sequential_units(
    state: &mut StateDict,
    prefix: &str,
    first: usize,
    dim: usize,
    device: &Device,
) -> Result<[ResidualUnit; 3]> {
    let unit = |state: &mut StateDict, offset: usize, dilation: usize| {
        let name = format!("{prefix}.{}", first + offset);
        ResidualUnit::load(state, &name, UnitNaming::Sequential, dim, dilation, dim, device)
    };
    Ok([unit(state, 0, 1)?, unit(state, 1, 3)?, unit(state, 2, 9)?])
}

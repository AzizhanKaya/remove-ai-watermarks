//! EnCodec 24 kHz (`facebook/encodec_24khz`, Transformers `EncodecModel` layout).
//!
//! Mirrors `_internal/audio_codec.py`: `model(x, bandwidth=...)` on one mono 24 kHz stem —
//! a causal reflect-padded SEANet encoder with LSTM, Euclidean residual quantization with
//! `bandwidth * 4 / 3` codebooks, the mirrored decoder, and trimming to the input length.

use std::path::Path;

use burn::tensor::module::embedding;
use burn::tensor::{Device, Int, Tensor, s};

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub, HubFile};
use crate::io::{check_mono, mono_tensor, read_finite, read_i64};
use crate::layers::{PadKind, SeanetConfig, SeanetDecoder, SeanetEncoder, SeanetLayout};
use crate::weights::StateDict;

pub const MODEL_REPOSITORY: &str = "facebook/encodec_24khz";
pub const MODEL_REVISION: &str = "c1dbe2ae3f1de713481a3b3e7c47f357092ee040";
pub const CHECKPOINT_FILENAME: &str = "model.safetensors";
pub const CHECKPOINT_SHA256: &str = "37a7cb100f71a29e6c1d815aca8666a1d7ea8885ebe44c306c751a5103559d57";
pub const CONFIG_SHA256: &str = "cd7da7da3b3b34daee262e213e3daa24dc01b4fdc3aadd38ed760f0d6f955871";
pub const SAMPLE_RATE: u32 = 24_000;
pub const HOP_LENGTH: usize = 320;
pub const TARGET_BANDWIDTHS: [f64; 5] = [1.5, 3.0, 6.0, 12.0, 24.0];

pub const CHECKPOINT_FILE: HubFile<'static> =
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, CHECKPOINT_FILENAME).with_sha256(CHECKPOINT_SHA256);
pub const CONFIG_FILE: HubFile<'static> =
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, "config.json").with_sha256(CONFIG_SHA256);

const CODEBOOKS: usize = 32;
const CODEBOOK_SIZE: usize = 1024;
const HIDDEN: usize = 128;
const FRAME_RATE: f64 = 75.0;

/// `encodec_24khz` SEANet hyper-parameters.
pub fn seanet_config() -> SeanetConfig {
    SeanetConfig {
        channels: 1,
        dimension: HIDDEN,
        n_filters: 32,
        n_residual_layers: 1,
        ratios: vec![8, 5, 4, 2],
        kernel_size: 7,
        last_kernel_size: 7,
        residual_kernel_size: 3,
        dilation_base: 2,
        compress: 2,
        lstm_layers: 2,
        causal: true,
        elu_alpha: 1.0,
        pad: PadKind::Reflect,
        true_skip: false,
        layout: SeanetLayout::Transformers,
    }
}

/// Codebooks used at a target bandwidth (`get_num_quantizers_for_bandwidth`).
pub fn num_quantizers(bandwidth_kbps: f64) -> Result<usize> {
    if !TARGET_BANDWIDTHS.contains(&bandwidth_kbps) {
        return Err(GpuError::Input(format!(
            "EnCodec bandwidth must be one of {TARGET_BANDWIDTHS:?} kbps"
        )));
    }
    let per_quantizer = (CODEBOOK_SIZE as f64).log2() * FRAME_RATE;
    Ok(((bandwidth_kbps * 1000.0 / per_quantizer).floor() as usize).max(1))
}

/// `EncodecEuclideanCodebook`: nearest code by squared Euclidean distance.
#[derive(Debug, Clone)]
struct Codebook {
    embed: Tensor<2>,
    embed_t: Tensor<2>,
    embed_sq: Tensor<2>,
}

impl Codebook {
    fn load(state: &mut StateDict, prefix: &str, device: &Device) -> Result<Self> {
        let embed: Tensor<2> = state.tensor(&format!("{prefix}.embed"), [CODEBOOK_SIZE, HIDDEN], device)?;
        for buffer in ["cluster_size", "embed_avg", "inited"] {
            state.take_any(&format!("{prefix}.{buffer}"))?;
        }
        let embed_t = embed.clone().transpose();
        let embed_sq = embed.clone().powf_scalar(2.0).sum_dim(1).transpose();
        Ok(Self {
            embed,
            embed_t,
            embed_sq,
        })
    }

    /// Codes of `[batch, hidden, frames]` as `[batch, frames]`.
    fn encode(&self, z: Tensor<3>) -> Tensor<2, Int> {
        let [batch, hidden, frames] = z.dims();
        let flat = z.swap_dims(1, 2).reshape([batch * frames, hidden]);
        let x_sq = flat.clone().powf_scalar(2.0).sum_dim(1);
        let dist = ((x_sq - flat.matmul(self.embed_t.clone()).mul_scalar(2.0)) + self.embed_sq.clone()).neg();
        dist.argmax(1).reshape([batch, frames])
    }

    fn decode(&self, codes: Tensor<2, Int>) -> Tensor<3> {
        embedding(self.embed.clone(), codes).swap_dims(1, 2)
    }
}

#[derive(Debug, Clone)]
pub struct EncodecCodec {
    encoder: SeanetEncoder,
    codebooks: Vec<Codebook>,
    decoder: SeanetDecoder,
    device: ComputeDevice,
}

impl EncodecCodec {
    /// Resolve the pinned checkpoint through the hub cache (downloading if allowed) and load it.
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        let config = hub.resolve(&CONFIG_FILE, fetch)?;
        let checkpoint = hub.resolve(&CHECKPOINT_FILE, fetch)?;
        if config.parent() != checkpoint.parent() {
            return Err(GpuError::Config(
                "pinned EnCodec artifacts must share one snapshot directory".to_owned(),
            ));
        }
        Self::from_checkpoint(&checkpoint, device)
    }

    /// Load from a local `model.safetensors` without digest checks.
    pub fn from_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_safetensors(path, &[])?;
        state.fold_weight_norm()?;
        let burn = device.burn();
        let config = seanet_config();
        let encoder = SeanetEncoder::load(&mut state, "encoder", &config, burn)?;
        let decoder = SeanetDecoder::load(&mut state, "decoder", &config, burn)?;
        let codebooks = (0..CODEBOOKS)
            .map(|i| Codebook::load(&mut state, &format!("quantizer.layers.{i}.codebook"), burn))
            .collect::<Result<Vec<_>>>()?;
        state.finish()?;
        Ok(Self {
            encoder,
            codebooks,
            decoder,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// Residual codes per codebook for mono 24 kHz samples at `bandwidth_kbps`.
    pub fn encode(&self, samples: &[f32], bandwidth_kbps: f64) -> Result<Vec<Vec<u32>>> {
        let (codes, _) = self.quantize(samples, bandwidth_kbps)?;
        codes
            .into_iter()
            .map(|code| Ok(read_i64(code)?.into_iter().map(|c| c as u32).collect()))
            .collect()
    }

    /// `model(x, bandwidth=...).audio_values`, trimmed to the input length.
    pub fn reconstruct(&self, samples: &[f32], bandwidth_kbps: f64) -> Result<Vec<f32>> {
        let (_, quantized) = self.quantize(samples, bandwidth_kbps)?;
        let decoded = self.decoder.forward(quantized);
        read_finite(decoded.slice(s![.., .., 0..samples.len()]))
    }

    fn quantize(&self, samples: &[f32], bandwidth_kbps: f64) -> Result<(Vec<Tensor<2, Int>>, Tensor<3>)> {
        check_mono(samples, "EnCodec")?;
        let active = num_quantizers(bandwidth_kbps)?;
        let latent = self.encoder.forward(mono_tensor(samples, self.device.burn()));
        let mut residual = latent.clone();
        let mut quantized = latent.zeros_like();
        let mut codes = Vec::with_capacity(active);
        for codebook in &self.codebooks[..active] {
            let code = codebook.encode(residual.clone());
            let part = codebook.decode(code.clone());
            residual = residual - part.clone();
            quantized = quantized + part;
            codes.push(code);
        }
        Ok((codes, quantized))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandwidth_selects_codebooks() {
        assert_eq!(num_quantizers(1.5).unwrap(), 2);
        assert_eq!(num_quantizers(6.0).unwrap(), 8);
        assert_eq!(num_quantizers(24.0).unwrap(), 32);
        assert!(num_quantizers(7.0).is_err());
    }
}

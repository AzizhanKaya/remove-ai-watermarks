//! `WanVideoToVideoPipeline.__call__` for one window, as `video_wan._run_wan_window` drives it:
//! encode with `latent_dist.mode()`, normalize by the VAE latent statistics, noise to the
//! strength's first timestep with `torch.randn` from a CPU generator seeded per window, run
//! the UniPC tail with CFG off (`guidance_scale=1.0`), un-normalize and decode.
//!
//! Spatial padding to the model's multiple (16 for Wan 2.1, 32 for Wan 2.2) and the
//! guide-size resize stay with the caller; the temporal `1 + 4k` padding (repeat the last
//! frame, trim the output) happens here.
//!
//! On an 8 GB card Wan 2.2's int8 transformer and its VAE take turns
//! ([`BlockResidency::Swapped`]): the VAE reloads from its checkpoint for each encode and
//! decode, and the transformer blocks upload from host memory for each denoising pass.

use std::path::PathBuf;

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub};
use crate::rng::torch_randn;

use super::config::VaeConfig;
use super::prompt::PromptEmbedding;
use super::scheduler::UniPcScheduler;
use super::transformer::{
    BlockResidency, TextContext, TransformerFiles, WAN21_TRANSFORMER, WAN22_TRANSFORMER, WanTransformer,
};
use super::vae3d::{VaeFiles, VaeTiles, WAN21_VAE, WAN22_VAE, WanVae};
use super::{FLOW_SHIFT, TEMPORAL_FACTOR, WAN22_FLOW_SHIFT};

/// One supported checkpoint: its pinned VAE and transformer and its scheduler shift.
#[derive(Debug, Clone, Copy)]
pub struct WanModel {
    pub vae: VaeFiles,
    pub transformer: TransformerFiles,
    pub flow_shift: f64,
    /// Too large for f16 blocks next to the VAE on 8 GB: int8 blocks swapped with the VAE.
    pub swapped: bool,
    /// VAE tiling: diffusers' default tiles, as `video_wan.py` enables them for Wan 2.1 (Wan
    /// 2.2's causal caches alone take ~3 GB untiled at 480p).
    pub vae_tiles: Option<VaeTiles>,
    /// Encode tiles, when they differ from [`Self::vae_tiles`].
    pub vae_encode_tiles: Option<VaeTiles>,
}

impl WanModel {
    /// The fastest residency that fits `device`.
    pub fn residency_for(&self, device: &ComputeDevice) -> BlockResidency {
        #[cfg(feature = "gpu-runtime")]
        if self.swapped && device.is_gpu() {
            return BlockResidency::Swapped(super::Precision::Int8);
        }
        BlockResidency::for_device(device)
    }
}

/// `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` with `video_wan.py`'s `flow_shift=3.0`.
pub const WAN21_T2V_1_3B: WanModel = WanModel {
    vae: WAN21_VAE,
    transformer: WAN21_TRANSFORMER,
    flow_shift: FLOW_SHIFT,
    swapped: false,
    vae_tiles: Some(VaeTiles::DIFFUSERS),
    vae_encode_tiles: Some(VaeTiles::DIFFUSERS),
};

/// `Wan-AI/Wan2.2-TI2V-5B-Diffusers` with its scheduler's default shift (Rust-only).
pub const WAN22_TI2V_5B: WanModel = WanModel {
    vae: WAN22_VAE,
    transformer: WAN22_TRANSFORMER,
    flow_shift: WAN22_FLOW_SHIFT,
    swapped: true,
    vae_tiles: Some(VaeTiles::DIFFUSERS),
    vae_encode_tiles: Some(VaeTiles::WAN22_ENCODE),
};

impl WanModel {
    /// `vae` with this model's tiling.
    pub fn tile(&self, vae: WanVae) -> WanVae {
        vae.with_tiles(self.vae_tiles)
            .with_encode_tiles(self.vae_encode_tiles)
    }
}

/// The VAE on the device for good, or reloaded from its checkpoint for each encode and decode.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant, reason = "one per pipeline")]
enum VaeSlot {
    Resident(WanVae),
    Reloaded {
        config: VaeConfig,
        checkpoint: PathBuf,
        model: WanModel,
        device: ComputeDevice,
    },
}

impl VaeSlot {
    fn config(&self) -> &VaeConfig {
        match self {
            Self::Resident(vae) => vae.config(),
            Self::Reloaded { config, .. } => config,
        }
    }

    /// Run `stage` on the VAE, freeing a reloaded one's device memory afterwards.
    fn with<R>(&self, stage: impl FnOnce(&WanVae) -> Result<R>) -> Result<R> {
        match self {
            Self::Resident(vae) => stage(vae),
            Self::Reloaded {
                config,
                checkpoint,
                model,
                device,
            } => {
                let vae = model.tile(WanVae::from_checkpoint(config.clone(), checkpoint, device)?);
                let out = stage(&vae);
                drop(vae);
                device.burn().memory_cleanup();
                out
            }
        }
    }
}

/// The VAE, the transformer and the projected default prompt, ready to regenerate windows.
#[derive(Debug, Clone)]
pub struct WanPipeline {
    vae: VaeSlot,
    transformer: WanTransformer,
    context: TextContext,
    flow_shift: f64,
}

impl WanPipeline {
    /// Wan 2.1 (flow shift 3) from loaded models.
    pub fn new(vae: WanVae, transformer: WanTransformer, prompt: &PromptEmbedding) -> Self {
        Self::assemble(VaeSlot::Resident(vae), transformer, prompt, FLOW_SHIFT)
    }

    fn assemble(
        vae: VaeSlot,
        transformer: WanTransformer,
        prompt: &PromptEmbedding,
        flow_shift: f64,
    ) -> Self {
        let context = transformer.encode_text(prompt);
        Self {
            vae,
            transformer,
            context,
            flow_shift,
        }
    }

    /// Load the pinned Wan 2.1 VAE and transformer plus the Python-produced default-prompt cache.
    pub fn load(
        hub: &Hub,
        fetch: Fetch,
        device: &ComputeDevice,
        residency: BlockResidency,
        prompt_sha256: Option<&str>,
    ) -> Result<Self> {
        Self::load_model(hub, fetch, &WAN21_T2V_1_3B, device, residency, prompt_sha256)
    }

    /// Load a pinned `model` with the default-prompt cache (Wan 2.2 shares Wan 2.1's UMT5-XXL
    /// text encoder, so the cache serves both). A swapping `residency` keeps the VAE off the
    /// device between its stages too.
    pub fn load_model(
        hub: &Hub,
        fetch: Fetch,
        model: &WanModel,
        device: &ComputeDevice,
        residency: BlockResidency,
        prompt_sha256: Option<&str>,
    ) -> Result<Self> {
        let prompt = PromptEmbedding::load_cached(prompt_sha256)?;
        Self::from_files(hub, fetch, model, device, residency, &prompt)
    }

    /// [`Self::load_model`] with an explicit prompt embedding.
    pub fn from_files(
        hub: &Hub,
        fetch: Fetch,
        model: &WanModel,
        device: &ComputeDevice,
        residency: BlockResidency,
        prompt: &PromptEmbedding,
    ) -> Result<Self> {
        let vae = if residency.swaps() {
            let (config, checkpoint) = WanVae::resolve_files(hub, fetch, &model.vae)?;
            VaeSlot::Reloaded {
                config,
                checkpoint,
                model: *model,
                device: device.clone(),
            }
        } else {
            VaeSlot::Resident(model.tile(WanVae::load_files(hub, fetch, &model.vae, device)?))
        };
        let transformer = WanTransformer::load_files(hub, fetch, &model.transformer, device, residency)?;
        Ok(Self::assemble(vae, transformer, prompt, model.flow_shift))
    }

    /// The resident VAE (`None` when it reloads per stage).
    pub fn vae(&self) -> Option<&WanVae> {
        match &self.vae {
            VaeSlot::Resident(vae) => Some(vae),
            VaeSlot::Reloaded { .. } => None,
        }
    }

    pub fn transformer(&self) -> &WanTransformer {
        &self.transformer
    }

    /// Pixels per latent along each side (8 for Wan 2.1, 16 for Wan 2.2).
    pub fn spatial_factor(&self) -> usize {
        self.vae.config().spatial_factor()
    }

    /// The side multiple windows need: one latent patch.
    pub fn spatial_multiple(&self) -> usize {
        self.spatial_factor() * self.transformer.config().patch()
    }

    /// Normalized latents `(mode - mean) / std`, `[z, 1 + k, H / f, W / f]`, of `1 + 4k` frames.
    pub fn encode(&self, frames: &[f32], (count, height, width): (usize, usize, usize)) -> Result<Vec<f32>> {
        let mut latents = self.vae.with(|vae| vae.encode(frames, count, height, width))?;
        let config = self.vae.config();
        let plane = latents.len() / config.z_dim;
        for ((channel, mean), std) in latents
            .chunks_exact_mut(plane)
            .zip(&config.latents_mean)
            .zip(&config.latents_std)
        {
            let inv_std = 1.0 / std;
            for value in channel {
                *value = (*value - mean) * inv_std;
            }
        }
        Ok(latents)
    }

    /// Frames `[1 + 4(T - 1), 3, f·h, f·w]` in `[-1, 1]` of normalized latents `[z, T, h, w]`.
    pub fn decode(
        &self,
        latents: &[f32],
        (frames, height, width): (usize, usize, usize),
    ) -> Result<Vec<f32>> {
        let plane = frames * height * width;
        let channels = self.vae.config().z_dim;
        if plane == 0 || latents.len() != channels * plane {
            return Err(GpuError::Input(format!(
                "Wan latents hold {} values, expected {channels}x{frames}x{height}x{width}",
                latents.len()
            )));
        }
        let mut raw = latents.to_vec();
        let config = self.vae.config();
        for ((channel, mean), std) in raw
            .chunks_exact_mut(plane)
            .zip(&config.latents_mean)
            .zip(&config.latents_std)
        {
            let inv_std = 1.0 / std;
            for value in channel {
                *value = *value / inv_std + mean;
            }
        }
        self.vae.with(|vae| vae.decode(&raw, frames, height, width))
    }

    /// Noise normalized latents to the first timestep `strength` keeps, then denoise them
    /// (uploading swapped transformer blocks for the pass).
    pub fn denoise(
        &mut self,
        latents: &[f32],
        shape: (usize, usize, usize),
        strength: f64,
        steps: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let mut scheduler = UniPcScheduler::new(steps, self.flow_shift)?;
        let timesteps = scheduler.strength_timesteps(strength).to_vec();
        let Some(&first) = timesteps.first() else {
            return Err(GpuError::Input(format!(
                "strength {strength} keeps no denoising step out of {steps}"
            )));
        };
        let noise = torch_randn(latents.len(), seed)
            .ok_or_else(|| GpuError::Input("Wan latents are too small to noise".to_owned()))?;
        let sample = scheduler.add_noise(latents, &noise, first)?;
        self.transformer.unpark();
        let denoised = self.run_steps(&mut scheduler, &timesteps, sample, shape);
        self.transformer.park();
        denoised
    }

    fn run_steps(
        &self,
        scheduler: &mut UniPcScheduler,
        timesteps: &[i64],
        mut sample: Vec<f32>,
        shape: (usize, usize, usize),
    ) -> Result<Vec<f32>> {
        for &t in timesteps {
            let velocity = self
                .transformer
                .forward(&sample, shape, t as f32, &self.context)?;
            sample = scheduler.step(&velocity, t, &sample)?;
        }
        Ok(sample)
    }

    /// Regenerate one window of RGB frames `[T, 3, H, W]` in `[-1, 1]` (`H`, `W` multiples of
    /// [`Self::spatial_multiple`]).
    ///
    /// Mirrors `WanVideoToVideoPipeline(video, strength, num_inference_steps=steps,
    /// guidance_scale=1.0, generator=torch.Generator("cpu").manual_seed(seed))`: returns the
    /// same number of frames, each in `[-1, 1]`.
    pub fn regenerate_chunk(
        &mut self,
        frames: &[f32],
        (count, height, width): (usize, usize, usize),
        strength: f64,
        steps: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let multiple = self.spatial_multiple();
        if count == 0 || !height.is_multiple_of(multiple) || !width.is_multiple_of(multiple) {
            return Err(GpuError::Input(format!(
                "Wan windows need frames and sides divisible by {multiple}, got {count}x{height}x{width}"
            )));
        }
        let frame = 3 * height * width;
        if frames.len() != count * frame {
            return Err(GpuError::Input(format!(
                "Wan window holds {} values, expected {count}x3x{height}x{width}",
                frames.len()
            )));
        }
        let padded = pad_to_causal_length(frames, count, frame);
        let padded_count = padded.len() / frame;
        let latents = self.encode(&padded, (padded_count, height, width))?;
        let factor = self.spatial_factor();
        let shape = (
            1 + (padded_count - 1) / TEMPORAL_FACTOR,
            height / factor,
            width / factor,
        );
        let denoised = self.denoise(&latents, shape, strength, steps, seed)?;
        let mut decoded = self.decode(&denoised, shape)?;
        decoded.truncate(count * frame);
        Ok(decoded)
    }
}

/// Repeat the last frame until the count is `1 + 4k` (`_run_wan_window`'s `pad_t`).
fn pad_to_causal_length(frames: &[f32], count: usize, frame: usize) -> Vec<f32> {
    let pad = (TEMPORAL_FACTOR - (count - 1) % TEMPORAL_FACTOR) % TEMPORAL_FACTOR;
    let mut padded = frames.to_vec();
    let last = &frames[(count - 1) * frame..];
    for _ in 0..pad {
        padded.extend_from_slice(last);
    }
    padded
}

/// `uint8` BGR frames `[T, H, W, 3]` to RGB `[T, 3, H, W]` as diffusers preprocesses PIL frames:
/// `2 * (x / 255) - 1`.
pub fn bgr_u8_to_unit(frames: &[u8], height: usize, width: usize) -> Vec<f32> {
    bgr_to_unit(frames, height, width, 255.0)
}

/// High-depth `uint16` BGR frames to RGB `[T, 3, H, W]`: `2 * (x / 65535) - 1`.
pub fn bgr_u16_to_unit(frames: &[u16], height: usize, width: usize) -> Vec<f32> {
    bgr_to_unit(frames, height, width, 65535.0)
}

/// Decoded RGB `[T, 3, H, W]` to `uint8` BGR: `rint(clamp(x * 0.5 + 0.5, 0, 1) * 255)`.
pub fn unit_to_bgr_u8(samples: &[f32], height: usize, width: usize) -> Vec<u8> {
    unit_to_bgr(samples, height, width, 255.0)
        .into_iter()
        .map(|v| v as u8)
        .collect()
}

/// Decoded RGB `[T, 3, H, W]` to `uint16` BGR: `rint(clamp(x * 0.5 + 0.5, 0, 1) * 65535)`.
pub fn unit_to_bgr_u16(samples: &[f32], height: usize, width: usize) -> Vec<u16> {
    unit_to_bgr(samples, height, width, 65535.0)
        .into_iter()
        .map(|v| v as u16)
        .collect()
}

fn bgr_to_unit<T: Copy + Into<f32>>(frames: &[T], height: usize, width: usize, peak: f32) -> Vec<f32> {
    let pixels = height * width;
    let mut out = vec![0.0f32; frames.len()];
    for (frame, dst) in frames
        .chunks_exact(3 * pixels)
        .zip(out.chunks_exact_mut(3 * pixels))
    {
        for (p, bgr) in frame.chunks_exact(3).enumerate() {
            for (c, &value) in bgr.iter().rev().enumerate() {
                dst[c * pixels + p] = 2.0 * (value.into() / peak) - 1.0;
            }
        }
    }
    out
}

fn unit_to_bgr(samples: &[f32], height: usize, width: usize, peak: f32) -> Vec<f32> {
    let pixels = height * width;
    let mut out = vec![0.0f32; samples.len()];
    for (frame, dst) in samples
        .chunks_exact(3 * pixels)
        .zip(out.chunks_exact_mut(3 * pixels))
    {
        for p in 0..pixels {
            for c in 0..3 {
                let unit = (frame[c * pixels + p] * 0.5 + 0.5).clamp(0.0, 1.0);
                dst[3 * p + (2 - c)] = (unit * peak).round_ties_even();
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_padding_repeats_the_last_frame() {
        let frames: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let padded = pad_to_causal_length(&frames, 3, 2);
        assert_eq!(padded, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
        assert_eq!(pad_to_causal_length(&frames[..2], 1, 2).len(), 2);
    }

    #[test]
    fn pixel_conversions_round_trip() {
        let bgr = [0u8, 128, 255, 7, 64, 200];
        let unit = bgr_u8_to_unit(&bgr, 1, 2);
        assert_eq!(unit[0], 2.0 * (255.0f32 / 255.0) - 1.0);
        assert_eq!(unit_to_bgr_u8(&unit, 1, 2), bgr);
        let deep = [0u16, 40_000, 65_535];
        assert_eq!(unit_to_bgr_u16(&bgr_u16_to_unit(&deep, 1, 1), 1, 1), deep);
    }
}

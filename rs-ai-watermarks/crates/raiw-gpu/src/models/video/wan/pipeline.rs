//! `WanVideoToVideoPipeline.__call__` for one window, as `video_wan._run_wan_window` drives it:
//! encode with `latent_dist.mode()`, normalize by the VAE latent statistics, noise to the
//! strength's first timestep with `torch.randn` from a CPU generator seeded per window, run
//! the UniPC tail with CFG off (`guidance_scale=1.0`), un-normalize and decode.
//!
//! Spatial padding to multiples of 16 and the guide-size resize stay with the caller;
//! the temporal `1 + 4k` padding (repeat the last frame, trim the output) happens here.

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub};
use crate::rng::torch_randn;

use super::prompt::PromptEmbedding;
use super::scheduler::UniPcScheduler;
use super::transformer::{BlockResidency, TextContext, WanTransformer};
use super::vae3d::{LATENTS_MEAN, LATENTS_STD, WanVae};
use super::{FLOW_SHIFT, LATENT_CHANNELS, SPATIAL_FACTOR, SPATIAL_MULTIPLE, TEMPORAL_FACTOR};

/// The VAE, the transformer and the projected default prompt, ready to regenerate windows.
#[derive(Debug, Clone)]
pub struct WanPipeline {
    vae: WanVae,
    transformer: WanTransformer,
    context: TextContext,
}

impl WanPipeline {
    pub fn new(vae: WanVae, transformer: WanTransformer, prompt: &PromptEmbedding) -> Self {
        let context = transformer.encode_text(prompt);
        Self {
            vae,
            transformer,
            context,
        }
    }

    /// Load the pinned VAE and transformer plus the Python-produced default-prompt cache.
    pub fn load(
        hub: &Hub,
        fetch: Fetch,
        device: &ComputeDevice,
        residency: BlockResidency,
        prompt_sha256: Option<&str>,
    ) -> Result<Self> {
        let prompt = PromptEmbedding::load_cached(prompt_sha256)?;
        let vae = WanVae::load_from_hub(hub, fetch, device)?;
        let transformer = WanTransformer::load_from_hub(hub, fetch, device, residency)?;
        Ok(Self::new(vae, transformer, &prompt))
    }

    pub fn vae(&self) -> &WanVae {
        &self.vae
    }

    pub fn transformer(&self) -> &WanTransformer {
        &self.transformer
    }

    /// Normalized latents `(mode - mean) / std`, `[16, 1 + k, H / 8, W / 8]`, of `1 + 4k` frames.
    pub fn encode(&self, frames: &[f32], (count, height, width): (usize, usize, usize)) -> Result<Vec<f32>> {
        let mut latents = self.vae.encode(frames, count, height, width)?;
        let plane = latents.len() / LATENT_CHANNELS;
        for (c, channel) in latents.chunks_exact_mut(plane).enumerate() {
            let inv_std = 1.0 / LATENTS_STD[c];
            for value in channel {
                *value = (*value - LATENTS_MEAN[c]) * inv_std;
            }
        }
        Ok(latents)
    }

    /// Frames `[1 + 4(T - 1), 3, 8h, 8w]` in `[-1, 1]` of normalized latents `[16, T, h, w]`.
    pub fn decode(
        &self,
        latents: &[f32],
        (frames, height, width): (usize, usize, usize),
    ) -> Result<Vec<f32>> {
        let plane = frames * height * width;
        if plane == 0 || latents.len() != LATENT_CHANNELS * plane {
            return Err(GpuError::Input(format!(
                "Wan latents hold {} values, expected {LATENT_CHANNELS}x{frames}x{height}x{width}",
                latents.len()
            )));
        }
        let mut raw = latents.to_vec();
        for (c, channel) in raw.chunks_exact_mut(plane).enumerate() {
            let inv_std = 1.0 / LATENTS_STD[c];
            for value in channel {
                *value = *value / inv_std + LATENTS_MEAN[c];
            }
        }
        self.vae.decode(&raw, frames, height, width)
    }

    /// Noise normalized latents to the first timestep `strength` keeps, then denoise them.
    pub fn denoise(
        &self,
        latents: &[f32],
        shape: (usize, usize, usize),
        strength: f64,
        steps: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let mut scheduler = UniPcScheduler::new(steps, FLOW_SHIFT)?;
        let timesteps = scheduler.strength_timesteps(strength).to_vec();
        let Some(&first) = timesteps.first() else {
            return Err(GpuError::Input(format!(
                "strength {strength} keeps no denoising step out of {steps}"
            )));
        };
        let noise = torch_randn(latents.len(), seed)
            .ok_or_else(|| GpuError::Input("Wan latents are too small to noise".to_owned()))?;
        let mut sample = scheduler.add_noise(latents, &noise, first)?;
        for t in timesteps {
            let velocity = self
                .transformer
                .forward(&sample, shape, t as f32, &self.context)?;
            sample = scheduler.step(&velocity, t, &sample)?;
        }
        Ok(sample)
    }

    /// Regenerate one window of RGB frames `[T, 3, H, W]` in `[-1, 1]` (`H`, `W` multiples of 16).
    ///
    /// Mirrors `WanVideoToVideoPipeline(video, strength, num_inference_steps=steps,
    /// guidance_scale=1.0, generator=torch.Generator("cpu").manual_seed(seed))`: returns the
    /// same number of frames, each in `[-1, 1]`.
    pub fn regenerate_chunk(
        &self,
        frames: &[f32],
        (count, height, width): (usize, usize, usize),
        strength: f64,
        steps: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        if count == 0 || !height.is_multiple_of(SPATIAL_MULTIPLE) || !width.is_multiple_of(SPATIAL_MULTIPLE) {
            return Err(GpuError::Input(format!(
                "Wan windows need frames and sides divisible by {SPATIAL_MULTIPLE}, got {count}x{height}x{width}"
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
        let shape = (
            1 + (padded_count - 1) / TEMPORAL_FACTOR,
            height / SPATIAL_FACTOR,
            width / SPATIAL_FACTOR,
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

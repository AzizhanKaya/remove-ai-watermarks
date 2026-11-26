//! Stable Diffusion `AutoencoderKL` (`stabilityai/sd-vae-ft-mse`), the 2-D video VAE behind
//! `video invisible --profile synthid`, the Seedance profile and the `video message` regeneration.
//!
//! Mirrors `video_invisible._encode_frame_latents` / `_decode_frame_latents`: frames are
//! encoded one at a time (`enable_slicing`), latents are `latent_dist.mode() * 0.18215`, and
//! decoding divides by the same factor. Every heavy convolution runs as a chunked im2col matmul
//! on GPUs, so no single dispatch grows with the frame size.

use std::path::Path;

use burn::tensor::activation::{silu, softmax};
use burn::tensor::{Device, Tensor, TensorData};

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub, HubFile};
use crate::io::read_finite;
use crate::layers::norm::GroupNorm;
use crate::layers::{Conv2d, Conv2dSpec};
use crate::models::video::tiling::{Plane, Tiling, run_tiled};
use crate::weights::StateDict;

pub const MODEL_REPOSITORY: &str = "stabilityai/sd-vae-ft-mse";
pub const MODEL_REVISION: &str = "31f26fdeee1355a5c34592e401dd41e45d25a493";
pub const CHECKPOINT_SHA256: &str = "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815";
pub const CONFIG_SHA256: &str = "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e";
/// The certified SynthID profile is defined against this latent scaling.
pub const SCALING_FACTOR: f32 = 0.18215;
/// Spatial downsampling between pixels and latents.
pub const LATENT_MULTIPLE: usize = 8;
pub const LATENT_CHANNELS: usize = 4;

pub const CHECKPOINT_FILE: HubFile<'static> = HubFile::new(
    MODEL_REPOSITORY,
    MODEL_REVISION,
    "diffusion_pytorch_model.safetensors",
)
.with_sha256(CHECKPOINT_SHA256);
pub const CONFIG_FILE: HubFile<'static> =
    HubFile::new(MODEL_REPOSITORY, MODEL_REVISION, "config.json").with_sha256(CONFIG_SHA256);

const CHANNELS: [usize; 4] = [128, 256, 512, 512];
/// `tile_sample_min_size` (config `sample_size`) and `tile_latent_min_size`, overlap 0.25.
const TILE_SAMPLE: usize = 256;
const TILE_LATENT: usize = TILE_SAMPLE / LATENT_MULTIPLE;
const ENCODE_TILING: Tiling = Tiling {
    tile: TILE_SAMPLE,
    stride: TILE_SAMPLE * 3 / 4,
    blend: TILE_LATENT / 4,
    limit: TILE_LATENT * 3 / 4,
};
const DECODE_TILING: Tiling = Tiling {
    tile: TILE_LATENT,
    stride: TILE_LATENT * 3 / 4,
    blend: TILE_SAMPLE / 4,
    limit: TILE_SAMPLE * 3 / 4,
};
const GROUPS: usize = 32;
const EPS: f64 = 1e-6;

/// `ResnetBlock2D`: two GroupNorm-SiLU-conv stages plus a 1x1 shortcut when widths differ.
#[derive(Debug, Clone)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn load(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<Self> {
        let shortcut = if input == output {
            None
        } else {
            Some(Conv2d::load(
                state,
                &format!("{prefix}.conv_shortcut"),
                Conv2dSpec::new(input, output, 1),
                device,
            )?)
        };
        Ok(Self {
            norm1: GroupNorm::load(state, &format!("{prefix}.norm1"), GROUPS, input, EPS, device)?,
            conv1: Conv2d::load(
                state,
                &format!("{prefix}.conv1"),
                Conv2dSpec::new(input, output, 3).padding(1),
                device,
            )?,
            norm2: GroupNorm::load(state, &format!("{prefix}.norm2"), GROUPS, output, EPS, device)?,
            conv2: Conv2d::load(
                state,
                &format!("{prefix}.conv2"),
                Conv2dSpec::new(output, output, 3).padding(1),
                device,
            )?,
            shortcut,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x.clone())));
        let h = self.conv2.forward(silu(self.norm2.forward(h)));
        match &self.shortcut {
            Some(shortcut) => shortcut.forward(x) + h,
            None => x + h,
        }
    }
}

/// Single-head spatial self-attention of the mid block (legacy `query/key/value/proj_attn`).
#[derive(Debug, Clone)]
struct Attention {
    norm: GroupNorm,
    query: (Tensor<2>, Tensor<1>),
    key: (Tensor<2>, Tensor<1>),
    value: (Tensor<2>, Tensor<1>),
    proj: (Tensor<2>, Tensor<1>),
    channels: usize,
}

impl Attention {
    fn load(state: &mut StateDict, prefix: &str, channels: usize, device: &Device) -> Result<Self> {
        let mut linear = |name: &str| -> Result<(Tensor<2>, Tensor<1>)> {
            let weight: Tensor<2> =
                state.tensor(&format!("{prefix}.{name}.weight"), [channels, channels], device)?;
            Ok((
                weight.transpose(),
                state.tensor(&format!("{prefix}.{name}.bias"), [channels], device)?,
            ))
        };
        let (query, key, value, proj) = (
            linear("query")?,
            linear("key")?,
            linear("value")?,
            linear("proj_attn")?,
        );
        Ok(Self {
            norm: GroupNorm::load(
                state,
                &format!("{prefix}.group_norm"),
                GROUPS,
                channels,
                EPS,
                device,
            )?,
            query,
            key,
            value,
            proj,
            channels,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [n, c, h, w] = x.dims();
        let tokens = self
            .norm
            .forward(x.clone())
            .reshape([n, c, h * w])
            .swap_dims(1, 2);
        let project = |(weight, bias): &(Tensor<2>, Tensor<1>), t: Tensor<3>| {
            t.matmul(weight.clone().unsqueeze_dim(0)) + bias.clone().unsqueeze_dims::<3>(&[0, 1])
        };
        let q = project(&self.query, tokens.clone());
        let k = project(&self.key, tokens.clone());
        let v = project(&self.value, tokens);
        let scale = 1.0 / (self.channels as f64).sqrt();
        let weights = softmax(q.matmul(k.swap_dims(1, 2)).mul_scalar(scale), 2);
        let out = project(&self.proj, weights.matmul(v));
        x + out.swap_dims(1, 2).reshape([n, c, h, w])
    }
}

#[derive(Debug, Clone)]
struct MidBlock {
    first: ResnetBlock,
    attention: Attention,
    second: ResnetBlock,
}

impl MidBlock {
    fn load(state: &mut StateDict, prefix: &str, channels: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            first: ResnetBlock::load(state, &format!("{prefix}.resnets.0"), channels, channels, device)?,
            attention: Attention::load(state, &format!("{prefix}.attentions.0"), channels, device)?,
            second: ResnetBlock::load(state, &format!("{prefix}.resnets.1"), channels, channels, device)?,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.second.forward(self.attention.forward(self.first.forward(x)))
    }
}

#[derive(Debug, Clone)]
struct Stage {
    resnets: Vec<ResnetBlock>,
    /// Encoder downsampler (pad right/bottom by one, stride-2 conv) or decoder upsampler (nearest x2, conv).
    resample: Option<Conv2d>,
}

#[derive(Debug, Clone)]
pub struct SdVae {
    encoder_in: Conv2d,
    down: Vec<Stage>,
    encoder_mid: MidBlock,
    encoder_norm: GroupNorm,
    encoder_out: Conv2d,
    quant_conv: Conv2d,
    post_quant_conv: Conv2d,
    decoder_in: Conv2d,
    decoder_mid: MidBlock,
    up: Vec<Stage>,
    decoder_norm: GroupNorm,
    decoder_out: Conv2d,
    tiling: bool,
    device: ComputeDevice,
}

impl SdVae {
    /// Resolve the pinned safetensors through the hub cache (downloading if allowed) and load them.
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        let config = hub.resolve(&CONFIG_FILE, fetch)?;
        let checkpoint = hub.resolve(&CHECKPOINT_FILE, fetch)?;
        if config.parent() != checkpoint.parent() {
            return Err(GpuError::Config(
                "pinned VAE artifacts must share one snapshot directory".to_owned(),
            ));
        }
        Self::from_checkpoint(&checkpoint, device)
    }

    /// Load from a local `diffusion_pytorch_model.safetensors` without digest checks.
    pub fn from_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_safetensors(path, &[])?;
        let burn = device.burn();
        let conv3 = |state: &mut StateDict, name: &str, input: usize, output: usize| {
            Conv2d::load(state, name, Conv2dSpec::new(input, output, 3).padding(1), burn)
        };

        let encoder_in = conv3(&mut state, "encoder.conv_in", 3, CHANNELS[0])?;
        let mut down = Vec::with_capacity(CHANNELS.len());
        let mut width = CHANNELS[0];
        for (i, &output) in CHANNELS.iter().enumerate() {
            let prefix = format!("encoder.down_blocks.{i}");
            let resnets = (0..2)
                .map(|j| {
                    let input = if j == 0 { width } else { output };
                    ResnetBlock::load(&mut state, &format!("{prefix}.resnets.{j}"), input, output, burn)
                })
                .collect::<Result<Vec<_>>>()?;
            let resample = if i + 1 < CHANNELS.len() {
                let spec = Conv2dSpec::new(output, output, 3).stride(2);
                Some(Conv2d::load(
                    &mut state,
                    &format!("{prefix}.downsamplers.0.conv"),
                    spec,
                    burn,
                )?)
            } else {
                None
            };
            down.push(Stage { resnets, resample });
            width = output;
        }
        let top = CHANNELS[CHANNELS.len() - 1];
        let encoder_mid = MidBlock::load(&mut state, "encoder.mid_block", top, burn)?;
        let encoder_norm = GroupNorm::load(&mut state, "encoder.conv_norm_out", GROUPS, top, EPS, burn)?;
        let encoder_out = conv3(&mut state, "encoder.conv_out", top, 2 * LATENT_CHANNELS)?;
        let quant_conv = Conv2d::load(
            &mut state,
            "quant_conv",
            Conv2dSpec::new(2 * LATENT_CHANNELS, 2 * LATENT_CHANNELS, 1),
            burn,
        )?;
        let post_quant_conv = Conv2d::load(
            &mut state,
            "post_quant_conv",
            Conv2dSpec::new(LATENT_CHANNELS, LATENT_CHANNELS, 1),
            burn,
        )?;

        let decoder_in = conv3(&mut state, "decoder.conv_in", LATENT_CHANNELS, top)?;
        let decoder_mid = MidBlock::load(&mut state, "decoder.mid_block", top, burn)?;
        let mut up = Vec::with_capacity(CHANNELS.len());
        let mut width = top;
        for (i, &output) in CHANNELS.iter().rev().enumerate() {
            let prefix = format!("decoder.up_blocks.{i}");
            let resnets = (0..3)
                .map(|j| {
                    let input = if j == 0 { width } else { output };
                    ResnetBlock::load(&mut state, &format!("{prefix}.resnets.{j}"), input, output, burn)
                })
                .collect::<Result<Vec<_>>>()?;
            let resample = if i + 1 < CHANNELS.len() {
                Some(conv3(
                    &mut state,
                    &format!("{prefix}.upsamplers.0.conv"),
                    output,
                    output,
                )?)
            } else {
                None
            };
            up.push(Stage { resnets, resample });
            width = output;
        }
        let decoder_norm = GroupNorm::load(
            &mut state,
            "decoder.conv_norm_out",
            GROUPS,
            CHANNELS[0],
            EPS,
            burn,
        )?;
        let decoder_out = conv3(&mut state, "decoder.conv_out", CHANNELS[0], 3)?;
        state.finish()?;
        Ok(Self {
            encoder_in,
            down,
            encoder_mid,
            encoder_norm,
            encoder_out,
            quant_conv,
            post_quant_conv,
            decoder_in,
            decoder_mid,
            up,
            decoder_norm,
            decoder_out,
            tiling: false,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// `vae.enable_tiling()`: frames wider or taller than 256 px are processed as blended
    /// 256 px tiles exactly like diffusers (the synthid profile enables it above 768 px).
    #[must_use]
    pub fn with_tiling(mut self, tiling: bool) -> Self {
        self.tiling = tiling;
        self
    }

    pub fn set_tiling(&mut self, tiling: bool) {
        self.tiling = tiling;
    }

    /// Scaled latents (`mode() * 0.18215`) of RGB frames in `[-1, 1]`, NCHW `[frames, 3, h, w]`.
    ///
    /// Returns NCHW `[frames, 4, h / 8, w / 8]`; `h` and `w` must be multiples of 8.
    pub fn encode(&self, pixels: &[f32], frames: usize, height: usize, width: usize) -> Result<Vec<f32>> {
        check_geometry(pixels.len(), frames, 3, height, width, LATENT_MULTIPLE)?;
        let plane = 3 * height * width;
        let tiled = self.tiling && (height > TILE_SAMPLE || width > TILE_SAMPLE);
        let mut out = Vec::with_capacity(frames * LATENT_CHANNELS * plane / 192);
        for frame in pixels.chunks_exact(plane) {
            let input = Plane::new(3, height, width, frame.to_vec());
            let moments = if tiled {
                run_tiled(&input, ENCODE_TILING, |tile| self.moments(tile))?
            } else {
                self.moments(&input)?
            };
            let mean = LATENT_CHANNELS * moments.height * moments.width;
            out.extend(moments.data[..mean].iter().map(|v| v * SCALING_FACTOR));
        }
        Ok(out)
    }

    /// Decoded RGB samples (`decoder(z / 0.18215)`) of scaled latents NCHW `[frames, 4, h, w]`.
    ///
    /// Returns NCHW `[frames, 3, 8h, 8w]`, unclamped like `vae.decode(...).sample`.
    pub fn decode(&self, latents: &[f32], frames: usize, height: usize, width: usize) -> Result<Vec<f32>> {
        check_geometry(latents.len(), frames, LATENT_CHANNELS, height, width, 1)?;
        let plane = LATENT_CHANNELS * height * width;
        let tiled = self.tiling && (height > TILE_LATENT || width > TILE_LATENT);
        let mut out = Vec::with_capacity(frames * plane * 48);
        for frame in latents.chunks_exact(plane) {
            let input = Plane::new(
                LATENT_CHANNELS,
                height,
                width,
                frame.iter().map(|v| v / SCALING_FACTOR).collect(),
            );
            let decoded = if tiled {
                run_tiled(&input, DECODE_TILING, |tile| self.decode_plane(tile))?
            } else {
                self.decode_plane(&input)?
            };
            out.extend(decoded.data);
        }
        Ok(out)
    }

    /// Encoder + `quant_conv` moments (mean then log-variance) of one RGB plane.
    fn moments(&self, input: &Plane) -> Result<Plane> {
        let x: Tensor<4> = Tensor::from_data(
            TensorData::new(input.data.clone(), [1, 3, input.height, input.width]),
            self.device.burn(),
        );
        let (h, w) = (input.height / LATENT_MULTIPLE, input.width / LATENT_MULTIPLE);
        Ok(Plane::new(
            2 * LATENT_CHANNELS,
            h,
            w,
            read_finite(self.encode_frame(x))?,
        ))
    }

    /// `post_quant_conv` + decoder of one unscaled latent plane.
    fn decode_plane(&self, input: &Plane) -> Result<Plane> {
        let z: Tensor<4> = Tensor::from_data(
            TensorData::new(
                input.data.clone(),
                [1, LATENT_CHANNELS, input.height, input.width],
            ),
            self.device.burn(),
        );
        let (h, w) = (input.height * LATENT_MULTIPLE, input.width * LATENT_MULTIPLE);
        Ok(Plane::new(3, h, w, read_finite(self.decode_frame(z))?))
    }

    fn encode_frame(&self, x: Tensor<4>) -> Tensor<4> {
        let mut h = self.encoder_in.forward(x);
        for stage in &self.down {
            h = stage.resnets.iter().fold(h, |h, block| block.forward(h));
            if let Some(conv) = &stage.resample {
                h = conv.forward_padded(h, [(0, 1), (0, 1)]);
            }
        }
        let h = self
            .encoder_out
            .forward(silu(self.encoder_norm.forward(self.encoder_mid.forward(h))));
        self.quant_conv.forward(h)
    }

    fn decode_frame(&self, z: Tensor<4>) -> Tensor<4> {
        let z = self.post_quant_conv.forward(z);
        let mut h = self.decoder_mid.forward(self.decoder_in.forward(z));
        for stage in &self.up {
            h = stage.resnets.iter().fold(h, |h, block| block.forward(h));
            if let Some(conv) = &stage.resample {
                h = conv.forward(upsample_nearest2(h));
            }
        }
        self.decoder_out.forward(silu(self.decoder_norm.forward(h)))
    }
}

/// `F.interpolate(scale_factor=2, mode="nearest")`.
pub(crate) fn upsample_nearest2(x: Tensor<4>) -> Tensor<4> {
    let [n, c, h, w] = x.dims();
    x.reshape([n, c, h, 1, w, 1])
        .expand([n, c, h, 2, w, 2])
        .reshape([n, c, 2 * h, 2 * w])
}

fn check_geometry(
    len: usize,
    frames: usize,
    channels: usize,
    height: usize,
    width: usize,
    multiple: usize,
) -> Result<()> {
    if frames == 0
        || height == 0
        || width == 0
        || !height.is_multiple_of(multiple)
        || !width.is_multiple_of(multiple)
    {
        return Err(GpuError::Input(format!(
            "VAE frames need positive sides divisible by {multiple}"
        )));
    }
    if len != frames * channels * height * width {
        return Err(GpuError::Input(format!(
            "VAE buffer holds {len} values, expected {frames}x{channels}x{height}x{width}"
        )));
    }
    Ok(())
}

/// `uint8` BGR frames (`[frames, h, w, 3]`) to RGB NCHW in `[-1, 1]` (`/ 127.5 - 1`).
pub fn bgr_u8_to_unit_rgb(frames: &[u8], height: usize, width: usize) -> Vec<f32> {
    let pixels = height * width;
    let mut out = vec![0.0f32; frames.len()];
    for (frame, dst) in frames
        .chunks_exact(3 * pixels)
        .zip(out.chunks_exact_mut(3 * pixels))
    {
        for (p, bgr) in frame.chunks_exact(3).enumerate() {
            for (c, &value) in bgr.iter().rev().enumerate() {
                dst[c * pixels + p] = f32::from(value) / 127.5 - 1.0;
            }
        }
    }
    out
}

/// Decoded RGB NCHW samples to `uint8` BGR frames: `round(clamp(x / 2 + 0.5, 0, 1) * 255)`.
pub fn unit_rgb_to_bgr_u8(samples: &[f32], height: usize, width: usize) -> Vec<u8> {
    let pixels = height * width;
    let mut out = vec![0u8; samples.len()];
    for (frame, dst) in samples
        .chunks_exact(3 * pixels)
        .zip(out.chunks_exact_mut(3 * pixels))
    {
        for p in 0..pixels {
            for c in 0..3 {
                let unit = (frame[c * pixels + p] / 2.0 + 0.5).clamp(0.0, 1.0);
                dst[3 * p + (2 - c)] = (unit * 255.0).round_ties_even() as u8;
            }
        }
    }
    out
}

/// High-depth `uint16` BGR frames to RGB NCHW in `[-1, 1]` (`/ 32767.5 - 1`).
pub fn bgr_u16_to_unit_rgb(frames: &[u16], height: usize, width: usize) -> Vec<f32> {
    let pixels = height * width;
    let mut out = vec![0.0f32; frames.len()];
    for (frame, dst) in frames
        .chunks_exact(3 * pixels)
        .zip(out.chunks_exact_mut(3 * pixels))
    {
        for (p, bgr) in frame.chunks_exact(3).enumerate() {
            for (c, &value) in bgr.iter().rev().enumerate() {
                dst[c * pixels + p] = f32::from(value) / 32767.5 - 1.0;
            }
        }
    }
    out
}

/// Decoded RGB NCHW samples to `uint16` BGR frames: `round(clamp(x / 2 + 0.5, 0, 1) * 65535)`.
pub fn unit_rgb_to_bgr_u16(samples: &[f32], height: usize, width: usize) -> Vec<u16> {
    let pixels = height * width;
    let mut out = vec![0u16; samples.len()];
    for (frame, dst) in samples
        .chunks_exact(3 * pixels)
        .zip(out.chunks_exact_mut(3 * pixels))
    {
        for p in 0..pixels {
            for c in 0..3 {
                let unit = (frame[c * pixels + p] / 2.0 + 0.5).clamp(0.0, 1.0);
                dst[3 * p + (2 - c)] = (unit * 65535.0).round_ties_even() as u16;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_conversions_round_trip_like_python() {
        let bgr = [10u8, 128, 255, 0, 64, 200];
        let unit = bgr_u8_to_unit_rgb(&bgr, 1, 2);
        assert_eq!(unit[0], 255.0 / 127.5 - 1.0);
        assert_eq!(unit[1], 200.0 / 127.5 - 1.0);
        assert_eq!(unit_rgb_to_bgr_u8(&unit, 1, 2), bgr);
    }

    #[test]
    fn nearest_upsample_repeats_pixels() {
        let device = Device::flex();
        let x = Tensor::<4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);
        let up = crate::io::read_f32(upsample_nearest2(x)).unwrap();
        assert_eq!(up[..8], [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
    }
}

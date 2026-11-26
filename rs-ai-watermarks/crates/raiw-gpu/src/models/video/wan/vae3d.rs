//! Wan 2.1 `AutoencoderKLWan` (`vae/` subfolder, float32), the causal 3-D video VAE.
//!
//! Clips live on the device as `[T, C, H, W]` (time as the batch axis). Each causal `Conv3d`
//! is a sum of per-time-tap `Conv2d` calls on the chunked im2col path, and a [`TimeCache`]
//! carries the last two input frames of every convolution across chunks exactly like
//! diffusers' `feat_cache`: the encoder sees the first frame alone and then chunks of four,
//! the decoder one latent frame at a time. Clips wider or taller than 256 px use Wan's own
//! `tiled_encode`/`tiled_decode` geometry (256 px tiles, 192 px stride), as `video_wan.py`
//! enables tiling and slicing.

use std::path::Path;

use burn::tensor::activation::silu;
use burn::tensor::{Device, Tensor, TensorData, s};

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub};
use crate::io::read_finite;
use crate::layers::{Conv2d, Conv2dSpec};
use crate::models::video::tiling::{Plane, Tiling, run_tiled};
use crate::models::video::vae::upsample_nearest2;
use crate::weights::{HostTensor, StateDict};

use super::attention::scaled_dot_product;
use super::{LATENT_CHANNELS, SPATIAL_FACTOR, TEMPORAL_FACTOR, VAE_CHECKPOINT, VAE_CONFIG};

const BASE_DIM: usize = 96;
const DIM_MULT: [usize; 4] = [1, 2, 4, 4];
const RES_BLOCKS: usize = 2;
const TIME_DOWNSAMPLE: [bool; 3] = [false, true, true];
/// `CACHE_T`: frames each causal convolution remembers from the previous chunk.
const CACHE_FRAMES: usize = 2;
const TILE_SAMPLE: usize = 256;
const TILE_SAMPLE_STRIDE: usize = 192;
const TILE_LATENT: usize = TILE_SAMPLE / SPATIAL_FACTOR;
const TILE_LATENT_STRIDE: usize = TILE_SAMPLE_STRIDE / SPATIAL_FACTOR;
const ENCODE_TILING: Tiling = Tiling {
    tile: TILE_SAMPLE,
    stride: TILE_SAMPLE_STRIDE,
    blend: TILE_LATENT - TILE_LATENT_STRIDE,
    limit: TILE_LATENT_STRIDE,
};
const DECODE_TILING: Tiling = Tiling {
    tile: TILE_LATENT,
    stride: TILE_LATENT_STRIDE,
    blend: TILE_SAMPLE - TILE_SAMPLE_STRIDE,
    limit: TILE_SAMPLE_STRIDE,
};

/// `vae.config.latents_mean` / `latents_std`: the pipeline normalizes latents with them.
pub const LATENTS_MEAN: [f32; LATENT_CHANNELS] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632,
    -0.1922, -0.9497, 0.2503, -0.2921,
];
pub const LATENTS_STD: [f32; LATENT_CHANNELS] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382,
    1.1253, 2.8251, 1.916,
];

/// One `feat_cache` entry.
#[derive(Debug, Clone, Default)]
enum Slot {
    #[default]
    Empty,
    /// diffusers' `"Rep"` marker: the first decoder chunk skipped the temporal upsample.
    Repeat,
    Frames(Box<Tensor<4>>),
}

impl Slot {
    fn tail(frames: Tensor<4>) -> Self {
        Self::Frames(Box::new(frames))
    }

    fn frames(&self) -> Option<&Tensor<4>> {
        match self {
            Self::Frames(frames) => Some(frames.as_ref()),
            _ => None,
        }
    }
}

/// Per-convolution causal state, visited in call order once per chunk (`feat_idx`).
#[derive(Debug, Default)]
struct TimeCache {
    slots: Vec<Slot>,
    cursor: usize,
}

impl TimeCache {
    fn rewind(&mut self) {
        self.cursor = 0;
    }

    fn next(&mut self) -> &mut Slot {
        if self.cursor == self.slots.len() {
            self.slots.push(Slot::Empty);
        }
        self.cursor += 1;
        &mut self.slots[self.cursor - 1]
    }
}

/// The last `CACHE_FRAMES` frames of `x`, topped up with the previous slot's last frame.
fn cache_tail(x: &Tensor<4>, previous: Option<&Tensor<4>>) -> Tensor<4> {
    let frames = x.dims()[0];
    let tail = x.clone().slice(s![frames.saturating_sub(CACHE_FRAMES)..]);
    match previous {
        Some(prev) if frames < CACHE_FRAMES => {
            let last = prev.dims()[0] - 1;
            Tensor::cat(vec![prev.clone().slice(s![last..]), tail], 0)
        }
        _ => tail,
    }
}

/// A 1x1 convolution over `[T, C, H, W]` as `weight @ x` on the channel axis.
#[derive(Debug, Clone)]
struct Pointwise {
    weight: Tensor<3>,
    bias: Option<Tensor<4>>,
}

impl Pointwise {
    fn from_parts(weight: Tensor<2>, bias: Option<Tensor<1>>) -> Self {
        let [out, _] = weight.dims();
        Self {
            weight: weight.unsqueeze_dim(0),
            bias: bias.map(|b| b.reshape([1, out, 1, 1])),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [t, c, h, w] = x.dims();
        let out = self.weight.dims()[1];
        let y = self
            .weight
            .clone()
            .matmul(x.reshape([t, c, h * w]))
            .reshape([t, out, h, w]);
        match &self.bias {
            Some(bias) => y + bias.clone(),
            None => y,
        }
    }
}

/// Per-time-tap frame convolutions, oldest first; only the newest carries the bias.
#[derive(Debug, Clone)]
enum Taps {
    Spatial(Vec<Conv2d>),
    Pointwise(Vec<Pointwise>),
}

impl Taps {
    fn len(&self) -> usize {
        match self {
            Self::Spatial(taps) => taps.len(),
            Self::Pointwise(taps) => taps.len(),
        }
    }

    fn forward(&self, k: usize, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Spatial(taps) => taps[k].forward(x),
            Self::Pointwise(taps) => taps[k].forward(x),
        }
    }
}

/// `WanCausalConv3d`: `2 * pad_t` zero (or cached) frames in front, none behind.
#[derive(Debug, Clone)]
struct CausalConv3d {
    taps: Taps,
    stride: usize,
    pad: usize,
}

impl CausalConv3d {
    /// `kernel = (k_t, k_s)`, `padding = (p_t, p_s)`, time stride `stride`.
    fn load(
        state: &mut StateDict,
        prefix: &str,
        (input, output): (usize, usize),
        (k_t, k_s): (usize, usize),
        (p_t, p_s): (usize, usize),
        stride: usize,
        device: &Device,
    ) -> Result<Self> {
        let weight = state.take(&format!("{prefix}.weight"), &[output, input, k_t, k_s, k_s])?;
        let bias: Tensor<1> = state.tensor(&format!("{prefix}.bias"), [output], device)?;
        let tap = |k: usize| {
            let data = time_tap(&weight, k, (input, output), (k_t, k_s));
            (data, (k + 1 == k_t).then(|| bias.clone()))
        };
        let taps = if k_s == 1 {
            Taps::Pointwise(
                (0..k_t)
                    .map(|k| {
                        let (data, bias) = tap(k);
                        let w = Tensor::from_data(TensorData::new(data, [output, input]), device);
                        Pointwise::from_parts(w, bias)
                    })
                    .collect(),
            )
        } else {
            let spec = Conv2dSpec::new(input, output, k_s).padding(p_s);
            Taps::Spatial(
                (0..k_t)
                    .map(|k| {
                        let (data, bias) = tap(k);
                        let w = Tensor::from_data(TensorData::new(data, [output, input, k_s, k_s]), device);
                        Conv2d::from_parts(w, bias, spec, device)
                    })
                    .collect(),
            )
        };
        Ok(Self {
            taps,
            stride,
            pad: 2 * p_t,
        })
    }

    fn pointwise(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::load(state, prefix, (input, output), (1, 1), (0, 0), 1, device)
    }

    fn cube(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::load(state, prefix, (input, output), (3, 3), (1, 1), 1, device)
    }

    /// Convolve `x` preceded by `cache` frames (if any) and the remaining zero padding.
    fn forward(&self, x: Tensor<4>, cache: Option<&Tensor<4>>) -> Tensor<4> {
        let (x, zeros) = match cache {
            Some(prefix) if self.pad > 0 => {
                let cached = prefix.dims()[0];
                (
                    Tensor::cat(vec![prefix.clone(), x], 0),
                    self.pad.saturating_sub(cached),
                )
            }
            _ => (x, self.pad),
        };
        let frames = x.dims()[0];
        let k_t = self.taps.len();
        debug_assert!(zeros + frames >= k_t, "causal conv needs a full kernel span");
        let out_frames = (zeros + frames).saturating_sub(k_t) / self.stride + 1;
        let mut out: Option<Tensor<4>> = None;
        // Newest tap first: it reads a real frame for every output, so it seeds the sum
        for k in (0..k_t).rev() {
            let first = if k >= zeros {
                0
            } else {
                (zeros - k).div_ceil(self.stride)
            };
            if first >= out_frames {
                continue;
            }
            let start = first * self.stride + k - zeros;
            let end = start + (out_frames - first - 1) * self.stride + 1;
            let input = if self.stride == 1 {
                x.clone().slice(s![start..end])
            } else {
                let step = self.stride as isize;
                x.clone().slice(s![start..end;step])
            };
            let y = self.taps.forward(k, input);
            out = Some(match out {
                None if first == 0 => y,
                None => {
                    let [_, c, h, w] = y.dims();
                    let lead = Tensor::zeros([first, c, h, w], &y.device());
                    Tensor::cat(vec![lead, y], 0)
                }
                Some(acc) if first == 0 => acc + y,
                Some(acc) => {
                    let tail = acc.clone().slice(s![first..]) + y;
                    acc.slice_assign(s![first..], tail)
                }
            });
        }
        out.unwrap_or(x)
    }

    /// A cached call: consume one slot and leave this chunk's tail in it.
    fn forward_cached(&self, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
        let slot = cache.next();
        let tail = cache_tail(&x, slot.frames());
        let out = self.forward(x, slot.frames());
        *slot = Slot::tail(tail);
        out
    }
}

/// Weights of time tap `k` of a `[out, in, k_t, k_s, k_s]` kernel as `[out, in, k_s, k_s]`.
fn time_tap(
    weight: &HostTensor,
    k: usize,
    (input, output): (usize, usize),
    (k_t, k_s): (usize, usize),
) -> Vec<f32> {
    let plane = k_s * k_s;
    let mut tap = Vec::with_capacity(output * input * plane);
    for pair in 0..output * input {
        let start = (pair * k_t + k) * plane;
        tap.extend_from_slice(&weight.data[start..start + plane]);
    }
    tap
}

/// `WanRMS_norm`: `F.normalize(x, dim=1) * sqrt(C) * gamma` over channels.
#[derive(Debug, Clone)]
struct ChannelRmsNorm {
    scale: Tensor<4>,
}

impl ChannelRmsNorm {
    fn load(
        state: &mut StateDict,
        prefix: &str,
        channels: usize,
        images: bool,
        device: &Device,
    ) -> Result<Self> {
        let name = format!("{prefix}.gamma");
        let shape: &[usize] = if images {
            &[channels, 1, 1]
        } else {
            &[channels, 1, 1, 1]
        };
        let gamma = state.take(&name, shape)?;
        let scale = (channels as f64).sqrt() as f32;
        let data: Vec<f32> = gamma.data.iter().map(|g| g * scale).collect();
        Ok(Self {
            scale: Tensor::from_data(TensorData::new(data, [1, channels, 1, 1]), device),
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let norm = x.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-12);
        x / norm * self.scale.clone()
    }
}

/// `WanResidualBlock`.
#[derive(Debug, Clone)]
struct ResidualBlock {
    norm1: ChannelRmsNorm,
    conv1: CausalConv3d,
    norm2: ChannelRmsNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn load(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<Self> {
        let shortcut = (input != output)
            .then(|| {
                CausalConv3d::pointwise(state, &format!("{prefix}.conv_shortcut"), input, output, device)
            })
            .transpose()?;
        Ok(Self {
            norm1: ChannelRmsNorm::load(state, &format!("{prefix}.norm1"), input, false, device)?,
            conv1: CausalConv3d::cube(state, &format!("{prefix}.conv1"), input, output, device)?,
            norm2: ChannelRmsNorm::load(state, &format!("{prefix}.norm2"), output, false, device)?,
            conv2: CausalConv3d::cube(state, &format!("{prefix}.conv2"), output, output, device)?,
            shortcut,
        })
    }

    fn forward(&self, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
        let h = match &self.shortcut {
            Some(shortcut) => shortcut.forward(x.clone(), None),
            None => x.clone(),
        };
        let x = self.conv1.forward_cached(silu(self.norm1.forward(x)), cache);
        let x = self.conv2.forward_cached(silu(self.norm2.forward(x)), cache);
        x + h
    }
}

/// `WanAttentionBlock`: single-head spatial self-attention within each frame.
#[derive(Debug, Clone)]
struct FrameAttention {
    norm: ChannelRmsNorm,
    qkv: Pointwise,
    proj: Pointwise,
    channels: usize,
}

impl FrameAttention {
    fn load(state: &mut StateDict, prefix: &str, channels: usize, device: &Device) -> Result<Self> {
        let mut pointwise = |name: &str, out: usize| -> Result<Pointwise> {
            let w = state.take(&format!("{prefix}.{name}.weight"), &[out, channels, 1, 1])?;
            let bias = state.tensor(&format!("{prefix}.{name}.bias"), [out], device)?;
            let w = Tensor::from_data(TensorData::new(w.data, [out, channels]), device);
            Ok(Pointwise::from_parts(w, Some(bias)))
        };
        Ok(Self {
            qkv: pointwise("to_qkv", 3 * channels)?,
            proj: pointwise("proj", channels)?,
            norm: ChannelRmsNorm::load(state, &format!("{prefix}.norm"), channels, true, device)?,
            channels,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [t, c, h, w] = x.dims();
        let qkv = self
            .qkv
            .forward(self.norm.forward(x.clone()))
            .reshape([t, 3 * c, h * w])
            .swap_dims(1, 2);
        let head = |i: usize| {
            qkv.clone()
                .slice(s![.., .., i * self.channels..(i + 1) * self.channels])
                .unsqueeze_dim::<4>(1)
        };
        let out = scaled_dot_product(head(0), head(1), head(2))
            .squeeze_dim::<3>(1)
            .swap_dims(1, 2)
            .reshape([t, c, h, w]);
        x + self.proj.forward(out)
    }
}

/// `WanMidBlock`: resnet, attention, resnet.
#[derive(Debug, Clone)]
struct MidBlock {
    first: ResidualBlock,
    attention: FrameAttention,
    second: ResidualBlock,
}

impl MidBlock {
    fn load(state: &mut StateDict, prefix: &str, channels: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            first: ResidualBlock::load(state, &format!("{prefix}.resnets.0"), channels, channels, device)?,
            attention: FrameAttention::load(state, &format!("{prefix}.attentions.0"), channels, device)?,
            second: ResidualBlock::load(state, &format!("{prefix}.resnets.1"), channels, channels, device)?,
        })
    }

    fn forward(&self, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
        let x = self.first.forward(x, cache);
        self.second.forward(self.attention.forward(x), cache)
    }
}

/// `WanResample` in one of its four modes.
#[derive(Debug, Clone)]
enum Resample {
    Down {
        conv: Conv2d,
        time: Option<CausalConv3d>,
    },
    Up {
        conv: Conv2d,
        time: Option<CausalConv3d>,
    },
}

impl Resample {
    fn load_down(
        state: &mut StateDict,
        prefix: &str,
        dim: usize,
        temporal: bool,
        device: &Device,
    ) -> Result<Self> {
        let spec = Conv2dSpec::new(dim, dim, 3).stride(2);
        let conv = Conv2d::load(state, &format!("{prefix}.resample.1"), spec, device)?;
        let time = temporal
            .then(|| {
                CausalConv3d::load(
                    state,
                    &format!("{prefix}.time_conv"),
                    (dim, dim),
                    (3, 1),
                    (0, 0),
                    2,
                    device,
                )
            })
            .transpose()?;
        Ok(Self::Down { conv, time })
    }

    fn load_up(
        state: &mut StateDict,
        prefix: &str,
        dim: usize,
        temporal: bool,
        device: &Device,
    ) -> Result<Self> {
        let spec = Conv2dSpec::new(dim, dim / 2, 3).padding(1);
        let conv = Conv2d::load(state, &format!("{prefix}.resample.1"), spec, device)?;
        let time = temporal
            .then(|| {
                CausalConv3d::load(
                    state,
                    &format!("{prefix}.time_conv"),
                    (dim, 2 * dim),
                    (3, 1),
                    (1, 0),
                    1,
                    device,
                )
            })
            .transpose()?;
        Ok(Self::Up { conv, time })
    }

    fn forward(&self, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
        match self {
            Self::Down { conv, time } => {
                let x = conv.forward_padded(x, [(0, 1), (0, 1)]);
                match time {
                    Some(time) => downsample_time(time, x, cache),
                    None => x,
                }
            }
            Self::Up { conv, time } => {
                let x = match time {
                    Some(time) => upsample_time(time, x, cache),
                    None => x,
                };
                conv.forward(upsample_nearest2(x))
            }
        }
    }
}

/// `downsample3d`: the first chunk passes through; later ones convolve with the last cached frame.
fn downsample_time(time: &CausalConv3d, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
    let slot = cache.next();
    let frames = x.dims()[0];
    let tail = x.clone().slice(s![frames - 1..]);
    let out = match slot.frames() {
        None => x,
        Some(prev) => {
            let last = prev.dims()[0] - 1;
            time.forward(Tensor::cat(vec![prev.clone().slice(s![last..]), x], 0), None)
        }
    };
    *slot = Slot::tail(tail);
    out
}

/// `upsample3d`: the first chunk marks `"Rep"`; later ones double the frames via `time_conv`.
fn upsample_time(time: &CausalConv3d, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
    let slot = cache.next();
    if matches!(slot, Slot::Empty) {
        *slot = Slot::Repeat;
        return x;
    }
    let [frames, c, h, w] = x.dims();
    let tail = match slot {
        Slot::Repeat if frames < CACHE_FRAMES => {
            let tail = x.clone().slice(s![frames.saturating_sub(CACHE_FRAMES)..]);
            let zeros = Tensor::zeros([CACHE_FRAMES - tail.dims()[0], c, h, w], &x.device());
            Tensor::cat(vec![zeros, tail], 0)
        }
        _ => cache_tail(&x, slot.frames()),
    };
    let y = time.forward(x, slot.frames());
    *slot = Slot::tail(tail);
    // `[T, 2C]` -> `[2T, C]`: frame `2t` takes the first channel half, `2t + 1` the second
    y.reshape([2 * frames, c, h, w])
}

/// Residual blocks plus an optional resampler: one encoder level or one decoder up block.
#[derive(Debug, Clone)]
struct Stage {
    resnets: Vec<ResidualBlock>,
    resample: Option<Resample>,
}

impl Stage {
    fn forward(&self, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
        let x = self.resnets.iter().fold(x, |x, resnet| resnet.forward(x, cache));
        match &self.resample {
            Some(resample) => resample.forward(x, cache),
            None => x,
        }
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    conv_in: CausalConv3d,
    stages: Vec<Stage>,
    mid: MidBlock,
    norm_out: ChannelRmsNorm,
    conv_out: CausalConv3d,
}

impl Encoder {
    fn load(state: &mut StateDict, device: &Device) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(1).chain(DIM_MULT).map(|m| m * BASE_DIM).collect();
        let conv_in = CausalConv3d::cube(state, "encoder.conv_in", 3, dims[0], device)?;
        let mut stages = Vec::new();
        // `down_blocks` is one flat list: resnets, then the level's resampler
        let mut index = 0;
        for (i, pair) in dims.windows(2).enumerate() {
            let (mut input, output) = (pair[0], pair[1]);
            let mut resnets = Vec::new();
            for _ in 0..RES_BLOCKS {
                let prefix = format!("encoder.down_blocks.{index}");
                resnets.push(ResidualBlock::load(state, &prefix, input, output, device)?);
                input = output;
                index += 1;
            }
            let resample = if i + 1 < DIM_MULT.len() {
                let prefix = format!("encoder.down_blocks.{index}");
                index += 1;
                Some(Resample::load_down(
                    state,
                    &prefix,
                    output,
                    TIME_DOWNSAMPLE[i],
                    device,
                )?)
            } else {
                None
            };
            stages.push(Stage { resnets, resample });
        }
        let top = dims[dims.len() - 1];
        Ok(Self {
            conv_in,
            stages,
            mid: MidBlock::load(state, "encoder.mid_block", top, device)?,
            norm_out: ChannelRmsNorm::load(state, "encoder.norm_out", top, false, device)?,
            conv_out: CausalConv3d::cube(state, "encoder.conv_out", top, 2 * LATENT_CHANNELS, device)?,
        })
    }

    fn forward(&self, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
        let x = self.conv_in.forward_cached(x, cache);
        let x = self.stages.iter().fold(x, |x, stage| stage.forward(x, cache));
        let x = silu(self.norm_out.forward(self.mid.forward(x, cache)));
        self.conv_out.forward_cached(x, cache)
    }
}

#[derive(Debug, Clone)]
struct Decoder {
    conv_in: CausalConv3d,
    mid: MidBlock,
    blocks: Vec<Stage>,
    norm_out: ChannelRmsNorm,
    conv_out: CausalConv3d,
}

impl Decoder {
    fn load(state: &mut StateDict, device: &Device) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(DIM_MULT[DIM_MULT.len() - 1])
            .chain(DIM_MULT.iter().rev().copied())
            .map(|m| m * BASE_DIM)
            .collect();
        let conv_in = CausalConv3d::cube(state, "decoder.conv_in", LATENT_CHANNELS, dims[0], device)?;
        let mid = MidBlock::load(state, "decoder.mid_block", dims[0], device)?;
        let mut blocks = Vec::new();
        for (i, pair) in dims.windows(2).enumerate() {
            let output = pair[1];
            let mut input = if i > 0 { pair[0] / 2 } else { pair[0] };
            let prefix = format!("decoder.up_blocks.{i}");
            let mut resnets = Vec::new();
            for j in 0..=RES_BLOCKS {
                resnets.push(ResidualBlock::load(
                    state,
                    &format!("{prefix}.resnets.{j}"),
                    input,
                    output,
                    device,
                )?);
                input = output;
            }
            let resample = (i + 1 < DIM_MULT.len())
                .then(|| {
                    let temporal = TIME_DOWNSAMPLE[TIME_DOWNSAMPLE.len() - 1 - i];
                    Resample::load_up(state, &format!("{prefix}.upsamplers.0"), output, temporal, device)
                })
                .transpose()?;
            blocks.push(Stage { resnets, resample });
        }
        let out = dims[dims.len() - 1];
        Ok(Self {
            conv_in,
            mid,
            blocks,
            norm_out: ChannelRmsNorm::load(state, "decoder.norm_out", out, false, device)?,
            conv_out: CausalConv3d::cube(state, "decoder.conv_out", out, 3, device)?,
        })
    }

    fn forward(&self, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
        let x = self.mid.forward(self.conv_in.forward_cached(x, cache), cache);
        let x = self.blocks.iter().fold(x, |x, stage| stage.forward(x, cache));
        self.conv_out
            .forward_cached(silu(self.norm_out.forward(x)), cache)
    }
}

/// `AutoencoderKLWan` from `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (`vae/`).
#[derive(Debug, Clone)]
pub struct WanVae {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: CausalConv3d,
    post_quant_conv: CausalConv3d,
    tiling: bool,
    device: ComputeDevice,
}

impl WanVae {
    /// Resolve the pinned `vae/` safetensors through the hub cache (downloading if allowed).
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        let config = hub.resolve(&VAE_CONFIG, fetch)?;
        let checkpoint = hub.resolve(&VAE_CHECKPOINT, fetch)?;
        if config.parent() != checkpoint.parent() {
            return Err(GpuError::Config(
                "pinned Wan VAE artifacts must share one snapshot directory".to_owned(),
            ));
        }
        Self::from_checkpoint(&checkpoint, device)
    }

    /// Load a local `vae/diffusion_pytorch_model.safetensors` without digest checks.
    pub fn from_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_safetensors(path, &[])?;
        let burn = device.burn();
        let encoder = Encoder::load(&mut state, burn)?;
        let decoder = Decoder::load(&mut state, burn)?;
        let z2 = 2 * LATENT_CHANNELS;
        let quant_conv = CausalConv3d::pointwise(&mut state, "quant_conv", z2, z2, burn)?;
        let post_quant_conv = CausalConv3d::pointwise(
            &mut state,
            "post_quant_conv",
            LATENT_CHANNELS,
            LATENT_CHANNELS,
            burn,
        )?;
        state.finish()?;
        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            tiling: true,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// `enable_tiling()` (on by default, as `video_wan.py` loads it).
    #[must_use]
    pub fn with_tiling(mut self, tiling: bool) -> Self {
        self.tiling = tiling;
        self
    }

    pub fn set_tiling(&mut self, tiling: bool) {
        self.tiling = tiling;
    }

    /// `encode(video).latent_dist.mode()` of `1 + 4k` RGB frames in `[-1, 1]`, `[T, 3, H, W]`.
    ///
    /// Returns the unnormalized latent means as `[16, 1 + k, H / 8, W / 8]`.
    pub fn encode(&self, frames: &[f32], count: usize, height: usize, width: usize) -> Result<Vec<f32>> {
        check_clip(frames.len(), count, 3, height, width, SPATIAL_FACTOR)?;
        if !(count - 1).is_multiple_of(TEMPORAL_FACTOR) {
            return Err(GpuError::Input(format!(
                "Wan VAE encodes 1 + 4k frames, got {count}"
            )));
        }
        let input = Plane::new(count * 3, height, width, frames.to_vec());
        let tiled = self.tiling && (height > TILE_SAMPLE || width > TILE_SAMPLE);
        let moments = if tiled {
            run_tiled(&input, ENCODE_TILING, |tile| self.encode_plane(tile, count))?
        } else {
            self.encode_plane(&input, count)?
        };
        let latent_frames = 1 + (count - 1) / TEMPORAL_FACTOR;
        let (h, w) = (height / SPATIAL_FACTOR, width / SPATIAL_FACTOR);
        check_assembled(&moments, latent_frames * LATENT_CHANNELS, h, w)?;
        Ok(swap_time_channels(
            &moments.data,
            latent_frames,
            LATENT_CHANNELS,
            h * w,
        ))
    }

    /// `decode(z).sample` of unnormalized latents `[16, T, h, w]`, clamped to `[-1, 1]`.
    ///
    /// Returns frames `[1 + 4(T - 1), 3, 8h, 8w]`.
    pub fn decode(
        &self,
        latents: &[f32],
        latent_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        check_clip(latents.len(), latent_frames, LATENT_CHANNELS, height, width, 1)?;
        let time_major = swap_time_channels(latents, LATENT_CHANNELS, latent_frames, height * width);
        let input = Plane::new(latent_frames * LATENT_CHANNELS, height, width, time_major);
        let tiled = self.tiling && (height > TILE_LATENT || width > TILE_LATENT);
        let mut decoded = if tiled {
            run_tiled(&input, DECODE_TILING, |tile| {
                self.decode_plane(tile, latent_frames)
            })?
        } else {
            self.decode_plane(&input, latent_frames)?
        };
        let frames = 1 + TEMPORAL_FACTOR * (latent_frames - 1);
        check_assembled(
            &decoded,
            frames * 3,
            height * SPATIAL_FACTOR,
            width * SPATIAL_FACTOR,
        )?;
        for value in &mut decoded.data {
            *value = value.clamp(-1.0, 1.0);
        }
        Ok(decoded.data)
    }

    /// Encoder + `quant_conv` means of one (tile of a) clip, time-major `[T', 16, h, w]`.
    fn encode_plane(&self, input: &Plane, count: usize) -> Result<Plane> {
        let (h, w) = (input.height, input.width);
        let clip: Tensor<4> = Tensor::from_data(
            TensorData::new(input.data.clone(), [count, 3, h, w]),
            self.device.burn(),
        );
        let mut cache = TimeCache::default();
        let mut data = Vec::new();
        let mut start = 0;
        while start < count {
            let end = if start == 0 { 1 } else { start + TEMPORAL_FACTOR };
            cache.rewind();
            let moments = self.quant_conv.forward(
                self.encoder
                    .forward(clip.clone().slice(s![start..end]), &mut cache),
                None,
            );
            data.extend(read_finite(moments.slice(s![.., 0..LATENT_CHANNELS]))?);
            start = end;
        }
        let latent_frames = 1 + (count - 1) / TEMPORAL_FACTOR;
        Ok(Plane::new(
            latent_frames * LATENT_CHANNELS,
            h / SPATIAL_FACTOR,
            w / SPATIAL_FACTOR,
            data,
        ))
    }

    /// `post_quant_conv` + decoder of one (tile of a) latent clip, time-major `[T, 3, H, W]`.
    fn decode_plane(&self, input: &Plane, latent_frames: usize) -> Result<Plane> {
        let (h, w) = (input.height, input.width);
        let z: Tensor<4> = Tensor::from_data(
            TensorData::new(input.data.clone(), [latent_frames, LATENT_CHANNELS, h, w]),
            self.device.burn(),
        );
        let z = self.post_quant_conv.forward(z, None);
        let mut cache = TimeCache::default();
        let mut data = Vec::new();
        for i in 0..latent_frames {
            cache.rewind();
            data.extend(read_finite(
                self.decoder.forward(z.clone().slice(s![i..i + 1]), &mut cache),
            )?);
        }
        let frames = 1 + TEMPORAL_FACTOR * (latent_frames - 1);
        Ok(Plane::new(
            frames * 3,
            h * SPATIAL_FACTOR,
            w * SPATIAL_FACTOR,
            data,
        ))
    }
}

/// `[a, b, n]` -> `[b, a, n]` on the host (time-major <-> channel-major clips).
pub(crate) fn swap_time_channels(data: &[f32], a: usize, b: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for i in 0..a {
        for j in 0..b {
            let src = (i * b + j) * n;
            let dst = (j * a + i) * n;
            out[dst..dst + n].copy_from_slice(&data[src..src + n]);
        }
    }
    out
}

fn check_clip(
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
            "Wan VAE clips need frames and sides divisible by {multiple}"
        )));
    }
    if len != frames * channels * height * width {
        return Err(GpuError::Input(format!(
            "Wan VAE buffer holds {len} values, expected {frames}x{channels}x{height}x{width}"
        )));
    }
    Ok(())
}

fn check_assembled(plane: &Plane, channels: usize, height: usize, width: usize) -> Result<()> {
    if (plane.channels, plane.height, plane.width) == (channels, height, width) {
        Ok(())
    } else {
        Err(GpuError::Config(format!(
            "tiled Wan VAE produced {}x{}x{}, expected {channels}x{height}x{width}",
            plane.channels, plane.height, plane.width
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::read_f32;

    #[test]
    fn causal_conv_matches_zero_padded_sum() {
        let device = Device::flex();
        let mut tensors = std::collections::BTreeMap::new();
        let weight = HostTensor::new(vec![1, 1, 3, 1, 1], vec![1.0, 10.0, 100.0]).unwrap();
        tensors.insert("c.weight".to_owned(), weight);
        tensors.insert("c.bias".to_owned(), HostTensor::new(vec![1], vec![0.5]).unwrap());
        let mut state = StateDict::from_tensors("mem", tensors);
        let conv = CausalConv3d::load(&mut state, "c", (1, 1), (3, 1), (1, 0), 1, &device).unwrap();
        let x = Tensor::<4>::from_floats([[[[1.0]]], [[[2.0]]], [[[3.0]]]], &device);
        let full = read_f32(conv.forward(x.clone(), None)).unwrap();
        assert_eq!(full, vec![100.5, 210.5, 321.5]);
        // Chunked with a one-frame cache reproduces the tail of the full sequence
        let mut cache = TimeCache::default();
        let first = conv.forward_cached(x.clone().slice(s![0..1]), &mut cache);
        cache.rewind();
        let rest = conv.forward_cached(x.slice(s![1..3]), &mut cache);
        assert_eq!(read_f32(first).unwrap(), vec![100.5]);
        assert_eq!(read_f32(rest).unwrap(), vec![210.5, 321.5]);
    }

    #[test]
    fn time_channel_swap_round_trips() {
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let swapped = swap_time_channels(&data, 2, 3, 4);
        assert_eq!(swapped[4..8], [12.0, 13.0, 14.0, 15.0]);
        assert_eq!(swap_time_channels(&swapped, 3, 2, 4), data);
    }
}

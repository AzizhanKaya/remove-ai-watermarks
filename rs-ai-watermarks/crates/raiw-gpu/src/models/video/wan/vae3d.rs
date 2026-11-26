//! `AutoencoderKLWan` (`vae/` subfolder, float32), the causal 3-D video VAE of Wan 2.1 and,
//! from its `config.json` ([`VaeConfig`]), Wan 2.2 TI2V 5B.
//!
//! Clips live on the device as `[T, C, H, W]` (time as the batch axis). Each causal `Conv3d`
//! is a sum of per-time-tap `Conv2d` calls on the chunked im2col path, and a [`TimeCache`]
//! carries the last two input frames of every convolution across chunks exactly like
//! diffusers' `feat_cache`: the encoder sees the first frame alone and then chunks of four,
//! the decoder one latent frame at a time. Wan 2.1 clips wider or taller than 256 px use Wan's
//! own `tiled_encode`/`tiled_decode` geometry (256 px tiles, 192 px stride), as `video_wan.py`
//! enables tiling and slicing.
//!
//! Wan 2.2 patchifies 2x2 pixels into 12 channels and wraps every level in a parameter-free
//! shortcut ([`AvgDown`] in the encoder, [`DupUp`] in the decoder); diffusers runs it untiled.

use std::path::{Path, PathBuf};

use burn::tensor::activation::silu;
use burn::tensor::{Device, Tensor, TensorData, s};

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub, HubFile};
use crate::io::read_finite;
use crate::layers::{Conv2d, Conv2dSpec};
use crate::models::video::tiling::{Plane, Tiling, run_tiled};
use crate::models::video::vae::upsample_nearest2;
use crate::weights::{HostTensor, StateDict};

use super::attention::scaled_dot_product;
use super::config::VaeConfig;
use super::{
    SPATIAL_FACTOR, TEMPORAL_FACTOR, VAE_CHECKPOINT, VAE_CONFIG, WAN22_VAE_CHECKPOINT, WAN22_VAE_CONFIG,
};
#[cfg(feature = "gpu-runtime")]
use crate::kernels::{self, ConvShape};

/// `CACHE_T`: frames each causal convolution remembers from the previous chunk.
const CACHE_FRAMES: usize = 2;
// Wan 2.1 tiling geometry (8x spatial factor)
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

/// A checkpoint's pinned `vae/config.json` and weights.
#[derive(Debug, Clone, Copy)]
pub struct VaeFiles {
    pub config: HubFile<'static>,
    pub checkpoint: HubFile<'static>,
}

/// `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
pub const WAN21_VAE: VaeFiles = VaeFiles {
    config: VAE_CONFIG,
    checkpoint: VAE_CHECKPOINT,
};

/// `Wan-AI/Wan2.2-TI2V-5B-Diffusers`.
pub const WAN22_VAE: VaeFiles = VaeFiles {
    config: WAN22_VAE_CONFIG,
    checkpoint: WAN22_VAE_CHECKPOINT,
};

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
    /// Every tap at once through the CubeCL implicit-GEMM kernel (stride 1, GPU).
    #[cfg(feature = "gpu-runtime")]
    Kernel(Box<KernelConv>),
}

impl Taps {
    fn len(&self) -> usize {
        match self {
            Self::Spatial(taps) => taps.len(),
            Self::Pointwise(taps) => taps.len(),
            #[cfg(feature = "gpu-runtime")]
            Self::Kernel(conv) => conv.shape.time_kernel,
        }
    }

    fn forward(&self, k: usize, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Spatial(taps) => taps[k].forward(x),
            Self::Pointwise(taps) => taps[k].forward(x),
            #[cfg(feature = "gpu-runtime")]
            Self::Kernel(_) => unreachable!("kernel convolutions run all taps at once"),
        }
    }
}

/// A stride-1 convolution on the CubeCL kernel: `[K, C_out]` weight, bias, geometry.
#[cfg(feature = "gpu-runtime")]
#[derive(Debug, Clone)]
struct KernelConv {
    weight: Tensor<2>,
    bias: Tensor<1>,
    shape: ConvShape,
}

#[cfg(feature = "gpu-runtime")]
impl KernelConv {
    /// From a PyTorch `[C_out, C_in, k_t, k_s, k_s]` (or `[C_out, C_in, k_s, k_s]`) weight.
    fn new(weight: &[f32], bias: Tensor<1>, shape: ConvShape, device: &Device) -> Self {
        let matrix = kernels::conv_weight(weight, &shape);
        let weight = Tensor::from_data(
            TensorData::new(matrix, [shape.depth(), shape.weight_stride()]),
            device,
        );
        Self { weight, bias, shape }
    }

    fn forward(&self, x: Tensor<4>, zeros: usize, residual: Option<Tensor<4>>) -> Tensor<4> {
        kernels::causal_conv3d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            residual,
            zeros,
            self.shape,
        )
    }
}

/// Whether convolutions on `device` run the CubeCL kernels (any GPU in a GPU build).
fn use_kernels(device: &Device) -> bool {
    cfg!(feature = "gpu-runtime") && *device != Device::flex()
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
        #[cfg(feature = "gpu-runtime")]
        if stride == 1 && 2 * p_s + 1 == k_s && use_kernels(device) {
            let shape = ConvShape {
                in_channels: input,
                out_channels: output,
                time_kernel: k_t,
                kernel: k_s,
                padding: p_s,
                upsample: false,
            };
            return Ok(Self {
                taps: Taps::Kernel(Box::new(KernelConv::new(&weight.data, bias, shape, device))),
                stride,
                pad: 2 * p_t,
            });
        }
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
        self.forward_with(x, cache, None)
    }

    /// [`Self::forward`] plus `residual` (fused into the kernel's epilogue on the GPU).
    fn forward_with(
        &self,
        x: Tensor<4>,
        cache: Option<&Tensor<4>>,
        residual: Option<Tensor<4>>,
    ) -> Tensor<4> {
        #[cfg(feature = "gpu-runtime")]
        if let Taps::Kernel(conv) = &self.taps {
            let (x, zeros) = match cache {
                Some(prefix) if self.pad > 0 => {
                    let zeros = self.pad.saturating_sub(prefix.dims()[0]);
                    (Tensor::cat(vec![prefix.clone(), x], 0), zeros)
                }
                _ => (x, self.pad),
            };
            return conv.forward(x, zeros, residual);
        }
        let out = self.forward_taps(x, cache);
        match residual {
            Some(residual) => out + residual,
            None => out,
        }
    }

    /// The per-time-tap sum of frame convolutions.
    fn forward_taps(&self, x: Tensor<4>, cache: Option<&Tensor<4>>) -> Tensor<4> {
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
        self.forward_cached_with(x, cache, None)
    }

    fn forward_cached_with(
        &self,
        x: Tensor<4>,
        cache: &mut TimeCache,
        residual: Option<Tensor<4>>,
    ) -> Tensor<4> {
        let slot = cache.next();
        let tail = cache_tail(&x, slot.frames());
        let out = self.forward_with(x, slot.frames(), residual);
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
        self.conv2
            .forward_cached_with(silu(self.norm2.forward(x)), cache, Some(h))
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
        conv: UpConv,
        time: Option<CausalConv3d>,
    },
}

/// The up path's 3x3 convolution of the 2x nearest-upsampled frames.
#[derive(Debug, Clone)]
enum UpConv {
    Plain(Conv2d),
    /// Reads the input through the upsampling, so the 4x frames are never materialized.
    #[cfg(feature = "gpu-runtime")]
    Kernel(KernelConv),
}

impl UpConv {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Plain(conv) => conv.forward(upsample_nearest2(x)),
            #[cfg(feature = "gpu-runtime")]
            Self::Kernel(conv) => conv.forward(x, 0, None),
        }
    }
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

    /// `upsample2d` / `upsample3d` from `dim` to `out` channels (`dim / 2` in Wan 2.1).
    fn load_up(
        state: &mut StateDict,
        prefix: &str,
        (dim, out): (usize, usize),
        temporal: bool,
        device: &Device,
    ) -> Result<Self> {
        let name = format!("{prefix}.resample.1");
        let conv = if use_kernels(device) {
            up_kernel(state, &name, (dim, out), device)?
        } else {
            UpConv::Plain(Conv2d::load(
                state,
                &name,
                Conv2dSpec::new(dim, out, 3).padding(1),
                device,
            )?)
        };
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
                conv.forward(x)
            }
        }
    }
}

#[cfg(feature = "gpu-runtime")]
fn up_kernel(
    state: &mut StateDict,
    name: &str,
    (dim, out): (usize, usize),
    device: &Device,
) -> Result<UpConv> {
    let shape = ConvShape {
        in_channels: dim,
        out_channels: out,
        time_kernel: 1,
        kernel: 3,
        padding: 1,
        upsample: true,
    };
    let weight = state.take(&format!("{name}.weight"), &[out, dim, 3, 3])?;
    let bias = state.tensor(&format!("{name}.bias"), [out], device)?;
    Ok(UpConv::Kernel(KernelConv::new(&weight.data, bias, shape, device)))
}

#[cfg(not(feature = "gpu-runtime"))]
fn up_kernel(_: &mut StateDict, _: &str, _: (usize, usize), _: &Device) -> Result<UpConv> {
    unreachable!("kernels are off without gpu-runtime")
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

/// `AvgDown3D`: `time x space x space` blocks folded into channels, then averaged over
/// groups of consecutive channels (front-padding time with zeros to a whole block).
#[derive(Debug, Clone, Copy)]
struct AvgDown {
    out: usize,
    time: usize,
    space: usize,
}

impl AvgDown {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [t, c, h, w] = x.dims();
        let (ft, fs, out) = (self.time, self.space, self.out);
        if ft * fs == 1 && c == out {
            return x;
        }
        let pad = (ft - t % ft) % ft;
        let x = if pad == 0 {
            x
        } else {
            Tensor::cat(vec![Tensor::zeros([pad, c, h, w], &x.device()), x], 0)
        };
        let (t, h, w) = ((t + pad) / ft, h / fs, w / fs);
        let group = c * ft * fs * fs / out;
        // `[t, ft, c, h, fs, w, fs]` -> `[t, c, ft, fs, fs, h, w]`: channel `c·f + a·fs² + b·fs + d`
        x.reshape([t, ft, c, h, fs, w, fs])
            .permute([0, 2, 1, 4, 6, 3, 5])
            .reshape([t, out, group, h * w])
            .mean_dim(2)
            .reshape([t, out, h, w])
    }
}

/// `DupUp3D`: channels repeated, then unfolded into `time x space x space` blocks; the first
/// chunk drops the frames before its first.
#[derive(Debug, Clone, Copy)]
struct DupUp {
    out: usize,
    time: usize,
    space: usize,
    repeats: usize,
}

impl DupUp {
    fn new(input: usize, out: usize, time: usize, space: usize) -> Self {
        Self {
            out,
            time,
            space,
            repeats: out * time * space * space / input,
        }
    }

    fn forward(&self, x: Tensor<4>, first_chunk: bool) -> Tensor<4> {
        let [t, _, h, w] = x.dims();
        let (ft, fs, out) = (self.time, self.space, self.out);
        // `repeat_interleave`, then `[t, out, ft, fs, fs, h, w]` -> `[t, ft, out, h, fs, w, fs]`
        let y = x
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, self.repeats)
            .reshape([t, out, ft, fs, fs, h, w])
            .permute([0, 2, 1, 5, 3, 6, 4])
            .reshape([t * ft, out, h * fs, w * fs]);
        if first_chunk && ft > 1 {
            y.slice(s![ft - 1..])
        } else {
            y
        }
    }
}

/// Wan 2.2's parameter-free path around a whole level.
#[derive(Debug, Clone, Copy)]
enum Shortcut {
    Down(AvgDown),
    Up(DupUp),
}

/// One encoder level or decoder up block: residual blocks, an optional resampler and (Wan
/// 2.2's `WanResidualDownBlock` / `WanResidualUpBlock`) a shortcut around both.
#[derive(Debug, Clone)]
struct Stage {
    resnets: Vec<ResidualBlock>,
    resample: Option<Resample>,
    shortcut: Option<Shortcut>,
}

impl Stage {
    fn forward(&self, x: Tensor<4>, cache: &mut TimeCache, first_chunk: bool) -> Tensor<4> {
        let input = self.shortcut.is_some().then(|| x.clone());
        let x = self.resnets.iter().fold(x, |x, resnet| resnet.forward(x, cache));
        let x = match &self.resample {
            Some(resample) => resample.forward(x, cache),
            None => x,
        };
        match (self.shortcut, input) {
            (Some(Shortcut::Down(down)), Some(input)) => x + down.forward(input),
            (Some(Shortcut::Up(up)), Some(input)) => x + up.forward(input, first_chunk),
            _ => x,
        }
    }
}

/// `count` residual blocks from `input` to `output` channels, the `j`-th under `name(j)`.
fn load_resnets(
    state: &mut StateDict,
    name: impl Fn(usize) -> String,
    (input, output): (usize, usize),
    count: usize,
    device: &Device,
) -> Result<Vec<ResidualBlock>> {
    (0..count)
        .map(|j| {
            ResidualBlock::load(
                state,
                &name(j),
                if j == 0 { input } else { output },
                output,
                device,
            )
        })
        .collect()
}

/// `2` when the flag is set, else `1`.
const fn factor(flag: bool) -> usize {
    if flag { 2 } else { 1 }
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
    fn load(state: &mut StateDict, config: &VaeConfig, device: &Device) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(1)
            .chain(config.dim_mult.iter().copied())
            .map(|m| m * config.base_dim)
            .collect();
        let conv_in = CausalConv3d::cube(state, "encoder.conv_in", config.in_channels, dims[0], device)?;
        let levels = config.dim_mult.len();
        let mut stages = Vec::new();
        // Wan 2.1's `down_blocks` is one flat list: resnets, then the level's resampler
        let mut index = 0;
        for (i, pair) in dims.windows(2).enumerate() {
            let (input, output) = (pair[0], pair[1]);
            let down = i + 1 < levels;
            let temporal = down && config.time_downsample[i];
            let stage = if config.is_residual {
                let prefix = format!("encoder.down_blocks.{i}");
                let resnets = load_resnets(
                    state,
                    |j| format!("{prefix}.resnets.{j}"),
                    (input, output),
                    config.num_res_blocks,
                    device,
                )?;
                let resample = down
                    .then(|| {
                        Resample::load_down(state, &format!("{prefix}.downsampler"), output, temporal, device)
                    })
                    .transpose()?;
                let shortcut = AvgDown {
                    out: output,
                    time: factor(temporal),
                    space: factor(down),
                };
                Stage {
                    resnets,
                    resample,
                    shortcut: Some(Shortcut::Down(shortcut)),
                }
            } else {
                let resnets = load_resnets(
                    state,
                    |j| format!("encoder.down_blocks.{}", index + j),
                    (input, output),
                    config.num_res_blocks,
                    device,
                )?;
                index += config.num_res_blocks;
                let resample = if down {
                    let prefix = format!("encoder.down_blocks.{index}");
                    index += 1;
                    Some(Resample::load_down(state, &prefix, output, temporal, device)?)
                } else {
                    None
                };
                Stage {
                    resnets,
                    resample,
                    shortcut: None,
                }
            };
            stages.push(stage);
        }
        let top = dims[dims.len() - 1];
        Ok(Self {
            conv_in,
            stages,
            mid: MidBlock::load(state, "encoder.mid_block", top, device)?,
            norm_out: ChannelRmsNorm::load(state, "encoder.norm_out", top, false, device)?,
            conv_out: CausalConv3d::cube(state, "encoder.conv_out", top, 2 * config.z_dim, device)?,
        })
    }

    fn forward(&self, x: Tensor<4>, cache: &mut TimeCache) -> Tensor<4> {
        let x = self.conv_in.forward_cached(x, cache);
        let x = self
            .stages
            .iter()
            .fold(x, |x, stage| stage.forward(x, cache, false));
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
    fn load(state: &mut StateDict, config: &VaeConfig, device: &Device) -> Result<Self> {
        let mult = &config.dim_mult;
        let dims: Vec<usize> = std::iter::once(mult[mult.len() - 1])
            .chain(mult.iter().rev().copied())
            .map(|m| m * config.decoder_dim())
            .collect();
        let conv_in = CausalConv3d::cube(state, "decoder.conv_in", config.z_dim, dims[0], device)?;
        let mid = MidBlock::load(state, "decoder.mid_block", dims[0], device)?;
        let mut blocks = Vec::new();
        for (i, pair) in dims.windows(2).enumerate() {
            let output = pair[1];
            // Wan 2.1 halves the channels in each upsampler and so feeds half to the next block
            let input = if i > 0 && !config.is_residual {
                pair[0] / 2
            } else {
                pair[0]
            };
            let up = i + 1 < mult.len();
            let temporal = up && config.time_downsample[config.time_downsample.len() - 1 - i];
            let prefix = format!("decoder.up_blocks.{i}");
            let resnets = load_resnets(
                state,
                |j| format!("{prefix}.resnets.{j}"),
                (input, output),
                config.num_res_blocks + 1,
                device,
            )?;
            let (name, upsampled) = if config.is_residual {
                ("upsampler", output)
            } else {
                ("upsamplers.0", output / 2)
            };
            let resample = up
                .then(|| {
                    let prefix = format!("{prefix}.{name}");
                    Resample::load_up(state, &prefix, (output, upsampled), temporal, device)
                })
                .transpose()?;
            let shortcut = (config.is_residual && up)
                .then(|| Shortcut::Up(DupUp::new(input, output, factor(temporal), 2)));
            blocks.push(Stage {
                resnets,
                resample,
                shortcut,
            });
        }
        let out = dims[dims.len() - 1];
        Ok(Self {
            conv_in,
            mid,
            blocks,
            norm_out: ChannelRmsNorm::load(state, "decoder.norm_out", out, false, device)?,
            conv_out: CausalConv3d::cube(state, "decoder.conv_out", out, config.out_channels, device)?,
        })
    }

    fn forward(&self, x: Tensor<4>, cache: &mut TimeCache, first_chunk: bool) -> Tensor<4> {
        let x = self.mid.forward(self.conv_in.forward_cached(x, cache), cache);
        let x = self
            .blocks
            .iter()
            .fold(x, |x, stage| stage.forward(x, cache, first_chunk));
        self.conv_out
            .forward_cached(silu(self.norm_out.forward(x)), cache)
    }
}

/// `patchify`: `[T, C, H, W]` -> `[T, C·p², H / p, W / p]`, channel `c·p² + pw·p + ph`.
fn patchify(x: Tensor<4>, p: usize) -> Tensor<4> {
    if p == 1 {
        return x;
    }
    let [t, c, h, w] = x.dims();
    x.reshape([t, c, h / p, p, w / p, p])
        .permute([0, 1, 5, 3, 2, 4])
        .reshape([t, c * p * p, h / p, w / p])
}

/// `unpatchify`, the inverse of [`patchify`].
fn unpatchify(x: Tensor<4>, p: usize) -> Tensor<4> {
    if p == 1 {
        return x;
    }
    let [t, c, h, w] = x.dims();
    let c = c / (p * p);
    x.reshape([t, c, p, p, h, w])
        .permute([0, 1, 4, 3, 5, 2])
        .reshape([t, c, h * p, w * p])
}

/// `AutoencoderKLWan` from a Wan 2.1 or Wan 2.2 Diffusers checkpoint (`vae/`).
#[derive(Debug, Clone)]
pub struct WanVae {
    config: VaeConfig,
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: CausalConv3d,
    post_quant_conv: CausalConv3d,
    tiling: bool,
    device: ComputeDevice,
}

impl WanVae {
    /// Resolve the pinned Wan 2.1 `vae/` through the hub cache (downloading if allowed).
    pub fn load(device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(&Hub::from_env(), Fetch::Download, device)
    }

    pub fn load_from_hub(hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        Self::load_files(hub, fetch, &WAN21_VAE, device)
    }

    /// The pinned VAE `files` of any supported checkpoint.
    pub fn load_files(hub: &Hub, fetch: Fetch, files: &VaeFiles, device: &ComputeDevice) -> Result<Self> {
        let (config, checkpoint) = Self::resolve_files(hub, fetch, files)?;
        Self::from_checkpoint(config, &checkpoint, device)
    }

    /// The verified config and local checkpoint path of pinned `files`, for later
    /// [`Self::from_checkpoint`] reloads.
    pub fn resolve_files(hub: &Hub, fetch: Fetch, files: &VaeFiles) -> Result<(VaeConfig, PathBuf)> {
        let config = hub.resolve(&files.config, fetch)?;
        let checkpoint = hub.resolve(&files.checkpoint, fetch)?;
        if config.parent() != checkpoint.parent() {
            return Err(GpuError::Config(
                "pinned Wan VAE artifacts must share one snapshot directory".to_owned(),
            ));
        }
        Ok((VaeConfig::from_file(&config)?, checkpoint))
    }

    /// Load a local `vae/diffusion_pytorch_model.safetensors` without digest checks.
    pub fn from_checkpoint(config: VaeConfig, path: &Path, device: &ComputeDevice) -> Result<Self> {
        let mut state = StateDict::from_safetensors(path, &[])?;
        let burn = device.burn();
        let encoder = Encoder::load(&mut state, &config, burn)?;
        let decoder = Decoder::load(&mut state, &config, burn)?;
        let (z, z2) = (config.z_dim, 2 * config.z_dim);
        let quant_conv = CausalConv3d::pointwise(&mut state, "quant_conv", z2, z2, burn)?;
        let post_quant_conv = CausalConv3d::pointwise(&mut state, "post_quant_conv", z, z, burn)?;
        state.finish()?;
        Ok(Self {
            // diffusers leaves tiling off; `video_wan.py` enables it for Wan 2.1
            tiling: config.patch() == 1,
            config,
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// Latent channels (`z_dim`).
    pub fn latent_channels(&self) -> usize {
        self.config.z_dim
    }

    /// Pixels per latent along each side.
    pub fn spatial_factor(&self) -> usize {
        self.config.spatial_factor()
    }

    /// `enable_tiling()` (on by default for Wan 2.1, as `video_wan.py` loads it; Wan 2.2's
    /// patchified tiling is not ported).
    #[must_use]
    pub fn with_tiling(mut self, tiling: bool) -> Self {
        self.tiling = tiling;
        self
    }

    pub fn set_tiling(&mut self, tiling: bool) {
        self.tiling = tiling;
    }

    /// Whether a `(height, width)` clip runs tiled, `limit` being the untiled side limit.
    fn tiled(&self, (height, width): (usize, usize), limit: usize) -> Result<bool> {
        if !self.tiling || (height <= limit && width <= limit) {
            return Ok(false);
        }
        if self.config.patch() > 1 {
            return Err(GpuError::Config(
                "tiling is not ported for the patchified (Wan 2.2) VAE".to_owned(),
            ));
        }
        Ok(true)
    }

    /// `encode(video).latent_dist.mode()` of `1 + 4k` RGB frames in `[-1, 1]`, `[T, 3, H, W]`.
    ///
    /// Returns the unnormalized latent means as `[z, 1 + k, H / f, W / f]` (`f` the
    /// [`Self::spatial_factor`]).
    pub fn encode(&self, frames: &[f32], count: usize, height: usize, width: usize) -> Result<Vec<f32>> {
        let (f, z) = (self.spatial_factor(), self.latent_channels());
        check_clip(frames.len(), count, 3, height, width, f)?;
        if !(count - 1).is_multiple_of(TEMPORAL_FACTOR) {
            return Err(GpuError::Input(format!(
                "Wan VAE encodes 1 + 4k frames, got {count}"
            )));
        }
        let input = Plane::new(count * 3, height, width, frames.to_vec());
        let moments = if self.tiled((height, width), TILE_SAMPLE)? {
            run_tiled(&input, ENCODE_TILING, |tile| self.encode_plane(tile, count))?
        } else {
            self.encode_plane(&input, count)?
        };
        let latent_frames = 1 + (count - 1) / TEMPORAL_FACTOR;
        let (h, w) = (height / f, width / f);
        check_assembled(&moments, latent_frames * z, h, w)?;
        Ok(swap_time_channels(&moments.data, latent_frames, z, h * w))
    }

    /// `decode(z).sample` of unnormalized latents `[z, T, h, w]`, clamped to `[-1, 1]`.
    ///
    /// Returns frames `[1 + 4(T - 1), 3, f·h, f·w]`.
    pub fn decode(
        &self,
        latents: &[f32],
        latent_frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        let (f, z) = (self.spatial_factor(), self.latent_channels());
        check_clip(latents.len(), latent_frames, z, height, width, 1)?;
        let time_major = swap_time_channels(latents, z, latent_frames, height * width);
        let input = Plane::new(latent_frames * z, height, width, time_major);
        let mut decoded = if self.tiled((height, width), TILE_LATENT)? {
            run_tiled(&input, DECODE_TILING, |tile| {
                self.decode_plane(tile, latent_frames)
            })?
        } else {
            self.decode_plane(&input, latent_frames)?
        };
        let frames = 1 + TEMPORAL_FACTOR * (latent_frames - 1);
        check_assembled(&decoded, frames * 3, height * f, width * f)?;
        for value in &mut decoded.data {
            *value = value.clamp(-1.0, 1.0);
        }
        Ok(decoded.data)
    }

    /// Encoder + `quant_conv` means of one (tile of a) clip, time-major `[T', z, h, w]`.
    fn encode_plane(&self, input: &Plane, count: usize) -> Result<Plane> {
        let (h, w) = (input.height, input.width);
        let (f, z) = (self.spatial_factor(), self.latent_channels());
        let clip: Tensor<4> = Tensor::from_data(
            TensorData::new(input.data.clone(), [count, 3, h, w]),
            self.device.burn(),
        );
        let clip = patchify(clip, self.config.patch());
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
            data.extend(read_finite(moments.slice(s![.., 0..z]))?);
            start = end;
        }
        let latent_frames = 1 + (count - 1) / TEMPORAL_FACTOR;
        Ok(Plane::new(latent_frames * z, h / f, w / f, data))
    }

    /// `post_quant_conv` + decoder of one (tile of a) latent clip, time-major `[T, 3, H, W]`.
    fn decode_plane(&self, input: &Plane, latent_frames: usize) -> Result<Plane> {
        let (h, w) = (input.height, input.width);
        let (f, patch) = (self.spatial_factor(), self.config.patch());
        let z: Tensor<4> = Tensor::from_data(
            TensorData::new(input.data.clone(), [latent_frames, self.latent_channels(), h, w]),
            self.device.burn(),
        );
        let z = self.post_quant_conv.forward(z, None);
        let mut cache = TimeCache::default();
        let mut data = Vec::new();
        for i in 0..latent_frames {
            cache.rewind();
            let frames = self
                .decoder
                .forward(z.clone().slice(s![i..i + 1]), &mut cache, i == 0);
            data.extend(read_finite(unpatchify(frames, patch))?);
        }
        let frames = 1 + TEMPORAL_FACTOR * (latent_frames - 1);
        Ok(Plane::new(frames * 3, h * f, w * f, data))
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
    fn patchify_orders_channels_like_diffusers() {
        let device = Device::flex();
        // One 2x2 frame [[a, b], [c, d]] -> channels `pw·2 + ph`: a, c, b, d
        let x = Tensor::<4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);
        let packed = patchify(x.clone(), 2);
        assert_eq!(packed.dims(), [1, 4, 1, 1]);
        assert_eq!(read_f32(packed.clone()).unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(read_f32(unpatchify(packed, 2)).unwrap(), read_f32(x).unwrap());
    }

    #[test]
    fn avg_down_pads_time_and_averages_channel_groups() {
        let device = Device::flex();
        let x = Tensor::<4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);
        // One frame, front-padded to a time pair: group 0 holds the zero frame, group 1 the real one
        let down = AvgDown {
            out: 2,
            time: 2,
            space: 2,
        };
        let y = down.forward(x);
        assert_eq!(y.dims(), [1, 2, 1, 1]);
        assert_eq!(read_f32(y).unwrap(), vec![0.0, 2.5]);
    }

    #[test]
    fn dup_up_spreads_repeated_channels() {
        let device = Device::flex();
        // 2 -> 1 channels over 2x2 space: rows come from channels 0 and 1
        let x = Tensor::<4>::from_floats([[[[5.0]], [[7.0]]]], &device);
        let y = DupUp::new(2, 1, 1, 2).forward(x.clone(), true);
        assert_eq!(y.dims(), [1, 1, 2, 2]);
        assert_eq!(read_f32(y).unwrap(), vec![5.0, 5.0, 7.0, 7.0]);
        // Time doubling drops the leading frame of the first chunk only
        let up = DupUp::new(2, 2, 2, 2);
        assert_eq!(up.forward(x.clone(), false).dims(), [2, 2, 2, 2]);
        let first = up.forward(x, true);
        assert_eq!(first.dims(), [1, 2, 2, 2]);
        assert_eq!(read_f32(first).unwrap(), [vec![5.0; 4], vec![7.0; 4]].concat());
    }

    #[test]
    fn time_channel_swap_round_trips() {
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let swapped = swap_time_channels(&data, 2, 3, 4);
        assert_eq!(swapped[4..8], [12.0, 13.0, 14.0, 15.0]);
        assert_eq!(swap_time_channels(&swapped, 3, 2, 4), data);
    }
}

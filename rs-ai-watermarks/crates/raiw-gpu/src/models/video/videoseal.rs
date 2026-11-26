//! VideoSeal / PixelSeal / ChunkySeal message detectors (`video message`).
//!
//! Mirrors `_internal/video_message_models.py`: RGB frames in `[0, 1]` are resized to 256x256
//! with antialiased bilinear filtering when needed, mapped to `[-1, 1]`, and run through a
//! ConvNeXtV2 extractor plus pixel decoder whose mean-pooled embedding yields `1 + nbits`
//! logits per frame (presence first, then message bits). VideoSeal weights come from the
//! TorchScript graph, PixelSeal from its checkpoint's `detector.*` tensors.

use std::path::Path;

use burn::tensor::activation::gelu;
use burn::tensor::ops::PadMode;
use burn::tensor::{Device, Tensor, TensorData};

use crate::device::ComputeDevice;
use crate::error::{GpuError, Result};
use crate::hub::{Fetch, Hub};
use crate::io::read_finite;
use crate::layers::{Conv2d, Conv2dSpec};
use crate::weights::StateDict;

pub const SOURCE_COMMIT: &str = "870ca7fb33578b90f14c602016b6c2788096226e";
pub const IMAGE_SIZE: usize = 256;
const CHUNK_FRAMES: usize = 16;
const NORM_EPS: f64 = 1e-6;
const GRN_EPS: f64 = 1e-6;

/// Which public detector to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    VideoSeal,
    PixelSeal,
    ChunkySeal,
}

/// A pinned public asset: URL, SHA-256 and the cache file name Python uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Asset {
    pub url: &'static str,
    pub sha256: &'static str,
    pub cache_path: &'static str,
}

pub const VIDEOSEAL_GRAPH: Asset = Asset {
    url: "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit",
    sha256: "5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b",
    cache_path: "video-models/videoseal-5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b.jit",
};
pub const PIXELSEAL_CHECKPOINT: Asset = Asset {
    url: "https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth",
    sha256: "0c5665cff20eb6ce1b5aaa7d91c19dafb418bfee32d02dd3344e4ed60d9d75bd",
    cache_path: "video-models/pixelseal-0c5665cff20eb6ce1b5aaa7d91c19dafb418bfee32d02dd3344e4ed60d9d75bd.pth",
};

impl Scheme {
    /// Message bits after the presence logit.
    pub const fn bits(self) -> usize {
        match self {
            Self::VideoSeal | Self::PixelSeal => 256,
            Self::ChunkySeal => 1024,
        }
    }

    fn config(self) -> ConvNextConfig {
        match self {
            Self::VideoSeal | Self::PixelSeal => ConvNextConfig {
                dims: [96, 192, 384, 768],
                depths: [3, 3, 9, 3],
                stem_stride: 4,
            },
            Self::ChunkySeal => ConvNextConfig {
                dims: [362, 724, 1448, 2896],
                depths: [3, 3, 27, 3],
                stem_stride: 2,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ConvNextConfig {
    dims: [usize; 4],
    depths: [usize; 4],
    /// The stem is always a 4x4 patch conv; ChunkySeal strides it by 2 instead of 4.
    stem_stride: usize,
}

const STEM_KERNEL: usize = 4;

/// `LayerNorm(data_format="channels_first")` over `[N, C, H, W]`.
#[derive(Debug, Clone)]
struct ChannelNorm {
    weight: Tensor<4>,
    bias: Tensor<4>,
}

impl ChannelNorm {
    fn load(state: &mut StateDict, prefix: &str, channels: usize, device: &Device) -> Result<Self> {
        let weight: Tensor<1> = state.tensor(&format!("{prefix}.weight"), [channels], device)?;
        let bias: Tensor<1> = state.tensor(&format!("{prefix}.bias"), [channels], device)?;
        Ok(Self {
            weight: weight.reshape([1, channels, 1, 1]),
            bias: bias.reshape([1, channels, 1, 1]),
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let mean = x.clone().mean_dim(1);
        let centered = x - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(1);
        centered / var.add_scalar(NORM_EPS).sqrt() * self.weight.clone() + self.bias.clone()
    }
}

/// `nn.Linear` applied over the last axis of `[N, H, W, C]`.
#[derive(Debug, Clone)]
struct Linear {
    weight_t: Tensor<3>,
    bias: Tensor<3>,
}

impl Linear {
    fn load(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<Self> {
        let weight: Tensor<2> = state.tensor(&format!("{prefix}.weight"), [output, input], device)?;
        let bias: Tensor<1> = state.tensor(&format!("{prefix}.bias"), [output], device)?;
        Ok(Self {
            weight_t: weight.transpose().unsqueeze_dim(0),
            bias: bias.reshape([1, 1, output]),
        })
    }

    /// One `[N*H*W, C]` matmul; per-row batches would leave most GPU matmul tiles empty.
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [n, h, w, c] = x.dims();
        let output = self.bias.dims()[2];
        (x.reshape([1, n * h * w, c]).matmul(self.weight_t.clone()) + self.bias.clone())
            .reshape([n, h, w, output])
    }
}

/// ConvNeXtV2 block: depthwise 7x7, LayerNorm, MLP with GELU and GRN, residual.
#[derive(Debug, Clone)]
struct Block {
    dwconv: Conv2d,
    norm_w: Tensor<4>,
    norm_b: Tensor<4>,
    pwconv1: Linear,
    gamma: Tensor<4>,
    beta: Tensor<4>,
    pwconv2: Linear,
}

impl Block {
    fn load(state: &mut StateDict, prefix: &str, dim: usize, device: &Device) -> Result<Self> {
        let norm_w: Tensor<1> = state.tensor(&format!("{prefix}.norm.weight"), [dim], device)?;
        let norm_b: Tensor<1> = state.tensor(&format!("{prefix}.norm.bias"), [dim], device)?;
        Ok(Self {
            dwconv: Conv2d::load(
                state,
                &format!("{prefix}.dwconv"),
                Conv2dSpec::new(dim, dim, 7).padding(3).groups(dim),
                device,
            )?,
            norm_w: norm_w.reshape([1, 1, 1, dim]),
            norm_b: norm_b.reshape([1, 1, 1, dim]),
            pwconv1: Linear::load(state, &format!("{prefix}.pwconv1"), dim, 4 * dim, device)?,
            gamma: state.tensor(&format!("{prefix}.grn.gamma"), [1, 1, 1, 4 * dim], device)?,
            beta: state.tensor(&format!("{prefix}.grn.beta"), [1, 1, 1, 4 * dim], device)?,
            pwconv2: Linear::load(state, &format!("{prefix}.pwconv2"), 4 * dim, dim, device)?,
        })
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let h = self.dwconv.forward(x.clone()).permute([0, 2, 3, 1]);
        let mean = h.clone().mean_dim(3);
        let centered = h - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(3);
        let h = centered / var.add_scalar(NORM_EPS).sqrt() * self.norm_w.clone() + self.norm_b.clone();
        let h = gelu(self.pwconv1.forward(h));
        let gx = h.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1).sqrt();
        let nx = gx.clone() / gx.mean_dim(3).add_scalar(GRN_EPS);
        let h = self.gamma.clone() * (h.clone() * nx) + self.beta.clone() + h;
        x + self.pwconv2.forward(h).permute([0, 3, 1, 2])
    }
}

#[derive(Debug, Clone)]
struct Stage {
    /// Stem (conv then norm) for the first stage, norm then 2x2 conv afterwards.
    norm: ChannelNorm,
    conv: Conv2d,
    norm_first: bool,
    blocks: Vec<Block>,
}

#[derive(Debug, Clone)]
pub struct VideoMessageDetector {
    scheme: Scheme,
    stages: Vec<Stage>,
    head_conv: Conv2d,
    head_norm: ChannelNorm,
    linear_w: Tensor<2>,
    linear_b: Tensor<1>,
    device: ComputeDevice,
}

impl VideoMessageDetector {
    /// Resolve the pinned public assets in the model cache (downloading if allowed) and load them.
    pub fn load(scheme: Scheme, device: &ComputeDevice) -> Result<Self> {
        Self::load_from_hub(scheme, &Hub::from_env(), Fetch::Download, device)
    }

    pub fn load_from_hub(scheme: Scheme, hub: &Hub, fetch: Fetch, device: &ComputeDevice) -> Result<Self> {
        let resolve = |asset: &Asset| hub.resolve_url(asset.url, asset.sha256, asset.cache_path, fetch);
        match scheme {
            Scheme::VideoSeal => Self::from_videoseal_graph(&resolve(&VIDEOSEAL_GRAPH)?, device),
            Scheme::PixelSeal => Self::from_pixelseal_checkpoint(&resolve(&PIXELSEAL_CHECKPOINT)?, device),
            // The 3.1 GB detector slice is cut from a 13 GB checkpoint; only reuse the cached one
            Scheme::ChunkySeal => {
                let path = raiw_core::model_cache_dir().join(CHUNKYSEAL_SLICE);
                if !path.is_file() {
                    return Err(GpuError::NotCached {
                        repo: "https://dl.fbaipublicfiles.com/videoseal/chunkyseal/checkpoint.pth".to_owned(),
                        revision: CHUNKYSEAL_SHA256.to_owned(),
                        filename: CHUNKYSEAL_SLICE.to_owned(),
                    });
                }
                if raiw_core::sha256_file(&path).map_err(|err| GpuError::io(&path, err))? != CHUNKYSEAL_SHA256
                {
                    return Err(GpuError::DigestMismatch {
                        filename: CHUNKYSEAL_SLICE.to_owned(),
                        actual: "cached slice".to_owned(),
                        expected: CHUNKYSEAL_SHA256.to_owned(),
                    });
                }
                Self::from_chunkyseal_slice(&path, device)
            }
        }
    }

    /// Detector tensors (`detector.*`) of the VideoSeal TorchScript graph.
    pub fn from_videoseal_graph(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let state = StateDict::from_torch_zip(path, Some("detector"), &[], |_| true)?;
        Self::from_state(Scheme::VideoSeal, state, device)
    }

    /// `model["detector.*"]` of the PixelSeal training checkpoint.
    pub fn from_pixelseal_checkpoint(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let state = StateDict::from_torch_zip(path, Some("model"), &[("^detector\\.", "")], |name| {
            name.starts_with("detector.")
        })?;
        Self::from_state(Scheme::PixelSeal, state, device)
    }

    /// The raw ZIP64 detector slice Python caches for ChunkySeal (`chunkyseal-<sha>.slice.pth`).
    ///
    /// Entries `checkpoint/data/228..` hold the detector parameters as stored little-endian f32
    /// in `state_dict()` order; names, sizes and CRCs are verified without unpickling anything.
    pub fn from_chunkyseal_slice(path: &Path, device: &ComputeDevice) -> Result<Self> {
        let state = read_chunkyseal_slice(path, &parameter_layout(Scheme::ChunkySeal))?;
        Self::from_state(Scheme::ChunkySeal, state, device)
    }

    /// Build from detector tensors named like the PyTorch module (`convnext.*`, `pixel_decoder.*`).
    pub fn from_state(scheme: Scheme, mut state: StateDict, device: &ComputeDevice) -> Result<Self> {
        let config = scheme.config();
        let burn = device.burn();
        let mut stages = Vec::with_capacity(4);
        for (i, (&dim, &depth)) in config.dims.iter().zip(&config.depths).enumerate() {
            let down = format!("convnext.downsample_layers.{i}");
            let (norm, conv, norm_first) = if i == 0 {
                let spec = Conv2dSpec::new(3, dim, STEM_KERNEL).stride(config.stem_stride);
                (
                    ChannelNorm::load(&mut state, &format!("{down}.1"), dim, burn)?,
                    Conv2d::load(&mut state, &format!("{down}.0"), spec, burn)?,
                    false,
                )
            } else {
                let previous = config.dims[i - 1];
                (
                    ChannelNorm::load(&mut state, &format!("{down}.0"), previous, burn)?,
                    Conv2d::load(
                        &mut state,
                        &format!("{down}.1"),
                        Conv2dSpec::new(previous, dim, 2).stride(2),
                        burn,
                    )?,
                    true,
                )
            };
            let blocks = (0..depth)
                .map(|b| Block::load(&mut state, &format!("convnext.stages.{i}.{b}"), dim, burn))
                .collect::<Result<Vec<_>>>()?;
            stages.push(Stage {
                norm,
                conv,
                norm_first,
                blocks,
            });
        }
        let embed = config.dims[3];
        let block = "pixel_decoder.output_upscaling.0.upsample_block";
        let head_conv = Conv2d::load(
            &mut state,
            &format!("{block}.2"),
            Conv2dSpec::new(embed, embed, 3).no_bias(),
            burn,
        )?;
        let head_norm = ChannelNorm::load(&mut state, &format!("{block}.3"), embed, burn)?;
        let linear_w: Tensor<2> =
            state.tensor("pixel_decoder.linear.weight", [scheme.bits() + 1, embed], burn)?;
        let linear_b = state.tensor("pixel_decoder.linear.bias", [scheme.bits() + 1], burn)?;
        state.finish()?;
        Ok(Self {
            scheme,
            stages,
            head_conv,
            head_norm,
            linear_w: linear_w.transpose(),
            linear_b,
            device: device.clone(),
        })
    }

    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    pub fn device(&self) -> &ComputeDevice {
        &self.device
    }

    /// Raw logits `[frames][1 + bits]` (presence first) of RGB frames in `[0, 1]`, NHWC `[frames, h, w, 3]`.
    pub fn predict(
        &self,
        frames: &[f32],
        count: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if count == 0 || height == 0 || width == 0 || frames.len() != count * height * width * 3 {
            return Err(GpuError::Input(
                "video message frames must be nonempty TxHxWx3 RGB".to_owned(),
            ));
        }
        if !frames.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
            return Err(GpuError::Input(
                "video message frames must be finite RGB in [0, 1]".to_owned(),
            ));
        }
        let plane = height * width * 3;
        let mut out = Vec::with_capacity(count);
        for chunk in frames.chunks(CHUNK_FRAMES * plane) {
            let n = chunk.len() / plane;
            let nchw = resize_nhwc_to_nchw(chunk, n, height, width, IMAGE_SIZE);
            let x: Tensor<4> = Tensor::from_data(
                TensorData::new(nchw, [n, 3, IMAGE_SIZE, IMAGE_SIZE]),
                self.device.burn(),
            );
            let logits = read_finite(self.forward(x.mul_scalar(2.0).sub_scalar(1.0)))?;
            out.extend(logits.chunks_exact(self.scheme.bits() + 1).map(<[f32]>::to_vec));
        }
        Ok(out)
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let mut h = x;
        for stage in &self.stages {
            h = if stage.norm_first {
                stage.conv.forward(stage.norm.forward(h))
            } else {
                stage.norm.forward(stage.conv.forward(h))
            };
            h = stage.blocks.iter().fold(h, |h, block| block.forward(h));
        }
        let h = h.pad([(1, 1), (1, 1)], PadMode::Reflect);
        let h = gelu(self.head_norm.forward(self.head_conv.forward(h)));
        let [n, c, _, _] = h.dims();
        let pooled = h.mean_dim(3).mean_dim(2).reshape([n, c]);
        pooled.matmul(self.linear_w.clone()) + self.linear_b.clone().unsqueeze_dim(0)
    }
}

pub const CHUNKYSEAL_SHA256: &str = "a546154b6709abb31ca8c6d70ff1c88c9bd7ce973b95a4d2748ebf5746418ea7";
pub const CHUNKYSEAL_SLICE: &str =
    "video-models/chunkyseal-a546154b6709abb31ca8c6d70ff1c88c9bd7ce973b95a4d2748ebf5746418ea7.slice.pth";
const CHUNKYSEAL_FIRST_STORAGE_KEY: usize = 228;

/// Detector parameters in PyTorch `state_dict()` registration order.
pub fn parameter_layout(scheme: Scheme) -> Vec<(String, Vec<usize>)> {
    let config = scheme.config();
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |name: String, shape: &[usize]| out.push((name, shape.to_vec()));
    for (i, &dim) in config.dims.iter().enumerate() {
        let down = format!("convnext.downsample_layers.{i}");
        if i == 0 {
            push(format!("{down}.0.weight"), &[dim, 3, STEM_KERNEL, STEM_KERNEL]);
            push(format!("{down}.0.bias"), &[dim]);
            push(format!("{down}.1.weight"), &[dim]);
            push(format!("{down}.1.bias"), &[dim]);
        } else {
            let previous = config.dims[i - 1];
            push(format!("{down}.0.weight"), &[previous]);
            push(format!("{down}.0.bias"), &[previous]);
            push(format!("{down}.1.weight"), &[dim, previous, 2, 2]);
            push(format!("{down}.1.bias"), &[dim]);
        }
    }
    for (i, (&dim, &depth)) in config.dims.iter().zip(&config.depths).enumerate() {
        for b in 0..depth {
            let block = format!("convnext.stages.{i}.{b}");
            push(format!("{block}.dwconv.weight"), &[dim, 1, 7, 7]);
            push(format!("{block}.dwconv.bias"), &[dim]);
            push(format!("{block}.norm.weight"), &[dim]);
            push(format!("{block}.norm.bias"), &[dim]);
            push(format!("{block}.pwconv1.weight"), &[4 * dim, dim]);
            push(format!("{block}.pwconv1.bias"), &[4 * dim]);
            push(format!("{block}.grn.gamma"), &[1, 1, 1, 4 * dim]);
            push(format!("{block}.grn.beta"), &[1, 1, 1, 4 * dim]);
            push(format!("{block}.pwconv2.weight"), &[dim, 4 * dim]);
            push(format!("{block}.pwconv2.bias"), &[dim]);
        }
    }
    let embed = config.dims[3];
    let block = "pixel_decoder.output_upscaling.0.upsample_block";
    push(format!("{block}.2.weight"), &[embed, embed, 3, 3]);
    push(format!("{block}.3.weight"), &[embed]);
    push(format!("{block}.3.bias"), &[embed]);
    push(
        "pixel_decoder.linear.weight".to_owned(),
        &[scheme.bits() + 1, embed],
    );
    push("pixel_decoder.linear.bias".to_owned(), &[scheme.bits() + 1]);
    out
}

/// Walk the stored ZIP local entries of the detector slice, verifying names, sizes and CRCs.
fn read_chunkyseal_slice(path: &Path, layout: &[(String, Vec<usize>)]) -> Result<StateDict> {
    use std::io::Read;
    let fail = |reason: String| GpuError::Weights {
        path: path.to_path_buf(),
        reason,
    };
    let file = std::fs::File::open(path).map_err(|err| GpuError::io(path, err))?;
    let mut stream = std::io::BufReader::with_capacity(8 << 20, file);
    let mut read = |len: usize, what: &str| -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        stream
            .read_exact(&mut buf)
            .map_err(|err| fail(format!("truncated slice at {what}: {err}")))?;
        Ok(buf)
    };
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let mut tensors = std::collections::BTreeMap::new();
    for (index, (name, shape)) in layout.iter().enumerate() {
        let entry = format!("checkpoint/data/{}", CHUNKYSEAL_FIRST_STORAGE_KEY + index);
        let header = read(30, &entry)?;
        if header[..4] != *b"PK\x03\x04" || u16_at(&header, 8) != 0 {
            return Err(fail(format!("{entry}: not a stored ZIP entry")));
        }
        let flags = u16_at(&header, 6);
        let entry_name = read(usize::from(u16_at(&header, 26)), &entry)?;
        if entry_name != entry.as_bytes() {
            return Err(fail(format!("storage order mismatch at {entry}")));
        }
        read(usize::from(u16_at(&header, 28)), &entry)?;
        let count: usize = shape.iter().product();
        let payload = read(4 * count, &entry)?;
        let crc = crc32fast::hash(&payload);
        let (stored_crc, stored_len) = if flags & 0x0008 != 0 {
            let mut descriptor = read(16, &entry)?;
            if descriptor[..4] != *b"PK\x07\x08" {
                return Err(fail(format!("{entry}: missing data descriptor")));
            }
            let compact = u32_at(&descriptor, 8) as usize;
            if compact == payload.len() && u32_at(&descriptor, 12) as usize == payload.len() {
                (u32_at(&descriptor, 4), compact)
            } else {
                // Entries past the 4 GiB mark carry ZIP64 descriptors with 8-byte sizes
                descriptor.extend(read(8, &entry)?);
                let wide = |i: usize| {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&descriptor[i..i + 8]);
                    u64::from_le_bytes(b) as usize
                };
                let (compressed, uncompressed) = (wide(8), wide(16));
                if compressed != uncompressed {
                    return Err(fail(format!("{entry}: compressed ZIP64 entry")));
                }
                (u32_at(&descriptor, 4), compressed)
            }
        } else {
            (u32_at(&header, 14), u32_at(&header, 18) as usize)
        };
        if stored_crc != crc || stored_len != payload.len() {
            return Err(fail(format!("{entry}: corrupted sizes or CRC")));
        }
        let data = payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        tensors.insert(
            name.clone(),
            crate::weights::HostTensor::new(shape.clone(), data)?,
        );
    }
    Ok(StateDict::from_tensors(path, tensors))
}

/// NHWC frames to NCHW at `size x size` with `F.interpolate(bilinear, antialias=True)` semantics.
fn resize_nhwc_to_nchw(frames: &[f32], count: usize, height: usize, width: usize, size: usize) -> Vec<f32> {
    let rows = AaWeights::new(height, size);
    let cols = AaWeights::new(width, size);
    let mut out = vec![0.0f32; count * 3 * size * size];
    let mut horizontal = vec![0.0f64; height * size];
    for (f, frame) in frames.chunks_exact(height * width * 3).enumerate() {
        for c in 0..3 {
            for y in 0..height {
                for (x, (start, weights)) in cols.taps.iter().enumerate() {
                    horizontal[y * size + x] = weights
                        .iter()
                        .enumerate()
                        .map(|(k, w)| w * f64::from(frame[(y * width + start + k) * 3 + c]))
                        .sum();
                }
            }
            let plane = &mut out[(f * 3 + c) * size * size..(f * 3 + c + 1) * size * size];
            for (y, (start, weights)) in rows.taps.iter().enumerate() {
                for x in 0..size {
                    plane[y * size + x] = weights
                        .iter()
                        .enumerate()
                        .map(|(k, w)| w * horizontal[(start + k) * size + x])
                        .sum::<f64>() as f32;
                }
            }
        }
    }
    out
}

/// Per-output source ranges and normalized triangle weights (torch `_upsample_bilinear2d_aa`).
struct AaWeights {
    taps: Vec<(usize, Vec<f64>)>,
}

impl AaWeights {
    fn new(input: usize, output: usize) -> Self {
        let scale = input as f64 / output as f64;
        let support = if scale >= 1.0 { scale } else { 1.0 };
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let taps = (0..output)
            .map(|i| {
                let center = scale * (i as f64 + 0.5);
                let start = ((center - support + 0.5) as i64).max(0) as usize;
                let end = ((center + support + 0.5) as i64).min(input as i64) as usize;
                let mut weights: Vec<f64> = (start..end)
                    .map(|j| {
                        let x = (j as f64 - center + 0.5) * invscale;
                        (1.0 - x.abs()).max(0.0)
                    })
                    .collect();
                let total: f64 = weights.iter().sum();
                if total != 0.0 {
                    for w in &mut weights {
                        *w /= total;
                    }
                }
                (start, weights)
            })
            .collect();
        Self { taps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_resize_keeps_pixels() {
        let frames: Vec<f32> = (0..2 * 2 * 3).map(|i| i as f32 / 12.0).collect();
        let out = resize_nhwc_to_nchw(&frames, 1, 2, 2, 2);
        assert_eq!(out[..4], [frames[0], frames[3], frames[6], frames[9]]);
    }
}

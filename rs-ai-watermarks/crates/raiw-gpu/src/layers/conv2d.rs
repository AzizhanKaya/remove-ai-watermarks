//! 2-D convolution with PyTorch weight layout and a chunked im2col lowering for GPUs.

use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Tensor, s};

use crate::error::Result;
use crate::weights::StateDict;

/// Bytes of one im2col chunk; bounds VRAM and keeps every dispatch short.
const LOWERING_CHUNK_BYTES: usize = 96 << 20;

/// Static hyper-parameters of a `torch.nn.Conv2d` (square kernel, symmetric padding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
    pub bias: bool,
}

impl Conv2dSpec {
    pub const fn new(in_channels: usize, out_channels: usize, kernel: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: true,
        }
    }

    #[must_use]
    pub const fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    #[must_use]
    pub const fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    #[must_use]
    pub const fn groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    #[must_use]
    pub const fn no_bias(mut self) -> Self {
        self.bias = false;
        self
    }
}

/// GPU replacements for burn's direct conv2d kernels.
#[derive(Debug, Clone)]
enum Lowering {
    /// `[1, out, kernel * kernel * in]` for `weight @ im2col(x)`.
    Im2col(Box<Tensor<3>>),
    /// Per-tap `[1, channels, 1, 1]` weights of a stride-1 depthwise convolution.
    Depthwise(Vec<Tensor<4>>),
}

#[derive(Debug, Clone)]
pub struct Conv2d {
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    lowering: Option<Lowering>,
    spec: Conv2dSpec,
}

impl Conv2d {
    /// Load `<prefix>.weight` (`[out, in / groups, k, k]`) and `<prefix>.bias`.
    pub fn load(state: &mut StateDict, prefix: &str, spec: Conv2dSpec, device: &Device) -> Result<Self> {
        let k = spec.kernel;
        let weight: Tensor<4> = state.tensor(
            &format!("{prefix}.weight"),
            [spec.out_channels, spec.in_channels / spec.groups, k, k],
            device,
        )?;
        let bias = if spec.bias {
            Some(state.tensor(&format!("{prefix}.bias"), [spec.out_channels], device)?)
        } else {
            None
        };
        Ok(Self::from_parts(weight, bias, spec, device))
    }

    pub fn from_parts(weight: Tensor<4>, bias: Option<Tensor<1>>, spec: Conv2dSpec, device: &Device) -> Self {
        let k = spec.kernel;
        let depthwise = spec.groups == spec.in_channels && spec.out_channels == spec.in_channels;
        // GPU direct conv2d kernels trail these lowerings by a wide margin on the ported shapes
        let lowering = if *device == Device::flex() || k == 1 {
            None
        } else if spec.groups == 1 {
            let flat = [1, spec.out_channels, k * k * spec.in_channels];
            Some(Lowering::Im2col(Box::new(
                weight.clone().permute([0, 2, 3, 1]).reshape(flat),
            )))
        } else if depthwise && spec.stride == 1 {
            let taps = weight.clone().reshape([spec.out_channels, k * k]);
            let tap = |t: usize| taps.clone().narrow(1, t, 1).reshape([1, spec.out_channels, 1, 1]);
            Some(Lowering::Depthwise((0..k * k).map(tap).collect()))
        } else {
            None
        };
        Self {
            weight,
            bias,
            lowering,
            spec,
        }
    }

    pub fn spec(&self) -> &Conv2dSpec {
        &self.spec
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let p = self.spec.padding;
        self.forward_padded(x, [(p, p), (p, p)])
    }

    /// Convolve with explicit `[(top, bottom), (left, right)]` zero padding.
    pub fn forward_padded(&self, x: Tensor<4>, padding: [(usize, usize); 2]) -> Tensor<4> {
        let spec = &self.spec;
        let out = match &self.lowering {
            Some(Lowering::Im2col(weight)) => self.forward_im2col(&pad_spatial(x, padding), weight),
            Some(Lowering::Depthwise(taps)) => self.forward_depthwise(&pad_spatial(x, padding), taps),
            None => {
                let options = ConvOptions::new_with_padding([spec.stride; 2], padding, [1, 1], spec.groups);
                return conv2d(x, self.weight.clone(), self.bias.clone(), options);
            }
        };
        match &self.bias {
            Some(bias) => out + bias.clone().reshape([1, spec.out_channels, 1, 1]),
            None => out,
        }
    }

    /// Stride-1 depthwise convolution as a sum of per-channel scaled, shifted views.
    fn forward_depthwise(&self, padded: &Tensor<4>, taps: &[Tensor<4>]) -> Tensor<4> {
        let k = self.spec.kernel;
        let [_, _, height, width] = padded.dims();
        let (out_h, out_w) = (height + 1 - k, width + 1 - k);
        taps.iter()
            .enumerate()
            .map(|(t, w)| {
                let (i, j) = (t / k, t % k);
                padded.clone().slice(s![.., .., i..i + out_h, j..j + out_w]) * w.clone()
            })
            .reduce(|acc, tap| acc + tap)
            .expect("kernels have at least one tap")
    }

    /// Convolution as `weight @ im2col(x)`, chunked over output rows (strided taps for stride > 1).
    fn forward_im2col(&self, padded: &Tensor<4>, weight: &Tensor<3>) -> Tensor<4> {
        let spec = &self.spec;
        let (k, stride) = (spec.kernel, spec.stride);
        let [batch, channels, height, width] = padded.dims();
        let (out_h, out_w) = ((height - k) / stride + 1, (width - k) / stride + 1);
        let row_bytes = 4 * batch * channels * k * k * out_w;
        let rows = (LOWERING_CHUNK_BYTES / row_bytes.max(1)).clamp(1, out_h);
        let pieces: Vec<Tensor<4>> = (0..out_h)
            .step_by(rows)
            .map(|r0| {
                let n = rows.min(out_h - r0);
                let mut taps = Vec::with_capacity(k * k);
                for i in 0..k {
                    let row = r0 * stride + i;
                    let row_end = row + (n - 1) * stride + 1;
                    for j in 0..k {
                        let col_end = j + (out_w - 1) * stride + 1;
                        let tap = if stride == 1 {
                            padded.clone().slice(s![.., .., row..row_end, j..col_end])
                        } else {
                            padded
                                .clone()
                                .slice(s![.., .., row..row_end;stride, j..col_end;stride])
                        };
                        taps.push(tap);
                    }
                }
                let cols = Tensor::cat(taps, 1).reshape([batch, k * k * channels, n * out_w]);
                weight
                    .clone()
                    .matmul(cols)
                    .reshape([batch, spec.out_channels, n, out_w])
            })
            .collect();
        Tensor::cat(pieces, 2)
    }
}

fn pad_spatial(x: Tensor<4>, [(top, bottom), (left, right)]: [(usize, usize); 2]) -> Tensor<4> {
    if top + bottom + left + right > 0 {
        x.pad([(top, bottom), (left, right)], 0.0)
    } else {
        x
    }
}

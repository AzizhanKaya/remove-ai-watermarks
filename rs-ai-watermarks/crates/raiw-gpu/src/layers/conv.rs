//! 1-D convolutions with PyTorch weight layouts (weight-norm already folded).

use burn::tensor::module::{conv_transpose1d, conv1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Device, Tensor, s};

use crate::error::Result;
use crate::weights::StateDict;

/// Static hyper-parameters of a `torch.nn.Conv1d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    /// Symmetric zero padding applied by the convolution itself.
    pub padding: usize,
    pub bias: bool,
}

impl Conv1dSpec {
    pub const fn new(in_channels: usize, out_channels: usize, kernel: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel,
            stride: 1,
            dilation: 1,
            groups: 1,
            padding: 0,
            bias: true,
        }
    }

    #[must_use]
    pub const fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    #[must_use]
    pub const fn dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    #[must_use]
    pub const fn groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    #[must_use]
    pub const fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    #[must_use]
    pub const fn no_bias(mut self) -> Self {
        self.bias = false;
        self
    }

    /// Receptive field of one output sample.
    pub const fn effective_kernel(&self) -> usize {
        (self.kernel - 1) * self.dilation + 1
    }
}

/// GPU replacements for burn's direct conv1d kernels, which trail these 4-100x on long sequences.
#[derive(Debug, Clone)]
enum Lowering {
    /// `[1, out, kernel * in]` for `weight @ im2col(x)`.
    Im2col(Box<Tensor<3>>),
    /// Per-tap `[1, channels, 1]` weights of a depthwise convolution, applied as shifted multiply-adds.
    Depthwise(Vec<Tensor<3>>),
}

impl Lowering {
    fn new(weight: &Tensor<3>, spec: &Conv1dSpec) -> Option<Self> {
        if spec.groups == 1 {
            let flat = [1, spec.out_channels, spec.kernel * spec.in_channels];
            return Some(Self::Im2col(Box::new(
                weight.clone().swap_dims(1, 2).reshape(flat),
            )));
        }
        (spec.groups == spec.in_channels && spec.out_channels == spec.in_channels).then(|| {
            let taps =
                (0..spec.kernel).map(|k| weight.clone().narrow(2, k, 1).reshape([1, spec.out_channels, 1]));
            Self::Depthwise(taps.collect())
        })
    }
}

#[derive(Debug, Clone)]
pub struct Conv1d {
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    lowering: Option<Lowering>,
    spec: Conv1dSpec,
}

/// Bytes of one im2col chunk; bounds VRAM and keeps each dispatch short.
const LOWERING_CHUNK_BYTES: usize = 64 << 20;

/// `len` samples of `x` from `start`, every `stride` samples.
fn strided(x: &Tensor<3>, start: usize, len: usize, stride: usize) -> Tensor<3> {
    if stride == 1 {
        x.clone().narrow(2, start, len)
    } else {
        x.clone()
            .slice(s![.., .., start..start + (len - 1) * stride + 1;stride])
    }
}

impl Conv1d {
    /// Load `<prefix>.weight` (`[out, in / groups, kernel]`) and `<prefix>.bias`.
    pub fn load(state: &mut StateDict, prefix: &str, spec: Conv1dSpec, device: &Device) -> Result<Self> {
        let weight: Tensor<3> = state.tensor(
            &format!("{prefix}.weight"),
            [spec.out_channels, spec.in_channels / spec.groups, spec.kernel],
            device,
        )?;
        let bias = if spec.bias {
            Some(state.tensor(&format!("{prefix}.bias"), [spec.out_channels], device)?)
        } else {
            None
        };
        // burn-flex already runs im2col + gemm and a depthwise fast path
        let lowering = (*device != Device::flex())
            .then(|| Lowering::new(&weight, &spec))
            .flatten();
        Ok(Self {
            weight,
            bias,
            lowering,
            spec,
        })
    }

    pub fn spec(&self) -> &Conv1dSpec {
        &self.spec
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_padded(x, (self.spec.padding, self.spec.padding))
    }

    /// Convolve with explicit (possibly asymmetric) zero padding instead of the spec's.
    pub fn forward_padded(&self, x: Tensor<3>, padding: (usize, usize)) -> Tensor<3> {
        let spec = &self.spec;
        let out = match &self.lowering {
            Some(Lowering::Im2col(weight)) => self.forward_im2col(&pad_time(x, padding), weight),
            Some(Lowering::Depthwise(taps)) => self.forward_depthwise(&pad_time(x, padding), taps),
            None => {
                let options =
                    ConvOptions::new_with_padding([spec.stride], [padding], [spec.dilation], spec.groups);
                return conv1d(x, self.weight.clone(), self.bias.clone(), options);
            }
        };
        match &self.bias {
            Some(bias) => out + bias.clone().reshape([1, spec.out_channels, 1]),
            None => out,
        }
    }

    /// Output samples of the already padded input length.
    fn output_len(&self, padded: usize) -> usize {
        let span = (self.spec.kernel - 1) * self.spec.dilation;
        padded
            .checked_sub(span + 1)
            .map_or(0, |room| room / self.spec.stride + 1)
    }

    /// `weight @ im2col(x)` with strided taps, chunked along time.
    fn forward_im2col(&self, padded: &Tensor<3>, weight: &Tensor<3>) -> Tensor<3> {
        let spec = &self.spec;
        let [batch, channels, length] = padded.dims();
        let out_len = self.output_len(length);
        let chunk = (LOWERING_CHUNK_BYTES / (4 * batch * channels * spec.kernel)).max(1);
        let pieces: Vec<Tensor<3>> = (0..out_len)
            .step_by(chunk)
            .map(|start| {
                let len = chunk.min(out_len - start);
                let mut taps: Vec<Tensor<3>> = (0..spec.kernel)
                    .map(|k| strided(padded, start * spec.stride + k * spec.dilation, len, spec.stride))
                    .collect();
                let cols = if taps.len() == 1 {
                    taps.remove(0)
                } else {
                    Tensor::cat(taps, 1)
                };
                weight.clone().matmul(cols)
            })
            .collect();
        Tensor::cat(pieces, 2)
    }

    /// Depthwise convolution as a sum of per-channel scaled, shifted views.
    fn forward_depthwise(&self, padded: &Tensor<3>, taps: &[Tensor<3>]) -> Tensor<3> {
        let spec = &self.spec;
        let out_len = self.output_len(padded.dims()[2]);
        taps.iter()
            .enumerate()
            .map(|(k, w)| strided(padded, k * spec.dilation, out_len, spec.stride) * w.clone())
            .reduce(|acc, tap| acc + tap)
            .expect("kernels have at least one tap")
    }
}

fn pad_time(x: Tensor<3>, (left, right): (usize, usize)) -> Tensor<3> {
    if left + right > 0 {
        x.pad([(left, right)], 0.0)
    } else {
        x
    }
}

/// Static hyper-parameters of a `torch.nn.ConvTranspose1d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose1dSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub groups: usize,
    pub bias: bool,
}

impl ConvTranspose1dSpec {
    pub const fn new(in_channels: usize, out_channels: usize, kernel: usize, stride: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel,
            stride,
            padding: 0,
            output_padding: 0,
            groups: 1,
            bias: true,
        }
    }

    #[must_use]
    pub const fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    #[must_use]
    pub const fn output_padding(mut self, output_padding: usize) -> Self {
        self.output_padding = output_padding;
        self
    }

    /// Stride-long tap groups each input sample spreads over.
    const fn phases(&self) -> usize {
        self.kernel.div_ceil(self.stride)
    }
}

#[derive(Debug, Clone)]
pub struct ConvTranspose1d {
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    /// `[1, phases * stride * out, in]` (taps zero-filled to whole strides) for matmul + overlap-add.
    lowered: Option<Tensor<3>>,
    spec: ConvTranspose1dSpec,
}

impl ConvTranspose1d {
    /// Load `<prefix>.weight` (`[in, out / groups, kernel]`) and `<prefix>.bias`.
    pub fn load(
        state: &mut StateDict,
        prefix: &str,
        spec: ConvTranspose1dSpec,
        device: &Device,
    ) -> Result<Self> {
        let weight: Tensor<3> = state.tensor(
            &format!("{prefix}.weight"),
            [spec.in_channels, spec.out_channels / spec.groups, spec.kernel],
            device,
        )?;
        let bias = if spec.bias {
            Some(state.tensor(&format!("{prefix}.bias"), [spec.out_channels], device)?)
        } else {
            None
        };
        // GPU transposed-conv kernels walk every tap per output; a matmul does the reduction instead
        let lowered = (spec.groups == 1 && *device != Device::flex()).then(|| {
            let taps = spec.phases() * spec.stride;
            let per_tap = weight.clone().permute([2, 1, 0]);
            let per_tap = if taps > spec.kernel {
                let fill = Tensor::zeros([taps - spec.kernel, spec.out_channels, spec.in_channels], device);
                Tensor::cat(vec![per_tap, fill], 0)
            } else {
                per_tap
            };
            per_tap.reshape([1, taps * spec.out_channels, spec.in_channels])
        });
        Ok(Self {
            weight,
            bias,
            lowered,
            spec,
        })
    }

    pub fn spec(&self) -> &ConvTranspose1dSpec {
        &self.spec
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let spec = &self.spec;
        if let Some(weight) = &self.lowered {
            let out = self.forward_lowered(&x, weight);
            return match &self.bias {
                Some(bias) => out + bias.clone().reshape([1, spec.out_channels, 1]),
                None => out,
            };
        }
        let options = ConvTransposeOptions::new(
            [spec.stride],
            [spec.padding],
            [spec.output_padding],
            [1],
            spec.groups,
        );
        conv_transpose1d(x, self.weight.clone(), self.bias.clone(), options)
    }

    /// Per-tap products `weight @ x`, then each stride-long phase is overlap-added into place.
    fn forward_lowered(&self, x: &Tensor<3>, weight: &Tensor<3>) -> Tensor<3> {
        let spec = &self.spec;
        let (stride, out, phases) = (spec.stride, spec.out_channels, spec.phases());
        let [batch, _, steps] = x.dims();
        let chunk = (LOWERING_CHUNK_BYTES / (4 * batch * phases * stride * out)).max(1);
        let pieces: Vec<Tensor<3>> = (0..steps)
            .step_by(chunk)
            .map(|start| {
                weight
                    .clone()
                    .matmul(x.clone().narrow(2, start, chunk.min(steps - start)))
            })
            .collect();
        let products = Tensor::cat(pieces, 2).reshape([batch, phases, stride, out, steps]);
        let full = (0..phases)
            .map(|j| {
                let phase = products
                    .clone()
                    .narrow(1, j, 1)
                    .reshape([batch, stride, out, steps])
                    .permute([0, 2, 3, 1])
                    .reshape([batch, out, steps * stride]);
                pad_time(phase, (j * stride, (phases - 1 - j) * stride))
            })
            .reduce(|acc, phase| acc + phase)
            .expect("transposed convolutions have at least one phase");
        let length = (steps - 1) * stride + spec.kernel + spec.output_padding - 2 * spec.padding;
        let full_len = (steps + phases - 1) * stride;
        let tail = (spec.padding + length).saturating_sub(full_len);
        pad_time(full, (0, tail)).narrow(2, spec.padding, length)
    }
}

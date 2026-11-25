//! 1-D convolutions with PyTorch weight layouts (weight-norm already folded).

use burn::tensor::module::{conv_transpose1d, conv1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Device, Tensor};

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

#[derive(Debug, Clone)]
pub struct Conv1d {
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    spec: Conv1dSpec,
}

impl Conv1d {
    /// Load `<prefix>.weight` (`[out, in / groups, kernel]`) and `<prefix>.bias`.
    pub fn load(state: &mut StateDict, prefix: &str, spec: Conv1dSpec, device: &Device) -> Result<Self> {
        let weight = state.tensor(
            &format!("{prefix}.weight"),
            [spec.out_channels, spec.in_channels / spec.groups, spec.kernel],
            device,
        )?;
        let bias = if spec.bias {
            Some(state.tensor(&format!("{prefix}.bias"), [spec.out_channels], device)?)
        } else {
            None
        };
        Ok(Self { weight, bias, spec })
    }

    pub fn spec(&self) -> &Conv1dSpec {
        &self.spec
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_padded(x, (self.spec.padding, self.spec.padding))
    }

    /// Convolve with explicit (possibly asymmetric) zero padding instead of the spec's.
    pub fn forward_padded(&self, x: Tensor<3>, padding: (usize, usize)) -> Tensor<3> {
        let options = ConvOptions::new_with_padding(
            [self.spec.stride],
            [padding],
            [self.spec.dilation],
            self.spec.groups,
        );
        conv1d(x, self.weight.clone(), self.bias.clone(), options)
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
}

#[derive(Debug, Clone)]
pub struct ConvTranspose1d {
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
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
        let weight = state.tensor(
            &format!("{prefix}.weight"),
            [spec.in_channels, spec.out_channels / spec.groups, spec.kernel],
            device,
        )?;
        let bias = if spec.bias {
            Some(state.tensor(&format!("{prefix}.bias"), [spec.out_channels], device)?)
        } else {
            None
        };
        Ok(Self { weight, bias, spec })
    }

    pub fn spec(&self) -> &ConvTranspose1dSpec {
        &self.spec
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let spec = &self.spec;
        let options = ConvTransposeOptions::new(
            [spec.stride],
            [spec.padding],
            [spec.output_padding],
            [1],
            spec.groups,
        );
        conv_transpose1d(x, self.weight.clone(), self.bias.clone(), options)
    }
}

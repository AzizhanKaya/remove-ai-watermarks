//! `torch.nn.Linear` (and pointwise convolutions) over the last axis.

use burn::tensor::module::linear;
use burn::tensor::{Device, Tensor, TensorData};

use crate::error::Result;
use crate::weights::{HostTensor, StateDict};

#[derive(Debug, Clone)]
pub struct Linear {
    /// `[in, out]`: a transposed view of the uploaded PyTorch `[out, in]` layout.
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
}

impl Linear {
    /// Load `<prefix>.weight` (`[out, in]`) and `<prefix>.bias`.
    pub fn load(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::load_shaped(state, prefix, input, output, 0, device)
    }

    /// Load a pointwise convolution stored as `[out, in, 1, ...]` with `unit_dims` trailing ones.
    pub fn load_pointwise(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        output: usize,
        unit_dims: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::load_shaped(state, prefix, input, output, unit_dims, device)
    }

    fn load_shaped(
        state: &mut StateDict,
        prefix: &str,
        input: usize,
        output: usize,
        unit_dims: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut shape = vec![output, input];
        shape.extend(std::iter::repeat_n(1, unit_dims));
        let weight = state.take(&format!("{prefix}.weight"), &shape)?;
        let bias = state.optional(&format!("{prefix}.bias"), [output], device)?;
        Ok(Self::from_host(weight, input, output, bias, device))
    }

    /// Build from a host `[out, in]` weight (any trailing unit dims).
    pub fn from_host(
        weight: HostTensor,
        input: usize,
        output: usize,
        bias: Option<Tensor<1>>,
        device: &Device,
    ) -> Self {
        let weight: Tensor<2> = Tensor::from_data(TensorData::new(weight.data, [output, input]), device);
        Self {
            weight: weight.transpose(),
            bias,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        linear(x, self.weight.clone(), self.bias.clone())
    }

    /// Output features.
    pub fn output(&self) -> usize {
        self.weight.dims()[1]
    }
}

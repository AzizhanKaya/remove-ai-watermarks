//! Normalization layers in PyTorch layouts.

use burn::tensor::{Device, Tensor};

use crate::error::Result;
use crate::weights::StateDict;

/// `torch.nn.GroupNorm` over `[N, C, H, W]` with a per-channel affine.
#[derive(Debug, Clone)]
pub struct GroupNorm {
    groups: usize,
    eps: f64,
    weight: Tensor<4>,
    bias: Tensor<4>,
}

impl GroupNorm {
    pub fn load(
        state: &mut StateDict,
        prefix: &str,
        groups: usize,
        channels: usize,
        eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let weight: Tensor<1> = state.tensor(&format!("{prefix}.weight"), [channels], device)?;
        let bias: Tensor<1> = state.tensor(&format!("{prefix}.bias"), [channels], device)?;
        Ok(Self {
            groups,
            eps,
            weight: weight.reshape([1, channels, 1, 1]),
            bias: bias.reshape([1, channels, 1, 1]),
        })
    }

    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [n, c, h, w] = x.dims();
        let grouped = x.reshape([n, self.groups, (c / self.groups) * h * w]);
        let mean = grouped.clone().mean_dim(2);
        let centered = grouped - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(2);
        let normed = (centered / var.add_scalar(self.eps).sqrt()).reshape([n, c, h, w]);
        normed * self.weight.clone() + self.bias.clone()
    }
}

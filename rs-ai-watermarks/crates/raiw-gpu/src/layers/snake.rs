//! Snake activation `x + sin(alpha * x)^2 / (alpha + 1e-9)` (DAC, SNAC, BigVGAN).

use burn::tensor::{Device, Tensor};

use crate::error::Result;
use crate::weights::StateDict;

#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor<3>,
    inv_alpha: Tensor<3>,
}

impl Snake1d {
    /// Load `<prefix>.alpha` of shape `[1, channels, 1]`.
    pub fn load(state: &mut StateDict, prefix: &str, channels: usize, device: &Device) -> Result<Self> {
        let alpha: Tensor<3> = state.tensor(&format!("{prefix}.alpha"), [1, channels, 1], device)?;
        let inv_alpha = alpha.clone().add_scalar(1e-9).recip();
        Ok(Self { alpha, inv_alpha })
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let wave = self.alpha.clone().mul(x.clone()).sin();
        x + self.inv_alpha.clone().mul(wave.clone().mul(wave))
    }
}

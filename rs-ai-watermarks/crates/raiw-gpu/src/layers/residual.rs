//! Snake residual unit shared by DAC and SNAC: `x + conv1x1(snake(conv_k7(snake(x))))`.

use burn::tensor::{Device, Tensor, s};

use super::conv::{Conv1d, Conv1dSpec};
use super::snake::Snake1d;
use crate::error::Result;
use crate::weights::StateDict;

/// Parameter naming of the four sub-layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitNaming {
    /// Transformers DAC: `snake1`, `conv1`, `snake2`, `conv2`.
    Named,
    /// SNAC `nn.Sequential`: `block.0` .. `block.3`.
    Sequential,
}

impl UnitNaming {
    fn names(self) -> [&'static str; 4] {
        match self {
            Self::Named => ["snake1", "conv1", "snake2", "conv2"],
            Self::Sequential => ["block.0", "block.1", "block.2", "block.3"],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    /// Kernel-7 dilated unit; `groups == dim` gives SNAC's depthwise variant.
    pub fn load(
        state: &mut StateDict,
        prefix: &str,
        naming: UnitNaming,
        dim: usize,
        dilation: usize,
        groups: usize,
        device: &Device,
    ) -> Result<Self> {
        let [s1, c1, s2, c2] = naming.names().map(|name| format!("{prefix}.{name}"));
        let wide = Conv1dSpec::new(dim, dim, 7)
            .dilation(dilation)
            .padding(3 * dilation)
            .groups(groups);
        Ok(Self {
            snake1: Snake1d::load(state, &s1, dim, device)?,
            conv1: Conv1d::load(state, &c1, wide, device)?,
            snake2: Snake1d::load(state, &s2, dim, device)?,
            conv2: Conv1d::load(state, &c2, Conv1dSpec::new(dim, dim, 1), device)?,
        })
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = self.conv2.forward(
            self.snake2
                .forward(self.conv1.forward(self.snake1.forward(x.clone()))),
        );
        let crop = (x.dims()[2] - y.dims()[2]) / 2;
        if crop > 0 {
            let end = x.dims()[2] - crop;
            x.slice(s![.., .., crop..end]) + y
        } else {
            x + y
        }
    }
}

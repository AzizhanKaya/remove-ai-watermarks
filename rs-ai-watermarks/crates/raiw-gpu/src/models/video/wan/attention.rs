//! Memory-bounded attention and the normalizations shared by the Wan VAE and transformer.

use burn::tensor::activation::softmax;
use burn::tensor::module::layer_norm;
use burn::tensor::{Device, Tensor, s};

use crate::error::Result;
use crate::weights::StateDict;

/// Score bytes one attention dispatch may cover; bounds VRAM and keeps every dispatch short.
const SCORE_BUDGET_BYTES: usize = 32 << 20;

/// `F.scaled_dot_product_attention(q, k, v)` with the default `1 / sqrt(head_dim)` scale over
/// `[batch, heads, tokens, head_dim]`, split over heads and query rows within the score budget.
pub fn scaled_dot_product(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    let [batch, heads, queries, _] = q.dims();
    let row_bytes = 4 * batch * k.dims()[2];
    let rows = (SCORE_BUDGET_BYTES / row_bytes).clamp(1, queries);
    let group = if rows == queries {
        (SCORE_BUDGET_BYTES / (row_bytes * queries)).clamp(1, heads)
    } else {
        1
    };
    if rows == queries && group == heads {
        return attend(q, k, v);
    }
    let by_head: Vec<Tensor<4>> = (0..heads)
        .step_by(group)
        .map(|h0| {
            let h1 = (h0 + group).min(heads);
            let (kh, vh) = (k.clone().slice(s![.., h0..h1]), v.clone().slice(s![.., h0..h1]));
            let pieces: Vec<Tensor<4>> = (0..queries)
                .step_by(rows)
                .map(|r0| {
                    let r1 = (r0 + rows).min(queries);
                    attend(q.clone().slice(s![.., h0..h1, r0..r1]), kh.clone(), vh.clone())
                })
                .collect();
            Tensor::cat(pieces, 2)
        })
        .collect();
    Tensor::cat(by_head, 1)
}

/// Materialized scores within the budget; RADV takes minutes to build the fused flash kernels.
fn attend(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    let scale = 1.0 / (q.dims()[3] as f64).sqrt();
    softmax(q.matmul(k.swap_dims(2, 3)).mul_scalar(scale), 3).matmul(v)
}

/// `torch.nn.RMSNorm` over the last axis: `x * rsqrt(mean(x^2) + eps) * weight`.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor<1>,
    eps: f64,
}

impl RmsNorm {
    pub fn load(state: &mut StateDict, prefix: &str, dim: usize, eps: f64, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: state.tensor(&format!("{prefix}.weight"), [dim], device)?,
            eps,
        })
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let rms = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(D - 1)
            .add_scalar(self.eps)
            .sqrt();
        x / rms * self.weight.clone().unsqueeze()
    }
}

/// `FP32LayerNorm` over the last axis, optionally with a learned affine.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    gamma: Tensor<1>,
    beta: Option<Tensor<1>>,
    eps: f64,
}

impl LayerNorm {
    pub fn plain(dim: usize, eps: f64, device: &Device) -> Self {
        Self {
            gamma: Tensor::ones([dim], device),
            beta: None,
            eps,
        }
    }

    pub fn load(state: &mut StateDict, prefix: &str, dim: usize, eps: f64, device: &Device) -> Result<Self> {
        Ok(Self {
            gamma: state.tensor(&format!("{prefix}.weight"), [dim], device)?,
            beta: Some(state.tensor(&format!("{prefix}.bias"), [dim], device)?),
            eps,
        })
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        layer_norm(x, self.gamma.clone(), self.beta.clone(), self.eps)
    }
}

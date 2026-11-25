//! Factorized, l2-normalized vector quantizers (DAC and SNAC residual codebooks).

use burn::tensor::module::{avg_pool1d, embedding};
use burn::tensor::ops::AvgPoolOptions;
use burn::tensor::{Device, Int, Tensor};

use super::conv::{Conv1d, Conv1dSpec};
use crate::error::Result;
use crate::weights::StateDict;

/// How the nearest code is scored, mirroring each reference's exact arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeScore {
    /// DAC: `-(|e|^2 - 2 e.c) + |c|^2` over normalized vectors.
    Distance,
    /// SNAC: plain cosine similarity `e.c`.
    Cosine,
}

/// Shape of one codebook stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VqSpec {
    /// Channels of the latent being quantized.
    pub input_dim: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    /// Temporal pooling before lookup (SNAC multi-scale); 1 for DAC.
    pub stride: usize,
    pub score: CodeScore,
}

/// One codebook with its 1x1 in/out projections and an optional temporal stride.
#[derive(Debug, Clone)]
pub struct VectorQuantizer {
    in_proj: Conv1d,
    out_proj: Conv1d,
    codebook: Tensor<2>,
    /// Normalized codebook, transposed to `[dim, size]`.
    codebook_norm_t: Tensor<2>,
    /// `|c|^2` of the normalized codebook as `[1, size]`.
    codebook_sq: Tensor<2>,
    stride: usize,
    score: CodeScore,
}

impl VectorQuantizer {
    /// Load `<prefix>.{in_proj,out_proj,codebook}` (weight-norm folded).
    pub fn load(state: &mut StateDict, prefix: &str, spec: VqSpec, device: &Device) -> Result<Self> {
        let (input, dim) = (spec.input_dim, spec.codebook_dim);
        let in_proj = Conv1d::load(
            state,
            &format!("{prefix}.in_proj"),
            Conv1dSpec::new(input, dim, 1),
            device,
        )?;
        let out_proj = Conv1d::load(
            state,
            &format!("{prefix}.out_proj"),
            Conv1dSpec::new(dim, input, 1),
            device,
        )?;
        let codebook: Tensor<2> = state.tensor(
            &format!("{prefix}.codebook.weight"),
            [spec.codebook_size, dim],
            device,
        )?;
        let normalized = l2_normalize(codebook.clone());
        let codebook_sq = normalized.clone().powf_scalar(2.0).sum_dim(1).transpose();
        Ok(Self {
            in_proj,
            out_proj,
            codebook,
            codebook_norm_t: normalized.transpose(),
            codebook_sq,
            stride: spec.stride,
            score: spec.score,
        })
    }

    /// Nearest code per frame of `z` (`[batch, input_dim, time]`) as `[batch, time / stride]`.
    pub fn encode(&self, z: Tensor<3>) -> Tensor<2, Int> {
        let z = if self.stride > 1 {
            avg_pool1d(z, AvgPoolOptions::new([self.stride]).with_stride([self.stride]))
        } else {
            z
        };
        let latents = self.in_proj.forward(z);
        let [batch, dim, frames] = latents.dims();
        let flat = l2_normalize(latents.swap_dims(1, 2).reshape([batch * frames, dim]));
        let dot = flat.clone().matmul(self.codebook_norm_t.clone());
        let scores = match self.score {
            CodeScore::Cosine => dot,
            CodeScore::Distance => {
                let e_sq = flat.powf_scalar(2.0).sum_dim(1);
                (dot.mul_scalar(2.0) - e_sq) + self.codebook_sq.clone()
            }
        };
        scores.argmax(1).reshape([batch, frames])
    }

    /// Project codes back to `[batch, input_dim, time]`, repeating each frame `stride` times.
    pub fn decode(&self, codes: Tensor<2, Int>) -> Tensor<3> {
        let latents = embedding(self.codebook.clone(), codes).swap_dims(1, 2);
        let out = self.out_proj.forward(latents);
        if self.stride > 1 {
            let [batch, channels, frames] = out.dims();
            out.unsqueeze_dim::<4>(3)
                .expand([batch, channels, frames, self.stride])
                .reshape([batch, channels, frames * self.stride])
        } else {
            out
        }
    }
}

/// `F.normalize(x, dim=1)`: rows divided by `max(||row||, 1e-12)`.
pub fn l2_normalize(x: Tensor<2>) -> Tensor<2> {
    let norm = x.clone().powf_scalar(2.0).sum_dim(1).sqrt().clamp_min(1e-12);
    x / norm
}

/// Residual quantization over the first `active` codebooks: returns per-codebook codes and the sum.
pub fn residual_quantize(quantizers: &[VectorQuantizer], z: Tensor<3>) -> (Vec<Tensor<2, Int>>, Tensor<3>) {
    let mut residual = z.clone();
    let mut quantized = z.zeros_like();
    let mut codes = Vec::with_capacity(quantizers.len());
    for quantizer in quantizers {
        let code = quantizer.encode(residual.clone());
        let z_i = quantizer.decode(code.clone());
        quantized = quantized + z_i.clone();
        residual = residual - z_i;
        codes.push(code);
    }
    (codes, quantized)
}

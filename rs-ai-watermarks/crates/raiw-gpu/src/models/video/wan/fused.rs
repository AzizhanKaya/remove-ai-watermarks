//! The GPU transformer block: resident weights driving the CubeCL kernels ([`crate::kernels`]).
//!
//! One block, per denoising step:
//!
//! 1. adaLN LayerNorm → one fused QKV GEMM → RMSNorm + RoPE on Q and K → flash self-attention
//!    → output GEMM closing with the gated residual;
//! 2. affine LayerNorm → query GEMM → RMSNorm → flash cross-attention over this block's text
//!    keys and values (projected once per prompt, [`CrossKv`]) → output GEMM + residual;
//! 3. adaLN LayerNorm → GELU GEMM → down GEMM closing with the gated residual.
//!
//! The residual stream stays float32; GEMM and attention operands use the storage
//! [`Precision`] (f16 halves memory traffic and keeps all 30 blocks resident in 2.7 GB, the way
//! Python runs the model in bfloat16 on CUDA), always with float32 accumulation.

use burn::tensor::{DType, Device, Tensor, TensorData, s};

use crate::error::Result;
use crate::kernels::{self, Activation, DenseWeight};
use crate::weights::StateDict;

use super::config::TransformerConfig;

const EPS: f32 = 1e-6;
/// Tokens per feed-forward chunk, bounding the `[tokens, 8960]` intermediate.
const FFN_TOKENS: usize = 16_384;

/// Storage precision of the kernel path's weights and GEMM/attention operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Half the bandwidth and memory of float32; Python's CUDA path runs bfloat16.
    #[default]
    F16,
    /// f16 activations, per-channel int8 weights: half the memory of f16 at the same speed
    /// (the 5B Wan 2.2 transformer fits an 8 GB card only this way).
    Int8,
    /// Same kernels in float32 (parity checks).
    F32,
}

impl Precision {
    /// The activation (operand) dtype.
    pub fn dtype(self) -> DType {
        match self {
            Self::F16 | Self::Int8 => DType::F16,
            Self::F32 => DType::F32,
        }
    }
}

/// `x·W + b` with the bias float32 and the weight at the storage precision: f16 and int8
/// weights are stored `[in, out]` for the packed-math GEMM, f32 ones in PyTorch's `[out, in]`.
#[derive(Debug, Clone)]
struct Dense {
    weight: DenseWeight,
    bias: Tensor<1>,
}

impl Dense {
    /// Stack the `[output, input]` projections under `prefixes` into one `[n * output, input]`.
    fn load(
        state: &mut StateDict,
        prefixes: &[String],
        (input, output): (usize, usize),
        precision: Precision,
        device: &Device,
    ) -> Result<Self> {
        let rows = prefixes.len() * output;
        let (mut weight, mut bias) = (Vec::with_capacity(rows * input), Vec::with_capacity(rows));
        for prefix in prefixes {
            weight.extend(state.take(&format!("{prefix}.weight"), &[output, input])?.data);
            bias.extend(state.take(&format!("{prefix}.bias"), &[output])?.data);
        }
        let weight = match precision {
            Precision::Int8 => DenseWeight::int8(&weight, (rows, input), device),
            Precision::F16 => {
                DenseWeight::packed(Tensor::from_data(TensorData::new(weight, [rows, input]), device))
            }
            Precision::F32 => {
                DenseWeight::torch(Tensor::from_data(TensorData::new(weight, [rows, input]), device))
            }
        };
        Ok(Self {
            weight,
            bias: Tensor::from_data(TensorData::new(bias, [rows]), device),
        })
    }

    fn to_device(&self, device: &Device) -> Self {
        Self {
            weight: self.weight.to_device(device),
            bias: self.bias.clone().to_device(device),
        }
    }

    fn apply(&self, x: Tensor<2>, activation: Activation, dtype: DType) -> Tensor<2> {
        kernels::linear(x, &self.weight, self.bias.clone(), activation, dtype)
    }

    /// `residual + (x·W + b) * gate`.
    fn gated(&self, x: Tensor<2>, residual: Tensor<2>, gate: Tensor<1>) -> Tensor<2> {
        kernels::linear_gated(x, &self.weight, self.bias.clone(), residual, gate)
    }

    /// `residual + x·W + b`.
    fn residual(&self, x: Tensor<2>, residual: Tensor<2>) -> Tensor<2> {
        kernels::linear_residual(x, &self.weight, self.bias.clone(), residual)
    }
}

fn store(t: Tensor<2>, dtype: DType) -> Tensor<2> {
    if t.dtype() == dtype { t } else { t.cast(dtype) }
}

/// One block's text keys and values, `[text tokens, dim]` at the storage precision.
#[derive(Debug, Clone)]
pub struct CrossKv {
    k: Tensor<2>,
    v: Tensor<2>,
}

/// Rotary cos/sin tables `[tokens, head_dim / 2]` for one patch grid.
#[derive(Debug, Clone)]
pub struct RopeTables {
    cos: Tensor<2>,
    sin: Tensor<2>,
    head_dim: usize,
}

impl RopeTables {
    pub fn new((frames, rows, cols): (usize, usize, usize), head_dim: usize, device: &Device) -> Self {
        let (cos, sin) = super::rope::tables(head_dim, (frames, rows, cols));
        let shape = [frames * rows * cols, head_dim / 2];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
            head_dim,
        }
    }

    fn pair(&self) -> (Tensor<2>, Tensor<2>) {
        (self.cos.clone(), self.sin.clone())
    }
}

/// `WanTransformerBlock` on the kernel path.
#[derive(Debug, Clone)]
pub struct FusedBlock {
    /// `scale_shift_table` `[6, dim]`: shift, scale, gate (attention) then (feed-forward).
    table: Tensor<2>,
    dim: usize,
    heads: usize,
    qkv: Dense,
    norm_q: Tensor<1>,
    norm_k: Tensor<1>,
    out: Dense,
    norm2_weight: Tensor<1>,
    norm2_bias: Tensor<1>,
    cross_q: Dense,
    cross_kv: Dense,
    cross_norm_q: Tensor<1>,
    cross_norm_k: Tensor<1>,
    cross_out: Dense,
    ffn_in: Dense,
    ffn_out: Dense,
    dtype: DType,
}

impl FusedBlock {
    pub fn load(
        state: &mut StateDict,
        prefix: &str,
        config: &TransformerConfig,
        precision: Precision,
        device: &Device,
    ) -> Result<Self> {
        let (dim, ffn) = (config.dim(), config.ffn_dim);
        let dtype = precision.dtype();
        let names = |parts: &[&str]| parts.iter().map(|p| format!("{prefix}.{p}")).collect::<Vec<_>>();
        let mut dense = |parts: &[&str], shape| Dense::load(state, &names(parts), shape, precision, device);
        let qkv = dense(&["attn1.to_q", "attn1.to_k", "attn1.to_v"], (dim, dim))?;
        let out = dense(&["attn1.to_out.0"], (dim, dim))?;
        let cross_q = dense(&["attn2.to_q"], (dim, dim))?;
        let cross_kv = dense(&["attn2.to_k", "attn2.to_v"], (dim, dim))?;
        let cross_out = dense(&["attn2.to_out.0"], (dim, dim))?;
        let ffn_in = dense(&["ffn.net.0.proj"], (dim, ffn))?;
        let ffn_out = dense(&["ffn.net.2"], (ffn, dim))?;
        let table: Tensor<3> = state.tensor(&format!("{prefix}.scale_shift_table"), [1, 6, dim], device)?;
        let mut vector = |name: &str| state.tensor::<1>(&format!("{prefix}.{name}"), [dim], device);
        Ok(Self {
            table: table.reshape([6, dim]),
            dim,
            heads: config.heads,
            qkv,
            norm_q: vector("attn1.norm_q.weight")?,
            norm_k: vector("attn1.norm_k.weight")?,
            out,
            norm2_weight: vector("norm2.weight")?,
            norm2_bias: vector("norm2.bias")?,
            cross_q,
            cross_kv,
            cross_norm_q: vector("attn2.norm_q.weight")?,
            cross_norm_k: vector("attn2.norm_k.weight")?,
            cross_out,
            ffn_in,
            ffn_out,
            dtype,
        })
    }

    /// A copy on `device`: [`super::BlockResidency::Swapped`] keeps blocks on the host and
    /// uploads them only while the transformer runs.
    #[must_use]
    pub fn to_device(&self, device: &Device) -> Self {
        let vector = |t: &Tensor<1>| t.clone().to_device(device);
        Self {
            table: self.table.clone().to_device(device),
            dim: self.dim,
            heads: self.heads,
            qkv: self.qkv.to_device(device),
            norm_q: vector(&self.norm_q),
            norm_k: vector(&self.norm_k),
            out: self.out.to_device(device),
            norm2_weight: vector(&self.norm2_weight),
            norm2_bias: vector(&self.norm2_bias),
            cross_q: self.cross_q.to_device(device),
            cross_kv: self.cross_kv.to_device(device),
            cross_norm_q: vector(&self.cross_norm_q),
            cross_norm_k: vector(&self.cross_norm_k),
            cross_out: self.cross_out.to_device(device),
            ffn_in: self.ffn_in.to_device(device),
            ffn_out: self.ffn_out.to_device(device),
            dtype: self.dtype,
        }
    }

    /// Project the text context `[text tokens, dim]` (float32) to this block's keys and values.
    pub fn cross_kv(&self, text: &Tensor<2>) -> CrossKv {
        self.project_text(&self.cross_kv, self.cross_norm_k.clone(), text)
    }

    /// [`Self::cross_kv`] of a block parked on another device: only its key/value projection
    /// visits `device`.
    pub fn cross_kv_on(&self, text: &Tensor<2>, device: &Device) -> CrossKv {
        let dense = self.cross_kv.to_device(device);
        self.project_text(&dense, self.cross_norm_k.clone().to_device(device), text)
    }

    fn project_text(&self, cross_kv: &Dense, norm_k: Tensor<1>, text: &Tensor<2>) -> CrossKv {
        let dim = self.dim;
        let kv = cross_kv.apply(store(text.clone(), self.dtype), Activation::Identity, self.dtype);
        CrossKv {
            k: kernels::rms_norm(kv.clone(), norm_k, 0, dim, EPS, self.dtype),
            v: kv.slice(s![.., dim..2 * dim]),
        }
    }

    /// One block over `h` `[tokens, dim]` (float32) at timestep projection `timestep` `[6, dim]`.
    pub fn forward(
        &self,
        h: Tensor<2>,
        cross: &CrossKv,
        timestep: &Tensor<2>,
        rope: &RopeTables,
    ) -> Tensor<2> {
        let (dtype, dim, heads) = (self.dtype, self.dim, self.heads);
        let modulation = self.table.clone() + timestep.clone();
        let row = |i: usize| modulation.clone().slice(s![i..i + 1]).reshape([dim]);

        let x = kernels::modulated_layer_norm(h.clone(), row(1).add_scalar(1.0), row(0), EPS, dtype);
        let qkv = self.qkv.apply(x, Activation::Identity, dtype);
        let rope_qk = |offset: usize, weight: &Tensor<1>| {
            kernels::rms_norm_rope(
                qkv.clone(),
                weight.clone(),
                rope.pair(),
                offset,
                dim,
                rope.head_dim,
                EPS,
                dtype,
            )
        };
        let (q, k) = (rope_qk(0, &self.norm_q), rope_qk(dim, &self.norm_k));
        let v = qkv.slice(s![.., 2 * dim..3 * dim]);
        let h = self
            .out
            .gated(kernels::flash_attention(q, k, v, heads), h, row(2));

        let x = kernels::modulated_layer_norm(
            h.clone(),
            self.norm2_weight.clone(),
            self.norm2_bias.clone(),
            EPS,
            dtype,
        );
        let q = self.cross_q.apply(x, Activation::Identity, dtype);
        let q = kernels::rms_norm(q, self.cross_norm_q.clone(), 0, dim, EPS, dtype);
        let attended = kernels::flash_attention(q, cross.k.clone(), cross.v.clone(), heads);
        let h = self.cross_out.residual(attended, h);

        let x = kernels::modulated_layer_norm(h.clone(), row(4).add_scalar(1.0), row(3), EPS, dtype);
        self.feed_forward(x, h, &row(5))
    }

    /// `h + ffn(x) * gate`, in token chunks of at most [`FFN_TOKENS`].
    fn feed_forward(&self, x: Tensor<2>, h: Tensor<2>, gate: &Tensor<1>) -> Tensor<2> {
        let run = |x: Tensor<2>, h: Tensor<2>| {
            let hidden = self.ffn_in.apply(x, Activation::GeluTanh, self.dtype);
            self.ffn_out.gated(hidden, h, gate.clone())
        };
        let tokens = x.dims()[0];
        if tokens <= FFN_TOKENS {
            return run(x, h);
        }
        let chunks = (0..tokens)
            .step_by(FFN_TOKENS)
            .map(|t0| {
                let t1 = (t0 + FFN_TOKENS).min(tokens);
                run(x.clone().slice(s![t0..t1]), h.clone().slice(s![t0..t1]))
            })
            .collect();
        Tensor::cat(chunks, 0)
    }
}

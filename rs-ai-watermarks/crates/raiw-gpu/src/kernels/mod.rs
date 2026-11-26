//! Hand-written CubeCL kernels, reached from `Tensor<D>` through a burn backend extension.
//!
//! - [`attention`]: FlashAttention-2 forward (online softmax, scores never leave the chip).
//! - [`gemm`]: register-tiled `x·wᵀ + b` with fused GELU / gated-residual epilogues.
//! - [`hgemm`]: the same product in packed f16 math for `[in, out]` f16 weights.
//! - [`norm`]: LayerNorm + modulation, and RMSNorm + rotary embedding.
//! - [`conv`]: causal 3-D convolution as an implicit GEMM (bias, residual, 2x upsampling fused).
//!
//! Every op is a [`KernelOps`] method implemented for `CubeBackend`; the generated `Fusion` arm
//! records it lazily in the fusion stream, so it orders correctly with surrounding burn ops.
//! The `Tensor` wrappers below accept GPU (CubeCL) tensors only; CPU callers keep burn ops.
//! Kernels store f16 or f32 and accumulate in f32.

pub mod attention;
pub mod conv;
pub mod gemm;
pub mod hgemm;
pub mod norm;

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::{DType, Device, Int, Shape, Tensor, TensorData};
use burn_cubecl::CubeBackend;
use burn_cubecl::tensor::CubeTensor;

pub use attention::FlashTile;
pub use conv::{ConvShape, conv_weight};
pub use gemm::{Activation, GemmTile, WeightLayout};

#[backend_extension(Cube, Fusion)]
pub trait KernelOps: Backend {
    /// Multi-head attention over row-major `[tokens, heads * head_dim]` tensors.
    #[fusion(dtype = q, shape = q)]
    fn flash_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        heads: usize,
        scale: f32,
        tile: FlashTile,
    ) -> FloatTensor<Self>;

    /// `activation(x·W + bias)` with `W` stored in `layout`.
    #[fusion(dtype = *dtype, shape = product_shape(x, weight, layout))]
    fn linear(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        activation: Activation,
        dtype: DType,
        layout: WeightLayout,
        tile: GemmTile,
    ) -> FloatTensor<Self>;

    /// `residual + (x·W + bias) * gate` (float32).
    #[fusion(dtype = residual, shape = residual)]
    fn linear_gated(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        residual: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        layout: WeightLayout,
    ) -> FloatTensor<Self>;

    /// `residual + x·W + bias` (float32).
    #[fusion(dtype = residual, shape = residual)]
    fn linear_residual(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        residual: FloatTensor<Self>,
        layout: WeightLayout,
    ) -> FloatTensor<Self>;

    /// `activation(x·(Q·diag(scale)) + bias)` for int8 `[in, out]` weights `Q`.
    #[fusion(dtype = *dtype, shape = Shape::new([x[0], weight[1]]))]
    fn linear_q8(
        x: FloatTensor<Self>,
        weight: IntTensor<Self>,
        scale: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        activation: Activation,
        dtype: DType,
    ) -> FloatTensor<Self>;

    /// `residual + (x·(Q·diag(scale)) + bias) * gate` (float32).
    #[fusion(dtype = residual, shape = residual)]
    fn linear_q8_gated(
        x: FloatTensor<Self>,
        weight: IntTensor<Self>,
        scale: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        residual: FloatTensor<Self>,
        gate: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// `residual + x·(Q·diag(scale)) + bias` (float32).
    #[fusion(dtype = residual, shape = residual)]
    fn linear_q8_residual(
        x: FloatTensor<Self>,
        weight: IntTensor<Self>,
        scale: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        residual: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// `x` in row-major memory order.
    #[fusion(dtype = x, shape = x)]
    fn contiguous(x: FloatTensor<Self>) -> FloatTensor<Self>;

    /// `LayerNorm(x) * mul + add`.
    #[fusion(dtype = *dtype, shape = x)]
    fn modulated_layer_norm(
        x: FloatTensor<Self>,
        mul: FloatTensor<Self>,
        add: FloatTensor<Self>,
        eps: f32,
        dtype: DType,
    ) -> FloatTensor<Self>;

    /// `RMSNorm(x[:, offset..offset + width]) * weight`.
    #[fusion(dtype = *dtype, shape = window_shape(x, *width))]
    fn rms_norm(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        offset: usize,
        width: usize,
        eps: f32,
        dtype: DType,
    ) -> FloatTensor<Self>;

    /// [`KernelOps::rms_norm`] followed by RoPE over `head_dim` heads (tables `[rows, head_dim / 2]`).
    #[fusion(dtype = *dtype, shape = window_shape(x, *width))]
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_rope(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        cos: FloatTensor<Self>,
        sin: FloatTensor<Self>,
        offset: usize,
        width: usize,
        head_dim: usize,
        eps: f32,
        dtype: DType,
    ) -> FloatTensor<Self>;
}

/// Extension methods continued: the VAE convolution.
#[backend_extension(Cube, Fusion)]
pub trait ConvOps: Backend {
    /// Causal convolution after `zeros` virtual zero frames ([`conv::causal_conv3d`]).
    #[fusion(dtype = x, shape = conv_output(x, *zeros, shape))]
    fn causal_conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        zeros: usize,
        shape: ConvShape,
    ) -> FloatTensor<Self>;

    /// [`ConvOps::causal_conv3d`] plus `residual` (same shape as the output).
    #[fusion(dtype = x, shape = residual)]
    fn causal_conv3d_residual(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        residual: FloatTensor<Self>,
        zeros: usize,
        shape: ConvShape,
    ) -> FloatTensor<Self>;
}

impl ConvOps for CubeBackend {
    fn causal_conv3d(
        x: CubeTensor,
        weight: CubeTensor,
        bias: CubeTensor,
        zeros: usize,
        shape: ConvShape,
    ) -> CubeTensor {
        conv::causal_conv3d(x, weight, bias, None, zeros, shape)
    }

    fn causal_conv3d_residual(
        x: CubeTensor,
        weight: CubeTensor,
        bias: CubeTensor,
        residual: CubeTensor,
        zeros: usize,
        shape: ConvShape,
    ) -> CubeTensor {
        conv::causal_conv3d(x, weight, bias, Some(residual), zeros, shape)
    }
}

fn conv_output(x: &Shape, zeros: usize, shape: &ConvShape) -> Shape {
    let scale = if shape.upsample { 2 } else { 1 };
    Shape::new([
        zeros + x[0] + 1 - shape.time_kernel,
        shape.out_channels,
        scale * x[2],
        scale * x[3],
    ])
}

/// The fusion metadata hands every ordinary argument by reference.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn product_shape(x: &Shape, weight: &Shape, layout: &WeightLayout) -> Shape {
    match layout {
        WeightLayout::OutIn => Shape::new([x[0], weight[0]]),
        WeightLayout::InOut => Shape::new([x[0], weight[1]]),
    }
}

/// One GEMM with its epilogue: the packed-f16 kernel for `[in, out]` f16 weights, the
/// register-tiled kernel for PyTorch's `[out, in]`.
fn dense(
    x: CubeTensor,
    weight: CubeTensor,
    bias: CubeTensor,
    epilogue: gemm::Epilogue,
    dtype: DType,
    layout: WeightLayout,
    tile: GemmTile,
) -> CubeTensor {
    match layout {
        WeightLayout::OutIn => gemm::linear(x, weight, bias, epilogue, dtype, tile),
        WeightLayout::InOut => hgemm::linear(x, weight, None, bias, epilogue, dtype),
    }
}

fn window_shape(x: &Shape, width: usize) -> Shape {
    Shape::new([x[0], width])
}

impl KernelOps for CubeBackend {
    fn flash_attention(
        q: CubeTensor,
        k: CubeTensor,
        v: CubeTensor,
        heads: usize,
        scale: f32,
        tile: FlashTile,
    ) -> CubeTensor {
        attention::flash_attention_with(q, k, v, heads, scale, tile)
    }

    fn linear(
        x: CubeTensor,
        weight: CubeTensor,
        bias: CubeTensor,
        activation: Activation,
        dtype: DType,
        layout: WeightLayout,
        tile: GemmTile,
    ) -> CubeTensor {
        let epilogue = gemm::Epilogue {
            activation,
            ..gemm::Epilogue::default()
        };
        dense(x, weight, bias, epilogue, dtype, layout, tile)
    }

    fn linear_gated(
        x: CubeTensor,
        weight: CubeTensor,
        bias: CubeTensor,
        residual: CubeTensor,
        gate: CubeTensor,
        layout: WeightLayout,
    ) -> CubeTensor {
        let tile = gemm::default_tile(x.dtype);
        let epilogue = gemm::Epilogue {
            residual: Some(residual),
            gate: Some(gate),
            ..gemm::Epilogue::default()
        };
        dense(x, weight, bias, epilogue, DType::F32, layout, tile)
    }

    fn linear_residual(
        x: CubeTensor,
        weight: CubeTensor,
        bias: CubeTensor,
        residual: CubeTensor,
        layout: WeightLayout,
    ) -> CubeTensor {
        let tile = gemm::default_tile(x.dtype);
        let epilogue = gemm::Epilogue {
            residual: Some(residual),
            ..gemm::Epilogue::default()
        };
        dense(x, weight, bias, epilogue, DType::F32, layout, tile)
    }

    fn linear_q8(
        x: CubeTensor,
        weight: CubeTensor,
        scale: CubeTensor,
        bias: CubeTensor,
        activation: Activation,
        dtype: DType,
    ) -> CubeTensor {
        let epilogue = gemm::Epilogue {
            activation,
            ..gemm::Epilogue::default()
        };
        hgemm::linear(x, weight, Some(scale), bias, epilogue, dtype)
    }

    fn linear_q8_gated(
        x: CubeTensor,
        weight: CubeTensor,
        scale: CubeTensor,
        bias: CubeTensor,
        residual: CubeTensor,
        gate: CubeTensor,
    ) -> CubeTensor {
        let epilogue = gemm::Epilogue {
            residual: Some(residual),
            gate: Some(gate),
            ..gemm::Epilogue::default()
        };
        hgemm::linear(x, weight, Some(scale), bias, epilogue, DType::F32)
    }

    fn linear_q8_residual(
        x: CubeTensor,
        weight: CubeTensor,
        scale: CubeTensor,
        bias: CubeTensor,
        residual: CubeTensor,
    ) -> CubeTensor {
        let epilogue = gemm::Epilogue {
            residual: Some(residual),
            ..gemm::Epilogue::default()
        };
        hgemm::linear(x, weight, Some(scale), bias, epilogue, DType::F32)
    }

    fn contiguous(x: CubeTensor) -> CubeTensor {
        burn_cubecl::kernel::into_contiguous(x)
    }

    fn modulated_layer_norm(
        x: CubeTensor,
        mul: CubeTensor,
        add: CubeTensor,
        eps: f32,
        dtype: DType,
    ) -> CubeTensor {
        norm::modulated_layer_norm(x, mul, add, eps, dtype)
    }

    fn rms_norm(
        x: CubeTensor,
        weight: CubeTensor,
        offset: usize,
        width: usize,
        eps: f32,
        dtype: DType,
    ) -> CubeTensor {
        norm::rms_norm_rope(x, offset, width, weight, None, width, eps, dtype)
    }

    fn rms_norm_rope(
        x: CubeTensor,
        weight: CubeTensor,
        cos: CubeTensor,
        sin: CubeTensor,
        offset: usize,
        width: usize,
        head_dim: usize,
        eps: f32,
        dtype: DType,
    ) -> CubeTensor {
        let tables = norm::RopeTables { cos, sin };
        norm::rms_norm_rope(x, offset, width, weight, Some(tables), head_dim, eps, dtype)
    }
}

type Kernels = Dispatch;

/// `softmax(q·kᵀ / sqrt(d))·v` per head; `q` `[seq_q, heads * d]`, `k`/`v` `[seq_kv, heads * d]`,
/// all f32 or all f16.
pub fn flash_attention(q: Tensor<2>, k: Tensor<2>, v: Tensor<2>, heads: usize) -> Tensor<2> {
    let tile = attention::default_tile(q.dtype());
    let scale = 1.0 / (q.dims()[1] as f32 / heads as f32).sqrt();
    flash_attention_tuned(q, k, v, heads, scale, tile)
}

/// [`flash_attention`] with an explicit scale and tile.
pub fn flash_attention_tuned(
    q: Tensor<2>,
    k: Tensor<2>,
    v: Tensor<2>,
    heads: usize,
    scale: f32,
    tile: FlashTile,
) -> Tensor<2> {
    Tensor::from_dispatch(<Kernels as KernelOps>::flash_attention(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        heads,
        scale,
        tile,
    ))
}

/// A dense layer's weight in the layout its GEMM kernel reads.
#[derive(Debug, Clone)]
pub struct DenseWeight {
    matrix: Matrix,
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant, reason = "one per layer")]
enum Matrix {
    Float {
        matrix: Tensor<2>,
        layout: WeightLayout,
    },
    /// `[in, out]` int8 values and `[out]` float32 scales.
    Int8 {
        values: Tensor<2, Int>,
        scale: Tensor<1>,
    },
}

impl DenseWeight {
    /// PyTorch's `[out, in]` weight, any float storage dtype.
    pub fn torch(weight: Tensor<2>) -> Self {
        Self {
            matrix: Matrix::Float {
                matrix: weight,
                layout: WeightLayout::OutIn,
            },
        }
    }

    /// The `[out, in]` weight transposed once into the packed-f16 kernel's `[in, out]` f16.
    pub fn packed(weight: Tensor<2>) -> Self {
        let matrix = weight.transpose().cast(DType::F16);
        Self {
            matrix: Matrix::Float {
                // materialized now rather than on every call
                matrix: Tensor::from_dispatch(<Kernels as KernelOps>::contiguous(matrix.into_dispatch())),
                layout: WeightLayout::InOut,
            },
        }
    }

    /// Symmetric per-output-channel int8 of a host `[out, in]` float32 weight (row-major),
    /// for the packed-f16 kernel: half the memory of f16, quantization error ≤ max|w| / 254.
    pub fn int8(weight: &[f32], (output, input): (usize, usize), device: &Device) -> Self {
        assert_eq!(weight.len(), output * input, "int8 weight shape");
        let scale: Vec<f32> = weight
            .chunks_exact(input)
            .map(|row| row.iter().fold(0f32, |m, w| m.max(w.abs())) / 127.0)
            .collect();
        let mut values = vec![0i8; output * input];
        for (o, (row, &s)) in weight.chunks_exact(input).zip(&scale).enumerate() {
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for (i, w) in row.iter().enumerate() {
                values[i * output + o] = (w * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
        // from_data widens to the default int dtype; the kernel reads bytes
        let values: Tensor<2, Int> = Tensor::from_data(TensorData::new(values, [input, output]), device);
        Self {
            matrix: Matrix::Int8 {
                values: values.cast(DType::I8),
                scale: Tensor::from_data(TensorData::new(scale, [output]), device),
            },
        }
    }

    /// A copy on `device` (host parking of resident weights between pipeline stages).
    #[must_use]
    pub fn to_device(&self, device: &Device) -> Self {
        let matrix = match &self.matrix {
            Matrix::Float { matrix, layout } => Matrix::Float {
                matrix: matrix.clone().to_device(device),
                layout: *layout,
            },
            Matrix::Int8 { values, scale } => {
                let values = values.clone().to_device(device);
                Matrix::Int8 {
                    // a cross-backend move may widen the int dtype again
                    values: if values.dtype() == DType::I8 {
                        values
                    } else {
                        values.cast(DType::I8)
                    },
                    scale: scale.clone().to_device(device),
                }
            }
        };
        Self { matrix }
    }

    /// The dtype the activations must share (f16 for int8 weights).
    pub fn dtype(&self) -> DType {
        match &self.matrix {
            Matrix::Float { matrix, .. } => matrix.dtype(),
            Matrix::Int8 { .. } => DType::F16,
        }
    }
}

/// `activation(x·W + bias)` for `x` `[m, k]` (the weight's dtype) and `bias` `[n]` float32;
/// the result has `dtype`.
pub fn linear(
    x: Tensor<2>,
    weight: &DenseWeight,
    bias: Tensor<1>,
    activation: Activation,
    dtype: DType,
) -> Tensor<2> {
    let tile = gemm::default_tile(x.dtype());
    linear_tuned(x, weight, bias, activation, dtype, tile)
}

/// [`linear`] with an explicit tile (for `[out, in]` weights).
pub fn linear_tuned(
    x: Tensor<2>,
    weight: &DenseWeight,
    bias: Tensor<1>,
    activation: Activation,
    dtype: DType,
    tile: GemmTile,
) -> Tensor<2> {
    let (x, bias) = (x.into_dispatch(), bias.into_dispatch());
    Tensor::from_dispatch(match &weight.matrix {
        Matrix::Float { matrix, layout } => <Kernels as KernelOps>::linear(
            x,
            matrix.clone().into_dispatch(),
            bias,
            activation,
            dtype,
            *layout,
            tile,
        ),
        Matrix::Int8 { values, scale } => <Kernels as KernelOps>::linear_q8(
            x,
            values.clone().into_dispatch(),
            scale.clone().into_dispatch(),
            bias,
            activation,
            dtype,
        ),
    })
}

/// `residual + (x·W + bias) * gate`, float32.
pub fn linear_gated(
    x: Tensor<2>,
    weight: &DenseWeight,
    bias: Tensor<1>,
    residual: Tensor<2>,
    gate: Tensor<1>,
) -> Tensor<2> {
    let (x, bias, residual, gate) = (
        x.into_dispatch(),
        bias.into_dispatch(),
        residual.into_dispatch(),
        gate.into_dispatch(),
    );
    Tensor::from_dispatch(match &weight.matrix {
        Matrix::Float { matrix, layout } => <Kernels as KernelOps>::linear_gated(
            x,
            matrix.clone().into_dispatch(),
            bias,
            residual,
            gate,
            *layout,
        ),
        Matrix::Int8 { values, scale } => <Kernels as KernelOps>::linear_q8_gated(
            x,
            values.clone().into_dispatch(),
            scale.clone().into_dispatch(),
            bias,
            residual,
            gate,
        ),
    })
}

/// `residual + x·W + bias`, float32.
pub fn linear_residual(
    x: Tensor<2>,
    weight: &DenseWeight,
    bias: Tensor<1>,
    residual: Tensor<2>,
) -> Tensor<2> {
    let (x, bias, residual) = (x.into_dispatch(), bias.into_dispatch(), residual.into_dispatch());
    Tensor::from_dispatch(match &weight.matrix {
        Matrix::Float { matrix, layout } => <Kernels as KernelOps>::linear_residual(
            x,
            matrix.clone().into_dispatch(),
            bias,
            residual,
            *layout,
        ),
        Matrix::Int8 { values, scale } => <Kernels as KernelOps>::linear_q8_residual(
            x,
            values.clone().into_dispatch(),
            scale.clone().into_dispatch(),
            bias,
            residual,
        ),
    })
}

/// `LayerNorm(x) * mul + add` over the rows of `x`; `mul`, `add` `[width]` float32.
pub fn modulated_layer_norm(
    x: Tensor<2>,
    mul: Tensor<1>,
    add: Tensor<1>,
    eps: f32,
    dtype: DType,
) -> Tensor<2> {
    Tensor::from_dispatch(<Kernels as KernelOps>::modulated_layer_norm(
        x.into_dispatch(),
        mul.into_dispatch(),
        add.into_dispatch(),
        eps,
        dtype,
    ))
}

/// `RMSNorm(x[:, offset..offset + width]) * weight` (`weight` `[width]` float32).
pub fn rms_norm(
    x: Tensor<2>,
    weight: Tensor<1>,
    offset: usize,
    width: usize,
    eps: f32,
    dtype: DType,
) -> Tensor<2> {
    Tensor::from_dispatch(<Kernels as KernelOps>::rms_norm(
        x.into_dispatch(),
        weight.into_dispatch(),
        offset,
        width,
        eps,
        dtype,
    ))
}

/// Causal convolution of `x` `[T, C_in, H, W]` after `zeros` virtual zero frames, with the
/// `[K, C_out]` weight from [`conv_weight`] and an optional residual added to the output.
pub fn causal_conv3d(
    x: Tensor<4>,
    weight: Tensor<2>,
    bias: Tensor<1>,
    residual: Option<Tensor<4>>,
    zeros: usize,
    shape: ConvShape,
) -> Tensor<4> {
    let (x, weight, bias) = (x.into_dispatch(), weight.into_dispatch(), bias.into_dispatch());
    Tensor::from_dispatch(match residual {
        Some(residual) => <Kernels as ConvOps>::causal_conv3d_residual(
            x,
            weight,
            bias,
            residual.into_dispatch(),
            zeros,
            shape,
        ),
        None => <Kernels as ConvOps>::causal_conv3d(x, weight, bias, zeros, shape),
    })
}

/// [`rms_norm`] followed by RoPE over `head_dim` heads with `[rows, head_dim / 2]` tables.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_rope(
    x: Tensor<2>,
    weight: Tensor<1>,
    (cos, sin): (Tensor<2>, Tensor<2>),
    offset: usize,
    width: usize,
    head_dim: usize,
    eps: f32,
    dtype: DType,
) -> Tensor<2> {
    Tensor::from_dispatch(<Kernels as KernelOps>::rms_norm_rope(
        x.into_dispatch(),
        weight.into_dispatch(),
        cos.into_dispatch(),
        sin.into_dispatch(),
        offset,
        width,
        head_dim,
        eps,
        dtype,
    ))
}

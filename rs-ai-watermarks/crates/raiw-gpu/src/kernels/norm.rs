//! Row normalizations fused with what follows them in a DiT block.
//!
//! - [`modulated_layer_norm`]: `LayerNorm(x) * mul + add` (adaLN: `mul = 1 + scale`,
//!   `add = shift`; an affine LayerNorm is `mul = gamma`, `add = beta`).
//! - [`rms_norm_rope`]: `RMSNorm(x) * weight`, then rotary embedding per head, reading a column
//!   window of a wider row (one projection out of a fused QKV output).
//!
//! One cube normalizes one row; each thread keeps its slice of the row in registers, so the
//! statistics take two passes (mean, then centered variance) without re-reading memory.
//! Inputs and outputs are row-major; statistics and arithmetic stay float32.

use burn::tensor::DType;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use half::f16;

/// Threads per row; must divide the row width in vectors.
const THREADS: usize = 128;
/// Elements per access.
const VECTOR: usize = 4;
/// Planes a cube may span at the smallest plane width (32).
const MAX_PLANES: usize = THREADS / 32;

/// Sum over the cube; every unit receives the total.
#[cube]
fn cube_sum(value: f32, partials: &mut Shared<[f32]>) -> f32 {
    let planes = CUBE_DIM / PLANE_DIM;
    let plane_total = plane_sum(value);
    sync_cube();
    if UNIT_POS_PLANE == 0 {
        partials[(UNIT_POS / PLANE_DIM) as usize] = plane_total;
    }
    sync_cube();
    let mut total = 0.0;
    for p in 0..planes {
        total += partials[p as usize];
    }
    total
}

#[cube(launch)]
fn modulated_layer_norm_kernel<I: Float, O: Float, N: Size>(
    x: &[Vector<I, N>],
    mul: &[Vector<f32, N>],
    add: &[Vector<f32, N>],
    out: &mut [Vector<O, N>],
    eps: f32,
    #[comptime] width: usize,
) {
    let per_thread = comptime!(width / VECTOR / THREADS);
    let row = CUBE_POS_X as usize * comptime!(width / VECTOR);
    let tid = UNIT_POS as usize;
    let mut partials = Shared::<[f32]>::new_slice(MAX_PLANES);
    let mut values = Array::<Vector<f32, N>>::new(per_thread);
    let mut sum = 0.0;
    #[unroll]
    for i in 0..per_thread {
        let v = Vector::<f32, N>::cast_from(x[row + tid + i * THREADS]);
        values[i] = v;
        sum += v.vector_sum();
    }
    let mean = cube_sum(sum, &mut partials) / comptime!(width as f32);
    let mut squares = 0.0;
    #[unroll]
    for i in 0..per_thread {
        let c = values[i] - Vector::new(mean);
        values[i] = c;
        squares += (c * c).vector_sum();
    }
    let inv = (cube_sum(squares, &mut partials) / comptime!(width as f32) + eps).inverse_sqrt();
    #[unroll]
    for i in 0..per_thread {
        let col = tid + i * THREADS;
        out[row + col] = Vector::cast_from(fma(values[i] * Vector::new(inv), mul[col], add[col]));
    }
}

#[cube(launch)]
fn rms_norm_rope_kernel<I: Float, O: Float, N: Size>(
    x: &[Vector<I, N>],
    weight: &[Vector<f32, N>],
    cos: &[Vector<f32, N>],
    sin: &[Vector<f32, N>],
    out: &mut [Vector<O, N>],
    x_stride: usize,
    x_offset: usize,
    eps: f32,
    #[comptime] width: usize,
    #[comptime] head_dim: usize,
    #[comptime] rope: bool,
) {
    let vectors = comptime!(width / VECTOR);
    let per_thread = comptime!(width / VECTOR / THREADS);
    let head_vectors = comptime!(head_dim / VECTOR);
    // cos/sin rows hold head_dim / 2 pair angles: two vectors' worth per head vector pair
    let table_vectors = comptime!(head_dim / 2 / VECTOR);
    let token = CUBE_POS_X as usize;
    let src = token * x_stride + x_offset;
    let tid = UNIT_POS as usize;
    let mut partials = Shared::<[f32]>::new_slice(MAX_PLANES);
    let mut values = Array::<Vector<f32, N>>::new(per_thread);
    let mut squares = 0.0;
    #[unroll]
    for i in 0..per_thread {
        let v = Vector::<f32, N>::cast_from(x[src + tid + i * THREADS]);
        values[i] = v;
        squares += (v * v).vector_sum();
    }
    let inv = (cube_sum(squares, &mut partials) / comptime!(width as f32) + eps).inverse_sqrt();
    #[unroll]
    for i in 0..per_thread {
        let col = tid + i * THREADS;
        let mut y = values[i] * Vector::new(inv) * weight[col];
        if comptime!(rope) {
            // a vector holds pairs (2p, 2p + 1) and (2p + 2, 2p + 3) of one head
            let pair_vector = col % head_vectors;
            let table = token * table_vectors + pair_vector / 2;
            let c = cos[table];
            let s = sin[table];
            let half = (pair_vector % 2) * 2;
            let mut rotated = Vector::<f32, N>::empty();
            #[unroll]
            for p in 0..comptime!(VECTOR / 2) {
                let even = y.extract(2 * p);
                let odd = y.extract(2 * p + 1);
                let cp = select(half == 0, c.extract(p), c.extract(p + 2));
                let sp = select(half == 0, s.extract(p), s.extract(p + 2));
                rotated.insert(2 * p, even * cp - odd * sp);
                rotated.insert(2 * p + 1, even * sp + odd * cp);
            }
            y = rotated;
        }
        out[token * vectors + col] = Vector::cast_from(y);
    }
}

fn row_width(t: &CubeTensor) -> (usize, usize) {
    let shape = t.meta.shape();
    assert_eq!(shape.num_dims(), 2, "row kernels take [rows, width]");
    (shape[0], shape[1])
}

/// One cube per row.
fn per_row(rows: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static(rows as u32, 1, 1),
        CubeDim::new_1d(THREADS as u32),
    )
}

/// `LayerNorm(x) * mul + add` over rows of `x` `[rows, width]`; `mul`, `add` `[width]` float32.
pub fn modulated_layer_norm(
    x: CubeTensor,
    mul: CubeTensor,
    add: CubeTensor,
    eps: f32,
    dtype: DType,
) -> CubeTensor {
    let (x, mul, add) = (into_contiguous(x), into_contiguous(mul), into_contiguous(add));
    let (rows, width) = row_width(&x);
    assert!(
        width.is_multiple_of(VECTOR * THREADS),
        "row width must be a multiple of {}",
        VECTOR * THREADS
    );
    assert_eq!(
        (mul.dtype, add.dtype),
        (DType::F32, DType::F32),
        "mul/add must be float32"
    );
    assert_eq!((mul.meta.num_elements(), add.meta.num_elements()), (width, width));
    let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), [rows, width].into(), dtype);
    let (count, dim) = per_row(rows);
    macro_rules! launch {
        ($i:ty, $o:ty) => {
            modulated_layer_norm_kernel::launch::<$i, $o>(
                &x.client,
                count,
                dim,
                VECTOR,
                x.clone().into_buffer_arg(),
                mul.clone().into_buffer_arg(),
                add.clone().into_buffer_arg(),
                out.clone().into_buffer_arg(),
                eps,
                width,
            )
        };
    }
    match (x.dtype, dtype) {
        (DType::F32, DType::F32) => launch!(f32, f32),
        (DType::F32, DType::F16) => launch!(f32, f16),
        (DType::F16, DType::F16) => launch!(f16, f16),
        (DType::F16, DType::F32) => launch!(f16, f32),
        other => panic!("layer norm supports f16/f32, got {other:?}"),
    }
    out
}

/// Rotary tables `[rows, head_dim / 2]` (cos, sin) float32.
#[derive(Debug, Clone)]
pub struct RopeTables {
    pub cos: CubeTensor,
    pub sin: CubeTensor,
}

/// `RMSNorm(x[:, offset..offset + width]) * weight`, then RoPE per `head_dim` head when `rope`
/// is given; `x` is `[rows, stride]` with `stride >= offset + width`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_rope(
    x: CubeTensor,
    offset: usize,
    width: usize,
    weight: CubeTensor,
    rope: Option<RopeTables>,
    head_dim: usize,
    eps: f32,
    dtype: DType,
) -> CubeTensor {
    let (x, weight) = (into_contiguous(x), into_contiguous(weight));
    let (rows, stride) = row_width(&x);
    assert!(offset + width <= stride, "column window past the row");
    assert!(
        width.is_multiple_of(VECTOR * THREADS)
            && offset.is_multiple_of(VECTOR)
            && stride.is_multiple_of(VECTOR)
    );
    assert!(head_dim.is_multiple_of(2 * VECTOR) && width.is_multiple_of(head_dim));
    assert_eq!(
        (weight.dtype, weight.meta.num_elements()),
        (DType::F32, width),
        "weight must be [width] float32"
    );
    let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), [rows, width].into(), dtype);
    let with_rope = rope.is_some();
    // without rope the tables are never read; the weight stands in for both
    let (cos, sin) = match rope {
        Some(RopeTables { cos, sin }) => {
            assert_eq!(
                cos.meta.num_elements(),
                rows * head_dim / 2,
                "rope tables must be [rows, head_dim / 2]"
            );
            (into_contiguous(cos), into_contiguous(sin))
        }
        None => (weight.clone(), weight.clone()),
    };
    let (count, dim) = per_row(rows);
    macro_rules! launch {
        ($i:ty, $o:ty) => {
            rms_norm_rope_kernel::launch::<$i, $o>(
                &x.client,
                count,
                dim,
                VECTOR,
                x.clone().into_buffer_arg(),
                weight.clone().into_buffer_arg(),
                cos.clone().into_buffer_arg(),
                sin.clone().into_buffer_arg(),
                out.clone().into_buffer_arg(),
                stride / VECTOR,
                offset / VECTOR,
                eps,
                width,
                head_dim,
                with_rope,
            )
        };
    }
    match (x.dtype, dtype) {
        (DType::F32, DType::F32) => launch!(f32, f32),
        (DType::F32, DType::F16) => launch!(f32, f16),
        (DType::F16, DType::F16) => launch!(f16, f16),
        (DType::F16, DType::F32) => launch!(f16, f32),
        other => panic!("rms norm supports f16/f32, got {other:?}"),
    }
    out
}

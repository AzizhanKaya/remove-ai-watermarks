//! Half-precision GEMM in packed f16 math: `out = epilogue(a·b + bias)` with `b` `[k, n]`.
//!
//! RDNA executes two f16 FMAs per `v_pk_fma_f16`, twice the `v_fma_mix_f32` rate the
//! [`super::gemm`] kernel is bound by. The product is an outer product per `k`: thread `(r, c)`
//! holds rows `8r..8r + 8` and columns `8c..8c + 8` as eight f16x8 partial sums, multiplies a
//! broadcast `a[row, k]` by the f16x8 `b[k, 8c..8c + 8]`, and flushes the partials into float32
//! accumulators every [`CHUNK`] values of `k`. With unit-scale activations the chunked f16
//! sums stay below the rounding bfloat16 inputs already carry in Python's CUDA path.
//!
//! Layout: `a` `[m, k]` row-major, `b` `[k, n]` (the weight transposed once at load), both f16;
//! `k` and `n` multiples of 8. One cube owns a 128x128 output tile and walks `k` in 32-deep
//! slabs through shared memory with the next slab prefetched into registers.

use burn::tensor::DType;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use half::f16;

use super::gemm::{Activation, Epilogue, gelu_tanh};

const VECTOR: usize = 8;
const ROW_THREADS: usize = 16;
const COL_THREADS: usize = 16;
const THREADS: usize = ROW_THREADS * COL_THREADS;
const ROWS: usize = 8;
const BLOCK_ROWS: usize = ROW_THREADS * ROWS;
const BLOCK_COLS: usize = COL_THREADS * VECTOR;
const SLAB: usize = 32;
/// `k` values per f16 partial sum before it joins the float32 accumulator.
pub const CHUNK: usize = 16;
/// Shared row strides in vectors (one vector of padding).
const LDA: usize = SLAB / VECTOR + 1;
const LDB: usize = BLOCK_COLS / VECTOR + 1;

#[cube(launch)]
fn hgemm_kernel<F: Float, O: Float, N: Size>(
    a: &[Vector<F, N>],
    b: &[Vector<F, N>],
    bias: &[f32],
    residual: ComptimeOption<&[f32]>,
    gate: ComptimeOption<&[f32]>,
    out: &mut [O],
    m_begin: usize,
    m: usize,
    cols: usize,
    k: usize,
    #[comptime] activation: Activation,
) {
    let mut a_tile = Shared::<[Vector<F, N>]>::new_aligned_slice(comptime!(BLOCK_ROWS * LDA), 16usize);
    let mut b_tile = Shared::<[Vector<F, N>]>::new_aligned_slice(comptime!(SLAB * LDB), 16usize);

    let tid = UNIT_POS as usize;
    let r = tid / COL_THREADS;
    let c = tid % COL_THREADS;
    let m0 = m_begin + CUBE_POS_Y as usize * BLOCK_ROWS;
    let n0 = CUBE_POS_X as usize * BLOCK_COLS;
    let (kv, nv) = (k / VECTOR, cols / VECTOR);

    let mut acc = Array::<Vector<f32, N>>::new(ROWS);
    let mut part = Array::<Vector<F, N>>::new(ROWS);
    #[unroll]
    for i in 0..ROWS {
        acc[i] = Vector::new(0.0);
        part[i] = Vector::new(F::new(0.0));
    }

    // two A and two B vectors per thread per slab, at fixed offsets
    let a_loads = comptime!(BLOCK_ROWS * SLAB / VECTOR / THREADS);
    let b_loads = comptime!(SLAB * BLOCK_COLS / VECTOR / THREADS);
    let mut a_regs = Array::<Vector<F, N>>::new(a_loads);
    let mut b_regs = Array::<Vector<F, N>>::new(b_loads);
    fetch::<F, N>(
        a,
        &mut a_regs,
        tid,
        m0,
        m,
        0,
        kv,
        comptime!(SLAB / VECTOR),
        a_loads,
    );
    fetch::<F, N>(
        b,
        &mut b_regs,
        tid,
        0,
        k,
        n0 / VECTOR,
        nv,
        comptime!(BLOCK_COLS / VECTOR),
        b_loads,
    );

    let slabs = k.div_ceil(SLAB);
    for s in 0..slabs {
        sync_cube();
        #[unroll]
        for it in 0..a_loads {
            let idx = tid + it * THREADS;
            a_tile[(idx / comptime!(SLAB / VECTOR)) * LDA + idx % comptime!(SLAB / VECTOR)] = a_regs[it];
        }
        #[unroll]
        for it in 0..b_loads {
            let idx = tid + it * THREADS;
            b_tile[(idx / comptime!(BLOCK_COLS / VECTOR)) * LDB + idx % comptime!(BLOCK_COLS / VECTOR)] =
                b_regs[it];
        }
        sync_cube();
        let next = (s + 1) * SLAB;
        if next < k {
            fetch::<F, N>(
                a,
                &mut a_regs,
                tid,
                m0,
                m,
                next / VECTOR,
                kv,
                comptime!(SLAB / VECTOR),
                a_loads,
            );
            fetch::<F, N>(
                b,
                &mut b_regs,
                tid,
                next,
                k,
                n0 / VECTOR,
                nv,
                comptime!(BLOCK_COLS / VECTOR),
                b_loads,
            );
        }

        #[unroll]
        for chunk in 0..comptime!(SLAB / CHUNK) {
            #[unroll]
            for kk in 0..comptime!(CHUNK / VECTOR) {
                let kvec = chunk * comptime!(CHUNK / VECTOR) + kk;
                let mut av = Array::<Vector<F, N>>::new(ROWS);
                #[unroll]
                for i in 0..ROWS {
                    av[i] = a_tile[(r * ROWS + i) * LDA + kvec];
                }
                #[unroll]
                for kl in 0..VECTOR {
                    let bv = b_tile[(kvec * VECTOR + kl) * LDB + c];
                    #[unroll]
                    for i in 0..ROWS {
                        part[i] = fma(Vector::new(av[i].extract(kl)), bv, part[i]);
                    }
                }
            }
            #[unroll]
            for i in 0..ROWS {
                acc[i] += Vector::<f32, N>::cast_from(part[i]);
                part[i] = Vector::new(F::new(0.0));
            }
        }
    }

    #[unroll]
    for i in 0..ROWS {
        let row = m0 + r * ROWS + i;
        if row < m {
            let values = acc[i];
            #[unroll]
            for e in 0..VECTOR {
                let col = n0 + c * VECTOR + e;
                if col < cols {
                    let mut y = values.extract(e) + bias[col];
                    if comptime!(activation == Activation::GeluTanh) {
                        y = gelu_tanh(y);
                    }
                    let at = row * cols + col;
                    #[comptime]
                    match residual {
                        ComptimeOption::Some(residual) =>
                        {
                            #[comptime]
                            match gate {
                                ComptimeOption::Some(gate) => y = residual[at] + y * gate[col],
                                ComptimeOption::None => y += residual[at],
                            }
                        }
                        ComptimeOption::None => {}
                    }
                    out[at] = O::cast_from(y);
                }
            }
        }
    }
}

/// `loads` vectors of a `[rows, width]`-vector slab window: row `row0 + idx / width`, vector
/// column `col0 + idx % width`, zero outside `[rows, cols]` (branch-free).
#[cube]
#[allow(clippy::too_many_arguments)]
fn fetch<F: Float, N: Size>(
    src: &[Vector<F, N>],
    regs: &mut Array<Vector<F, N>>,
    tid: usize,
    row0: usize,
    rows: usize,
    col0: usize,
    cols: usize,
    #[comptime] width: usize,
    #[comptime] loads: usize,
) {
    #[unroll]
    for it in 0..loads {
        let idx = tid + it * THREADS;
        let row = row0 + idx / width;
        let col = col0 + idx % width;
        let ok = row < rows && col < cols;
        let value = src[select(ok, row * cols + col, 0)];
        regs[it] = select(ok, value, Vector::new(F::new(0.0)));
    }
}

/// Work per dispatch, inside the amdgpu ring timeout at any plausible speed.
const DISPATCH_FLOPS: usize = 200_000_000_000;

/// `epilogue(x·w + bias)` for f16 `x` `[m, k]` and `w` `[k, n]`, `bias` `[n]` float32.
pub fn linear(
    x: CubeTensor,
    weight: CubeTensor,
    bias: CubeTensor,
    epilogue: Epilogue,
    dtype: DType,
) -> CubeTensor {
    let (x, weight, bias) = (into_contiguous(x), into_contiguous(weight), into_contiguous(bias));
    let [m, k] = dims(&x);
    let [wk, n] = dims(&weight);
    assert_eq!(k, wk, "x and weight disagree on the reduction size");
    assert!(
        x.dtype == DType::F16 && weight.dtype == DType::F16,
        "packed GEMM is f16"
    );
    assert!(
        k % VECTOR == 0 && n % VECTOR == 0,
        "k and n must be multiples of {VECTOR}"
    );
    assert_eq!(
        (bias.dtype, bias.meta.num_elements()),
        (DType::F32, n),
        "bias must be [n] float32"
    );
    let residual = epilogue.residual.map(|r| {
        let r = into_contiguous(r);
        assert_eq!(
            (r.dtype, dims(&r)),
            (DType::F32, [m, n]),
            "residual must be [m, n] float32"
        );
        r
    });
    let gate = epilogue.gate.map(|g| {
        let g = into_contiguous(g);
        assert_eq!(
            (g.dtype, g.meta.num_elements()),
            (DType::F32, n),
            "gate must be [n] float32"
        );
        g
    });
    assert!(gate.is_none() || residual.is_some(), "a gate needs a residual");
    assert!(
        residual.is_none() || dtype == DType::F32,
        "a fused residual stays float32"
    );

    let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), [m, n].into(), dtype);
    let client = x.client.clone();
    let row_blocks = m.div_ceil(BLOCK_ROWS);
    let per_dispatch = (DISPATCH_FLOPS / (2 * BLOCK_ROWS * n * k).max(1)).max(1);
    for first in (0..row_blocks).step_by(per_dispatch) {
        let count = CubeCount::Static(
            n.div_ceil(BLOCK_COLS) as u32,
            (row_blocks - first).min(per_dispatch) as u32,
            1,
        );
        macro_rules! launch {
            ($out:ty) => {
                hgemm_kernel::launch::<f16, $out>(
                    &client,
                    count,
                    CubeDim::new_1d(THREADS as u32),
                    VECTOR,
                    x.clone().into_buffer_arg(),
                    weight.clone().into_buffer_arg(),
                    bias.clone().into_buffer_arg(),
                    residual.clone().map(|r| r.into_buffer_arg()).into(),
                    gate.clone().map(|g| g.into_buffer_arg()).into(),
                    out.clone().into_buffer_arg(),
                    first * BLOCK_ROWS,
                    m,
                    n,
                    k,
                    epilogue.activation,
                )
            };
        }
        match dtype {
            DType::F16 => launch!(f16),
            DType::F32 => launch!(f32),
            other => panic!("packed GEMM writes f16 or f32, got {other:?}"),
        }
    }
    out
}

fn dims(t: &CubeTensor) -> [usize; 2] {
    let shape = t.meta.shape();
    assert_eq!(shape.num_dims(), 2, "linear takes 2-D tensors");
    [shape[0], shape[1]]
}

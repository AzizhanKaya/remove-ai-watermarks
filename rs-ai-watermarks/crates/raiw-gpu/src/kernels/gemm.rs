//! Register-tiled GEMM with a fused epilogue: `out = epilogue(a·bᵀ + bias)`.
//!
//! `a` is `[m, k]` activations and `b` is `[n, k]`, PyTorch's `Linear.weight` layout, so both
//! operands are read along `k` and no weight transpose is ever materialized. One cube owns a
//! `BR x BC` output tile and walks `k` in `BK`-wide slabs staged through shared memory, with the
//! next slab prefetched into registers while the current one computes. Thread `(r, c)` owns
//! rows `r * MR + i` and columns `c + TC * j` (interleaved, so a plane's shared reads stay
//! conflict-free). Storage is f16 or f32, accumulation f32 (RDNA folds the f16 conversions
//! into `v_fma_mix_f32`).
//!
//! The tile is fixed per dtype rather than autotuned: autotuning burn's matmul on the Wan
//! feed-forward shapes ran a candidate past the amdgpu ring timeout and reset the GPU.
//!
//! Epilogues: GELU (tanh form), and the gated residual `residual + y * gate` that closes every
//! Wan transformer sub-block.

use burn::tensor::DType;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use half::f16;

/// Tile shape of one GEMM cube.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmTile {
    pub row_threads: usize,
    /// Thread columns (a power of two).
    pub col_threads: usize,
    /// Output rows per thread (`MR`).
    pub rows: usize,
    /// Output columns per thread (`MC`).
    pub cols: usize,
    /// Reduction slab (`BK`).
    pub depth: usize,
    /// Elements per global and shared-memory access along `k`.
    pub vector: usize,
    /// Extra vectors per shared row, against bank conflicts.
    pub pad: usize,
}

impl GemmTile {
    pub const fn block_rows(&self) -> usize {
        self.row_threads * self.rows
    }

    pub const fn block_cols(&self) -> usize {
        self.col_threads * self.cols
    }

    pub const fn threads(&self) -> usize {
        self.row_threads * self.col_threads
    }

    pub const fn shared_bytes(&self, elem: usize) -> usize {
        (self.block_rows() + self.block_cols()) * (self.depth + self.pad * self.vector) * elem
    }
}

/// 128x128 outputs, 8x8 per thread, 32-deep slabs, 8-wide accesses (20 KB shared at f16).
pub const TILE_F16: GemmTile = GemmTile {
    row_threads: 16,
    col_threads: 16,
    rows: 8,
    cols: 8,
    depth: 32,
    vector: 8,
    pad: 1,
};

/// The f32 tile: 4-wide accesses, 40 KB shared.
pub const TILE_F32: GemmTile = GemmTile {
    vector: 4,
    ..TILE_F16
};

/// The default tile for a storage dtype.
pub fn default_tile(dtype: DType) -> GemmTile {
    if dtype == DType::F16 { TILE_F16 } else { TILE_F32 }
}

/// How a weight matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WeightLayout {
    /// PyTorch's `Linear.weight`, `[out, in]`: the register-tiled kernel.
    #[default]
    OutIn,
    /// `[in, out]` f16 (transposed once at load): the packed-math [`super::hgemm`] kernel.
    InOut,
}

/// Pointwise activation applied after the bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Activation {
    #[default]
    Identity,
    /// `gelu(x, approximate="tanh")`, evaluated as `x * sigmoid(2·sqrt(2/π)·(x + 0.044715x³))`,
    /// which is the same function and cannot overflow into NaN.
    GeluTanh,
}

#[cube(launch)]
fn gemm_kernel<F: Float, O: Float, N: Size>(
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
    lda: usize,
    ldb: usize,
    ldo: usize,
    #[comptime] tile: GemmTile,
    #[comptime] activation: Activation,
) {
    let tc = comptime!(tile.col_threads);
    let mr = comptime!(tile.rows);
    let mc = comptime!(tile.cols);
    let br = comptime!(tile.row_threads * tile.rows);
    let bc = comptime!(tile.col_threads * tile.cols);
    let width = N::value();
    let kv = comptime!(tile.depth / width);
    let ld = comptime!(tile.depth / width + tile.pad);

    let mut a_tile = Shared::<[Vector<F, N>]>::new_slice(comptime!(br * ld));
    let mut b_tile = Shared::<[Vector<F, N>]>::new_slice(comptime!(bc * ld));

    let tid = UNIT_POS as usize;
    let row0 = (tid / tc) * mr;
    let col_id = tid % tc;
    let m0 = m_begin + CUBE_POS_Y as usize * br;
    let n0 = CUBE_POS_X as usize * bc;
    let k_vectors = k / width;

    let mut acc = Array::<f32>::new(comptime!(mr * mc));
    #[unroll]
    for x in 0..comptime!(mr * mc) {
        acc[x] = 0.0;
    }

    let a_loads = comptime!((tile.row_threads * tile.rows * tile.depth / width).div_ceil(tile.threads()));
    let b_loads = comptime!((tile.col_threads * tile.cols * tile.depth / width).div_ceil(tile.threads()));
    let mut a_regs = Array::<Vector<F, N>>::new(a_loads);
    let mut b_regs = Array::<Vector<F, N>>::new(b_loads);
    fetch_slab::<F, N>(a, &mut a_regs, m0, m, 0, k_vectors, lda, tid, br, tile);
    fetch_slab::<F, N>(b, &mut b_regs, n0, cols, 0, k_vectors, ldb, tid, bc, tile);

    let slabs = k.div_ceil(comptime!(tile.depth));
    for s in 0..slabs {
        sync_cube();
        stash_slab::<F, N>(&a_regs, &mut a_tile, tid, br, tile);
        stash_slab::<F, N>(&b_regs, &mut b_tile, tid, bc, tile);
        sync_cube();
        let next = (s + 1) * kv;
        if next < k_vectors {
            fetch_slab::<F, N>(a, &mut a_regs, m0, m, next, k_vectors, lda, tid, br, tile);
            fetch_slab::<F, N>(b, &mut b_regs, n0, cols, next, k_vectors, ldb, tid, bc, tile);
        }

        #[unroll]
        for kk in 0..kv {
            let mut av = Array::<Vector<f32, N>>::new(mr);
            #[unroll]
            for i in 0..mr {
                av[i] = Vector::cast_from(a_tile[(row0 + i) * ld + kk]);
            }
            #[unroll]
            for j in 0..mc {
                let bv = Vector::<f32, N>::cast_from(b_tile[(col_id + j * tc) * ld + kk]);
                #[unroll]
                for i in 0..mr {
                    let mut sum = acc[i * mc + j];
                    let ai = av[i];
                    #[unroll]
                    for e in 0..width {
                        sum = fma(ai.extract(e), bv.extract(e), sum);
                    }
                    acc[i * mc + j] = sum;
                }
            }
        }
    }

    #[unroll]
    for j in 0..mc {
        let col = n0 + col_id + j * tc;
        if col < cols {
            let shift = bias[col];
            #[unroll]
            for i in 0..mr {
                let row = m0 + row0 + i;
                if row < m {
                    let mut y = acc[i * mc + j] + shift;
                    if comptime!(activation == Activation::GeluTanh) {
                        y = gelu_tanh(y);
                    }
                    let at = row * ldo + col;
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

/// `gelu(y, approximate="tanh") = y · sigmoid(2·sqrt(2/π)·(y + 0.044715·y³))`.
#[cube]
pub(crate) fn gelu_tanh(y: f32) -> f32 {
    let inner = comptime!((8.0f32 / core::f32::consts::PI).sqrt()) * (y + 0.044_715 * y * y * y);
    y / (1.0 + (-inner).exp())
}

/// Load this thread's share of the `rows x BK` slab starting at row `row0`, vector `k0`.
#[cube]
fn fetch_slab<F: Float, N: Size>(
    src: &[Vector<F, N>],
    regs: &mut Array<Vector<F, N>>,
    row0: usize,
    rows: usize,
    k0: usize,
    k_vectors: usize,
    stride: usize,
    tid: usize,
    #[comptime] block: usize,
    #[comptime] tile: GemmTile,
) {
    let width = N::value();
    let threads = comptime!(tile.threads());
    let kv = comptime!(tile.depth / width);
    let total = comptime!(block * tile.depth / width);
    // branch-free: masked lanes read element 0 and keep zero
    #[unroll]
    for it in 0..comptime!(total.div_ceil(threads)) {
        let idx = tid + it * threads;
        let row = row0 + idx / kv;
        let col = k0 + idx % kv;
        let ok = (comptime!(total % threads == 0) || idx < total) && row < rows && col < k_vectors;
        let value = src[select(ok, row * stride + col, 0)];
        regs[it] = select(ok, value, Vector::new(F::new(0.0)));
    }
}

/// Store [`fetch_slab`] registers into the padded shared slab.
#[cube]
fn stash_slab<F: Float, N: Size>(
    regs: &Array<Vector<F, N>>,
    dst: &mut Shared<[Vector<F, N>]>,
    tid: usize,
    #[comptime] block: usize,
    #[comptime] tile: GemmTile,
) {
    let width = N::value();
    let threads = comptime!(tile.threads());
    let kv = comptime!(tile.depth / width);
    let ld = comptime!(tile.depth / width + tile.pad);
    let total = comptime!(block * tile.depth / width);
    #[unroll]
    for it in 0..comptime!(total.div_ceil(threads)) {
        let idx = tid + it * threads;
        if idx < total {
            dst[(idx / kv) * ld + idx % kv] = regs[it];
        }
    }
}

/// Work per GEMM dispatch, well inside the amdgpu ring timeout at any plausible speed.
const DISPATCH_FLOPS: usize = 200_000_000_000;

/// Optional epilogue operands of [`linear`].
#[derive(Debug, Clone, Default)]
pub struct Epilogue {
    pub activation: Activation,
    /// `[m, n]` float32 added after the activation.
    pub residual: Option<CubeTensor>,
    /// `[n]` float32 scaling the product before the residual add.
    pub gate: Option<CubeTensor>,
}

/// `epilogue(x·wᵀ + bias)` for `x` `[m, k]`, `w` `[n, k]` (same dtype), `bias` `[n]` float32;
/// the output has `dtype` (float32 whenever a residual is fused).
pub fn linear(
    x: CubeTensor,
    weight: CubeTensor,
    bias: CubeTensor,
    epilogue: Epilogue,
    dtype: DType,
    tile: GemmTile,
) -> CubeTensor {
    let (x, weight, bias) = (into_contiguous(x), into_contiguous(weight), into_contiguous(bias));
    let [m, k] = dims(&x);
    let [n, wk] = dims(&weight);
    assert_eq!(k, wk, "x and weight disagree on the reduction size");
    assert_eq!(x.dtype, weight.dtype, "x and weight dtypes differ");
    assert_eq!(bias.dtype, DType::F32, "bias must be float32");
    assert_eq!(bias.meta.num_elements(), n, "bias length");
    assert!(k % tile.vector == 0, "k must be a multiple of the vector width");
    assert!(tile.depth.is_multiple_of(tile.vector) && tile.col_threads.is_power_of_two());
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
    let cube_dim = CubeDim::new_1d(tile.threads() as u32);
    let col_blocks = n.div_ceil(tile.block_cols()) as u32;
    let row_blocks = m.div_ceil(tile.block_rows());
    let per_dispatch = (DISPATCH_FLOPS / (2 * tile.block_rows() * n * k).max(1)).max(1);
    for first in (0..row_blocks).step_by(per_dispatch) {
        let count = CubeCount::Static(col_blocks, (row_blocks - first).min(per_dispatch) as u32, 1);
        macro_rules! launch {
            ($elem:ty, $out:ty) => {
                gemm_kernel::launch::<$elem, $out>(
                    &client,
                    count,
                    cube_dim,
                    tile.vector,
                    x.clone().into_buffer_arg(),
                    weight.clone().into_buffer_arg(),
                    bias.clone().into_buffer_arg(),
                    residual.clone().map(|r| r.into_buffer_arg()).into(),
                    gate.clone().map(|g| g.into_buffer_arg()).into(),
                    out.clone().into_buffer_arg(),
                    first * tile.block_rows(),
                    m,
                    n,
                    k,
                    k / tile.vector,
                    k / tile.vector,
                    n,
                    tile,
                    epilogue.activation,
                )
            };
        }
        match (x.dtype, dtype) {
            (DType::F16, DType::F16) => launch!(f16, f16),
            (DType::F16, DType::F32) => launch!(f16, f32),
            (DType::F32, DType::F32) => launch!(f32, f32),
            (DType::F32, DType::F16) => launch!(f32, f16),
            other => panic!("linear supports f16/f32 storage, got {other:?}"),
        }
        let rows = (row_blocks - first).min(per_dispatch) * tile.block_rows();
        super::submit_heavy(&client, 2 * rows * n * k);
    }
    out
}

fn dims(t: &CubeTensor) -> [usize; 2] {
    let shape = t.meta.shape();
    assert_eq!(shape.num_dims(), 2, "linear takes 2-D tensors");
    [shape[0], shape[1]]
}

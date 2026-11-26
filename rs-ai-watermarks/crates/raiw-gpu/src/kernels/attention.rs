//! FlashAttention-2 forward for GPUs without matrix cores (written for RDNA1 under RADV).
//!
//! One cube owns `BR` query rows of one head and streams the keys and values through shared
//! memory `BC` rows at a time, keeping the running row max, row sum and output in registers
//! (online softmax), so the `[seq_q, seq_kv]` scores never reach global memory.
//!
//! Layout: queries, keys, values and the output are row-major `[tokens, heads * head_dim]`
//! (the projections' natural layout), so no head transposes run before or after.
//! Threads form a `TR x TC` grid: thread `(r, c)` owns query rows `MR·r..MR·r + MR`, key
//! columns `c + TC·j` and output vectors `c + TC·j`. The `TC` threads of one row are adjacent
//! lanes of one plane, so the row max reduces with `TC.log2()` lane shuffles (DPP on RDNA) and
//! the row sum once at the end. Interleaved columns keep a plane's shared reads on distinct
//! banks or broadcasts; P is stored transposed so a thread's per-key P reads merge into one.
//! Keys and values can share one shared buffer, and the next tile's rows are prefetched into
//! registers while the current tile computes.
//!
//! Storage `F` is `f32` (parity) or `f16` (half the bandwidth and shared memory). Q·Kᵀ and the
//! softmax accumulate in `f32` (`v_fma_mix_f32` folds the f16 conversions); P·V can sum
//! `pv_chunk` keys in packed f16 before joining the `f32` output.

use burn::tensor::DType;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use half::f16;

/// Tile shape of one attention cube.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlashTile {
    /// Thread rows (`TR`).
    pub row_threads: usize,
    /// Thread columns sharing one query row (`TC`, a power of two within one plane).
    pub col_threads: usize,
    /// Query rows per thread (`MR`).
    pub rows: usize,
    /// Key columns per thread (`MC`).
    pub cols: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Elements per global and shared-memory access.
    pub vector: usize,
    /// Extra vectors per shared Q/K/V row, against bank conflicts.
    pub pad: usize,
    /// Keys and values take turns in one shared buffer (one extra barrier per tile) so larger
    /// tiles keep two cubes resident per compute unit.
    pub shared_kv: bool,
    /// Keys per storage-precision partial sum of P·V before it joins the f32 output (packed
    /// f16 FMAs, two per instruction); 0 accumulates every product in f32.
    pub pv_chunk: usize,
}

impl FlashTile {
    pub const fn query_rows(&self) -> usize {
        self.row_threads * self.rows
    }

    pub const fn key_rows(&self) -> usize {
        self.col_threads * self.cols
    }

    pub const fn threads(&self) -> usize {
        self.row_threads * self.col_threads
    }

    /// Shared-memory bytes for storage elements of `elem` bytes.
    pub const fn shared_bytes(&self, elem: usize) -> usize {
        let row = self.head_dim + self.pad * self.vector;
        let kv_buffers = if self.shared_kv { 1 } else { 2 };
        ((self.query_rows() + kv_buffers * self.key_rows()) * row + self.key_rows() * (self.query_rows() + 4))
            * elem
    }
}

/// Default f16 tile for head dimension 128: 64x32 scores, 4x2 per thread, 8-wide accesses,
/// 29 KB shared (two cubes per compute unit), P·V in packed f16 over 16 keys. Fastest of the
/// RX 5500 XT sweep (`tests/dit_kernels.rs`): 4.6 TFLOP/s at 4096 tokens (3.9 with f32 P·V).
/// The partial sums keep the output within 1.5e-3 of float64, under bfloat16 P's own error.
pub const TILE_F16: FlashTile = FlashTile {
    row_threads: 16,
    col_threads: 16,
    rows: 4,
    cols: 2,
    head_dim: 128,
    vector: 8,
    pad: 1,
    shared_kv: true,
    pv_chunk: 16,
};

/// Default f32 tile for head dimension 128: same shape, 4-wide, f32 P·V, 58 KB shared (one
/// cube per compute unit), 2.2 TFLOP/s.
pub const TILE_F32: FlashTile = FlashTile {
    vector: 4,
    pv_chunk: 0,
    ..TILE_F16
};

/// Masked scores sit far below any real logit but stay finite, so `exp(m_old - m_new)` never
/// sees `inf - inf`.
const MASKED: f32 = -1.0e30;

#[cube(launch)]
fn flash_attention_kernel<F: Float, N: Size>(
    q: &[Vector<F, N>],
    k: &[Vector<F, N>],
    v: &[Vector<F, N>],
    out: &mut [Vector<F, N>],
    q_begin: usize,
    seq_q: usize,
    seq_kv: usize,
    q_stride: usize,
    kv_stride: usize,
    out_stride: usize,
    scale: f32,
    #[comptime] tile: FlashTile,
) {
    let tc = comptime!(tile.col_threads);
    let mr = comptime!(tile.rows);
    let mc = comptime!(tile.cols);
    let threads = comptime!(tile.row_threads * tile.col_threads);
    let br = comptime!(tile.row_threads * tile.rows);
    let bc = comptime!(tile.col_threads * tile.cols);
    let width = N::value();
    // vectors per head row, per padded shared row, and per thread in the output
    let dv = comptime!(tile.head_dim / width);
    let ld = comptime!(tile.head_dim / width + tile.pad);
    let ov = comptime!(tile.head_dim / width / tile.col_threads);
    // P is stored transposed, [key][query]; a thread's MR rows are adjacent and 8-byte
    // aligned, so its per-key P reads merge into one load
    let ldp = comptime!(tile.row_threads * tile.rows + 4);
    let shuffles = comptime!(tile.col_threads.trailing_zeros() as usize);

    let mut qs = Shared::<[Vector<F, N>]>::new_slice(comptime!(br * ld));
    let mut ks = Shared::<[Vector<F, N>]>::new_slice(comptime!(bc * ld));
    let mut vs = Shared::<[Vector<F, N>]>::new_slice(comptime!(if tile.shared_kv { 1 } else { bc * ld }));
    let mut ps = Shared::<[F]>::new_slice(comptime!(bc * ldp));

    let tid = UNIT_POS as usize;
    let row0 = (tid / tc) * mr;
    let col_id = tid % tc;
    let head = CUBE_POS_Y as usize * dv;
    let q0 = q_begin + CUBE_POS_X as usize * br;

    // Q tile, zero past the last query row
    let q_loads = comptime!((br * dv).div_ceil(threads));
    #[unroll]
    for it in 0..q_loads {
        let idx = tid + it * threads;
        if idx < br * dv {
            let row = idx / dv;
            let col = idx % dv;
            let mut value = Vector::<F, N>::new(F::new(0.0));
            if q0 + row < seq_q {
                value = q[(q0 + row) * q_stride + head + col];
            }
            qs[row * ld + col] = value;
        }
    }

    let mut row_max = Array::<f32>::new(mr);
    let mut row_sum = Array::<f32>::new(mr);
    let mut acc = Array::<Vector<f32, N>>::new(comptime!(mr * ov));
    #[unroll]
    for i in 0..mr {
        row_max[i] = MASKED;
        row_sum[i] = 0.0;
        #[unroll]
        for j in 0..ov {
            acc[i * ov + j] = Vector::new(0.0);
        }
    }
    let mut scores = Array::<f32>::new(comptime!(mr * mc));

    // Keys and values of the next tile wait in registers while this one computes
    let kv_loads = comptime!((bc * dv).div_ceil(threads));
    let mut key_regs = Array::<Vector<F, N>>::new(kv_loads);
    let mut value_regs = Array::<Vector<F, N>>::new(kv_loads);
    fetch_rows::<F, N>(k, &mut key_regs, 0, seq_kv, kv_stride, head, tid, tile);
    if comptime!(!tile.shared_kv) {
        fetch_rows::<F, N>(v, &mut value_regs, 0, seq_kv, kv_stride, head, tid, tile);
    }
    let tiles = seq_kv.div_ceil(bc);
    for t in 0..tiles {
        let kv0 = t * bc;
        let next = kv0 + bc;
        // the previous tile's P·V is done with ks, vs and ps
        sync_cube();
        stash_rows::<F, N>(&key_regs, &mut ks, tid, tile);
        if comptime!(!tile.shared_kv) {
            stash_rows::<F, N>(&value_regs, &mut vs, tid, tile);
        }
        sync_cube();
        if comptime!(tile.shared_kv) {
            fetch_rows::<F, N>(v, &mut value_regs, kv0, seq_kv, kv_stride, head, tid, tile);
        } else if next < seq_kv {
            fetch_rows::<F, N>(k, &mut key_regs, next, seq_kv, kv_stride, head, tid, tile);
            fetch_rows::<F, N>(v, &mut value_regs, next, seq_kv, kv_stride, head, tid, tile);
        }

        // S = Q·Kᵀ for rows row0 + i, columns col_id + tc·j
        #[unroll]
        for x in 0..comptime!(mr * mc) {
            scores[x] = 0.0;
        }
        for d in 0..dv {
            let mut qv = Array::<Vector<f32, N>>::new(mr);
            #[unroll]
            for i in 0..mr {
                qv[i] = Vector::cast_from(qs[(row0 + i) * ld + d]);
            }
            #[unroll]
            for j in 0..mc {
                let kv = Vector::<f32, N>::cast_from(ks[(col_id + j * tc) * ld + d]);
                #[unroll]
                for i in 0..mr {
                    let mut s = scores[i * mc + j];
                    let qi = qv[i];
                    #[unroll]
                    for e in 0..width {
                        s = fma(qi.extract(e), kv.extract(e), s);
                    }
                    scores[i * mc + j] = s;
                }
            }
        }

        // online softmax; P goes to shared memory for the P·V product
        #[unroll]
        for i in 0..mr {
            let mut tile_max = f32::new(MASKED);
            #[unroll]
            for j in 0..mc {
                let mut s = scores[i * mc + j] * scale;
                if kv0 + col_id + j * tc >= seq_kv {
                    s = MASKED;
                }
                scores[i * mc + j] = s;
                tile_max = max(tile_max, s);
            }
            #[unroll]
            for b in 0..shuffles {
                tile_max = max(tile_max, plane_shuffle_xor(tile_max, comptime!(1u32 << b)));
            }
            let new_max = max(row_max[i], tile_max);
            let alpha = (row_max[i] - new_max).exp();
            row_max[i] = new_max;
            let mut sum = 0.0;
            #[unroll]
            for j in 0..mc {
                let p = (scores[i * mc + j] - new_max).exp();
                sum += p;
                ps[(col_id + j * tc) * ldp + row0 + i] = F::cast_from(p);
            }
            row_sum[i] = row_sum[i] * alpha + sum;
            let alpha_v = Vector::<f32, N>::new(alpha);
            #[unroll]
            for j in 0..ov {
                acc[i * ov + j] *= alpha_v;
            }
        }
        if comptime!(tile.shared_kv) {
            // every key read is done: the values take the buffer
            sync_cube();
            stash_rows::<F, N>(&value_regs, &mut ks, tid, tile);
            if next < seq_kv {
                fetch_rows::<F, N>(k, &mut key_regs, next, seq_kv, kv_stride, head, tid, tile);
            }
        }
        sync_cube();

        // O += P·V
        if comptime!(tile.pv_chunk == 0) {
            for c in 0..bc {
                let mut pv = Array::<f32>::new(mr);
                #[unroll]
                for i in 0..mr {
                    pv[i] = f32::cast_from(ps[c * ldp + row0 + i]);
                }
                #[unroll]
                for j in 0..ov {
                    let vv = if comptime!(tile.shared_kv) {
                        Vector::<f32, N>::cast_from(ks[c * ld + col_id + j * tc])
                    } else {
                        Vector::<f32, N>::cast_from(vs[c * ld + col_id + j * tc])
                    };
                    #[unroll]
                    for i in 0..mr {
                        acc[i * ov + j] = fma(Vector::new(pv[i]), vv, acc[i * ov + j]);
                    }
                }
            }
        } else {
            // storage-precision partial sums over pv_chunk keys (packed f16 FMAs), flushed to f32
            let chunk = comptime!(tile.pv_chunk);
            #[unroll]
            for blk in 0..comptime!(bc / tile.pv_chunk) {
                let mut part = Array::<Vector<F, N>>::new(comptime!(mr * ov));
                #[unroll]
                for x in 0..comptime!(mr * ov) {
                    part[x] = Vector::new(F::new(0.0));
                }
                #[unroll]
                for cc in 0..chunk {
                    let c = blk * chunk + cc;
                    let mut pv = Array::<F>::new(mr);
                    #[unroll]
                    for i in 0..mr {
                        pv[i] = ps[c * ldp + row0 + i];
                    }
                    #[unroll]
                    for j in 0..ov {
                        let vv = if comptime!(tile.shared_kv) {
                            ks[c * ld + col_id + j * tc]
                        } else {
                            vs[c * ld + col_id + j * tc]
                        };
                        #[unroll]
                        for i in 0..mr {
                            part[i * ov + j] = fma(Vector::new(pv[i]), vv, part[i * ov + j]);
                        }
                    }
                }
                #[unroll]
                for x in 0..comptime!(mr * ov) {
                    acc[x] += Vector::<f32, N>::cast_from(part[x]);
                }
            }
        }
    }

    #[unroll]
    for i in 0..mr {
        let mut sum = row_sum[i];
        #[unroll]
        for b in 0..shuffles {
            sum += plane_shuffle_xor(sum, comptime!(1u32 << b));
        }
        let row = q0 + row0 + i;
        if row < seq_q {
            let inv = Vector::<f32, N>::new(1.0 / sum);
            #[unroll]
            for j in 0..ov {
                out[row * out_stride + head + col_id + j * tc] = Vector::cast_from(acc[i * ov + j] * inv);
            }
        }
    }
}

/// Load this thread's share of `BC` key or value rows starting at `row0` (zero past `rows`).
#[cube]
fn fetch_rows<F: Float, N: Size>(
    src: &[Vector<F, N>],
    regs: &mut Array<Vector<F, N>>,
    row0: usize,
    rows: usize,
    stride: usize,
    head: usize,
    tid: usize,
    #[comptime] tile: FlashTile,
) {
    let width = N::value();
    let threads = comptime!(tile.row_threads * tile.col_threads);
    let dv = comptime!(tile.head_dim / width);
    let total = comptime!(tile.col_threads * tile.cols * tile.head_dim / width);
    // branch-free: masked lanes read element 0 and keep zero
    #[unroll]
    for it in 0..comptime!(total.div_ceil(threads)) {
        let idx = tid + it * threads;
        let row = idx / dv;
        let ok = (comptime!(total % threads == 0) || idx < total) && row0 + row < rows;
        let value = src[select(ok, (row0 + row) * stride + head + idx % dv, 0)];
        regs[it] = select(ok, value, Vector::new(F::new(0.0)));
    }
}

/// Store [`fetch_rows`] registers into a padded shared tile.
#[cube]
fn stash_rows<F: Float, N: Size>(
    regs: &Array<Vector<F, N>>,
    dst: &mut Shared<[Vector<F, N>]>,
    tid: usize,
    #[comptime] tile: FlashTile,
) {
    let width = N::value();
    let threads = comptime!(tile.row_threads * tile.col_threads);
    let dv = comptime!(tile.head_dim / width);
    let ld = comptime!(tile.head_dim / width + tile.pad);
    let total = comptime!(tile.col_threads * tile.cols * tile.head_dim / width);
    #[unroll]
    for it in 0..comptime!(total.div_ceil(threads)) {
        let idx = tid + it * threads;
        if idx < total {
            dst[(idx / dv) * ld + idx % dv] = regs[it];
        }
    }
}

/// The default tile for a storage dtype.
pub fn default_tile(dtype: DType) -> FlashTile {
    if dtype == DType::F16 { TILE_F16 } else { TILE_F32 }
}

/// `softmax(q·kᵀ · scale)·v` per head over row-major `[tokens, heads * head_dim]` tensors,
/// with an explicit tile. Queries, keys and values share the head dimension.
pub fn flash_attention_with(
    q: CubeTensor,
    k: CubeTensor,
    v: CubeTensor,
    heads: usize,
    scale: f32,
    tile: FlashTile,
) -> CubeTensor {
    let (q, k, v) = (into_contiguous(q), into_contiguous(k), into_contiguous(v));
    let (seq_q, width) = dims(&q);
    let (seq_kv, kv_width) = dims(&k);
    let vector = tile.vector;
    assert_eq!(dims(&v), (seq_kv, kv_width), "keys and values must match");
    assert_eq!(width, kv_width, "queries and keys must share heads * head_dim");
    assert_eq!(
        width,
        heads * tile.head_dim,
        "tile head_dim does not match the tensors"
    );
    assert!(
        tile.head_dim.is_multiple_of(vector * tile.col_threads),
        "head_dim must split over col threads"
    );
    assert!(
        tile.col_threads.is_power_of_two() && tile.col_threads <= 32,
        "col threads must fit a plane"
    );
    assert!(
        tile.pv_chunk == 0 || tile.key_rows().is_multiple_of(tile.pv_chunk),
        "pv_chunk must divide the key rows"
    );
    assert!(q.dtype == k.dtype && k.dtype == v.dtype, "q, k, v dtypes differ");

    let out = empty_device_contiguous_dtype(
        q.client.clone(),
        q.device.clone(),
        q.meta.shape().clone(),
        q.dtype,
    );
    let client = q.client.clone();
    let cube_dim = CubeDim::new_1d(tile.threads() as u32);
    let strides = (
        row_stride(&q) / vector,
        row_stride(&k) / vector,
        row_stride(&out) / vector,
    );
    assert_eq!(
        row_stride(&k),
        row_stride(&v),
        "keys and values must share a layout"
    );
    // Query blocks per dispatch, so no single dispatch nears the amdgpu ring timeout
    let block_flops = 4 * tile.query_rows() * seq_kv.max(1) * width;
    let blocks_per_dispatch = (DISPATCH_FLOPS / block_flops).max(1);
    let blocks = seq_q.div_ceil(tile.query_rows());
    for first in (0..blocks).step_by(blocks_per_dispatch) {
        let count = CubeCount::Static((blocks - first).min(blocks_per_dispatch) as u32, heads as u32, 1);
        let q_begin = first * tile.query_rows();
        macro_rules! launch {
            ($elem:ty) => {
                flash_attention_kernel::launch::<$elem>(
                    &client,
                    count,
                    cube_dim,
                    vector,
                    q.clone().into_buffer_arg(),
                    k.clone().into_buffer_arg(),
                    v.clone().into_buffer_arg(),
                    out.clone().into_buffer_arg(),
                    q_begin,
                    seq_q,
                    seq_kv,
                    strides.0,
                    strides.1,
                    strides.2,
                    scale,
                    tile,
                )
            };
        }
        match q.dtype {
            DType::F32 => launch!(f32),
            DType::F16 => launch!(f16),
            other => panic!("flash attention supports f32 and f16, got {other:?}"),
        }
    }
    out
}

/// Work per attention dispatch: about 0.1 s at the RX 5500 XT's f32 rate.
const DISPATCH_FLOPS: usize = 400_000_000_000;

fn dims(t: &CubeTensor) -> (usize, usize) {
    let shape = t.meta.shape();
    assert_eq!(
        shape.num_dims(),
        2,
        "flash attention takes [tokens, heads * head_dim]"
    );
    (shape[0], shape[1])
}

fn row_stride(t: &CubeTensor) -> usize {
    let strides = t.meta.strides();
    assert_eq!(strides[1], 1, "rows must be contiguous");
    strides[0]
}

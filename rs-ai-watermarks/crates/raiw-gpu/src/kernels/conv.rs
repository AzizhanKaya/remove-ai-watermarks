//! Causal 3-D convolution as an implicit GEMM: the im2col matrix never reaches memory.
//!
//! Per output frame, `out[co, p] = bias[co] + Σ_k w[k, co] · col[k, p]` with
//! `k = ((dt · ks + dy) · ks + dx) · C_in + ci` and `col[k, p]` read straight from the input
//! frames (zero outside the frame, and for the virtual zero frames a causal convolution pads
//! in front). One cube owns 128 pixels x `8 · TC` output channels and walks `k` in 16-deep
//! slabs through shared memory, prefetching the next slab into registers.
//!
//! Each thread loads one fixed quad of pixels, so their coordinates are decoded once; with
//! `C_in % 16 == 0` a whole slab shares one tap, so a slab costs one tap decode and one bounds
//! test per pixel. Thread `(r, c)` multiplies the pixel quads `r` and `r + 16` by the channel
//! quads `c` and `c + TC` (an outer product per `k`: four 16-byte reads, adjacent across lanes,
//! feed 64 FMAs). The epilogue adds the bias and, optionally, a residual (a resnet shortcut);
//! the input can be read through 2x nearest upsampling (Wan's `Resample` up path), so upsampled
//! frames are never materialized either.
//!
//! Layouts: input `[T_in, C_in, H, W]`, weight `[K, C_out⁺]` with `C_out⁺` rounded up to a
//! multiple of 4 ([`conv_weight`]), output `[T_out, C_out, H_out, W_out]`, all float32 (the VAE
//! runs in float32, as in Python).

use burn::tensor::DType;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;

/// Static geometry of one convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvShape {
    pub in_channels: usize,
    pub out_channels: usize,
    /// Time taps (`k_t`); the input holds every frame they read, after `zeros` virtual ones.
    pub time_kernel: usize,
    /// Square spatial kernel side (`k_s`), stride 1.
    pub kernel: usize,
    /// Symmetric spatial zero padding (in output coordinates).
    pub padding: usize,
    /// Read the input through 2x nearest upsampling.
    pub upsample: bool,
}

impl ConvShape {
    pub const fn depth(&self) -> usize {
        self.time_kernel * self.kernel * self.kernel * self.in_channels
    }

    /// Weight row length: output channels rounded up to whole vectors.
    pub const fn weight_stride(&self) -> usize {
        self.out_channels.div_ceil(VECTOR) * VECTOR
    }
}

const VECTOR: usize = 4;
/// Pixel quads per thread row group and pixels per cube.
const ROW_THREADS: usize = 16;
const PIXELS: usize = 2 * VECTOR * ROW_THREADS;
const QUADS: usize = PIXELS / VECTOR;
const SLAB: usize = 16;

/// Column threads (a power of two): the largest whose `8 · TC` channels tile `out_channels`
/// exactly, else 16 with masked channels. On the RX 5500 XT a masked 16 beats an exact 12
/// (96 channels: 2.3 vs 1.8 TFLOP/s) and an exact 8 beats a masked 16 (192: 2.5 vs 2.3).
fn col_threads(out_channels: usize) -> usize {
    [16, 8, 4, 2]
        .into_iter()
        .find(|tc| out_channels.is_multiple_of(2 * VECTOR * tc))
        .unwrap_or(if out_channels > 64 { 16 } else { 2 })
}

#[cube(launch)]
fn conv_kernel<F: Float, N: Size>(
    x: &[F],
    w: &[Vector<F, N>],
    bias: &[F],
    residual: ComptimeOption<&[F]>,
    out: &mut [F],
    t_begin: usize,
    zeros: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    #[comptime] shape: ConvShape,
    #[comptime] tc: usize,
) {
    let threads = comptime!(ROW_THREADS * tc);
    let bc = comptime!(2 * VECTOR * tc);
    let lda = comptime!(QUADS + 1);
    let ldb = comptime!(2 * tc + 1);
    let depth = comptime!(shape.depth());
    let cout = comptime!(shape.out_channels);
    let w_stride = comptime!(shape.weight_stride() / VECTOR);

    let mut a_tile = Shared::<[Vector<F, N>]>::new_aligned_slice(comptime!(SLAB * (QUADS + 1)), 16usize);
    let mut b_tile = Shared::<[Vector<F, N>]>::new_aligned_slice(comptime!(SLAB * (2 * tc + 1)), 16usize);

    let tid = UNIT_POS as usize;
    let pixels = out_h * out_w;
    let m0 = CUBE_POS_X as usize * PIXELS;
    let n0 = CUBE_POS_Y as usize * bc;
    let t = t_begin + CUBE_POS_Z as usize;
    let plane = in_h * in_w;

    // A loads: a fixed pixel quad, k rows k_first + it * (threads / QUADS)
    let quad = tid % QUADS;
    let k_first = tid / QUADS;
    let mut py = Array::<usize>::new(VECTOR);
    let mut px = Array::<usize>::new(VECTOR);
    let mut p_valid = Array::<bool>::new(VECTOR);
    #[unroll]
    for l in 0..VECTOR {
        let p = m0 + quad * VECTOR + l;
        py[l] = p / out_w;
        px[l] = p % out_w;
        p_valid[l] = p < pixels;
    }
    let a_loads = comptime!((SLAB * QUADS).div_ceil(ROW_THREADS * tc));
    let mut a_regs = Array::<Vector<F, N>>::new(a_loads);
    // B loads: two weight vectors at fixed (row, column) offsets
    let mut b_row = Array::<usize>::new(2usize);
    let mut b_col = Array::<usize>::new(2usize);
    #[unroll]
    for it in 0..2 {
        let e = tid + it * threads;
        b_row[it] = e / comptime!(2 * tc);
        b_col[it] = e % comptime!(2 * tc);
    }
    let mut b_regs = Array::<Vector<F, N>>::new(2usize);

    let r = tid / tc;
    let c = tid % tc;
    let mut acc = Array::<f32>::new(64usize);
    #[unroll]
    for i in 0..64 {
        acc[i] = 0.0;
    }

    fetch_col::<F, N>(
        x,
        &mut a_regs,
        0,
        t,
        zeros,
        &py,
        &px,
        &p_valid,
        in_w,
        out_h,
        out_w,
        plane,
        k_first,
        shape,
        tc,
    );
    fetch_weight::<F, N>(w, &mut b_regs, 0, n0 / VECTOR, &b_row, &b_col, w_stride, depth);
    let slabs = comptime!(shape.depth().div_ceil(SLAB));
    for s in 0..slabs {
        sync_cube();
        #[unroll]
        for it in 0..a_loads {
            let k = k_first + it * comptime!(ROW_THREADS * tc / QUADS);
            if comptime!((SLAB * QUADS).is_multiple_of(ROW_THREADS * tc)) || k < SLAB {
                a_tile[k * lda + quad] = a_regs[it];
            }
        }
        #[unroll]
        for it in 0..2 {
            b_tile[b_row[it] * ldb + b_col[it]] = b_regs[it];
        }
        sync_cube();
        let next = (s + 1) * SLAB;
        if next < depth {
            fetch_col::<F, N>(
                x,
                &mut a_regs,
                next,
                t,
                zeros,
                &py,
                &px,
                &p_valid,
                in_w,
                out_h,
                out_w,
                plane,
                k_first,
                shape,
                tc,
            );
            fetch_weight::<F, N>(w, &mut b_regs, next, n0 / VECTOR, &b_row, &b_col, w_stride, depth);
        }
        #[unroll]
        for kl in 0..SLAB {
            let a0 = Vector::<f32, N>::cast_from(a_tile[kl * lda + r]);
            let a1 = Vector::<f32, N>::cast_from(a_tile[kl * lda + ROW_THREADS + r]);
            let b0 = Vector::<f32, N>::cast_from(b_tile[kl * ldb + c]);
            let b1 = Vector::<f32, N>::cast_from(b_tile[kl * ldb + tc + c]);
            #[unroll]
            for i in 0..VECTOR {
                #[unroll]
                for j in 0..VECTOR {
                    acc[i * 8 + j] = fma(a0.extract(i), b0.extract(j), acc[i * 8 + j]);
                    acc[i * 8 + 4 + j] = fma(a0.extract(i), b1.extract(j), acc[i * 8 + 4 + j]);
                    acc[(4 + i) * 8 + j] = fma(a1.extract(i), b0.extract(j), acc[(4 + i) * 8 + j]);
                    acc[(4 + i) * 8 + 4 + j] = fma(a1.extract(i), b1.extract(j), acc[(4 + i) * 8 + 4 + j]);
                }
            }
        }
    }

    // accumulator (i, j): pixel quad (i / 4 ? r + 16 : r), channel quad (j / 4 ? c + tc : c)
    #[unroll]
    for j in 0..8 {
        let co = n0 + (c + (j / 4) * tc) * VECTOR + j % 4;
        if co < cout {
            let shift = f32::cast_from(bias[co]);
            let base = (t * cout + co) * pixels;
            #[unroll]
            for i in 0..8 {
                let q = m0 + (r + (i / 4) * ROW_THREADS) * VECTOR + i % 4;
                if q < pixels {
                    let mut y = acc[i * 8 + j] + shift;
                    #[comptime]
                    match residual {
                        ComptimeOption::Some(residual) => y += f32::cast_from(residual[base + q]),
                        ComptimeOption::None => {}
                    }
                    out[base + q] = F::cast_from(y);
                }
            }
        }
    }
}

/// This thread's pixel quad of `col[k, ·]` for its rows of the slab starting at `k0`.
#[cube]
#[allow(clippy::too_many_arguments, clippy::trivially_copy_pass_by_ref)]
fn fetch_col<F: Float, N: Size>(
    x: &[F],
    regs: &mut Array<Vector<F, N>>,
    k0: usize,
    t: usize,
    zeros: usize,
    py: &Array<usize>,
    px: &Array<usize>,
    p_valid: &Array<bool>,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    plane: usize,
    k_first: usize,
    #[comptime] shape: ConvShape,
    #[comptime] tc: usize,
) {
    let ks = comptime!(shape.kernel);
    let cin = comptime!(shape.in_channels);
    let pad = comptime!(shape.padding);
    let depth = comptime!(shape.depth());
    let step = comptime!(ROW_THREADS * tc / QUADS);
    let loads = comptime!((SLAB * QUADS).div_ceil(ROW_THREADS * tc));
    if comptime!(shape.in_channels.is_multiple_of(SLAB)) {
        // one tap for the whole slab: decode it once, test each pixel once
        let tap = k0 / cin;
        let ci0 = k0 % cin;
        let dt = tap / comptime!(ks * ks);
        let dy = (tap / ks) % ks;
        let dx = tap % ks;
        let frame_ok = t + dt >= zeros;
        let frame = select(frame_ok, t + dt - zeros, 0);
        let mut offset = Array::<usize>::new(VECTOR);
        let mut inside = Array::<bool>::new(VECTOR);
        #[unroll]
        for l in 0..VECTOR {
            let yy = py[l] + dy;
            let xx = px[l] + dx;
            let ok = frame_ok && p_valid[l] && yy >= pad && yy < out_h + pad && xx >= pad && xx < out_w + pad;
            let mut sy = select(ok, yy - pad, 0);
            let mut sx = select(ok, xx - pad, 0);
            if comptime!(shape.upsample) {
                sy /= 2;
                sx /= 2;
            }
            inside[l] = ok;
            offset[l] = sy * in_w + sx;
        }
        // branch-free: masked lanes read element 0 and keep zero
        #[unroll]
        for it in 0..loads {
            let k = k_first + it * step;
            let channel = (frame * cin + ci0 + k) * plane;
            let mut v = Vector::<F, N>::empty();
            #[unroll]
            for l in 0..VECTOR {
                let ok =
                    inside[l] && (comptime!((SLAB * QUADS).is_multiple_of(ROW_THREADS * tc)) || k < SLAB);
                let value = x[select(ok, channel + offset[l], 0)];
                v.insert(l, select(ok, value, F::new(0.0)));
            }
            regs[it] = v;
        }
    } else {
        #[unroll]
        for it in 0..loads {
            let k = k0 + k_first + it * step;
            let ci = k % cin;
            let tap = k / cin;
            let dt = tap / comptime!(ks * ks);
            let dy = (tap / ks) % ks;
            let dx = tap % ks;
            let mut v = Vector::<F, N>::empty();
            #[unroll]
            for l in 0..VECTOR {
                let yy = py[l] + dy;
                let xx = px[l] + dx;
                let ok = p_valid[l]
                    && k < depth
                    && k_first + it * step < SLAB
                    && t + dt >= zeros
                    && yy >= pad
                    && yy < out_h + pad
                    && xx >= pad
                    && xx < out_w + pad;
                let mut value = F::new(0.0);
                if ok {
                    let mut sy = yy - pad;
                    let mut sx = xx - pad;
                    if comptime!(shape.upsample) {
                        sy /= 2;
                        sx /= 2;
                    }
                    value = x[((t + dt - zeros) * cin + ci) * plane + sy * in_w + sx];
                }
                v.insert(l, value);
            }
            regs[it] = v;
        }
    }
}

/// This thread's two weight vectors of the slab starting at row `k0`, vector column `n0v`.
#[cube]
#[allow(clippy::too_many_arguments, clippy::trivially_copy_pass_by_ref)]
fn fetch_weight<F: Float, N: Size>(
    w: &[Vector<F, N>],
    regs: &mut Array<Vector<F, N>>,
    k0: usize,
    n0v: usize,
    rows: &Array<usize>,
    cols: &Array<usize>,
    w_stride: usize,
    depth: usize,
) {
    #[unroll]
    for it in 0..2 {
        let k = k0 + rows[it];
        let col = n0v + cols[it];
        let ok = k < depth && col < w_stride;
        let value = w[select(ok, k * w_stride + col, 0)];
        regs[it] = select(ok, value, Vector::new(F::new(0.0)));
    }
}

/// Work per convolution dispatch, inside the amdgpu ring timeout at any plausible speed.
const DISPATCH_FLOPS: usize = 250_000_000_000;

/// Rearrange a PyTorch `[C_out, C_in, k_t, k_s, k_s]` kernel into the `[K, C_out⁺]` matrix the
/// kernel reads: `K` ordered `(dt, dy, dx, ci)`, columns zero-padded to whole vectors.
pub fn conv_weight(weight: &[f32], shape: &ConvShape) -> Vec<f32> {
    let (cin, cout, kt, ks) = (
        shape.in_channels,
        shape.out_channels,
        shape.time_kernel,
        shape.kernel,
    );
    let stride = shape.weight_stride();
    let mut out = vec![0.0; shape.depth() * stride];
    for co in 0..cout {
        for ci in 0..cin {
            for dt in 0..kt {
                for dy in 0..ks {
                    for dx in 0..ks {
                        let src = (((co * cin + ci) * kt + dt) * ks + dy) * ks + dx;
                        let k = ((dt * ks + dy) * ks + dx) * cin + ci;
                        out[k * stride + co] = weight[src];
                    }
                }
            }
        }
    }
    out
}

/// Causal convolution of `x` `[T_in, C_in, H, W]` after `zeros` virtual zero frames:
/// `T_out = zeros + T_in - k_t + 1` output frames of `[C_out, H_out, W_out]`, where
/// `H_out = H` (or `2H` when upsampling) with `padding = (k_s - 1) / 2`.
pub fn causal_conv3d(
    x: CubeTensor,
    weight: CubeTensor,
    bias: CubeTensor,
    residual: Option<CubeTensor>,
    zeros: usize,
    shape: ConvShape,
) -> CubeTensor {
    let (x, weight, bias) = (into_contiguous(x), into_contiguous(weight), into_contiguous(bias));
    let dims = x.meta.shape();
    assert_eq!(dims.num_dims(), 4, "conv input is [T, C, H, W]");
    let (frames, channels, in_h, in_w) = (dims[0], dims[1], dims[2], dims[3]);
    assert_eq!(channels, shape.in_channels, "conv input channels");
    assert_eq!(2 * shape.padding + 1, shape.kernel, "conv keeps the spatial size");
    assert!(
        zeros + frames >= shape.time_kernel,
        "causal conv needs a full kernel span"
    );
    assert_eq!(
        weight.meta.num_elements(),
        shape.depth() * shape.weight_stride(),
        "conv weight is [K, C_out rounded up to 4] (conv_weight)"
    );
    assert_eq!(bias.meta.num_elements(), shape.out_channels, "conv bias");
    assert!(x.dtype == DType::F32 && weight.dtype == DType::F32 && bias.dtype == DType::F32);
    let (out_h, out_w) = if shape.upsample {
        (2 * in_h, 2 * in_w)
    } else {
        (in_h, in_w)
    };
    let out_frames = zeros + frames + 1 - shape.time_kernel;
    let out_shape = [out_frames, shape.out_channels, out_h, out_w];
    let residual = residual.map(|r| {
        let r = into_contiguous(r);
        assert_eq!(
            r.meta.shape().clone(),
            out_shape.into(),
            "residual must match the output"
        );
        r
    });
    let out = empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), out_shape.into(), DType::F32);
    let tc = col_threads(shape.out_channels);
    let pixels = out_h * out_w;
    let frame_flops = 2 * pixels * shape.out_channels * shape.depth();
    let per_dispatch = (DISPATCH_FLOPS / frame_flops.max(1)).max(1);
    let blocks = (
        pixels.div_ceil(PIXELS) as u32,
        shape.out_channels.div_ceil(2 * VECTOR * tc) as u32,
    );
    for first in (0..out_frames).step_by(per_dispatch) {
        let frames = (out_frames - first).min(per_dispatch);
        let count = CubeCount::Static(blocks.0, blocks.1, frames as u32);
        conv_kernel::launch::<f32>(
            &x.client,
            count,
            CubeDim::new_1d((ROW_THREADS * tc) as u32),
            VECTOR,
            x.clone().into_buffer_arg(),
            weight.clone().into_buffer_arg(),
            bias.clone().into_buffer_arg(),
            residual.clone().map(CubeTensor::into_buffer_arg).into(),
            out.clone().into_buffer_arg(),
            first,
            zeros,
            in_h,
            in_w,
            out_h,
            out_w,
            shape,
            tc,
        );
        super::submit_heavy(&x.client, frames * frame_flops);
    }
    out
}

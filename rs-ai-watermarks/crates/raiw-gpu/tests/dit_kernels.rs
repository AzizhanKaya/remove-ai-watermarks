//! The DiT CubeCL kernels (flash attention, GEMM epilogues, fused norms) against float64 host
//! references, plus GPU timings.
//!
//! Timings: `cargo test --release -p raiw-gpu --test dit_kernels -- --ignored --nocapture --test-threads=1`.

#![cfg(feature = "gpu-runtime")]

use std::time::Instant;

use raiw_gpu::burn::tensor::{DType, Tensor, TensorData, s};
use raiw_gpu::io::read_f32;
use raiw_gpu::kernels::attention::{FlashTile, TILE_F16, TILE_F32};
use raiw_gpu::kernels::{
    Activation, ConvShape, DenseWeight, causal_conv3d, conv_weight, flash_attention, flash_attention_tuned,
    linear, linear_gated, modulated_layer_norm, rms_norm_rope,
};
use raiw_gpu::models::video::wan::attention::scaled_dot_product;
use raiw_gpu::models::video::wan::rope;
use raiw_gpu::{ComputeDevice, DeviceRequest};

const HEAD_DIM: usize = 128;

fn seeded(len: usize, seed: u32, amplitude: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            amplitude * (f64::from(state) / 4_294_967_296.0 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn gpu() -> Option<ComputeDevice> {
    ComputeDevice::select(DeviceRequest::Auto)
        .ok()
        .filter(ComputeDevice::is_gpu)
}

/// Float64 `softmax(q·kᵀ/sqrt(d))·v` over `[tokens, heads * d]` rows.
fn reference(q: &[f32], k: &[f32], v: &[f32], seq_q: usize, seq_kv: usize, heads: usize) -> Vec<f64> {
    let width = heads * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let mut out = vec![0.0; seq_q * width];
    let mut scores = vec![0.0; seq_kv];
    for h in 0..heads {
        for i in 0..seq_q {
            let qi = &q[i * width + h * HEAD_DIM..][..HEAD_DIM];
            for (j, s) in scores.iter_mut().enumerate() {
                let kj = &k[j * width + h * HEAD_DIM..][..HEAD_DIM];
                *s = qi
                    .iter()
                    .zip(kj)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>()
                    * scale;
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = scores.iter().map(|s| (s - max).exp()).sum();
            let row = &mut out[i * width + h * HEAD_DIM..][..HEAD_DIM];
            for (j, s) in scores.iter().enumerate() {
                let p = (s - max).exp() / total;
                let vj = &v[j * width + h * HEAD_DIM..][..HEAD_DIM];
                for (o, x) in row.iter_mut().zip(vj) {
                    *o += p * f64::from(*x);
                }
            }
        }
    }
    out
}

fn upload(values: &[f32], rows: usize, width: usize, device: &ComputeDevice, dtype: DType) -> Tensor<2> {
    let t: Tensor<2> = Tensor::from_data(TensorData::new(values.to_vec(), [rows, width]), device.burn());
    if dtype == DType::F32 { t } else { t.cast(dtype) }
}

fn upload1(values: &[f32], device: &ComputeDevice) -> Tensor<1> {
    Tensor::from_data(TensorData::new(values.to_vec(), [values.len()]), device.burn())
}

fn read(t: Tensor<2>) -> Vec<f32> {
    read_f32(t.cast(DType::F32)).unwrap()
}

/// Inputs as the kernel sees them after storage rounding.
fn stored(values: Vec<f32>, dtype: DType) -> Vec<f32> {
    if dtype == DType::F16 {
        round_f16(&values)
    } else {
        values
    }
}

/// Max error relative to the reference's largest magnitude.
fn rel_err(actual: &[f32], expected: &[f64]) -> f64 {
    let scale = expected.iter().fold(1e-12, |m, x| f64::max(m, x.abs()));
    max_err(actual, expected) / scale
}

fn round_f16(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
}

fn max_err(actual: &[f32], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0, f64::max)
}

struct Case {
    seq_q: usize,
    seq_kv: usize,
    heads: usize,
    /// q and k amplitude: 3 gives peaky rows (online rescale), 0.3 near-uniform rows (long sums)
    amplitude: f32,
}

const CASES: [Case; 6] = [
    Case {
        seq_q: 1,
        seq_kv: 1,
        heads: 1,
        amplitude: 3.0,
    },
    Case {
        seq_q: 37,
        seq_kv: 53,
        heads: 2,
        amplitude: 3.0,
    },
    Case {
        seq_q: 64,
        seq_kv: 64,
        heads: 1,
        amplitude: 3.0,
    },
    Case {
        seq_q: 200,
        seq_kv: 512,
        heads: 3,
        amplitude: 3.0,
    },
    Case {
        seq_q: 130,
        seq_kv: 1000,
        heads: 2,
        amplitude: 3.0,
    },
    Case {
        seq_q: 70,
        seq_kv: 4100,
        heads: 1,
        amplitude: 0.3,
    },
];

fn check(device: &ComputeDevice, dtype: DType, tile: FlashTile, limit: f64) {
    for (n, case) in CASES.iter().enumerate() {
        let width = case.heads * HEAD_DIM;
        let mut q = seeded(case.seq_q * width, 3 * n as u32 + 1, case.amplitude);
        let mut k = seeded(case.seq_kv * width, 3 * n as u32 + 2, case.amplitude);
        let mut v = seeded(case.seq_kv * width, 3 * n as u32 + 3, 1.0);
        if dtype == DType::F16 {
            (q, k, v) = (round_f16(&q), round_f16(&k), round_f16(&v));
        }
        let expected = reference(&q, &k, &v, case.seq_q, case.seq_kv, case.heads);
        let out = flash_attention_tuned(
            upload(&q, case.seq_q, width, device, dtype),
            upload(&k, case.seq_kv, width, device, dtype),
            upload(&v, case.seq_kv, width, device, dtype),
            case.heads,
            1.0 / (HEAD_DIM as f32).sqrt(),
            tile,
        );
        let actual = read_f32(out.cast(DType::F32)).unwrap();
        let err = max_err(&actual, &expected);
        eprintln!(
            "{dtype:?} pv{} {}x{}x{}: max err {err:.2e}",
            tile.pv_chunk, case.seq_q, case.seq_kv, case.heads
        );
        assert!(err <= limit, "{dtype:?} case {n}: max err {err}");
    }
}

#[test]
fn flash_attention_f32_matches_reference() {
    let Some(device) = gpu() else { return };
    check(&device, DType::F32, TILE_F32, 1e-5);
}

#[test]
fn flash_attention_f16_matches_reference() {
    let Some(device) = gpu() else { return };
    // f16 output rounding of values in [-1, 1] is 2^-11; the packed P·V partial sums add as much
    for pv_chunk in [0, 8, 16, TILE_F16.pv_chunk] {
        check(&device, DType::F16, FlashTile { pv_chunk, ..TILE_F16 }, 1.5e-3);
    }
}

fn gelu_tanh(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh())
}

/// Float64 `x·wᵀ + b` with optional GELU and gated residual.
fn linear_reference(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    (m, k, n): (usize, usize, usize),
    gelu: bool,
    residual: Option<(&[f32], &[f32])>,
) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for r in 0..m {
        for c in 0..n {
            let dot: f64 = (0..k)
                .map(|i| f64::from(x[r * k + i]) * f64::from(w[c * k + i]))
                .sum();
            let mut y = dot + f64::from(b[c]);
            if gelu {
                y = gelu_tanh(y);
            }
            if let Some((res, gate)) = residual {
                y = f64::from(res[r * n + c]) + y * f64::from(gate[c]);
            }
            out[r * n + c] = y;
        }
    }
    out
}

#[test]
fn linear_matches_reference() {
    let Some(device) = gpu() else { return };
    let shapes = [(5, 64, 7), (130, 256, 200), (257, 1536, 136)];
    for dtype in [DType::F32, DType::F16] {
        for (n_case, &(m, k, n)) in shapes.iter().enumerate() {
            let seed = 10 * n_case as u32;
            let x = stored(seeded(m * k, seed + 1, 1.0), dtype);
            let w = stored(seeded(n * k, seed + 2, 1.0 / (k as f32).sqrt()), dtype);
            let b = seeded(n, seed + 3, 0.5);
            let res = seeded(m * n, seed + 4, 2.0);
            let gate = seeded(n, seed + 5, 1.5);
            let (xt, wt) = (upload(&x, m, k, &device, dtype), upload(&w, n, k, &device, dtype));
            let bt = upload1(&b, &device);
            // int8 is checked against its own dequantized weights: the kernel, not the rounding
            let dequantized = dequantize_int8(&w, (n, k));
            let mut weights = vec![("torch", DenseWeight::torch(wt.clone()), &w)];
            if dtype == DType::F16 && n % 8 == 0 {
                weights.push(("packed", DenseWeight::packed(wt), &w));
                weights.push(("int8", DenseWeight::int8(&w, (n, k), device.burn()), &dequantized));
            }
            for (kind, weight, w) in weights {
                let plain = read(linear(
                    xt.clone(),
                    &weight,
                    bt.clone(),
                    Activation::Identity,
                    DType::F32,
                ));
                let gelu = read(linear(
                    xt.clone(),
                    &weight,
                    bt.clone(),
                    Activation::GeluTanh,
                    dtype,
                ));
                let gated = read(linear_gated(
                    xt.clone(),
                    &weight,
                    bt.clone(),
                    upload(&res, m, n, &device, DType::F32),
                    upload1(&gate, &device),
                ));
                let errs = [
                    rel_err(&plain, &linear_reference(&x, w, &b, (m, k, n), false, None)),
                    rel_err(&gelu, &linear_reference(&x, w, &b, (m, k, n), true, None)),
                    rel_err(
                        &gated,
                        &linear_reference(&x, w, &b, (m, k, n), false, Some((&res, &gate))),
                    ),
                ];
                eprintln!(
                    "linear {kind} {dtype:?} {m}x{k}x{n}: plain {:.1e} gelu {:.1e} gated {:.1e}",
                    errs[0], errs[1], errs[2]
                );
                // f32 accumulation: f16 adds only the gelu output rounding; the packed kernel's
                // 16-term f16 partial sums stay under bfloat16 input rounding (~4e-3)
                let limit = match (dtype, kind) {
                    (DType::F32, _) => [2e-6, 2e-6, 2e-6],
                    (_, "torch") => [2e-5, 1e-3, 2e-5],
                    _ => [2e-3, 2e-3, 2e-3],
                };
                for (err, limit) in errs.iter().zip(limit) {
                    assert!(*err <= limit, "{kind} {dtype:?} {m}x{k}x{n}: {errs:?}");
                }
            }
            if dtype == DType::F16 {
                let rounding = rel_err(&dequantized, &w.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
                eprintln!("int8 weight rounding {m}x{k}x{n}: {rounding:.1e} of max |w|");
            }
        }
    }
}

/// `DenseWeight::int8`'s rounding, back in float32 `[out, in]`.
fn dequantize_int8(w: &[f32], (output, input): (usize, usize)) -> Vec<f32> {
    w.chunks_exact(input)
        .flat_map(|row| {
            let scale = row.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            row.iter()
                .map(move |v| (v * inv).round().clamp(-127.0, 127.0) * scale)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .take(output * input)
        .collect()
}

#[test]
fn modulated_layer_norm_matches_reference() {
    let Some(device) = gpu() else { return };
    let (rows, width) = (37, 1536);
    // a large common offset checks the two-pass variance
    let x: Vec<f32> = seeded(rows * width, 1, 4.0).iter().map(|v| v + 50.0).collect();
    let mul = seeded(width, 2, 1.0);
    let add = seeded(width, 3, 1.0);
    let mut expected = vec![0.0; rows * width];
    for r in 0..rows {
        let row = &x[r * width..][..width];
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / width as f64;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / width as f64;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for c in 0..width {
            expected[r * width + c] =
                (f64::from(row[c]) - mean) * inv * f64::from(mul[c]) + f64::from(add[c]);
        }
    }
    for dtype in [DType::F32, DType::F16] {
        let out = read(modulated_layer_norm(
            upload(&x, rows, width, &device, DType::F32),
            upload1(&mul, &device),
            upload1(&add, &device),
            1e-6,
            dtype,
        ));
        let err = rel_err(&out, &expected);
        eprintln!("layer norm -> {dtype:?}: {err:.1e}");
        // f32: x itself carries 50 * 2^-24 of rounding against a unit-scale result
        assert!(
            err <= if dtype == DType::F16 { 1e-3 } else { 5e-6 },
            "{dtype:?}: {err}"
        );
    }
}

#[test]
fn rms_norm_rope_matches_reference() {
    let Some(device) = gpu() else { return };
    let grid = (2, 3, 5);
    let rows = grid.0 * grid.1 * grid.2;
    let (width, stride, offset) = (1536, 4608, 1536);
    let x = seeded(rows * stride, 4, 3.0);
    let weight = seeded(width, 5, 1.0);
    let (cos, sin) = rope::tables(HEAD_DIM, grid);
    let pairs = HEAD_DIM / 2;
    let mut expected = vec![0.0; rows * width];
    for r in 0..rows {
        let row = &x[r * stride + offset..][..width];
        let ms = row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / width as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        let y: Vec<f64> = row
            .iter()
            .zip(&weight)
            .map(|(&v, &w)| f64::from(v) * inv * f64::from(w))
            .collect();
        for c in (0..width).step_by(2) {
            let p = (c % HEAD_DIM) / 2;
            let (cs, sn) = (f64::from(cos[r * pairs + p]), f64::from(sin[r * pairs + p]));
            expected[r * width + c] = y[c] * cs - y[c + 1] * sn;
            expected[r * width + c + 1] = y[c] * sn + y[c + 1] * cs;
        }
    }
    let tables = (
        upload(&cos, rows, pairs, &device, DType::F32),
        upload(&sin, rows, pairs, &device, DType::F32),
    );
    for (input, output) in [(DType::F32, DType::F32), (DType::F16, DType::F16)] {
        let xs = stored(x.clone(), input);
        let out = read(rms_norm_rope(
            upload(&xs, rows, stride, &device, input),
            upload1(&weight, &device),
            tables.clone(),
            offset,
            width,
            HEAD_DIM,
            1e-6,
            output,
        ));
        let err = rel_err(&out, &expected);
        eprintln!("rms norm + rope {input:?} -> {output:?}: {err:.1e}");
        assert!(
            err <= if output == DType::F16 { 2e-3 } else { 1e-6 },
            "{input:?}: {err}"
        );
    }
}

/// Float64 causal convolution, PyTorch weight layout `[C_out, C_in, k_t, k_s, k_s]`.
fn conv_reference(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    frames: usize,
    (h, wd): (usize, usize),
    zeros: usize,
    s: &ConvShape,
) -> Vec<f64> {
    let (cin, cout, kt, ks, pad) = (s.in_channels, s.out_channels, s.time_kernel, s.kernel, s.padding);
    let (oh, ow) = if s.upsample { (2 * h, 2 * wd) } else { (h, wd) };
    let out_frames = zeros + frames + 1 - kt;
    let mut out = vec![0.0; out_frames * cout * oh * ow];
    for t in 0..out_frames {
        for co in 0..cout {
            for y in 0..oh {
                for xx in 0..ow {
                    let mut sum = f64::from(b[co]);
                    for dt in 0..kt {
                        let Some(src_t) = (t + dt).checked_sub(zeros) else {
                            continue;
                        };
                        for ci in 0..cin {
                            for dy in 0..ks {
                                for dx in 0..ks {
                                    let (Some(sy), Some(sx)) =
                                        ((y + dy).checked_sub(pad), (xx + dx).checked_sub(pad))
                                    else {
                                        continue;
                                    };
                                    if sy >= oh || sx >= ow {
                                        continue;
                                    }
                                    let (sy, sx) = if s.upsample { (sy / 2, sx / 2) } else { (sy, sx) };
                                    let xv = x[((src_t * cin + ci) * h + sy) * wd + sx];
                                    let wv = w[(((co * cin + ci) * kt + dt) * ks + dy) * ks + dx];
                                    sum += f64::from(xv) * f64::from(wv);
                                }
                            }
                        }
                    }
                    out[((t * cout + co) * oh + y) * ow + xx] = sum;
                }
            }
        }
    }
    out
}

#[test]
fn causal_conv3d_matches_reference() {
    let Some(device) = gpu() else { return };
    let shape = |cin, cout, kt, ks, upsample| ConvShape {
        in_channels: cin,
        out_channels: cout,
        time_kernel: kt,
        kernel: ks,
        padding: (ks - 1) / 2,
        upsample,
    };
    // (shape, frames, (h, w), zeros, residual)
    let cases = [
        (shape(16, 96, 3, 3, false), 2, (11, 13), 2, true),
        (shape(3, 96, 3, 3, false), 4, (9, 17), 0, false),
        (shape(32, 16, 1, 3, true), 2, (5, 7), 0, false),
        (shape(24, 48, 3, 1, false), 3, (6, 21), 1, true),
        (shape(40, 3, 3, 3, false), 3, (12, 10), 1, false),
    ];
    for (n, (s, frames, (h, w), zeros, with_residual)) in cases.into_iter().enumerate() {
        let seed = 40 + 5 * n as u32;
        let x = seeded(frames * s.in_channels * h * w, seed, 1.0);
        let fan_in = (s.time_kernel * s.kernel * s.kernel * s.in_channels) as f32;
        let weight = seeded(s.out_channels * s.depth(), seed + 1, 1.0 / fan_in.sqrt());
        let bias = seeded(s.out_channels, seed + 2, 0.5);
        let mut expected = conv_reference(&x, &weight, &bias, frames, (h, w), zeros, &s);
        let out_frames = zeros + frames + 1 - s.time_kernel;
        let (oh, ow) = if s.upsample { (2 * h, 2 * w) } else { (h, w) };
        let out_shape = [out_frames, s.out_channels, oh, ow];
        let residual = with_residual.then(|| seeded(expected.len(), seed + 3, 2.0));
        if let Some(res) = &residual {
            expected
                .iter_mut()
                .zip(res)
                .for_each(|(e, r)| *e += f64::from(*r));
        }
        let xt: Tensor<4> =
            Tensor::from_data(TensorData::new(x, [frames, s.in_channels, h, w]), device.burn());
        let wt = upload(
            &conv_weight(&weight, &s),
            s.depth(),
            s.weight_stride(),
            &device,
            DType::F32,
        );
        let rt = residual.map(|r| Tensor::<4>::from_data(TensorData::new(r, out_shape), device.burn()));
        let out = causal_conv3d(xt, wt, upload1(&bias, &device), rt, zeros, s);
        assert_eq!(out.dims(), out_shape, "case {n} shape");
        let err = rel_err(&read_f32(out).unwrap(), &expected);
        eprintln!("conv case {n} {s:?}: {err:.1e}");
        assert!(err <= 2e-6, "case {n}: {err}");
    }
}

fn bench(label: &str, device: &ComputeDevice, mut run: impl FnMut() -> Tensor<2>, flops: f64) -> f64 {
    let warm = run();
    device.sync().unwrap();
    drop(warm);
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let start = Instant::now();
        let out = run();
        device.sync().unwrap();
        best = best.min(start.elapsed().as_secs_f64());
        drop(out);
    }
    println!(
        "{label:<44} {:>9.2} ms  {:>6.2} TFLOP/s",
        best * 1e3,
        flops / best / 1e12
    );
    best
}

/// Wan-shaped problems scaled down to short dispatches; `RAIW_ATTN_TOKENS` overrides the length.
#[test]
#[ignore = "GPU timing run"]
fn flash_attention_timings() {
    let Some(device) = gpu() else { return };
    let tokens: usize = std::env::var("RAIW_ATTN_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let heads = 12;
    let width = heads * HEAD_DIM;
    let q = seeded(tokens * width, 1, 3.0);
    let k = seeded(tokens * width, 2, 3.0);
    let v = seeded(tokens * width, 3, 1.0);
    let flops = 4.0 * (tokens * tokens * width) as f64;
    for dtype in [DType::F32, DType::F16] {
        let (qt, kt, vt) = (
            upload(&q, tokens, width, &device, dtype),
            upload(&k, tokens, width, &device, dtype),
            upload(&v, tokens, width, &device, dtype),
        );
        bench(
            &format!("flash {dtype:?} self {tokens}"),
            &device,
            || flash_attention(qt.clone(), kt.clone(), vt.clone(), heads),
            flops,
        );
        let ctx = 512;
        let (kc, vc) = (
            upload(&k[..ctx * width], ctx, width, &device, dtype),
            upload(&v[..ctx * width], ctx, width, &device, dtype),
        );
        bench(
            &format!("flash {dtype:?} cross {tokens}x{ctx}"),
            &device,
            || flash_attention(qt.clone(), kc.clone(), vc.clone(), heads),
            4.0 * (tokens * ctx * width) as f64,
        );
    }
    // the materialized path Wan used before
    let heads4 = |t: Tensor<2>| {
        t.reshape([tokens, heads, HEAD_DIM])
            .swap_dims(0, 1)
            .unsqueeze_dim::<4>(0)
    };
    let (qt, kt, vt) = (
        upload(&q, tokens, width, &device, DType::F32),
        upload(&k, tokens, width, &device, DType::F32),
        upload(&v, tokens, width, &device, DType::F32),
    );
    bench(
        &format!("chunked burn f32 self {tokens}"),
        &device,
        || {
            let out = scaled_dot_product(heads4(qt.clone()), heads4(kt.clone()), heads4(vt.clone()));
            out.squeeze_dim::<3>(0).swap_dims(0, 1).reshape([tokens, width])
        },
        flops,
    );
}

/// Tile sweep on one self-attention shape.
#[test]
#[ignore = "GPU tuning run"]
fn flash_attention_tile_sweep() {
    let Some(device) = gpu() else { return };
    let tokens = env_usize("RAIW_ATTN_TOKENS", 4096);
    let heads = 12;
    let width = heads * HEAD_DIM;
    let q = seeded(tokens * width, 1, 3.0);
    let k = seeded(tokens * width, 2, 3.0);
    let v = seeded(tokens * width, 3, 1.0);
    let flops = 4.0 * (tokens * tokens * width) as f64;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let tile = |tr, tc, mr, mc, vector, shared_kv| FlashTile {
        row_threads: tr,
        col_threads: tc,
        rows: mr,
        cols: mc,
        vector,
        shared_kv,
        ..TILE_F16
    };
    let tiles = [
        TILE_F32,
        TILE_F16,
        tile(16, 16, 2, 2, 4, true),
        tile(16, 16, 4, 2, 4, true),
        tile(32, 8, 2, 2, 4, true),
        tile(16, 16, 2, 4, 4, true),
    ];
    // RAIW_ATTN_TILES="tr,tc,mr,mc,vector,pad,shared_kv,pv_chunk;..." and RAIW_ATTN_DTYPE=f16|f32
    let tiles: Vec<FlashTile> = match std::env::var("RAIW_ATTN_TILES") {
        Ok(spec) => spec
            .split(';')
            .map(|t| {
                let n: Vec<usize> = t.split(',').map(|x| x.trim().parse().unwrap()).collect();
                FlashTile {
                    pad: n[5],
                    pv_chunk: n.get(7).copied().unwrap_or(0),
                    ..tile(n[0], n[1], n[2], n[3], n[4], n.get(6) == Some(&1))
                }
            })
            .collect(),
        Err(_) => tiles.to_vec(),
    };
    let dtypes = match std::env::var("RAIW_ATTN_DTYPE").as_deref() {
        Ok("f16") => vec![DType::F16],
        Ok("f32") => vec![DType::F32],
        _ => vec![DType::F32, DType::F16],
    };
    for dtype in dtypes {
        let (qt, kt, vt) = (
            upload(&q, tokens, width, &device, dtype),
            upload(&k, tokens, width, &device, dtype),
            upload(&v, tokens, width, &device, dtype),
        );
        for &tile in &tiles {
            let elem = if dtype == DType::F32 { 4 } else { 2 };
            if tile.shared_bytes(elem) > 65_536 {
                continue;
            }
            bench(
                &format!(
                    "{dtype:?} {}x{} thr {}x{} per v{}{} pv{} {}KB",
                    tile.row_threads,
                    tile.col_threads,
                    tile.rows,
                    tile.cols,
                    tile.vector,
                    if tile.shared_kv { " kv1" } else { "" },
                    tile.pv_chunk,
                    tile.shared_bytes(elem) / 1024
                ),
                &device,
                || flash_attention_tuned(qt.clone(), kt.clone(), vt.clone(), heads, scale, tile),
                flops,
            );
        }
    }
}

/// The GEMM kernel on the Wan block's projection shapes (`RAIW_ATTN_TOKENS` rows).
///
/// burn's autotuned matmul runs only on the 1536x1536 shape: tuning it on the feed-forward
/// shapes ran a candidate past the amdgpu ring timeout and reset the GPU.
#[test]
#[ignore = "GPU timing run"]
fn wan_linear_throughput() {
    let Some(device) = gpu() else { return };
    let tokens = env_usize("RAIW_ATTN_TOKENS", 1024);
    for dtype in [DType::F32, DType::F16] {
        for (input, output, act) in [
            (1536, 4608, Activation::Identity),
            (1536, 1536, Activation::Identity),
            (1536, 8960, Activation::GeluTanh),
            (8960, 1536, Activation::Identity),
            (3072, 9216, Activation::Identity),
            (3072, 14336, Activation::GeluTanh),
            (14336, 3072, Activation::Identity),
        ] {
            let x = upload(&seeded(tokens * input, 5, 1.0), tokens, input, &device, dtype);
            let host = seeded(input * output, 6, 0.02);
            let w = upload(&host, output, input, &device, dtype);
            let b = upload1(&seeded(output, 7, 0.1), &device);
            let mut weights = vec![("torch", DenseWeight::torch(w.clone()))];
            if dtype == DType::F16 {
                weights.push(("packed", DenseWeight::packed(w)));
                weights.push(("int8", DenseWeight::int8(&host, (output, input), device.burn())));
            }
            for (kind, weight) in weights {
                bench(
                    &format!("linear {kind} {dtype:?} [{tokens}x{input}]->{output} {act:?}"),
                    &device,
                    || linear(x.clone(), &weight, b.clone(), act, dtype),
                    2.0 * (tokens * input * output) as f64,
                );
            }
        }
    }
    let x = upload(&seeded(tokens * 1536, 5, 1.0), tokens, 1536, &device, DType::F32);
    let w = upload(&seeded(1536 * 1536, 6, 0.02), 1536, 1536, &device, DType::F32);
    bench(
        &format!("burn matmul F32 [{tokens}x1536]x[1536x1536]"),
        &device,
        || x.clone().matmul(w.clone()),
        2.0 * (tokens * 1536 * 1536) as f64,
    );
}

/// GEMM tile sweep on the feed-forward up projection (`RAIW_GEMM_TILES="tr,tc,mr,mc,depth,vector,pad;..."`).
#[test]
#[ignore = "GPU tuning run"]
fn gemm_tile_sweep() {
    use raiw_gpu::kernels::{GemmTile, gemm::TILE_F16, linear_tuned};
    let Some(device) = gpu() else { return };
    let tokens = env_usize("RAIW_ATTN_TOKENS", 4096);
    let (input, output) = (1536, 8960);
    let x = upload(
        &seeded(tokens * input, 5, 1.0),
        tokens,
        input,
        &device,
        DType::F16,
    );
    let w = upload(
        &seeded(input * output, 6, 0.02),
        output,
        input,
        &device,
        DType::F16,
    );
    let b = upload1(&seeded(output, 7, 0.1), &device);
    let tile = |tr, tc, mr, mc, depth, vector, pad| GemmTile {
        row_threads: tr,
        col_threads: tc,
        rows: mr,
        cols: mc,
        depth,
        vector,
        pad,
    };
    let tiles: Vec<GemmTile> = match std::env::var("RAIW_GEMM_TILES") {
        Ok(spec) => spec
            .split(';')
            .map(|t| {
                let n: Vec<usize> = t.split(',').map(|x| x.trim().parse().unwrap()).collect();
                tile(n[0], n[1], n[2], n[3], n[4], n[5], n[6])
            })
            .collect(),
        Err(_) => vec![
            TILE_F16,
            tile(16, 16, 8, 8, 64, 8, 1),
            tile(16, 16, 8, 8, 32, 8, 2),
            tile(16, 16, 8, 8, 16, 8, 1),
            tile(16, 16, 8, 4, 32, 8, 1),
            tile(8, 16, 8, 8, 32, 8, 1),
            tile(16, 8, 8, 8, 32, 8, 1),
        ],
    };
    for t in tiles {
        if t.shared_bytes(2) > 65_536 {
            continue;
        }
        bench(
            &format!(
                "gemm f16 {}x{} thr {}x{} per depth {} v{} pad {} {}KB",
                t.row_threads,
                t.col_threads,
                t.rows,
                t.cols,
                t.depth,
                t.vector,
                t.pad,
                t.shared_bytes(2) / 1024
            ),
            &device,
            || {
                linear_tuned(
                    x.clone(),
                    &DenseWeight::torch(w.clone()),
                    b.clone(),
                    Activation::GeluTanh,
                    DType::F16,
                    t,
                )
            },
            2.0 * (tokens * input * output) as f64,
        );
    }
}

/// The causal conv kernel on Wan VAE decoder shapes (one output frame) against the im2col
/// `Conv2d` path it replaces (three time taps, summed).
#[test]
#[ignore = "GPU timing run"]
fn vae_conv_throughput() {
    use raiw_gpu::layers::{Conv2d, Conv2dSpec};
    let Some(device) = gpu() else { return };
    let scale = env_usize("RAIW_CONV_SCALE", 1);
    for (channels, h, w) in [(384, 60, 104), (384, 120, 208), (192, 240, 416), (96, 480, 832)] {
        let (h, w) = (h / scale, w / scale);
        let s = ConvShape {
            in_channels: channels,
            out_channels: channels,
            time_kernel: 3,
            kernel: 3,
            padding: 1,
            upsample: false,
        };
        let flops = 2.0 * (h * w * channels * s.depth()) as f64;
        let x: Tensor<4> = Tensor::from_data(
            TensorData::new(seeded(3 * channels * h * w, 1, 1.0), [3, channels, h, w]),
            device.burn(),
        );
        let weight = seeded(channels * s.depth(), 2, 0.02);
        let wt = upload(
            &conv_weight(&weight, &s),
            s.depth(),
            s.weight_stride(),
            &device,
            DType::F32,
        );
        let bias = upload1(&seeded(channels, 3, 0.1), &device);
        bench4(
            &format!("causal conv kernel {channels}ch {h}x{w}"),
            &device,
            || causal_conv3d(x.clone(), wt.clone(), bias.clone(), None, 0, s),
            flops,
        );
        let spec = Conv2dSpec::new(channels, channels, 3).padding(1);
        let taps: Vec<Conv2d> = (0..3)
            .map(|t| {
                let k: Tensor<4> = Tensor::from_data(
                    TensorData::new(
                        seeded(channels * channels * 9, 10 + t, 0.02),
                        [channels, channels, 3, 3],
                    ),
                    device.burn(),
                );
                Conv2d::from_parts(k, None, spec, device.burn())
            })
            .collect();
        bench4(
            &format!("im2col conv2d x3  {channels}ch {h}x{w}"),
            &device,
            || {
                (0..3)
                    .map(|t| taps[t].forward(x.clone().slice(s![t..t + 1])))
                    .reduce(|a, b| a + b)
                    .unwrap()
            },
            flops,
        );
    }
}

fn bench4(label: &str, device: &ComputeDevice, mut run: impl FnMut() -> Tensor<4>, flops: f64) {
    bench(
        label,
        device,
        || run().flatten::<1>(0, 3).unsqueeze_dim::<2>(0),
        flops,
    );
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

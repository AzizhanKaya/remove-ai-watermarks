//! GPU-vs-CPU kernel cross-checks on shapes the ported models hit.

use raiw_gpu::burn::tensor::activation::sigmoid;
use raiw_gpu::burn::tensor::module::conv2d;
use raiw_gpu::burn::tensor::ops::ConvOptions;
use raiw_gpu::burn::tensor::{Tensor, TensorData};
use raiw_gpu::io::read_f32;
use raiw_gpu::{ComputeDevice, DeviceRequest};

fn seeded(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (f64::from(state) / 4_294_967_296.0 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn gpu() -> Option<ComputeDevice> {
    ComputeDevice::select(DeviceRequest::Auto)
        .ok()
        .filter(ComputeDevice::is_gpu)
}

fn conv_diff(device: &ComputeDevice, input: [usize; 4], out_channels: usize) -> f32 {
    let [n, c, h, w] = input;
    let x = seeded(n * c * h * w, 1);
    let weight = seeded(out_channels * c * 9, 2);
    let bias = seeded(out_channels, 3);
    let run = |dev: &ComputeDevice| {
        let x: Tensor<4> = Tensor::from_data(TensorData::new(x.clone(), input), dev.burn());
        let k: Tensor<4> = Tensor::from_data(
            TensorData::new(weight.clone(), [out_channels, c, 3, 3]),
            dev.burn(),
        );
        let b: Tensor<1> = Tensor::from_data(TensorData::new(bias.clone(), [out_channels]), dev.burn());
        read_f32(conv2d(x, k, Some(b), ConvOptions::new([1, 1], [1, 1], [1, 1], 1))).unwrap()
    };
    let cpu = run(&ComputeDevice::cpu());
    let other = run(device);
    cpu.iter()
        .zip(&other)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn instance_norm(x: Tensor<4>) -> Tensor<4> {
    let [n, c, h, w] = x.dims();
    let flat = x.reshape([n, c, h * w]);
    let mean = flat.clone().mean_dim(2);
    let centered = flat - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    (centered / var.add_scalar(1e-5).sqrt()).reshape([n, c, h, w])
}

#[test]
fn gated_instance_norm_matches_cpu() {
    let Some(device) = gpu() else {
        return;
    };
    for frames in [67, 71] {
        let shape = [1, 128, 1024, frames];
        let len = shape.iter().product();
        let (a, b) = (seeded(len, 4), seeded(len, 5));
        let run = |dev: &ComputeDevice| {
            let a: Tensor<4> = Tensor::from_data(TensorData::new(a.clone(), shape), dev.burn());
            let b: Tensor<4> = Tensor::from_data(TensorData::new(b.clone(), shape), dev.burn());
            let h = a * sigmoid(b);
            let staged = {
                let mean = h.clone().mean_dim(3).mean_dim(2);
                let centered = h.clone() - mean;
                let var = centered.clone().powf_scalar(2.0).mean_dim(3).mean_dim(2);
                centered / var.add_scalar(1e-5).sqrt()
            };
            let mut out = read_f32(instance_norm(h)).unwrap();
            out.extend(read_f32(staged).unwrap());
            out
        };
        let cpu = run(&ComputeDevice::cpu());
        let other = run(&device);
        let diff = cpu
            .iter()
            .zip(&other)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        eprintln!("gated instance norm 128x1024x{frames}: max diff {diff:.2e}");
        assert!(diff < 1e-3, "gated instance norm {frames}: diff {diff}");
    }
}

#[test]
#[ignore = "micro-benchmark of conv1d lowering on the GPU"]
fn conv1d_lowering_timings() {
    use raiw_gpu::burn::tensor::module::conv1d;
    let Some(device) = gpu() else {
        return;
    };
    for (channels, length, kernel) in [(256, 1250, 7), (96, 44_100, 7), (768, 689, 7), (512, 4000, 1)] {
        let x: Tensor<3> = Tensor::from_data(
            TensorData::new(seeded(channels * length, 1), [1, channels, length]),
            device.burn(),
        );
        let w: Tensor<3> = Tensor::from_data(
            TensorData::new(
                seeded(channels * channels * kernel, 2),
                [channels, channels, kernel],
            ),
            device.burn(),
        );
        let pad = kernel / 2;
        let direct = || {
            let options = ConvOptions::new_with_padding([1], [(pad, pad)], [1], 1);
            conv1d(x.clone(), w.clone(), None, options)
        };
        let lowered = || {
            let padded = x.clone().pad([(pad, pad)], 0.0);
            let taps: Vec<Tensor<3>> = (0..kernel).map(|k| padded.clone().narrow(2, k, length)).collect();
            let cols = Tensor::cat(taps, 1);
            let weight = w
                .clone()
                .swap_dims(1, 2)
                .reshape([1, channels, channels * kernel]);
            weight.matmul(cols)
        };
        let time = |run: &dyn Fn() -> Tensor<3>| {
            read_f32(run()).unwrap();
            let start = std::time::Instant::now();
            for _ in 0..3 {
                read_f32(run()).unwrap();
            }
            start.elapsed().as_secs_f64() / 3.0
        };
        let a = read_f32(direct()).unwrap();
        let b = read_f32(lowered()).unwrap();
        let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        eprintln!(
            "conv1d {channels}ch L={length} k={kernel}: direct {:.4}s lowered {:.4}s (diff {diff:.1e})",
            time(&direct),
            time(&lowered)
        );
    }
}

#[test]
#[ignore = "micro-benchmark of conv2d lowering on the GPU"]
fn conv2d_lowering_timings() {
    use raiw_gpu::layers::{Conv2d, Conv2dSpec};
    let Some(device) = gpu() else {
        return;
    };
    for (channels, side) in [(128, 256), (256, 128), (512, 64)] {
        let x: Tensor<4> = Tensor::from_data(
            TensorData::new(seeded(channels * side * side, 1), [1, channels, side, side]),
            device.burn(),
        );
        let w: Tensor<4> = Tensor::from_data(
            TensorData::new(seeded(channels * channels * 9, 2), [channels, channels, 3, 3]),
            device.burn(),
        );
        let spec = Conv2dSpec::new(channels, channels, 3).padding(1).no_bias();
        let lowered = Conv2d::from_parts(w.clone(), None, spec, device.burn());
        let direct = || {
            conv2d(
                x.clone(),
                w.clone(),
                None,
                ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
            )
        };
        let time = |run: &dyn Fn() -> Tensor<4>| {
            read_f32(run()).unwrap();
            let start = std::time::Instant::now();
            read_f32(run()).unwrap();
            start.elapsed().as_secs_f64()
        };
        let a = read_f32(direct()).unwrap();
        let b = read_f32(lowered.forward(x.clone())).unwrap();
        let diff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        eprintln!(
            "conv2d {channels}ch {side}x{side}: direct {:.4}s lowered {:.4}s (diff {diff:.1e})",
            time(&direct),
            time(&|| lowered.forward(x.clone()))
        );
    }
}

#[test]
fn conv2d_matches_cpu_on_spectrogram_shapes() {
    let Some(device) = gpu() else {
        return;
    };
    for frames in [67, 71] {
        for (channels, out) in [(1, 128), (128, 128), (128, 5)] {
            let diff = conv_diff(&device, [1, channels, 1024, frames], out);
            eprintln!("conv2d {channels}->{out} on 1024x{frames}: max diff {diff:.2e}");
            assert!(diff < 1e-3, "conv2d {channels}->{out} 1024x{frames} diff {diff}");
        }
    }
}

/// A layer's state dict from seeded weights.
fn seeded_state(tensors: &[(&str, Vec<usize>)]) -> raiw_gpu::StateDict {
    let map = tensors
        .iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let data = seeded(shape.iter().product(), 10 + i as u32);
            (
                (*name).to_owned(),
                raiw_gpu::HostTensor::new(shape.clone(), data).unwrap(),
            )
        })
        .collect();
    raiw_gpu::StateDict::from_tensors("seeded", map)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "output lengths differ");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn conv1d_lowerings_match_cpu() {
    use raiw_gpu::layers::{Conv1d, Conv1dSpec};
    let Some(device) = gpu() else {
        return;
    };
    let specs = [
        Conv1dSpec::new(64, 128, 8).stride(4).padding(2),
        Conv1dSpec::new(32, 64, 10).stride(5),
        Conv1dSpec::new(48, 48, 7).dilation(3).padding(9).groups(48),
        Conv1dSpec::new(48, 48, 7).stride(2).padding(3).groups(48),
        Conv1dSpec::new(40, 24, 7).dilation(2).padding(6),
        Conv1dSpec::new(40, 24, 1).no_bias(),
    ];
    for spec in specs {
        let length = 1_003;
        let x = seeded(2 * spec.in_channels * length, 7);
        let run = |dev: &ComputeDevice| {
            let mut state = seeded_state(&[
                (
                    "c.weight",
                    vec![spec.out_channels, spec.in_channels / spec.groups, spec.kernel],
                ),
                ("c.bias", vec![spec.out_channels]),
            ]);
            let conv = Conv1d::load(&mut state, "c", spec, dev.burn()).unwrap();
            let x: Tensor<3> = Tensor::from_data(
                TensorData::new(x.clone(), [2, spec.in_channels, length]),
                dev.burn(),
            );
            let mut out = read_f32(conv.forward(x.clone())).unwrap();
            out.extend(read_f32(conv.forward_padded(x, (5, 1))).unwrap());
            out
        };
        let diff = max_diff(&run(&ComputeDevice::cpu()), &run(&device));
        eprintln!("conv1d {spec:?}: max diff {diff:.2e}");
        assert!(diff < 1e-4, "conv1d {spec:?}: diff {diff}");
    }
}

#[test]
fn conv_transpose1d_lowering_matches_cpu() {
    use raiw_gpu::layers::{ConvTranspose1d, ConvTranspose1dSpec};
    let Some(device) = gpu() else {
        return;
    };
    let specs = [
        ConvTranspose1dSpec::new(64, 32, 16, 8).padding(4),
        ConvTranspose1dSpec::new(16, 8, 20, 20),
        ConvTranspose1dSpec::new(24, 12, 6, 3)
            .padding(2)
            .output_padding(1),
        ConvTranspose1dSpec::new(24, 12, 7, 3)
            .padding(1)
            .output_padding(2),
        ConvTranspose1dSpec::new(8, 8, 4, 2).output_padding(1),
    ];
    for spec in specs {
        let steps = 157;
        let x = seeded(2 * spec.in_channels * steps, 8);
        let run = |dev: &ComputeDevice| {
            let mut state = seeded_state(&[
                ("t.weight", vec![spec.in_channels, spec.out_channels, spec.kernel]),
                ("t.bias", vec![spec.out_channels]),
            ]);
            let conv = ConvTranspose1d::load(&mut state, "t", spec, dev.burn()).unwrap();
            let x: Tensor<3> = Tensor::from_data(
                TensorData::new(x.clone(), [2, spec.in_channels, steps]),
                dev.burn(),
            );
            read_f32(conv.forward(x)).unwrap()
        };
        let diff = max_diff(&run(&ComputeDevice::cpu()), &run(&device));
        eprintln!("conv_transpose1d {spec:?}: max diff {diff:.2e}");
        assert!(diff < 1e-4, "conv_transpose1d {spec:?}: diff {diff}");
    }
}

#[test]
#[ignore = "micro-benchmark of depthwise conv2d lowering on the GPU"]
fn depthwise_conv2d_timings() {
    use raiw_gpu::layers::{Conv2d, Conv2dSpec};
    let Some(device) = gpu() else {
        return;
    };
    for (channels, side) in [(96, 64), (192, 32), (384, 16), (768, 8)] {
        let shape = [16, channels, side, side];
        let x: Tensor<4> = Tensor::from_data(
            TensorData::new(seeded(shape.iter().product(), 1), shape),
            device.burn(),
        );
        let w: Tensor<4> = Tensor::from_data(
            TensorData::new(seeded(channels * 49, 2), [channels, 1, 7, 7]),
            device.burn(),
        );
        let spec = Conv2dSpec::new(channels, channels, 7)
            .padding(3)
            .groups(channels)
            .no_bias();
        let lowered = Conv2d::from_parts(w.clone(), None, spec, device.burn());
        let options = ConvOptions::new([1, 1], [3, 3], [1, 1], channels);
        let direct = || conv2d(x.clone(), w.clone(), None, options.clone());
        let time = |run: &dyn Fn() -> Tensor<4>| {
            read_f32(run()).unwrap();
            let start = std::time::Instant::now();
            for _ in 0..3 {
                read_f32(run()).unwrap();
            }
            start.elapsed().as_secs_f64() / 3.0
        };
        let diff = max_diff(
            &read_f32(direct()).unwrap(),
            &read_f32(lowered.forward(x.clone())).unwrap(),
        );
        eprintln!(
            "dwconv2d {channels}ch {side}x{side} x16: direct {:.4}s lowered {:.4}s (diff {diff:.1e})",
            time(&direct),
            time(&|| lowered.forward(x.clone()))
        );
    }
}

#[test]
#[ignore = "micro-benchmark of the SD VAE im2col matmul shapes on the GPU"]
fn vae_matmul_throughput() {
    use raiw_gpu::burn::tensor::FloatDType;
    let Some(device) = gpu() else {
        return;
    };
    for (m, k, n) in [(512, 4608, 9216), (256, 2304, 36864), (128, 1152, 36864)] {
        for dtype in [FloatDType::F32, FloatDType::F16] {
            let upload = |data, shape: [usize; 3]| {
                Tensor::<3>::from_data(TensorData::new(data, shape), device.burn()).cast(dtype)
            };
            let (a, b) = (
                upload(seeded(m * k, 1), [1, m, k]),
                upload(seeded(k * n, 2), [1, k, n]),
            );
            let run = || a.clone().matmul(b.clone()).sum().cast(FloatDType::F32);
            read_f32(run()).unwrap();
            let start = std::time::Instant::now();
            for _ in 0..3 {
                read_f32(run()).unwrap();
            }
            let secs = start.elapsed().as_secs_f64() / 3.0;
            let tflops = 2.0 * (m * k * n) as f64 / secs / 1e12;
            eprintln!("matmul {dtype:?} {m}x{k} @ {k}x{n}: {secs:.4}s ({tflops:.2} TFLOP/s)");
        }
    }
}

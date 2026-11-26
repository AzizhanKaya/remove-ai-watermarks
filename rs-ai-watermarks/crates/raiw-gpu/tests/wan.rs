//! Wan 2.1 parity against `goldens/gpu_wan/*.json` (diffusers float32 on CPU).

mod common;

use std::process::Command;
use std::time::Instant;

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use common::{as_f64s, devices, fetch, loaded, max_abs_diff, strided_diff, synthetic_signal, tolerance};
use raiw_core::testing::{fixture, golden};
use raiw_gpu::models::video::wan::pipeline::{bgr_u8_to_unit, unit_to_bgr_u8};
use raiw_gpu::models::video::wan::prompt::DEFAULT_PROMPT;
use raiw_gpu::models::video::wan::{
    BlockResidency, PromptEmbedding, UniPcScheduler, WanPipeline, WanTransformer, WanVae,
};
use raiw_gpu::rng::torch_randn;
use raiw_gpu::{GpuError, Hub, StateDict};
use serde_json::Value;

#[test]
fn scheduler_matches_diffusers() {
    let golden = golden("gpu_wan", "scheduler");
    let reference = UniPcScheduler::new(20, 3.0).unwrap();
    let timesteps: Vec<i64> = golden["timesteps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_i64().unwrap())
        .collect();
    assert_eq!(reference.timesteps(), timesteps.as_slice());
    let sigma_diff = max_abs_diff(reference.sigmas(), &as_f64s(&golden["sigmas"]));
    assert!(sigma_diff == 0.0, "sigma diff {sigma_diff}");
    let n = dim(&golden, "samples");
    for (name, case) in golden["cases"].as_object().unwrap() {
        let mut scheduler = UniPcScheduler::new(20, 3.0).unwrap();
        let strength = case["strength"].as_f64().unwrap();
        let steps = scheduler.strength_timesteps(strength).to_vec();
        let expected: Vec<i64> = case["timesteps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_i64().unwrap())
            .collect();
        assert_eq!(steps, expected, "{name} timesteps");
        let trajectory = case["trajectory"].as_array().unwrap();
        let original = synthetic_signal(n, 977, 80);
        let noise = torch_randn(n, 81).unwrap();
        let mut sample = scheduler.add_noise(&original, &noise, steps[0]).unwrap();
        let mut worst = strided_diff(&sample, &trajectory[0]);
        for (i, &t) in steps.iter().enumerate() {
            let velocity: Vec<f32> = synthetic_signal(n, 977, 90 + i as u32)
                .iter()
                .zip(&sample)
                .map(|(v, x)| v + 0.25 * x)
                .collect();
            sample = scheduler.step(&velocity, t, &sample).unwrap();
            worst = worst.max(strided_diff(&sample, &trajectory[i + 1]));
        }
        eprintln!("unipc {name}: max diff {worst:.2e}");
        assert!(worst <= 1e-5, "{name}: trajectory diff {worst}");
    }
}

#[test]
fn prompt_cache_path_and_reader_match_python() {
    let golden = golden("gpu_wan", "prompt");
    assert_eq!(
        PromptEmbedding::cache_file_name(DEFAULT_PROMPT),
        golden["default_cache_file"].as_str().unwrap()
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("embeds.pt");
    let bytes = BASE64_STANDARD
        .decode(golden["pt_b64"].as_str().unwrap())
        .unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let mut state = StateDict::from_torch_zip(&path, None, &[], |_| true).unwrap();
    let tensor = state.take_any("").unwrap();
    assert_eq!(tensor.shape, [1, 4, 8]);
    assert_eq!(max_abs_diff(&tensor.data, &as_f64s(&golden["values"])), 0.0);
    assert!(matches!(
        PromptEmbedding::from_file(&path, None),
        Err(GpuError::ShapeMismatch { .. })
    ));
    let digest = raiw_core::sha256_bytes(&bytes);
    assert!(matches!(
        PromptEmbedding::from_file(&path, Some(&"0".repeat(64))),
        Err(GpuError::DigestMismatch { .. })
    ));
    assert!(matches!(
        PromptEmbedding::from_file(&path, Some(&digest)),
        Err(GpuError::ShapeMismatch { .. })
    ));
}

fn dim(record: &Value, key: &str) -> usize {
    record[key].as_u64().unwrap() as usize
}

fn clip(record: &Value) -> (Vec<f32>, usize, usize, usize) {
    let (frames, height, width) = (dim(record, "frames"), dim(record, "height"), dim(record, "width"));
    let pixels = synthetic_signal(
        frames * 3 * height * width,
        dim(record, "rate") as u32,
        dim(record, "seed") as u32,
    );
    (pixels, frames, height, width)
}

#[test]
fn vae_matches_diffusers() {
    let golden = golden("gpu_wan", "vae");
    for device in devices() {
        let Some(vae) = loaded(
            WanVae::load_from_hub(&Hub::from_env(), fetch(), &device),
            "Wan VAE",
        ) else {
            return;
        };
        for (name, case) in golden["cases"].as_object().unwrap() {
            let (pixels, frames, height, width) = clip(&case["input"]);
            let start = Instant::now();
            let latents = vae.encode(&pixels, frames, height, width).unwrap();
            let encode = start.elapsed().as_secs_f64();
            let latent_diff = strided_diff(&latents, &case["latents"]);
            let shape = &case["latent_shape"];
            let (t, h, w) = (dim_at(shape, 1), dim_at(shape, 2), dim_at(shape, 3));
            let start = Instant::now();
            let decoded = vae.decode(&latents, t, h, w).unwrap();
            let decode = start.elapsed().as_secs_f64();
            let decoded_diff = strided_diff(&decoded, &case["decoded"]);
            eprintln!(
                "wan vae {name} on {device}: latents {latent_diff:.2e} decoded {decoded_diff:.2e} \
                 (encode {encode:.2}s decode {decode:.2}s)"
            );
            let limit = tolerance(&device, 1e-4, 5e-4);
            assert!(
                latent_diff <= limit,
                "{name} on {device}: latent diff {latent_diff}"
            );
            assert!(
                decoded_diff <= limit,
                "{name} on {device}: decoded diff {decoded_diff}"
            );
        }
    }
}

#[test]
fn transformer_matches_diffusers() {
    let golden = golden("gpu_wan", "transformer");
    let input = &golden["latents"];
    let (frames, height, width) = (dim(input, "frames"), dim(input, "height"), dim(input, "width"));
    let latents = synthetic_signal(
        16 * frames * height * width,
        dim(input, "rate") as u32,
        dim(input, "seed") as u32,
    );
    let context = &golden["context"];
    let prompt = PromptEmbedding::from_values(synthetic_signal(
        dim(context, "tokens") * dim(context, "dim"),
        dim(context, "rate") as u32,
        dim(context, "seed") as u32,
    ))
    .unwrap();
    for device in devices() {
        let transformer =
            WanTransformer::load_from_hub(&Hub::from_env(), fetch(), &device, BlockResidency::Streamed);
        let Some(transformer) = loaded(transformer, "Wan transformer") else {
            return;
        };
        let text = transformer.encode_text(&prompt);
        for (name, case) in golden["cases"].as_object().unwrap() {
            let timestep = case["timestep"].as_f64().unwrap() as f32;
            let start = Instant::now();
            let out = transformer
                .forward(&latents, (frames, height, width), timestep, &text)
                .unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            let diff = strided_diff(&out, &case["output"]);
            let scale = case["output"]["abs_mean"].as_f64().unwrap();
            eprintln!(
                "wan transformer {name} on {device}: diff {diff:.2e} (|out| {scale:.3}, {elapsed:.2}s)"
            );
            assert!(
                diff <= tolerance(&device, 1e-4, 5e-4),
                "{name} on {device}: diff {diff}"
            );
        }
    }
}

/// Fixture frames decoded by the generator's ffmpeg command, BGR `[T, H, W, 3]`.
fn fixture_frames(record: &Value) -> Option<Vec<u8>> {
    let strings = |key: &str| -> Vec<String> {
        record[key]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect()
    };
    let output = Command::new("ffmpeg")
        .args(strings("before"))
        .arg(fixture(record["fixture"].as_str()?))
        .args(strings("after"))
        .output()
        .ok()?;
    if raiw_core::sha256_bytes(&output.stdout) != record["decoded_sha256"].as_str()? {
        eprintln!("skipping fixture frames: local ffmpeg decodes different bytes");
        return None;
    }
    Some(output.stdout)
}

#[test]
fn pipeline_matches_run_wan_window() {
    let golden = golden("gpu_wan", "pipeline");
    let input = &golden["input"];
    let Some(bgr) = fixture_frames(input) else {
        return;
    };
    let (frames, height, width) = (dim(input, "frames"), dim(input, "height"), dim(input, "width"));
    let pixels = bgr_u8_to_unit(&bgr, height, width);
    let context = &golden["context"];
    let prompt = PromptEmbedding::from_values(synthetic_signal(
        dim(context, "tokens") * dim(context, "dim"),
        dim(context, "rate") as u32,
        dim(context, "seed") as u32,
    ))
    .unwrap();
    for device in devices() {
        // float32 burn ops match diffusers' float32 window to one level; the GPU's f16
        // kernels (the default there) measured max 1, mean 0.03 levels
        let mut engines = vec![(BlockResidency::Streamed, (1u8, 0.01))];
        #[cfg(feature = "gpu-runtime")]
        if device.is_gpu() {
            engines.push((
                BlockResidency::Fused(raiw_gpu::models::video::wan::Precision::F16),
                (2, 0.1),
            ));
        }
        for (residency, (max_levels, mean_levels)) in engines {
            let hub = Hub::from_env();
            let Some(vae) = loaded(WanVae::load_from_hub(&hub, fetch(), &device), "Wan VAE") else {
                return;
            };
            let transformer = WanTransformer::load_from_hub(&hub, fetch(), &device, residency);
            let Some(transformer) = loaded(transformer, "Wan transformer") else {
                return;
            };
            let pipeline = WanPipeline::new(vae, transformer, &prompt);
            for (name, case) in golden["cases"].as_object().unwrap() {
                let strength = case["strength"].as_f64().unwrap();
                let (steps, seed) = (dim(case, "steps"), case["seed"].as_u64().unwrap());
                let start = Instant::now();
                let out = pipeline
                    .regenerate_chunk(&pixels, (frames, height, width), strength, steps, seed)
                    .unwrap();
                let elapsed = start.elapsed().as_secs_f64();
                let out = unit_to_bgr_u8(&out, height, width);
                let expected = BASE64_STANDARD
                    .decode(case["output_bgr_b64"].as_str().unwrap())
                    .unwrap();
                assert_eq!(out.len(), expected.len(), "{name} output length");
                let diffs: Vec<u8> = out.iter().zip(&expected).map(|(a, b)| a.abs_diff(*b)).collect();
                let max = diffs.iter().copied().max().unwrap_or(0);
                let mean = diffs.iter().map(|&d| f64::from(d)).sum::<f64>() / diffs.len() as f64;
                let python = case["seconds_cpu_fp32"].as_f64().unwrap();
                eprintln!(
                    "wan pipeline {name} {residency:?} on {device}: max {max} mean {mean:.4} levels \
                     ({elapsed:.1}s; diffusers fp32 cpu {python:.1}s)"
                );
                assert!(
                    max <= max_levels && mean <= mean_levels,
                    "{name} {residency:?} on {device}: max {max} mean {mean}"
                );
            }
        }
    }
}

/// Reduced-size 480p timings for projecting a full window (run with `--ignored --nocapture`).
#[test]
#[ignore = "timing run; needs the cached Wan VAE and transformer"]
fn wan_480p_timing() {
    let Some(device) = devices().into_iter().find(raiw_gpu::ComputeDevice::is_gpu) else {
        eprintln!("skipping: no GPU");
        return;
    };
    let hub = Hub::from_env();
    let Some(vae) = loaded(WanVae::load_from_hub(&hub, fetch(), &device), "Wan VAE") else {
        return;
    };
    let (height, width) = (480, 832);
    for frames in [1usize, 5] {
        let pixels = synthetic_signal(frames * 3 * height * width, 977, 3);
        let start = Instant::now();
        let latents = vae.encode(&pixels, frames, height, width).unwrap();
        let encode = start.elapsed().as_secs_f64();
        let latent_frames = 1 + (frames - 1) / 4;
        let start = Instant::now();
        vae.decode(&latents, latent_frames, height / 8, width / 8)
            .unwrap();
        let decode = start.elapsed().as_secs_f64();
        println!("wan vae {frames}x{height}x{width} on {device}: encode {encode:.2}s decode {decode:.2}s");
    }
    drop(vae);
    // RAIW_WAN_RESIDENCY=streamed|fused-f32 overrides the device default (fused f16 on a GPU);
    // RAIW_WAN_FRAMES="1,5,21" picks the latent frame counts (21 = one 81-frame window)
    let residency = match std::env::var("RAIW_WAN_RESIDENCY").as_deref() {
        Ok("streamed") => BlockResidency::Streamed,
        #[cfg(feature = "gpu-runtime")]
        Ok("fused-f32") => BlockResidency::Fused(raiw_gpu::models::video::wan::Precision::F32),
        _ => BlockResidency::for_device(&device),
    };
    let start = Instant::now();
    let transformer = WanTransformer::load_from_hub(&hub, fetch(), &device, residency);
    let Some(transformer) = loaded(transformer, "Wan transformer") else {
        return;
    };
    println!(
        "wan transformer {residency:?} load: {:.1}s",
        start.elapsed().as_secs_f64()
    );
    let prompt = PromptEmbedding::from_values(synthetic_signal(512 * 4096, 977, 70)).unwrap();
    let text = transformer.encode_text(&prompt);
    let (rows, cols) = (height / 8, width / 8);
    let frames: Vec<usize> = std::env::var("RAIW_WAN_FRAMES").ok().map_or_else(
        || vec![1, 3, 5],
        |v| v.split(',').map(|f| f.trim().parse().unwrap()).collect(),
    );
    for latent_frames in frames {
        let latents = synthetic_signal(16 * latent_frames * rows * cols, 977, 4);
        let tokens = latent_frames * rows * cols / 4;
        for pass in ["cold", "warm"] {
            let start = Instant::now();
            transformer
                .forward(&latents, (latent_frames, rows, cols), 618.0, &text)
                .unwrap();
            println!(
                "wan transformer {latent_frames} latent frames ({tokens} tokens) {pass} on {device}: {:.2}s",
                start.elapsed().as_secs_f64()
            );
        }
    }
}

/// Host cost of streaming one block from the memory-mapped shards (run with `--ignored --nocapture`).
#[test]
#[ignore = "timing run; needs the cached transformer"]
fn transformer_block_streaming_timing() {
    let hub = Hub::from_env();
    let shards: Vec<_> = raiw_gpu::models::video::wan::TRANSFORMER_SHARDS
        .iter()
        .filter_map(|file| hub.cached(file))
        .collect();
    if shards.len() != 2 {
        eprintln!("skipping: transformer shards not cached");
        return;
    }
    for index in [5usize, 6, 5] {
        let scope = format!("blocks.{index}.");
        let start = Instant::now();
        let state = StateDict::from_safetensors_shards(&shards, |name| name.starts_with(&scope)).unwrap();
        eprintln!(
            "block {index}: {} tensors read in {:.3}s",
            state.len(),
            start.elapsed().as_secs_f64()
        );
    }
}

fn dim_at(shape: &Value, i: usize) -> usize {
    shape[i].as_u64().unwrap() as usize
}

fn cached_shards() -> Option<Vec<std::path::PathBuf>> {
    let hub = Hub::from_env();
    let shards: Vec<_> = raiw_gpu::models::video::wan::TRANSFORMER_SHARDS
        .iter()
        .filter_map(|file| hub.cached(file))
        .collect();
    (shards.len() == 2).then_some(shards)
}

/// Max |a - b| over max |b|.
fn relative_diff(actual: &[f32], expected: &[f32]) -> f64 {
    let scale = expected.iter().fold(1e-12f64, |m, x| m.max(f64::from(x.abs())));
    let diff = actual
        .iter()
        .zip(expected)
        .fold(0.0f64, |m, (a, b)| m.max(f64::from((a - b).abs())));
    diff / scale
}

/// The fused kernel blocks against the burn-op blocks on the checkpoint's first blocks.
#[cfg(feature = "gpu-runtime")]
#[test]
fn fused_blocks_match_reference_blocks() {
    use raiw_gpu::models::video::wan::Precision;
    let Some(device) = devices().into_iter().find(raiw_gpu::ComputeDevice::is_gpu) else {
        return;
    };
    let Some(shards) = cached_shards() else {
        eprintln!("skipping: transformer shards not cached");
        return;
    };
    let layers = 3;
    // 3 x 10 x 13 = 390 tokens: partial attention and GEMM tiles on every axis
    let (frames, height, width) = (3, 20, 26);
    let latents = synthetic_signal(16 * frames * height * width, 977, 4);
    let prompt = PromptEmbedding::from_values(synthetic_signal(512 * 4096, 977, 70)).unwrap();
    let run = |residency| {
        let model = WanTransformer::from_shards_truncated(
            raiw_gpu::models::video::wan::TransformerConfig::WAN21_T2V_1_3B,
            shards.clone(),
            &device,
            residency,
            layers,
        )
        .unwrap();
        let text = model.encode_text(&prompt);
        let start = Instant::now();
        let out = model
            .forward(&latents, (frames, height, width), 618.0, &text)
            .unwrap();
        (out, start.elapsed().as_secs_f64())
    };
    let (reference, seconds) = run(BlockResidency::Resident);
    eprintln!("reference {layers} blocks: {seconds:.2}s");
    for (precision, limit) in [
        (Precision::F32, 1e-4),
        (Precision::F16, 2e-2),
        (Precision::Int8, 2e-2),
    ] {
        let (fused, seconds) = run(BlockResidency::Fused(precision));
        let diff = relative_diff(&fused, &reference);
        eprintln!("fused {precision:?} {layers} blocks: relative diff {diff:.2e} ({seconds:.2}s)");
        assert!(diff <= limit, "{precision:?}: relative diff {diff}");
    }
}

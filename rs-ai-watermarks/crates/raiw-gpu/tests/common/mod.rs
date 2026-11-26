//! Shared helpers for the golden parity tests: inputs, devices, comparisons.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use raiw_core::testing::fixture;
use raiw_gpu::{ComputeDevice, DeviceRequest, GpuError};
use serde_json::Value;

/// `gpu_common.synthetic_signal`: two tones plus LCG noise.
pub fn synthetic_signal(samples: usize, sample_rate: u32, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..samples)
        .map(|n| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = f64::from(state) / 4_294_967_296.0 * 2.0 - 1.0;
            let t = n as f64 / f64::from(sample_rate);
            let tau = 2.0 * std::f64::consts::PI;
            (0.3 * (tau * 440.0 * t).sin() + 0.2 * (tau * 1234.5 * t + 0.3).sin() + 0.05 * noise) as f32
        })
        .collect()
}

/// Little-endian int16 PCM (base64) to the exact float32 values Python decoded.
pub fn pcm16(payload: &str) -> Vec<f32> {
    let bytes = BASE64_STANDARD
        .decode(payload)
        .expect("golden PCM payload is base64");
    bytes
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32767.0)
        .collect()
}

/// Decode a fixture's audio with the same ffmpeg command as the generator; `None` without ffmpeg.
pub fn fixture_audio(record: &Value) -> Option<Vec<f32>> {
    let name = record["fixture"].as_str()?;
    let path: PathBuf = fixture(name);
    let rate = record["sample_rate"].as_u64()?.to_string();
    let start = format_seconds(record["start"].as_f64()?);
    let duration = format_seconds(record["duration"].as_f64()?);
    let args = ["-v", "error", "-ss", &start, "-t", &duration, "-i"];
    let output = Command::new("ffmpeg")
        .args(args)
        .arg(&path)
        .args(["-vn", "-ac", "1", "-ar", &rate, "-f", "f32le", "-"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let digest = raiw_core::sha256_bytes(&output.stdout);
    if Some(digest.as_str()) != record["decoded_sha256"].as_str() {
        eprintln!("skipping {name}: local ffmpeg decodes different bytes than the golden");
        return None;
    }
    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let keep = record["truncate_to"]
        .as_u64()
        .map_or(samples.len(), |n| n as usize);
    Some(samples[..keep].to_vec())
}

fn format_seconds(value: f64) -> String {
    format!("{value:?}")
}

/// Rebuild a case input recorded by the generators.
pub fn case_input(record: &Value) -> Option<Vec<f32>> {
    match record["kind"].as_str()? {
        "synthetic" => Some(synthetic_signal(
            record["samples"].as_u64()? as usize,
            record["sample_rate"].as_u64()? as u32,
            record["seed"].as_u64()? as u32,
        )),
        "pcm16" => Some(pcm16(record["pcm16_b64"].as_str()?)),
        "fixture" => fixture_audio(record),
        _ => None,
    }
}

/// The devices to check: always CPU, plus the auto-selected GPU when there is one.
///
/// `RAIW_TEST_DEVICES=cpu` or `=gpu` restricts the run to one backend.
pub fn devices() -> Vec<ComputeDevice> {
    let only = std::env::var("RAIW_TEST_DEVICES").unwrap_or_default();
    let mut out = Vec::new();
    if only != "gpu" {
        out.push(ComputeDevice::cpu());
    }
    if only != "cpu"
        && let Ok(gpu) = ComputeDevice::select(DeviceRequest::Auto)
        && gpu.is_gpu()
    {
        out.push(gpu);
    }
    out
}

/// Parity tolerance: tight on CPU, looser for GPU reduction order and fast-math.
pub fn tolerance(device: &ComputeDevice, cpu: f64, gpu: f64) -> f64 {
    if device.is_gpu() { gpu } else { cpu }
}

/// Cache-only unless `RAIW_GPU_DOWNLOAD` allows fetching missing checkpoints.
pub fn fetch() -> raiw_gpu::Fetch {
    if std::env::var_os("RAIW_GPU_DOWNLOAD").is_some() {
        raiw_gpu::Fetch::Download
    } else {
        raiw_gpu::Fetch::CacheOnly
    }
}

/// Skip a test when a pinned checkpoint is not in the local hub cache.
pub fn loaded<T>(result: Result<T, GpuError>, what: &str) -> Option<T> {
    match result {
        Ok(model) => Some(model),
        Err(GpuError::NotCached { .. }) => {
            eprintln!("skipping {what}: checkpoint not cached");
            None
        }
        Err(err) => panic!("{what}: {err}"),
    }
}

/// Max abs difference between a Rust buffer sampled like `gpu_common.strided` and the golden excerpt.
pub fn strided_diff(actual: &[f32], golden: &Value) -> f64 {
    assert_eq!(
        actual.len() as u64,
        golden["len"].as_u64().unwrap(),
        "output length"
    );
    let step = golden["step"].as_u64().unwrap() as usize;
    let excerpt = golden["excerpt"].as_array().unwrap();
    let mut worst = 0.0f64;
    for (i, expected) in excerpt.iter().enumerate() {
        let diff = (f64::from(actual[i * step]) - expected.as_f64().unwrap()).abs();
        worst = worst.max(diff);
    }
    let rms = (actual.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / actual.len() as f64).sqrt();
    worst.max((rms - golden["rms"].as_f64().unwrap()).abs())
}

/// Max abs difference between two equally long buffers.
pub fn max_abs_diff(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (f64::from(a) - e).abs())
        .fold(0.0, f64::max)
}

pub fn as_f64s(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

pub fn as_codes(value: &Value) -> Vec<Vec<u32>> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_u64().unwrap() as u32)
                .collect()
        })
        .collect()
}

/// Fraction of mismatching codebook indices.
pub fn code_mismatch(actual: &[Vec<u32>], expected: &[Vec<u32>]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "codebook count");
    let mut total = 0usize;
    let mut wrong = 0usize;
    for (a, e) in actual.iter().zip(expected) {
        assert_eq!(a.len(), e.len(), "codebook length");
        total += a.len();
        wrong += a.iter().zip(e).filter(|(x, y)| x != y).count();
    }
    wrong as f64 / total.max(1) as f64
}

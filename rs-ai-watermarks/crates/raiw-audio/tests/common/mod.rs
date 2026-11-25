//! Shared golden-test helpers: bit-identical signal synthesis and array fingerprints.
#![allow(dead_code)]

use std::f64::consts::PI;
use std::path::Path;
use std::process::{Command, Stdio};

use raiw_audio::Frames;
use raiw_core::sha256_bytes;
use serde_json::Value;

const TWO_PI: f64 = 2.0 * PI;
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

pub fn golden(name: &str) -> Value {
    raiw_core::testing::golden("audio", name)
}

/// SplitMix64 uniforms in `[0, 1)`, as `audio_common.splitmix_uniform`.
pub fn splitmix_uniform(seed: u64, count: usize) -> Vec<f64> {
    (1..=count as u64)
        .map(|i| {
            let mut z = seed.wrapping_add(i.wrapping_mul(GOLDEN));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        })
        .collect()
}

fn num(spec: &Value, key: &str, default: f64) -> f64 {
    spec.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn component(spec: &Value, frames: usize, rate: f64) -> Vec<f64> {
    let t = |n: usize| n as f64 / rate;
    let amp = num(spec, "amp", 1.0);
    match spec["kind"].as_str().unwrap() {
        "tone" => {
            let (freq, phase) = (num(spec, "freq", 0.0), num(spec, "phase", 0.0));
            (0..frames)
                .map(|n| amp * (TWO_PI * freq * t(n) + phase).sin())
                .collect()
        }
        "noise" => splitmix_uniform(spec["seed"].as_u64().unwrap(), frames)
            .into_iter()
            .map(|u| amp * (2.0 * u - 1.0))
            .collect(),
        "chirp" => {
            let (f0, f1) = (num(spec, "f0", 0.0), num(spec, "f1", 0.0));
            let k = (f1 - f0) / (frames as f64 / rate);
            (0..frames)
                .map(|n| {
                    let t = t(n);
                    amp * (TWO_PI * (f0 * t + 0.5 * k * t * t)).sin()
                })
                .collect()
        }
        "burst" => {
            let (period, on) = (
                spec["period"].as_u64().unwrap() as usize,
                spec["on"].as_u64().unwrap() as usize,
            );
            let freq = num(spec, "freq", 0.0);
            (0..frames)
                .map(|n| amp * (TWO_PI * freq * t(n)).sin() * if n % period < on { 1.0 } else { 0.0 })
                .collect()
        }
        "dc" => vec![num(spec, "value", 0.0); frames],
        "silence" => vec![0.0; frames],
        other => panic!("unknown component {other}"),
    }
}

/// `(frames, channels)` float32 samples for a signal spec, as `audio_common.synth`.
pub fn synth(spec: &Value) -> Frames {
    let frames = spec["frames"].as_u64().unwrap() as usize;
    let rate = spec["rate"].as_f64().unwrap();
    let channels: Vec<Vec<f32>> = spec["channels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|parts| {
            let mut value = vec![0.0f64; frames];
            for part in parts.as_array().unwrap() {
                for (v, c) in value.iter_mut().zip(component(part, frames, rate)) {
                    *v += c;
                }
            }
            value.into_iter().map(|v| v as f32).collect()
        })
        .collect();
    Frames::from_channels(&channels)
}

pub fn sha_f32(values: &[f32]) -> String {
    sha256_bytes(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}

pub fn sha_f64(values: &[f64]) -> String {
    sha256_bytes(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}

/// Compare a float array against an `audio_common.digest`; returns whether it is bit-identical.
pub fn check_digest(actual: &[f64], sha: &str, expected: &Value, tolerance: f64, context: &str) -> bool {
    let shape: usize = expected["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .product();
    assert_eq!(actual.len(), shape, "{context}: length");
    let exact = sha == expected["sha256"].as_str().unwrap();
    let scale = expected["max_abs"].as_f64().unwrap().max(1e-12);
    let close = |a: f64, e: &Value, what: &str| {
        let e = e.as_f64().unwrap();
        assert!(
            (a - e).abs() <= tolerance * scale.max(e.abs()),
            "{context}: {what}: {a} != {e} (tol {tolerance})"
        );
    };
    let sum: f64 = actual.iter().sum();
    let sum_sq: f64 = actual.iter().map(|v| v * v).sum();
    close(sum, &expected["sum"], "sum");
    let sq_scale = expected["sum_sq"].as_f64().unwrap().max(1e-12);
    assert!(
        (sum_sq - expected["sum_sq"].as_f64().unwrap()).abs() <= tolerance * sq_scale.max(1.0) * 10.0,
        "{context}: sum_sq {sum_sq} != {}",
        expected["sum_sq"]
    );
    let max_abs = actual.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    close(max_abs, &expected["max_abs"], "max_abs");
    for (i, (pick, value)) in expected["picks"]
        .as_array()
        .unwrap()
        .iter()
        .zip(expected["values"].as_array().unwrap())
        .enumerate()
    {
        close(
            actual[pick.as_u64().unwrap() as usize],
            value,
            &format!("value #{i}"),
        );
    }
    for (i, value) in expected["head"].as_array().unwrap().iter().enumerate() {
        close(actual[i], value, &format!("head[{i}]"));
    }
    let tail = expected["tail"].as_array().unwrap();
    for (i, value) in tail.iter().enumerate() {
        close(
            actual[actual.len() - tail.len() + i],
            value,
            &format!("tail[{i}]"),
        );
    }
    exact
}

pub fn check_f32(actual: &[f32], expected: &Value, tolerance: f64, context: &str) -> bool {
    let wide: Vec<f64> = actual.iter().map(|&v| f64::from(v)).collect();
    check_digest(&wide, &sha_f32(actual), expected, tolerance, context)
}

pub fn check_f64(actual: &[f64], expected: &Value, tolerance: f64, context: &str) -> bool {
    check_digest(actual, &sha_f64(actual), expected, tolerance, context)
}

pub fn ffmpeg() -> String {
    std::env::var("RAIW_FFMPEG").unwrap_or_else(|_| "ffmpeg".into())
}

/// The `audio_common.encode_raw` ffmpeg invocation.
pub fn encode_raw(samples: &Frames, rate: u32, path: &Path, args: &[&str], layout: Option<&str>) {
    let mut command = Command::new(ffmpeg());
    command.args([
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        &rate.to_string(),
        "-ac",
    ]);
    command.arg(samples.channels().to_string());
    if let Some(layout) = layout {
        command.args(["-channel_layout", layout]);
    }
    command
        .args(["-i", "pipe:0"])
        .args(args)
        .args(["-fflags", "+bitexact", "-flags:a", "+bitexact"])
        .arg(path);
    let mut child = command.stdin(Stdio::piped()).spawn().expect("ffmpeg runs");
    {
        use std::io::Write;
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(&samples.to_le_bytes()).unwrap();
    }
    assert!(child.wait().unwrap().success(), "ffmpeg encode failed");
}

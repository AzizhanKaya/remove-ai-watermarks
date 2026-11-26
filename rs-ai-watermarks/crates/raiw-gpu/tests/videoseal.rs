//! VideoSeal-family detector parity against `goldens/gpu_videoseal/<scheme>.json`.

mod common;

use std::process::Command;

use common::{as_f64s, devices, fetch, loaded, max_abs_diff, synthetic_signal, tolerance};
use raiw_core::testing::{fixture, golden_path};
use raiw_gpu::Hub;
use raiw_gpu::models::video::{Scheme, VideoMessageDetector};
use serde_json::Value;

/// `(frames NHWC RGB in [0, 1], count, height, width)` of a recorded case input.
fn case_frames(record: &Value) -> Option<(Vec<f32>, usize, usize, usize)> {
    let dim = |key: &str| record[key].as_u64().map(|v| v as usize);
    let (count, height, width) = (dim("frames")?, dim("height")?, dim("width")?);
    match record["kind"].as_str()? {
        "synthetic" => {
            let values = synthetic_signal(
                count * height * width * 3,
                dim("rate")? as u32,
                dim("seed")? as u32,
            );
            Some((
                values.iter().map(|v| (v + 0.5).clamp(0.0, 1.0)).collect(),
                count,
                height,
                width,
            ))
        }
        "fixture_frames" => {
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
            Some((
                output.stdout.iter().map(|&b| f32::from(b) / 255.0).collect(),
                count,
                height,
                width,
            ))
        }
        _ => None,
    }
}

fn check(scheme: Scheme, name: &str) {
    let path = golden_path("gpu_videoseal", name);
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("skipping {name}: no golden");
        return;
    };
    let golden: Value = serde_json::from_str(&text).unwrap();
    for device in devices() {
        let Some(detector) = loaded(
            VideoMessageDetector::load_from_hub(scheme, &Hub::from_env(), fetch(), &device),
            name,
        ) else {
            return;
        };
        for (case_name, case) in golden["cases"].as_object().unwrap() {
            let Some((frames, count, height, width)) = case_frames(&case["input"]) else {
                continue;
            };
            let logits = detector.predict(&frames, count, height, width).unwrap();
            let presence: Vec<f32> = logits.iter().map(|row| row[0]).collect();
            let message: Vec<f32> = logits.iter().flat_map(|row| row[1..].iter().copied()).collect();
            let expected: Vec<f64> = case["message"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(as_f64s)
                .collect();
            // Logits reach tens; compare relative to the largest magnitude in the case
            let scale = expected.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            let presence_diff = max_abs_diff(&presence, &as_f64s(&case["presence"])) / scale;
            let message_diff = max_abs_diff(&message, &expected) / scale;
            eprintln!(
                "{name} {case_name} on {device}: presence {presence_diff:.2e} message {message_diff:.2e}"
            );
            // ChunkySeal's 36-block, 2896-wide trunk amplifies f32 summation order on real frames
            let tol = match scheme {
                Scheme::ChunkySeal => 5e-3,
                _ => tolerance(&device, 1e-4, 2e-3),
            };
            // Bits whose reference logit is within the tolerance of zero may legitimately flip
            let flipped = message
                .iter()
                .zip(&expected)
                .filter(|(a, e)| (**a > 0.0) != (**e > 0.0) && e.abs() > tol * scale)
                .count();
            eprintln!("{name} {case_name} on {device}: {flipped} decisive message bit signs differ");
            assert!(
                presence_diff <= tol && message_diff <= tol && flipped == 0,
                "{name} {case_name} on {device}"
            );
        }
    }
}

/// The slice loader's parameter order and shapes, checked against the real VideoSeal graph.
#[test]
fn parameter_layout_covers_videoseal_graph() {
    use raiw_gpu::models::video::videoseal::{VIDEOSEAL_GRAPH, parameter_layout};
    let Some(path) = loaded(
        Hub::from_env().resolve_url(
            VIDEOSEAL_GRAPH.url,
            VIDEOSEAL_GRAPH.sha256,
            VIDEOSEAL_GRAPH.cache_path,
            fetch(),
        ),
        "VideoSeal graph",
    ) else {
        return;
    };
    let mut state = raiw_gpu::StateDict::from_torch_zip(&path, Some("detector"), &[], |_| true).unwrap();
    for (name, shape) in parameter_layout(Scheme::VideoSeal) {
        state.take(&name, &shape).unwrap();
    }
    state.finish().unwrap();
}

#[test]
fn videoseal_matches_python_reference() {
    check(Scheme::VideoSeal, "videoseal");
}

#[test]
fn pixelseal_matches_python_reference() {
    check(Scheme::PixelSeal, "pixelseal");
}

/// Needs the 3.1 GB detector slice that `video_message_models` caches; skipped otherwise.
#[test]
fn chunkyseal_matches_python_reference() {
    check(Scheme::ChunkySeal, "chunkyseal");
}

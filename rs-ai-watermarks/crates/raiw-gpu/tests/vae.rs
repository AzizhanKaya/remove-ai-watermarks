//! `SdVae` parity against `goldens/gpu_vae/autoencoder.json`.

mod common;

use std::process::Command;

use common::{as_f64s, devices, fetch, loaded, max_abs_diff, strided_diff, synthetic_signal};
use raiw_core::testing::{fixture, golden};
use raiw_gpu::Hub;
use raiw_gpu::models::video::SdVae;
use raiw_gpu::models::video::vae::bgr_u8_to_unit_rgb;
use serde_json::Value;

/// `(pixels, frames, height, width)` of a recorded case input.
fn case_pixels(record: &Value) -> Option<(Vec<f32>, usize, usize, usize)> {
    let dim = |key: &str| record[key].as_u64().map(|v| v as usize);
    match record["kind"].as_str()? {
        "synthetic" => {
            let (frames, height, width) = (dim("frames")?, dim("height")?, dim("width")?);
            let pixels = synthetic_signal(
                frames * 3 * height * width,
                dim("rate")? as u32,
                dim("seed")? as u32,
            );
            Some((pixels, frames, height, width))
        }
        "fixture_frame" => {
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
                eprintln!("skipping fixture frame: local ffmpeg decodes different bytes");
                return None;
            }
            let (height, width) = (dim("height")?, dim("width")?);
            Some((
                bgr_u8_to_unit_rgb(&output.stdout, height, width),
                1,
                height,
                width,
            ))
        }
        _ => None,
    }
}

/// One 512x512 frame, matching `tools/bench/gpu_video.py` (run with `--ignored --nocapture`).
#[test]
#[ignore = "timing run; needs the cached VAE"]
fn autoencoder_timing() {
    for device in devices() {
        let Some(vae) = loaded(SdVae::load_from_hub(&Hub::from_env(), fetch(), &device), "SD VAE") else {
            return;
        };
        let side = 512;
        let pixels = synthetic_signal(3 * side * side, 977, 1);
        let latents = vae.encode(&pixels, 1, side, side).unwrap();
        vae.decode(&latents, 1, side / 8, side / 8).unwrap();
        let start = std::time::Instant::now();
        let latents = vae.encode(&pixels, 1, side, side).unwrap();
        let encode = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        vae.decode(&latents, 1, side / 8, side / 8).unwrap();
        let decode = start.elapsed().as_secs_f64();
        println!("vae 512x512 on {device}: encode {encode:.3}s decode {decode:.3}s");
    }
}

#[test]
fn autoencoder_matches_diffusers() {
    let golden = golden("gpu_vae", "autoencoder");
    for device in devices() {
        let Some(mut vae) = loaded(SdVae::load_from_hub(&Hub::from_env(), fetch(), &device), "SD VAE") else {
            return;
        };
        for (name, case) in golden["cases"].as_object().unwrap() {
            let Some((pixels, frames, height, width)) = case_pixels(&case["input"]) else {
                continue;
            };
            vae.set_tiling(case["tiled"].as_bool().unwrap_or(false));
            let latents = vae.encode(&pixels, frames, height, width).unwrap();
            let latent_diff = max_abs_diff(&latents, &as_f64s(&case["latents"]));
            let decoded = vae.decode(&latents, frames, height / 8, width / 8).unwrap();
            let decoded_diff = strided_diff(&decoded, &case["decoded"]);
            eprintln!("vae {name} on {device}: latents {latent_diff:.2e} decoded {decoded_diff:.2e}");
            assert!(
                latent_diff <= 1e-4,
                "{name} on {device}: latent diff {latent_diff}"
            );
            assert!(
                decoded_diff <= 1e-4,
                "{name} on {device}: decoded diff {decoded_diff}"
            );
        }
    }
}

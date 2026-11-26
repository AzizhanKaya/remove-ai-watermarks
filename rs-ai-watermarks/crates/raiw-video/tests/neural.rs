//! The burn-backed neural runtimes end to end against Python (CPU torch) on short clips: SynthID
//! regeneration (8-bit, native guide, unstabilized, 10-bit), Seedance, and `video message`.
//!
//! `RAIW_VIDEO_TEST_DEVICE` picks the device (`auto` by default); models come from the caches.
#![cfg(feature = "gpu")]

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use base64::Engine;
use raiw_core::testing::{fixture, golden};
use raiw_cv::Mat;
use raiw_video::decode::VideoReader;
use raiw_video::invisible::{InvisibleOptions, regenerate_video_candidate};
use raiw_video::{
    MessageRemovalOptions, MessageScheme, SeedanceQuality, VideoRuntime, detect_video_watermark_presence,
    identify_video_message, regenerate_seedance_experimental, remove_video_message,
};
use serde_json::Value;

fn device() -> String {
    std::env::var("RAIW_VIDEO_TEST_DEVICE").unwrap_or_else(|_| "auto".into())
}

fn runtime() -> VideoRuntime {
    VideoRuntime::discover_on(&device()).expect("runtime")
}

fn b64(text: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(text)
        .expect("base64")
}

/// Rebuild the golden clips from the tracked fixture and check their bytes.
fn rebuild(doc: &Value, dir: &Path) -> Vec<(String, PathBuf)> {
    let base: Vec<String> = serde_json::from_value(doc["base_args"].clone()).expect("base");
    let source = fixture(doc["fixture"].as_str().expect("fixture"));
    let mut out = Vec::new();
    for (name, args) in doc["clips"].as_object().expect("clips") {
        let path = dir.join(format!("{name}.mp4"));
        let args: Vec<String> = args
            .as_array()
            .expect("args")
            .iter()
            .map(|a| match a.as_str().expect("arg") {
                "{fixture}" => source.to_string_lossy().into_owned(),
                other => other.to_owned(),
            })
            .collect();
        let status = Command::new("ffmpeg")
            .args(&base)
            .args(&args)
            .arg(&path)
            .status()
            .expect("ffmpeg");
        assert!(status.success(), "building {name}");
        let digest = common::sha256_hex(&std::fs::read(&path).expect("clip"));
        assert_eq!(
            Some(digest.as_str()),
            doc["sha256"][name].as_str(),
            "clip {name} differs"
        );
        out.push((name.clone(), path));
    }
    out
}

fn frames<T: raiw_video::invisible::VaePixel>(path: &Path) -> Vec<Mat<T>> {
    let mut reader = VideoReader::<T>::open(path).expect("open");
    std::iter::from_fn(|| reader.read().expect("read")).collect()
}

/// PSNR and maximum absolute difference between two decoded files of one depth.
fn frame_parity<T: raiw_video::invisible::VaePixel>(a: &Path, b: &Path) -> (f64, f64) {
    let (fa, fb) = (frames::<T>(a), frames::<T>(b));
    assert_eq!(fa.len(), fb.len(), "frame counts differ");
    let (mut se, mut n, mut max) = (0.0f64, 0usize, 0.0f64);
    for (x, y) in fa.iter().zip(&fb) {
        assert_eq!(x.size(), y.size());
        for (p, q) in x.data().iter().zip(y.data()) {
            let d = (p.to_f64() - q.to_f64()).abs();
            se += d * d;
            max = max.max(d);
            n += 1;
        }
    }
    let peak = T::PEAK;
    (10.0 * (peak * peak / (se / n as f64).max(1e-12)).log10(), max)
}

fn close(actual: f64, expected: &Value, tolerance: f64, what: &str) {
    let expected = expected.as_f64().expect(what);
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: rust {actual} vs python {expected} (tolerance {tolerance})"
    );
}

#[test]
#[ignore = "parity run: about four minutes of VAE work on the GPU (Seedance guides at 512 px)"]
fn vae_recipes_track_python() {
    common::ffmpeg_threads();
    let doc = golden("video_seedance", "recipes");
    let dir = tempfile::tempdir().expect("dir");
    let clips: std::collections::HashMap<String, PathBuf> = rebuild(&doc, dir.path()).into_iter().collect();
    let runtime = runtime();
    let vae = runtime.neural.vae.as_ref().expect("vae");
    let device = device();
    for (name, case) in doc["cases"].as_object().expect("cases") {
        let clip = &clips[case["clip"].as_str().expect("clip")];
        let output = dir.path().join(format!("{name}.mp4"));
        let opts = &case["options"];
        let started = Instant::now();
        let metrics = if name.starts_with("seedance") {
            let quality =
                SeedanceQuality::parse(opts["quality"].as_str().expect("quality")).expect("quality");
            regenerate_seedance_experimental(vae.as_ref(), clip, &output, &device, quality).expect("seedance")
        } else {
            let options = InvisibleOptions {
                profile: "synthid".into(),
                long_side: opts["long_side"].as_u64().expect("long side") as usize,
                native_guide: opts["native_guide"].as_bool().unwrap_or(false),
                temporal_stabilize: opts["temporal_stabilize"].as_bool().unwrap_or(true),
                batch_size: opts["batch_size"].as_u64().map_or(4, |b| b as usize),
                device: device.clone(),
                ..InvisibleOptions::default()
            };
            regenerate_video_candidate(vae.as_ref(), clip, &output, &options, None, None).expect("regenerate")
        };
        let elapsed = started.elapsed().as_secs_f64();
        let want = &case["metrics"];
        assert_eq!(
            metrics.frames as u64,
            want["frames"].as_u64().expect("frames"),
            "{name}"
        );
        assert_eq!(
            metrics.width as u64,
            want["width"].as_u64().expect("width"),
            "{name}"
        );
        assert_eq!(
            metrics.height as u64,
            want["height"].as_u64().expect("height"),
            "{name}"
        );
        close(metrics.fps, &want["fps"], 1e-9, "fps");
        let parity = case["output_b64"].as_str().map(|text| {
            let python = dir.path().join(format!("{name}-python.mp4"));
            std::fs::write(&python, b64(text)).expect("python output");
            if name.ends_with("10bit") {
                frame_parity::<u16>(&output, &python)
            } else {
                frame_parity::<u8>(&output, &python)
            }
        });
        eprintln!(
            "{name}: rust {elapsed:.2}s vs python {:.2}s; psnr {:.4} vs {}; ratio {:.4} vs {}; frames rust-vs-python {:?}",
            case["python_seconds"].as_f64().unwrap_or(f64::NAN),
            metrics.psnr_db,
            want["psnr_db"],
            metrics.temporal_residual_ratio,
            want["temporal_residual_ratio"],
            parity
        );
        // Stabilizer flow decisions flip on GPU-vs-CPU rounding, so the motion ratio is relative.
        let ratio = want["temporal_residual_ratio"].as_f64().expect("ratio");
        close(metrics.psnr_db, &want["psnr_db"], 0.25, &format!("{name} psnr"));
        close(
            metrics.temporal_residual_ratio,
            &want["temporal_residual_ratio"],
            0.04 * ratio,
            &format!("{name} temporal ratio"),
        );
        if let Some((psnr, _)) = parity {
            assert!(psnr > 36.0, "{name}: decoded output PSNR against Python {psnr}");
        }
    }
}

#[test]
#[ignore = "timing run: VAE batches versus the host-side orchestration"]
fn vae_phase_timings() {
    common::ffmpeg_threads();
    let doc = golden("video_seedance", "recipes");
    let dir = tempfile::tempdir().expect("dir");
    let clips: std::collections::HashMap<String, PathBuf> = rebuild(&doc, dir.path()).into_iter().collect();
    let runtime = runtime();
    let vae = runtime.neural.vae.as_ref().expect("vae");
    let frames: Vec<Mat<f32>> = (0..4)
        .map(|_| Mat::from_vec(256, 144, 3, vec![0.1; 256 * 144 * 3]).expect("frame"))
        .collect();
    vae.encode(&frames[..1]).expect("warm up");
    for round in 0..2 {
        let started = Instant::now();
        let latents = vae.encode(&frames).expect("encode");
        let encoded = started.elapsed().as_secs_f64();
        let started = Instant::now();
        vae.decode(&latents).expect("decode");
        eprintln!(
            "round {round}: encode 4x256x144 {encoded:.2}s, decode {:.2}s",
            started.elapsed().as_secs_f64()
        );
    }
    let started = Instant::now();
    let options = InvisibleOptions {
        profile: "synthid".into(),
        long_side: 256,
        device: device(),
        ..InvisibleOptions::default()
    };
    regenerate_video_candidate(
        vae.as_ref(),
        &clips["veo_320"],
        &dir.path().join("t.mp4"),
        &options,
        None,
        None,
    )
    .expect("regen");
    eprintln!("synthid_256 end to end {:.2}s", started.elapsed().as_secs_f64());
}

#[test]
fn video_messages_track_python() {
    common::ffmpeg_threads();
    let doc = golden("video_messages", "clip");
    let dir = tempfile::tempdir().expect("dir");
    let clip = dir.path().join("marked.mp4");
    std::fs::write(&clip, b64(doc["clip_b64"].as_str().expect("clip"))).expect("clip");
    let reference = doc["reference"].as_str().expect("reference");
    let cases = &doc["cases"];
    let runtime = runtime();

    let started = Instant::now();
    let report =
        identify_video_message(&runtime, &clip, reference, MessageScheme::VideoSeal, true).expect("identify");
    let identify_seconds = started.elapsed().as_secs_f64();
    let want = &cases["identify"];
    assert_eq!(report.status, want["status"].as_str().expect("status"));
    let reading = serde_json::to_value(report.reading.as_ref().expect("reading")).expect("json");
    close(
        reading["max_bit_accuracy"].as_f64().expect("max"),
        &want["reading"]["max_bit_accuracy"],
        0.02,
        "identify max bit accuracy",
    );
    for key in ["avg", "squared_avg", "l1norm_avg", "l2norm_avg"] {
        close(
            reading["whole_clip"][key].as_f64().expect("score"),
            &want["reading"]["whole_clip"][key],
            0.02,
            key,
        );
    }

    // PixelSeal's 1.2 GB checkpoint stays out of the default run (RSS budget); see `identify_other`.
    if std::env::var_os("RAIW_VIDEO_TEST_PIXELSEAL").is_some() {
        let other = identify_video_message(&runtime, &clip, &"0".repeat(64), MessageScheme::PixelSeal, true)
            .expect("pixelseal");
        assert_eq!(
            other.status,
            cases["identify_other"]["status"].as_str().expect("status")
        );
    }

    let presence =
        detect_video_watermark_presence(&runtime, &clip, MessageScheme::VideoSeal).expect("detect");
    let want = &cases["detect"];
    assert_eq!(presence.status, want["status"].as_str().expect("status"));
    close(
        presence.mean_presence_prob.expect("prob"),
        &want["mean_presence_prob"],
        0.02,
        "presence",
    );
    close(
        presence.bit_sign_consistency.expect("consistency"),
        &want["bit_sign_consistency"],
        0.03,
        "consistency",
    );

    let output = dir.path().join("remove_auto.mp4");
    let started = Instant::now();
    let removed = remove_video_message(
        &runtime,
        &clip,
        &output,
        &MessageRemovalOptions {
            reference_hex: reference.into(),
            long_side: 128,
            device: device(),
            ..MessageRemovalOptions::default()
        },
    )
    .expect("remove");
    let remove_seconds = started.elapsed().as_secs_f64();
    let want = &cases["remove_auto"];
    assert_eq!(removed.method.as_deref(), want["method"].as_str());
    assert_eq!(removed.noise_std, want["noise_std"].as_f64());
    assert_eq!(removed.verified_removed, want["verified_removed"].as_bool());
    assert_eq!(removed.copied_audio_streams, vec![1]);
    assert_eq!(removed.quality_passed, want["quality_passed"].as_bool());
    let quality = removed.quality.as_ref().expect("quality");
    for (key, tolerance) in [
        ("source_psnr_db", 0.3),
        ("source_ssim", 0.01),
        ("temporal_residual_ratio", 0.05),
    ] {
        close(
            quality[key].as_f64().expect(key),
            &want["quality"][key],
            tolerance,
            key,
        );
    }
    eprintln!(
        "video message: identify {identify_seconds:.2}s, remove(auto) {remove_seconds:.2}s; max bit accuracy {} vs {}; quality {} / {} vs {} / {}",
        reading["max_bit_accuracy"],
        cases["identify"]["reading"]["max_bit_accuracy"],
        quality["source_psnr_db"],
        quality["source_ssim"],
        want["quality"]["source_psnr_db"],
        want["quality"]["source_ssim"]
    );
}

//! Timings matching `tools/bench/gpu_audio.py` (run with `--ignored --nocapture --test-threads=1`).
//!
//! `RAIW_BENCH_DEVICES=gpu` or `=cpu` limits the run to one backend; one model is resident at a time.

mod common;

use std::hint::black_box;
use std::time::Instant;

use common::loaded;
use raiw_gpu::models::audio::audioseal::AudioSealDetector;
use raiw_gpu::models::audio::dac::DacCodec;
use raiw_gpu::models::audio::encodec::EncodecCodec;
use raiw_gpu::models::audio::perth::PerthDetector;
use raiw_gpu::models::audio::silentcipher::{SilentCipherDecoder, normalize_power};
use raiw_gpu::models::audio::snac::SnacCodec;
use raiw_gpu::models::audio::wavmark::WavMarkDecoder;
use raiw_gpu::{ComputeDevice, DeviceRequest, Fetch, Hub};

const REPEATS: usize = 2;
/// Codec inputs stay short: multi-second 44.1 kHz DAC/SNAC graphs trip the amdgpu ring timeout.
const CODEC_SECONDS: f64 = 1.0;

fn signal(rate: u32, seconds: f64) -> Vec<f32> {
    common::synthetic_signal((f64::from(rate) * seconds) as usize, rate, 0)
}

/// Best of `REPEATS` runs after one warm-up (which also absorbs autotuning).
fn best_seconds(mut run: impl FnMut()) -> f64 {
    run();
    (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn bench_devices() -> Vec<ComputeDevice> {
    let only = std::env::var("RAIW_BENCH_DEVICES").unwrap_or_default();
    let mut out = Vec::new();
    if only != "cpu"
        && let Ok(gpu) = ComputeDevice::select(DeviceRequest::Auto)
        && gpu.is_gpu()
    {
        out.push(gpu);
    }
    if only != "gpu" {
        out.push(ComputeDevice::cpu());
    }
    out
}

fn report(model: &str, seconds_audio: f64, device: &ComputeDevice, secs: f64) {
    println!("{model:<20} {seconds_audio:>4.1}s audio  {secs:>8.3}s  on {device}");
}

/// Cold phases of `video message detect`: load, first chunk (compile + tune), warm chunks.
#[test]
#[ignore = "timing run; needs cached checkpoints"]
fn video_message_timings() {
    use raiw_gpu::models::video::{Scheme, VideoMessageDetector};
    let hub = Hub::from_env();
    for device in bench_devices() {
        let start = Instant::now();
        let Some(model) = loaded(
            VideoMessageDetector::load_from_hub(Scheme::VideoSeal, &hub, Fetch::CacheOnly, &device),
            "VideoSeal",
        ) else {
            continue;
        };
        println!(
            "videoseal load {:>8.3}s  on {device}",
            start.elapsed().as_secs_f64()
        );
        let frames: Vec<f32> = (0..16 * 256 * 256 * 3)
            .map(|i| (i % 251) as f32 / 251.0)
            .collect();
        for (label, count) in [("cold16", 16), ("warm16", 16), ("tail10", 10), ("warm10", 10)] {
            let start = Instant::now();
            black_box(
                model
                    .predict(&frames[..count * 256 * 256 * 3], count, 256, 256)
                    .unwrap(),
            );
            println!(
                "videoseal {label} {:>8.3}s  on {device}",
                start.elapsed().as_secs_f64()
            );
        }
    }
}

#[test]
#[ignore = "timing run; needs cached checkpoints"]
fn audio_model_timings() {
    let hub = Hub::from_env();
    for device in bench_devices() {
        if let Some(model) = loaded(
            AudioSealDetector::load_from_hub(&hub, Fetch::CacheOnly, &device),
            "AudioSeal",
        ) {
            let audio = signal(16_000, 10.0);
            report(
                "audioseal",
                10.0,
                &device,
                best_seconds(|| drop(black_box(model.detect(&audio, 16_000)))),
            );
        }
        if let Some(model) = loaded(DacCodec::load_from_hub(&hub, Fetch::CacheOnly, &device), "DAC") {
            let mut audio = signal(44_100, CODEC_SECONDS);
            audio.truncate(audio.len() / 512 * 512);
            let codes = model.encode(&audio, 3).unwrap();
            report(
                "dac_encode",
                CODEC_SECONDS,
                &device,
                best_seconds(|| drop(black_box(model.encode(&audio, 3)))),
            );
            report(
                "dac_decode",
                CODEC_SECONDS,
                &device,
                best_seconds(|| drop(black_box(model.decode(&codes)))),
            );
        }
        if let Some(model) = loaded(SnacCodec::load_from_hub(&hub, Fetch::CacheOnly, &device), "SNAC") {
            let audio = signal(44_100, CODEC_SECONDS);
            report(
                "snac",
                CODEC_SECONDS,
                &device,
                best_seconds(|| drop(black_box(model.reconstruct(&audio, 3)))),
            );
        }
        if let Some(model) = loaded(
            EncodecCodec::load_from_hub(&hub, Fetch::CacheOnly, &device),
            "EnCodec",
        ) {
            let audio = signal(24_000, CODEC_SECONDS);
            let secs = best_seconds(|| drop(black_box(model.reconstruct(&audio, 6.0))));
            report("encodec_6kbps", CODEC_SECONDS, &device, secs);
        }
        if let Some(model) = loaded(
            WavMarkDecoder::load_from_hub(&hub, Fetch::CacheOnly, &device),
            "WavMark",
        ) {
            let audio = signal(16_000, 2.0);
            report(
                "wavmark_21_windows",
                2.0,
                &device,
                best_seconds(|| drop(black_box(model.read(&audio, 16_000)))),
            );
        }
        if let Some(model) = loaded(
            PerthDetector::load_from_hub(&hub, Fetch::CacheOnly, &device),
            "Perth",
        ) {
            let audio = signal(16_000, 10.0);
            report(
                "perth",
                10.0,
                &device,
                best_seconds(|| drop(black_box(model.read(&audio, 16_000)))),
            );
        }
        if let Some(model) = loaded(
            SilentCipherDecoder::load_from_hub(&hub, Fetch::CacheOnly, &device),
            "SilentCipher",
        ) {
            let audio = normalize_power(&signal(44_100, 3.0)).unwrap();
            report(
                "silentcipher_phase",
                3.0,
                &device,
                best_seconds(|| drop(black_box(model.scores(&audio)))),
            );
        }
    }
}

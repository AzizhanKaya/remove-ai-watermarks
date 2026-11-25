//! Release timings matching `tools/bench/gpu_audio.py` (run with `--ignored --nocapture`).
//!
//! `RAIW_BENCH_DEVICES=gpu` or `=cpu` limits the run to one backend.

mod common;

use std::time::Instant;

use common::loaded;
use raiw_gpu::models::audio::audioseal::AudioSealDetector;
use raiw_gpu::models::audio::dac::DacCodec;
use raiw_gpu::models::audio::perth::PerthDetector;
use raiw_gpu::models::audio::silentcipher::{SilentCipherDecoder, normalize_power};
use raiw_gpu::models::audio::snac::SnacCodec;
use raiw_gpu::models::audio::wavmark::WavMarkDecoder;
use raiw_gpu::{ComputeDevice, DeviceRequest, Fetch, Hub};

const REPEATS: usize = 3;

fn signal(rate: u32, seconds: f64) -> Vec<f32> {
    common::synthetic_signal((f64::from(rate) * seconds) as usize, rate, 0)
}

fn median_seconds(mut run: impl FnMut()) -> f64 {
    run();
    let mut samples: Vec<f64> = (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[REPEATS / 2]
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

#[test]
#[ignore = "timing run; needs cached checkpoints and a release build"]
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
                median_seconds(|| drop(model.detect(&audio, 16_000).unwrap())),
            );
        }
        if let Some(model) = loaded(DacCodec::load_from_hub(&hub, Fetch::CacheOnly, &device), "DAC") {
            let mut audio = signal(44_100, 5.0);
            audio.truncate(audio.len() / 512 * 512);
            report(
                "dac",
                5.0,
                &device,
                median_seconds(|| drop(model.decode(&model.encode(&audio, 3).unwrap()).unwrap())),
            );
        }
        if let Some(model) = loaded(SnacCodec::load_from_hub(&hub, Fetch::CacheOnly, &device), "SNAC") {
            let audio = signal(44_100, 5.0);
            report(
                "snac",
                5.0,
                &device,
                median_seconds(|| drop(model.reconstruct(&audio, 3).unwrap())),
            );
        }
        if let Some(model) = loaded(
            WavMarkDecoder::load_from_hub(&hub, Fetch::CacheOnly, &device),
            "WavMark",
        ) {
            let audio = signal(16_000, 10.0);
            report(
                "wavmark",
                10.0,
                &device,
                median_seconds(|| drop(model.read(&audio, 16_000).unwrap())),
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
                median_seconds(|| drop(model.read(&audio, 16_000).unwrap())),
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
                median_seconds(|| drop(model.scores(&audio).unwrap())),
            );
        }
    }
}

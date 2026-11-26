//! Release-mode timings of the hot paths; compare with `tools/bench/audio.py`.
//!
//! `cargo test --release -p raiw-audio --test bench -- --ignored --nocapture`

mod common;

use std::time::Instant;

use common::{encode_raw, synth};
use raiw_audio::codecs::raw::blend_hf_crossover;
use raiw_audio::dsp::resample_poly;
use raiw_audio::io::{FfmpegTools, probe_audio, read_all};
use raiw_audio::perceptual::compute_perceptual_metrics;
use raiw_audio::quality::measure_fidelity;
use raiw_audio::{Frames, RationalClock, spectral};
use serde_json::json;

fn spec(rate: u32, seconds: usize, seed: u64) -> serde_json::Value {
    json!({
        "rate": rate,
        "frames": rate as usize * seconds,
        "channels": [
            [{"kind": "noise", "seed": seed, "amp": 0.3}, {"kind": "tone", "freq": 1250.0, "amp": 0.1}],
            [{"kind": "noise", "seed": seed + 1, "amp": 0.3}, {"kind": "tone", "freq": 440.0, "amp": 0.1}],
        ],
    })
}

fn best<T>(repeat: usize, mut f: impl FnMut() -> T) -> f64 {
    (0..repeat)
        .map(|_| {
            let start = Instant::now();
            std::hint::black_box(f());
            start.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn resample_stereo(x: &Frames, up: usize, down: usize) -> Frames {
    let channels: Vec<Vec<f32>> = x
        .to_channels()
        .iter()
        .map(|c| resample_poly(c, up, down))
        .collect();
    Frames::from_channels(&channels)
}

#[test]
#[ignore = "timing run; use --release --ignored --nocapture"]
fn hot_path_timings() {
    let x48 = synth(&spec(48_000, 60, 1));
    let x44 = synth(&spec(44_100, 60, 3));
    let x16 = synth(&spec(16_000, 60, 5));
    let mut results = serde_json::Map::new();
    let mut record = |name: &str, seconds: f64| {
        results.insert(name.to_owned(), json!((seconds * 1e4).round() / 1e4));
    };
    record(
        "resample_48k_to_16k_60s_stereo",
        best(3, || resample_stereo(&x48, 1, 3)),
    );
    record(
        "resample_44k1_to_16k_60s_stereo",
        best(3, || resample_stereo(&x44, 160, 441)),
    );
    record(
        "resample_16k_to_44k1_60s_stereo",
        best(3, || resample_stereo(&x16, 441, 160)),
    );
    let clock = RationalClock::standard(x48.frames(), 48_000).unwrap();
    record(
        "clock_forward_inverse_48k_60s_stereo",
        best(3, || clock.inverse(&clock.forward(&x48).unwrap()).unwrap()),
    );
    let chunk = 48_000 * 8;
    record(
        "spectral_filter_48k_60s_stereo",
        best(3, || {
            let chunks = (0..x48.frames())
                .step_by(chunk)
                .map(|s| Ok(x48.slice(s..s + chunk)));
            spectral::filter_chunks(chunks, 48_000)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        }),
    );
    let x30 = x44.slice(0..44_100 * 30);
    let processed = x30.scale(0.9);
    record(
        "hf_crossover_44k1_30s_stereo",
        best(3, || {
            blend_hf_crossover(&processed, &x30, 44_100, 14_000.0).unwrap()
        }),
    );
    let tools = FfmpegTools::discover().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let (source, output) = (dir.path().join("a.wav"), dir.path().join("b.wav"));
    encode_raw(&x48, 48_000, &source, &["-c:a", "pcm_f32le"], None);
    encode_raw(&x48.scale(0.9), 48_000, &output, &["-c:a", "pcm_f32le"], None);
    let info = probe_audio(&tools, &source, 0).unwrap();
    record(
        "decode_48k_60s_stereo",
        best(3, || read_all(&tools, &source, &info).unwrap()),
    );
    record(
        "measure_fidelity_48k_60s_stereo",
        best(3, || {
            measure_fidelity(&tools, &source, &info, &output, 0).unwrap()
        }),
    );
    record(
        "perceptual_metrics_48k_60s_stereo",
        best(1, || {
            compute_perceptual_metrics(&tools, &source, &info, &output, 0, None).unwrap()
        }),
    );
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}

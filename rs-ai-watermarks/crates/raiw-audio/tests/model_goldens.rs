//! Real burn models (raiw-gpu) against `tools/goldens/audio_models.py`.
//!
//! `cargo test -p raiw-audio --features gpu --test model_goldens -- --test-threads=1`
#![cfg(feature = "gpu")]

mod common;

use std::time::Instant;

use common::{check_f32, golden};
use raiw_audio::decoders::DecoderSet;
use raiw_audio::io::read_all;
use raiw_audio::options::{Device, Method, Scheme};
use raiw_audio::{
    AudioRuntime, IdentifyOptions, Permissions, ProcessingArgs, RemovalRequest, identify_audio,
    remove_audio_invisible,
};
use raiw_core::testing::fixture;
use raiw_core::{JsonStyle, to_py_json};
use serde_json::Value;

/// Only one model family: AudioSeal decoders, or nothing.
fn runtime(audioseal: bool) -> AudioRuntime {
    let runtime = AudioRuntime::discover();
    let decoders = DecoderSet {
        audioseal: audioseal.then(|| runtime.decoders.audioseal.clone()).flatten(),
        ..DecoderSet::default()
    };
    runtime.with_decoders(decoders)
}

fn json<T: serde::Serialize>(value: &T, root: &str) -> Value {
    let repo = raiw_core::testing::repo_root().canonicalize().unwrap();
    let text = to_py_json(value, JsonStyle::COMPACT)
        .replace(&repo.display().to_string(), "<ROOT0>")
        .replace(root, "<ROOT0>");
    serde_json::from_str(&text).unwrap()
}

/// Same decisions everywhere; probabilities within `tolerance` (burn vs torch kernels).
fn compare_detections(actual: &Value, expected: &Value, tolerance: f64, context: &str) -> f64 {
    assert_eq!(
        actual["watermark_status"], expected["watermark_status"],
        "{context}: status"
    );
    let (a, e) = (
        actual["detections"].as_array().unwrap(),
        expected["detections"].as_array().unwrap(),
    );
    assert_eq!(a.len(), e.len(), "{context}: detection count");
    let mut worst = 0.0f64;
    for (da, de) in a.iter().zip(e) {
        for key in [
            "scheme",
            "channel",
            "status",
            "decoder_sample_rate",
            "inapplicable",
        ] {
            assert_eq!(da[key], de[key], "{context}: {key}");
        }
        let readings = |d: &Value| -> Vec<Value> {
            let mut all = vec![d["whole_clip"].clone()];
            for key in ["windows", "whole_segments"] {
                all.extend(d[key].as_array().unwrap().iter().map(|w| w["reading"].clone()));
            }
            all
        };
        for (ra, re) in readings(da).iter().zip(readings(de)) {
            for key in ["mean_detect_prob", "frac_above_threshold"] {
                let diff = (ra[key].as_f64().unwrap() - re[key].as_f64().unwrap()).abs();
                worst = worst.max(diff);
                assert!(diff <= tolerance, "{context}: {key} {ra} vs {re}");
            }
        }
    }
    worst
}

#[test]
fn audioseal_identify_on_a_fixture_matches_python_decisions() {
    let golden = golden("models_identify");
    let path = fixture(golden["fixture"].as_str().unwrap());
    let runtime = runtime(true);
    let options = IdentifyOptions {
        scheme: Scheme::AudioSeal,
        ..IdentifyOptions::default()
    };
    let start = Instant::now();
    let report = identify_audio(&runtime, &path, &options).unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    let worst = compare_detections(
        &json(&report, "/nonexistent"),
        &golden["report"],
        2e-3,
        "identify",
    );
    println!(
        "identify (AudioSeal, {}): {elapsed:.2} s, max probability drift {worst:.2e}",
        report.stream.sample_rate
    );
    runtime.release_models();
}

#[test]
fn spectral_removal_with_real_audioseal_matches_python() {
    let golden = golden("models_remove");
    let path = fixture(golden["fixture"].as_str().unwrap());
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let output = root.join("clean.wav");
    let runtime = runtime(true);
    let request = RemovalRequest {
        args: ProcessingArgs {
            scheme: Scheme::AudioSeal,
            method: Method::Spectral,
            ..ProcessingArgs::default()
        },
        permissions: Permissions {
            allow_unverified: true,
            ..Permissions::default()
        },
        low_memory: true,
        ..RemovalRequest::default()
    };
    let start = Instant::now();
    let result = remove_audio_invisible(&runtime, &path, &output, &request).unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    let actual = json(&result, &root.display().to_string());
    let expected = &golden["result"];
    assert_eq!(actual["verified_removed"], expected["verified_removed"]);
    assert_eq!(actual["quality_passed"], expected["quality_passed"]);
    compare_detections(&actual["before"], &expected["before"], 2e-3, "before");
    let worst = compare_detections(&actual["after"], &expected["after"], 2e-3, "after");
    raiw_core::testing::assert_json_close(&actual["fidelity"], &expected["fidelity"], 1e-9, "fidelity");
    let tools = runtime.tools.clone().unwrap();
    let info = raiw_audio::io::probe_audio(&tools, &output, 0).unwrap();
    let samples = read_all(&tools, &output, &info).unwrap();
    assert!(
        check_f32(samples.as_slice(), &golden["output"], 1e-12, "spectral output"),
        "spectral output bit-identical"
    );
    println!("remove (spectral + AudioSeal before/after): {elapsed:.2} s, max probability drift {worst:.2e}");
}

#[test]
fn dac_resynthesis_on_a_fixture_matches_python() {
    let golden = golden("models_dac");
    let path = fixture(golden["fixture"].as_str().unwrap());
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("dac.wav");
    let runtime = runtime(false);
    let device = std::env::var("RAIW_AUDIO_TEST_DEVICE").map_or(Device::Cpu, |d| d.parse().unwrap());
    let request = RemovalRequest {
        args: ProcessingArgs {
            method: Method::Dac,
            verify: false,
            vocoder_device: device,
            ..ProcessingArgs::default()
        },
        permissions: Permissions {
            force: true,
            ..Permissions::default()
        },
        low_memory: true,
        ..RemovalRequest::default()
    };
    let start = Instant::now();
    let result = remove_audio_invisible(&runtime, &path, &output, &request).unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    let expected = &golden["result"]["fidelity"];
    let actual = serde_json::to_value(result.fidelity.as_ref().unwrap()).unwrap();
    assert_eq!(actual["frames"], expected["frames"]);
    let snr_diff = (actual["snr_db"].as_f64().unwrap() - expected["snr_db"].as_f64().unwrap()).abs();
    let rms_diff =
        (actual["output_rms"].as_f64().unwrap() / expected["output_rms"].as_f64().unwrap() - 1.0).abs();
    println!(
        "dac ({device}): {elapsed:.2} s, SNR {} vs {} dB, output RMS ratio drift {rms_diff:.2e}",
        actual["snr_db"], expected["snr_db"]
    );
    assert!(snr_diff < 0.05, "SNR drift {snr_diff}");
    assert!(rms_diff < 1e-3, "output RMS drift {rms_diff}");
    assert_eq!(
        result.quality_passed,
        golden["result"]["quality_passed"].as_bool()
    );
}

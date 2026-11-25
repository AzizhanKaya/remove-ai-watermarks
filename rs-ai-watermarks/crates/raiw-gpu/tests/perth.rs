//! Perth decoder parity against `goldens/gpu_perth/decoder.json`.

mod common;

use common::{case_input, devices, fetch, loaded, strided_diff, tolerance};
use raiw_core::testing::golden;
use raiw_gpu::Hub;
use raiw_gpu::dsp::resample_poly;
use raiw_gpu::models::audio::perth::{PerthDetector, SAMPLE_RATE};

#[test]
fn resampler_matches_librosa_polyphase() {
    let golden = golden("gpu_perth", "decoder");
    for (name, case) in golden["cases"].as_object().unwrap() {
        let Some(samples) = case_input(&case["input"]) else {
            continue;
        };
        let native = resample_poly(&samples, 2, 1);
        let diff = strided_diff(&native, &case["resampled"]);
        assert!(diff <= 1e-5, "{name}: resampled diff {diff}");
    }
}

#[test]
fn decoder_matches_python_reference() {
    let golden = golden("gpu_perth", "decoder");
    for device in devices() {
        let Some(detector) = loaded(
            PerthDetector::load_from_hub(&Hub::from_env(), fetch(), &device),
            "Perth",
        ) else {
            return;
        };
        let tol = tolerance(&device, 1e-4, 2e-3);
        for (name, case) in golden["cases"].as_object().unwrap() {
            let Some(samples) = case_input(&case["input"]) else {
                eprintln!("skipping {name}: input unavailable");
                continue;
            };
            let reading = detector.read(&samples, SAMPLE_RATE).unwrap();
            let diff = (f64::from(reading.raw) - case["raw"].as_f64().unwrap()).abs();
            eprintln!("perth {name} on {device}: raw diff {diff:.2e}");
            assert!(diff <= tol, "{name} on {device}: raw diff {diff}");
            assert_eq!(
                reading.detected(),
                case["detected"].as_bool().unwrap(),
                "{name} on {device}"
            );
        }
    }
}

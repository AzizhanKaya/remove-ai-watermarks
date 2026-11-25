//! AudioSeal detector parity against `goldens/gpu_audioseal/detector.json`.

mod common;

use common::{as_f64s, case_input, devices, fetch, loaded, max_abs_diff, strided_diff, tolerance};
use raiw_core::testing::golden;
use raiw_gpu::Hub;
use raiw_gpu::models::audio::audioseal::{AudioSealDetector, SAMPLE_RATE};

#[test]
fn detector_matches_python_reference() {
    let golden = golden("gpu_audioseal", "detector");
    for device in devices() {
        let Some(detector) = loaded(
            AudioSealDetector::load_from_hub(&Hub::from_env(), fetch(), &device),
            "AudioSeal",
        ) else {
            return;
        };
        let tol = tolerance(&device, 1e-4, 2e-3);
        for (name, case) in golden["cases"].as_object().unwrap() {
            let Some(samples) = case_input(&case["input"]) else {
                eprintln!("skipping {name}: input unavailable");
                continue;
            };
            let reading = detector.detect(&samples, SAMPLE_RATE).unwrap();
            let prob_diff = strided_diff(&reading.probabilities, &case["probabilities"]);
            let message_diff = max_abs_diff(&reading.message, &as_f64s(&case["message"]));
            let mean_diff = (reading.mean_detect_prob() - case["mean_detect_prob"].as_f64().unwrap()).abs();
            let frac_diff =
                (reading.frac_above_threshold() - case["frac_above_threshold"].as_f64().unwrap()).abs();
            eprintln!(
                "audioseal {name} on {device}: probs {prob_diff:.2e} message {message_diff:.2e} mean {mean_diff:.2e}"
            );
            assert!(
                prob_diff <= tol,
                "{name} on {device}: probability diff {prob_diff}"
            );
            assert!(
                message_diff <= tol,
                "{name} on {device}: message diff {message_diff}"
            );
            assert!(mean_diff <= tol, "{name} on {device}: mean diff {mean_diff}");
            assert!(frac_diff <= 1e-3, "{name} on {device}: fraction diff {frac_diff}");
            assert_eq!(
                reading.label(),
                case["label"].as_str().unwrap(),
                "{name} on {device}"
            );
            assert_eq!(
                reading.detected(),
                case["detected"].as_bool().unwrap(),
                "{name} on {device}"
            );
        }
    }
}

#[test]
fn rejects_wrong_rate_and_nonfinite_input() {
    let device = raiw_gpu::ComputeDevice::cpu();
    let Some(detector) = loaded(
        AudioSealDetector::load_from_hub(&Hub::from_env(), fetch(), &device),
        "AudioSeal",
    ) else {
        return;
    };
    assert!(detector.detect(&[0.0; 160], 44_100).is_err());
    assert!(detector.detect(&[], SAMPLE_RATE).is_err());
    assert!(detector.detect(&[f32::NAN; 16], SAMPLE_RATE).is_err());
}

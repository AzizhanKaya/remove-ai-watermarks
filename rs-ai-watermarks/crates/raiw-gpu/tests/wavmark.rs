//! WavMark decoder parity against `goldens/gpu_wavmark/decoder.json`.

mod common;

use common::{as_f64s, case_input, devices, fetch, loaded, max_abs_diff, tolerance};
use raiw_core::testing::golden;
use raiw_gpu::Hub;
use raiw_gpu::models::audio::wavmark::{SAMPLE_RATE, WINDOW, WavMarkDecoder};

#[test]
fn decoder_matches_python_reference() {
    let golden = golden("gpu_wavmark", "decoder");
    for device in devices() {
        let Some(decoder) = loaded(
            WavMarkDecoder::load_from_hub(&Hub::from_env(), fetch(), &device),
            "WavMark",
        ) else {
            return;
        };
        let tol = tolerance(&device, 1e-4, 2e-3);
        for (name, case) in golden["cases"].as_object().unwrap() {
            let Some(samples) = case_input(&case["input"]) else {
                eprintln!("skipping {name}: input unavailable");
                continue;
            };
            for (start, expected) in case["raw"].as_object().unwrap() {
                let start: usize = start.parse().unwrap();
                let raw = decoder.decode(&[&samples[start..start + WINDOW]]).unwrap();
                let diff = max_abs_diff(&raw[0], &as_f64s(expected));
                eprintln!("wavmark {name}@{start} on {device}: raw diff {diff:.2e}");
                assert!(diff <= tol, "{name}@{start} on {device}: raw diff {diff}");
            }
            let reading = decoder.read(&samples, SAMPLE_RATE).unwrap();
            let expected = &case["reading"];
            let payload = reading
                .payload
                .map(|bits| bits.iter().map(|&b| u64::from(b)).collect::<Vec<_>>());
            let golden_payload = expected["payload"]
                .as_array()
                .map(|bits| bits.iter().map(|b| b.as_u64().unwrap()).collect());
            assert_eq!(payload, golden_payload, "{name} on {device}: payload");
            let starts: Vec<u64> = reading.start_samples.iter().map(|&s| s as u64).collect();
            let golden_starts: Vec<u64> = expected["start_samples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_u64().unwrap())
                .collect();
            assert_eq!(starts, golden_starts, "{name} on {device}: sync locations");
        }
    }
}

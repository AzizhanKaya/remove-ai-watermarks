//! DAC and SNAC parity against `goldens/gpu_codecs/{dac,snac,randn}.json`.

mod common;

use common::{
    as_codes, as_f64s, case_input, code_mismatch, devices, fetch, loaded, max_abs_diff, strided_diff,
    tolerance,
};
use raiw_core::testing::golden;
use raiw_gpu::models::audio::dac::{DacCodec, FIXED_QUANTIZERS};
use raiw_gpu::models::audio::encodec::EncodecCodec;
use raiw_gpu::models::audio::snac::SnacCodec;
use raiw_gpu::rng::torch_randn;
use raiw_gpu::{Fetch, Hub};

#[test]
fn torch_randn_stream_matches() {
    let golden = golden("gpu_codecs", "randn");
    for (n, values) in golden["draws"].as_object().unwrap() {
        let n: usize = n.parse().unwrap();
        let actual = torch_randn(n, 0).unwrap();
        let diff = max_abs_diff(&actual, &as_f64s(values));
        assert!(diff <= 1e-5, "randn({n}) diff {diff}");
    }
}

#[test]
fn dac_matches_python_reference() {
    let golden = golden("gpu_codecs", "dac");
    for device in devices() {
        let Some(codec) = loaded(
            DacCodec::load_from_hub(&Hub::from_env(), Fetch::CacheOnly, &device),
            "DAC",
        ) else {
            return;
        };
        for (name, case) in golden["cases"].as_object().unwrap() {
            let Some(samples) = case_input(&case["input"]) else {
                eprintln!("skipping {name}: input unavailable");
                continue;
            };
            let codes = codec.encode(&samples, FIXED_QUANTIZERS).unwrap();
            let mismatch = code_mismatch(&codes.codebooks, &as_codes(&case["codes"]));
            let decoded = codec.decode(&codes).unwrap();
            let reconstructed = codec.reconstruct(&samples).unwrap();
            let diff = strided_diff(&decoded, &case["output"]);
            let self_diff = decoded
                .iter()
                .zip(&reconstructed)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("dac {name} on {device}: codes mismatch {mismatch:.3} output diff {diff:.2e}");
            assert!(
                mismatch <= tolerance(&device, 0.0, 0.02),
                "{name} on {device}: code mismatch {mismatch}"
            );
            if mismatch == 0.0 {
                assert!(
                    diff <= tolerance(&device, 1e-4, 2e-3),
                    "{name} on {device}: output diff {diff}"
                );
            }
            assert!(
                self_diff <= 1e-5,
                "{name} on {device}: decode(encode) vs reconstruct {self_diff}"
            );
        }
    }
}

#[test]
fn encodec_matches_python_reference() {
    let golden = golden("gpu_codecs", "encodec");
    for device in devices() {
        let Some(codec) = loaded(
            EncodecCodec::load_from_hub(&Hub::from_env(), fetch(), &device),
            "EnCodec",
        ) else {
            return;
        };
        for (name, case) in golden["cases"].as_object().unwrap() {
            let Some(samples) = case_input(&case["input"]) else {
                eprintln!("skipping {name}: input unavailable");
                continue;
            };
            let bandwidth = case["bandwidth"].as_f64().unwrap();
            let codes = codec.encode(&samples, bandwidth).unwrap();
            let mismatch = code_mismatch(&codes, &as_codes(&case["codes"]));
            let decoded = codec.reconstruct(&samples, bandwidth).unwrap();
            let diff = strided_diff(&decoded, &case["output"]);
            eprintln!("encodec {name} on {device}: codes mismatch {mismatch:.3} output diff {diff:.2e}");
            assert!(
                mismatch <= tolerance(&device, 0.0, 0.02),
                "{name} on {device}: code mismatch {mismatch}"
            );
            if mismatch == 0.0 {
                assert!(
                    diff <= tolerance(&device, 1e-4, 2e-3),
                    "{name} on {device}: output diff {diff}"
                );
            }
        }
    }
}

#[test]
fn snac_matches_python_reference() {
    let golden = golden("gpu_codecs", "snac");
    for device in devices() {
        let Some(codec) = loaded(
            SnacCodec::load_from_hub(&Hub::from_env(), Fetch::CacheOnly, &device),
            "SNAC",
        ) else {
            return;
        };
        for (name, case) in golden["cases"].as_object().unwrap() {
            let Some(samples) = case_input(&case["input"]) else {
                eprintln!("skipping {name}: input unavailable");
                continue;
            };
            let codes = codec.encode(&samples, 3).unwrap();
            let mismatch = code_mismatch(&codes, &as_codes(&case["codes"]));
            let decoded = codec.reconstruct(&samples, 3).unwrap();
            let diff = strided_diff(&decoded, &case["output"]);
            eprintln!("snac {name} on {device}: codes mismatch {mismatch:.3} output diff {diff:.2e}");
            assert!(
                mismatch <= tolerance(&device, 0.0, 0.02),
                "{name} on {device}: code mismatch {mismatch}"
            );
            if mismatch == 0.0 {
                assert!(
                    diff <= tolerance(&device, 1e-4, 2e-3),
                    "{name} on {device}: output diff {diff}"
                );
            }
        }
    }
}

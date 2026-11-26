//! SilentCipher decoder parity against `goldens/gpu_silentcipher/decoder.json`.

mod common;

use common::{case_input, devices, fetch, loaded, tolerance};
use raiw_core::testing::golden;
use raiw_gpu::Hub;
use raiw_gpu::models::audio::silentcipher::{
    SAMPLE_RATE, SilentCipherDecoder, decode_sequence, normalize_power,
};

#[test]
fn decoder_matches_python_reference() {
    let golden = golden("gpu_silentcipher", "decoder");
    for device in devices() {
        let Some(decoder) = loaded(
            SilentCipherDecoder::load_from_hub(&Hub::from_env(), fetch(), &device),
            "SilentCipher",
        ) else {
            return;
        };
        let tol = tolerance(&device, 1e-4, 2e-3);
        for (name, case) in golden["cases"].as_object().unwrap() {
            let Some(samples) = case_input(&case["input"]) else {
                eprintln!("skipping {name}: input unavailable");
                continue;
            };
            let normalized = normalize_power(&samples).unwrap();
            for (shift, expected) in case["shifts"].as_object().unwrap() {
                let shift: usize = shift.parse().unwrap();
                let scores = decoder.scores(&normalized[shift..]).unwrap();
                assert_eq!(
                    scores.frames as u64,
                    expected["frames"].as_u64().unwrap(),
                    "{name}@{shift}: frames"
                );
                let mut worst = 0.0f64;
                let mut at = (0, 0, 0.0, 0.0);
                for (symbol, row) in expected["scores"].as_array().unwrap().iter().enumerate() {
                    for (t, (actual, golden)) in
                        scores.row(symbol).iter().zip(row.as_array().unwrap()).enumerate()
                    {
                        let golden = golden.as_f64().unwrap();
                        let diff = (f64::from(*actual) - golden).abs() / golden.abs().max(1.0);
                        if diff > worst {
                            worst = diff;
                            at = (symbol, t, f64::from(*actual), golden);
                        }
                    }
                }
                if worst > tol {
                    eprintln!("worst at symbol {} frame {}: {} vs {}", at.0, at.1, at.2, at.3);
                }
                let (payload, consistency) = decode_sequence(&scores).unwrap();
                eprintln!(
                    "silentcipher {name}@{shift} on {device}: score diff {worst:.2e} consistency {consistency:.3}"
                );
                assert!(worst <= tol, "{name}@{shift} on {device}: score diff {worst}");
                assert!(
                    payload.is_none() == expected["payload"].is_null(),
                    "{name}@{shift}: payload presence"
                );
                let golden_consistency = expected["consistency"].as_f64().unwrap();
                assert!(
                    (consistency - golden_consistency).abs() <= 0.02,
                    "{name}@{shift}: consistency {consistency}"
                );
            }
            // The full 205-phase receiver sweep is minutes of compute; opt in with RAIW_SLOW
            if let Some(expected) = case
                .get("reading")
                .filter(|_| std::env::var_os("RAIW_SLOW").is_some())
            {
                let reading = decoder.read(&samples, SAMPLE_RATE).unwrap();
                assert_eq!(
                    reading.payload.is_none(),
                    expected["payload"].is_null(),
                    "{name}: receiver payload"
                );
                assert_eq!(
                    reading.phases_checked as u64,
                    expected["phases_checked"].as_u64().unwrap(),
                    "{name}: phases"
                );
                let best = expected["symbol_consistency"].as_f64().unwrap();
                assert!(
                    (reading.symbol_consistency - best).abs() <= 0.02,
                    "{name}: consistency"
                );
            }
        }
    }
}

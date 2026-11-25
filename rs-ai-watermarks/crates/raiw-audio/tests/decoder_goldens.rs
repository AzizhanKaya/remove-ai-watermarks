//! Receiver logic against `tools/goldens/audio_decoders.py` (real Python readers on raw outputs).

mod common;

use common::{golden, splitmix_uniform};
use raiw_audio::decoders::silentcipher::{SymbolScores, decode_sequence};
use raiw_audio::decoders::wavmark::SYNC_PATTERN;
use raiw_audio::decoders::{
    AudioSealNetwork, AudioSealOutput, DecoderError, PerthNetwork, WavMarkNetwork, audioseal, perth, wavmark,
};
use serde_json::{Value, json};

fn uniform32(seed: u64, count: usize, lo: f64, hi: f64) -> Vec<f32> {
    splitmix_uniform(seed, count).into_iter().map(|u| (lo + (hi - lo) * u) as f32).collect()
}

fn spec_values(spec: &Value) -> Vec<f32> {
    uniform32(
        spec["seed"].as_u64().unwrap(),
        spec["count"].as_u64().unwrap() as usize,
        spec["lo"].as_f64().unwrap(),
        spec["hi"].as_f64().unwrap(),
    )
}

fn outcome<T: serde::Serialize>(result: Result<T, DecoderError>) -> Value {
    match result {
        Ok(reading) => json!({ "ok": reading }),
        Err(DecoderError::Invalid(message)) => json!({ "error": { "type": "ValueError", "message": message } }),
        Err(DecoderError::Failed(message) | DecoderError::ModelCheck(message)) => {
            json!({ "error": { "type": "RuntimeError", "message": message } })
        }
    }
}

struct RawAudioSeal(AudioSealOutput);

impl AudioSealNetwork for RawAudioSeal {
    fn detect(&self, _samples: &[f32]) -> Result<AudioSealOutput, DecoderError> {
        Ok(self.0.clone())
    }
}

#[test]
fn audioseal_reader_matches_python_exactly() {
    let golden = golden("decoders");
    for case in golden["audioseal"].as_array().unwrap() {
        let net = RawAudioSeal(AudioSealOutput {
            probabilities: spec_values(&case["probs"]),
            message: spec_values(&case["message"]),
        });
        assert_eq!(outcome(audioseal::read(&net, &[0.0; 10])), case["result"], "{}", case["probs"]);
    }
}

struct RawWavMark {
    every: usize,
}

impl WavMarkNetwork for RawWavMark {
    fn decode(&self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>, DecoderError> {
        Ok(windows
            .iter()
            .map(|w| {
                let start = w[0] as usize;
                let mut raw = uniform32(start as u64 + 7, 32, 0.0, 1.0);
                if self.every > 0 && start % self.every == 0 {
                    for (slot, bit) in raw.iter_mut().zip(SYNC_PATTERN) {
                        *slot = f32::from(bit);
                    }
                }
                raw
            })
            .collect())
    }
}

struct Signature {
    flips: [usize; 2],
}

impl WavMarkNetwork for Signature {
    fn decode(&self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>, DecoderError> {
        let reference: Vec<f32> =
            b"default ".iter().flat_map(|byte| (0..8).rev().map(move |i| f32::from(byte >> i & 1))).collect();
        Ok(windows
            .iter()
            .enumerate()
            .map(|(index, w)| {
                let bits = &reference[index * 32..(index + 1) * 32];
                let noise = uniform32(w[0] as u64 + 11, 32, 0.0, 0.4);
                let mut raw: Vec<f32> = bits.iter().zip(&noise).map(|(b, n)| b * 0.6 + n).collect();
                for k in 0..self.flips[index] {
                    raw[(k * 7) % 32] = 1.0 - bits[(k * 7) % 32];
                }
                raw
            })
            .collect())
    }
}

#[test]
fn wavmark_receivers_match_python_exactly() {
    let golden = golden("decoders");
    assert_eq!(json!(SYNC_PATTERN), golden["pattern"]);
    for case in golden["wavmark"].as_array().unwrap() {
        let length = case["length"].as_u64().unwrap() as usize;
        let samples: Vec<f32> = (0..length).map(|v| v as f32).collect();
        let net = RawWavMark { every: case["every"].as_u64().unwrap() as usize };
        assert_eq!(outcome(wavmark::read(&net, &samples)), case["result"], "wavmark {length}");
    }
    for case in golden["openvoice"].as_array().unwrap() {
        let length = case["length"].as_u64().unwrap() as usize;
        let flips: Vec<usize> = case["flips"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let samples: Vec<f32> = (0..length).map(|v| v as f32).collect();
        let net = Signature { flips: [flips[0], flips[1]] };
        let actual = match wavmark::read_openvoice_default(&net, &samples) {
            Ok((whole, fragments)) => json!({ "whole": whole, "fragments": fragments }),
            Err(err) => json!({ "error": { "type": "ValueError", "message": err.to_string() } }),
        };
        assert_eq!(actual, case["result"], "openvoice {length} {flips:?}");
    }
}

struct RawPerth(Vec<f32>);

impl PerthNetwork for RawPerth {
    fn presence(&self, _samples: &[f32]) -> Result<Vec<f32>, DecoderError> {
        Ok(self.0.clone())
    }
}

#[test]
fn perth_reader_matches_python() {
    let golden = golden("decoders");
    for case in golden["perth"].as_array().unwrap() {
        let raw: Vec<f32> =
            case["raw"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().parse::<f64>().unwrap() as f32).collect();
        let len = if case.get("short").is_some() { 15_999 } else { 16_000 };
        assert_eq!(outcome(perth::read(&RawPerth(raw), &vec![0.0; len])), case["result"], "{}", case["raw"]);
    }
}

fn scores(kind: &str, seed: u64, frames: usize) -> Vec<f32> {
    let mut out = uniform32(seed, 5 * frames, 0.0, 0.5);
    let symbols: Vec<usize> = match kind {
        "valid" => std::iter::once(0).chain((0..20).map(|i| 1 + (i * 3) % 4)).collect(),
        "two_zeros" => [0, 0].into_iter().chain((0..19).map(|i| 1 + i % 4)).collect(),
        "no_zero" => (0..21).map(|i| 1 + i % 4).collect(),
        _ => splitmix_uniform(seed + 1, 21).into_iter().map(|v| (v * 5.0) as usize % 5).collect(),
    };
    for t in 0..frames {
        out[symbols[t % 21] * frames + t] += 1.0;
    }
    for k in 0..(seed % 5) as usize {
        let t = (k * 13 + 5) % frames;
        out[((symbols[t % 21] + 1) % 5) * frames + t] += 2.0;
    }
    out
}

#[test]
fn silentcipher_sequence_decoding_matches_python() {
    let golden = golden("decoders");
    for case in golden["silentcipher"].as_array().unwrap() {
        let (kind, seed, frames) = (
            case["kind"].as_str().unwrap(),
            case["seed"].as_u64().unwrap(),
            case["frames"].as_u64().unwrap() as usize,
        );
        let values = scores(kind, seed, frames);
        let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
        assert!((sum - case["scores_sum"].as_f64().unwrap()).abs() < 1e-6, "{kind} {seed}: input");
        let actual = match decode_sequence(&SymbolScores { rows: 5, frames, values }) {
            Ok((payload, consistency)) => json!({ "payload": payload, "consistency": consistency }),
            Err(err) => json!({ "error": err.to_string() }),
        };
        assert_eq!(actual, case["result"], "silentcipher {kind} {seed} {frames}");
    }
}

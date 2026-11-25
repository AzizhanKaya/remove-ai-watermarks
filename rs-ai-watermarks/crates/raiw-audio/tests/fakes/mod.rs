//! Deterministic fake networks mirroring `tools/goldens/audio_pipeline.py`.
#![allow(dead_code)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use raiw_audio::codecs::{CodecError, CodecKind, CodecRequest, CodecSet, NeuralCodec};
use raiw_audio::decoders::silentcipher::SymbolScores;
use raiw_audio::decoders::wavmark::SYNC_PATTERN;
use raiw_audio::decoders::{
    AudioSealNetwork, AudioSealOutput, DecoderError, DecoderSet, PerthNetwork, SilentCipherNetwork,
    WavMarkNetwork,
};
use raiw_audio::dsp::resample_rates;
use raiw_audio::metadata::MetadataSeam;
use raiw_audio::report::Markers;
use serde_json::Value;

/// `abs(2 * mean(x * exp(-2j pi f t)))` with `t = n / rate`.
pub fn band_amplitude(samples: &[f32], rate: f64, freq: f64) -> f64 {
    let w = -2.0 * PI * freq;
    let (mut re, mut im) = (0.0, 0.0);
    for (n, &x) in samples.iter().enumerate() {
        let theta = w * (n as f64 / rate);
        re += f64::from(x) * theta.cos();
        im += f64::from(x) * theta.sin();
    }
    let n = samples.len() as f64;
    2.0 * (re / n).hypot(im / n)
}

fn mean_square_f32(samples: &[f32]) -> f32 {
    samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32
}

pub struct FakeAudioSeal(pub String);

impl AudioSealNetwork for FakeAudioSeal {
    fn detect(&self, samples: &[f32]) -> Result<AudioSealOutput, DecoderError> {
        let found = match self.0.as_str() {
            "energy" => mean_square_f32(samples) > 0.001,
            "positive" | "resistant" => true,
            "negative" => false,
            _ => band_amplitude(samples, 16_000.0, 1250.0) > 0.01,
        };
        if self.0 == "crash" && !found {
            return Err(DecoderError::Failed("AudioSeal post-verify crashed".into()));
        }
        let prob = if found { 0.9f32 } else { 0.1f32 };
        Ok(AudioSealOutput {
            probabilities: vec![prob; samples.len()],
            message: (0..16).map(|i| (i % 2) as f32).collect(),
        })
    }
}

pub struct FakeWavMark;

impl WavMarkNetwork for FakeWavMark {
    fn decode(&self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>, DecoderError> {
        Ok(windows
            .iter()
            .map(|w| {
                if band_amplitude(w, 16_000.0, 1250.0) > 0.01 {
                    SYNC_PATTERN
                        .iter()
                        .map(|&b| f32::from(b))
                        .chain((0..16).map(|i| ((i + 1) % 2) as f32))
                        .collect()
                } else {
                    vec![0.25; 32]
                }
            })
            .collect())
    }
}

pub struct FakePerth;

impl PerthNetwork for FakePerth {
    fn presence(&self, samples: &[f32]) -> Result<Vec<f32>, DecoderError> {
        Ok(vec![if band_amplitude(samples, 16_000.0, 1250.0) > 0.01 {
            0.8
        } else {
            0.2
        }])
    }
}

pub struct FakeSilentCipher;

impl SilentCipherNetwork for FakeSilentCipher {
    fn device(&self) -> &'static str {
        "cpu"
    }

    fn resample(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, DecoderError> {
        Ok(resample_rates(samples, sample_rate, 44_100))
    }

    fn scores(&self, normalized: &[f32], shift: usize) -> Result<SymbolScores, DecoderError> {
        let carrier = &normalized[shift.min(normalized.len())..];
        let found = band_amplitude(carrier, 44_100.0, 1250.0) > 0.005;
        let symbols: Vec<usize> = if found {
            std::iter::once(0).chain((0..20).map(|i| 1 + i % 4)).collect()
        } else {
            vec![0; 21]
        };
        let mut values = vec![0.0f32; 5 * 63];
        for t in 0..63 {
            values[symbols[t % 21] * 63 + t] = 1.0;
        }
        Ok(SymbolScores {
            rows: 5,
            frames: 63,
            values,
        })
    }
}

/// `codec_fn(mode)`: `comb`, `zero` or a gain.
pub struct FakeCodec {
    pub kind: CodecKind,
    pub mode: String,
}

impl NeuralCodec for FakeCodec {
    fn kind(&self) -> CodecKind {
        self.kind
    }

    fn reconstruct(&self, stem: &[f32], _request: &CodecRequest) -> Result<Vec<f32>, CodecError> {
        Ok(match self.mode.as_str() {
            "comb" => (0..stem.len())
                .map(|n| {
                    let shifted = if stem.len() > 18 && n >= 18 {
                        stem[n - 18]
                    } else {
                        0.0
                    };
                    0.5 * (stem[n] + shifted)
                })
                .collect(),
            "zero" => stem.iter().map(|v| v * 0.0).collect(),
            gain => {
                let gain: f32 = gain.parse::<f64>().map_err(|e| CodecError(e.to_string()))? as f32;
                stem.iter().map(|v| v * gain).collect()
            }
        })
    }
}

pub fn decoders(fakes: &Value) -> DecoderSet {
    let mode = |key: &str| fakes.get(key).and_then(Value::as_str).map(str::to_owned);
    DecoderSet {
        audioseal: mode("audioseal").map(|m| Arc::new(FakeAudioSeal(m)) as Arc<dyn AudioSealNetwork>),
        wavmark: mode("wavmark").map(|_| Arc::new(FakeWavMark) as Arc<dyn WavMarkNetwork>),
        perth: mode("perth").map(|_| Arc::new(FakePerth) as Arc<dyn PerthNetwork>),
        silentcipher: mode("silentcipher")
            .map(|_| Arc::new(FakeSilentCipher) as Arc<dyn SilentCipherNetwork>),
    }
}

pub fn codecs(fakes: &Value) -> CodecSet {
    let codec = |key: &str, kind| {
        fakes.get(key).and_then(Value::as_str).map(|mode| {
            Arc::new(FakeCodec {
                kind,
                mode: mode.to_owned(),
            }) as Arc<dyn NeuralCodec>
        })
    };
    CodecSet {
        encodec: codec("encodec", CodecKind::Encodec),
        dac: codec("dac", CodecKind::Dac),
        snac: codec("snac", CodecKind::Snac),
    }
}

/// Markers only on the scenario source; `markers_claim_ai` as Python recorded it.
#[derive(Debug, Clone)]
pub struct FakeMetadata {
    pub source: PathBuf,
    pub markers: Markers,
    pub claims_ai: bool,
}

impl MetadataSeam for FakeMetadata {
    fn ai_metadata(&self, path: &Path) -> raiw_audio::Result<Markers> {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        Ok(if canonical == self.source {
            self.markers.clone()
        } else {
            Markers::new()
        })
    }

    fn unreadable_reason(&self, _path: &Path) -> Option<String> {
        None
    }

    fn markers_claim_ai(&self, markers: &Markers) -> bool {
        !markers.is_empty() && *markers == self.markers && self.claims_ai
    }

    fn strip_and_verify(
        &self,
        source: &Path,
        output: &Path,
        _keep_standard: bool,
    ) -> raiw_audio::Result<(PathBuf, Markers)> {
        std::fs::copy(source, output)?;
        Ok((output.to_path_buf(), Markers::new()))
    }
}

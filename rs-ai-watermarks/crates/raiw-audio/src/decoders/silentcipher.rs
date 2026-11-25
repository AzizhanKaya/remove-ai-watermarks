//! Experimental pinned SilentCipher-44k receiver with exhaustive phase checks (`_internal/silentcipher.py`).
//!
//! The profile needs three complete message repetitions and at least 0.9 symbol agreement; these
//! are project receiver criteria, not SDK defaults.

use std::sync::Arc;

use serde::Serialize;

use super::DecoderError;

pub const MODEL_REPOSITORY: &str = "sony/silentcipher";
pub const MODEL_REVISION: &str = "a1c4d021905e0dc5b24be5f68db5fc4dba410ee1";
pub const SAMPLE_RATE: u32 = 44_100;
pub const WINDOW_SECONDS: usize = 3;
pub const MIN_READ_SAMPLES: usize = SAMPLE_RATE as usize * WINDOW_SECONDS;
pub const PERIOD: usize = 21;
pub const MIN_CYCLES: usize = 3;
pub const THRESHOLD: f64 = 0.9;
pub const HOP_LENGTH: usize = 2048;
pub const PHASE_STEP: usize = 10;
const TARGET_POWER: f64 = 0.002_837_200_844_477_648;

/// Raw `(5, frames)` symbol scores, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolScores {
    pub rows: usize,
    pub frames: usize,
    pub values: Vec<f32>,
}

/// The released SilentCipher decoder and its STFT, plus librosa's default resampler to 44.1 kHz.
pub trait SilentCipherNetwork: Send + Sync {
    /// The device label reported in readings (`cpu`, `cuda`, ...).
    fn device(&self) -> &str;
    /// `librosa.resample(samples, orig_sr=sample_rate, target_sr=44100)`.
    fn resample(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>, DecoderError>;
    /// `decoder(magnitude(signal[shift:], window)[:, None])[0, 0]` on power-normalized input.
    fn scores(&self, normalized: &[f32], shift: usize) -> Result<SymbolScores, DecoderError>;
    /// A CPU instance for retrying device (not numerical) failures.
    fn cpu_fallback(&self) -> Option<Arc<dyn SilentCipherNetwork>> {
        None
    }
}

/// Accepted payload or a complete negative under this receiver profile.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SilentCipherReading {
    pub payload: Option<Vec<u8>>,
    pub symbol_consistency: f64,
    pub phase_offset: Option<usize>,
    pub phases_checked: usize,
    pub device: String,
}

impl SilentCipherReading {
    pub const fn detected(&self) -> bool {
        self.payload.is_some()
    }
}

/// `_decode_sequence`: framing-terminated five-byte payload and the symbol agreement.
pub fn decode_sequence(scores: &SymbolScores) -> Result<(Option<Vec<u8>>, f64), DecoderError> {
    if scores.rows != 5
        || scores.values.len() != 5 * scores.frames
        || !scores.values.iter().all(|v| v.is_finite())
    {
        return Err(DecoderError::ModelCheck(
            "SilentCipher returned invalid raw neural values".into(),
        ));
    }
    let length = scores.frames / PERIOD * PERIOD;
    if length < PERIOD * MIN_CYCLES {
        return Err(DecoderError::ModelCheck(
            "SilentCipher returned too few complete message repetitions".into(),
        ));
    }
    let predicted: Vec<usize> = (0..length)
        .map(|t| {
            let mut best = 0;
            for row in 1..5 {
                if scores.values[row * scores.frames + t] > scores.values[best * scores.frames + t] {
                    best = row;
                }
            }
            best
        })
        .collect();
    let cycles = length / PERIOD;
    let symbols: Vec<usize> = (0..PERIOD)
        .map(|position| {
            let mut counts = [0usize; 5];
            for cycle in 0..cycles {
                counts[predicted[cycle * PERIOD + position]] += 1;
            }
            (0..5).fold(0, |best, s| if counts[s] > counts[best] { s } else { best })
        })
        .collect();
    let agree = predicted
        .iter()
        .enumerate()
        .filter(|(i, p)| **p == symbols[i % PERIOD])
        .count();
    let consistency = agree as f64 / length as f64;
    let zeros: Vec<usize> = (0..PERIOD).filter(|&i| symbols[i] == 0).collect();
    let [end] = zeros.as_slice() else {
        return Ok((None, consistency));
    };
    let letters: Vec<usize> = symbols[end + 1..]
        .iter()
        .chain(&symbols[..*end])
        .map(|s| s - 1)
        .collect();
    if letters.len() != 20 || letters.iter().any(|&l| l > 3) {
        return Ok((None, consistency));
    }
    let payload = letters
        .chunks(4)
        .map(|four| four.iter().fold(0u8, |acc, &l| (acc << 2) | l as u8))
        .collect();
    Ok((Some(payload), consistency))
}

fn read_native(
    network: &dyn SilentCipherNetwork,
    samples: &[f32],
) -> Result<SilentCipherReading, DecoderError> {
    let squares: Vec<f32> = samples.iter().map(|v| v * v).collect();
    let power = super::mean_f32(&squares) as f32;
    if !power.is_finite() || power <= 0.0 {
        return Err(DecoderError::ModelCheck(
            "SilentCipher cannot normalize silent or invalid input power".into(),
        ));
    }
    let gain = (TARGET_POWER as f32 / power).sqrt();
    let values: Vec<f32> = samples.iter().map(|v| v * gain).collect();
    let mut best = 0.0f64;
    let mut checked = 0;
    for shift in (0..HOP_LENGTH).step_by(PHASE_STEP) {
        let scores = network.scores(&values, shift)?;
        let (payload, consistency) = decode_sequence(&scores)?;
        checked += 1;
        best = best.max(consistency);
        if let Some(payload) = payload
            && consistency >= THRESHOLD
        {
            return Ok(SilentCipherReading {
                payload: Some(payload),
                symbol_consistency: consistency,
                phase_offset: Some(shift),
                phases_checked: checked,
                device: network.device().to_owned(),
            });
        }
    }
    Ok(SilentCipherReading {
        payload: None,
        symbol_consistency: best,
        phase_offset: None,
        phases_checked: checked,
        device: network.device().to_owned(),
    })
}

/// `silentcipher.read(model, samples, sample_rate=...)`.
pub fn read(
    network: &dyn SilentCipherNetwork,
    samples: &[f32],
    sample_rate: u32,
) -> Result<SilentCipherReading, DecoderError> {
    if sample_rate == 0
        || samples.len() < WINDOW_SECONDS * sample_rate as usize
        || !samples.iter().all(|v| v.is_finite())
    {
        return Err(DecoderError::Invalid(
            "SilentCipher requires at least three seconds of finite mono audio".into(),
        ));
    }
    let resampled;
    let native = if sample_rate == SAMPLE_RATE {
        samples
    } else {
        resampled = network.resample(samples, sample_rate)?;
        &resampled
    };
    if native.len() < MIN_READ_SAMPLES || !native.iter().all(|v| v.is_finite()) {
        return Err(DecoderError::Failed(
            "SilentCipher resampling returned invalid samples".into(),
        ));
    }
    match read_native(network, native) {
        Err(DecoderError::Failed(message)) if network.device() != "cpu" => match network.cpu_fallback() {
            Some(cpu) => read_native(cpu.as_ref(), native),
            None => Err(DecoderError::Failed(message)),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores_for(symbols: &[usize], cycles: usize) -> SymbolScores {
        let frames = symbols.len() * cycles;
        let mut values = vec![0.0; 5 * frames];
        for t in 0..frames {
            values[symbols[t % symbols.len()] * frames + t] = 1.0;
        }
        SymbolScores {
            rows: 5,
            frames,
            values,
        }
    }

    #[test]
    fn terminated_payload_decodes_five_bytes() {
        let mut symbols = vec![0usize];
        symbols.extend((0..20).map(|i| 1 + i % 4));
        let (payload, consistency) = decode_sequence(&scores_for(&symbols, 3)).unwrap();
        assert_eq!(consistency, 1.0);
        assert_eq!(payload, Some(vec![0b0001_1011; 5]));
    }

    #[test]
    fn short_sequences_are_model_check_errors() {
        let symbols = vec![0usize; 21];
        assert!(matches!(
            decode_sequence(&scores_for(&symbols, 2)),
            Err(DecoderError::ModelCheck(_))
        ));
    }
}

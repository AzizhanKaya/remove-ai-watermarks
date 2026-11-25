//! Hash-pinned public WavMark receiver (`_internal/wavmark.py`), plus OpenVoice default framing.

use std::fmt::Write;

use serde::Serialize;

use super::DecoderError;

pub const MODEL_REPOSITORY: &str = "M4869/WavMark";
pub const MODEL_REVISION: &str = "0ad3c7b74f641bddb61f6b85cdf2de0d93a5bfef";
pub const MODEL_SHA256: &str = "7a3d873d115c2c644685fb4cf109bdb299718296b88c871d3cf1c8e4cbc13144";
pub const SAMPLE_RATE: u32 = 16_000;
pub const MIN_READ_SAMPLES: usize = 16_000;
pub const OPENVOICE_START_SAMPLES: [usize; 2] = [0, 32_000];
pub const OPENVOICE_PREFIX_SAMPLES: usize = 48_000;
pub const OPENVOICE_REFERENCE: &[u8; 8] = b"default ";
pub const SIGNATURE_TAIL_THRESHOLD: f64 = 1e-6;
/// `wavmark.wm_add_util.fix_pattern[:16]` of the pinned package.
pub const SYNC_PATTERN: [u8; 16] = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0];
const STEP: usize = SAMPLE_RATE as usize / 20;
const BATCH: usize = 64;

/// The WavMark decoder forward pass: each 16000-sample window yields 32 raw message values.
pub trait WavMarkNetwork: Send + Sync {
    fn decode(&self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>, DecoderError>;
}

/// A payload accepted by the upstream exact synchronization-pattern check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WavMarkReading {
    pub payload: Option<Vec<u8>>,
    pub start_samples: Vec<usize>,
}

impl WavMarkReading {
    pub const fn detected(&self) -> bool {
        self.payload.is_some()
    }
}

/// Known-reference correlation; binomial tails are not population false-positive rates.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WavMarkSignatureReading {
    pub start_samples: Vec<usize>,
    pub decoded_bits: Vec<u8>,
    pub reference_hex: String,
    pub matching_bits: usize,
    pub adjusted_binomial_tail: f64,
    pub framing: String,
}

impl WavMarkSignatureReading {
    /// The fixed, multiple-comparison-corrected signature criterion.
    pub fn detected(&self) -> bool {
        self.adjusted_binomial_tail < SIGNATURE_TAIL_THRESHOLD
    }
}

fn decode_checked(
    network: &dyn WavMarkNetwork,
    windows: &[&[f32]],
    message: &str,
) -> Result<Vec<Vec<bool>>, DecoderError> {
    let raw = network.decode(windows)?;
    if raw.len() != windows.len()
        || raw
            .iter()
            .any(|row| row.len() != 32 || !row.iter().all(|v| v.is_finite()))
    {
        return Err(DecoderError::Failed(message.into()));
    }
    Ok(raw
        .into_iter()
        .map(|row| row.into_iter().map(|v| v >= 0.5).collect())
        .collect())
}

/// `wavmark.read`: 50 ms stride over every complete one-second window, including the final one.
pub fn read(network: &dyn WavMarkNetwork, samples: &[f32]) -> Result<WavMarkReading, DecoderError> {
    if samples.len() < MIN_READ_SAMPLES || !samples.iter().all(|v| v.is_finite()) {
        return Err(DecoderError::Invalid(format!(
            "WavMark requires at least {MIN_READ_SAMPLES} finite mono samples at 16 kHz"
        )));
    }
    let end = samples.len() - SAMPLE_RATE as usize;
    let mut starts: Vec<usize> = (0..=end).step_by(STEP).collect();
    if starts.last() != Some(&end) {
        starts.push(end);
    }
    let mut locations = Vec::new();
    let mut messages: Vec<Vec<bool>> = Vec::new();
    for points in starts.chunks(BATCH) {
        let windows: Vec<&[f32]> = points
            .iter()
            .map(|&s| &samples[s..s + SAMPLE_RATE as usize])
            .collect();
        let bits = decode_checked(network, &windows, "WavMark returned invalid raw message values")?;
        for (&start, message) in points.iter().zip(bits) {
            if message[..16]
                .iter()
                .zip(SYNC_PATTERN)
                .all(|(&b, p)| u8::from(b) == p)
            {
                locations.push(start);
                messages.push(message[16..].to_vec());
            }
        }
    }
    if messages.is_empty() {
        return Ok(WavMarkReading {
            payload: None,
            start_samples: Vec::new(),
        });
    }
    let payload = (0..16)
        .map(|bit| {
            let ones = messages.iter().filter(|m| m[bit]).count();
            u8::from(ones as f64 / messages.len() as f64 >= 0.5)
        })
        .collect();
    Ok(WavMarkReading {
        payload: Some(payload),
        start_samples: locations,
    })
}

fn binomial_tail(matches: usize, size: usize) -> f64 {
    let mut total: u128 = 0;
    let mut comb: u128 = 1;
    for k in 0..=size {
        if k >= matches {
            total += comb;
        }
        comb = comb * (size - k) as u128 / (k + 1) as u128;
    }
    total as f64 / (1u128 << size) as f64
}

fn score(
    values: &[bool],
    reference: &[u8],
    positions: Vec<usize>,
    comparisons: f64,
) -> WavMarkSignatureReading {
    let expected: Vec<bool> = reference
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |i| byte >> i & 1 == 1))
        .collect();
    let matches = values.iter().zip(&expected).filter(|(a, b)| a == b).count();
    WavMarkSignatureReading {
        start_samples: positions,
        decoded_bits: values.iter().map(|&b| u8::from(b)).collect(),
        reference_hex: reference.iter().fold(String::new(), |mut hex, b| {
            let _ = write!(hex, "{b:02x}");
            hex
        }),
        matching_bits: matches,
        adjusted_binomial_tail: (comparisons * binomial_tail(matches, expected.len())).min(1.0),
        framing: "openvoice-default".into(),
    }
}

/// `read_openvoice_default`: raw 32-bit blocks at native samples 0 and 32000, without sync bits.
pub fn read_openvoice_default(
    network: &dyn WavMarkNetwork,
    samples: &[f32],
) -> Result<(WavMarkSignatureReading, Vec<WavMarkSignatureReading>), DecoderError> {
    if samples.len() < MIN_READ_SAMPLES || !samples.iter().all(|v| v.is_finite()) {
        return Err(DecoderError::Invalid(
            "OpenVoice default framing requires at least 16000 finite native mono samples".into(),
        ));
    }
    let starts: Vec<usize> = OPENVOICE_START_SAMPLES
        .into_iter()
        .filter(|s| s + MIN_READ_SAMPLES <= samples.len())
        .collect();
    let windows: Vec<&[f32]> = starts
        .iter()
        .map(|&s| &samples[s..s + MIN_READ_SAMPLES])
        .collect();
    let decoded = decode_checked(network, &windows, "WavMark returned invalid raw signature values")?;
    let comparisons = if starts.len() == 2 { 3.0 } else { 1.0 };
    let fragments: Vec<WavMarkSignatureReading> = decoded
        .iter()
        .zip(&starts)
        .enumerate()
        .map(|(i, (bits, &start))| {
            score(
                bits,
                &OPENVOICE_REFERENCE[i * 4..(i + 1) * 4],
                vec![start],
                comparisons,
            )
        })
        .collect();
    let flat: Vec<bool> = decoded.concat();
    let mut whole = score(
        &flat,
        &OPENVOICE_REFERENCE[..starts.len() * 4],
        starts,
        comparisons,
    );
    for fragment in &fragments {
        whole.adjusted_binomial_tail = whole.adjusted_binomial_tail.min(fragment.adjusted_binomial_tail);
    }
    Ok((whole, fragments))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Positive(usize);

    impl WavMarkNetwork for Positive {
        fn decode(&self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>, DecoderError> {
            Ok(windows
                .iter()
                .map(|w| {
                    let mut row = vec![0.0; 32];
                    if w[0] as usize == self.0 {
                        for (slot, bit) in row.iter_mut().zip(SYNC_PATTERN) {
                            *slot = f32::from(bit);
                        }
                    }
                    row
                })
                .collect())
        }
    }

    #[test]
    fn final_valid_window_is_always_checked() {
        for length in [16_000usize, 20_000, 20_001] {
            let end = length - 16_000;
            let samples: Vec<f32> = (0..length).map(|v| v as f32).collect();
            let reading = read(&Positive(end), &samples).unwrap();
            assert_eq!(reading.payload, Some(vec![0; 16]));
            assert_eq!(reading.start_samples, vec![end]);
        }
    }

    #[test]
    fn binomial_tail_of_full_agreement() {
        assert_eq!(binomial_tail(32, 32), 1.0 / 4_294_967_296.0);
        assert_eq!(binomial_tail(0, 32), 1.0);
    }
}

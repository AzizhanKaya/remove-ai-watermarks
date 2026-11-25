//! Full-track DAC and SNAC resynthesis on the 44.1 kHz rational grid (`audio_dac.py`, `audio_snac.py`).
//!
//! Raw mid/side stems, silent-stem skipping, quiet-stem gain staging, stem RMS restoration, an
//! optional zero-phase high-frequency crossover and a 0.999 peak ceiling.

use rayon::prelude::*;

use super::encodec::{left_right, mid_side};
use super::{CodecKind, CodecRequest, NeuralCodec, StereoMode};
use crate::clock::RationalClock;
use crate::dsp::numpy::{rms_f32, sum};
use crate::dsp::{Band, Section, butter_sos, sosfiltfilt};
use crate::error::{AudioError, Result};
use crate::frames::Frames;

pub const DAC_REPOSITORY: &str = "descript/dac_44khz";
pub const DAC_REVISION: &str = "c1bc521685adf9cfe247bc39a5ca58917eda1ac4";
pub const DAC_CHECKPOINT_SHA256: &str = "6128ebff483a41422b0164d079a3773b0d8d82e64c4293d775994cbf8baf913a";
pub const SNAC_REPOSITORY: &str = "hubertsiuzdak/snac_44khz";
pub const SNAC_REVISION: &str = "873ebef9718b89660340c6f55a2b515e98cfa1d9";
pub const SNAC_CHECKPOINT_SHA256: &str = "b0a676cbdc8d1cc53186f6d777bc956fb7932ceacdc657a4c3741646e9e7ead0";
pub const FIXED_QUANTIZERS: usize = 3;
pub const DAC_HOP: usize = 512;
pub const SNAC_HOP: usize = 3072;
/// Default crossover on tracks at or above 44.1 kHz.
pub const DEFAULT_PRESERVE_HF_HZ: f64 = 14_000.0;
const SILENT_STEM_RMS: f64 = 1e-5;
const PEAK_CEILING: f64 = 0.999;

/// `validate_preserve_hf_hz`.
pub fn validate_preserve_hf_hz(preserve_hf_hz: f64, sample_rate: u32) -> Result<()> {
    let upper = f64::from(sample_rate) * 0.45;
    if !preserve_hf_hz.is_finite() || !(8000.0..=upper).contains(&preserve_hf_hz) {
        return Err(AudioError::invalid(format!(
            "preserve_hf_hz must be between 8000 and {upper:.0} Hz for {sample_rate} Hz"
        )));
    }
    Ok(())
}

fn interleaved_rms(values: &[f64]) -> f64 {
    let squares: Vec<f64> = values.iter().map(|v| v * v).collect();
    (sum(&squares) / values.len() as f64).sqrt()
}

/// `blend_hf_crossover`: keep the original's band above `preserve_hf_hz`, level-matched below it.
pub fn blend_hf_crossover(
    processed: &Frames,
    original: &Frames,
    sample_rate: u32,
    preserve_hf_hz: f64,
) -> Result<Frames> {
    validate_preserve_hf_hz(preserve_hf_hz, sample_rate)?;
    if processed.frames() < 64 {
        return Ok(processed.clone());
    }
    let rate = f64::from(sample_rate);
    let low = butter_sos(8, Band::Lowpass(preserve_hf_hz), rate)?;
    let high = butter_sos(8, Band::Highpass(preserve_hf_hz), rate)?;
    let (processed_channels, original_channels) = (processed.to_channels(), original.to_channels());
    let jobs: Vec<(&[Section], &Vec<f32>)> = processed_channels
        .iter()
        .map(|c| (low.as_slice(), c))
        .chain(original_channels.iter().map(|c| (low.as_slice(), c)))
        .chain(original_channels.iter().map(|c| (high.as_slice(), c)))
        .collect();
    let filtered: Vec<Vec<f64>> = jobs
        .par_iter()
        .map(|(sos, c)| sosfiltfilt(sos, c, None).map_err(AudioError::from))
        .collect::<Result<_>>()?;
    let interleave = |channels: &[Vec<f64>]| -> Vec<f64> {
        let frames = channels.first().map_or(0, Vec::len);
        (0..frames)
            .flat_map(|i| channels.iter().map(move |c| c[i]))
            .collect()
    };
    let width = processed.channels();
    let lp_proc = interleave(&filtered[..width]);
    let lp_orig = interleave(&filtered[width..2 * width]);
    let hp_orig = interleave(&filtered[2 * width..]);
    let (in_rms, out_rms) = (interleaved_rms(&lp_orig), interleaved_rms(&lp_proc));
    let gain = (in_rms >= 1e-6 && out_rms >= 1e-6).then(|| in_rms / out_rms);
    let blended: Vec<f32> = lp_proc
        .iter()
        .zip(&hp_orig)
        .map(|(&lp, &hp)| (gain.map_or(lp, |g| lp * g) + hp) as f32)
        .collect();
    Ok(Frames::new(blended, processed.channels())?)
}

/// `resynthesize` for DAC or SNAC: a finite mono or stereo signal on its exact native grid.
pub fn resynthesize(
    codec: &dyn NeuralCodec,
    samples: &Frames,
    sample_rate: u32,
    mode: StereoMode,
    preserve_hf_hz: Option<f64>,
    request: &CodecRequest,
) -> Result<Frames> {
    let kind = codec.kind();
    let label = kind.label();
    let hop = if kind == CodecKind::Snac {
        SNAC_HOP
    } else {
        DAC_HOP
    };
    if let Some(hf) = preserve_hf_hz {
        validate_preserve_hf_hz(hf, sample_rate)?;
    }
    if samples.is_empty() || !matches!(samples.channels(), 1 | 2) {
        return Err(AudioError::invalid(format!(
            "{label} expects a nonempty floating (frames, 1 or 2 channels) matrix"
        )));
    }
    let clock = RationalClock::new(samples.frames(), sample_rate, 44_100, hop, hop)?;
    if !samples.is_finite() {
        return Err(AudioError::invalid(format!("{label} input must be finite")));
    }
    let mid_side_mode = samples.channels() == 2 && mode == StereoMode::MidSide;
    let stems = if mid_side_mode {
        mid_side(samples)
    } else {
        samples.clone()
    };
    let stem_channels = stems.to_channels();
    let stem_rms: Vec<f64> = stem_channels.iter().map(|c| rms_f32(c)).collect();
    let active: Vec<usize> = (0..stem_channels.len())
        .filter(|&c| stem_rms[c] >= SILENT_STEM_RMS)
        .collect();
    if kind == CodecKind::Snac && active.is_empty() {
        return Ok(Frames::zeros(samples.frames(), samples.channels()));
    }
    let to_model = Frames::from_channels(
        &active
            .iter()
            .map(|&c| stem_channels[c].clone())
            .collect::<Vec<_>>(),
    );
    let prepared = if active.is_empty() {
        None
    } else {
        Some(clock.pad_model(&clock.forward(&to_model)?)?)
    };
    if prepared.as_ref().is_some_and(|p| !p.is_finite()) {
        return Err(AudioError::invalid(format!(
            "{label} prepared input must be finite"
        )));
    }
    let mut outputs = Vec::with_capacity(stem_channels.len());
    for (channel, stem) in stem_channels.iter().enumerate() {
        let Some(position) = active.iter().position(|&c| c == channel) else {
            outputs.push(vec![0.0f32; stem.len()]);
            continue;
        };
        let model_stem = prepared.as_ref().map(|p| p.channel(position)).unwrap_or_default();
        let peak_in = model_stem.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if peak_in > 1e-6 {
            (0.50 / f64::from(peak_in).max(1e-6)).min(32.0)
        } else {
            1.0
        };
        let scale32 = scale as f32;
        let codec_in: Vec<f32> = model_stem.iter().map(|v| v * scale32).collect();
        let decoded = codec
            .reconstruct(&codec_in, request)
            .map_err(|e| AudioError::failed(e.0))?;
        if decoded.len() < clock.padded_frames() {
            return Err(AudioError::failed(format!(
                "{label} returned fewer samples than expected for the padded model grid"
            )));
        }
        let decoded: Vec<f32> = decoded[..clock.padded_frames()]
            .iter()
            .map(|v| v / scale32)
            .collect();
        if !decoded.iter().all(|v| v.is_finite()) {
            return Err(AudioError::failed(format!("{label} returned nonfinite samples")));
        }
        let restored = clock
            .inverse(&clock.unpad_model(&Frames::mono(decoded))?)?
            .into_vec();
        let out_rms = rms_f32(&restored);
        let restored = if out_rms > 1e-6 {
            let gain = (stem_rms[channel] / out_rms) as f32;
            restored.into_iter().map(|v| v * gain).collect()
        } else {
            restored
        };
        outputs.push(restored);
    }
    let mut delivered = Frames::from_channels(&outputs);
    if mid_side_mode {
        delivered = left_right(&delivered);
    }
    if let Some(hf) = preserve_hf_hz {
        delivered = blend_hf_crossover(&delivered, samples, sample_rate, hf)?;
    }
    let peak = f64::from(delivered.peak());
    if peak > PEAK_CEILING {
        delivered = delivered.scale((PEAK_CEILING / peak) as f32);
    }
    if delivered.frames() != samples.frames()
        || delivered.channels() != samples.channels()
        || !delivered.is_finite()
    {
        return Err(AudioError::failed(format!(
            "{label} returned invalid native sample geometry or values"
        )));
    }
    Ok(delivered)
}

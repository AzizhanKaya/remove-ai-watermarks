//! Tagged 5.1/7.1 stem routing: FL/FR, FC and surround pairs through the codec, LFE low-passed.

use crate::dsp::{Band, butter_sos, sosfiltfilt};
use crate::error::{AudioError, Result};
use crate::frames::Frames;
use crate::io::AudioStreamInfo;

const SURROUND_51: [&str; 2] = ["5.1", "5.1(side)"];
const SURROUND_71: [&str; 3] = ["7.1", "7.1(wide)", "7.1(wide-side)"];

/// `_is_mono_or_stereo`: a labelled mono or stereo layout.
pub fn is_mono_or_stereo(info: &AudioStreamInfo) -> bool {
    matches!(info.channel_layout.as_deref(), Some("mono" | "stereo")) && matches!(info.channels, 1 | 2)
}

/// `_is_tagged_surround`: a declared 5.1 or 7.1 layout.
pub fn is_tagged_surround(info: &AudioStreamInfo) -> bool {
    let layout = info.channel_layout.as_deref().unwrap_or("");
    (info.channels == 6 && SURROUND_51.contains(&layout))
        || (info.channels == 8 && SURROUND_71.contains(&layout))
}

/// `_lowpass_lfe_channel`: strip energy above 120 Hz from the LFE while keeping subwoofer bass.
pub fn lowpass_lfe(lfe: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
    if lfe.len() <= 16 || sample_rate <= 300 {
        return Ok(lfe.to_vec());
    }
    let sos = butter_sos(4, Band::Lowpass(120.0), f64::from(sample_rate))?;
    let wide: Vec<f64> = lfe.iter().map(|&v| f64::from(v)).collect();
    let pad = (3 * (2 * sos.len() + 1)).min(wide.len() - 1);
    Ok(sosfiltfilt(&sos, &wide, Some(pad))?
        .into_iter()
        .map(|v| v as f32)
        .collect())
}

/// `_transform_surround_stems`.
pub fn transform_surround_stems(
    samples: &Frames,
    sample_rate: u32,
    stem: &mut dyn FnMut(&Frames) -> Result<Frames>,
) -> Result<Frames> {
    let channels = samples.channels();
    if matches!(channels, 1 | 2) {
        return stem(samples);
    }
    if !matches!(channels, 6 | 8) {
        return Err(AudioError::invalid(
            "Surround stem decomposition requires 6 (5.1) or 8 (7.1) channels",
        ));
    }
    let front = stem(&samples.select_channels(0..2))?;
    let center = stem(&samples.select_channels(2..3))?;
    let lfe = Frames::mono(lowpass_lfe(&samples.channel(3), sample_rate)?);
    let mut blocks = vec![front, center, lfe, stem(&samples.select_channels(4..6))?];
    if channels == 8 {
        blocks.push(stem(&samples.select_channels(6..8))?);
    }
    Ok(Frames::hstack(&blocks))
}

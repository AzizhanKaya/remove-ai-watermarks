//! Optional pinned EnCodec resynthesis preserving native geometry (`_internal/audio_codec.py`).

use rayon::prelude::*;

use super::{CodecKind, CodecRequest, NeuralCodec, StereoMode};
use crate::chunks::transform_chunks;
use crate::dsp::resample_rates;
use crate::error::{AudioError, Result};
use crate::frames::Frames;
use crate::options::valid_bandwidth;

pub const MODEL_REPOSITORY: &str = "facebook/encodec_24khz";
pub const MODEL_REVISION: &str = "c1dbe2ae3f1de713481a3b3e7c47f357092ee040";
pub const CHECKPOINT_SHA256: &str = "37a7cb100f71a29e6c1d815aca8666a1d7ea8885ebe44c306c751a5103559d57";
pub const CONTEXT_SECONDS: usize = 1;

/// `(L + R) / 2` and `(L - R) / 2`.
pub fn mid_side(samples: &Frames) -> Frames {
    let (l, r) = (samples.channel(0), samples.channel(1));
    let mid = l.iter().zip(&r).map(|(a, b)| 0.5 * (a + b)).collect();
    let side = l.iter().zip(&r).map(|(a, b)| 0.5 * (a - b)).collect();
    Frames::from_channels(&[mid, side])
}

/// `M + S`, `M - S`.
pub fn left_right(stems: &Frames) -> Frames {
    let (m, s) = (stems.channel(0), stems.channel(1));
    let left = m.iter().zip(&s).map(|(a, b)| a + b).collect();
    let right = m.iter().zip(&s).map(|(a, b)| a - b).collect();
    Frames::from_channels(&[left, right])
}

fn check_args(sample_rate: u32, bandwidth: f64) -> Result<()> {
    if !valid_bandwidth(bandwidth) || sample_rate == 0 {
        return Err(AudioError::invalid(
            "EnCodec requires a positive sample rate and bandwidth 6, 12, or 24 kbps",
        ));
    }
    Ok(())
}

/// `audio_codec.resynthesize`: channels independently or as mid/side, trimming only padding.
pub fn resynthesize(
    codec: &dyn NeuralCodec,
    samples: &Frames,
    sample_rate: u32,
    mode: StereoMode,
    request: &CodecRequest,
) -> Result<Frames> {
    check_args(sample_rate, request.bandwidth_kbps)?;
    if samples.is_empty() || !samples.is_finite() {
        return Err(AudioError::invalid(
            "EnCodec requires finite nonempty sample frames and channels",
        ));
    }
    let use_ms = mode == StereoMode::MidSide && samples.channels() == 2;
    let codec_in = if use_ms {
        mid_side(samples)
    } else {
        samples.clone()
    };
    let model_rate = CodecKind::Encodec.model_rate();
    let stems: Vec<Vec<f32>> = codec_in
        .to_channels()
        .par_iter()
        .map(|channel| {
            let native = resample_rates(channel, sample_rate, model_rate);
            let reconstructed = codec
                .reconstruct(&native, request)
                .map_err(|e| AudioError::failed(e.0))?;
            Ok(resample_rates(&reconstructed, model_rate, sample_rate))
        })
        .collect::<Result<_>>()?;
    let frames = samples.frames();
    if stems.iter().any(|s| s.len() < frames) || stems.len() != samples.channels() {
        return Err(AudioError::failed(
            "EnCodec returned insufficient samples or changed channels",
        ));
    }
    let trimmed: Vec<Vec<f32>> = stems
        .into_iter()
        .map(|mut s| {
            s.truncate(frames);
            s
        })
        .collect();
    let trimmed = Frames::from_channels(&trimmed);
    let result = if use_ms { left_right(&trimmed) } else { trimmed };
    if !result.is_finite() {
        return Err(AudioError::failed("EnCodec returned nonfinite samples"));
    }
    Ok(result)
}

/// `resynthesize_chunks`: one second of neighboring context, exact central samples.
pub fn resynthesize_chunks<'a, I>(
    codec: &'a dyn NeuralCodec,
    chunks: I,
    sample_rate: u32,
    mode: StereoMode,
    request: CodecRequest,
) -> Result<impl Iterator<Item = Result<Frames>> + 'a>
where
    I: Iterator<Item = Result<Frames>> + 'a,
{
    check_args(sample_rate, request.bandwidth_kbps)?;
    transform_chunks(
        chunks,
        move |padded: &Frames| resynthesize(codec, padded, sample_rate, mode, &request),
        sample_rate as usize * CONTEXT_SECONDS,
    )
}

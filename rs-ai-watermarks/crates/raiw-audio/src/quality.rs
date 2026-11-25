//! Delivered-audio fidelity against the decoded marked source (`_internal/audio_quality.py`).

use std::path::Path;

use raiw_core::constants::{AUDIO_LOSSY_OUTPUT_EXTENSIONS, AUDIO_OPUS_EXTENSIONS};
use raiw_core::fs::suffix_lower;
use serde::Serialize;

use crate::error::{AudioError, Result};
use crate::frames::Frames;
use crate::io::{AudioReader, AudioStreamInfo, DEFAULT_CHUNK_FRAMES, FfmpegTools, probe_audio};

/// Delivered-file sample differences against the decoded marked source.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioFidelity {
    pub frames: usize,
    pub rms_error: f64,
    pub max_abs_error: f64,
    pub snr_db: Option<f64>,
    pub peak_amplitude: f64,
    pub samples_above_full_scale: usize,
    pub source_rms: f64,
    pub output_rms: f64,
    pub source_peak: f64,
    pub source_samples_above_full_scale: usize,
    pub worst_window_rms_ratio: f64,
    pub max_side_mid_drift_db: Option<f64>,
}

/// Model-free delivered audio integrity against source energy and full scale.
pub fn audio_quality_passed(f: &AudioFidelity, lossy: bool) -> bool {
    if f.frames == 0 {
        return false;
    }
    let finite = [
        f.rms_error,
        f.max_abs_error,
        f.peak_amplitude,
        f.source_rms,
        f.output_rms,
        f.source_peak,
        f.worst_window_rms_ratio,
    ];
    if !finite.iter().all(|v| v.is_finite()) {
        return false;
    }
    if let Some(drift) = f.max_side_mid_drift_db
        && (!drift.is_finite() || drift.abs() > if lossy { 9.0 } else { 6.0 })
    {
        return false;
    }
    if f.source_rms >= 1e-3 && f.output_rms < (f.source_rms * 0.005).max(1e-5) {
        return false;
    }
    if f.source_rms >= 1e-4 && f.output_rms > f.source_rms * 4.0 {
        return false;
    }
    if f.source_rms >= 1e-3 && (f.worst_window_rms_ratio < 0.005 || f.worst_window_rms_ratio > 4.0) {
        return false;
    }
    if lossy {
        let allowed = f.source_samples_above_full_scale.max(16.max(f.frames / 500));
        if f.samples_above_full_scale > allowed || f.peak_amplitude > 1.05f64.max(f.source_peak * 1.05) {
            return false;
        }
    } else if f.samples_above_full_scale > f.source_samples_above_full_scale
        || (f.source_peak <= 1.0 && f.peak_amplitude > 1.0)
    {
        return false;
    }
    true
}

/// Whether `output` is a lossy delivery for [`audio_quality_passed`].
pub fn is_lossy_output(output: &Path) -> bool {
    AUDIO_LOSSY_OUTPUT_EXTENSIONS.contains(&suffix_lower(output).as_str())
}

/// `_opus_aligned_decode_info`: decode Opus deliveries back on the source clock.
pub fn aligned_decode_info(
    tools: &FfmpegTools,
    info: &AudioStreamInfo,
    output: &Path,
    stream: usize,
    delivered: Option<&AudioStreamInfo>,
    strict_geometry: bool,
) -> Result<(AudioStreamInfo, bool)> {
    let suffix = suffix_lower(output);
    let delivered = match delivered {
        Some(d) => d.clone(),
        None => probe_audio(tools, output, stream)?,
    };
    let opus_clock = AUDIO_OPUS_EXTENSIONS.contains(&suffix.as_str()) || delivered.codec == "opus";
    if strict_geometry {
        let expected_rate = if opus_clock { 48_000 } else { info.sample_rate };
        if (delivered.channels, delivered.sample_rate) != (info.channels, expected_rate) {
            return Err(AudioError::failed(
                "Audio output changed the source sample rate or channel count",
            ));
        }
        if info.channel_layout.is_some()
            && !AUDIO_LOSSY_OUTPUT_EXTENSIONS.contains(&suffix.as_str())
            && delivered.channel_layout != info.channel_layout
        {
            return Err(AudioError::failed(
                "Audio output changed the source channel layout",
            ));
        }
    }
    let mut decode = delivered;
    if opus_clock {
        decode.sample_rate = info.sample_rate;
    }
    Ok((decode, opus_clock))
}

/// Accumulates 1-second active windows, global energy and stereo side/mid drift.
#[derive(Debug, Default)]
struct FidelityAccumulator {
    channels: usize,
    win_target: usize,
    power: f64,
    output_power: f64,
    residual: f64,
    max_error: f64,
    peak: f64,
    source_peak: f64,
    count: usize,
    above: usize,
    source_above: usize,
    win_frames: usize,
    win_src: f64,
    win_out: f64,
    mid_a: f64,
    side_a: f64,
    mid_b: f64,
    side_b: f64,
    min_ratio: Option<f64>,
    max_ratio: Option<f64>,
    max_drift: Option<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl FidelityAccumulator {
    fn new(channels: usize, sample_rate: u32) -> Self {
        Self {
            channels,
            win_target: (sample_rate as usize).max(1),
            ..Self::default()
        }
    }

    fn flush(&mut self) {
        if self.win_frames == 0 {
            return;
        }
        let frames = self.win_frames as f64;
        let denom = (self.win_frames * self.channels) as f64;
        let (src, out) = ((self.win_src / denom).sqrt(), (self.win_out / denom).sqrt());
        if src >= 1e-3 {
            let ratio = out / src;
            self.min_ratio = Some(self.min_ratio.map_or(ratio, |m| m.min(ratio)));
            self.max_ratio = Some(self.max_ratio.map_or(ratio, |m| m.max(ratio)));
        }
        if self.channels == 2 {
            let (ma, sa) = ((self.mid_a / frames).sqrt(), (self.side_a / frames).sqrt());
            let (mb, sb) = ((self.mid_b / frames).sqrt(), (self.side_b / frames).sqrt());
            if ma >= 1e-3 && mb >= 1e-3 && (sa >= 1e-4 || sb >= 1e-4) {
                let drift = (20.0 * ((sb.max(1e-6) / mb) / (sa.max(1e-6) / ma)).log10()).abs();
                self.max_drift = Some(self.max_drift.map_or(drift, |m| m.max(drift)));
            }
        }
        self.win_frames = 0;
        (self.win_src, self.win_out) = (0.0, 0.0);
        (self.mid_a, self.side_a, self.mid_b, self.side_b) = (0.0, 0.0, 0.0, 0.0);
    }

    fn feed(&mut self, before: &Frames, after: &Frames) -> Result<()> {
        if !before.is_finite() || !after.is_finite() {
            return Err(AudioError::failed("Delivered audio contains nonfinite samples"));
        }
        let c = self.channels;
        let a: Vec<f64> = before.as_slice().iter().map(|&v| f64::from(v)).collect();
        let b: Vec<f64> = after.as_slice().iter().map(|&v| f64::from(v)).collect();
        for (&x, &y) in a.iter().zip(&b) {
            let delta = y - x;
            self.residual += delta * delta;
            self.max_error = self.max_error.max(delta.abs());
            self.source_peak = self.source_peak.max(x.abs());
            self.peak = self.peak.max(y.abs());
            self.source_above += usize::from(x.abs() > 1.0);
            self.above += usize::from(y.abs() > 1.0);
        }
        let len = before.frames();
        self.count += len;
        let mut pos = 0;
        while pos < len {
            let take = (self.win_target - self.win_frames).min(len - pos);
            let seg_a = &a[pos * c..(pos + take) * c];
            let seg_b = &b[pos * c..(pos + take) * c];
            let (pa, pb) = if c == 2 {
                let split = |s: &[f64]| -> (Vec<f64>, Vec<f64>) {
                    (
                        s.iter().step_by(2).copied().collect(),
                        s.iter().skip(1).step_by(2).copied().collect(),
                    )
                };
                let ((la, ra), (lb, rb)) = (split(seg_a), split(seg_b));
                let pa = dot(&la, &la) + dot(&ra, &ra);
                let pb = dot(&lb, &lb) + dot(&rb, &rb);
                let (ca, cb) = (dot(&la, &ra), dot(&lb, &rb));
                self.mid_a += 0.25 * (pa + 2.0 * ca);
                self.side_a += 0.25 * (pa - 2.0 * ca);
                self.mid_b += 0.25 * (pb + 2.0 * cb);
                self.side_b += 0.25 * (pb - 2.0 * cb);
                (pa, pb)
            } else {
                (dot(seg_a, seg_a), dot(seg_b, seg_b))
            };
            self.power += pa;
            self.output_power += pb;
            self.win_src += pa;
            self.win_out += pb;
            self.win_frames += take;
            pos += take;
            if self.win_frames >= self.win_target {
                self.flush();
            }
        }
        Ok(())
    }

    fn finish(mut self) -> AudioFidelity {
        let min_tail = (self.win_target / 4).max(1);
        if self.win_frames >= min_tail || (self.min_ratio.is_none() && self.win_frames > 0) {
            self.flush();
        }
        let values = (self.count * self.channels) as f64;
        let snr =
            (self.power > 0.0 && self.residual > 0.0).then(|| 10.0 * (self.power / self.residual).log10());
        let worst = match (self.min_ratio, self.max_ratio) {
            (Some(min), Some(max)) => {
                if max > 4.0 {
                    max
                } else {
                    min
                }
            }
            _ => 1.0,
        };
        let drift = if self.channels == 2 {
            Some(self.max_drift.unwrap_or(0.0))
        } else {
            self.max_drift
        };
        AudioFidelity {
            frames: self.count,
            rms_error: (self.residual / values).sqrt(),
            max_abs_error: self.max_error,
            snr_db: snr,
            peak_amplitude: self.peak,
            samples_above_full_scale: self.above,
            source_rms: (self.power / values).sqrt(),
            output_rms: (self.output_power / values).sqrt(),
            source_peak: self.source_peak,
            source_samples_above_full_scale: self.source_above,
            worst_window_rms_ratio: worst,
            max_side_mid_drift_db: drift,
        }
    }
}

/// `measure_fidelity(source, info, output)`: decode both files and compare every sample frame.
pub fn measure_fidelity(
    tools: &FfmpegTools,
    source: &Path,
    info: &AudioStreamInfo,
    output: &Path,
    stream: usize,
) -> Result<AudioFidelity> {
    let (decode_info, opus_clock) = aligned_decode_info(tools, info, output, stream, None, true)?;
    let adts = suffix_lower(output) == ".aac";
    let relaxed = opus_clock || adts;
    let tail_slack = if opus_clock {
        2.max(info.sample_rate as usize / 100)
    } else if adts {
        4096
    } else {
        0
    };
    let mut left = AudioReader::open(tools, source, info, DEFAULT_CHUNK_FRAMES)?;
    let mut right = AudioReader::open(tools, output, &decode_info, DEFAULT_CHUNK_FRAMES)?;
    let mut acc = FidelityAccumulator::new(info.channels, info.sample_rate);
    let (mut total_before, mut total_after) = (0usize, 0usize);
    loop {
        let before = left.next().transpose()?;
        let after = right.next().transpose()?;
        total_before += before.as_ref().map_or(0, Frames::frames);
        total_after += after.as_ref().map_or(0, Frames::frames);
        let (before, after) = match (before, after) {
            (None, None) => break,
            (Some(b), Some(a)) if !relaxed && b.frames() == a.frames() && b.channels() == a.channels() => {
                (b, a)
            }
            (Some(b), Some(a)) if relaxed => {
                if b.channels() != a.channels() {
                    return Err(AudioError::failed("Delivered audio changed the channel count"));
                }
                let len = b.frames().min(a.frames());
                (b.slice(0..len), a.slice(0..len))
            }
            _ if relaxed => break,
            _ => {
                return Err(AudioError::failed(
                    "Delivered audio changed the decoded sample-frame count",
                ));
            }
        };
        acc.feed(&before, &after)?;
    }
    if relaxed {
        for block in left {
            total_before += block?.frames();
        }
        for block in right {
            total_after += block?.frames();
        }
        if total_after.abs_diff(total_before) > tail_slack || acc.count == 0 {
            return Err(AudioError::failed(
                "Delivered audio changed the decoded sample-frame count",
            ));
        }
    }
    Ok(acc.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fidelity() -> AudioFidelity {
        AudioFidelity {
            frames: 16_000,
            rms_error: 0.01,
            max_abs_error: 0.02,
            snr_db: Some(20.0),
            peak_amplitude: 0.5,
            samples_above_full_scale: 0,
            source_rms: 0.2,
            output_rms: 0.19,
            source_peak: 0.5,
            source_samples_above_full_scale: 0,
            worst_window_rms_ratio: 0.95,
            max_side_mid_drift_db: Some(0.5),
        }
    }

    #[test]
    fn gate_rejects_silence_gain_and_drift() {
        assert!(audio_quality_passed(&fidelity(), false));
        assert!(!audio_quality_passed(
            &AudioFidelity {
                output_rms: 1e-6,
                ..fidelity()
            },
            false
        ));
        assert!(!audio_quality_passed(
            &AudioFidelity {
                output_rms: 0.9,
                ..fidelity()
            },
            false
        ));
        assert!(!audio_quality_passed(
            &AudioFidelity {
                worst_window_rms_ratio: 0.001,
                ..fidelity()
            },
            false
        ));
        assert!(!audio_quality_passed(
            &AudioFidelity {
                max_side_mid_drift_db: Some(7.0),
                ..fidelity()
            },
            false
        ));
        assert!(audio_quality_passed(
            &AudioFidelity {
                max_side_mid_drift_db: Some(7.0),
                ..fidelity()
            },
            true
        ));
        assert!(!audio_quality_passed(
            &AudioFidelity {
                peak_amplitude: 1.01,
                ..fidelity()
            },
            false
        ));
        assert!(audio_quality_passed(
            &AudioFidelity {
                peak_amplitude: 1.01,
                ..fidelity()
            },
            true
        ));
        assert!(!audio_quality_passed(
            &AudioFidelity {
                frames: 0,
                ..fidelity()
            },
            false
        ));
    }
}

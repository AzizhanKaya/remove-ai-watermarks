//! Phase-invariant multi-resolution STFT and 64-band log-Mel distances (`compute_perceptual_metrics`).

use std::path::Path;

use rayon::prelude::*;
use serde::Serialize;

use crate::dsp::stft::StftPlan;
use crate::error::Result;
use crate::frames::Frames;
use crate::io::{AudioReader, AudioStreamInfo, FfmpegTools};
use crate::quality::aligned_decode_info;

pub const RESOLUTIONS: [(usize, usize); 3] = [(512, 128), (1024, 256), (2048, 512)];
pub const MEL_BANDS: usize = 64;
pub const BLOCK_FRAMES: usize = 163_840;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioPerceptualMetrics {
    pub mr_stft_loss: f64,
    pub mr_stft_sc: f64,
    pub mr_stft_log_mag: f64,
    pub mel_distance_db: f64,
}

/// `_mel_filterbank(sample_rate, n_fft, n_mels)`, row-major `(n_mels, n_fft / 2 + 1)`.
pub fn mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f64>> {
    let rate = f64::from(sample_rate);
    let (mut f_min, mut f_max) = (20.0, 20_000.0f64.min(0.49 * rate));
    if f_max <= f_min {
        f_min = 0.0;
        f_max = 0.49 * rate;
    }
    let to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let (lo, hi) = (to_mel(f_min), to_mel(f_max));
    let count = n_mels + 2;
    let step = (hi - lo) / (count - 1) as f64;
    let hz: Vec<f64> = (0..count)
        .map(|i| {
            let mel = if i == count - 1 { hi } else { i as f64 * step + lo };
            700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
        })
        .collect();
    let spacing = 1.0 / (n_fft as f64 * (1.0 / rate));
    let freqs: Vec<f64> = (0..=n_fft / 2).map(|k| k as f64 * spacing).collect();
    (0..n_mels)
        .map(|m| {
            let (left, center, right) = (hz[m], hz[m + 1], hz[m + 2]);
            let norm = (right - left).max(1e-9);
            freqs
                .iter()
                .map(|&f| {
                    let up = (f - left) / (center - left).max(1e-9);
                    let down = (right - f) / (right - center).max(1e-9);
                    0.0f64.max(up.min(down)) * (2.0 / norm)
                })
                .collect()
        })
        .collect()
}

#[derive(Debug, Default, Clone, Copy)]
struct Sums {
    diff_sq: f64,
    src_sq: f64,
    log_l1: f64,
    bins: usize,
    mel_sq_db: f64,
    mel_bins: usize,
}

impl std::ops::Add for Sums {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            diff_sq: self.diff_sq + o.diff_sq,
            src_sq: self.src_sq + o.src_sq,
            log_l1: self.log_l1 + o.log_l1,
            bins: self.bins + o.bins,
            mel_sq_db: self.mel_sq_db + o.mel_sq_db,
            mel_bins: self.mel_bins + o.mel_bins,
        }
    }
}

struct Metrics {
    plans: Vec<StftPlan>,
    mel: Vec<Vec<f64>>,
}

impl Metrics {
    fn new(sample_rate: u32) -> Self {
        let plans = RESOLUTIONS
            .iter()
            .map(|&(n, hop)| StftPlan::hann(n, hop))
            .collect();
        Self {
            plans,
            mel: mel_filterbank(sample_rate, RESOLUTIONS[2].0, MEL_BANDS),
        }
    }

    fn mel_db(&self, mag: &[f64]) -> Vec<f64> {
        self.mel
            .iter()
            .map(|row| {
                let energy: f64 = row.iter().zip(mag).map(|(w, m)| m * m * w).sum();
                10.0 * energy.max(1e-10).log10()
            })
            .collect()
    }

    fn channel(&self, plan_index: usize, a: &[f64], b: &[f64]) -> Sums {
        let plan = &self.plans[plan_index];
        let with_mel = plan_index == RESOLUTIONS.len() - 1;
        (0..plan.frame_count(a.len()))
            .into_par_iter()
            .fold(
                || {
                    (
                        plan.scratch(),
                        vec![0.0; plan.bins()],
                        vec![0.0; plan.bins()],
                        Sums::default(),
                    )
                },
                |(mut scratch, mut ma, mut mb, mut sums), frame| {
                    plan.magnitudes(a, frame, &mut scratch, &mut ma);
                    plan.magnitudes(b, frame, &mut scratch, &mut mb);
                    for (&x, &y) in ma.iter().zip(&mb) {
                        sums.diff_sq += (y - x) * (y - x);
                        sums.src_sq += x * x;
                        sums.log_l1 += (y.max(1e-5).ln() - x.max(1e-5).ln()).abs();
                    }
                    sums.bins += ma.len();
                    if with_mel {
                        for (x, y) in self.mel_db(&ma).into_iter().zip(self.mel_db(&mb)) {
                            sums.mel_sq_db += (y - x) * (y - x);
                        }
                        sums.mel_bins += MEL_BANDS;
                    }
                    (scratch, ma, mb, sums)
                },
            )
            .map(|(_, _, _, sums)| sums)
            .reduce(Sums::default, |x, y| x + y)
    }

    fn block(&self, a: &Frames, b: &Frames, totals: &mut [Sums; 3]) {
        if a.frames() == 0 {
            return;
        }
        for c in 0..a.channels() {
            let ca: Vec<f64> = a.channel(c).into_iter().map(f64::from).collect();
            let cb: Vec<f64> = b.channel(c).into_iter().map(f64::from).collect();
            for (i, total) in totals.iter_mut().enumerate() {
                *total = *total + self.channel(i, &ca, &cb);
            }
        }
    }
}

/// `compute_perceptual_metrics(source, info, output)` over bounded 163840-frame blocks.
pub fn compute_perceptual_metrics(
    tools: &FfmpegTools,
    source: &Path,
    info: &AudioStreamInfo,
    output: &Path,
    stream: usize,
    delivered: Option<&AudioStreamInfo>,
) -> Result<AudioPerceptualMetrics> {
    let (decode_info, _) = aligned_decode_info(tools, info, output, stream, delivered, false)?;
    let metrics = Metrics::new(info.sample_rate);
    let mut totals = [Sums::default(); 3];
    let left = AudioReader::open(tools, source, info, BLOCK_FRAMES)?;
    let right = AudioReader::open(tools, output, &decode_info, BLOCK_FRAMES)?;
    let (mut buf_a, mut buf_b) = (Frames::empty(info.channels), Frames::empty(decode_info.channels));
    for (before, after) in left.zip(right) {
        let (before, after) = (before?, after?);
        let len = before.frames().min(after.frames());
        if len == 0 {
            continue;
        }
        buf_a.append(&before.slice(0..len));
        buf_b.append(&after.slice(0..len));
        while buf_a.frames() >= BLOCK_FRAMES {
            metrics.block(
                &buf_a.slice(0..BLOCK_FRAMES),
                &buf_b.slice(0..BLOCK_FRAMES),
                &mut totals,
            );
            buf_a = buf_a.slice(BLOCK_FRAMES..buf_a.frames());
            buf_b = buf_b.slice(BLOCK_FRAMES..buf_b.frames());
        }
    }
    metrics.block(&buf_a, &buf_b, &mut totals);
    Ok(summarize(&totals))
}

fn summarize(totals: &[Sums; 3]) -> AudioPerceptualMetrics {
    let scs: Vec<f64> = totals
        .iter()
        .map(|s| {
            if s.src_sq > 1e-12 {
                s.diff_sq.sqrt() / s.src_sq.sqrt().max(1e-7)
            } else {
                0.0
            }
        })
        .collect();
    let lms: Vec<f64> = totals.iter().map(|s| s.log_l1 / s.bins.max(1) as f64).collect();
    let mean_sc = scs.iter().sum::<f64>() / scs.len() as f64;
    let mean_lm = lms.iter().sum::<f64>() / lms.len() as f64;
    let mel = &totals[2];
    AudioPerceptualMetrics {
        mr_stft_loss: mean_sc + mean_lm,
        mr_stft_sc: mean_sc,
        mr_stft_log_mag: mean_lm,
        mel_distance_db: (mel.mel_sq_db / mel.mel_bins.max(1) as f64).sqrt(),
    }
}

/// Perceptual distances between two in-memory signals at one sample rate (block-wise like the file path).
pub fn perceptual_between(source: &Frames, output: &Frames, sample_rate: u32) -> AudioPerceptualMetrics {
    let metrics = Metrics::new(sample_rate);
    let mut totals = [Sums::default(); 3];
    let len = source.frames().min(output.frames());
    let mut start = 0;
    while start < len {
        let end = (start + BLOCK_FRAMES).min(len);
        metrics.block(&source.slice(start..end), &output.slice(start..end), &mut totals);
        start = end;
    }
    summarize(&totals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_signals_have_zero_distance() {
        let x = Frames::mono((0..5000).map(|n| (n as f32 * 0.01).sin() * 0.3).collect());
        let m = perceptual_between(&x, &x, 16_000);
        assert_eq!((m.mr_stft_loss, m.mel_distance_db), (0.0, 0.0));
    }

    #[test]
    fn filterbank_rows_are_triangles() {
        let fb = mel_filterbank(48_000, 2048, 64);
        assert_eq!((fb.len(), fb[0].len()), (64, 1025));
        assert!(fb.iter().all(|row| row.iter().all(|&w| w >= 0.0)));
    }
}

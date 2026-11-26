//! Host STFT/ISTFT matching `torch.stft`/`torch.istft` with `center=True`, reflect padding
//! and a periodic Hann window (computed in f64 with realfft).

use std::fmt;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::error::{GpuError, Result};

/// One-sided complex spectrogram, row-major `[frame][bin]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrogram {
    pub frames: usize,
    pub bins: usize,
    pub values: Vec<Complex<f64>>,
}

impl Spectrogram {
    pub fn at(&self, frame: usize, bin: usize) -> Complex<f64> {
        self.values[frame * self.bins + bin]
    }
}

#[derive(Clone)]
pub struct Stft {
    n_fft: usize,
    hop: usize,
    window: Vec<f64>,
    forward: Arc<dyn RealToComplex<f64>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
}

impl fmt::Debug for Stft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stft")
            .field("n_fft", &self.n_fft)
            .field("hop", &self.hop)
            .finish_non_exhaustive()
    }
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let window = (0..n_fft)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / n_fft as f64).cos())
            .collect();
        Self {
            n_fft,
            hop,
            window,
            forward: planner.plan_fft_forward(n_fft),
            inverse: planner.plan_fft_inverse(n_fft),
        }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Frames produced for a signal of `len` samples.
    pub fn frames(&self, len: usize) -> usize {
        1 + len / self.hop
    }

    /// `torch.stft(x, n_fft, hop, window=hann(n_fft), center=True, pad_mode="reflect")`.
    pub fn forward(&self, signal: &[f32]) -> Result<Spectrogram> {
        let half = self.n_fft / 2;
        if signal.len() <= half {
            return Err(GpuError::Input(format!(
                "STFT input of {} samples is too short for reflect padding",
                signal.len()
            )));
        }
        let n = signal.len() as isize;
        let reflected = |i: isize| -> f64 {
            let j = if i < 0 {
                -i
            } else if i >= n {
                2 * (n - 1) - i
            } else {
                i
            };
            f64::from(signal[j as usize])
        };
        let frames = self.frames(signal.len());
        let bins = self.bins();
        let mut input = self.forward.make_input_vec();
        let mut spectrum = self.forward.make_output_vec();
        let mut values = Vec::with_capacity(frames * bins);
        for frame in 0..frames {
            let start = (frame * self.hop) as isize - half as isize;
            for (k, slot) in input.iter_mut().enumerate() {
                *slot = reflected(start + k as isize) * self.window[k];
            }
            self.forward
                .process(&mut input, &mut spectrum)
                .map_err(|err| GpuError::Input(format!("STFT failed: {err}")))?;
            values.extend_from_slice(&spectrum);
        }
        Ok(Spectrogram { frames, bins, values })
    }

    /// `torch.istft(spec, n_fft, hop, window=hann(n_fft), center=True)` with the default length.
    pub fn inverse(&self, spec: &Spectrogram) -> Result<Vec<f32>> {
        if spec.bins != self.bins() || spec.frames == 0 || spec.values.len() != spec.frames * spec.bins {
            return Err(GpuError::Input(
                "ISTFT spectrogram geometry does not match the transform".to_owned(),
            ));
        }
        let total = self.n_fft + self.hop * (spec.frames - 1);
        let mut signal = vec![0.0f64; total];
        let mut envelope = vec![0.0f64; total];
        let mut bins = self.inverse.make_input_vec();
        let mut frame_out = self.inverse.make_output_vec();
        let scale = 1.0 / self.n_fft as f64;
        let last = bins.len() - 1;
        for frame in 0..spec.frames {
            bins.copy_from_slice(&spec.values[frame * spec.bins..(frame + 1) * spec.bins]);
            bins[0].im = 0.0;
            if self.n_fft.is_multiple_of(2) {
                bins[last].im = 0.0;
            }
            self.inverse
                .process(&mut bins, &mut frame_out)
                .map_err(|err| GpuError::Input(format!("ISTFT failed: {err}")))?;
            let offset = frame * self.hop;
            for (k, &value) in frame_out.iter().enumerate() {
                signal[offset + k] += value * scale * self.window[k];
                envelope[offset + k] += self.window[k] * self.window[k];
            }
        }
        let start = self.n_fft / 2;
        let end = total - self.n_fft / 2;
        signal[start..end]
            .iter()
            .zip(&envelope[start..end])
            .map(|(&s, &e)| {
                if e > 1e-11 {
                    Ok((s / e) as f32)
                } else {
                    Err(GpuError::Input("ISTFT window envelope vanishes".to_owned()))
                }
            })
            .collect()
    }
}

/// `scipy.signal.resample_poly(x, up, down)` with its default Kaiser(5.0) anti-aliasing FIR.
pub fn resample_poly(x: &[f32], up: usize, down: usize) -> Vec<f32> {
    let divisor = gcd(up, down);
    let (up, down) = (up / divisor, down / divisor);
    if up == down {
        return x.to_vec();
    }
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let taps = firwin_kaiser(2 * half_len + 1, 1.0 / max_rate as f64, 5.0);
    let pre_pad = down - half_len % down;
    let pre_remove = (half_len + pre_pad) / down;
    let mut filter = vec![0.0f64; pre_pad];
    filter.extend(taps.iter().map(|t| t * up as f64));
    let out_len = (x.len() * up).div_ceil(down);
    (0..out_len)
        .map(|o| {
            let k = (o + pre_remove) * down;
            let first = (k + 1).saturating_sub(filter.len()).div_ceil(up);
            let last = (k / up).min(x.len().saturating_sub(1));
            (first..=last)
                .map(|i| f64::from(x[i]) * filter[k - i * up])
                .sum::<f64>() as f32
        })
        .collect()
}

/// `scipy.signal.firwin(numtaps, cutoff, window=("kaiser", beta))`, low-pass, DC-normalized.
fn firwin_kaiser(numtaps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let alpha = (numtaps - 1) as f64 / 2.0;
    let sinc = |x: f64| {
        if x == 0.0 {
            1.0
        } else {
            (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
        }
    };
    let denom = bessel_i0(beta);
    let taps: Vec<f64> = (0..numtaps)
        .map(|n| {
            let m = n as f64 - alpha;
            let ratio = m / alpha;
            let window = bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / denom;
            cutoff * sinc(cutoff * m) * window
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    taps.into_iter().map(|t| t / sum).collect()
}

fn bessel_i0(x: f64) -> f64 {
    let quarter = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..64 {
        term *= quarter / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_poly_doubles_length_and_keeps_tones() {
        let x: Vec<f32> = (0..1600)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let y = resample_poly(&x, 2, 1);
        assert_eq!(y.len(), 3200);
        let expected = (2.0 * std::f64::consts::PI * 440.0 * 1000.0 / 32_000.0).sin();
        assert!((f64::from(y[1000]) - expected).abs() < 1e-3);
        assert!((bessel_i0(5.0) - 27.239_871_823_604_45).abs() < 1e-9);
    }

    #[test]
    fn istft_inverts_stft() {
        let stft = Stft::new(1000, 400);
        let signal: Vec<f32> = (0..16_000).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
        let spec = stft.forward(&signal).unwrap();
        assert_eq!((spec.frames, spec.bins), (41, 501));
        let back = stft.inverse(&spec).unwrap();
        assert_eq!(back.len(), signal.len());
        let worst = back
            .iter()
            .zip(&signal)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "{worst}");
    }
}

//! Framed real FFT magnitudes (`sliding_window_view(x, n_fft)[::hop]`, windowed, `np.abs(np.fft.rfft(...))`).

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

use super::window::hann_periodic;

pub struct StftPlan {
    n_fft: usize,
    hop: usize,
    window: Vec<f64>,
    fft: Arc<dyn RealToComplex<f64>>,
}

impl std::fmt::Debug for StftPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StftPlan")
            .field("n_fft", &self.n_fft)
            .field("hop", &self.hop)
            .finish_non_exhaustive()
    }
}

/// Per-thread buffers for [`StftPlan::magnitudes`].
#[derive(Debug)]
pub struct StftScratch {
    input: Vec<f64>,
    spectrum: Vec<realfft::num_complex::Complex<f64>>,
    work: Vec<realfft::num_complex::Complex<f64>>,
}

impl StftPlan {
    /// A periodic-Hann STFT.
    pub fn hann(n_fft: usize, hop: usize) -> Self {
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(n_fft);
        Self {
            n_fft,
            hop,
            window: hann_periodic(n_fft),
            fft,
        }
    }

    pub const fn n_fft(&self) -> usize {
        self.n_fft
    }

    pub const fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Frames over `len` samples, after zero-padding a short signal to one frame.
    pub const fn frame_count(&self, len: usize) -> usize {
        let len = if len < self.n_fft { self.n_fft } else { len };
        (len - self.n_fft) / self.hop + 1
    }

    pub fn scratch(&self) -> StftScratch {
        StftScratch {
            input: self.fft.make_input_vec(),
            spectrum: self.fft.make_output_vec(),
            work: self.fft.make_scratch_vec(),
        }
    }

    /// Magnitudes of frame `index` of `signal` into `out` (length [`Self::bins`]).
    pub fn magnitudes(&self, signal: &[f64], index: usize, scratch: &mut StftScratch, out: &mut [f64]) {
        let start = index * self.hop;
        for (i, (slot, w)) in scratch.input.iter_mut().zip(&self.window).enumerate() {
            *slot = signal.get(start + i).copied().unwrap_or(0.0) * w;
        }
        self.fft
            .process_with_scratch(&mut scratch.input, &mut scratch.spectrum, &mut scratch.work)
            .expect("buffers come from the same plan");
        for (slot, c) in out.iter_mut().zip(&scratch.spectrum) {
            *slot = c.re.hypot(c.im);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_signals_yield_one_frame() {
        let plan = StftPlan::hann(512, 128);
        assert_eq!(plan.frame_count(10), 1);
        assert_eq!(plan.frame_count(512 + 128 * 3), 4);
    }

    #[test]
    fn cosine_peaks_in_its_bin() {
        let plan = StftPlan::hann(64, 16);
        let signal: Vec<f64> = (0..64)
            .map(|n| (2.0 * std::f64::consts::PI * 4.0 * n as f64 / 64.0).cos())
            .collect();
        let mut scratch = plan.scratch();
        let mut out = vec![0.0; plan.bins()];
        plan.magnitudes(&signal, 0, &mut scratch, &mut out);
        let peak = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(peak, 4);
    }
}

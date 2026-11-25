//! 2D FFT helpers and `cv::phaseCorrelate`.

use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};

use crate::element::{Depth, Element};
use crate::error::{CvError, Result};
use crate::mat::MatView;
use crate::types::Point2d;

/// `cv::getOptimalDFTSize`: the smallest `2^a 3^b 5^c >= n`.
pub fn optimal_dft_size(n: usize) -> usize {
    let mut best = usize::MAX;
    let mut p2 = 1usize;
    while p2 < 2 * n.max(1) {
        let mut p3 = p2;
        while p3 < 2 * n.max(1) {
            let mut p5 = p3;
            while p5 < n {
                p5 *= 5;
            }
            best = best.min(p5);
            p3 *= 3;
        }
        p2 *= 2;
    }
    best.max(1)
}

/// A complex `rows x cols` buffer with row-major storage.
pub(crate) struct Spectrum {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Complex64>,
}

impl Spectrum {
    pub fn from_real<T: Element>(src: &MatView<'_, T>, rows: usize, cols: usize) -> Self {
        let mut data = vec![Complex64::new(0.0, 0.0); rows * cols];
        for y in 0..src.height() {
            for (x, v) in src.row(y).iter().enumerate() {
                data[y * cols + x] = Complex64::new(v.to_f64(), 0.0);
            }
        }
        Self { rows, cols, data }
    }

    pub fn transform(&mut self, inverse: bool) {
        let mut planner = FftPlanner::<f64>::new();
        let (row, col): (std::sync::Arc<dyn Fft<f64>>, std::sync::Arc<dyn Fft<f64>>) = if inverse {
            (planner.plan_fft_inverse(self.cols), planner.plan_fft_inverse(self.rows))
        } else {
            (planner.plan_fft_forward(self.cols), planner.plan_fft_forward(self.rows))
        };
        self.data.par_chunks_mut(self.cols).for_each(|r| row.process(r));
        let (rows, cols) = (self.rows, self.cols);
        let mut columns: Vec<Vec<Complex64>> = (0..cols).map(|x| (0..rows).map(|y| self.data[y * cols + x]).collect()).collect();
        columns.par_iter_mut().for_each(|c| col.process(c));
        for (x, c) in columns.iter().enumerate() {
            for (y, v) in c.iter().enumerate() {
                self.data[y * cols + x] = *v;
            }
        }
    }
}

/// `cv::phaseCorrelate(a, b)`: the sub-pixel shift of `b` relative to `a` and the peak response.
pub fn phase_correlate<'a, 'b, T: Element>(a: impl Into<MatView<'a, T>>, b: impl Into<MatView<'b, T>>) -> Result<(Point2d, f64)> {
    let (a, b) = (a.into(), b.into());
    if a.size() != b.size() || a.channels() != 1 || b.channels() != 1 || a.is_empty() {
        return Err(CvError::ShapeMismatch { op: "phase_correlate" });
    }
    let (m, n) = (optimal_dft_size(a.height()), optimal_dft_size(a.width()));
    let mut fa = Spectrum::from_real(&a, m, n);
    let mut fb = Spectrum::from_real(&b, m, n);
    fa.transform(false);
    fb.transform(false);
    let eps = if T::DEPTH == Depth::F32 { f64::from(f32::EPSILON) } else { f64::EPSILON };
    fa.data.par_iter_mut().zip(fb.data.par_iter()).for_each(|(p, q)| {
        let prod = *p * q.conj();
        let mag = prod.norm();
        *p = prod * (mag / (mag * mag + eps));
    });
    fa.transform(true);
    let mut c: Vec<f64> = fa.data.iter().map(|v| v.re).collect();
    fft_shift(&mut c, m, n);
    let mut peak = (0usize, 0usize);
    let mut best = f64::NEG_INFINITY;
    for (i, &v) in c.iter().enumerate() {
        if v > best {
            best = v;
            peak = (i % n, i / n);
        }
    }
    let (minr, maxr) = (peak.1.saturating_sub(2), (peak.1 + 2).min(m - 1));
    let (minc, maxc) = (peak.0.saturating_sub(2), (peak.0 + 2).min(n - 1));
    let (mut cx, mut cy, mut sum) = (0.0, 0.0, 0.0);
    for y in minr..=maxr {
        for x in minc..=maxc {
            let v = c[y * n + x];
            cx += x as f64 * v;
            cy += y as f64 * v;
            sum += v;
        }
    }
    let response = sum / (m * n) as f64;
    let sum = sum + f64::EPSILON;
    let shift = Point2d::new(n as f64 / 2.0 - cx / sum, m as f64 / 2.0 - cy / sum);
    Ok((shift, response))
}

/// Swap quadrants so the zero frequency sits at the center (`fftShift` in `phasecorr.cpp`).
fn fft_shift(c: &mut [f64], rows: usize, cols: usize) {
    if rows == 1 && cols == 1 {
        return;
    }
    let src = c.to_vec();
    let (ym, xm) = (rows / 2, cols / 2);
    for y in 0..rows {
        for x in 0..cols {
            c[((y + ym) % rows) * cols + (x + xm) % cols] = src[y * cols + x];
        }
    }
}

//! Window functions and the windowed-sinc FIR design used by `scipy.signal.resample_poly`.

use std::f64::consts::PI;

use super::numpy;

/// Modified Bessel function of the first kind, order zero (power series, ~1 ulp for |x| <= 30).
pub fn bessel_i0(x: f64) -> f64 {
    let quarter = 0.25 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut compensation = 0.0;
    for k in 1..500 {
        let kf = f64::from(k);
        term *= quarter / (kf * kf);
        let y = term - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
        if term < sum * 1e-18 {
            break;
        }
    }
    sum
}

/// `scipy.signal.windows.kaiser(m, beta, sym=True)`.
pub fn kaiser(m: usize, beta: f64) -> Vec<f64> {
    if m <= 1 {
        return vec![1.0; m];
    }
    let alpha = (m - 1) as f64 / 2.0;
    let denom = bessel_i0(beta);
    (0..m)
        .map(|n| {
            let r = (n as f64 - alpha) / alpha;
            bessel_i0(beta * (1.0 - r * r).sqrt()) / denom
        })
        .collect()
}

/// `np.sinc`.
fn sinc(x: f64) -> f64 {
    let y = PI * if x == 0.0 { 1.0e-20 } else { x };
    y.sin() / y
}

/// `scipy.signal.firwin(numtaps, cutoff, window=("kaiser", beta))` for a low-pass with unit DC gain.
pub fn firwin_kaiser(numtaps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    let window = kaiser(numtaps, beta);
    let h: Vec<f64> = (0..numtaps)
        .zip(&window)
        .map(|(i, w)| {
            let m = i as f64 - alpha;
            (cutoff * sinc(cutoff * m) - 0.0 * sinc(0.0 * m)) * w
        })
        .collect();
    let scale = numpy::sum(&h);
    h.into_iter().map(|v| v / scale).collect()
}

/// The periodic Hann window `0.5 - 0.5 cos(2 pi n / n_fft)`.
pub fn hann_periodic(n_fft: usize) -> Vec<f64> {
    (0..n_fft)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / n_fft as f64).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_matches_reference_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-16);
        assert!((bessel_i0(5.0) - 27.239_871_823_604_45).abs() < 1e-12);
    }

    #[test]
    fn firwin_has_unit_dc_gain() {
        let h = firwin_kaiser(61, 1.0 / 3.0, 5.0);
        assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }
}

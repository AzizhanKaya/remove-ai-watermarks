//! `scipy.signal.butter(..., output="sos", fs=...)`: analog prototype, band transform, bilinear map, `zpk2sos`.

use std::cmp::Ordering;
use std::f64::consts::PI;

use super::complex::{C64, prod};

/// One biquad `[b0, b1, b2, 1, a1, a2]`.
pub type Section = [f64; 6];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Band {
    Lowpass(f64),
    Highpass(f64),
    Bandstop(f64, f64),
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DesignError {
    #[error("digital filter critical frequencies must be 0 < Wn < fs/2")]
    CriticalFrequency,
    #[error("filter order must be positive")]
    Order,
}

/// Butterworth second-order sections, as `butter(order, band, btype, fs=sample_rate, output="sos")`.
pub fn butter_sos(order: usize, band: Band, sample_rate: f64) -> Result<Vec<Section>, DesignError> {
    if order == 0 {
        return Err(DesignError::Order);
    }
    let normalize = |f: f64| f / (sample_rate / 2.0);
    let warp = |wn: f64| -> Result<f64, DesignError> {
        if wn <= 0.0 || wn >= 1.0 {
            return Err(DesignError::CriticalFrequency);
        }
        Ok(2.0 * 2.0 * (PI * wn / 2.0).tan())
    };
    let poles = buttap(order);
    let (zeros, poles, gain) = match band {
        Band::Lowpass(f) => lp2lp(&poles, warp(normalize(f))?),
        Band::Highpass(f) => lp2hp(&poles, warp(normalize(f))?),
        Band::Bandstop(lo, hi) => {
            let (w0, w1) = (warp(normalize(lo))?, warp(normalize(hi))?);
            lp2bs(&poles, (w0 * w1).sqrt(), w1 - w0)
        }
    };
    let (zeros, poles, gain) = bilinear(&zeros, &poles, gain, 2.0);
    Ok(zpk2sos(zeros, poles, gain))
}

fn buttap(order: usize) -> Vec<C64> {
    let n = order as i64;
    let scale = C64::real((2 * n) as f64);
    (0..n)
        .map(|i| {
            let m = (-n + 1 + 2 * i) as f64;
            let theta = C64::new(0.0, PI * m).quot(scale);
            -C64::cis(theta.im)
        })
        .collect()
}

type Zpk = (Vec<C64>, Vec<C64>, f64);

fn lp2lp(poles: &[C64], wo: f64) -> Zpk {
    let degree = poles.len() as f64;
    let poles = poles.iter().map(|&p| C64::real(wo) * p).collect();
    (Vec::new(), poles, wo.powf(degree))
}

fn lp2hp(poles: &[C64], wo: f64) -> Zpk {
    let gain = C64::real(1.0)
        .quot(prod(&poles.iter().map(|&p| -p).collect::<Vec<_>>()))
        .re;
    let mapped = poles.iter().map(|&p| C64::real(wo).quot(p)).collect();
    (vec![C64::real(0.0); poles.len()], mapped, gain)
}

fn lp2bs(poles: &[C64], wo: f64, bw: f64) -> Zpk {
    let degree = poles.len();
    let half = C64::real(bw / 2.0);
    let wo2 = wo.powf(2.0);
    let hp: Vec<C64> = poles.iter().map(|&p| half.quot(p)).collect();
    let shifted = |p: C64| (p.square() - C64::real(wo2)).sqrt();
    let mut mapped: Vec<C64> = hp.iter().map(|&p| p + shifted(p)).collect();
    mapped.extend(hp.iter().map(|&p| p - shifted(p)));
    let mut zeros = vec![C64::new(0.0, wo); degree];
    zeros.extend(std::iter::repeat_n(C64::new(0.0, -wo), degree));
    let gain = C64::real(1.0)
        .quot(prod(&poles.iter().map(|&p| -p).collect::<Vec<_>>()))
        .re;
    (zeros, mapped, gain)
}

fn bilinear(zeros: &[C64], poles: &[C64], gain: f64, fs: f64) -> Zpk {
    let fs2 = C64::real(2.0 * fs);
    let degree = poles.len() - zeros.len();
    let mut mapped_zeros: Vec<C64> = zeros.iter().map(|&z| (fs2 + z).quot(fs2 - z)).collect();
    let mapped_poles = poles.iter().map(|&p| (fs2 + p).quot(fs2 - p)).collect();
    mapped_zeros.extend(std::iter::repeat_n(C64::real(-1.0), degree));
    let num = prod(&zeros.iter().map(|&z| fs2 - z).collect::<Vec<_>>());
    let den = prod(&poles.iter().map(|&p| fs2 - p).collect::<Vec<_>>());
    (mapped_zeros, mapped_poles, gain * num.quot(den).re)
}

/// `_cplxreal`: averaged positive-imaginary representatives of conjugate pairs, then reals.
fn cplxreal(values: &[C64]) -> (Vec<C64>, Vec<f64>) {
    let tol = 100.0 * f64::EPSILON;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| {
        a.re.partial_cmp(&b.re)
            .unwrap_or(Ordering::Equal)
            .then(a.im.abs().partial_cmp(&b.im.abs()).unwrap_or(Ordering::Equal))
    });
    let is_real = |z: &C64| z.im.abs() <= tol * z.abs();
    let reals: Vec<f64> = sorted.iter().filter(|z| is_real(z)).map(|z| z.re).collect();
    let complex: Vec<C64> = sorted.into_iter().filter(|z| !is_real(z)).collect();
    let mut positive: Vec<C64> = complex.iter().copied().filter(|z| z.im > 0.0).collect();
    let mut negative: Vec<C64> = complex.iter().copied().filter(|z| z.im < 0.0).collect();
    let n = positive.len().min(negative.len());
    let mut start = 0;
    while start < positive.len() {
        let mut stop = start;
        while stop + 1 < positive.len()
            && positive[stop + 1].re - positive[stop].re <= tol * positive[stop].abs()
        {
            stop += 1;
        }
        if stop > start {
            let by_imag = |a: &C64, b: &C64| a.im.abs().partial_cmp(&b.im.abs()).unwrap_or(Ordering::Equal);
            positive[start..=stop].sort_by(by_imag);
            if stop < negative.len() {
                negative[start..=stop].sort_by(by_imag);
            }
        }
        start = stop + 1;
    }
    let averaged = (0..n)
        .map(|i| (positive[i] + negative[i].conj()).quot(C64::real(2.0)))
        .collect();
    (averaged, reals)
}

/// `np.poly` on at most two roots, with `zpk2tf`'s conjugate-pair real cast.
fn poly2(roots: &[C64]) -> Vec<f64> {
    match roots {
        [] => vec![1.0],
        [r] => vec![1.0, -r.re],
        [a, b] => {
            let (na, nb) = (-*a, -*b);
            let mid = (nb.re * 1.0 - 0.0 * nb.im) + (na.re * 1.0 - na.im * 0.0);
            let last = na * nb;
            vec![1.0, mid, last.re]
        }
        _ => unreachable!("second-order sections hold at most two roots"),
    }
}

fn single_section(zeros: &[C64], poles: &[C64]) -> Section {
    let b = poly2(zeros);
    let a = poly2(poles);
    let mut sos = [0.0; 6];
    sos[3 - b.len()..3].copy_from_slice(&b);
    sos[6 - a.len()..6].copy_from_slice(&a);
    sos
}

fn nearest(from: &[C64], to: C64, want: Option<bool>) -> usize {
    let mut order: Vec<usize> = (0..from.len()).collect();
    order.sort_by(|&i, &j| {
        (from[i] - to)
            .abs()
            .partial_cmp(&(from[j] - to).abs())
            .unwrap_or(Ordering::Equal)
    });
    order
        .into_iter()
        .find(|&i| want.is_none_or(|real| from[i].is_real() == real))
        .unwrap_or(0)
}

fn worst(values: &[C64]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if (1.0 - v.abs()).abs() < (1.0 - values[best].abs()).abs() {
            best = i;
        }
    }
    best
}

/// `zpk2sos(z, p, k, pairing="nearest")` for a digital filter.
fn zpk2sos(zeros: Vec<C64>, poles: Vec<C64>, gain: f64) -> Vec<Section> {
    let mut z = zeros;
    let mut p = poles;
    while p.len() < z.len() {
        p.push(C64::real(0.0));
    }
    while z.len() < p.len() {
        z.push(C64::real(0.0));
    }
    let n_sections = p.len().div_ceil(2);
    if p.len() % 2 == 1 {
        p.push(C64::real(0.0));
        z.push(C64::real(0.0));
    }
    let concat = |(pairs, reals): (Vec<C64>, Vec<f64>)| {
        pairs
            .into_iter()
            .chain(reals.into_iter().map(C64::real))
            .collect::<Vec<_>>()
    };
    let mut z = concat(cplxreal(&z));
    let mut p = concat(cplxreal(&p));
    let mut sos = vec![[0.0; 6]; n_sections];
    for si in (0..n_sections).rev() {
        let p1 = p.remove(worst(&p));
        let reals_left = p.iter().filter(|v| v.is_real()).count();
        if p1.is_real() && reals_left == 0 {
            let z1 = z.remove(nearest(&z, p1, Some(true)));
            sos[si] = single_section(&[z1, C64::real(0.0)], &[p1, C64::real(0.0)]);
        } else if p.len() + 1 == z.len()
            && !p1.is_real()
            && reals_left == 1
            && z.iter().filter(|v| v.is_real()).count() == 1
        {
            let z1 = z.remove(nearest(&z, p1, Some(false)));
            sos[si] = single_section(&[z1, z1.conj()], &[p1, p1.conj()]);
        } else {
            let p2 = if p1.is_real() {
                let real_idx: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_real()).collect();
                let candidates: Vec<C64> = real_idx.iter().map(|&i| p[i]).collect();
                p.remove(real_idx[worst(&candidates)])
            } else {
                p1.conj()
            };
            if z.is_empty() {
                sos[si] = single_section(&[], &[p1, p2]);
            } else {
                let z1 = z.remove(nearest(&z, p1, None));
                if !z1.is_real() {
                    sos[si] = single_section(&[z1, z1.conj()], &[p1, p2]);
                } else if z.is_empty() {
                    sos[si] = single_section(&[z1], &[p1, p2]);
                } else {
                    let z2 = z.remove(nearest(&z, p1, Some(true)));
                    sos[si] = single_section(&[z1, z2], &[p1, p2]);
                }
            }
        }
    }
    for value in &mut sos[0][..3] {
        *value *= gain;
    }
    sos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandstop_has_four_normalized_sections() {
        let sos = butter_sos(4, Band::Bandstop(1000.0, 1500.0), 48000.0).unwrap();
        assert_eq!(sos.len(), 4);
        assert!(sos.iter().all(|s| s[3] == 1.0));
    }

    #[test]
    fn rejects_frequencies_at_nyquist() {
        assert!(butter_sos(4, Band::Lowpass(8000.0), 16000.0).is_err());
    }
}

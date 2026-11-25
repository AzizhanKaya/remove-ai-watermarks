//! `scipy.signal.sosfilt`, `sosfilt_zi` and `sosfiltfilt(padtype="odd")` in SciPy's evaluation order.

use super::iir::Section;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FilterError {
    #[error("The length of the input vector x must be greater than padlen, which is {0}.")]
    TooShort(usize),
}

/// Samples whose odd extension is computed in their own precision before widening to float64.
pub trait FilterSample: Copy + Send + Sync {
    fn widen(self) -> f64;
    /// `2 * edge - value` in the sample's dtype.
    fn reflect(edge: Self, value: Self) -> f64;
}

impl FilterSample for f32 {
    fn widen(self) -> f64 {
        f64::from(self)
    }
    fn reflect(edge: Self, value: Self) -> f64 {
        f64::from(2.0 * edge - value)
    }
}

impl FilterSample for f64 {
    fn widen(self) -> f64 {
        self
    }
    fn reflect(edge: Self, value: Self) -> f64 {
        2.0 * edge - value
    }
}

/// Direct-form II transposed cascade, updating `state` in place.
pub fn sosfilt_in_place(sos: &[Section], x: &mut [f64], state: &mut [[f64; 2]]) {
    cascade(sos, x.iter_mut(), state);
}

fn cascade<'a>(sos: &[Section], samples: impl Iterator<Item = &'a mut f64>, state: &mut [[f64; 2]]) {
    for sample in samples {
        let mut cur = *sample;
        for (s, z) in sos.iter().zip(state.iter_mut()) {
            let new = s[0] * cur + z[0];
            z[0] = s[1] * cur - s[4] * new + z[1];
            z[1] = s[2] * cur - s[5] * new;
            cur = new;
        }
        *sample = cur;
    }
}

/// `sosfilt_zi`: steady-state initial conditions for a unit step.
pub fn sosfilt_zi(sos: &[Section]) -> Vec<[f64; 2]> {
    let mut scale = 1.0;
    sos.iter()
        .map(|s| {
            let zi = lfilter_zi(s);
            let section = [scale * zi[0], scale * zi[1]];
            scale *= ((s[0] + s[1]) + s[2]) / ((s[3] + s[4]) + s[5]);
            section
        })
        .collect()
}

/// `lfilter_zi` for one normalized biquad (SciPy 1.18 closed form): with the unit-step limit
/// `y = sum(b) / sum(a)` and `d = b - y a`, `zi = [d2 + d1, d2]`.
fn lfilter_zi(s: &Section) -> [f64; 2] {
    let y_inf = ((s[0] + s[1]) + s[2]) / ((s[3] + s[4]) + s[5]);
    let d1 = s[1] - y_inf * s[4];
    let d2 = s[2] - y_inf * s[5];
    [d2 + d1, d2]
}

/// Number of taps that `sosfiltfilt` derives its default pad from.
fn ntaps(sos: &[Section]) -> usize {
    let zero_b2 = sos.iter().filter(|s| s[2] == 0.0).count();
    let zero_a2 = sos.iter().filter(|s| s[5] == 0.0).count();
    2 * sos.len() + 1 - zero_b2.min(zero_a2)
}

/// Zero-phase forward-backward filtering of one channel with odd extension.
pub fn sosfiltfilt<T: FilterSample>(
    sos: &[Section],
    x: &[T],
    padlen: Option<usize>,
) -> Result<Vec<f64>, FilterError> {
    let edge = padlen.unwrap_or(3 * ntaps(sos));
    let n = x.len();
    if n <= edge {
        return Err(FilterError::TooShort(edge));
    }
    let mut ext = Vec::with_capacity(n + 2 * edge);
    if edge > 0 {
        let first = x[0];
        ext.extend((1..=edge).rev().map(|k| T::reflect(first, x[k])));
    }
    ext.extend(x.iter().map(|v| v.widen()));
    if edge > 0 {
        let last = x[n - 1];
        ext.extend((1..=edge).map(|k| T::reflect(last, x[n - 1 - k])));
    }
    let zi = sosfilt_zi(sos);
    let scaled = |v: f64| zi.iter().map(|z| [z[0] * v, z[1] * v]).collect::<Vec<_>>();
    let mut state = scaled(ext[0]);
    cascade(sos, ext.iter_mut(), &mut state);
    let mut state = scaled(ext[ext.len() - 1]);
    cascade(sos, ext.iter_mut().rev(), &mut state);
    Ok(ext[edge..ext.len() - edge].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::iir::{Band, butter_sos};

    #[test]
    fn steady_state_matches_dc_gain() {
        let sos = butter_sos(4, Band::Lowpass(1000.0), 48000.0).unwrap();
        let mut x = vec![1.0; 64];
        let mut state = sosfilt_zi(&sos);
        sosfilt_in_place(&sos, &mut x, &mut state);
        assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-9));
    }

    #[test]
    fn short_input_is_rejected() {
        let sos = butter_sos(4, Band::Lowpass(1000.0), 48000.0).unwrap();
        assert_eq!(
            sosfiltfilt(&sos, &[0.0f32; 10], None),
            Err(FilterError::TooShort(15))
        );
        let single = sosfiltfilt(&sos, &[0.5f32; 1], Some(0)).unwrap();
        assert!(single.len() == 1 && (single[0] - 0.5).abs() < 1e-12);
    }
}

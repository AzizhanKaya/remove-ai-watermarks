//! The handful of NumPy reductions the engines rely on, with NumPy's conventions.

/// `np.percentile(values, q)` with the default linear method; `values` is sorted in place.
pub fn percentile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let index = q / 100.0 * (values.len() - 1) as f64;
    let lo = index.floor() as usize;
    let hi = (lo + 1).min(values.len() - 1);
    let t = index - lo as f64;
    let (a, b) = (values[lo], values[hi]);
    let diff = b - a;
    if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
}

/// `np.median` of `float32` samples, averaging the middle pair in `float32`.
pub fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    let n = values.len();
    let mid = n / 2;
    let (_, upper, _) = values.select_nth_unstable_by(mid, f32::total_cmp);
    let upper = *upper;
    if n % 2 == 1 {
        return upper;
    }
    let lower = values[..mid].iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (lower + upper) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_linear() {
        let mut v = vec![4.0, 1.0, 3.0, 2.0];
        assert_eq!(percentile(&mut v, 25.0), 1.75);
        assert!((percentile(&mut v, 10.0) - 1.3).abs() < 1e-12);
        assert_eq!(percentile(&mut v, 100.0), 4.0);
    }

    #[test]
    fn median_even_and_odd() {
        assert_eq!(median_f32(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_f32(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}

//! Python numeric semantics the port must reproduce exactly.

/// `round(x)`: round half to even, as an integer.
pub fn round_int(value: f64) -> i64 {
    value.round_ties_even() as i64
}

/// `round(x, ndigits)`: the double nearest the correctly rounded decimal.
pub fn round_to(value: f64, ndigits: usize) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.ndigits$}").parse().unwrap_or(value)
}

/// Python's `f"{value:.{precision}g}"`.
pub fn format_g(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".into()
        } else if value > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let precision = precision.max(1);
    let sci = format!("{:.*e}", precision - 1, value);
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if exponent < -4 || exponent >= precision as i32 {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim_fraction(mantissa), exponent.abs())
    } else {
        let decimals = (precision as i32 - 1 - exponent).max(0) as usize;
        trim_fraction(&format!("{value:.decimals$}")).to_owned()
    }
}

fn trim_fraction(text: &str) -> &str {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text
    }
}

/// `np.linspace(0, n - 1, count, dtype=int)`: truncated evenly spaced indices.
pub fn linspace_indices(n: usize, count: usize) -> Vec<usize> {
    if count == 0 || n == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![0];
    }
    let stop = (n - 1) as f64;
    let step = stop / (count - 1) as f64;
    (0..count)
        .map(|i| {
            if i == count - 1 {
                n - 1
            } else {
                (i as f64 * step) as usize
            }
        })
        .collect()
}

/// `np.median` over a copy of `values` (mean of the two middle values when even).
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        f64::midpoint(values[mid - 1], values[mid])
    }
}

/// `np.median` of float32 data, computed in float32 like numpy.
pub fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        f32::midpoint(values[mid - 1], values[mid])
    }
}

/// `CPython`'s iteration order over a set of small non-negative ints (hash = value), for the
/// tie-breaking loops that iterate a `set`.
pub fn cpython_int_set_order(values: &[i64]) -> Vec<i64> {
    let mut unique: Vec<i64> = Vec::new();
    for &v in values {
        if !unique.contains(&v) {
            unique.push(v);
        }
    }
    let mut mask: u64 = 7;
    while (unique.len() as u64) * 5 >= mask * 3 {
        mask = (mask + 1) * 4 - 1;
    }
    let mut table: Vec<Option<i64>> = vec![None; (mask + 1) as usize];
    for &value in &unique {
        let hash = value as u64;
        let mut perturb = hash;
        let mut i = hash & mask;
        loop {
            let mut placed = false;
            if table[i as usize].is_none() {
                table[i as usize] = Some(value);
                break;
            }
            if i + 9 <= mask {
                for j in 1..=9 {
                    if table[(i + j) as usize].is_none() {
                        table[(i + j) as usize] = Some(value);
                        placed = true;
                        break;
                    }
                }
            }
            if placed {
                break;
            }
            perturb >>= 5;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }
    table.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_format_matches_python() {
        assert_eq!(format_g(24.0, 12), "24");
        assert_eq!(format_g(29.970_029_97, 12), "29.97002997");
        assert_eq!(format_g(30000.0 / 1001.0, 12), "29.97002997");
        assert_eq!(format_g(1e-7, 12), "1e-07");
        assert_eq!(format_g(123_456_789_012_345.0, 12), "1.23456789012e+14");
    }

    #[test]
    fn rounding_is_bankers() {
        assert_eq!(round_int(2.5), 2);
        assert_eq!(round_int(3.5), 4);
        assert_eq!(round_to(2.675, 2), 2.67);
        assert_eq!(round_to(26.785_04, 4), 26.785);
    }

    #[test]
    fn linspace_truncates() {
        assert_eq!(linspace_indices(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(linspace_indices(100, 3), vec![0, 49, 99]);
    }

    #[test]
    fn small_int_sets_iterate_by_hash_slot() {
        assert_eq!(cpython_int_set_order(&[56, 48, 44]), vec![56, 48, 44]);
        assert_eq!(cpython_int_set_order(&[42, 36, 33]), vec![33, 42, 36]);
        assert_eq!(cpython_int_set_order(&[16, 17, 18]), vec![16, 17, 18]);
    }
}

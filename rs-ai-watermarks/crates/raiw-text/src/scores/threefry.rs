//! Scalar Threefry2x32 bits and repetition mask of the open SynthID-Text reference.
//!
//! Algorithm reference: Google DeepMind synthid-text at 72810caf309d51f5985dc871a4bce42c3d0cb54d.

use std::collections::{HashMap, VecDeque};

use num_bigint::BigUint;

const PARITY: u32 = 0x1BD1_1BDA;
const ROTATIONS: [[u32; 4]; 2] = [[13, 15, 26, 6], [17, 29, 16, 24]];

/// The twenty-round Random123 Threefry2x32 block function.
pub fn threefry2x32(key: (u32, u32), first: u32, second: u32) -> (u32, u32) {
    let (left, right) = key;
    let words = [left, right, left ^ right ^ PARITY];
    let mut first = first.wrapping_add(left);
    let mut second = second.wrapping_add(right);
    for group in 0..5u32 {
        for &shift in &ROTATIONS[(group % 2) as usize] {
            first = first.wrapping_add(second);
            second = first ^ second.rotate_left(shift);
        }
        let injection = group + 1;
        first = first.wrapping_add(words[(injection % 3) as usize]);
        second = second
            .wrapping_add(words[((injection + 1) % 3) as usize])
            .wrapping_add(injection);
    }
    (first, second)
}

/// One bits and scored positions over an EOS-free continuation.
pub fn count_g_values(
    tokens: &[u32],
    key: u64,
    ngram_len: usize,
    depth: usize,
    context_history_size: usize,
) -> (u64, u64) {
    let base_key = ((key >> 32) as u32, key as u32);
    let width = ngram_len - 1;
    let mut recent: VecDeque<(u32, u32)> = VecDeque::new();
    let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
    let (mut ones, mut positions) = (0u64, 0u64);
    for end in width..tokens.len() {
        let context_key = tokens[end - width..end]
            .iter()
            .fold(base_key, |state, &previous| threefry2x32(state, 0, previous));
        let initial_zero = context_key == (0, 0) && recent.len() < context_history_size;
        if counts.get(&context_key).copied().unwrap_or(0) == 0 && !initial_zero {
            positions += 1;
            for (block, offset) in (0..depth).step_by(64).enumerate() {
                let (high, low) = threefry2x32(context_key, tokens[end], block as u32);
                let bits = (depth - offset).min(64);
                let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
                ones += u64::from(((u64::from(high) << 32 | u64::from(low)) & mask).count_ones());
            }
        }
        if recent.len() == context_history_size
            && let Some(oldest) = recent.pop_front()
            && let Some(count) = counts.get_mut(&oldest)
        {
            *count -= 1;
            if *count == 0 {
                counts.remove(&oldest);
            }
        }
        recent.push_back(context_key);
        *counts.entry(context_key).or_insert(0) += 1;
    }
    (ones, positions)
}

/// Correctly rounded `value / 2**shift` (CPython's `int / int` for a power-of-two divisor).
fn scaled_to_f64(value: &BigUint, shift: u64) -> f64 {
    let bits = value.bits();
    if bits == 0 {
        return 0.0;
    }
    let exponent = bits as i128 - 1 - i128::from(shift);
    let precision = if exponent >= -1022 { 53 } else { exponent + 1075 };
    if precision < 0 {
        return 0.0;
    }
    let drop = bits as i128 - precision;
    let (mantissa, weight) = if drop <= 0 {
        (value << ((-drop) as u64), drop)
    } else {
        let drop_bits = drop as u64;
        let mut mantissa: BigUint = value >> drop_bits;
        let remainder: BigUint = value - (&mantissa << drop_bits);
        let half = BigUint::from(1u8) << (drop_bits - 1);
        if remainder > half || (remainder == half && mantissa.bit(0)) {
            mantissa += 1u8;
        }
        (mantissa, drop)
    };
    let mantissa = mantissa.iter_u64_digits().next().unwrap_or(0) as f64;
    libm::ldexp(mantissa, (weight - i128::from(shift)) as i32)
}

/// CPython's `math.fsum`: the correctly rounded sum (Shewchuk's partials).
pub(crate) fn fsum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for mut x in values {
        let mut kept = 0;
        for index in 0..partials.len() {
            let mut y = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        if x != 0.0 {
            partials.push(x);
        }
    }
    let Some(mut hi) = partials.pop() else { return 0.0 };
    let mut lo = 0.0;
    while let Some(y) = partials.pop() {
        let x = hi;
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    if let Some(&next) = partials.last()
        && ((lo < 0.0 && next < 0.0) || (lo > 0.0 && next > 0.0))
    {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    hi
}

fn binomial(n: u64, k: u64) -> BigUint {
    let k = k.min(n - k);
    let mut value = BigUint::from(1u8);
    for index in 0..k {
        value *= n - index;
        value /= index + 1;
    }
    value
}

/// `P(X >= ones)` for `X ~ Binomial(trials, 1/2)`.
pub fn binomial_half_survival(ones: u64, trials: u64) -> f64 {
    if ones == 0 {
        return 1.0;
    }
    if ones <= trials / 2 {
        return 1.0 - binomial_half_survival(trials - ones + 1, trials);
    }
    let first = binomial(trials, ones);
    let scale = trials.saturating_sub(first.bits() + 1000);
    let mut term = scaled_to_f64(&first, trials - scale);
    let mut terms = Vec::new();
    for count in ones..=trials {
        if term == 0.0 {
            break;
        }
        terms.push(term);
        term *= (trials - count) as f64 / (count + 1) as f64;
    }
    libm::ldexp(fsum(terms), -(scale as i32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_vectors() {
        assert_eq!(threefry2x32((0, 0), 0, 0), (0x6B20_0159, 0x99BA_4EFE));
        assert_eq!(
            threefry2x32((u32::MAX, u32::MAX), u32::MAX, u32::MAX),
            (0x1CB9_96FC, 0xBB00_2BE7)
        );
        assert_eq!(
            threefry2x32((0x1319_8A2E, 0x0370_7344), 0x243F_6A88, 0x85A3_08D3),
            (0xC492_3A9C, 0x483D_F7A0)
        );
    }

    #[test]
    fn fsum_is_exact() {
        assert_eq!(fsum([0.1; 10]), 1.0);
        assert_eq!(fsum([1e100, 1.0, -1e100]), 1.0);
    }
}

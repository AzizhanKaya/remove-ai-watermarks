//! Exact rationals for stream timing, printed as Python's `str(Fraction)`.

use std::fmt;
use std::str::FromStr;

use serde::{Serialize, Serializer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fraction {
    num: i128,
    den: i128,
}

const fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Fraction {
    pub const ZERO: Self = Self { num: 0, den: 1 };

    pub fn new(num: i128, den: i128) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let g = gcd(num, den).max(1);
        let sign = if den < 0 { -1 } else { 1 };
        Some(Self {
            num: sign * num / g,
            den: sign * den / g,
        })
    }

    pub const fn integer(value: i128) -> Self {
        Self { num: value, den: 1 }
    }

    pub const fn numerator(self) -> i128 {
        self.num
    }

    pub const fn denominator(self) -> i128 {
        self.den
    }

    #[must_use]
    pub fn mul_int(self, value: i128) -> Self {
        Self::new(self.num * value, self.den).unwrap_or(Self::ZERO)
    }

    /// `float(Fraction)`.
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl fmt::Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl Serialize for Fraction {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl FromStr for Fraction {
    type Err = String;

    /// `Fraction(text)` for `a/b`, integers and plain decimals.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let bad = || format!("Invalid literal for Fraction: {text:?}");
        let text = text.trim();
        if let Some((num, den)) = text.split_once('/') {
            let num: i128 = num.trim().parse().map_err(|_| bad())?;
            let den: i128 = den.trim().parse().map_err(|_| bad())?;
            return Self::new(num, den).ok_or_else(bad);
        }
        let (negative, body) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        let (whole, frac) = body.split_once('.').unwrap_or((body, ""));
        if (whole.is_empty() && frac.is_empty())
            || !whole.chars().chain(frac.chars()).all(|c| c.is_ascii_digit())
        {
            return Err(bad());
        }
        let digits = format!("{whole}{frac}");
        let num: i128 = if digits.is_empty() {
            0
        } else {
            digits.parse().map_err(|_| bad())?
        };
        let den = 10i128.checked_pow(frac.len() as u32).ok_or_else(bad)?;
        Self::new(if negative { -num } else { num }, den).ok_or_else(bad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_like_python() {
        assert_eq!(Fraction::new(16_000, 16_000).unwrap().to_string(), "1");
        assert_eq!(Fraction::new(16_031, 16_000).unwrap().to_string(), "16031/16000");
        assert_eq!(
            "-0.021333".parse::<Fraction>().unwrap().to_string(),
            "-21333/1000000"
        );
        assert_eq!(
            "1/44100"
                .parse::<Fraction>()
                .unwrap()
                .mul_int(224_384)
                .to_string(),
            "224384/44100".parse::<Fraction>().unwrap().to_string()
        );
    }
}

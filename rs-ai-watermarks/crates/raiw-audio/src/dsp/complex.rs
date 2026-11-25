//! Complex arithmetic with NumPy's operation order, so filter designs match SciPy bit for bit.

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    #[must_use]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn is_real(self) -> bool {
        self.im == 0.0
    }

    /// `npy_cabs`, which is `hypot`.
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// `np.square` on complex input.
    #[must_use]
    pub fn square(self) -> Self {
        self * self
    }

    /// NumPy's Smith division (`CDOUBLE_divide`).
    #[must_use]
    pub fn quot(self, rhs: Self) -> Self {
        let (a, b, c, d) = (self.re, self.im, rhs.re, rhs.im);
        if c.abs() >= d.abs() {
            if c == 0.0 && d == 0.0 {
                return Self::new(a / c.abs(), b / c.abs());
            }
            let rat = d / c;
            let scl = 1.0 / (c + d * rat);
            Self::new((a + b * rat) * scl, (b - a * rat) * scl)
        } else {
            let rat = c / d;
            let scl = 1.0 / (d + c * rat);
            Self::new((a * rat + b) * scl, (b * rat - a) * scl)
        }
    }

    /// glibc `csqrt` for finite arguments.
    #[must_use]
    pub fn sqrt(self) -> Self {
        let (re, im) = (self.re, self.im);
        if im == 0.0 {
            return if re < 0.0 {
                Self::new(0.0, (-re).sqrt().copysign(im))
            } else {
                Self::new(re.sqrt().abs(), 0.0_f64.copysign(im))
            };
        }
        if re == 0.0 {
            let r = (0.5 * im.abs()).sqrt();
            return Self::new(r, r.copysign(im));
        }
        let d = re.hypot(im);
        let (r, s) = if re > 0.0 {
            let r = (0.5 * (d + re)).sqrt();
            (r, 0.5 * (im / r))
        } else {
            let s = (0.5 * (d - re)).sqrt();
            ((0.5 * (im / s)).abs(), s)
        };
        Self::new(r, s.copysign(im))
    }

    /// `exp(i * theta)` as glibc `cexp` evaluates a purely imaginary argument.
    #[must_use]
    pub fn cis(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }
}

impl Add for C64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for C64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for C64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Neg for C64 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

/// `np.prod` over complex values: a sequential product.
pub fn prod(values: &[C64]) -> C64 {
    values.iter().fold(C64::real(1.0), |acc, &v| acc * v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_of_negative_real_is_imaginary() {
        assert_eq!(C64::real(-4.0).sqrt(), C64::new(0.0, 2.0));
        let root = C64::new(-3.0, 4.0).sqrt();
        assert!((root.re - 1.0).abs() < 1e-15 && (root.im - 2.0).abs() < 1e-15);
    }

    #[test]
    fn division_by_real_multiplies_by_reciprocal() {
        let q = C64::new(1.0, 3.0).quot(C64::real(3.0));
        assert_eq!(q, C64::new(1.0 * (1.0 / 3.0), 3.0 * (1.0 / 3.0)));
    }
}

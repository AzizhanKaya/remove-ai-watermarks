//! Pixel element types and OpenCV's `saturate_cast` / `cvRound` semantics.

use std::fmt::Debug;

/// OpenCV matrix depth of an element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Depth {
    U8,
    I8,
    U16,
    I16,
    I32,
    F32,
    F64,
}

/// `cvRound(double)`: round half to even, `i32::MIN` outside the `i32` range (SSE2 `cvtsd2si`).
#[inline]
pub fn cv_round(v: f64) -> i32 {
    // Adding and removing 1.5 * 2^52 rounds half to even without a libm call.
    const MAGIC: f64 = 6_755_399_441_055_744.0;
    if v.abs() < 1_073_741_824.0 {
        ((v + MAGIC) - MAGIC) as i32
    } else if (-2_147_483_648.0..2_147_483_648.0).contains(&v) {
        v.round_ties_even() as i32
    } else {
        i32::MIN
    }
}

/// `cvRound(float)`.
#[inline]
pub fn cv_round_f32(v: f32) -> i32 {
    // 1.5 * 2^23 pins the integer bit position for |v| < 2^22.
    const MAGIC: f32 = 12_582_912.0;
    if v.abs() < 4_194_304.0 {
        ((v + MAGIC) - MAGIC) as i32
    } else if (-2_147_483_648.0..2_147_483_648.0).contains(&v) {
        v.round_ties_even() as i32
    } else {
        i32::MIN
    }
}

/// `cvFloor(double)`.
#[inline]
pub fn cv_floor(v: f64) -> i32 {
    let i = cv_round(v);
    i - i32::from(f64::from(i) > v)
}

/// `cvCeil(double)`.
#[inline]
pub fn cv_ceil(v: f64) -> i32 {
    let i = cv_round(v);
    i + i32::from(f64::from(i) < v)
}

#[inline]
pub fn saturate_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
pub fn saturate_u16(v: i32) -> u16 {
    v.clamp(0, 65_535) as u16
}

#[inline]
pub fn saturate_i16(v: i32) -> i16 {
    v.clamp(-32_768, 32_767) as i16
}

/// A matrix element: the numeric types OpenCV kernels accept.
pub trait Element: bytemuck::Pod + Default + Debug + PartialOrd + Send + Sync + 'static {
    const DEPTH: Depth;
    /// Smallest value, used as the neutral element of dilation (`-FLT_MAX` for floats).
    const LOWEST: Self;
    /// Largest value, used as the neutral element of erosion.
    const HIGHEST: Self;
    const ZERO: Self;

    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
    /// `saturate_cast<T>(float)`.
    fn from_f32(v: f32) -> Self;
    /// `saturate_cast<T>(double)`.
    fn from_f64(v: f64) -> Self;
    /// `saturate_cast<T>(int)`.
    fn from_i32(v: i32) -> Self;
    /// `cv::subtract` of one element: saturating for integers.
    #[inline]
    #[must_use]
    fn sub_sat(self, rhs: Self) -> Self {
        Self::from_f64(self.to_f64() - rhs.to_f64())
    }
}

macro_rules! int_element {
    ($t:ty, $depth:ident) => {
        impl Element for $t {
            const DEPTH: Depth = Depth::$depth;
            const LOWEST: Self = <$t>::MIN;
            const HIGHEST: Self = <$t>::MAX;
            const ZERO: Self = 0;

            #[inline]
            fn to_f32(self) -> f32 {
                self as f32
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn from_f32(v: f32) -> Self {
                Self::from_i32(cv_round_f32(v))
            }
            #[inline]
            fn from_f64(v: f64) -> Self {
                Self::from_i32(cv_round(v))
            }
            #[inline]
            fn from_i32(v: i32) -> Self {
                v.clamp(<$t>::MIN as i32, <$t>::MAX as i32) as $t
            }
            #[inline]
            fn sub_sat(self, rhs: Self) -> Self {
                self.saturating_sub(rhs)
            }
        }
    };
}

int_element!(u8, U8);
int_element!(i8, I8);
int_element!(u16, U16);
int_element!(i16, I16);

impl Element for i32 {
    const DEPTH: Depth = Depth::I32;
    const LOWEST: Self = i32::MIN;
    const HIGHEST: Self = i32::MAX;
    const ZERO: Self = 0;

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        cv_round_f32(v)
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        cv_round(v)
    }
    #[inline]
    fn from_i32(v: i32) -> Self {
        v
    }
}

impl Element for f32 {
    const DEPTH: Depth = Depth::F32;
    const LOWEST: Self = -f32::MAX;
    const HIGHEST: Self = f32::MAX;
    const ZERO: Self = 0.0;

    #[inline]
    fn sub_sat(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn from_i32(v: i32) -> Self {
        v as f32
    }
}

impl Element for f64 {
    const DEPTH: Depth = Depth::F64;
    const LOWEST: Self = -f64::MAX;
    const HIGHEST: Self = f64::MAX;
    const ZERO: Self = 0.0;

    #[inline]
    fn sub_sat(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        f64::from(v)
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn from_i32(v: i32) -> Self {
        f64::from(v)
    }
}

/// Floating-point element types (`f32`, `f64`).
pub trait Float: Element + num_like::FloatOps {}
impl Float for f32 {}
impl Float for f64 {}

pub(crate) mod num_like {
    use std::ops::{Add, Div, Mul, Neg, Sub};

    /// The arithmetic subset float kernels need.
    pub trait FloatOps:
        Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + PartialOrd
    {
        fn of_f32(v: f32) -> Self;
        fn of_f64(v: f64) -> Self;
        fn sqrt(self) -> Self;
        fn abs(self) -> Self;
        fn is_finite(self) -> bool;
    }

    impl FloatOps for f32 {
        #[inline]
        fn of_f32(v: f32) -> Self {
            v
        }
        #[inline]
        fn of_f64(v: f64) -> Self {
            v as f32
        }
        #[inline]
        fn sqrt(self) -> Self {
            f32::sqrt(self)
        }
        #[inline]
        fn abs(self) -> Self {
            f32::abs(self)
        }
        #[inline]
        fn is_finite(self) -> bool {
            f32::is_finite(self)
        }
    }

    impl FloatOps for f64 {
        #[inline]
        fn of_f32(v: f32) -> Self {
            f64::from(v)
        }
        #[inline]
        fn of_f64(v: f64) -> Self {
            v
        }
        #[inline]
        fn sqrt(self) -> Self {
            f64::sqrt(self)
        }
        #[inline]
        fn abs(self) -> Self {
            f64::abs(self)
        }
        #[inline]
        fn is_finite(self) -> bool {
            f64::is_finite(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_matches_cv_round() {
        assert_eq!(cv_round(2.5), 2);
        assert_eq!(cv_round(3.5), 4);
        assert_eq!(cv_round(-2.5), -2);
        assert_eq!(cv_round(f64::NAN), i32::MIN);
        assert_eq!(cv_floor(-0.5), -1);
        assert_eq!(cv_ceil(0.2), 1);
        assert_eq!(u8::from_f32(255.5), 255);
        assert_eq!(u8::from_f32(1e10), 0);
        assert_eq!(u8::from_f64(-3.0), 0);
    }
}

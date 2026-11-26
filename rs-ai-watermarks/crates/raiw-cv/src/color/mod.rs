//! `cv::cvtColor` for the conversions the Python package uses.

mod lab;
mod tables;

use rayon::prelude::*;

use crate::element::Element;
use crate::error::{CvError, Result, invalid};
use crate::mat::{Mat, MatView};

pub use lab::LabElement;

/// Supported `cv::ColorConversionCodes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorConversion {
    BgrToRgb,
    RgbToBgr,
    BgraToRgba,
    RgbaToBgra,
    BgraToBgr,
    RgbaToRgb,
    BgrToBgra,
    BgrToGray,
    RgbToGray,
    BgraToGray,
    GrayToBgr,
    GrayToBgra,
    BgrToHsv,
    RgbToHsv,
    BgrToYCrCb,
    RgbToYCrCb,
    YCrCbToBgr,
    YCrCbToRgb,
    BgrToYuv,
    RgbToYuv,
    YuvToBgr,
    YuvToRgb,
    BgrToLab,
    RgbToLab,
    LabToBgr,
    LabToRgb,
}

/// Element types with color-space arithmetic.
pub trait ColorElement: Element + LabElement {
    #[doc(hidden)]
    fn gray(b: Self, g: Self, r: Self) -> Self;
    #[doc(hidden)]
    fn ycc(b: Self, g: Self, r: Self, crcb: bool) -> [Self; 3];
    #[doc(hidden)]
    fn ycc_inv(y: Self, c1: Self, c2: Self, crcb: bool) -> [Self; 3];
    #[doc(hidden)]
    fn hsv(b: Self, g: Self, r: Self) -> [Self; 3];
    #[doc(hidden)]
    const ALPHA_MAX: Self;
    #[doc(hidden)]
    const HAS_HSV: bool = true;
}

// Fixed-point BT.601 coefficients (`yuv_shift = 14`, `gray_shift = 15`).
// Every intermediate of the 8- and 16-bit formulas stays below 2^31, so `i32` is exact.
type Fx = i32;
const YUV_SHIFT: Fx = 14;
const R2Y: Fx = 4899;
const G2Y: Fx = 9617;
const B2Y: Fx = 1868;
const YCRI: Fx = 11682;
const YCBI: Fx = 9241;
const R2VI: Fx = 14369;
const B2UI: Fx = 8061;
const CR2RI: Fx = 22987;
const CR2GI: Fx = -11698;
const CB2GI: Fx = -5636;
const CB2BI: Fx = 29049;
const V2RI: Fx = 18678;
const V2GI: Fx = -9519;
const U2GI: Fx = -6472;
const U2BI: Fx = 33292;
const RY15: Fx = 9798;
const GY15: Fx = 19235;
const BY15: Fx = 3735;

#[inline]
const fn descale(x: Fx, n: Fx) -> Fx {
    (x + (1 << (n - 1))) >> n
}

fn ycc_int(b: Fx, g: Fx, r: Fx, crcb: bool, half: Fx) -> [Fx; 3] {
    let (c3, c4) = if crcb { (YCRI, YCBI) } else { (R2VI, B2UI) };
    let delta = half << YUV_SHIFT;
    let y = descale(b * B2Y + g * G2Y + r * R2Y, YUV_SHIFT);
    let cr = descale((r - y) * c3 + delta, YUV_SHIFT);
    let cb = descale((b - y) * c4 + delta, YUV_SHIFT);
    if crcb { [y, cr, cb] } else { [y, cb, cr] }
}

fn ycc_inv_int(y: Fx, c1: Fx, c2: Fx, crcb: bool, half: Fx) -> [Fx; 3] {
    let (cr, cb) = if crcb { (c1, c2) } else { (c2, c1) };
    let (k0, k1, k2, k3) = if crcb {
        (CR2RI, CR2GI, CB2GI, CB2BI)
    } else {
        (V2RI, V2GI, U2GI, U2BI)
    };
    let b = y + descale((cb - half) * k3, YUV_SHIFT);
    let g = y + descale((cb - half) * k2 + (cr - half) * k1, YUV_SHIFT);
    let r = y + descale((cr - half) * k0, YUV_SHIFT);
    [b, g, r]
}

impl ColorElement for u8 {
    const ALPHA_MAX: Self = 255;

    #[inline]
    fn gray(b: u8, g: u8, r: u8) -> u8 {
        descale(Fx::from(b) * BY15 + Fx::from(g) * GY15 + Fx::from(r) * RY15, 15) as u8
    }
    #[inline]
    fn ycc(b: u8, g: u8, r: u8, crcb: bool) -> [u8; 3] {
        ycc_int(Fx::from(b), Fx::from(g), Fx::from(r), crcb, 128).map(|v| v.clamp(0, 255) as u8)
    }
    #[inline]
    fn ycc_inv(y: u8, c1: u8, c2: u8, crcb: bool) -> [u8; 3] {
        ycc_inv_int(Fx::from(y), Fx::from(c1), Fx::from(c2), crcb, 128).map(|v| v.clamp(0, 255) as u8)
    }
    #[inline]
    fn hsv(b: u8, g: u8, r: u8) -> [u8; 3] {
        tables::hsv_u8(b, g, r)
    }
}

impl ColorElement for u16 {
    const ALPHA_MAX: Self = 65_535;

    #[inline]
    fn gray(b: u16, g: u16, r: u16) -> u16 {
        let sum = u32::from(b) * BY15 as u32 + u32::from(g) * GY15 as u32 + u32::from(r) * RY15 as u32;
        ((sum + (1 << 14)) >> 15) as u16
    }
    #[inline]
    fn ycc(b: u16, g: u16, r: u16, crcb: bool) -> [u16; 3] {
        ycc_int(Fx::from(b), Fx::from(g), Fx::from(r), crcb, 32_768).map(|v| v.clamp(0, 65_535) as u16)
    }
    #[inline]
    fn ycc_inv(y: u16, c1: u16, c2: u16, crcb: bool) -> [u16; 3] {
        ycc_inv_int(Fx::from(y), Fx::from(c1), Fx::from(c2), crcb, 32_768).map(|v| v.clamp(0, 65_535) as u16)
    }
    const HAS_HSV: bool = false;

    #[inline]
    fn hsv(b: u16, g: u16, r: u16) -> [u16; 3] {
        [b, g, r]
    }
}

const R2YF: f32 = 0.299;
const G2YF: f32 = 0.587;
const B2YF: f32 = 0.114;

impl ColorElement for f32 {
    const ALPHA_MAX: Self = 1.0;

    #[inline]
    fn gray(b: f32, g: f32, r: f32) -> f32 {
        b * B2YF + g * G2YF + r * R2YF
    }
    #[inline]
    fn ycc(b: f32, g: f32, r: f32, crcb: bool) -> [f32; 3] {
        let (c3, c4) = if crcb {
            (0.713f32, 0.564f32)
        } else {
            (0.877f32, 0.492f32)
        };
        let y = b * B2YF + g * G2YF + r * R2YF;
        let cr = (r - y) * c3 + 0.5;
        let cb = (b - y) * c4 + 0.5;
        if crcb { [y, cr, cb] } else { [y, cb, cr] }
    }
    #[inline]
    fn ycc_inv(y: f32, c1: f32, c2: f32, crcb: bool) -> [f32; 3] {
        let (cr, cb) = if crcb { (c1, c2) } else { (c2, c1) };
        let (k0, k1, k2, k3) = if crcb {
            (1.403f32, -0.714f32, -0.344f32, 1.773f32)
        } else {
            (1.140f32, -0.581f32, -0.395f32, 2.032f32)
        };
        let (cr, cb) = (cr - 0.5, cb - 0.5);
        [y + cb * k3, y + cb * k2 + cr * k1, y + cr * k0]
    }
    #[inline]
    fn hsv(b: f32, g: f32, r: f32) -> [f32; 3] {
        hsv_f32(b, g, r, 360.0)
    }
}

/// `RGB2HSV_f` for one pixel with hue range `hrange`.
fn hsv_f32(b: f32, g: f32, r: f32, hrange: f32) -> [f32; 3] {
    let hscale = hrange * (1.0 / 360.0);
    let v = b.max(g).max(r);
    let vmin = b.min(g).min(r);
    let diff = v - vmin;
    let s = diff / (v.abs() + f32::EPSILON);
    let diff = (60.0 / f64::from(diff + f32::EPSILON)) as f32;
    let mut h = if v == r {
        (g - b) * diff
    } else if v == g {
        (b - r) * diff + 120.0
    } else {
        (r - g) * diff + 240.0
    };
    if h < 0.0 {
        h += 360.0;
    }
    [h * hscale, s, v]
}

/// `cv::cvtColor`.
pub fn cvt_color<'a, T: ColorElement>(
    src: impl Into<MatView<'a, T>>,
    code: ColorConversion,
) -> Result<Mat<T>> {
    use ColorConversion as C;
    let src = src.into();
    let scn = src.channels();
    let need = |ok: bool| {
        if ok {
            Ok(())
        } else {
            Err(CvError::Channels {
                op: "cvt_color",
                channels: scn,
            })
        }
    };
    // Each expansion gives the closure a single call site with compile-time channel counts.
    macro_rules! convert {
        ($dcn:literal, |$s:ident, $d:ident| $body:expr) => {
            match scn {
                1 => per_pixel::<T, 1, $dcn>(&src, |$s, $d| $body),
                3 => per_pixel::<T, 3, $dcn>(&src, |$s, $d| $body),
                _ => per_pixel::<T, 4, $dcn>(&src, |$s, $d| $body),
            }
        };
    }
    match code {
        C::BgrToRgb | C::RgbToBgr => {
            need(scn == 3)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&[s[2], s[1], s[0]])))
        }
        C::BgraToRgba | C::RgbaToBgra => {
            need(scn == 4)?;
            Ok(convert!(4, |s, d| d.copy_from_slice(&[s[2], s[1], s[0], s[3]])))
        }
        C::BgraToBgr | C::RgbaToRgb => {
            need(scn == 4)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&s[..3])))
        }
        C::BgrToBgra => {
            need(scn == 3)?;
            Ok(convert!(4, |s, d| d.copy_from_slice(&[
                s[0],
                s[1],
                s[2],
                T::ALPHA_MAX
            ])))
        }
        C::BgrToGray | C::BgraToGray => {
            need(scn == 3 || scn == 4)?;
            Ok(convert!(1, |s, d| d[0] = T::gray(s[0], s[1], s[2])))
        }
        C::RgbToGray => {
            need(scn == 3 || scn == 4)?;
            Ok(convert!(1, |s, d| d[0] = T::gray(s[2], s[1], s[0])))
        }
        C::GrayToBgr => {
            need(scn == 1)?;
            Ok(convert!(3, |s, d| d.fill(s[0])))
        }
        C::GrayToBgra => {
            need(scn == 1)?;
            Ok(convert!(4, |s, d| d.copy_from_slice(&[
                s[0],
                s[0],
                s[0],
                T::ALPHA_MAX
            ])))
        }
        C::BgrToHsv | C::RgbToHsv if !T::HAS_HSV => {
            Err(invalid("cvt_color", "HSV needs uint8 or float32 input"))
        }
        C::BgrToHsv => {
            need(scn == 3 || scn == 4)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&T::hsv(s[0], s[1], s[2]))))
        }
        C::RgbToHsv => {
            need(scn == 3 || scn == 4)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&T::hsv(s[2], s[1], s[0]))))
        }
        C::BgrToYCrCb => {
            need(scn == 3 || scn == 4)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&T::ycc(s[0], s[1], s[2], true))))
        }
        C::BgrToYuv => {
            need(scn == 3 || scn == 4)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&T::ycc(s[0], s[1], s[2], false))))
        }
        C::RgbToYCrCb => {
            need(scn == 3 || scn == 4)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&T::ycc(s[2], s[1], s[0], true))))
        }
        C::RgbToYuv => {
            need(scn == 3 || scn == 4)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&T::ycc(s[2], s[1], s[0], false))))
        }
        C::YCrCbToBgr => {
            need(scn == 3)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&T::ycc_inv(s[0], s[1], s[2], true))))
        }
        C::YuvToBgr => {
            need(scn == 3)?;
            Ok(convert!(3, |s, d| d.copy_from_slice(&T::ycc_inv(s[0], s[1], s[2], false))))
        }
        C::YCrCbToRgb | C::YuvToRgb => {
            need(scn == 3)?;
            let crcb = code == C::YCrCbToRgb;
            Ok(convert!(3, |s, d| {
                let [b, g, r] = T::ycc_inv(s[0], s[1], s[2], crcb);
                d.copy_from_slice(&[r, g, b]);
            }))
        }
        C::BgrToLab | C::RgbToLab => {
            need(scn == 3 || scn == 4)?;
            T::rgb_to_lab(&src, code == C::BgrToLab)
                .ok_or_else(|| invalid("cvt_color", "Lab needs float32 input"))
        }
        C::LabToBgr | C::LabToRgb => {
            need(scn == 3)?;
            T::lab_to_rgb(&src, code == C::LabToBgr)
                .ok_or_else(|| invalid("cvt_color", "Lab needs float32 input"))
        }
    }
}

/// Map every `S`-channel source pixel to a `D`-channel destination pixel, row-parallel.
pub(crate) fn per_pixel<T: Element, const S: usize, const D: usize>(
    src: &MatView<'_, T>,
    f: impl Fn(&[T], &mut [T]) + Sync,
) -> Mat<T> {
    let mut dst = Mat::zeros(src.width(), src.height(), D);
    dst.par_rows_mut()
        .enumerate()
        .for_each(|(y, row)| crate::simd::vectorized(pixel_loop::<T, _, S, D>, (src.row(y), row, &f)));
    dst
}

#[inline(always)]
fn pixel_loop<T: Element, F: Fn(&[T], &mut [T]), const S: usize, const D: usize>(
    (src, dst, f): (&[T], &mut [T], &F),
) {
    for (a, b) in src.chunks_exact(S).zip(dst.chunks_exact_mut(D)) {
        f(a, b);
    }
}

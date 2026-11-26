//! Per-element arithmetic: `addWeighted`, `convertScaleAbs`, `magnitude`, `absdiff`, bitwise ops.

use rayon::prelude::*;

use crate::element::{Depth, Element};
use crate::error::{CvError, Result};
use crate::mat::{Mat, MatView};

fn zip_map<'a, 'b, T: Element, U: Element>(
    a: impl Into<MatView<'a, T>>,
    b: impl Into<MatView<'b, T>>,
    op: &'static str,
    f: impl Fn(T, T) -> U + Sync,
) -> Result<Mat<U>> {
    let (a, b) = (a.into(), b.into());
    if a.size() != b.size() || a.channels() != b.channels() {
        return Err(CvError::ShapeMismatch { op });
    }
    let mut out = Mat::zeros(a.width(), a.height(), a.channels());
    out.par_rows_mut().enumerate().for_each(|(y, row)| {
        for ((o, &x), &z) in row.iter_mut().zip(a.row(y)).zip(b.row(y)) {
            *o = f(x, z);
        }
    });
    Ok(out)
}

/// `cv::addWeighted(a, alpha, b, beta, gamma)`: double arithmetic for float inputs (fused in
/// OpenCV's 8-wide AVX2 blocks, plain in the tail), single precision with saturation for integers.
pub fn add_weighted<'a, 'b, T: Element>(
    a: impl Into<MatView<'a, T>>,
    alpha: f64,
    b: impl Into<MatView<'b, T>>,
    beta: f64,
    gamma: f64,
) -> Result<Mat<T>> {
    let (a, b) = (a.into(), b.into());
    if a.size() != b.size() || a.channels() != b.channels() {
        return Err(CvError::ShapeMismatch { op: "add_weighted" });
    }
    let float = matches!(T::DEPTH, Depth::F32 | Depth::F64 | Depth::I32);
    let (al, be, ga) = (alpha as f32, beta as f32, gamma as f32);
    let continuous = a.is_contiguous() && b.is_contiguous();
    let row_len = a.row_len();
    let total = if continuous { row_len * a.height() } else { row_len };
    let fused_end = total / 8 * 8;
    let mut out = Mat::zeros(a.width(), a.height(), a.channels());
    out.par_rows_mut().enumerate().for_each(|(y, row)| {
        let base = if continuous { y * row_len } else { 0 };
        for (i, ((o, &x), &z)) in row.iter_mut().zip(a.row(y)).zip(b.row(y)).enumerate() {
            *o = if float {
                let (x, z) = (x.to_f64(), z.to_f64());
                T::from_f64(if base + i < fused_end {
                    x.mul_add(alpha, z.mul_add(beta, gamma))
                } else {
                    x * alpha + z * beta + gamma
                })
            } else {
                T::from_f32(x.to_f32() * al + (z.to_f32() * be + ga))
            };
        }
    });
    Ok(out)
}

/// `cv::convertScaleAbs(src, alpha, beta)`: `saturate_cast<uchar>(|src * alpha + beta|)`.
/// Long rows use the fused multiply-add of OpenCV's SIMD path.
pub fn convert_scale_abs<'a, T: Element>(src: impl Into<MatView<'a, T>>, alpha: f64, beta: f64) -> Mat<u8> {
    let src = src.into();
    let (a, b) = (alpha as f32, beta as f32);
    let run = if src.is_contiguous() {
        src.row_len() * src.height()
    } else {
        src.row_len()
    };
    let fused = run >= 16;
    let mut out = Mat::zeros(src.width(), src.height(), src.channels());
    out.par_rows_mut().enumerate().for_each(|(y, row)| {
        for (o, &v) in row.iter_mut().zip(src.row(y)) {
            let x = v.to_f32();
            let s = if fused { x.mul_add(a, b) } else { x * a + b };
            *o = u8::from_f32(s.abs());
        }
    });
    out
}

/// `cv::magnitude(x, y)`.
pub fn magnitude<'a, 'b, T: Element + crate::element::Float>(
    x: impl Into<MatView<'a, T>>,
    y: impl Into<MatView<'b, T>>,
) -> Result<Mat<T>> {
    zip_map(x, y, "magnitude", |a, b| {
        T::from_f64(if T::DEPTH == Depth::F32 {
            let (a, b) = (a.to_f32(), b.to_f32());
            f64::from((a * a + b * b).sqrt())
        } else {
            let (a, b) = (a.to_f64(), b.to_f64());
            (a * a + b * b).sqrt()
        })
    })
}

/// `cv::absdiff(a, b)`.
pub fn abs_diff<'a, 'b, T: Element>(
    a: impl Into<MatView<'a, T>>,
    b: impl Into<MatView<'b, T>>,
) -> Result<Mat<T>> {
    zip_map(a, b, "abs_diff", |x, y| {
        T::from_f64((x.to_f64() - y.to_f64()).abs())
    })
}

/// `cv::bitwise_and`.
pub fn bitwise_and<'a, 'b>(a: impl Into<MatView<'a, u8>>, b: impl Into<MatView<'b, u8>>) -> Result<Mat<u8>> {
    zip_map(a, b, "bitwise_and", |x, y| x & y)
}

/// `cv::bitwise_or`.
pub fn bitwise_or<'a, 'b>(a: impl Into<MatView<'a, u8>>, b: impl Into<MatView<'b, u8>>) -> Result<Mat<u8>> {
    zip_map(a, b, "bitwise_or", |x, y| x | y)
}

/// `cv::bitwise_not`.
pub fn bitwise_not<'a>(a: impl Into<MatView<'a, u8>>) -> Mat<u8> {
    a.into().to_mat().map(|v| !v)
}

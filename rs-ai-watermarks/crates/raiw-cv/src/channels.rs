//! Channel plumbing: `split`, `merge`, `extractChannel`, `insertChannel`, `transpose`, `flip`.

use rayon::prelude::*;

use crate::element::Element;
use crate::error::{CvError, Result, invalid};
use crate::mat::{Mat, MatView};

/// `cv::split`.
pub fn split<'a, T: Element>(src: impl Into<MatView<'a, T>>) -> Vec<Mat<T>> {
    let src = src.into();
    (0..src.channels()).map(|c| extract_planar(&src, c)).collect()
}

fn extract_planar<T: Element>(src: &MatView<'_, T>, c: usize) -> Mat<T> {
    let cn = src.channels();
    let mut out = Mat::zeros(src.width(), src.height(), 1);
    out.par_rows_mut().enumerate().for_each(|(y, row)| {
        for (o, px) in row.iter_mut().zip(src.row(y).chunks_exact(cn)) {
            *o = px[c];
        }
    });
    out
}

/// `cv::extractChannel(src, channel)`.
pub fn extract_channel<'a, T: Element>(src: impl Into<MatView<'a, T>>, channel: usize) -> Result<Mat<T>> {
    let src = src.into();
    if channel >= src.channels() {
        return Err(CvError::Channels { op: "extract_channel", channels: src.channels() });
    }
    Ok(extract_planar(&src, channel))
}

/// `cv::merge` of single-channel planes of equal size.
pub fn merge<T: Element>(planes: &[MatView<'_, T>]) -> Result<Mat<T>> {
    let first = planes.first().ok_or_else(|| invalid("merge", "no planes"))?;
    if planes.iter().any(|p| p.size() != first.size() || p.channels() != 1) {
        return Err(CvError::ShapeMismatch { op: "merge" });
    }
    let cn = planes.len();
    let mut out = Mat::zeros(first.width(), first.height(), cn);
    out.par_rows_mut().enumerate().for_each(|(y, row)| {
        for (c, p) in planes.iter().enumerate() {
            for (px, &v) in row.chunks_exact_mut(cn).zip(p.row(y)) {
                px[c] = v;
            }
        }
    });
    Ok(out)
}

/// `cv::insertChannel(plane, dst, channel)`.
pub fn insert_channel<'a, T: Element>(plane: impl Into<MatView<'a, T>>, dst: &mut Mat<T>, channel: usize) -> Result<()> {
    let plane = plane.into();
    if plane.size() != dst.size() || plane.channels() != 1 || channel >= dst.channels() {
        return Err(CvError::ShapeMismatch { op: "insert_channel" });
    }
    let cn = dst.channels();
    dst.par_rows_mut().enumerate().for_each(|(y, row)| {
        for (px, &v) in row.chunks_exact_mut(cn).zip(plane.row(y)) {
            px[channel] = v;
        }
    });
    Ok(())
}

/// `cv::transpose`.
pub fn transpose<'a, T: Element>(src: impl Into<MatView<'a, T>>) -> Mat<T> {
    let src = src.into();
    let cn = src.channels();
    let mut out = Mat::zeros(src.height(), src.width(), cn);
    out.par_rows_mut().enumerate().for_each(|(x, row)| {
        for (y, px) in row.chunks_exact_mut(cn).enumerate() {
            px.copy_from_slice(src.pixel(x, y));
        }
    });
    out
}

/// `cv::flip`: `horizontal` mirrors columns, `vertical` mirrors rows.
pub fn flip<'a, T: Element>(src: impl Into<MatView<'a, T>>, horizontal: bool, vertical: bool) -> Mat<T> {
    let src = src.into();
    let (w, h, cn) = (src.width(), src.height(), src.channels());
    let mut out = Mat::zeros(w, h, cn);
    out.par_rows_mut().enumerate().for_each(|(y, row)| {
        let sy = if vertical { h - 1 - y } else { y };
        for (x, px) in row.chunks_exact_mut(cn).enumerate() {
            let sx = if horizontal { w - 1 - x } else { x };
            px.copy_from_slice(src.pixel(sx, sy));
        }
    });
    out
}

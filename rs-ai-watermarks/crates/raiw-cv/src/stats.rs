//! Reductions: `minMaxLoc`, `meanStdDev`, `normalize`, `calcHist`.

use crate::element::Element;
use crate::error::{CvError, Result, invalid};
use crate::mat::{Mat, MatView};
use crate::types::Point;

/// Result of [`min_max_loc`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMaxLoc {
    pub min: f64,
    pub max: f64,
    pub min_loc: Point,
    pub max_loc: Point,
}

/// `cv::minMaxLoc` on a single-channel image: first occurrence in row-major order, NaN ignored.
pub fn min_max_loc<'a, T: Element>(src: impl Into<MatView<'a, T>>) -> Result<MinMaxLoc> {
    let src = src.into();
    if src.channels() != 1 {
        return Err(CvError::Channels { op: "min_max_loc", channels: src.channels() });
    }
    let mut out = MinMaxLoc { min: f64::MAX, max: -f64::MAX, min_loc: Point::new(-1, -1), max_loc: Point::new(-1, -1) };
    let mut first = true;
    for y in 0..src.height() {
        for (x, &v) in src.row(y).iter().enumerate() {
            let v = v.to_f64();
            if v.is_nan() {
                continue;
            }
            let at = Point::new(x as i32, y as i32);
            if first || v < out.min {
                out.min = v;
                out.min_loc = at;
            }
            if first || v > out.max {
                out.max = v;
                out.max_loc = at;
            }
            first = false;
        }
    }
    Ok(out)
}

/// `cv::meanStdDev`: per-channel mean and population standard deviation.
pub fn mean_std_dev<'a, T: Element>(src: impl Into<MatView<'a, T>>) -> Result<(Vec<f64>, Vec<f64>)> {
    let src = src.into();
    if src.is_empty() {
        return Err(invalid("mean_std_dev", "empty input"));
    }
    let cn = src.channels();
    let mut s = vec![0.0f64; cn];
    let mut sq = vec![0.0f64; cn];
    for y in 0..src.height() {
        for px in src.row(y).chunks_exact(cn) {
            for (c, &v) in px.iter().enumerate() {
                let v = v.to_f64();
                s[c] += v;
                sq[c] += v * v;
            }
        }
    }
    let scale = 1.0 / (src.width() * src.height()) as f64;
    let mean: Vec<f64> = s.iter().map(|v| v * scale).collect();
    let sdv = sq.iter().zip(&mean).map(|(q, m)| (q * scale - m * m).max(0.0).sqrt()).collect();
    Ok((mean, sdv))
}

/// `cv::normalize(src, None, alpha, beta, NORM_MINMAX)` keeping the element type.
pub fn normalize_min_max<'a, T: Element>(src: impl Into<MatView<'a, T>>, alpha: f64, beta: f64) -> Result<Mat<T>> {
    let src = src.into();
    let mut smin = f64::MAX;
    let mut smax = -f64::MAX;
    for v in src.iter() {
        let v = v.to_f64();
        if v < smin {
            smin = v;
        }
        if v > smax {
            smax = v;
        }
    }
    let (dmin, dmax) = (alpha.min(beta), alpha.max(beta));
    let range = smax - smin;
    let scale = (dmax - dmin) * if range > f64::EPSILON { 1.0 / range } else { 0.0 };
    let shift = dmin - smin * scale;
    Ok(src.to_mat().convert_scaled(scale, shift))
}

/// `cv::normalize(src, dst)` with the default `NORM_L2`, `alpha = 1`.
pub fn normalize_l2<'a, T: Element>(src: impl Into<MatView<'a, T>>) -> Result<Mat<T>> {
    let src = src.into();
    let norm = src.iter().map(|v| v.to_f64() * v.to_f64()).sum::<f64>().sqrt();
    let scale = if norm > f64::EPSILON { 1.0 / norm } else { 0.0 };
    Ok(src.to_mat().convert_scaled(scale, 0.0))
}

/// `cv::calcHist([img], [0..cn], None, bins, ranges)` over every channel of one image.
/// Returns a row-major `f32` histogram (`bins[0] x bins[1] x ...`).
pub fn calc_hist<'a, T: Element>(src: impl Into<MatView<'a, T>>, bins: &[usize], ranges: &[(f64, f64)]) -> Result<Vec<f32>> {
    let src = src.into();
    let cn = src.channels();
    if bins.len() != cn || ranges.len() != cn || bins.contains(&0) {
        return Err(invalid("calc_hist", "one bin count and range per channel"));
    }
    let total: usize = bins.iter().product();
    let mut hist = vec![0.0f32; total];
    let scales: Vec<f64> = bins.iter().zip(ranges).map(|(&b, &(lo, hi))| b as f64 / (hi - lo)).collect();
    for y in 0..src.height() {
        'px: for px in src.row(y).chunks_exact(cn) {
            let mut idx = 0;
            for c in 0..cn {
                let (lo, hi) = ranges[c];
                let v = px[c].to_f64();
                if !(v >= lo && v < hi) {
                    continue 'px;
                }
                let bin = (((v - lo) * scales[c]).floor() as usize).min(bins[c] - 1);
                idx = idx * bins[c] + bin;
            }
            hist[idx] += 1.0;
        }
    }
    Ok(hist)
}

/// `cv::countNonZero`.
pub fn count_non_zero<'a, T: Element>(src: impl Into<MatView<'a, T>>) -> usize {
    src.into().iter().filter(|v| v.to_f64() != 0.0).count()
}

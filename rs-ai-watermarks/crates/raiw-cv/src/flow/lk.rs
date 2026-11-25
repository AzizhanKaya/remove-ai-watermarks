//! `cv::calcOpticalFlowPyrLK` for 8-bit single-channel images (`lkpyramid.cpp`), plus `pyrDown`.

use rayon::prelude::*;

use crate::border::{Border, copy_make_border};
use crate::element::{cv_floor, cv_round_f32};
use crate::error::{CvError, Result, invalid};
use crate::mat::{Mat, MatView};
use crate::types::Point2f;

const W_BITS: i32 = 14;
const FLT_SCALE: f32 = 1.0 / (1 << 20) as f32;

/// `cv::pyrDown` for 8-bit images: 5x5 binomial filter, reflect-101 border, `(sum + 128) >> 8`.
pub fn pyr_down<'a>(src: impl Into<MatView<'a, u8>>) -> Result<Mat<u8>> {
    let src = src.into();
    if src.is_empty() {
        return Err(invalid("pyr_down", "empty input"));
    }
    const K: [i32; 5] = [1, 4, 6, 4, 1];
    let (w, h, cn) = (src.width(), src.height(), src.channels());
    let (dw, dh) = (w.div_ceil(2), h.div_ceil(2));
    let border = Border::Reflect101;
    let xs: Vec<[usize; 5]> = (0..dw)
        .map(|x| std::array::from_fn(|k| border.interpolate(2 * x as isize + k as isize - 2, w).unwrap_or(0)))
        .collect();
    let mut dst = Mat::zeros(dw, dh, cn);
    dst.par_rows_mut().enumerate().for_each(|(y, row)| {
        let rows: [&[u8]; 5] =
            std::array::from_fn(|k| src.row(border.interpolate(2 * y as isize + k as isize - 2, h).unwrap_or(0)));
        for (x, px) in row.chunks_exact_mut(cn).enumerate() {
            for (c, o) in px.iter_mut().enumerate() {
                let mut sum = 0i32;
                for (r, &kr) in rows.iter().zip(&K) {
                    let line: i32 = xs[x].iter().zip(&K).map(|(&sx, &kx)| i32::from(r[sx * cn + c]) * kx).sum();
                    sum += line * kr;
                }
                *o = ((sum + 128) >> 8).clamp(0, 255) as u8;
            }
        }
    });
    Ok(dst)
}

/// Termination of the per-point Lucas-Kanade iterations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LkParams {
    pub win_width: usize,
    pub win_height: usize,
    pub max_level: usize,
    pub max_count: usize,
    pub epsilon: f64,
    pub min_eig_threshold: f64,
}

impl Default for LkParams {
    fn default() -> Self {
        Self { win_width: 21, win_height: 21, max_level: 3, max_count: 30, epsilon: 0.01, min_eig_threshold: 1e-4 }
    }
}

/// One tracked point: new position, found flag, and the mean absolute residual (`err`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LkTrack {
    pub point: Point2f,
    pub found: bool,
    pub error: f32,
}

/// A padded pyramid level: image and Scharr derivatives with a `win`-pixel border.
struct Level {
    width: usize,
    height: usize,
    pad_x: usize,
    pad_y: usize,
    img: Mat<u8>,
}

impl Level {
    #[inline]
    fn at(&self, x: i32, y: i32) -> i32 {
        i32::from(self.img.at((x + self.pad_x as i32) as usize, (y + self.pad_y as i32) as usize, 0))
    }
}

fn build_pyramid(img: &MatView<'_, u8>, p: &LkParams, max_level: usize) -> Result<Vec<Level>> {
    let (wx, wy) = (p.win_width, p.win_height);
    let mut levels = Vec::new();
    let mut current = img.to_mat();
    for level in 0..=max_level {
        let (w, h) = (current.width(), current.height());
        let padded = copy_make_border(&current, wy, wy, wx, wx, Border::Reflect101)?;
        levels.push(Level { width: w, height: h, pad_x: wx, pad_y: wy, img: padded });
        let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
        if nw <= wx || nh <= wy || level == max_level {
            break;
        }
        current = pyr_down(&current)?;
    }
    Ok(levels)
}

/// `calcScharrDeriv` with a zero border of `win` pixels: `(dx, dy)` as `i16` pairs.
fn scharr(level: &Level) -> (Vec<i16>, usize) {
    let (w, h) = (level.width, level.height);
    let (px, py) = (level.pad_x, level.pad_y);
    let stride = w + 2 * px;
    let mut d = vec![0i16; stride * (h + 2 * py) * 2];
    let at = |x: usize, y: usize| i32::from(level.img.at(x + px, y + py, 0));
    for y in 0..h {
        let (y0, y2) = (if y > 0 { y - 1 } else if h > 1 { 1 } else { 0 }, if y + 1 < h { y + 1 } else if h > 1 { h - 2 } else { 0 });
        let t0: Vec<i32> = (0..w).map(|x| (at(x, y0) + at(x, y2)) * 3 + at(x, y) * 10).collect();
        let t1: Vec<i32> = (0..w).map(|x| at(x, y2) - at(x, y0)).collect();
        let col = |x: isize| -> usize {
            if x < 0 {
                if w > 1 { 1 } else { 0 }
            } else if x as usize >= w {
                if w > 1 { w - 2 } else { 0 }
            } else {
                x as usize
            }
        };
        for x in 0..w as isize {
            let gx = t0[col(x + 1)] - t0[col(x - 1)];
            let gy = (t1[col(x + 1)] + t1[col(x - 1)]) * 3 + t1[x as usize] * 10;
            let i = ((y + py) * stride + x as usize + px) * 2;
            d[i] = gx as i16;
            d[i + 1] = gy as i16;
        }
    }
    (d, stride)
}

#[inline]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

fn weights(a: f32, b: f32) -> [i32; 4] {
    let s = (1 << W_BITS) as f32;
    let w00 = cv_round_f32((1.0 - a) * (1.0 - b) * s);
    let w01 = cv_round_f32(a * (1.0 - b) * s);
    let w10 = cv_round_f32((1.0 - a) * b * s);
    [w00, w01, w10, (1 << W_BITS) - w00 - w01 - w10]
}

/// `cv::calcOpticalFlowPyrLK(prev, next, points, None, winSize, maxLevel)`.
pub fn calc_optical_flow_pyr_lk<'a, 'b>(
    prev: impl Into<MatView<'a, u8>>,
    next: impl Into<MatView<'b, u8>>,
    points: &[Point2f],
    params: LkParams,
) -> Result<Vec<LkTrack>> {
    let (prev, next) = (prev.into(), next.into());
    if prev.size() != next.size() || prev.channels() != 1 || next.channels() != 1 {
        return Err(CvError::ShapeMismatch { op: "pyr_lk" });
    }
    if params.win_width <= 2 || params.win_height <= 2 {
        return Err(invalid("pyr_lk", "window must exceed 2x2"));
    }
    let prev_pyr = build_pyramid(&prev, &params, params.max_level)?;
    let next_pyr = build_pyramid(&next, &params, params.max_level)?;
    let max_level = prev_pyr.len().min(next_pyr.len()) - 1;
    let max_count = params.max_count.min(100);
    let eps = params.epsilon.clamp(0.0, 10.0);
    let eps2 = eps * eps;
    let mut tracks: Vec<LkTrack> = points.iter().map(|&p| LkTrack { point: p, found: true, error: 0.0 }).collect();
    for level in (0..=max_level).rev() {
        let (pi, ni) = (&prev_pyr[level], &next_pyr[level]);
        let (deriv, dstride) = scharr(pi);
        tracks.par_iter_mut().zip(points.par_iter()).for_each(|(t, &p0)| {
            track_point(t, p0, level, max_level, pi, ni, &deriv, dstride, &params, max_count, eps2);
        });
    }
    Ok(tracks)
}

#[allow(clippy::too_many_arguments)]
fn track_point(
    t: &mut LkTrack,
    p0: Point2f,
    level: usize,
    max_level: usize,
    pi: &Level,
    ni: &Level,
    deriv: &[i16],
    dstride: usize,
    p: &LkParams,
    max_count: usize,
    eps2: f64,
) {
    let (ww, wh) = (p.win_width as i32, p.win_height as i32);
    let half = Point2f::new((ww - 1) as f32 * 0.5, (wh - 1) as f32 * 0.5);
    let lscale = (1.0 / f64::from(1u32 << level)) as f32;
    let prev_pt = Point2f::new(p0.x * lscale, p0.y * lscale);
    let mut next_pt = if level == max_level { prev_pt } else { Point2f::new(t.point.x * 2.0, t.point.y * 2.0) };
    t.point = next_pt;
    let pp = Point2f::new(prev_pt.x - half.x, prev_pt.y - half.y);
    let (ix, iy) = (cv_floor(f64::from(pp.x)), cv_floor(f64::from(pp.y)));
    let (w, h) = (pi.width as i32, pi.height as i32);
    if ix < -ww || ix >= w || iy < -wh || iy >= h {
        if level == 0 {
            t.found = false;
            t.error = 0.0;
        }
        return;
    }
    let iw = weights(pp.x - ix as f32, pp.y - iy as f32);
    let n = (ww * wh) as usize;
    let mut ival = vec![0i32; n];
    let mut dval = vec![(0i32, 0i32); n];
    let (mut a11, mut a12, mut a22) = (0.0f32, 0.0f32, 0.0f32);
    let d = |x: i32, y: i32, c: usize| -> i32 {
        i32::from(deriv[(((y + pi.pad_y as i32) as usize) * dstride + (x + pi.pad_x as i32) as usize) * 2 + c])
    };
    for y in 0..wh {
        for x in 0..ww {
            let (sx, sy) = (ix + x, iy + y);
            let v = descale(pi.at(sx, sy) * iw[0] + pi.at(sx + 1, sy) * iw[1] + pi.at(sx, sy + 1) * iw[2] + pi.at(sx + 1, sy + 1) * iw[3], W_BITS - 5);
            let gx = descale(d(sx, sy, 0) * iw[0] + d(sx + 1, sy, 0) * iw[1] + d(sx, sy + 1, 0) * iw[2] + d(sx + 1, sy + 1, 0) * iw[3], W_BITS);
            let gy = descale(d(sx, sy, 1) * iw[0] + d(sx + 1, sy, 1) * iw[1] + d(sx, sy + 1, 1) * iw[2] + d(sx + 1, sy + 1, 1) * iw[3], W_BITS);
            let k = (y * ww + x) as usize;
            ival[k] = v;
            dval[k] = (gx, gy);
            a11 += (gx * gx) as f32;
            a12 += (gx * gy) as f32;
            a22 += (gy * gy) as f32;
        }
    }
    let (a11, a12, a22) = (a11 * FLT_SCALE, a12 * FLT_SCALE, a22 * FLT_SCALE);
    let det = a11 * a22 - a12 * a12;
    let min_eig = (a22 + a11 - ((a11 - a22) * (a11 - a22) + 4.0 * a12 * a12).sqrt()) / (2 * ww * wh) as f32;
    if f64::from(min_eig) < p.min_eig_threshold || det < f32::EPSILON {
        if level == 0 {
            t.found = false;
        }
        return;
    }
    let inv_det = 1.0 / det;
    next_pt = Point2f::new(next_pt.x - half.x, next_pt.y - half.y);
    let mut prev_delta = Point2f::default();
    let (jw, jh) = (ni.width as i32, ni.height as i32);
    for j in 0..max_count {
        let (jx, jy) = (cv_floor(f64::from(next_pt.x)), cv_floor(f64::from(next_pt.y)));
        if jx < -ww || jx >= jw || jy < -wh || jy >= jh {
            if level == 0 {
                t.found = false;
            }
            break;
        }
        let jwt = weights(next_pt.x - jx as f32, next_pt.y - jy as f32);
        let (mut b1, mut b2) = (0.0f32, 0.0f32);
        for y in 0..wh {
            for x in 0..ww {
                let (sx, sy) = (jx + x, jy + y);
                let v = descale(ni.at(sx, sy) * jwt[0] + ni.at(sx + 1, sy) * jwt[1] + ni.at(sx, sy + 1) * jwt[2] + ni.at(sx + 1, sy + 1) * jwt[3], W_BITS - 5);
                let k = (y * ww + x) as usize;
                let diff = v - ival[k];
                b1 += (diff * dval[k].0) as f32;
                b2 += (diff * dval[k].1) as f32;
            }
        }
        let (b1, b2) = (b1 * FLT_SCALE, b2 * FLT_SCALE);
        let delta = Point2f::new((a12 * b2 - a22 * b1) * inv_det, (a12 * b1 - a11 * b2) * inv_det);
        next_pt = Point2f::new(next_pt.x + delta.x, next_pt.y + delta.y);
        t.point = Point2f::new(next_pt.x + half.x, next_pt.y + half.y);
        if f64::from(delta.x) * f64::from(delta.x) + f64::from(delta.y) * f64::from(delta.y) <= eps2 {
            break;
        }
        if j > 0 && (delta.x + prev_delta.x).abs() < 0.01 && (delta.y + prev_delta.y).abs() < 0.01 {
            t.point = Point2f::new(t.point.x - delta.x * 0.5, t.point.y - delta.y * 0.5);
            break;
        }
        prev_delta = delta;
    }
    if t.found && level == 0 {
        let np = Point2f::new(t.point.x - half.x, t.point.y - half.y);
        let (jx, jy) = (cv_floor(f64::from(np.x)), cv_floor(f64::from(np.y)));
        if jx < -ww || jx >= jw || jy < -wh || jy >= jh {
            t.found = false;
            return;
        }
        let jwt = weights(np.x - jx as f32, np.y - jy as f32);
        let mut err = 0.0f32;
        for y in 0..wh {
            for x in 0..ww {
                let (sx, sy) = (jx + x, jy + y);
                let v = descale(ni.at(sx, sy) * jwt[0] + ni.at(sx + 1, sy) * jwt[1] + ni.at(sx, sy + 1) * jwt[2] + ni.at(sx + 1, sy + 1) * jwt[3], W_BITS - 5);
                err += ((v - ival[(y * ww + x) as usize]) as f32).abs();
            }
        }
        t.error = err * 1.0 / (32 * ww * wh) as f32;
    }
}

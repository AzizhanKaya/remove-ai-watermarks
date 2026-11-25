//! `cv::calcOpticalFlowFarneback` (box-filter variant, `flags = 0`), ported from `optflowgf.cpp`.

use crate::border::Border;
use crate::element::{Element, cv_floor, cv_round};
use crate::error::{CvError, Result, invalid};
use crate::filter::gaussian_blur;
use crate::mat::{Mat, MatView};
use crate::resize::resize;
use crate::types::Interpolation;

/// Farneback parameters, named like the OpenCV arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FarnebackParams {
    pub pyr_scale: f64,
    pub levels: usize,
    pub winsize: usize,
    pub iterations: usize,
    pub poly_n: usize,
    pub poly_sigma: f64,
}

impl Default for FarnebackParams {
    /// The package's call: `(0.5, 3, 15, 3, 5, 1.2, 0)`.
    fn default() -> Self {
        Self { pyr_scale: 0.5, levels: 3, winsize: 15, iterations: 3, poly_n: 5, poly_sigma: 1.2 }
    }
}

/// Dense flow from `prev` to `next`: a 2-channel `f32` field of `(dx, dy)`.
pub fn calc_optical_flow_farneback<'a, 'b, T: Element>(
    prev: impl Into<MatView<'a, T>>,
    next: impl Into<MatView<'b, T>>,
    params: FarnebackParams,
) -> Result<Mat<f32>> {
    let (prev, next) = (prev.into(), next.into());
    if prev.size() != next.size() || prev.channels() != 1 || next.channels() != 1 {
        return Err(CvError::ShapeMismatch { op: "farneback" });
    }
    if params.pyr_scale >= 1.0 || params.poly_n == 0 {
        return Err(invalid("farneback", "pyr_scale must be < 1 and poly_n positive"));
    }
    const MIN_SIZE: f64 = 32.0;
    let (w0, h0) = (prev.width() as f64, prev.height() as f64);
    let mut levels = 0;
    let mut scale = 1.0;
    while levels < params.levels {
        scale *= params.pyr_scale;
        if w0 * scale < MIN_SIZE || h0 * scale < MIN_SIZE {
            break;
        }
        levels += 1;
    }
    let images = [prev.to_mat().convert::<f32>(), next.to_mat().convert::<f32>()];
    let mut prev_flow: Option<Mat<f32>> = None;
    for k in (0..=levels).rev() {
        let scale = (0..k).fold(1.0, |s, _| s * params.pyr_scale);
        let sigma = (1.0 / scale - 1.0) * 0.5;
        let smooth = ((cv_round(sigma * 5.0) | 1).max(3)) as usize;
        let width = cv_round(w0 * scale) as usize;
        let height = cv_round(h0 * scale) as usize;
        let mut flow = match &prev_flow {
            None => Mat::zeros(width, height, 2),
            Some(pf) => {
                let mut f = resize(pf, width, height, Interpolation::Linear)?;
                let s = (1.0 / params.pyr_scale) as f32;
                f.data_mut().iter_mut().for_each(|v| *v *= s);
                f
            }
        };
        let mut r = Vec::with_capacity(2);
        for img in &images {
            let blurred = gaussian_blur(img, (smooth, smooth), sigma, sigma, Border::Reflect101)?;
            let small = resize(&blurred, width, height, Interpolation::Linear)?;
            r.push(poly_exp(&small, params.poly_n, params.poly_sigma));
        }
        let mut m = Mat::<f32>::zeros(width, height, 5);
        update_matrices(&r[0], &r[1], &flow, &mut m, 0, height);
        for i in 0..params.iterations {
            update_flow_blur(&r[0], &r[1], &mut flow, &mut m, params.winsize, i + 1 < params.iterations);
        }
        prev_flow = Some(flow);
    }
    prev_flow.ok_or_else(|| invalid("farneback", "no pyramid level"))
}

/// `FarnebackPrepareGaussian`: taps `g`, `x g`, `x^2 g` and four entries of the inverse moment matrix.
fn prepare_gaussian(n: usize, sigma: f64) -> (Vec<f32>, Vec<f32>, Vec<f32>, [f64; 4]) {
    let sigma = if sigma < f64::from(f32::EPSILON) { n as f64 * 0.3 } else { sigma };
    let ni = n as i64;
    let len = 2 * n + 1;
    let mut g = vec![0.0f32; len];
    let mut s = 0.0f64;
    for x in -ni..=ni {
        let v = (f64::from((-x * x) as i32) / (2.0 * sigma * sigma)).exp() as f32;
        g[(x + ni) as usize] = v;
        s += f64::from(v);
    }
    let s = 1.0 / s;
    let mut xg = vec![0.0f32; len];
    let mut xxg = vec![0.0f32; len];
    for x in -ni..=ni {
        let i = (x + ni) as usize;
        g[i] = (f64::from(g[i]) * s) as f32;
        xg[i] = (x as f64 * f64::from(g[i])) as f32;
        xxg[i] = ((x * x) as f64 * f64::from(g[i])) as f32;
    }
    let mut gm = [[0.0f64; 6]; 6];
    for y in -ni..=ni {
        for x in -ni..=ni {
            let gyx = g[(y + ni) as usize] * g[(x + ni) as usize];
            let (xf, yf) = (x as f32, y as f32);
            gm[0][0] += f64::from(gyx);
            gm[1][1] += f64::from(gyx * xf * xf);
            gm[3][3] += f64::from(gyx * xf * xf * xf * xf);
            gm[5][5] += f64::from(gyx * xf * xf * yf * yf);
        }
    }
    gm[2][2] = gm[1][1];
    gm[0][3] = gm[1][1];
    gm[0][4] = gm[1][1];
    gm[3][0] = gm[1][1];
    gm[4][0] = gm[1][1];
    gm[4][4] = gm[3][3];
    gm[3][4] = gm[5][5];
    gm[4][3] = gm[5][5];
    let inv = cholesky_inverse(gm);
    (g, xg, xxg, [inv[1][1], inv[0][3], inv[3][3], inv[5][5]])
}

/// `Mat::inv(DECOMP_CHOLESKY)` following `CholImpl` (reciprocal diagonal, two triangular solves).
fn cholesky_inverse(a: [[f64; 6]; 6]) -> [[f64; 6]; 6] {
    const M: usize = 6;
    let mut l = a;
    for i in 0..M {
        for j in 0..i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            l[i][j] = s * l[j][j];
        }
        let mut s = a[i][i];
        for k in 0..i {
            s -= l[i][k] * l[i][k];
        }
        l[i][i] = 1.0 / s.sqrt();
    }
    let mut b = [[0.0f64; 6]; 6];
    for (i, row) in b.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for i in 0..M {
        for j in 0..M {
            let mut s = b[i][j];
            for k in 0..i {
                s -= l[i][k] * b[k][j];
            }
            b[i][j] = s * l[i][i];
        }
    }
    for i in (0..M).rev() {
        for j in 0..M {
            let mut s = b[i][j];
            for k in (i + 1..M).rev() {
                s -= l[k][i] * b[k][j];
            }
            b[i][j] = s * l[i][i];
        }
    }
    b
}

/// `FarnebackPolyExp`: per-pixel quadratic coefficients `(r2..r6)`, 5 channels.
fn poly_exp(src: &Mat<f32>, n: usize, sigma: f64) -> Mat<f32> {
    let (g, xg, xxg, [ig11, ig03, ig33, ig55]) = prepare_gaussian(n, sigma);
    let (w, h) = (src.width(), src.height());
    let mut dst = Mat::<f32>::zeros(w, h, 5);
    let pad = n * 3;
    let mut row = vec![0.0f32; (w + 2 * n) * 3];
    for y in 0..h {
        let (g0, srow0) = (g[n], src.row(y));
        for x in 0..w {
            row[pad + x * 3] = srow0[x] * g0;
            row[pad + x * 3 + 1] = 0.0;
            row[pad + x * 3 + 2] = 0.0;
        }
        for k in 1..=n {
            let (g0, g1, g2) = (g[n + k], xg[n + k], xxg[n + k]);
            let s0 = src.row(y.saturating_sub(k));
            let s1 = src.row((y + k).min(h - 1));
            for x in 0..w {
                let p = s0[x] + s1[x];
                let r = &mut row[pad + x * 3..pad + x * 3 + 3];
                r[0] += g0 * p;
                r[1] += g1 * (s1[x] - s0[x]);
                r[2] += g2 * p;
            }
        }
        for x in 0..pad {
            row[pad - 1 - x] = row[pad + 2 - x];
            row[pad + w * 3 + x] = row[pad + w * 3 + x - 3];
        }
        let drow = dst.row_mut(y);
        let at = |i: isize| f64::from(row[(pad as isize + i) as usize]);
        for x in 0..w {
            let xi = x as isize;
            let g0 = f64::from(g[n]);
            let mut b1 = at(xi * 3) * g0;
            let mut b2 = 0.0;
            let mut b3 = at(xi * 3 + 1) * g0;
            let mut b4 = 0.0;
            let mut b5 = at(xi * 3 + 2) * g0;
            let mut b6 = 0.0;
            for k in 1..=n {
                let ki = k as isize;
                let (p, m) = ((xi + ki) * 3, (xi - ki) * 3);
                let tg = f64::from(row[(pad as isize + p) as usize] + row[(pad as isize + m) as usize]);
                let gk = f64::from(g[n + k]);
                b1 += tg * gk;
                b4 += tg * f64::from(xxg[n + k]);
                b2 += f64::from(row[(pad as isize + p) as usize] - row[(pad as isize + m) as usize]) * f64::from(xg[n + k]);
                b3 += f64::from(row[(pad as isize + p + 1) as usize] + row[(pad as isize + m + 1) as usize]) * gk;
                b6 += f64::from(row[(pad as isize + p + 1) as usize] - row[(pad as isize + m + 1) as usize]) * f64::from(xg[n + k]);
                b5 += f64::from(row[(pad as isize + p + 2) as usize] + row[(pad as isize + m + 2) as usize]) * gk;
            }
            let d = &mut drow[x * 5..x * 5 + 5];
            d[1] = (b2 * ig11) as f32;
            d[0] = (b3 * ig11) as f32;
            d[3] = (b1 * ig03 + b4 * ig33) as f32;
            d[2] = (b1 * ig03 + b5 * ig33) as f32;
            d[4] = (b6 * ig55) as f32;
        }
    }
    dst
}

/// `FarnebackUpdateMatrices` for rows `[y0, y1)`.
fn update_matrices(r0: &Mat<f32>, r1: &Mat<f32>, flow: &Mat<f32>, m: &mut Mat<f32>, y0: usize, y1: usize) {
    const BORDER: usize = 5;
    const WEIGHTS: [f32; BORDER] = [0.14, 0.14, 0.4472, 0.4472, 0.4472];
    let (w, h) = (flow.width(), flow.height());
    let r1d = r1.data();
    let step = w * 5;
    for y in y0..y1 {
        let frow = flow.row(y);
        let r0row = r0.row(y);
        for x in 0..w {
            let (dx, dy) = (frow[x * 2], frow[x * 2 + 1]);
            let (mut fx, mut fy) = (x as f32 + dx, y as f32 + dy);
            let (x1, y1i) = (cv_floor(f64::from(fx)), cv_floor(f64::from(fy)));
            fx -= x1 as f32;
            fy -= y1i as f32;
            let base = &r0row[x * 5..x * 5 + 5];
            let (mut r2, mut r3, r4, r5, r6);
            if (x1 as u32) < (w as u32).wrapping_sub(1) && (y1i as u32) < (h as u32).wrapping_sub(1) {
                let p = (y1i as usize) * step + x1 as usize * 5;
                let (a00, a01, a10, a11) = ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy);
                let s = |c: usize| a00 * r1d[p + c] + a01 * r1d[p + 5 + c] + a10 * r1d[p + step + c] + a11 * r1d[p + step + 5 + c];
                r2 = s(0);
                r3 = s(1);
                r4 = (base[2] + s(2)) * 0.5;
                r5 = (base[3] + s(3)) * 0.5;
                r6 = (base[4] + s(4)) * 0.25;
            } else {
                r2 = 0.0;
                r3 = 0.0;
                r4 = base[2];
                r5 = base[3];
                r6 = base[4] * 0.5;
            }
            r2 = (base[0] - r2) * 0.5;
            r3 = (base[1] - r3) * 0.5;
            r2 += r4 * dy + r6 * dx;
            r3 += r6 * dy + r5 * dx;
            let (mut r4, mut r5, mut r6) = (r4, r5, r6);
            if x.wrapping_sub(BORDER) >= w.wrapping_sub(BORDER * 2) || y.wrapping_sub(BORDER) >= h.wrapping_sub(BORDER * 2) {
                let low = |i: usize| if i < BORDER { WEIGHTS[i] } else { 1.0 };
                let high = |i: usize, n: usize| if i + BORDER >= n { WEIGHTS[n - i - 1] } else { 1.0 };
                let scale = low(x) * high(x, w) * low(y) * high(y, h);
                r2 *= scale;
                r3 *= scale;
                r4 *= scale;
                r5 *= scale;
                r6 *= scale;
            }
            let out = &mut m.row_mut(y)[x * 5..x * 5 + 5];
            out[0] = r4 * r4 + r6 * r6;
            out[1] = (r4 + r5) * r6;
            out[2] = r5 * r5 + r6 * r6;
            out[3] = r4 * r2 + r6 * r3;
            out[4] = r6 * r2 + r5 * r3;
        }
    }
}

/// `FarnebackUpdateFlow_Blur`: box-filtered normal equations, matrices refreshed in stripes.
fn update_flow_blur(r0: &Mat<f32>, r1: &Mat<f32>, flow: &mut Mat<f32>, m: &mut Mat<f32>, block: usize, update: bool) {
    let (w, h) = (flow.width(), flow.height());
    let half = block / 2;
    let min_stripe = (1024 / w).max(block);
    let scale = 1.0 / (block * block) as f64;
    let off = (half + 1) * 5;
    let mut vsum = vec![0.0f64; (w + half * 2 + 2) * 5];
    let row0 = m.row(0);
    for x in 0..w * 5 {
        vsum[off + x] = f64::from(row0[x]) * (half + 2) as f64;
    }
    for y in 1..half {
        let r = m.row(y.min(h - 1));
        for x in 0..w * 5 {
            vsum[off + x] += f64::from(r[x]);
        }
    }
    let mut y0 = 0usize;
    for y in 0..h {
        {
            let s0 = m.row((y as isize - half as isize - 1).max(0) as usize);
            let s1 = m.row((y + half).min(h - 1));
            for x in 0..w * 5 {
                vsum[off + x] += f64::from(s1[x]) - f64::from(s0[x]);
            }
        }
        for x in 0..off {
            vsum[off - 1 - x] = vsum[off + 4 - x];
            vsum[off + w * 5 + x] = vsum[off + w * 5 + x - 5];
        }
        let v = |i: isize| vsum[(off as isize + i) as usize];
        let k = (half + 2) as f64;
        let (mut g11, mut g12, mut g22, mut h1, mut h2) = (v(0) * k, v(1) * k, v(2) * k, v(3) * k, v(4) * k);
        for x in 1..half as isize {
            g11 += v(x * 5);
            g12 += v(x * 5 + 1);
            g22 += v(x * 5 + 2);
            h1 += v(x * 5 + 3);
            h2 += v(x * 5 + 4);
        }
        let frow = flow.row_mut(y);
        let hm = half as isize;
        for x in 0..w as isize {
            g11 += v((x + hm) * 5) - v((x - hm) * 5 - 5);
            g12 += v((x + hm) * 5 + 1) - v((x - hm) * 5 - 4);
            g22 += v((x + hm) * 5 + 2) - v((x - hm) * 5 - 3);
            h1 += v((x + hm) * 5 + 3) - v((x - hm) * 5 - 2);
            h2 += v((x + hm) * 5 + 4) - v((x - hm) * 5 - 1);
            let (a, b, c, d, e) = (g11 * scale, g12 * scale, g22 * scale, h1 * scale, h2 * scale);
            let idet = 1.0 / (a * c - b * b + 1e-3);
            frow[x as usize * 2] = ((a * e - b * d) * idet) as f32;
            frow[x as usize * 2 + 1] = ((c * d - b * e) * idet) as f32;
        }
        let y1 = if y == h - 1 { h as isize } else { y as isize - block as isize };
        if update && (y1 == h as isize || y1 >= (y0 + min_stripe) as isize) {
            update_matrices(r0, r1, flow, m, y0, y1 as usize);
            y0 = y1 as usize;
        }
    }
}

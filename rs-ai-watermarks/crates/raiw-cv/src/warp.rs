//! `cv::remap`, `cv::warpAffine` and `cv::getRotationMatrix2D`: 1/32-pixel coordinate quantization
//! and OpenCV's interpolation tables (15-bit fixed point for 8-bit images).

use std::sync::OnceLock;

use rayon::prelude::*;

use crate::border::Border;
use crate::element::{Depth, Element, cv_round, cv_round_f32, saturate_i16};
use crate::error::{CvError, Result, invalid};
use crate::mat::{Mat, MatView};
use crate::types::{Interpolation, Point2f};

const INTER_BITS: i32 = 5;
const TAB: i32 = 1 << INTER_BITS;
const COEF_BITS: i32 = 15;
const COEF_SCALE: i32 = 1 << COEF_BITS;

/// Element types `remap` / `warp_affine` accept; 8-bit images interpolate in fixed point.
pub trait WarpElement: Element {}
impl WarpElement for u8 {}
impl WarpElement for u16 {}
impl WarpElement for i16 {}
impl WarpElement for f32 {}

fn lanczos4_1d(x: f32) -> [f32; 8] {
    const S45: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const CS: [[f64; 2]; 8] =
        [[1.0, 0.0], [-S45, -S45], [0.0, 1.0], [S45, -S45], [-1.0, 0.0], [S45, S45], [0.0, -1.0], [-S45, S45]];
    let mut out = [0.0f32; 8];
    if x < f32::EPSILON {
        out[3] = 1.0;
        return out;
    }
    let y0 = f64::from(-(x + 3.0)) * std::f64::consts::PI * 0.25;
    let (s0, c0) = y0.sin_cos();
    let mut sum = 0.0f32;
    for (i, c) in out.iter_mut().enumerate() {
        let y = f64::from(-(x + 3.0 - i as f32)) * std::f64::consts::PI * 0.25;
        *c = ((CS[i][0] * s0 + CS[i][1] * c0) / (y * y)) as f32;
        sum += *c;
    }
    let inv = 1.0 / sum;
    for c in &mut out {
        *c *= inv;
    }
    out
}

fn kernel_1d(ksize: usize, x: f32) -> Vec<f32> {
    match ksize {
        2 => vec![1.0 - x, x],
        4 => {
            const A: f32 = -0.75;
            let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
            let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
            let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
            vec![c0, c1, c2, 1.0 - c0 - c1 - c2]
        }
        _ => lanczos4_1d(x).to_vec(),
    }
}

/// `initInterTab2D`: float and normalized fixed-point 2D kernels for all 32x32 sub-pixel phases.
struct Tables {
    ksize: usize,
    float: Vec<f32>,
    fixed: Vec<i16>,
}

fn build_tables(ksize: usize) -> Tables {
    let scale = 1.0f32 / TAB as f32;
    let one_d: Vec<Vec<f32>> = (0..TAB).map(|i| kernel_1d(ksize, i as f32 * scale)).collect();
    let kk = ksize * ksize;
    let phases = (TAB * TAB) as usize;
    let mut float = vec![0.0f32; phases * kk];
    let mut fixed = vec![0i16; phases * kk];
    for i in 0..TAB as usize {
        for j in 0..TAB as usize {
            let base = (i * TAB as usize + j) * kk;
            let (tab, itab) = (&mut float[base..base + kk], &mut fixed[base..base + kk]);
            let mut isum = 0i32;
            for k1 in 0..ksize {
                let vy = one_d[i][k1];
                for k2 in 0..ksize {
                    let v = vy * one_d[j][k2];
                    tab[k1 * ksize + k2] = v;
                    itab[k1 * ksize + k2] = saturate_i16(cv_round_f32(v * COEF_SCALE as f32));
                    isum += i32::from(itab[k1 * ksize + k2]);
                }
            }
            if isum != COEF_SCALE {
                // OpenCV scans the central 2x2 taps; for the 2x2 kernel this reaches into the next,
                // still-zero phase block, and corrections aimed there are later overwritten.
                let diff = isum - COEF_SCALE;
                let half = ksize / 2;
                let get = |i: usize| if i < kk { itab[i] } else { 0 };
                let (mut big, mut small) = (half * ksize + half, half * ksize + half);
                for k1 in half..half + 2 {
                    for k2 in half..half + 2 {
                        let i = k1 * ksize + k2;
                        if get(i) < get(small) {
                            small = i;
                        } else if get(i) > get(big) {
                            big = i;
                        }
                    }
                }
                let i = if diff < 0 { big } else { small };
                if i < kk {
                    itab[i] = (i32::from(itab[i]) - diff) as i16;
                }
            }
        }
    }
    Tables { ksize, float, fixed }
}

fn tables(interpolation: Interpolation) -> &'static Tables {
    static LINEAR: OnceLock<Tables> = OnceLock::new();
    static CUBIC: OnceLock<Tables> = OnceLock::new();
    static LANCZOS: OnceLock<Tables> = OnceLock::new();
    match interpolation {
        Interpolation::Cubic => CUBIC.get_or_init(|| build_tables(4)),
        Interpolation::Lanczos4 => LANCZOS.get_or_init(|| build_tables(8)),
        _ => LINEAR.get_or_init(|| build_tables(2)),
    }
}

/// A quantized source coordinate: integer pixel plus the `(y & 31) * 32 + (x & 31)` phase.
#[derive(Clone, Copy)]
struct Coord {
    x: i32,
    y: i32,
    phase: usize,
}

impl Coord {
    /// From fixed-point coordinates with `INTER_BITS` fractional bits.
    fn from_fixed(x: i32, y: i32) -> Self {
        Self { x: x >> INTER_BITS, y: y >> INTER_BITS, phase: ((y & (TAB - 1)) * TAB + (x & (TAB - 1))) as usize }
    }
}

struct Sampler<'s, 'a, T> {
    src: &'s MatView<'a, T>,
    border: Border,
    cval: Vec<T>,
    fixed_point: bool,
    tab: &'static Tables,
}

impl<'s, 'a, T: WarpElement> Sampler<'s, 'a, T> {
    fn new(src: &'s MatView<'a, T>, border: Border, interpolation: Interpolation) -> Self {
        let cval = (0..src.channels()).map(|c| T::from_f64(border.constant_value(c & 3))).collect();
        Self { src, border, cval, fixed_point: T::DEPTH == Depth::U8, tab: tables(interpolation) }
    }

    #[inline]
    fn at(&self, x: i32, y: i32, c: usize) -> T {
        self.src.at(x as usize, y as usize, c)
    }

    #[inline]
    fn index(&self, p: i32, len: usize) -> Option<i32> {
        self.border.interpolate(p as isize, len).map(|v| v as i32)
    }

    fn is_constant(&self) -> bool {
        matches!(self.border, Border::Constant(_))
    }

    #[inline]
    fn cast_fixed(sum: i32) -> T {
        T::from_i32((sum + (1 << (COEF_BITS - 1))) >> COEF_BITS)
    }

    fn nearest(&self, x: i32, y: i32, out: &mut [T]) {
        let (w, h) = (self.src.width(), self.src.height());
        if (0..w as i32).contains(&x) && (0..h as i32).contains(&y) {
            out.copy_from_slice(self.src.pixel(x as usize, y as usize));
            return;
        }
        let picked = match self.border {
            Border::Replicate => Some((x.clamp(0, w as i32 - 1), y.clamp(0, h as i32 - 1))),
            Border::Constant(_) => None,
            _ => self.index(x, w).zip(self.index(y, h)),
        };
        match picked {
            Some((sx, sy)) => out.copy_from_slice(self.src.pixel(sx as usize, sy as usize)),
            None => out.copy_from_slice(&self.cval),
        }
    }

    fn bilinear(&self, p: Coord, out: &mut [T]) {
        let (w, h) = (self.src.width() as i32, self.src.height() as i32);
        let (sx, sy) = (p.x, p.y);
        if self.is_constant() && (sx >= w || sx + 1 < 0 || sy >= h || sy + 1 < 0) {
            out.copy_from_slice(&self.cval);
            return;
        }
        let pick = |v: i32, len: i32| -> Option<i32> {
            if self.border == Border::Replicate { Some(v.clamp(0, len - 1)) } else { self.index(v, len as usize) }
        };
        let (x0, x1, y0, y1) = (pick(sx, w), pick(sx + 1, w), pick(sy, h), pick(sy + 1, h));
        for (c, o) in out.iter_mut().enumerate() {
            let v = |x: Option<i32>, y: Option<i32>| match (x, y) {
                (Some(x), Some(y)) => self.at(x, y, c),
                _ => self.cval[c],
            };
            let px = [v(x0, y0), v(x1, y0), v(x0, y1), v(x1, y1)];
            *o = if self.fixed_point {
                let wt = &self.tab.fixed[p.phase * 4..p.phase * 4 + 4];
                let sum: i32 = px.iter().zip(wt).map(|(t, &k)| t.to_f64() as i32 * i32::from(k)).sum();
                Self::cast_fixed(sum)
            } else {
                let wt = &self.tab.float[p.phase * 4..p.phase * 4 + 4];
                let sum = px[0].to_f32() * wt[0] + px[1].to_f32() * wt[1] + px[2].to_f32() * wt[2] + px[3].to_f32() * wt[3];
                T::from_f32(sum)
            };
        }
    }

    /// `remapBicubic` / `remapLanczos4`: an interior path and a border path with its own grouping.
    fn window(&self, p: Coord, out: &mut [T]) {
        let k = self.tab.ksize as i32;
        let kk = self.tab.ksize * self.tab.ksize;
        let (w, h) = (self.src.width() as i32, self.src.height() as i32);
        let (sx, sy) = (p.x - (k / 2 - 1), p.y - (k / 2 - 1));
        let fw = &self.tab.fixed[p.phase * kk..(p.phase + 1) * kk];
        let ww = &self.tab.float[p.phase * kk..(p.phase + 1) * kk];
        let tap = |r: i32, j: i32| (r * k + j) as usize;
        let interior = (sx as u32) < (w - (k - 1)).max(0) as u32 && (sy as u32) < (h - (k - 1)).max(0) as u32;
        if interior {
            for (c, o) in out.iter_mut().enumerate() {
                *o = if self.fixed_point {
                    let sum: i32 = (0..k)
                        .map(|r| {
                            (0..k).map(|j| self.at(sx + j, sy + r, c).to_f64() as i32 * i32::from(fw[tap(r, j)])).sum::<i32>()
                        })
                        .sum();
                    Self::cast_fixed(sum)
                } else {
                    let mut sum = 0.0f32;
                    for r in 0..k {
                        let mut row = self.at(sx, sy + r, c).to_f32() * ww[tap(r, 0)];
                        for j in 1..k {
                            row += self.at(sx + j, sy + r, c).to_f32() * ww[tap(r, j)];
                        }
                        sum += row;
                    }
                    T::from_f32(sum)
                };
            }
            return;
        }
        if self.is_constant() && (sx >= w || sx + k <= 0 || sy >= h || sy + k <= 0) {
            out.copy_from_slice(&self.cval);
            return;
        }
        let xs: Vec<Option<i32>> = (0..k).map(|i| self.index(sx + i, w as usize)).collect();
        let ys: Vec<Option<i32>> = (0..k).map(|i| self.index(sy + i, h as usize)).collect();
        for (c, o) in out.iter_mut().enumerate() {
            let cv = self.cval[c];
            *o = if self.fixed_point {
                let cvi = cv.to_f64() as i32;
                let mut sum = cvi * COEF_SCALE;
                for (i, y) in ys.iter().enumerate() {
                    let Some(y) = *y else { continue };
                    for (j, x) in xs.iter().enumerate() {
                        if let Some(x) = *x {
                            sum += (self.at(x, y, c).to_f64() as i32 - cvi) * i32::from(fw[tap(i as i32, j as i32)]);
                        }
                    }
                }
                Self::cast_fixed(sum)
            } else {
                let cvf = cv.to_f32();
                let mut sum = cvf;
                for (i, y) in ys.iter().enumerate() {
                    let Some(y) = *y else { continue };
                    for (j, x) in xs.iter().enumerate() {
                        if let Some(x) = *x {
                            sum += (self.at(x, y, c).to_f32() - cvf) * ww[tap(i as i32, j as i32)];
                        }
                    }
                }
                T::from_f32(sum)
            };
        }
    }

    fn sample(&self, interpolation: Interpolation, p: Coord, out: &mut [T]) {
        match interpolation {
            Interpolation::Cubic | Interpolation::Lanczos4 => self.window(p, out),
            _ => self.bilinear(p, out),
        }
    }
}

/// OpenCV maps `INTER_AREA` to bilinear for warps.
fn warp_interpolation(interpolation: Interpolation) -> Interpolation {
    if interpolation == Interpolation::Area { Interpolation::Linear } else { interpolation }
}

/// `cv::remap(src, map_x, map_y, interpolation, borderMode, borderValue)` with float maps.
pub fn remap<'a, 'b, 'c, T: WarpElement>(
    src: impl Into<MatView<'a, T>>,
    map_x: impl Into<MatView<'b, f32>>,
    map_y: impl Into<MatView<'c, f32>>,
    interpolation: Interpolation,
    border: Border,
) -> Result<Mat<T>> {
    let (src, mx, my) = (src.into(), map_x.into(), map_y.into());
    if mx.size() != my.size() || mx.channels() != 1 || my.channels() != 1 {
        return Err(CvError::ShapeMismatch { op: "remap" });
    }
    if src.is_empty() {
        return Err(invalid("remap", "empty source"));
    }
    let interpolation = warp_interpolation(interpolation);
    let sampler = Sampler::new(&src, border, interpolation);
    let cn = src.channels();
    let mut dst = Mat::zeros(mx.width(), mx.height(), cn);
    dst.par_rows_mut().enumerate().for_each(|(y, row)| {
        for ((px, &fx), &fy) in row.chunks_exact_mut(cn).zip(mx.row(y)).zip(my.row(y)) {
            if interpolation == Interpolation::Nearest {
                sampler.nearest(i32::from(saturate_i16(cv_round_f32(fx))), i32::from(saturate_i16(cv_round_f32(fy))), px);
            } else {
                let (sx, sy) = (cv_round_f32(fx * TAB as f32), cv_round_f32(fy * TAB as f32));
                let mut c = Coord::from_fixed(sx, sy);
                c.x = i32::from(saturate_i16(c.x));
                c.y = i32::from(saturate_i16(c.y));
                sampler.sample(interpolation, c, px);
            }
        }
    });
    Ok(dst)
}

/// A 2x3 affine matrix, row-major (`[[a, b, c], [d, e, f]]`).
pub type Affine = [[f64; 3]; 2];

/// `cv::getRotationMatrix2D(center, angle_degrees, scale)`.
pub fn get_rotation_matrix_2d(center: Point2f, angle: f64, scale: f64) -> Affine {
    let angle = angle * std::f64::consts::PI / 180.0;
    let alpha = angle.cos() * scale;
    let beta = angle.sin() * scale;
    let (cx, cy) = (f64::from(center.x), f64::from(center.y));
    [[alpha, beta, (1.0 - alpha) * cx - beta * cy], [-beta, alpha, beta * cx + (1.0 - alpha) * cy]]
}

/// `cv::invertAffineTransform`.
pub fn invert_affine(m: &Affine) -> Affine {
    let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22) = (m[1][1] * d, m[0][0] * d);
    let (a12, a21) = (m[0][1] * -d, m[1][0] * -d);
    let b1 = -a11 * m[0][2] - a12 * m[1][2];
    let b2 = -a21 * m[0][2] - a22 * m[1][2];
    [[a11, a12, b1], [a21, a22, b2]]
}

/// `cv::warpAffine(src, m, (width, height), flags=interpolation, borderMode, borderValue)`;
/// `m` maps source to destination (OpenCV's default, without `WARP_INVERSE_MAP`).
pub fn warp_affine<'a, T: WarpElement>(
    src: impl Into<MatView<'a, T>>,
    m: &Affine,
    width: usize,
    height: usize,
    interpolation: Interpolation,
    border: Border,
) -> Result<Mat<T>> {
    let src = src.into();
    if src.is_empty() || width == 0 || height == 0 {
        return Err(invalid("warp_affine", "empty source or destination"));
    }
    let interpolation = warp_interpolation(interpolation);
    let inv = invert_affine(m);
    const AB_BITS: i32 = 10;
    let ab_scale = f64::from(1 << AB_BITS);
    let round_delta = if interpolation == Interpolation::Nearest { (1 << AB_BITS) / 2 } else { (1 << AB_BITS) / TAB / 2 };
    let adelta: Vec<i32> = (0..width).map(|x| cv_round(inv[0][0] * x as f64 * ab_scale)).collect();
    let bdelta: Vec<i32> = (0..width).map(|x| cv_round(inv[1][0] * x as f64 * ab_scale)).collect();
    let sampler = Sampler::new(&src, border, interpolation);
    let cn = src.channels();
    let mut dst = Mat::zeros(width, height, cn);
    dst.par_rows_mut().enumerate().for_each(|(y, row)| {
        let x0 = cv_round((inv[0][1] * y as f64 + inv[0][2]) * ab_scale).wrapping_add(round_delta);
        let y0 = cv_round((inv[1][1] * y as f64 + inv[1][2]) * ab_scale).wrapping_add(round_delta);
        for (x, px) in row.chunks_exact_mut(cn).enumerate() {
            let (fx, fy) = (x0.wrapping_add(adelta[x]), y0.wrapping_add(bdelta[x]));
            if interpolation == Interpolation::Nearest {
                let (sx, sy) = (i32::from(saturate_i16(fx >> AB_BITS)), i32::from(saturate_i16(fy >> AB_BITS)));
                sampler.nearest(sx, sy, px);
            } else {
                let (fx, fy) = (fx >> (AB_BITS - INTER_BITS), fy >> (AB_BITS - INTER_BITS));
                let mut c = Coord::from_fixed(fx, fy);
                c.x = i32::from(saturate_i16(c.x));
                c.y = i32::from(saturate_i16(c.y));
                sampler.sample(interpolation, c, px);
            }
        }
    });
    Ok(dst)
}

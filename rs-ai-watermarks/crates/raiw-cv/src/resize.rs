//! `cv::resize`, reproducing `hal::resize` table construction and fixed-point arithmetic.

use rayon::prelude::*;

use crate::element::{Element, cv_floor, cv_round, saturate_i16, saturate_u8};
use crate::error::{Result, invalid};
use crate::mat::{Mat, MatView};
use crate::types::Interpolation;

const COEF_BITS: i32 = 11;
const COEF_SCALE: f32 = (1 << COEF_BITS) as f32;
const MIN_PARALLEL_ELEMENTS: usize = 1 << 15;

/// Element types `resize` accepts, with OpenCV's buffer and coefficient types.
pub trait Resizable: Element {
    #[doc(hidden)]
    type W: Copy + Default + Send + Sync;
    #[doc(hidden)]
    type A: Copy + Default + Send + Sync;
    #[doc(hidden)]
    type AreaW: AreaAccum;
    #[doc(hidden)]
    fn coef(c: f32) -> Self::A;
    #[doc(hidden)]
    fn mul(s: Self, a: Self::A) -> Self::W;
    #[doc(hidden)]
    fn one(s: Self) -> Self::W;
    #[doc(hidden)]
    fn add(a: Self::W, b: Self::W) -> Self::W;
    #[doc(hidden)]
    fn wmul(w: Self::W, b: Self::A) -> Self::W;
    #[doc(hidden)]
    fn cast(w: Self::W) -> Self;
    #[doc(hidden)]
    fn vlinear(s0: Self::W, s1: Self::W, b0: Self::A, b1: Self::A) -> Self {
        Self::cast(Self::add(Self::wmul(s0, b0), Self::wmul(s1, b1)))
    }
    /// Integer 2x2 area averaging uses `(sum + 2) >> 2` instead of float scaling.
    #[doc(hidden)]
    const ROUNDED_QUAD: bool;
    /// Block width of OpenCV's SIMD vertical pass for a `ksize`-tap window (0 = scalar only).
    #[doc(hidden)]
    fn simd_lanes(_ksize: usize) -> usize {
        0
    }
    /// One `float` term `S * b` of OpenCV's SIMD vertical pass `S0*b0 + (S1*b1 + (... + Sn*bn))`.
    #[doc(hidden)]
    fn vterm(_w: Self::W, _b: Self::A) -> f32 {
        0.0
    }
    /// Store a SIMD-pass `float` sum (`v_round` + saturating packs for integers).
    #[doc(hidden)]
    fn vfinish(_v: f32) -> Self {
        Self::ZERO
    }
    /// `(a + b + c + d + 2) >> 2`, the 2x2 integer area average.
    #[doc(hidden)]
    fn quad(a: Self, b: Self, c: Self, d: Self) -> Self {
        Self::from_i32(((a.to_f64() + b.to_f64() + c.to_f64() + d.to_f64()) as i32 + 2) >> 2)
    }
}

/// Accumulator of the area resamplers (`float` or `double`).
pub trait AreaAccum:
    Copy + Default + Send + Sync + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    fn of<T: Element>(v: T) -> Self;
    fn of_f32(v: f32) -> Self;
    fn store<T: Element>(self) -> T;
    #[must_use]
    fn div_count(self, count: usize) -> Self;
}

impl AreaAccum for f32 {
    #[inline]
    fn of<T: Element>(v: T) -> Self {
        v.to_f32()
    }
    #[inline]
    fn of_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn store<T: Element>(self) -> T {
        T::from_f32(self)
    }
    #[inline]
    fn div_count(self, count: usize) -> Self {
        self / count as f32
    }
}

impl AreaAccum for f64 {
    #[inline]
    fn of<T: Element>(v: T) -> Self {
        v.to_f64()
    }
    #[inline]
    fn of_f32(v: f32) -> Self {
        f64::from(v)
    }
    #[inline]
    fn store<T: Element>(self) -> T {
        T::from_f64(self)
    }
    #[inline]
    fn div_count(self, count: usize) -> Self {
        f64::from(self as f32 / count as f32)
    }
}

impl Resizable for u8 {
    type W = i32;
    type A = i16;
    type AreaW = f32;
    const ROUNDED_QUAD: bool = true;

    #[inline]
    fn coef(c: f32) -> i16 {
        saturate_i16(crate::element::cv_round_f32(c * COEF_SCALE))
    }
    #[inline]
    fn mul(s: u8, a: i16) -> i32 {
        i32::from(s) * i32::from(a)
    }
    #[inline]
    fn one(s: u8) -> i32 {
        i32::from(s) << COEF_BITS
    }
    #[inline]
    fn add(a: i32, b: i32) -> i32 {
        a.wrapping_add(b)
    }
    #[inline]
    fn wmul(w: i32, b: i16) -> i32 {
        w.wrapping_mul(i32::from(b))
    }
    #[inline]
    fn cast(w: i32) -> u8 {
        saturate_u8(w.wrapping_add(1 << (COEF_BITS * 2 - 1)) >> (COEF_BITS * 2))
    }
    #[inline]
    fn vlinear(s0: i32, s1: i32, b0: i16, b1: i16) -> u8 {
        let b0 = i32::from(b0);
        let b1 = i32::from(b1);
        ((((b0 * (s0 >> 4)) >> 16) + ((b1 * (s1 >> 4)) >> 16) + 2) >> 2) as u8
    }
    fn simd_lanes(ksize: usize) -> usize {
        if ksize == 4 { 8 } else { 0 }
    }
    #[inline]
    fn vterm(w: i32, b: i16) -> f32 {
        let scale = 1.0f32 / (1 << (2 * COEF_BITS)) as f32;
        w as f32 * (f32::from(b) * scale)
    }
    #[inline]
    fn vfinish(v: f32) -> u8 {
        saturate_u8(crate::element::cv_round_f32(v))
    }
    #[inline]
    fn quad(a: u8, b: u8, c: u8, d: u8) -> u8 {
        ((u16::from(a) + u16::from(b) + u16::from(c) + u16::from(d) + 2) >> 2) as u8
    }
}

macro_rules! float_buffered {
    ($t:ty, $w:ty, $quad:expr, $cubic_lanes:expr, $lanczos_lanes:expr) => {
        impl Resizable for $t {
            type W = $w;
            type A = f32;
            type AreaW = $w;
            const ROUNDED_QUAD: bool = $quad;

            #[inline]
            fn coef(c: f32) -> f32 {
                c
            }
            #[inline]
            fn mul(s: $t, a: f32) -> $w {
                <$w>::from(s) * <$w>::from(a)
            }
            #[inline]
            fn one(s: $t) -> $w {
                <$w>::from(s)
            }
            #[inline]
            fn add(a: $w, b: $w) -> $w {
                a + b
            }
            #[inline]
            fn wmul(w: $w, b: f32) -> $w {
                w * <$w>::from(b)
            }
            #[inline]
            fn cast(w: $w) -> $t {
                <$t as Element>::from_f64(f64::from(w))
            }
            fn simd_lanes(ksize: usize) -> usize {
                if ksize == 4 {
                    $cubic_lanes
                } else {
                    $lanczos_lanes
                }
            }
            #[inline]
            fn vterm(w: $w, b: f32) -> f32 {
                w * b
            }
            #[inline]
            fn vfinish(v: f32) -> $t {
                <$t as Element>::from_f32(v)
            }
        }
    };
}

float_buffered!(u16, f32, true, 8, 0);
float_buffered!(f32, f32, false, 4, 4);

impl Resizable for f64 {
    type W = f64;
    type A = f32;
    type AreaW = f64;
    const ROUNDED_QUAD: bool = false;

    #[inline]
    fn coef(c: f32) -> f32 {
        c
    }
    #[inline]
    fn mul(s: f64, a: f32) -> f64 {
        s * f64::from(a)
    }
    #[inline]
    fn one(s: f64) -> f64 {
        s
    }
    #[inline]
    fn add(a: f64, b: f64) -> f64 {
        a + b
    }
    #[inline]
    fn wmul(w: f64, b: f32) -> f64 {
        w * f64::from(b)
    }
    #[inline]
    fn cast(w: f64) -> f64 {
        w
    }
}

/// `cv::resize(src, dsize=(width, height), interpolation)`.
pub fn resize<'a, T: Resizable>(
    src: impl Into<MatView<'a, T>>,
    width: usize,
    height: usize,
    interpolation: Interpolation,
) -> Result<Mat<T>> {
    let src = src.into();
    if src.is_empty() || width == 0 || height == 0 {
        return Err(invalid("resize", "empty source or destination size"));
    }
    let inv_x = width as f64 / src.width() as f64;
    let inv_y = height as f64 / src.height() as f64;
    Ok(resize_impl(&src, width, height, inv_x, inv_y, interpolation))
}

/// `cv::resize(src, None, fx=fx, fy=fy, interpolation)`.
pub fn resize_by<'a, T: Resizable>(
    src: impl Into<MatView<'a, T>>,
    fx: f64,
    fy: f64,
    interpolation: Interpolation,
) -> Result<Mat<T>> {
    let src = src.into();
    if src.is_empty() || fx <= 0.0 || fy <= 0.0 {
        return Err(invalid("resize_by", "empty source or non-positive scale"));
    }
    let width = cv_round(src.width() as f64 * fx);
    let height = cv_round(src.height() as f64 * fy);
    if width <= 0 || height <= 0 {
        return Err(invalid("resize_by", "scale produces an empty image"));
    }
    Ok(resize_impl(
        &src,
        width as usize,
        height as usize,
        fx,
        fy,
        interpolation,
    ))
}

fn resize_impl<T: Resizable>(
    src: &MatView<'_, T>,
    width: usize,
    height: usize,
    inv_x: f64,
    inv_y: f64,
    mut interpolation: Interpolation,
) -> Mat<T> {
    let mut dst = Mat::zeros(width, height, src.channels());
    if src.size() == (width, height) {
        for y in 0..height {
            dst.row_mut(y).copy_from_slice(src.row(y));
        }
        return dst;
    }
    let scale_x = 1.0 / inv_x;
    let scale_y = 1.0 / inv_y;
    let iscale_x = cv_round(scale_x);
    let iscale_y = cv_round(scale_y);
    let area_fast = (scale_x - f64::from(iscale_x)).abs() < f64::EPSILON
        && (scale_y - f64::from(iscale_y)).abs() < f64::EPSILON;

    if interpolation == Interpolation::Nearest {
        nearest(src, &mut dst, inv_x, inv_y);
        return dst;
    }
    if interpolation == Interpolation::Linear && area_fast && iscale_x == 2 && iscale_y == 2 {
        interpolation = Interpolation::Area;
    }
    if interpolation == Interpolation::Area && scale_x >= 1.0 && scale_y >= 1.0 {
        if area_fast {
            area_fast_resize(src, &mut dst, iscale_x as usize, iscale_y as usize);
        } else {
            area_resize(src, &mut dst, scale_x, scale_y);
        }
        return dst;
    }
    let plan = Plan::new(src, width, height, scale_x, scale_y, inv_x, inv_y, interpolation);
    separable(src, &mut dst, &plan);
    dst
}

// Nearest neighbour

fn nearest<T: Element>(src: &MatView<'_, T>, dst: &mut Mat<T>, fx: f64, fy: f64) {
    let (sw, sh, cn) = (src.width(), src.height(), src.channels());
    let ifx = 1.0 / fx;
    let ify = 1.0 / fy;
    let xofs: Vec<usize> = (0..dst.width())
        .map(|x| (cv_floor(x as f64 * ifx).max(0) as usize).min(sw - 1) * cn)
        .collect();
    dst.par_rows_mut().enumerate().for_each(|(y, row)| {
        let sy = (cv_floor(y as f64 * ify).max(0) as usize).min(sh - 1);
        let s = src.row(sy);
        for (d, &ofs) in row.chunks_exact_mut(cn).zip(&xofs) {
            d.copy_from_slice(&s[ofs..ofs + cn]);
        }
    });
}

// Separable linear / cubic / Lanczos

struct Plan {
    ksize: usize,
    xofs: Vec<usize>,
    alpha: Vec<f32>,
    yofs: Vec<i32>,
    beta: Vec<f32>,
    xmin: usize,
    xmax: usize,
}

impl Plan {
    #[allow(clippy::too_many_arguments)]
    fn new<T: Element>(
        src: &MatView<'_, T>,
        dw: usize,
        dh: usize,
        scale_x: f64,
        scale_y: f64,
        inv_x: f64,
        inv_y: f64,
        interpolation: Interpolation,
    ) -> Self {
        let (sw, cn) = (src.width() as i32, src.channels());
        let area_mode = interpolation == Interpolation::Area;
        let ksize = match interpolation {
            Interpolation::Cubic => 4,
            Interpolation::Lanczos4 => 8,
            _ => 2,
        };
        let windowed = matches!(interpolation, Interpolation::Cubic | Interpolation::Lanczos4);
        let ksize2 = ksize as i32 / 2;
        let coeffs = |f: f32, out: &mut [f32]| match interpolation {
            Interpolation::Cubic => out.copy_from_slice(&cubic_coeffs(f)),
            Interpolation::Lanczos4 => out.copy_from_slice(&lanczos4_coeffs(f)),
            _ => {
                out[0] = 1.0 - f;
                out[1] = f;
            }
        };
        let source_pos = |d: usize, scale: f64, inv: f64| -> (i32, f32) {
            if area_mode {
                let s = cv_floor(d as f64 * scale);
                let f = ((d + 1) as f64 - f64::from(s + 1) * inv) as f32;
                (s, if f <= 0.0 { 0.0 } else { f - f.floor() })
            } else {
                let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
                let s = f.floor() as i32;
                (s, f - s as f32)
            }
        };

        let (mut xmin, mut xmax) = (0, dw);
        let mut xofs = vec![0; dw * cn];
        let mut alpha = vec![0.0; dw * ksize];
        for dx in 0..dw {
            let (mut sx, mut fx) = source_pos(dx, scale_x, inv_x);
            if sx < ksize2 - 1 {
                xmin = dx + 1;
                if sx < 0 && !windowed {
                    fx = 0.0;
                    sx = 0;
                }
            }
            if sx + ksize2 >= sw {
                xmax = xmax.min(dx);
                if sx >= sw - 1 && !windowed {
                    fx = 0.0;
                    sx = sw - 1;
                }
            }
            for k in 0..cn {
                xofs[dx * cn + k] = (sx as isize * cn as isize + k as isize) as usize;
            }
            coeffs(fx, &mut alpha[dx * ksize..(dx + 1) * ksize]);
        }
        let mut yofs = vec![0; dh];
        let mut beta = vec![0.0; dh * ksize];
        for dy in 0..dh {
            let (sy, fy) = source_pos(dy, scale_y, inv_y);
            yofs[dy] = sy;
            coeffs(fy, &mut beta[dy * ksize..(dy + 1) * ksize]);
        }
        Self {
            ksize,
            xofs,
            alpha,
            yofs,
            beta,
            xmin: xmin * cn,
            xmax: xmax * cn,
        }
    }
}

fn cubic_coeffs(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

fn lanczos4_coeffs(x: f32) -> [f32; 8] {
    const S45: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const CS: [[f64; 2]; 8] = [
        [1.0, 0.0],
        [-S45, -S45],
        [0.0, 1.0],
        [S45, -S45],
        [-1.0, 0.0],
        [S45, S45],
        [0.0, -1.0],
        [-S45, S45],
    ];
    let y0 = f64::from(-(x + 3.0)) * std::f64::consts::PI * 0.25;
    let (s0, c0) = y0.sin_cos();
    let mut out = [0.0f32; 8];
    let mut sum = 0.0f32;
    for (i, c) in out.iter_mut().enumerate() {
        let y0_ = x + 3.0 - i as f32;
        *c = if y0_.abs() >= 1e-6 {
            let y = f64::from(-y0_) * std::f64::consts::PI * 0.25;
            ((CS[i][0] * s0 + CS[i][1] * c0) / (y * y)) as f32
        } else {
            1e30
        };
        sum += *c;
    }
    let inv = 1.0 / sum;
    for c in &mut out {
        *c *= inv;
    }
    out
}

struct HJob<'a, T: Resizable>(&'a [T], &'a mut [T::W], &'a Plan, &'a [T::A], usize);

impl<T: Resizable> crate::simd::Kernel for HJob<'_, T> {
    type Output = ();

    #[inline(always)]
    fn run(self) {
        hresize(self.0, self.1, self.2, self.3, self.4);
    }
}

struct VJob<'a, T: Resizable>(&'a [&'a [T::W]], &'a [T::A], &'a mut [T]);

impl<T: Resizable> crate::simd::Kernel for VJob<'_, T> {
    type Output = ();

    #[inline(always)]
    fn run(self) {
        vresize(self.0, self.1, self.2);
    }
}

/// One output row of the 2x2 integer area average, `CN` channels per pixel.
#[inline(always)]
fn quad_row<T: Resizable, const CN: usize>((r0, r1, d): (&[T], &[T], &mut [T])) {
    for ((out, a0), a1) in d
        .chunks_exact_mut(CN)
        .zip(r0.chunks_exact(2 * CN))
        .zip(r1.chunks_exact(2 * CN))
    {
        for c in 0..CN {
            out[c] = T::quad(a0[c], a0[c + CN], a1[c], a1[c + CN]);
        }
    }
}

/// Bilinear horizontal pass over whole pixels: one offset and weight pair per `CN` channels.
#[inline(always)]
fn linear_pixels<T: Resizable, const CN: usize>(s: &[T], d: &mut [T::W], xofs: &[usize], alpha: &[T::A]) {
    for ((out, ofs), a) in d
        .chunks_exact_mut(CN)
        .zip(xofs.chunks_exact(CN))
        .zip(alpha.chunks_exact(2 * CN))
    {
        let sx = ofs[0];
        let (p, q) = (&s[sx..sx + CN], &s[sx + CN..sx + 2 * CN]);
        let (a0, a1) = (a[0], a[1]);
        for c in 0..CN {
            out[c] = T::add(T::mul(p[c], a0), T::mul(q[c], a1));
        }
    }
}

/// Horizontal pass; `alpha` holds `ksize` coefficients per destination element.
#[inline(always)]
fn hresize<T: Resizable>(s: &[T], d: &mut [T::W], plan: &Plan, alpha: &[T::A], cn: usize) {
    let ksize = plan.ksize;
    let swidth = s.len() as isize;
    if ksize == 2 {
        let xmax = plan.xmax.min(d.len());
        match cn {
            1 => linear_pixels::<T, 1>(s, &mut d[..xmax], &plan.xofs, alpha),
            3 => linear_pixels::<T, 3>(s, &mut d[..xmax], &plan.xofs, alpha),
            4 => linear_pixels::<T, 4>(s, &mut d[..xmax], &plan.xofs, alpha),
            _ => {
                for ((out, &sx), a) in d[..xmax].iter_mut().zip(&plan.xofs).zip(alpha.chunks_exact(2)) {
                    *out = T::add(T::mul(s[sx], a[0]), T::mul(s[sx + cn], a[1]));
                }
            }
        }
        for (out, &sx) in d[xmax..].iter_mut().zip(&plan.xofs[xmax..]) {
            *out = T::one(s[sx]);
        }
        return;
    }
    let back = (ksize / 2 - 1) * cn;
    for (dx, out) in d.iter_mut().enumerate() {
        let a = &alpha[dx * ksize..(dx + 1) * ksize];
        let start = plan.xofs[dx] as isize - back as isize;
        if dx < plan.xmin || dx >= plan.xmax {
            let mut v = T::W::default();
            for (j, &aj) in a.iter().enumerate() {
                let mut sxj = start + (j * cn) as isize;
                while sxj < 0 {
                    sxj += cn as isize;
                }
                while sxj >= swidth {
                    sxj -= cn as isize;
                }
                v = T::add(v, T::mul(s[sxj as usize], aj));
            }
            *out = v;
        } else {
            let start = start as usize;
            let mut v = T::mul(s[start], a[0]);
            for (j, &aj) in a.iter().enumerate().skip(1) {
                v = T::add(v, T::mul(s[start + j * cn], aj));
            }
            *out = v;
        }
    }
}

#[inline(always)]
fn vresize<T: Resizable>(rows: &[&[T::W]], beta: &[T::A], dst: &mut [T]) {
    if rows.len() == 2 {
        for (x, d) in dst.iter_mut().enumerate() {
            *d = T::vlinear(rows[0][x], rows[1][x], beta[0], beta[1]);
        }
        return;
    }
    // Row-wise accumulation keeps each element's summation order and vectorizes.
    let lanes = T::simd_lanes(rows.len());
    let n = rows.len();
    let simd_end = dst.len().checked_div(lanes).map_or(0, |q| q * lanes);
    let (head, tail) = dst.split_at_mut(simd_end);
    if !head.is_empty() {
        let mut acc: Vec<f32> = rows[n - 1][..simd_end]
            .iter()
            .map(|&w| T::vterm(w, beta[n - 1]))
            .collect();
        for k in (0..n - 1).rev() {
            for (a, &w) in acc.iter_mut().zip(&rows[k][..simd_end]) {
                *a += T::vterm(w, beta[k]);
            }
        }
        for (d, a) in head.iter_mut().zip(acc) {
            *d = T::vfinish(a);
        }
    }
    let len = simd_end + tail.len();
    let mut acc: Vec<T::W> = rows[0][simd_end..len]
        .iter()
        .map(|&w| T::wmul(w, beta[0]))
        .collect();
    for (row, &b) in rows.iter().zip(beta).skip(1) {
        for (a, &w) in acc.iter_mut().zip(&row[simd_end..len]) {
            *a = T::add(*a, T::wmul(w, b));
        }
    }
    for (d, a) in tail.iter_mut().zip(acc) {
        *d = T::cast(a);
    }
}

fn separable<T: Resizable>(src: &MatView<'_, T>, dst: &mut Mat<T>, plan: &Plan) {
    let ksize = plan.ksize;
    let cn = src.channels();
    let sh = src.height() as i32;
    let row_len = dst.row_len();
    let alpha: Vec<T::A> = (0..row_len)
        .flat_map(|e| {
            plan.alpha[(e / cn) * ksize..(e / cn + 1) * ksize]
                .iter()
                .map(|&a| T::coef(a))
        })
        .collect();
    let beta: Vec<T::A> = plan.beta.iter().map(|&b| T::coef(b)).collect();
    let band = band_rows(dst.height(), row_len);
    dst.data_mut()
        .par_chunks_mut(row_len * band)
        .enumerate()
        .for_each(|(chunk, out)| {
            let mut cache: Vec<(i32, Vec<T::W>)> = Vec::with_capacity(ksize + 1);
            let mut pool: Vec<Vec<T::W>> = Vec::new();
            let mut needed = vec![0i32; ksize];
            for (i, drow) in out.chunks_exact_mut(row_len).enumerate() {
                let dy = chunk * band + i;
                let sy0 = plan.yofs[dy];
                for (k, n) in needed.iter_mut().enumerate() {
                    *n = (sy0 - ksize as i32 / 2 + 1 + k as i32).clamp(0, sh - 1);
                }
                let mut i = 0;
                while i < cache.len() {
                    if needed.contains(&cache[i].0) {
                        i += 1;
                    } else {
                        pool.push(cache.swap_remove(i).1);
                    }
                }
                for &sy in &needed {
                    if !cache.iter().any(|(c, _)| *c == sy) {
                        let mut buf = pool.pop().unwrap_or_else(|| vec![T::W::default(); row_len]);
                        crate::simd::dispatch(HJob(src.row(sy as usize), &mut buf, plan, &alpha, cn));
                        cache.push((sy, buf));
                    }
                }
                let mut rows: [&[T::W]; 8] = [&[]; 8];
                for (slot, sy) in rows.iter_mut().zip(&needed) {
                    *slot = cache
                        .iter()
                        .find(|(c, _)| c == sy)
                        .map_or(&[][..], |(_, b)| b.as_slice());
                }
                let rows = &rows[..ksize];
                crate::simd::dispatch(VJob(rows, &beta[dy * ksize..(dy + 1) * ksize], drow));
            }
        });
}

fn band_rows(height: usize, row_len: usize) -> usize {
    if height * row_len < MIN_PARALLEL_ELEMENTS {
        return height.max(1);
    }
    let tasks = rayon::current_num_threads() * 4;
    height.div_ceil(tasks).clamp(1, 64)
}

// Area: integer factors

fn area_fast_resize<T: Resizable>(src: &MatView<'_, T>, dst: &mut Mat<T>, sx: usize, sy: usize) {
    let (sw, sh, cn) = (src.width(), src.height(), src.channels());
    let row_len = dst.row_len();
    let area = sx * sy;
    let scale = 1.0f32 / area as f32;
    let full = (sw / sx) * cn;
    let quad = sx == 2 && sy == 2 && matches!(cn, 1 | 3 | 4);
    let simd_quad_f32 = sx == 2 && sy == 2 && T::DEPTH == crate::Depth::F32 && matches!(cn, 1 | 4);
    dst.par_rows_mut().enumerate().for_each(|(dy, d)| {
        let y0 = dy * sy;
        if y0 >= sh {
            d.fill(T::ZERO);
            return;
        }
        let w = if y0 + sy <= sh { full } else { 0 };
        let mut dx = 0;
        if quad && T::ROUNDED_QUAD {
            let (r0, r1) = (src.row(y0), src.row(y0 + 1));
            match cn {
                1 => crate::simd::vectorized(quad_row::<T, 1>, (r0, r1, &mut d[..w])),
                3 => crate::simd::vectorized(quad_row::<T, 3>, (r0, r1, &mut d[..w])),
                _ => crate::simd::vectorized(quad_row::<T, 4>, (r0, r1, &mut d[..w])),
            }
            dx = w;
        } else if simd_quad_f32 {
            let (r0, r1) = (src.row(y0), src.row(y0 + 1));
            let lanes = if cn == 1 { (w / 4) * 4 } else { w };
            for (x, out) in d[..lanes].iter_mut().enumerate() {
                let i = (x / cn) * 2 * cn + x % cn;
                let s = (r0[i].to_f32() + r0[i + cn].to_f32()) + (r1[i].to_f32() + r1[i + cn].to_f32());
                *out = T::from_f32(s * 0.25);
            }
            dx = lanes;
        }
        for (x, out) in d.iter_mut().enumerate().take(w).skip(dx) {
            let base = (x / cn) * sx * cn + x % cn;
            let mut sum = T::AreaW::default();
            let taps: Vec<T> = (0..sy)
                .flat_map(|ky| (0..sx).map(move |kx| src.row(y0 + ky)[base + kx * cn]))
                .collect();
            let mut chunks = taps.chunks_exact(4);
            for c in &mut chunks {
                let a = T::AreaW::of(c[0]) + T::AreaW::of(c[1]);
                sum = sum + (a + T::AreaW::of(c[2]) + T::AreaW::of(c[3]));
            }
            for &t in chunks.remainder() {
                sum = sum + T::AreaW::of(t);
            }
            *out = (sum * T::AreaW::of_f32(scale)).store();
        }
        for (x, out) in d.iter_mut().enumerate().take(row_len).skip(w) {
            let px = x / cn;
            let base = px * sx * cn + x % cn;
            let mut sum = T::AreaW::default();
            let mut count = 0;
            for ky in 0..sy {
                if y0 + ky >= sh {
                    break;
                }
                let row = src.row(y0 + ky);
                for kx in 0..sx {
                    if px * sx + kx >= sw {
                        break;
                    }
                    sum = sum + T::AreaW::of(row[base + kx * cn]);
                    count += 1;
                }
            }
            *out = if count == 0 {
                T::ZERO
            } else {
                sum.div_count(count).store()
            };
        }
    });
}

// Area: fractional factors

#[derive(Clone, Copy)]
struct Decimate {
    si: usize,
    di: usize,
    alpha: f32,
}

fn area_table(ssize: usize, dsize: usize, cn: usize, scale: f64) -> Vec<Decimate> {
    let mut tab = Vec::with_capacity(ssize * 2);
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx2 = cv_floor(fsx2);
        let mut sx1 = crate::element::cv_ceil(fsx1);
        sx2 = sx2.min(ssize as i32 - 1);
        sx1 = sx1.min(sx2);
        if f64::from(sx1) - fsx1 > 1e-3 {
            tab.push(Decimate {
                di: dx * cn,
                si: (sx1 - 1) as usize * cn,
                alpha: ((f64::from(sx1) - fsx1) / cell) as f32,
            });
        }
        for sx in sx1..sx2 {
            tab.push(Decimate {
                di: dx * cn,
                si: sx as usize * cn,
                alpha: (1.0 / cell) as f32,
            });
        }
        if fsx2 - f64::from(sx2) > 1e-3 {
            let a = (fsx2 - f64::from(sx2)).min(1.0).min(cell) / cell;
            tab.push(Decimate {
                di: dx * cn,
                si: sx2 as usize * cn,
                alpha: a as f32,
            });
        }
    }
    tab
}

fn area_resize<T: Resizable>(src: &MatView<'_, T>, dst: &mut Mat<T>, scale_x: f64, scale_y: f64) {
    let cn = src.channels();
    let xtab = area_table(src.width(), dst.width(), cn, scale_x);
    let ytab = area_table(src.height(), dst.height(), 1, scale_y);
    let mut starts = vec![0usize; dst.height() + 1];
    let mut dy = 0;
    for (k, entry) in ytab.iter().enumerate() {
        if k == 0 || entry.di != ytab[k - 1].di {
            starts[dy] = k;
            dy += 1;
        }
    }
    starts[dy] = ytab.len();
    let row_len = dst.row_len();
    dst.par_rows_mut().enumerate().for_each(|(dy, d)| {
        let mut buf = vec![T::AreaW::default(); row_len];
        let mut sum = vec![T::AreaW::default(); row_len];
        for (j, entry) in ytab[starts[dy]..starts[dy + 1]].iter().enumerate() {
            let s = src.row(entry.si);
            buf.fill(T::AreaW::default());
            for x in &xtab {
                let a = T::AreaW::of_f32(x.alpha);
                for c in 0..cn {
                    buf[x.di + c] = buf[x.di + c] + T::AreaW::of(s[x.si + c]) * a;
                }
            }
            let beta = T::AreaW::of_f32(entry.alpha);
            if j == 0 {
                sum.iter_mut().zip(&buf).for_each(|(s, &b)| *s = beta * b);
            } else {
                sum.iter_mut().zip(&buf).for_each(|(s, &b)| *s = *s + beta * b);
            }
        }
        for (o, &s) in d.iter_mut().zip(&sum) {
            *o = s.store();
        }
    });
}

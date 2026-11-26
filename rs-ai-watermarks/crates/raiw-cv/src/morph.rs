//! Structuring elements, `dilate` / `erode` and `morphologyEx` with OpenCV's default border
//! (pixels outside the image never win).

use rayon::prelude::*;

use crate::element::Element;
use crate::error::{Result, invalid};
use crate::mat::{Mat, MatView};

/// `cv::MorphShapes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MorphShape {
    Rect,
    Cross,
    Ellipse,
}

/// `cv::MorphTypes` handled by [`morphology_ex`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MorphOp {
    Erode,
    Dilate,
    Open,
    Close,
    Gradient,
    TopHat,
    BlackHat,
}

/// A binary structuring element anchored at its center.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuringElement {
    width: usize,
    height: usize,
    mask: Vec<bool>,
}

impl StructuringElement {
    /// `cv::getStructuringElement(shape, (width, height))`.
    pub fn new(shape: MorphShape, width: usize, height: usize) -> Self {
        let shape = if width == 1 && height == 1 {
            MorphShape::Rect
        } else {
            shape
        };
        let (ax, ay) = (width / 2, height / 2);
        let (r, c) = (height as i64 / 2, width as i64 / 2);
        let inv_r2 = if r > 0 { 1.0 / (r * r) as f64 } else { 0.0 };
        let mut mask = vec![false; width * height];
        for i in 0..height {
            let (j1, j2) = match shape {
                MorphShape::Rect => (0, width),
                MorphShape::Cross if i == ay => (0, width),
                MorphShape::Cross => (ax, ax + 1),
                MorphShape::Ellipse => {
                    let dy = i as i64 - r;
                    if dy.abs() <= r {
                        let dx =
                            crate::element::cv_round(c as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt());
                        let dx = i64::from(dx);
                        (
                            (c - dx).max(0) as usize,
                            ((c + dx + 1).min(width as i64)) as usize,
                        )
                    } else {
                        (0, 0)
                    }
                }
            };
            for m in &mut mask[i * width + j1..i * width + j2.max(j1)] {
                *m = true;
            }
        }
        Self { width, height, mask }
    }

    /// `np.ones((height, width), np.uint8)`.
    pub fn rect(width: usize, height: usize) -> Self {
        Self::new(MorphShape::Rect, width, height)
    }

    /// `cv::getStructuringElement(MORPH_ELLIPSE, (size, size))`.
    pub fn ellipse(size: usize) -> Self {
        Self::new(MorphShape::Ellipse, size, size)
    }

    /// An arbitrary mask given row-major (nonzero = member).
    pub fn from_mask(width: usize, height: usize, mask: &[u8]) -> Result<Self> {
        if mask.len() != width * height || width == 0 || height == 0 {
            return Err(invalid("StructuringElement", "mask size mismatch"));
        }
        Ok(Self {
            width,
            height,
            mask: mask.iter().map(|&v| v != 0).collect(),
        })
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn contains(&self, x: usize, y: usize) -> bool {
        self.mask[y * self.width + x]
    }

    /// The element as a 0/1 `u8` image.
    pub fn to_mat(&self) -> Mat<u8> {
        Mat::from_fn(self.width, self.height, 1, |x, y, _| {
            u8::from(self.contains(x, y))
        })
    }

    /// Horizontal runs `(row, start, len)` of member pixels.
    fn runs(&self) -> Vec<(usize, usize, usize)> {
        let mut runs = Vec::new();
        for y in 0..self.height {
            let mut x = 0;
            while x < self.width {
                if self.contains(x, y) {
                    let start = x;
                    while x < self.width && self.contains(x, y) {
                        x += 1;
                    }
                    runs.push((y, start, x - start));
                } else {
                    x += 1;
                }
            }
        }
        runs
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Extremum {
    Max,
    Min,
}

impl Extremum {
    #[inline(always)]
    fn pick<T: PartialOrd + Copy>(self, a: T, b: T) -> T {
        match self {
            Self::Max => {
                if b > a {
                    b
                } else {
                    a
                }
            }
            Self::Min => {
                if b < a {
                    b
                } else {
                    a
                }
            }
        }
    }

    fn neutral<T: Element>(self) -> T {
        match self {
            Self::Max => T::LOWEST,
            Self::Min => T::HIGHEST,
        }
    }
}

/// Windowed extrema of one padded row for every power-of-two window up to `max_len`.
#[inline(always)]
fn doubling_table<T: Element>(row: &[T], cn: usize, max_len: usize, op: Extremum) -> Vec<Vec<T>> {
    let mut levels = vec![row.to_vec()];
    let mut span = 1;
    while span * 2 <= max_len {
        let prev = levels.last().map_or(&[][..], Vec::as_slice);
        let shift = (span * cn).min(prev.len());
        let mut next = prev.to_vec();
        let n = prev.len() - shift;
        crate::simd::vectorized(pick_pair::<T>, (&mut next[..n], &prev[..n], &prev[shift..], op));
        levels.push(next);
        span *= 2;
    }
    levels
}

fn morph_once<T: Element>(src: &MatView<'_, T>, se: &StructuringElement, op: Extremum) -> Mat<T> {
    let (w, h, cn) = (src.width(), src.height(), src.channels());
    let (ax, ay) = (se.width / 2, se.height / 2);
    let runs = se.runs();
    let mut out = Mat::filled(w, h, cn, op.neutral::<T>());
    if runs.is_empty() {
        return out;
    }
    let max_len = runs.iter().map(|r| r.2).max().unwrap_or(1);
    let mut segments: Vec<(usize, usize)> = runs.iter().map(|&(_, s, l)| (s, l)).collect();
    segments.sort_unstable();
    segments.dedup();
    let run_segment: Vec<(usize, usize)> = runs
        .iter()
        .map(|&(ky, s, l)| (ky, segments.binary_search(&(s, l)).unwrap_or(0)))
        .collect();
    let pad_left = ax;
    let padded_len = (w + se.width) * cn;
    let neutral = op.neutral::<T>();
    let row_len = w * cn;
    let band = h
        .div_ceil(rayon::current_num_threads().max(1) * 2)
        .max(se.height.max(8))
        .min(h.max(1));
    out.data_mut()
        .par_chunks_mut(row_len * band)
        .enumerate()
        .for_each(|(b, dst)| {
            let y0 = b * band;
            let rows = dst.len() / row_len;
            let first = y0 as isize - ay as isize;
            let nsrc = rows + se.height - 1;
            // Horizontal extrema of every distinct kernel segment for each source row of the band.
            let mut hbuf = vec![neutral; segments.len() * nsrc * row_len];
            let mut padded = vec![neutral; padded_len];
            let mut present = vec![false; nsrc];
            for (i, flag) in present.iter_mut().enumerate() {
                let sy = first + i as isize;
                if sy < 0 || sy as usize >= h {
                    continue;
                }
                *flag = true;
                padded[pad_left * cn..pad_left * cn + row_len].copy_from_slice(src.row(sy as usize));
                let levels = doubling_table(&padded, cn, max_len, op);
                for (si, &(start, len)) in segments.iter().enumerate() {
                    let p = (usize::BITS - 1 - len.leading_zeros()) as usize;
                    let tail = (len - (1 << p)) * cn;
                    let base = start * cn;
                    let a = &levels[p][base..base + row_len];
                    let c = &levels[p][base + tail..base + tail + row_len];
                    let target = &mut hbuf[(si * nsrc + i) * row_len..(si * nsrc + i + 1) * row_len];
                    crate::simd::vectorized(pick_pair::<T>, (target, a, c, op));
                }
            }
            for (i, drow) in dst.chunks_exact_mut(row_len).enumerate() {
                for &(ky, si) in &run_segment {
                    if !present[i + ky] {
                        continue;
                    }
                    let line = &hbuf[(si * nsrc + i + ky) * row_len..(si * nsrc + i + ky + 1) * row_len];
                    crate::simd::vectorized(pick_into::<T>, (drow, line, op));
                }
            }
        });
    out
}

/// `target[i] = pick(a[i], c[i])`.
#[inline(always)]
fn pick_pair<T: Element>((target, a, c, op): (&mut [T], &[T], &[T], Extremum)) {
    for ((t, &u), &v) in target.iter_mut().zip(a).zip(c) {
        *t = op.pick(u, v);
    }
}

/// `dst[i] = pick(dst[i], line[i])`.
#[inline(always)]
fn pick_into<T: Element>((dst, line, op): (&mut [T], &[T], Extremum)) {
    for (d, &v) in dst.iter_mut().zip(line) {
        *d = op.pick(*d, v);
    }
}

fn morph<'a, T: Element>(
    src: impl Into<MatView<'a, T>>,
    se: &StructuringElement,
    iterations: usize,
    op: Extremum,
) -> Result<Mat<T>> {
    let src = src.into();
    if src.is_empty() {
        return Err(invalid("morphology", "empty input"));
    }
    let mut out = morph_once(&src, se, op);
    for _ in 1..iterations {
        out = morph_once(&out.view(), se, op);
    }
    Ok(out)
}

/// `cv::dilate(src, kernel, iterations=iterations)`.
pub fn dilate<'a, T: Element>(
    src: impl Into<MatView<'a, T>>,
    se: &StructuringElement,
    iterations: usize,
) -> Result<Mat<T>> {
    morph(src, se, iterations.max(1), Extremum::Max)
}

/// `cv::erode(src, kernel, iterations=iterations)`.
pub fn erode<'a, T: Element>(
    src: impl Into<MatView<'a, T>>,
    se: &StructuringElement,
    iterations: usize,
) -> Result<Mat<T>> {
    morph(src, se, iterations.max(1), Extremum::Min)
}

fn subtract<T: Element>(a: &MatView<'_, T>, b: &Mat<T>) -> Mat<T> {
    let mut out = b.clone();
    out.par_rows_mut().enumerate().for_each(|(y, row)| {
        for (o, &s) in row.iter_mut().zip(a.row(y)) {
            *o = s.sub_sat(*o);
        }
    });
    out
}

/// `cv::morphologyEx(src, op, kernel, iterations=iterations)`.
pub fn morphology_ex<'a, T: Element>(
    src: impl Into<MatView<'a, T>>,
    op: MorphOp,
    se: &StructuringElement,
    iterations: usize,
) -> Result<Mat<T>> {
    let src = src.into();
    let it = iterations.max(1);
    let open = |s: &MatView<'_, T>| -> Result<Mat<T>> { dilate(&erode(s, se, it)?, se, it) };
    let close = |s: &MatView<'_, T>| -> Result<Mat<T>> { erode(&dilate(s, se, it)?, se, it) };
    match op {
        MorphOp::Erode => erode(src, se, it),
        MorphOp::Dilate => dilate(src, se, it),
        MorphOp::Open => open(&src),
        MorphOp::Close => close(&src),
        MorphOp::Gradient => {
            let d = dilate(src, se, it)?;
            Ok(subtract(&d.view(), &erode(src, se, it)?))
        }
        MorphOp::TopHat => Ok(subtract(&src, &open(&src)?)),
        MorphOp::BlackHat => {
            let c = close(&src)?;
            Ok(subtract(&c.view(), &src.to_mat()))
        }
    }
}

//! `cv::inpaint` (Telea fast marching and Navier-Stokes), a line-by-line port of `photo/src/inpaint.cpp`.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::element::{Depth, Element};
use crate::error::{CvError, Result, invalid};
use crate::mat::{Mat, MatView};
use crate::morph::{MorphShape, StructuringElement, dilate};

/// `cv::INPAINT_TELEA` / `cv::INPAINT_NS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InpaintMethod {
    Telea,
    NavierStokes,
}

const KNOWN: u8 = 0;
const BAND: u8 = 1;
const INSIDE: u8 = 2;
const CHANGE: u8 = 3;

#[derive(Clone, Copy)]
struct HeapElem {
    t: f32,
    order: u64,
    i: usize,
    j: usize,
}

impl PartialEq for HeapElem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for HeapElem {}

impl PartialOrd for HeapElem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapElem {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.t.partial_cmp(&other.t) {
            Some(Ordering::Equal) | None => self.order.cmp(&other.order),
            Some(o) => o,
        }
    }
}

/// `CvPriorityQueueFloat`: smallest distance first, insertion order breaks ties.
#[derive(Default)]
struct Queue {
    heap: BinaryHeap<Reverse<HeapElem>>,
    next: u64,
}

impl Queue {
    fn push(&mut self, i: usize, j: usize, t: f32) {
        self.heap.push(Reverse(HeapElem { t, order: self.next, i, j }));
        self.next += 1;
    }

    fn pop(&mut self) -> Option<(usize, usize)> {
        self.heap.pop().map(|Reverse(e)| (e.i, e.j))
    }

    fn add(&mut self, band: &[u8], cols: usize) {
        for (idx, &v) in band.iter().enumerate() {
            if v != 0 {
                self.push(idx / cols, idx % cols, 0.0);
            }
        }
    }
}

/// The padded `(rows + 2) x (cols + 2)` state grids.
struct Grid {
    rows: usize,
    cols: usize,
}

impl Grid {
    #[inline]
    fn at(&self, i: usize, j: usize) -> usize {
        i * self.cols + j
    }

    fn clear_border(&self, m: &mut [u8]) {
        for j in 0..self.cols {
            m[j] = 0;
            m[self.at(self.rows - 1, j)] = 0;
        }
        for i in 0..self.rows {
            m[self.at(i, 0)] = 0;
            m[self.at(i, self.cols - 1)] = 0;
        }
    }

    /// `dilate(src, element) - src` (saturating), outside pixels ignored.
    fn dilate_minus(&self, src: &[u8], element: &StructuringElement) -> Result<Vec<u8>> {
        let view = MatView::new(src, self.cols, self.rows, 1)?;
        let dilated = dilate(view, element, 1)?;
        Ok(dilated.data().iter().zip(src).map(|(&d, &s)| d.saturating_sub(s)).collect())
    }
}

fn solve(g: &Grid, f: &[u8], t: &[f32], (i1, j1): (usize, usize), (i2, j2): (usize, usize)) -> f32 {
    let a11 = f64::from(t[g.at(i1, j1)]);
    let a22 = f64::from(t[g.at(i2, j2)]);
    let m12 = a11.min(a22);
    let k1 = f[g.at(i1, j1)] != INSIDE;
    let k2 = f[g.at(i2, j2)] != INSIDE;
    let sol = match (k1, k2) {
        (true, true) if (a11 - a22).abs() >= 1.0 => 1.0 + m12,
        (true, true) => (a11 + a22 + (2.0 - (a11 - a22) * (a11 - a22)).sqrt()) * 0.5,
        (true, false) => 1.0 + a11,
        (false, true) => 1.0 + a22,
        (false, false) => 1.0 + m12,
    };
    sol as f32
}

#[inline]
fn min2(a: f32, b: f32) -> f32 {
    if a > b { b } else { a }
}

fn arrival(g: &Grid, f: &[u8], t: &[f32], i: usize, j: usize) -> f32 {
    let a = solve(g, f, t, (i - 1, j), (i, j - 1));
    let b = solve(g, f, t, (i + 1, j), (i, j - 1));
    let c = solve(g, f, t, (i - 1, j), (i, j + 1));
    let d = solve(g, f, t, (i + 1, j), (i, j + 1));
    min2(min2(a, b), min2(c, d))
}

fn neighbours(ii: usize, jj: usize) -> [(isize, isize); 4] {
    let (i, j) = (ii as isize, jj as isize);
    [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1)]
}

/// `icvCalcFMM` with `negate = true`: signed distances outside the hole.
fn outer_distances(g: &Grid, f: &mut [u8], t: &mut [f32], queue: &mut Queue) {
    while let Some((ii, jj)) = queue.pop() {
        f[g.at(ii, jj)] = CHANGE;
        for (i, j) in neighbours(ii, jj) {
            if i <= 0 || j <= 0 || i as usize >= g.rows || j as usize >= g.cols {
                continue;
            }
            let (i, j) = (i as usize, j as usize);
            if f[g.at(i, j)] == INSIDE {
                let dist = arrival(g, f, t, i, j);
                t[g.at(i, j)] = dist;
                f[g.at(i, j)] = BAND;
                queue.push(i, j, dist);
            }
        }
    }
    for (fv, tv) in f.iter_mut().zip(t.iter_mut()) {
        if *fv == CHANGE {
            *fv = KNOWN;
            *tv = -*tv;
        }
    }
}

/// Element types `inpaint` supports (OpenCV: 8U C1/C3, 16U C1, 32F C1).
pub trait InpaintElement: Element {
    #[doc(hidden)]
    fn round_telea(sat: f32) -> Self;
}

impl InpaintElement for u8 {
    fn round_telea(sat: f32) -> u8 {
        u8::from_f64(f64::from(sat) + 0.5)
    }
}

impl InpaintElement for u16 {
    fn round_telea(sat: f32) -> u16 {
        u16::from_f64(f64::from(sat) + 0.5)
    }
}

impl InpaintElement for f32 {
    fn round_telea(sat: f32) -> f32 {
        sat
    }
}

/// `cv::inpaint(src, mask, radius, method)`.
pub fn inpaint<'a, 'b, T: InpaintElement>(
    src: impl Into<MatView<'a, T>>,
    mask: impl Into<MatView<'b, u8>>,
    radius: f64,
    method: InpaintMethod,
) -> Result<Mat<T>> {
    let src = src.into();
    let mask = mask.into();
    let cn = src.channels();
    if src.size() != mask.size() || mask.channels() != 1 {
        return Err(CvError::ShapeMismatch { op: "inpaint" });
    }
    if !(cn == 1 || (cn == 3 && T::DEPTH == Depth::U8)) {
        return Err(invalid("inpaint", "8-bit 1/3-channel, 16-bit or float 1-channel images are supported"));
    }
    let range = crate::element::cv_round(radius).clamp(1, 100) as usize;
    let g = Grid { rows: src.height() + 2, cols: src.width() + 2 };
    let mut f = vec![KNOWN; g.rows * g.cols];
    for y in 0..mask.height() {
        for (x, &m) in mask.row(y).iter().enumerate() {
            if m != 0 {
                f[g.at(y + 1, x + 1)] = INSIDE;
            }
        }
    }
    g.clear_border(&mut f);
    let mut band = g.dilate_minus(&f, &StructuringElement::new(MorphShape::Cross, 3, 3))?;
    g.clear_border(&mut band);
    let mut queue = Queue::default();
    queue.add(&band, g.cols);
    let mut t = vec![1.0e6f32; f.len()];
    for (tv, &b) in t.iter_mut().zip(&band) {
        if b != 0 {
            *tv = 0.0;
        }
    }
    let mut out = src.to_mat();
    match method {
        InpaintMethod::Telea => {
            let mut ring = g.dilate_minus(&f, &StructuringElement::rect(2 * range + 1, 2 * range + 1))?;
            let mut outer = Queue::default();
            outer.add(&band, g.cols);
            for (r, &b) in ring.iter_mut().zip(&band) {
                *r = r.saturating_sub(b);
            }
            g.clear_border(&mut ring);
            outer_distances(&g, &mut ring, &mut t, &mut outer);
            march(&g, &mut f, &mut t, &mut out, range, &mut queue, telea_pixel::<T>);
        }
        InpaintMethod::NavierStokes => march(&g, &mut f, &mut t, &mut out, range, &mut queue, ns_pixel::<T>),
    }
    Ok(out)
}

type PixelFn<T> = fn(&Grid, &[u8], &[f32], &mut Mat<T>, usize, usize, usize);

fn march<T: InpaintElement>(
    g: &Grid,
    f: &mut [u8],
    t: &mut [f32],
    out: &mut Mat<T>,
    range: usize,
    queue: &mut Queue,
    pixel: PixelFn<T>,
) {
    while let Some((ii, jj)) = queue.pop() {
        f[g.at(ii, jj)] = KNOWN;
        for (i, j) in neighbours(ii, jj) {
            if i <= 0 || j <= 0 || i as usize > g.rows - 1 || j as usize > g.cols - 1 {
                continue;
            }
            let (i, j) = (i as usize, j as usize);
            if f[g.at(i, j)] == INSIDE {
                let dist = arrival(g, f, t, i, j);
                t[g.at(i, j)] = dist;
                pixel(g, f, t, out, range, i, j);
                f[g.at(i, j)] = BAND;
                queue.push(i, j, dist);
            }
        }
    }
}

/// Clamped image coordinates of `inpaint.cpp`'s `km`, `kp`, `lm`, `lp`.
#[derive(Clone, Copy)]
struct Taps {
    km: usize,
    kp: usize,
    lm: usize,
    lp: usize,
}

impl Taps {
    fn new(g: &Grid, k: usize, l: usize) -> Self {
        Self {
            km: k - 1 + usize::from(k == 1),
            kp: k - 1 - usize::from(k == g.rows - 2),
            lm: l - 1 + usize::from(l == 1),
            lp: l - 1 - usize::from(l == g.cols - 2),
        }
    }
}

fn window(g: &Grid, f: &[u8], i: usize, j: usize, range: usize) -> impl Iterator<Item = (usize, usize)> {
    let r = range as isize;
    let (ci, cj) = (i as isize, j as isize);
    let (rows, cols) = (g.rows as isize, g.cols as isize);
    (ci - r..=ci + r)
        .flat_map(move |k| (cj - r..=cj + r).map(move |l| (k, l)))
        .filter(move |&(k, l)| k > 0 && l > 0 && k < rows - 1 && l < cols - 1)
        .filter(move |&(k, l)| (l - cj) * (l - cj) + (k - ci) * (k - ci) <= r * r)
        .map(|(k, l)| (k as usize, l as usize))
        .filter(move |&(k, l)| f[k * g.cols + l] != INSIDE)
}

#[inline]
fn px<T: Element>(out: &Mat<T>, y: usize, x: usize, c: usize) -> f64 {
    out.at(x, y, c).to_f64()
}

fn telea_pixel<T: InpaintElement>(g: &Grid, f: &[u8], t: &[f32], out: &mut Mat<T>, range: usize, i: usize, j: usize) {
    let cn = out.channels();
    let tt = |a: usize, b: usize| t[g.at(a, b)];
    let known = |a: usize, b: usize| f[g.at(a, b)] != INSIDE;
    let grad_t_x = match (known(i, j + 1), known(i, j - 1)) {
        (true, true) => (tt(i, j + 1) - tt(i, j - 1)) * 0.5,
        (true, false) => tt(i, j + 1) - tt(i, j),
        (false, true) => tt(i, j) - tt(i, j - 1),
        (false, false) => 0.0,
    };
    let grad_t_y = match (known(i + 1, j), known(i - 1, j)) {
        (true, true) => (tt(i + 1, j) - tt(i - 1, j)) * 0.5,
        (true, false) => tt(i + 1, j) - tt(i, j),
        (false, true) => tt(i, j) - tt(i - 1, j),
        (false, false) => 0.0,
    };
    let mut ia = [0.0f32; 3];
    let mut jx = [0.0f32; 3];
    let mut jy = [0.0f32; 3];
    let mut s = [1.0e-20f32; 3];
    for (k, l) in window(g, f, i, j, range) {
        let tap = Taps::new(g, k, l);
        let rx = j as f32 - l as f32;
        let ry = i as f32 - k as f32;
        let len = rx * rx + ry * ry;
        let dst = (1.0 / (f64::from(len) * f64::from(len).sqrt())) as f32;
        let lev = (1.0 / (1.0 + f64::from((tt(k, l) - tt(i, j)).abs()))) as f32;
        let mut dir = rx * grad_t_x + ry * grad_t_y;
        if f64::from(dir.abs()) <= 0.01 {
            dir = 0.000_001;
        }
        let w = (dst * lev * dir).abs();
        for c in 0..cn {
            let d = |y0: usize, x0: usize, y1: usize, x1: usize| (px(out, y0, x0, c) - px(out, y1, x1, c)) as f32;
            let gx = match (known(k, l + 1), known(k, l - 1)) {
                (true, true) => d(tap.km, tap.lp + 1, tap.km, tap.lm - 1) * 2.0,
                (true, false) => d(tap.km, tap.lp + 1, tap.km, tap.lm),
                (false, true) => d(tap.km, tap.lp, tap.km, tap.lm - 1),
                (false, false) => 0.0,
            };
            let gy = match (known(k + 1, l), known(k - 1, l)) {
                (true, true) => d(tap.kp + 1, tap.lm, tap.km - 1, tap.lm) * 2.0,
                (true, false) => d(tap.kp + 1, tap.lm, tap.km, tap.lm),
                (false, true) => d(tap.kp, tap.lm, tap.km - 1, tap.lm),
                (false, false) => 0.0,
            };
            ia[c] += w * px(out, k - 1, l - 1, c) as f32;
            jx[c] -= w * (gx * rx);
            jy[c] -= w * (gy * ry);
            s[c] += w;
        }
    }
    for c in 0..cn {
        let norm = f64::from(jx[c] * jx[c] + jy[c] * jy[c]).sqrt() + f64::from(1.0e-20f32);
        let sat = (f64::from(ia[c] / s[c]) + f64::from(jx[c] + jy[c]) / norm) as f32;
        out.set(j - 1, i - 1, c, T::round_telea(sat));
    }
}

fn ns_pixel<T: InpaintElement>(g: &Grid, f: &[u8], _t: &[f32], out: &mut Mat<T>, range: usize, i: usize, j: usize) {
    let cn = out.channels();
    let known = |a: usize, b: usize| f[g.at(a, b)] != INSIDE;
    let mut ia = [0.0f32; 3];
    let mut s = [1.0e-20f32; 3];
    for (k, l) in window(g, f, i, j, range) {
        let tap = Taps::new(g, k, l);
        let (rx, ry) = if cn == 3 {
            (l as f32 - j as f32, k as f32 - i as f32)
        } else {
            (j as f32 - l as f32, i as f32 - k as f32)
        };
        let len = rx * rx + ry * ry;
        let dst = 1.0 / (len * len + 1.0);
        for c in 0..cn {
            let a = |y0: usize, x0: usize, y1: usize, x1: usize| (px(out, y0, x0, c) - px(out, y1, x1, c)).abs() as f32;
            let gx = match (known(k + 1, l), known(k - 1, l)) {
                (true, true) => a(tap.kp + 1, tap.lm, tap.kp, tap.lm) + a(tap.kp, tap.lm, tap.km - 1, tap.lm),
                (true, false) => a(tap.kp + 1, tap.lm, tap.kp, tap.lm) * 2.0,
                (false, true) => a(tap.kp, tap.lm, tap.km - 1, tap.lm) * 2.0,
                (false, false) => 0.0,
            };
            let gy = match (known(k, l + 1), known(k, l - 1)) {
                (true, true) => a(tap.km, tap.lp + 1, tap.km, tap.lm) + a(tap.km, tap.lm, tap.km, tap.lm - 1),
                (true, false) => a(tap.km, tap.lp + 1, tap.km, tap.lm) * 2.0,
                (false, true) => a(tap.km, tap.lm, tap.km, tap.lm - 1) * 2.0,
                (false, false) => 0.0,
            };
            let gx = -gx;
            let mut dir = rx * gx + ry * gy;
            if f64::from(dir.abs()) <= 0.01 {
                dir = 0.000_001;
            } else {
                dir = (f64::from(rx * gx + ry * gy) / f64::from(len * (gx * gx + gy * gy)).sqrt()).abs() as f32;
            }
            let w = dst * dir;
            ia[c] += w * px(out, k - 1, l - 1, c) as f32;
            s[c] += w;
        }
    }
    for c in 0..cn {
        out.set(j - 1, i - 1, c, T::from_f64(f64::from(ia[c]) / f64::from(s[c])));
    }
}

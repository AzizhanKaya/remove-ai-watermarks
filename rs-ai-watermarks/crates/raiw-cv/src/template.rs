//! `cv::matchTemplate` with `TM_CCOEFF_NORMED`, following `common_matchTemplate`'s normalization.
//!
//! The cross-correlation is the exact float64 sum: integer-valued inputs are exact in any order
//! (and in `f32` partial sums below 2^24), real-valued inputs keep the scalar reference order.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::{Arc, OnceLock};
use std::thread::LocalKey;

use bytemuck::Pod;
use num_complex::Complex64;
use pulp::{Simd, WithSimd};
use rayon::prelude::*;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

use crate::element::{Depth, Element};
use crate::error::{Result, invalid};
use crate::mat::{Mat, MatView};
use crate::stats::mean_std_dev;

/// Real-valued inputs correlate directly up to this many multiply-adds and through the FFT above.
const DIRECT_LIMIT: usize = 1 << 26;
/// Register block: output rows by vectors per row (eight independent accumulators).
const ROWS: usize = 4;
const VECTORS: usize = 2;
/// Widest vector in bytes (AVX-512); sizes the zero padding a block may read.
const MAX_VECTOR_BYTES: usize = 64;
/// Multiply-adds (or outputs) below which a stage stays on the calling thread.
const PARALLEL_WORK: usize = 1 << 20;
const PARALLEL_OUTPUTS: usize = 1 << 14;
const PARALLEL_FFT_POINTS: usize = 1 << 16;
/// Integer sums are exact up to these magnitudes (the `f64` one also bounds [`round`]).
const F32_EXACT: f64 = 16_777_216.0;
const F64_EXACT: f64 = 2_251_799_813_685_248.0;
/// Cost of one `n log2 n` FFT point in direct `f32` multiply-adds (measured).
const FFT_POINT_COST: f64 = 5.0;

/// Spare buffers kept per thread and element type, and the largest one worth keeping.
const SPARES: usize = 4;
const SPARE_BYTES: usize = 2 << 20;

thread_local! {
    /// Planners cache plans by size; a cached plan computes exactly what a fresh one does.
    static PLANNERS: RefCell<(RealFftPlanner<f64>, FftPlanner<f64>)> =
        RefCell::new((RealFftPlanner::new(), FftPlanner::new()));
}

/// Element types with a per-thread pool of spare work buffers: freed large buffers are unmapped
/// by the allocator and page-faulted back on the next call, which costs more than the matching.
trait Pooled: Copy + 'static {
    fn pool() -> &'static LocalKey<Cell<Vec<Vec<Self>>>>;
}

macro_rules! pooled {
    ($($t:ty),*) => {$(
        impl Pooled for $t {
            fn pool() -> &'static LocalKey<Cell<Vec<Vec<Self>>>> {
                thread_local!(static POOL: Cell<Vec<Vec<$t>>> = const { Cell::new(Vec::new()) });
                &POOL
            }
        }
    )*};
}

pooled!(f32, f64, Complex64);

/// A `len`-element buffer filled with `value`, reusing a spare that fits when there is one.
fn buffer<E: Pooled>(len: usize, value: E) -> Vec<E> {
    let mut spares = E::pool().take();
    let mut buf = match spares.iter().position(|b| b.capacity() >= len) {
        Some(fit) => spares.swap_remove(fit),
        None => spares.pop().unwrap_or_default(),
    };
    E::pool().set(spares);
    buf.clear();
    buf.resize(len, value);
    buf
}

/// Hand `buf` back to this thread's spares, which keep the largest few.
fn recycle<E: Pooled>(buf: Vec<E>) {
    if buf.capacity() * size_of::<E>() > SPARE_BYTES {
        return;
    }
    let mut spares = E::pool().take();
    spares.push(buf);
    if spares.len() > SPARES {
        spares.sort_unstable_by_key(Vec::capacity);
        spares.remove(0);
    }
    E::pool().set(spares);
}

/// `cv::matchTemplate(image, templ, TM_CCOEFF_NORMED)`; the result is `(W-w+1) x (H-h+1)` `f32`.
pub fn match_template_ccoeff_normed<'a, 'b, T: Element>(
    image: impl Into<MatView<'a, T>>,
    templ: impl Into<MatView<'b, T>>,
) -> Result<Mat<f32>> {
    TemplateSearch::new(image).match_template(templ)
}

/// One image prepared for [`match_template_ccoeff_normed`] with many templates: its integral
/// tables, planar copies and FFT spectra are built on first use and shared by every match.
#[derive(Debug)]
pub struct TemplateSearch<'a, T> {
    image: Image<'a, T>,
    integrals: OnceLock<(Vec<f64>, Vec<f64>)>,
    planes_f32: OnceLock<Vec<f32>>,
    planes_f64: OnceLock<Vec<f64>>,
    spectra: OnceLock<Vec<Vec<Complex64>>>,
}

/// The searched image, borrowed or owned (so a search can outlive the call that built it).
#[derive(Debug)]
enum Image<'a, T> {
    View(MatView<'a, T>),
    Owned(Mat<T>),
}

impl<'a, T: Element> TemplateSearch<'a, T> {
    pub fn new(image: impl Into<MatView<'a, T>>) -> Self {
        Self::with(Image::View(image.into()))
    }

    /// A search that owns its image.
    pub fn owned(image: Mat<T>) -> TemplateSearch<'static, T> {
        TemplateSearch::with(Image::Owned(image))
    }

    fn with(image: Image<'a, T>) -> Self {
        Self {
            image,
            integrals: OnceLock::new(),
            planes_f32: OnceLock::new(),
            planes_f64: OnceLock::new(),
            spectra: OnceLock::new(),
        }
    }

    pub fn image(&self) -> MatView<'_, T> {
        match &self.image {
            Image::View(view) => *view,
            Image::Owned(mat) => mat.view(),
        }
    }

    /// The `TM_CCOEFF_NORMED` scores of `templ` over the image.
    pub fn match_template<'b>(&self, templ: impl Into<MatView<'b, T>>) -> Result<Mat<f32>> {
        let img = self.image();
        let tpl = templ.into();
        let cn = img.channels();
        let fits = tpl.width() <= img.width() && tpl.height() <= img.height();
        if tpl.channels() != cn || !fits || tpl.is_empty() {
            return Err(invalid(
                "match_template",
                "template must fit inside the image with the same channels",
            ));
        }
        let (rw, rh) = (img.width() - tpl.width() + 1, img.height() - tpl.height() + 1);
        let (mean, sdv) = mean_std_dev(tpl)?;
        let mut templ_norm: f64 = sdv.iter().map(|s| s * s).sum();
        if templ_norm < f64::EPSILON {
            return Ok(Mat::filled(rw, rh, 1, 1.0));
        }
        let inv_area = 1.0 / (tpl.width() * tpl.height()) as f64;
        templ_norm = templ_norm.sqrt() / inv_area.sqrt();

        let corr = self.cross_correlation(&tpl, rw, rh);
        let (sum, sqsum) = cached(&self.integrals, || integrals(&img));
        let scores = Scores {
            corr: &corr,
            width: rw,
            sum,
            sqsum,
            stride: (img.width() + 1) * cn,
            tw: tpl.width(),
            th: tpl.height(),
            cn,
            mean: &mean,
            inv_area,
            templ_norm,
        };
        let mut out = Mat::<f32>::zeros(rw, rh, 1);
        let run = |first: usize, out: &mut [f32]| {
            crate::simd::dispatch(ScoreRows {
                scores: &scores,
                first,
                out,
            });
        };
        if rw * rh >= PARALLEL_OUTPUTS {
            let rows = 8;
            out.data_mut()
                .par_chunks_mut(rows * rw)
                .enumerate()
                .for_each(|(b, out)| run(b * rows, out));
        } else {
            run(0, out.data_mut());
        }
        recycle(corr);
        Ok(out)
    }

    /// `sum_{i,j,c} img[y+i, x+j, c] * tpl[i, j, c]` in double precision (OpenCV's float32 DFT
    /// drifts by up to ~4.5e-3 of the normalized score in near-flat windows; this stays exact).
    fn cross_correlation(&self, tpl: &MatView<'_, T>, rw: usize, rh: usize) -> Vec<f64> {
        match Summation::of(tpl) {
            Summation::Ordered if rw * rh * tpl.width() * tpl.height() * tpl.channels() > DIRECT_LIMIT => {
                self.fft_correlation(tpl, rw, rh, false)
            }
            Summation::Ordered => self.direct::<f64>(tpl, rw, rh, usize::MAX),
            Summation::Exact { taps, chunk } => {
                if fft_is_cheaper(&self.image(), rw * rh * taps, self.spectra.get().is_some()) {
                    self.fft_correlation(tpl, rw, rh, true)
                } else if let Some(chunk) = chunk {
                    self.direct::<f32>(tpl, rw, rh, chunk)
                } else {
                    self.direct::<f64>(tpl, rw, rh, usize::MAX)
                }
            }
        }
    }
}

/// `cell`'s value, built outside the cell's lock: the work runs rayon tasks, and a stolen task
/// sharing this search would otherwise re-enter the initialization (a racing build is dropped).
fn cached<V>(cell: &OnceLock<V>, init: impl FnOnce() -> V) -> &V {
    if let Some(value) = cell.get() {
        return value;
    }
    let _ = cell.set(init());
    cell.get_or_init(|| unreachable!("the cell was just set"))
}

impl<T> Drop for TemplateSearch<'_, T> {
    fn drop(&mut self) {
        if let Some((sum, sq)) = self.integrals.take() {
            recycle(sum);
            recycle(sq);
        }
        self.planes_f32.take().into_iter().for_each(recycle);
        self.planes_f64.take().into_iter().for_each(recycle);
        self.spectra.take().into_iter().flatten().for_each(recycle);
    }
}

/// The correlation, integral tables and template constants behind every score.
struct Scores<'a> {
    corr: &'a [f64],
    width: usize,
    sum: &'a [f64],
    sqsum: &'a [f64],
    stride: usize,
    tw: usize,
    th: usize,
    cn: usize,
    mean: &'a [f64],
    inv_area: f64,
    templ_norm: f64,
}

/// Window sums `top[p] - top[p + span] - bottom[p] + bottom[p + span]` of one integral table.
#[inline(always)]
fn window<'a>(top: &'a [f64], bottom: &'a [f64], span: usize) -> impl Iterator<Item = f64> + 'a {
    top.iter()
        .zip(&top[span..])
        .zip(bottom.iter().zip(&bottom[span..]))
        .map(|((a, b), (c, d))| a - b - c + d)
}

impl Scores<'_> {
    /// `common_matchTemplate`'s score from the correlation and the window statistics.
    #[inline(always)]
    fn score(&self, num: f64, wnd_mean2: f64, wnd_sum2: f64) -> f32 {
        let diff2 = (wnd_sum2 - wnd_mean2).max(0.0);
        let t = if diff2 <= 0.5f64.min(10.0 * f64::from(f32::EPSILON) * wnd_sum2) {
            0.0
        } else {
            diff2.sqrt() * self.templ_norm
        };
        let num = if num.abs() < t {
            num / t
        } else if num.abs() < t * 1.125 {
            if num > 0.0 { 1.0 } else { -1.0 }
        } else {
            0.0
        };
        num as f32
    }

    /// Score row `i`: a single-channel loop the compiler vectorizes, or the general `cn` one.
    #[inline(always)]
    fn row(&self, i: usize, out: &mut [f32]) {
        let (cn, stride) = (self.cn, self.stride);
        let span = self.tw * cn;
        let top = &self.sum[i * stride..][..stride];
        let bottom = &self.sum[(i + self.th) * stride..][..stride];
        let top_q = &self.sqsum[i * stride..][..stride];
        let bottom_q = &self.sqsum[(i + self.th) * stride..][..stride];
        let corr = &self.corr[i * self.width..][..self.width];
        if cn == 1 {
            let mean = self.mean[0];
            let windows = window(top, bottom, span).zip(window(top_q, bottom_q, span));
            for ((o, &corr), (t, q)) in out.iter_mut().zip(corr).zip(windows) {
                let num = corr - t * mean;
                *o = self.score(num, (0.0 + t * t) * self.inv_area, 0.0 + q);
            }
            return;
        }
        for (j, (o, &corr)) in out.iter_mut().zip(corr).enumerate() {
            let p = j * cn;
            let mut num = corr;
            let mut wnd_mean2 = 0.0;
            let mut wnd_sum2 = 0.0;
            for k in 0..cn {
                let t = top[p + k] - top[p + span + k] - bottom[p + k] + bottom[p + span + k];
                wnd_mean2 += t * t;
                num -= t * self.mean[k];
            }
            wnd_mean2 *= self.inv_area;
            for k in 0..cn {
                wnd_sum2 += top_q[p + k] - top_q[p + span + k] - bottom_q[p + k] + bottom_q[p + span + k];
            }
            *o = self.score(num, wnd_mean2, wnd_sum2);
        }
    }
}

/// Score rows `first..` into `out`.
struct ScoreRows<'a> {
    scores: &'a Scores<'a>,
    first: usize,
    out: &'a mut [f32],
}

impl crate::simd::Kernel for ScoreRows<'_> {
    type Output = ();

    #[inline(always)]
    fn run(self) {
        for (i, row) in self.out.chunks_mut(self.scores.width).enumerate() {
            self.scores.row(self.first + i, row);
        }
    }
}

/// `cv::integral(img, sum, sqsum, CV_64F)` with `(W+1) x (H+1)` interleaved channels.
fn integrals<T: Element>(img: &MatView<'_, T>) -> (Vec<f64>, Vec<f64>) {
    let (w, h, cn) = (img.width(), img.height(), img.channels());
    let stride = (w + 1) * cn;
    let mut sum = buffer(stride * (h + 1), 0.0);
    let mut sq = buffer(stride * (h + 1), 0.0);
    // u8 running sums are exact in i32 (squares included) for rows up to 33025 pixels
    let narrow = T::DEPTH == Depth::U8 && i32::try_from(w * 255 * 255).is_ok();
    let mut run_i = (vec![0i32; stride - cn], vec![0i32; stride - cn]);
    let mut run_f = (vec![0.0f64; stride - cn], vec![0.0f64; stride - cn]);
    for y in 0..h {
        let row = img.row(y);
        let (top, bottom) = sum.split_at_mut((y + 1) * stride);
        let (top_q, bottom_q) = sq.split_at_mut((y + 1) * stride);
        let (above, out) = (&top[y * stride + cn..], &mut bottom[cn..stride]);
        let (above_q, out_q) = (&top_q[y * stride + cn..], &mut bottom_q[cn..stride]);
        match bytemuck::try_cast_slice::<T, u8>(row) {
            Ok(bytes) if narrow => {
                running_sums(bytes, cn, &mut run_i.0, &mut run_i.1, |v| i32::from(*v));
                crate::simd::vectorized(add_row, (out, above, &run_i.0[..]));
                crate::simd::vectorized(add_row, (out_q, above_q, &run_i.1[..]));
            }
            _ => {
                running_sums(row, cn, &mut run_f.0, &mut run_f.1, |v| v.to_f64());
                crate::simd::vectorized(add_row, (out, above, &run_f.0[..]));
                crate::simd::vectorized(add_row, (out_q, above_q, &run_f.1[..]));
            }
        }
    }
    (sum, sq)
}

/// Per-channel running sums of values and squares along one interleaved row.
fn running_sums<V, A>(row: &[V], cn: usize, s: &mut [A], q: &mut [A], of: impl Fn(&V) -> A)
where
    A: Copy + Default + Add<Output = A> + Mul<Output = A>,
{
    if cn == 1 {
        let (mut a, mut b) = (A::default(), A::default());
        for ((s, q), v) in s.iter_mut().zip(q.iter_mut()).zip(row) {
            let v = of(v);
            a = a + v;
            b = b + v * v;
            *s = a;
            *q = b;
        }
        return;
    }
    let mut run = vec![(A::default(), A::default()); cn];
    let pixels = s
        .chunks_exact_mut(cn)
        .zip(q.chunks_exact_mut(cn))
        .zip(row.chunks_exact(cn));
    for ((s, q), px) in pixels {
        for (((s, q), v), (a, b)) in s.iter_mut().zip(q.iter_mut()).zip(px).zip(run.iter_mut()) {
            let v = of(v);
            *a = *a + v;
            *b = *b + v * v;
            *s = *a;
            *q = *b;
        }
    }
}

/// `out = above + run`, one integral-table row.
#[inline(always)]
fn add_row<R: Copy + Into<f64>>((out, above, run): (&mut [f64], &[f64], &[R])) {
    for ((o, &a), &r) in out.iter_mut().zip(above).zip(run) {
        *o = a + r.into();
    }
}

/// How the cross-correlation may be summed without changing its value.
#[derive(Clone, Copy)]
enum Summation {
    /// Integer-valued below 2^51, so any order is exact; `f32` stays exact for `chunk` taps.
    Exact { taps: usize, chunk: Option<usize> },
    /// Real-valued: the scalar reference order `(row, column, channel)` in `f64`.
    Ordered,
}

impl Summation {
    fn of<T: Element>(tpl: &MatView<'_, T>) -> Self {
        if !matches!(T::DEPTH, Depth::U8 | Depth::I8 | Depth::U16 | Depth::I16) {
            return Self::Ordered;
        }
        let peak = T::LOWEST.to_f64().abs().max(T::HIGHEST.to_f64());
        let (taps, kmax) = tpl.iter().fold((0usize, 0.0f64), |(n, m), v| {
            let v = v.to_f64().abs();
            (n + usize::from(v != 0.0), m.max(v))
        });
        let term = (peak * kmax).max(1.0);
        if taps as f64 * term > F64_EXACT {
            return Self::Ordered;
        }
        let chunk = (term <= F32_EXACT).then(|| (F32_EXACT / term) as usize);
        Self::Exact { taps, chunk }
    }
}

/// Whether the FFT planes still to transform (image unless `cached`, template, inverse) cost
/// less than `work` direct multiply-adds.
fn fft_is_cheaper<T: Element>(img: &MatView<'_, T>, work: usize, cached: bool) -> bool {
    let points = (fft_size(img.width()) * fft_size(img.height())) as f64;
    let planes = ((2 - usize::from(cached)) * img.channels() + 1) as f64;
    FFT_POINT_COST * planes * points * points.log2() < work as f64
}

/// A direct-correlation accumulator lane.
trait Lane: Pod + Pooled + PartialEq + Send + Sync {
    const ZERO: Self;
    type V<S: Simd>: Copy + Pod;
    fn of<T: Element>(v: T) -> Self;
    fn to_f64(self) -> f64;
    fn splat<S: Simd>(simd: S, v: Self) -> Self::V<S>;
    /// `acc + k * x`.
    fn mac<S: Simd>(simd: S, acc: Self::V<S>, k: Self::V<S>, x: Self::V<S>) -> Self::V<S>;
    fn vectors<S: Simd>(values: &[Self]) -> &[Self::V<S>];
    /// The search's planar copy of its image in this lane type.
    fn planes<'s, T>(search: &'s TemplateSearch<'_, T>) -> &'s OnceLock<Vec<Self>>;
}

/// Integer-valued sums: exact while partial sums stay below 2^24, so fused multiply-adds are free.
impl Lane for f32 {
    const ZERO: Self = 0.0;
    type V<S: Simd> = S::f32s;

    fn of<T: Element>(v: T) -> Self {
        v.to_f32()
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn splat<S: Simd>(simd: S, v: Self) -> S::f32s {
        simd.splat_f32s(v)
    }
    #[inline(always)]
    fn mac<S: Simd>(simd: S, acc: S::f32s, k: S::f32s, x: S::f32s) -> S::f32s {
        simd.mul_add_e_f32s(k, x, acc)
    }
    #[inline(always)]
    fn vectors<S: Simd>(values: &[Self]) -> &[S::f32s] {
        S::as_simd_f32s(values).0
    }
    fn planes<'s, T>(search: &'s TemplateSearch<'_, T>) -> &'s OnceLock<Vec<Self>> {
        &search.planes_f32
    }
}

/// The scalar reference: a separate multiply and add per tap in `f64`.
impl Lane for f64 {
    const ZERO: Self = 0.0;
    type V<S: Simd> = S::f64s;

    fn of<T: Element>(v: T) -> Self {
        v.to_f64()
    }
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn splat<S: Simd>(simd: S, v: Self) -> S::f64s {
        simd.splat_f64s(v)
    }
    #[inline(always)]
    fn mac<S: Simd>(simd: S, acc: S::f64s, k: S::f64s, x: S::f64s) -> S::f64s {
        simd.add_f64s(acc, simd.mul_f64s(k, x))
    }
    #[inline(always)]
    fn vectors<S: Simd>(values: &[Self]) -> &[S::f64s] {
        S::as_simd_f64s(values).0
    }
    fn planes<'s, T>(search: &'s TemplateSearch<'_, T>) -> &'s OnceLock<Vec<Self>> {
        &search.planes_f64
    }
}

/// Elements per register block row on the running CPU.
struct BlockSpan<L>(PhantomData<L>);

impl<L: Lane> WithSimd for BlockSpan<L> {
    type Output = usize;

    #[inline(always)]
    fn with_simd<S: Simd>(self, _simd: S) -> usize {
        VECTORS * size_of::<L::V<S>>() / size_of::<L>()
    }
}

impl<T: Element> TemplateSearch<'_, T> {
    /// Planar layout shared by every template: rows of `stride` and channels of `plane` elements,
    /// zero-padded for the widest register block at the last output row and column.
    fn layout<L: Lane>(&self) -> (usize, usize) {
        let img = self.image();
        let stride = img.width() + VECTORS * MAX_VECTOR_BYTES / size_of::<L>() - 1;
        (stride, stride * (img.height() + ROWS - 1))
    }

    /// Register-blocked correlation over the zero-padded planar copy of the image.
    fn direct<L: Lane>(&self, tpl: &MatView<'_, T>, rw: usize, rh: usize, chunk: usize) -> Vec<f64> {
        let (stride, plane) = self.layout::<L>();
        let src = cached(L::planes(self), || planar(&self.image(), stride, plane));
        let taps = taps::<T, L>(tpl, stride, plane);
        let span = pulp::Arch::new().dispatch(BlockSpan::<L>(PhantomData));
        let mut out = buffer(rw * rh, 0.0);
        let run = |first_row: usize, out: &mut [f64]| {
            pulp::Arch::new().dispatch(Direct {
                src,
                stride,
                taps: &taps,
                chunk,
                span,
                out,
                width: rw,
                first_row,
            });
        };
        if rw * rh * taps.len() >= PARALLEL_WORK {
            out.par_chunks_mut(ROWS * rw)
                .enumerate()
                .for_each(|(b, rows)| run(b * ROWS, rows));
        } else {
            run(0, &mut out);
        }
        out
    }
}

/// Planar copy of `img`: per channel, `plane` elements of zero-padded `stride`-wide rows.
fn planar<T: Element, L: Lane>(img: &MatView<'_, T>, stride: usize, plane: usize) -> Vec<L> {
    let cn = img.channels();
    let mut data = buffer(plane * cn, L::ZERO);
    for (c, channel) in data.chunks_mut(plane).enumerate() {
        for (y, dst) in channel.chunks_mut(stride).take(img.height()).enumerate() {
            let row = img.row(y);
            if cn == 1 {
                for (d, &v) in dst.iter_mut().zip(row) {
                    *d = L::of(v);
                }
            } else {
                for (d, px) in dst.iter_mut().zip(row.chunks_exact(cn)) {
                    *d = L::of(px[c]);
                }
            }
        }
    }
    data
}

/// Nonzero template taps in `(row, column, channel)` order, as offsets into the planar image.
fn taps<T: Element, L: Lane>(tpl: &MatView<'_, T>, stride: usize, plane: usize) -> Vec<(usize, L)> {
    let cn = tpl.channels();
    (0..tpl.height())
        .flat_map(|i| {
            tpl.row(i)
                .iter()
                .enumerate()
                .map(move |(n, &v)| ((n % cn) * plane + i * stride + n / cn, L::of(v)))
        })
        .filter(|&(_, k)| k != L::ZERO)
        .collect()
}

/// Correlation rows `first_row..` into `out`, in register blocks of `ROWS x span` outputs.
struct Direct<'a, L> {
    src: &'a [L],
    stride: usize,
    taps: &'a [(usize, L)],
    /// Taps per partial sum, flushed into the `f64` output between groups.
    chunk: usize,
    span: usize,
    out: &'a mut [f64],
    width: usize,
    first_row: usize,
}

impl<L: Lane> WithSimd for Direct<'_, L> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            src,
            stride,
            taps,
            chunk,
            span,
            out,
            width,
            first_row,
        } = self;
        debug_assert_eq!(span, VECTORS * size_of::<L::V<S>>() / size_of::<L>());
        for (b, rows) in out.chunks_mut(ROWS * width).enumerate() {
            let base = (first_row + b * ROWS) * stride;
            for x in (0..width).step_by(span) {
                for group in taps.chunks(chunk) {
                    let acc = block(simd, src, stride, base + x, group);
                    for (row, acc) in rows.chunks_mut(width).zip(&acc) {
                        let lanes: &[L] = bytemuck::cast_slice(acc);
                        for (o, a) in row[x..].iter_mut().zip(lanes) {
                            *o += a.to_f64();
                        }
                    }
                }
            }
        }
    }
}

/// One `ROWS x VECTORS` register block at `base`, summed over `taps` in order.
#[inline(always)]
fn block<S: Simd, L: Lane>(
    simd: S,
    src: &[L],
    stride: usize,
    base: usize,
    taps: &[(usize, L)],
) -> [[L::V<S>; VECTORS]; ROWS] {
    let span = VECTORS * size_of::<L::V<S>>() / size_of::<L>();
    let reach = (ROWS - 1) * stride + span;
    let mut acc = [[L::splat(simd, L::ZERO); VECTORS]; ROWS];
    for &(offset, k) in taps {
        let k = L::splat(simd, k);
        let window = &src[base + offset..][..reach];
        for (r, acc) in acc.iter_mut().enumerate() {
            let Some(x) = L::vectors::<S>(&window[r * stride..][..span]).first_chunk::<VECTORS>() else {
                unreachable!("a block row spans VECTORS vectors")
            };
            for (a, &x) in acc.iter_mut().zip(x) {
                *a = L::mac(simd, *a, k, x);
            }
        }
    }
    acc
}

fn fft_size(n: usize) -> usize {
    let mut best = n.next_power_of_two();
    for a in [1usize, 3, 5, 9, 15, 25, 27, 45, 75, 81] {
        let mut m = a;
        while m < n {
            m *= 2;
        }
        best = best.min(m);
    }
    best
}

impl<T: Element> TemplateSearch<'_, T> {
    /// Real-input 2D FFT correlation: R2C rows, blocked transpose, complex columns (kept
    /// transposed). Integer-valued sums are rounded back to their exact value.
    fn fft_correlation(&self, tpl: &MatView<'_, T>, rw: usize, rh: usize, integral: bool) -> Vec<f64> {
        let img = self.image();
        let plans = Plans::new(fft_size(img.width()), fft_size(img.height()));
        let spectra = cached(&self.spectra, || {
            (0..img.channels()).map(|c| plans.spectrum(&img, c)).collect()
        });
        let mut acc = buffer(plans.nc * plans.fh, ZERO);
        for (c, a) in spectra.iter().enumerate() {
            let b = plans.spectrum(tpl, c);
            plans.each(&mut acc, 1 << 12, |at, acc| {
                for (o, (a, b)) in acc.iter_mut().zip(a[at..].iter().zip(&b[at..])) {
                    *o += a * b.conj();
                }
            });
            recycle(b);
        }
        plans.columns(&mut acc, &*plans.inv);
        let mut rows = buffer(rh * plans.nc, ZERO);
        plans.transpose(&acc, plans.nc, plans.fh, &mut rows);
        recycle(acc);
        let norm = 1.0 / (plans.fw * plans.fh) as f64;
        let mut out = buffer(rw * rh, 0.0);
        let (fw, nc) = (plans.fw, plans.nc);
        let init = || (vec![ZERO; nc], vec![0.0; fw], plans.c2r.make_scratch_vec());
        plans.each_with(&mut out, rw, init, |(spec, line, scratch), at, dst| {
            spec.copy_from_slice(&rows[at / rw * nc..][..nc]);
            spec[0].im = 0.0;
            if fw.is_multiple_of(2) {
                spec[nc - 1].im = 0.0;
            }
            let _ = plans.c2r.process_with_scratch(spec, line, scratch);
            for (d, v) in dst.iter_mut().zip(line.iter()) {
                *d = if integral { round(v * norm) } else { v * norm };
            }
        });
        recycle(rows);
        out
    }
}

/// Nearest integer for `|v| < 2^51` (adding and removing 1.5 * 2^52, without a libm call).
fn round(v: f64) -> f64 {
    const MAGIC: f64 = 6_755_399_441_055_744.0;
    (v + MAGIC) - MAGIC
}

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// FFT plans for one padded size; stages run in parallel only for large transforms.
struct Plans {
    fw: usize,
    fh: usize,
    nc: usize,
    r2c: Arc<dyn RealToComplex<f64>>,
    c2r: Arc<dyn ComplexToReal<f64>>,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
    parallel: bool,
}

impl Plans {
    fn new(fw: usize, fh: usize) -> Self {
        let (r2c, c2r, fwd, inv) = PLANNERS.with_borrow_mut(|(real, complex)| {
            (
                real.plan_fft_forward(fw),
                real.plan_fft_inverse(fw),
                complex.plan_fft_forward(fh),
                complex.plan_fft_inverse(fh),
            )
        });
        Self {
            fw,
            fh,
            nc: fw / 2 + 1,
            r2c,
            c2r,
            fwd,
            inv,
            parallel: fw * fh >= PARALLEL_FFT_POINTS,
        }
    }

    /// `f(offset, chunk)` over `chunk`-sized pieces of `data`.
    fn each<E: Send>(&self, data: &mut [E], chunk: usize, f: impl Fn(usize, &mut [E]) + Sync + Send) {
        self.each_with(data, chunk, || (), |(), at, part| f(at, part));
    }

    /// `f(state, offset, chunk)` over `chunk`-sized pieces of `data`, with per-worker `state`.
    fn each_with<E: Send, S>(
        &self,
        data: &mut [E],
        chunk: usize,
        init: impl Fn() -> S + Sync + Send,
        f: impl Fn(&mut S, usize, &mut [E]) + Sync + Send,
    ) {
        if self.parallel {
            data.par_chunks_mut(chunk)
                .enumerate()
                .for_each_init(&init, |state, (i, part)| f(state, i * chunk, part));
        } else {
            let mut state = init();
            for (i, part) in data.chunks_mut(chunk).enumerate() {
                f(&mut state, i * chunk, part);
            }
        }
    }

    /// Transposed (`nc x fh`) spectrum of channel `c` of `src`, zero-padded to `fw x fh`.
    fn spectrum<T: Element>(&self, src: &MatView<'_, T>, c: usize) -> Vec<Complex64> {
        let (fw, nc, cn) = (self.fw, self.nc, src.channels());
        let mut rows = buffer(self.fh * nc, ZERO);
        let filled = &mut rows[..src.height() * nc];
        let init = || (vec![0.0; fw], self.r2c.make_scratch_vec());
        self.each_with(filled, nc, init, |(line, scratch), at, out| {
            let (head, tail) = line.split_at_mut(src.width());
            for (d, px) in head.iter_mut().zip(src.row(at / nc).chunks_exact(cn)) {
                *d = px[c].to_f64();
            }
            tail.fill(0.0);
            let _ = self.r2c.process_with_scratch(line, out, scratch);
        });
        let mut cols = buffer(nc * self.fh, ZERO);
        self.transpose(&rows, self.fh, nc, &mut cols);
        recycle(rows);
        self.columns(&mut cols, &*self.fwd);
        cols
    }

    /// One `fh`-point transform per column of the transposed layout.
    fn columns(&self, cols: &mut [Complex64], fft: &dyn Fft<f64>) {
        let init = || vec![ZERO; fft.get_inplace_scratch_len()];
        self.each_with(cols, self.fh * 8, init, |scratch, _, batch| {
            fft.process_with_scratch(batch, scratch);
        });
    }

    /// Cache-blocked transpose of a `rows x cols` row-major matrix into the leading rows of `dst`
    /// (`dst.len() / rows` of the `cols` output rows).
    fn transpose(&self, src: &[Complex64], rows: usize, cols: usize, dst: &mut [Complex64]) {
        const B: usize = 32;
        self.each(dst, rows * B, |at, block| {
            let (c0, n) = (at / rows, block.len() / rows);
            for (r, src) in src.chunks_exact(cols).enumerate() {
                for (k, &v) in src[c0..c0 + n].iter().enumerate() {
                    block[k * rows + r] = v;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(w: usize, h: usize, cn: usize, seed: u64) -> Mat<u8> {
        let mut state = seed;
        Mat::<u8>::from_fn(w, h, cn, |_, _, _| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (state >> 56) as u8
        })
    }

    #[test]
    fn fft_and_direct_correlation_agree() {
        for (w, h, cn) in [(83usize, 61usize, 1usize), (64, 48, 3), (75, 50, 1)] {
            let img = noise(w, h, cn, 9);
            let tpl = img.crop(crate::types::Rect::new(10, 7, 21, 13)).expect("crop");
            let (rw, rh) = (w - 20, h - 12);
            let search = TemplateSearch::new(&img);
            let direct = search.direct::<f64>(&tpl.view(), rw, rh, usize::MAX);
            let fft = search.fft_correlation(&tpl.view(), rw, rh, false);
            let worst = direct
                .iter()
                .zip(&fft)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
                .fold(0.0, f64::max);
            assert!(worst < 1e-9, "{w}x{h}x{cn}: relative error {worst}");
            let rounded = search.fft_correlation(&tpl.view(), rw, rh, true);
            assert_eq!(direct, rounded, "{w}x{h}x{cn}: rounded FFT");
        }
    }

    #[test]
    fn exact_f32_blocks_match_f64_sums() {
        let cases = [(37, 9, 1, 5, 3), (90, 41, 3, 17, 11), (300, 20, 1, 280, 4)];
        for (w, h, cn, tw, th) in cases {
            let img = noise(w, h, cn, 3);
            let tpl = noise(tw, th, cn, 5);
            let (rw, rh) = (w - tw + 1, h - th + 1);
            let search = TemplateSearch::new(&img);
            let reference = search.direct::<f64>(&tpl.view(), rw, rh, usize::MAX);
            let chunk = (F32_EXACT / (255.0 * 255.0)) as usize;
            let blocks = search.direct::<f32>(&tpl.view(), rw, rh, chunk);
            assert_eq!(reference, blocks, "{w}x{h}x{cn} / {tw}x{th}");
        }
    }

    #[test]
    fn shared_search_matches_single_calls() {
        let img = noise(120, 70, 1, 11);
        let real = img.convert::<f32>();
        let (search, search_real) = (TemplateSearch::new(&img), TemplateSearch::owned(real.clone()));
        for (seed, tw, th) in [(1, 9, 7), (2, 40, 14), (3, 9, 7), (4, 101, 60)] {
            let tpl = noise(tw, th, 1, seed);
            let shared = search.match_template(&tpl).expect("shared");
            let single = match_template_ccoeff_normed(&img, &tpl).expect("single");
            assert_eq!(shared, single, "{tw}x{th}");
            let tpl = tpl.convert::<f32>();
            let shared = search_real.match_template(&tpl).expect("shared f32");
            let single = match_template_ccoeff_normed(&real, &tpl).expect("single f32");
            assert_eq!(shared, single, "{tw}x{th} f32");
        }
    }
}

//! Separable and 2D linear filtering with OpenCV border semantics, row-band parallel.

use rayon::prelude::*;

use crate::border::{Border, border_table};
use crate::element::Element;
use crate::element::num_like::FloatOps;
use crate::mat::{Mat, MatView};

/// Floating work type of a filter (`float` or `double`, chosen like OpenCV's `bdepth`).
pub(crate) trait Work: FloatOps + Default + Send + Sync + 'static {
    fn from_elem<T: Element>(v: T) -> Self;
    fn to_elem<T: Element>(self) -> T;
}

impl Work for f32 {
    #[inline]
    fn from_elem<T: Element>(v: T) -> Self {
        v.to_f32()
    }
    #[inline]
    fn to_elem<T: Element>(self) -> T {
        T::from_f32(self)
    }
}

impl Work for f64 {
    #[inline]
    fn from_elem<T: Element>(v: T) -> Self {
        v.to_f64()
    }
    #[inline]
    fn to_elem<T: Element>(self) -> T {
        T::from_f64(self)
    }
}

/// One separable filter: a row pass into a buffer type and a column pass to the output.
pub(crate) trait SepOps: Sync {
    type S: Element;
    type B: Copy + Default + Send + Sync;
    type D: Element;
    fn kx_len(&self) -> usize;
    fn ky_len(&self) -> usize;
    /// `ext` holds the border-extended source row; writes `out.len()` filtered samples.
    fn row(&self, ext: &[Self::S], out: &mut [Self::B], cn: usize);
    /// `rows[k]` is the filtered row at vertical tap `k`.
    fn col(&self, rows: &[&[Self::B]], out: &mut [Self::D]);
}

/// Border-extend a source row into `ext` (`ext.len() == xs.len() * cn`).
pub(crate) fn extend_row<T: Element>(src: &[T], xs: &[Option<usize>], fill: &[T], ext: &mut [T]) {
    let cn = fill.len();
    for (e, sx) in ext.chunks_exact_mut(cn).zip(xs) {
        match sx {
            Some(sx) => e.copy_from_slice(&src[sx * cn..(sx + 1) * cn]),
            None => e.copy_from_slice(fill),
        }
    }
}

fn band_size(height: usize, taps: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    height.div_ceil(threads * 2).max(taps.max(16)).min(height.max(1))
}

pub(crate) fn run_separable<O: SepOps>(src: &MatView<'_, O::S>, ops: &O, border: Border) -> Mat<O::D> {
    let (w, h, cn) = (src.width(), src.height(), src.channels());
    let (kx, ky) = (ops.kx_len(), ops.ky_len());
    let (ax, ay) = (kx / 2, ky / 2);
    let xs = border_table(w, ax, kx - 1 - ax, border);
    let ys = border_table(h, ay, ky - 1 - ay, border);
    let fill: Vec<O::S> = (0..cn).map(|c| O::S::from_f64(border.constant_value(c))).collect();
    let row_len = w * cn;
    let mut dst = Mat::<O::D>::zeros(w, h, cn);
    let band = band_size(h, ky);
    dst.data_mut().par_chunks_mut(row_len * band).enumerate().for_each(|(b, out)| {
        let y0 = b * band;
        let rows_out = out.len() / row_len;
        let mut ext = vec![O::S::ZERO; xs.len() * cn];
        let mut filtered = vec![O::B::default(); (rows_out + ky - 1) * row_len];
        for (r, chunk) in filtered.chunks_exact_mut(row_len).enumerate() {
            match ys[y0 + r] {
                Some(sy) => extend_row(src.row(sy), &xs, &fill, &mut ext),
                None => ext.chunks_exact_mut(cn).for_each(|e| e.copy_from_slice(&fill)),
            }
            ops.row(&ext, chunk, cn);
        }
        let mut taps: Vec<&[O::B]> = Vec::with_capacity(ky);
        for (i, drow) in out.chunks_exact_mut(row_len).enumerate() {
            taps.clear();
            taps.extend((0..ky).map(|k| &filtered[(i + k) * row_len..(i + k + 1) * row_len]));
            ops.col(&taps, drow);
        }
    });
    dst
}

/// Float separable filter with OpenCV's symmetric-kernel summation order.
pub(crate) struct FloatSep<S, D, W> {
    pub kx: Vec<W>,
    pub ky: Vec<W>,
    pub delta: W,
    pub _types: std::marker::PhantomData<(S, D)>,
}

impl<S, D, W: Work> FloatSep<S, D, W> {
    pub fn new(kx: &[f64], ky: &[f64], delta: f64) -> Self {
        Self {
            kx: kx.iter().map(|&v| W::of_f64(v)).collect(),
            ky: ky.iter().map(|&v| W::of_f64(v)).collect(),
            delta: W::of_f64(delta),
            _types: std::marker::PhantomData,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Symmetry {
    Symmetric,
    Antisymmetric,
    General,
}

fn symmetry<W: Work>(k: &[W]) -> Symmetry {
    let n = k.len();
    if n % 2 == 1 && (0..n / 2).all(|i| k[i] == k[n - 1 - i]) {
        Symmetry::Symmetric
    } else if n % 2 == 1 && k[n / 2] == W::default() && (0..n / 2).all(|i| k[i] == -k[n - 1 - i]) {
        Symmetry::Antisymmetric
    } else {
        Symmetry::General
    }
}

impl<S: Element, D: Element, W: Work> SepOps for FloatSep<S, D, W> {
    type S = S;
    type B = W;
    type D = D;

    fn kx_len(&self) -> usize {
        self.kx.len()
    }

    fn ky_len(&self) -> usize {
        self.ky.len()
    }

    fn row(&self, ext: &[S], out: &mut [W], cn: usize) {
        let k = &self.kx;
        let half = k.len() / 2;
        let sym = if k.len() <= 5 { symmetry(k) } else { Symmetry::General };
        let x: Vec<W> = ext.iter().map(|&v| W::from_elem(v)).collect();
        let at = |i: usize| x[i];
        for (i, o) in out.iter_mut().enumerate() {
            let c = i + half * cn;
            *o = match sym {
                Symmetry::Symmetric => {
                    let mut s = k[half] * at(c);
                    for j in 1..=half {
                        s = s + k[half + j] * (at(c - j * cn) + at(c + j * cn));
                    }
                    s
                }
                Symmetry::Antisymmetric => {
                    let mut s = W::default();
                    for j in 1..=half {
                        s = s + k[half + j] * (at(c + j * cn) - at(c - j * cn));
                    }
                    s
                }
                Symmetry::General => {
                    let mut s = W::default();
                    for (j, &kj) in k.iter().enumerate() {
                        s = s + kj * at(i + j * cn);
                    }
                    s
                }
            };
        }
    }

    fn col(&self, rows: &[&[W]], out: &mut [D]) {
        // Row-wise accumulation keeps OpenCV's per-element summation order and vectorizes.
        let k = &self.ky;
        let half = k.len() / 2;
        let n = out.len();
        let mut acc = vec![self.delta; n];
        match symmetry(k) {
            Symmetry::Symmetric => {
                for (a, &v) in acc.iter_mut().zip(&rows[half][..n]) {
                    *a = k[half] * v + self.delta;
                }
                for j in 1..=half {
                    let (kj, up, down) = (k[half + j], &rows[half + j][..n], &rows[half - j][..n]);
                    for ((a, &p), &m) in acc.iter_mut().zip(up).zip(down) {
                        *a = *a + kj * (p + m);
                    }
                }
            }
            Symmetry::Antisymmetric => {
                for j in 1..=half {
                    let (kj, up, down) = (k[half + j], &rows[half + j][..n], &rows[half - j][..n]);
                    for ((a, &p), &m) in acc.iter_mut().zip(up).zip(down) {
                        *a = *a + kj * (p - m);
                    }
                }
            }
            Symmetry::General => {
                for (&kj, row) in k.iter().zip(rows) {
                    for (a, &v) in acc.iter_mut().zip(&row[..n]) {
                        *a = *a + kj * v;
                    }
                }
            }
        }
        for (o, a) in out.iter_mut().zip(acc) {
            *o = a.to_elem();
        }
    }
}

/// Bit-exact `GaussianBlurFixedPoint` with `ufixedpoint16` (8-bit, `FixedGauss<u8>`) or
/// `ufixedpoint32` (16-bit, `FixedGauss<u16>`) taps. Integer sums are exact, so the symmetric
/// row kernel folds into half the multiplies and the column pass runs in blocks.
pub(crate) struct FixedGauss<T> {
    kx: Vec<u32>,
    ky: Vec<u32>,
    _type: std::marker::PhantomData<T>,
}

impl<T: Element> FixedGauss<T> {
    pub fn new(ksize: (usize, usize), sigma: (f64, f64)) -> Self {
        let bits = if T::DEPTH == crate::Depth::U8 { 8 } else { 16 };
        let taps = |n, s| super::kernels::gaussian_kernel_fixed(n, s, bits).into_iter().map(|v| v as u32).collect();
        Self { kx: taps(ksize.0, sigma.0), ky: taps(ksize.1, sigma.1), _type: std::marker::PhantomData }
    }
}

macro_rules! fixed_gauss {
    ($t:ty, $buf:ty, $acc:ty, $shift:expr) => {
        impl SepOps for FixedGauss<$t> {
            type S = $t;
            type B = $buf;
            type D = $t;

            fn kx_len(&self) -> usize {
                self.kx.len()
            }

            fn ky_len(&self) -> usize {
                self.ky.len()
            }

            fn row(&self, ext: &[$t], out: &mut [$buf], cn: usize) {
                let half = self.kx.len() / 2;
                let kc = <$acc>::from(self.kx[half]);
                for (i, o) in out.iter_mut().enumerate() {
                    let c = i + half * cn;
                    let mut s = kc * <$acc>::from(ext[c]);
                    for j in 1..=half {
                        s += <$acc>::from(self.kx[half + j])
                            * (<$acc>::from(ext[c - j * cn]) + <$acc>::from(ext[c + j * cn]));
                    }
                    *o = s.min(<$acc>::from(<$buf>::MAX)) as $buf;
                }
            }

            fn col(&self, rows: &[&[$buf]], out: &mut [$t]) {
                const BLOCK: usize = 64;
                let round: $acc = 1 << ($shift - 1);
                for (start, chunk) in (0..out.len()).step_by(BLOCK).zip(out.chunks_mut(BLOCK)) {
                    let mut acc: [$acc; BLOCK] = [0; BLOCK];
                    for (&k, r) in self.ky.iter().zip(rows) {
                        let k = <$acc>::from(k);
                        for (a, &v) in acc.iter_mut().zip(&r[start..start + chunk.len()]) {
                            *a += k * <$acc>::from(v);
                        }
                    }
                    for (o, a) in chunk.iter_mut().zip(acc) {
                        *o = ((a + round) >> $shift).min(<$acc>::from(<$t>::MAX)) as $t;
                    }
                }
            }
        }
    };
}

fixed_gauss!(u8, u16, u32, 16);
fixed_gauss!(u16, u32, u64, 32);

/// `filter2D` by direct correlation over the nonzero taps, in row-major tap order.
pub(crate) fn run_filter2d<S: Element, D: Element, W: Work>(
    src: &MatView<'_, S>,
    kernel: &[f64],
    kw: usize,
    kh: usize,
    anchor: (usize, usize),
    delta: f64,
    border: Border,
) -> Mat<D> {
    let (w, h, cn) = (src.width(), src.height(), src.channels());
    let taps: Vec<(usize, usize, W)> = (0..kh)
        .flat_map(|ky| (0..kw).map(move |kx| (kx, ky)))
        .filter_map(|(kx, ky)| {
            let v = kernel[ky * kw + kx];
            (v != 0.0).then(|| (kx, ky, W::of_f64(v)))
        })
        .collect();
    let xs = border_table(w, anchor.0, kw - 1 - anchor.0, border);
    let ys = border_table(h, anchor.1, kh - 1 - anchor.1, border);
    let fill: Vec<S> = (0..cn).map(|c| S::from_f64(border.constant_value(c))).collect();
    let delta = W::of_f64(delta);
    let row_len = w * cn;
    let mut dst = Mat::<D>::zeros(w, h, cn);
    let band = band_size(h, kh);
    dst.data_mut().par_chunks_mut(row_len * band).enumerate().for_each(|(b, out)| {
        let y0 = b * band;
        let rows_out = out.len() / row_len;
        let ext: Vec<Vec<W>> = (y0..y0 + rows_out + kh - 1)
            .map(|ey| {
                let mut e = vec![S::ZERO; xs.len() * cn];
                match ys[ey] {
                    Some(sy) => extend_row(src.row(sy), &xs, &fill, &mut e),
                    None => e.chunks_exact_mut(cn).for_each(|p| p.copy_from_slice(&fill)),
                }
                e.into_iter().map(W::from_elem).collect()
            })
            .collect();
        for (i, drow) in out.chunks_exact_mut(row_len).enumerate() {
            for (x, o) in drow.iter_mut().enumerate() {
                let mut s = delta;
                for &(kx, ky, k) in &taps {
                    s = s + k * ext[i + ky][x + kx * cn];
                }
                *o = s.to_elem();
            }
        }
    });
    dst
}

//! `cv::matchTemplate` with `TM_CCOEFF_NORMED`, following `common_matchTemplate`'s normalization.

use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::FftPlanner;

use crate::element::Element;
use crate::error::{Result, invalid};
use crate::mat::{Mat, MatView};
use crate::stats::mean_std_dev;

/// Direct correlation is used below this many multiply-adds; larger searches go through the FFT.
const DIRECT_LIMIT: usize = 1 << 26;

/// `cv::matchTemplate(image, templ, TM_CCOEFF_NORMED)`; the result is `(W-w+1) x (H-h+1)` `f32`.
pub fn match_template_ccoeff_normed<'a, 'b, T: Element>(
    image: impl Into<MatView<'a, T>>,
    templ: impl Into<MatView<'b, T>>,
) -> Result<Mat<f32>> {
    let img = image.into();
    let tpl = templ.into();
    let cn = img.channels();
    if tpl.channels() != cn || tpl.width() > img.width() || tpl.height() > img.height() || tpl.is_empty() {
        return Err(invalid("match_template", "template must fit inside the image with the same channels"));
    }
    let (rw, rh) = (img.width() - tpl.width() + 1, img.height() - tpl.height() + 1);
    let (mean, sdv) = mean_std_dev(&tpl)?;
    let mut templ_norm: f64 = sdv.iter().map(|s| s * s).sum();
    if templ_norm < f64::EPSILON {
        return Ok(Mat::filled(rw, rh, 1, 1.0));
    }
    let inv_area = 1.0 / (tpl.width() * tpl.height()) as f64;
    templ_norm = templ_norm.sqrt() / inv_area.sqrt();

    let corr = cross_correlation(&img, &tpl, rw, rh);
    let (sum, sqsum) = integrals(&img);
    let (tw, th) = (tpl.width(), tpl.height());
    let stride = (img.width() + 1) * cn;
    let mut out = Mat::<f32>::zeros(rw, rh, 1);
    out.par_rows_mut().enumerate().for_each(|(i, row)| {
        for (j, o) in row.iter_mut().enumerate() {
            let mut num = corr[i * rw + j];
            let mut wnd_mean2 = 0.0;
            let mut wnd_sum2 = 0.0;
            let p0 = i * stride + j * cn;
            let p1 = p0 + tw * cn;
            let p2 = p0 + th * stride;
            let p3 = p2 + tw * cn;
            for k in 0..cn {
                let t = sum[p0 + k] - sum[p1 + k] - sum[p2 + k] + sum[p3 + k];
                wnd_mean2 += t * t;
                num -= t * mean[k];
            }
            wnd_mean2 *= inv_area;
            for k in 0..cn {
                wnd_sum2 += sqsum[p0 + k] - sqsum[p1 + k] - sqsum[p2 + k] + sqsum[p3 + k];
            }
            let diff2 = (wnd_sum2 - wnd_mean2).max(0.0);
            let t = if diff2 <= 0.5f64.min(10.0 * f64::from(f32::EPSILON) * wnd_sum2) {
                0.0
            } else {
                diff2.sqrt() * templ_norm
            };
            num = if num.abs() < t {
                num / t
            } else if num.abs() < t * 1.125 {
                if num > 0.0 { 1.0 } else { -1.0 }
            } else {
                0.0
            };
            *o = num as f32;
        }
    });
    Ok(out)
}

/// `cv::integral(img, sum, sqsum, CV_64F)` with `(W+1) x (H+1)` interleaved channels.
fn integrals<T: Element>(img: &MatView<'_, T>) -> (Vec<f64>, Vec<f64>) {
    let (w, h, cn) = (img.width(), img.height(), img.channels());
    let stride = (w + 1) * cn;
    let mut sum = vec![0.0; stride * (h + 1)];
    let mut sq = vec![0.0; stride * (h + 1)];
    for y in 0..h {
        let row = img.row(y);
        let mut s = vec![0.0f64; cn];
        let mut q = vec![0.0f64; cn];
        for x in 0..w {
            for c in 0..cn {
                let v = row[x * cn + c].to_f64();
                s[c] += v;
                q[c] += v * v;
                let at = (y + 1) * stride + (x + 1) * cn + c;
                sum[at] = sum[at - stride] + s[c];
                sq[at] = sq[at - stride] + q[c];
            }
        }
    }
    (sum, sq)
}

/// `sum_{i,j,c} img[y+i, x+j, c] * tpl[i, j, c]` in double precision (OpenCV's float32 DFT drifts
/// by up to ~4.5e-3 of the normalized score in near-flat windows; this stays at the exact value).
fn cross_correlation<T: Element>(img: &MatView<'_, T>, tpl: &MatView<'_, T>, rw: usize, rh: usize) -> Vec<f64> {
    let work = rw * rh * tpl.width() * tpl.height() * tpl.channels();
    if work <= DIRECT_LIMIT { direct_correlation(img, tpl, rw, rh) } else { fft_correlation(img, tpl, rw, rh) }
}

fn direct_correlation<T: Element>(img: &MatView<'_, T>, tpl: &MatView<'_, T>, rw: usize, rh: usize) -> Vec<f64> {
    let cn = img.channels();
    let tw = tpl.width();
    let taps: Vec<Vec<f64>> = (0..tpl.height()).map(|i| tpl.row(i).iter().map(|v| v.to_f64()).collect()).collect();
    let mut out = vec![0.0f64; rw * rh];
    out.par_chunks_mut(rw).enumerate().for_each(|(y, row)| {
        let mut acc = vec![0.0f64; rw];
        let mut line = vec![0.0f64; img.width() * cn];
        for (i, trow) in taps.iter().enumerate() {
            for (l, v) in line.iter_mut().zip(img.row(y + i)) {
                *l = v.to_f64();
            }
            for j in 0..tw {
                for c in 0..cn {
                    let k = trow[j * cn + c];
                    if k == 0.0 {
                        continue;
                    }
                    let off = j * cn + c;
                    for (x, a) in acc.iter_mut().enumerate() {
                        *a += k * line[off + x * cn];
                    }
                }
            }
        }
        row.copy_from_slice(&acc);
    });
    out
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

fn fft_correlation<T: Element>(img: &MatView<'_, T>, tpl: &MatView<'_, T>, rw: usize, rh: usize) -> Vec<f64> {
    let cn = img.channels();
    let (fw, fh) = (fft_size(img.width()), fft_size(img.height()));
    let mut planner = FftPlanner::<f64>::new();
    let (row_fwd, row_inv) = (planner.plan_fft_forward(fw), planner.plan_fft_inverse(fw));
    let (col_fwd, col_inv) = (planner.plan_fft_forward(fh), planner.plan_fft_inverse(fh));
    let transform = |src: &MatView<'_, T>, c: usize| -> Vec<Complex64> {
        let mut buf = vec![Complex64::new(0.0, 0.0); fw * fh];
        buf.par_chunks_mut(fw).enumerate().take(src.height()).for_each(|(y, row)| {
            for (x, v) in row.iter_mut().take(src.width()).enumerate() {
                *v = Complex64::new(src.at(x, y, c).to_f64(), 0.0);
            }
            row_fwd.process(row);
        });
        fft_columns(&mut buf, fw, fh, &*col_fwd);
        buf
    };
    let mut acc = vec![Complex64::new(0.0, 0.0); fw * fh];
    for c in 0..cn {
        let a = transform(img, c);
        let b = transform(tpl, c);
        acc.par_iter_mut().zip(a.par_iter().zip(b.par_iter())).for_each(|(o, (a, b))| *o += a * b.conj());
    }
    fft_columns(&mut acc, fw, fh, &*col_inv);
    acc.par_chunks_mut(fw).take(rh).for_each(|row| row_inv.process(row));
    let norm = 1.0 / (fw * fh) as f64;
    (0..rh).flat_map(|y| (0..rw).map(move |x| (y, x))).map(|(y, x)| acc[y * fw + x].re * norm).collect()
}

fn fft_columns(buf: &mut [Complex64], fw: usize, fh: usize, fft: &dyn rustfft::Fft<f64>) {
    let mut cols: Vec<Vec<Complex64>> = (0..fw).map(|x| (0..fh).map(|y| buf[y * fw + x]).collect()).collect();
    cols.par_iter_mut().for_each(|c| fft.process(c));
    for (x, col) in cols.iter().enumerate() {
        for (y, v) in col.iter().enumerate() {
            buf[y * fw + x] = *v;
        }
    }
}

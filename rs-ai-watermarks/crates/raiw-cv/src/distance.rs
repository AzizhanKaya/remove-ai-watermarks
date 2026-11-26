//! `cv::distanceTransform(src, DIST_L2, DIST_MASK_PRECISE)`: the Felzenszwalb exact Euclidean
//! transform of `trueDistTrans`, ported with its float arithmetic.

use rayon::prelude::*;

use crate::error::{Result, invalid};
use crate::mat::{Mat, MatView};

const PRECISE_DIST_MAX: usize = 1 << 16;

/// Euclidean distance of every pixel to the nearest zero pixel (`f32`).
pub fn distance_transform_l2<'a>(src: impl Into<MatView<'a, u8>>) -> Result<Mat<f32>> {
    let src = src.into();
    if src.channels() != 1 || src.is_empty() {
        return Err(invalid(
            "distance_transform",
            "non-empty single-channel input required",
        ));
    }
    let (n, m) = (src.width(), src.height());
    let sqr = |i: usize, limit: usize| -> u32 {
        if i >= limit || i >= PRECISE_DIST_MAX {
            u32::MAX
        } else {
            (i * i) as u32
        }
    };
    let mut columns = vec![0.0f32; n * m];
    let cols: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|x| {
            let mut d = vec![0usize; m];
            let mut dist = m - 1;
            for j in (0..m).rev() {
                dist = if src.at(x, j, 0) == 0 { 0 } else { dist + 1 };
                d[j] = dist;
            }
            let mut dist = m - 1;
            (0..m)
                .map(|j| {
                    dist = (dist + 1).min(d[j]);
                    sqr(dist, m) as f32
                })
                .collect()
        })
        .collect();
    for (x, col) in cols.iter().enumerate() {
        for (y, &v) in col.iter().enumerate() {
            columns[y * n + x] = v;
        }
    }
    let inv: Vec<f32> = (0..n)
        .map(|i| if i == 0 { 0.0 } else { (0.5 / i as f64) as f32 })
        .collect();
    let sqr_row: Vec<u32> = (0..n)
        .map(|i| {
            if i >= PRECISE_DIST_MAX {
                u32::MAX
            } else {
                (i * i) as u32
            }
        })
        .collect();
    columns
        .par_chunks_mut(n)
        .for_each(|d| row_pass(d, &sqr_row, &inv));
    Mat::from_vec(n, m, 1, columns)
}

fn row_pass(d: &mut [f32], sqr: &[u32], inv: &[f32]) {
    const INF: f32 = 1e15;
    let n = d.len();
    let mut f = vec![0.0f32; n];
    let mut z = vec![0.0f32; n + 1];
    let mut v = vec![0usize; n];
    z[0] = -INF;
    if n > 1 {
        z[1] = INF;
    }
    f[0] = d[0];
    let mut k = 0usize;
    for q in 1..n {
        let fq = d[q];
        f[q] = fq;
        loop {
            let p = v[k];
            let gap = if q < PRECISE_DIST_MAX {
                sqr[q].wrapping_sub(sqr[p]) as f32
            } else {
                (q + p) as f32 * (q - p) as f32
            };
            let s = (fq - d[p] + gap) * inv[q - p];
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                if k < n {
                    z[k + 1] = INF;
                }
                break;
            }
            if k == 0 {
                break;
            }
            k -= 1;
        }
    }
    let mut k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        d[q] = (sqr[q.abs_diff(p)] as f32 + f[p]).sqrt();
    }
}

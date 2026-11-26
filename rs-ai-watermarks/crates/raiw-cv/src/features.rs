//! `cv::cornerMinEigenVal` and `cv::goodFeaturesToTrack` (Shi-Tomasi corners).

use rayon::prelude::*;

use crate::border::Border;
use crate::element::{Depth, Element};
use crate::error::{CvError, Result, invalid};
use crate::filter::sobel;
use crate::mat::{Mat, MatView};
use crate::morph::{StructuringElement, dilate};
use crate::types::Point2f;

/// `cv::cornerMinEigenVal(src, block_size, ksize)` with the default reflect-101 border.
pub fn corner_min_eigen_val<'a, T: Element>(
    src: impl Into<MatView<'a, T>>,
    block_size: usize,
    ksize: usize,
) -> Result<Mat<f32>> {
    let src = src.into();
    if src.channels() != 1 || !matches!(T::DEPTH, Depth::U8 | Depth::F32) {
        return Err(invalid(
            "corner_min_eigen_val",
            "8-bit or float single-channel input required",
        ));
    }
    let mut scale = f64::from(1u32 << (ksize.max(1) - 1)) * block_size as f64;
    if T::DEPTH == Depth::U8 {
        scale *= 255.0;
    }
    let scale = 1.0 / scale;
    let border = Border::Reflect101;
    let dx: Mat<f32> = sobel(src, 1, 0, ksize, scale, 0.0, border)?;
    let dy: Mat<f32> = sobel(src, 0, 1, ksize, scale, 0.0, border)?;
    let (w, h) = (src.width(), src.height());
    let cov: Vec<[f32; 3]> = dx
        .data()
        .iter()
        .zip(dy.data())
        .map(|(&a, &b)| [a * a, a * b, b * b])
        .collect();
    let r = block_size / 2;
    let xs = crate::border::border_table(w, r, block_size - 1 - r, border);
    let ys = crate::border::border_table(h, r, block_size - 1 - r, border);
    let mut out = Mat::<f32>::zeros(w, h, 1);
    out.par_rows_mut().enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut s = [0.0f64; 3];
            for sy in ys[y..y + block_size].iter().flatten() {
                for sx in xs[x..x + block_size].iter().flatten() {
                    let c = cov[sy * w + sx];
                    for k in 0..3 {
                        s[k] += f64::from(c[k]);
                    }
                }
            }
            let a = s[0] as f32 * 0.5;
            let b = s[1] as f32;
            let c = s[2] as f32 * 0.5;
            *o = (a + c) - ((a - c) * (a - c) + b * b).sqrt();
        }
    });
    Ok(out)
}

/// `cv::goodFeaturesToTrack(image, max_corners, quality_level, min_distance, mask)`
/// with `blockSize = 3`, `gradientSize = 3` and the min-eigenvalue score.
pub fn good_features_to_track<'a, T: Element>(
    image: impl Into<MatView<'a, T>>,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
    mask: Option<MatView<'_, u8>>,
) -> Result<Vec<Point2f>> {
    let image = image.into();
    if let Some(m) = &mask
        && m.size() != image.size()
    {
        return Err(CvError::ShapeMismatch {
            op: "good_features_to_track",
        });
    }
    if image.is_empty() {
        return Ok(Vec::new());
    }
    let mut eig = corner_min_eigen_val(image, 3, 3)?;
    let allowed = |x: usize, y: usize| mask.as_ref().is_none_or(|m| m.at(x, y, 0) != 0);
    let mut max_val = f32::MIN;
    for y in 0..eig.height() {
        for (x, &v) in eig.row(y).iter().enumerate() {
            if allowed(x, y) && v > max_val {
                max_val = v;
            }
        }
    }
    let thresh = (f64::from(max_val) * quality_level) as f32;
    eig.data_mut().iter_mut().for_each(|v| {
        if *v <= thresh {
            *v = 0.0;
        }
    });
    let peaks = dilate(&eig, &StructuringElement::rect(3, 3), 1)?;
    let (w, h) = (eig.width(), eig.height());
    let mut candidates: Vec<(f32, usize)> = Vec::new();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let v = eig.at(x, y, 0);
            if v != 0.0 && v == peaks.at(x, y, 0) && allowed(x, y) {
                candidates.push((v, y * w + x));
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1))
    });
    let mut corners = Vec::new();
    if min_distance >= 1.0 {
        let cell = crate::element::cv_round(min_distance).max(1) as usize;
        let (gw, gh) = (w.div_ceil(cell), h.div_ceil(cell));
        let mut grid: Vec<Vec<Point2f>> = vec![Vec::new(); gw * gh];
        let min_d2 = min_distance * min_distance;
        for &(_, ofs) in &candidates {
            let (x, y) = (ofs % w, ofs / w);
            let (xc, yc) = (x / cell, y / cell);
            let good = (yc.saturating_sub(1)..=(yc + 1).min(gh - 1)).all(|yy| {
                (xc.saturating_sub(1)..=(xc + 1).min(gw - 1)).all(|xx| {
                    grid[yy * gw + xx].iter().all(|p| {
                        let (dx, dy) = (x as f32 - p.x, y as f32 - p.y);
                        f64::from(dx * dx + dy * dy) >= min_d2
                    })
                })
            });
            if good {
                let p = Point2f::new(x as f32, y as f32);
                grid[yc * gw + xc].push(p);
                corners.push(p);
                if max_corners > 0 && corners.len() == max_corners {
                    break;
                }
            }
        }
    } else {
        corners = candidates
            .iter()
            .map(|&(_, ofs)| Point2f::new((ofs % w) as f32, (ofs / w) as f32))
            .collect();
        if max_corners > 0 {
            corners.truncate(max_corners);
        }
    }
    Ok(corners)
}

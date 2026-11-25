//! Port of `_internal/image_alignment.py` (bounded displacement post-processing and error weights).
//!
//! The DIS optical flow estimate of `align_regenerated` is not ported; [`bound_flow`] reproduces
//! every step after `estimator.calc`, so a caller with a small-scale flow gets the same output.

use crate::border::Border;
use crate::distance::distance_transform_l2;
use crate::error::{CvError, Result, invalid};
use crate::filter::gaussian_blur;
use crate::mat::Mat;
use crate::resize::resize;
use crate::types::Interpolation;
use crate::warp::remap;

/// `np.gradient` of a single-channel plane: `(d/drow, d/dcol)`.
fn gradient(p: &Mat<f32>) -> Result<(Mat<f32>, Mat<f32>)> {
    let (w, h) = p.size();
    if w < 2 || h < 2 {
        return Err(invalid("gradient", "each axis needs at least two samples"));
    }
    let at = |x: usize, y: usize| p.at(x, y, 0);
    let dy = Mat::from_fn(w, h, 1, |x, y, _| match y {
        0 => at(x, 1) - at(x, 0),
        _ if y == h - 1 => at(x, h - 1) - at(x, h - 2),
        _ => (at(x, y + 1) - at(x, y - 1)) / 2.0,
    });
    let dx = Mat::from_fn(w, h, 1, |x, y, _| match x {
        0 => at(1, y) - at(0, y),
        _ if x == w - 1 => at(w - 1, y) - at(w - 2, y),
        _ => (at(x + 1, y) - at(x - 1, y)) / 2.0,
    });
    Ok((dy, dx))
}

fn plane(flow: &Mat<f32>, c: usize) -> Mat<f32> {
    Mat::from_fn(flow.width(), flow.height(), 1, |x, y, _| flow.at(x, y, c))
}

/// Scale a 2-channel flow so its Jacobian norm stays below 0.35.
fn bound_gradient(flow: &mut Mat<f32>) -> Result<()> {
    let (gy0, gx0) = gradient(&plane(flow, 0))?;
    let (gy1, gx1) = gradient(&plane(flow, 1))?;
    let mut max_norm = f32::MIN;
    for i in 0..gy0.data().len() {
        let s = ((gy0.data()[i] * gy0.data()[i] + gx0.data()[i] * gx0.data()[i]) + gy1.data()[i] * gy1.data()[i])
            + gx1.data()[i] * gx1.data()[i];
        max_norm = max_norm.max(s.sqrt());
    }
    let k = (0.35 / f64::from(max_norm).max(1e-6)).min(1.0) as f32;
    flow.data_mut().iter_mut().for_each(|v| *v *= k);
    Ok(())
}

fn displaced(donor: &Mat<u8>, flow: &Mat<f32>, interpolation: Interpolation) -> Result<Mat<u8>> {
    let (w, h) = donor.size();
    let mx = Mat::from_fn(w, h, 1, |x, y, _| x as f32 + flow.at(x, y, 0));
    let my = Mat::from_fn(w, h, 1, |x, y, _| y as f32 + flow.at(x, y, 1));
    remap(donor, &mx, &my, interpolation, Border::Reflect101)
}

/// Every step of `align_regenerated` after the DIS estimate: upsample, sigma-12 smoothing,
/// 16-pixel clamp, protected-box taper, Jacobian bound and the final remap.
pub fn bound_flow(
    donor: &Mat<u8>,
    small_flow: &Mat<f32>,
    protected_boxes: &[(usize, usize, usize, usize)],
    resampling: Interpolation,
) -> Result<(Mat<u8>, Mat<f32>)> {
    let (width, height) = donor.size();
    if small_flow.channels() != 2 {
        return Err(CvError::Channels { op: "bound_flow", channels: small_flow.channels() });
    }
    let (sw, sh) = small_flow.size();
    let mut flow = resize(small_flow, width, height, Interpolation::Linear)?;
    let (fx, fy) = ((width as f64 / sw as f64) as f32, (height as f64 / sh as f64) as f32);
    for px in flow.data_mut().chunks_exact_mut(2) {
        px[0] *= fx;
        px[1] *= fy;
    }
    let mut flow = gaussian_blur(&flow, (0, 0), 12.0, 0.0, Border::Reflect101)?;
    if flow.data().iter().any(|v| !v.is_finite()) {
        return Err(invalid("bound_flow", "Nonfinite alignment flow"));
    }
    for px in flow.data_mut().chunks_exact_mut(2) {
        let mag = (px[0] * px[0] + px[1] * px[1]).sqrt();
        let k = (16.0 / mag.max(1e-6)).min(1.0);
        px[0] *= k;
        px[1] *= k;
    }
    if !protected_boxes.is_empty() {
        let mut allowed = Mat::filled(width, height, 1, 1u8);
        for &(left, top, right, bottom) in protected_boxes {
            if !(left < right && right <= width && top < bottom && bottom <= height) {
                return Err(invalid("bound_flow", "Protected box outside image"));
            }
            allowed.roi_mut(crate::types::Rect::from_corners(left, top, right, bottom))?.fill(0);
        }
        let distance = distance_transform_l2(&allowed)?;
        for (px, &d) in flow.data_mut().chunks_exact_mut(2).zip(distance.data()) {
            let wgt = 0.5 - 0.5 * (std::f32::consts::PI * (d / 16.0).min(1.0)).cos();
            px[0] *= wgt;
            px[1] *= wgt;
        }
    }
    bound_gradient(&mut flow)?;
    Ok((displaced(donor, &flow, resampling)?, flow))
}

/// `attenuate_regenerated(donor, flow, weights)`: weighted displacement, one Lanczos-4 resample.
pub fn attenuate_regenerated(donor: &Mat<u8>, flow: &Mat<f32>, weights: &Mat<f32>) -> Result<(Mat<u8>, Mat<f32>)> {
    let (w, h) = donor.size();
    if donor.channels() != 3 || w.min(h) < 64 {
        return Err(invalid("attenuate_regenerated", "Expected uint8 RGB donor with sides at least 64 pixels"));
    }
    if flow.size() != (w, h) || flow.channels() != 2 || weights.size() != (w, h) || weights.channels() != 1 {
        return Err(invalid("attenuate_regenerated", "Flow and weight dimensions must match donor"));
    }
    if flow.data().iter().chain(weights.data()).any(|v| !v.is_finite()) {
        return Err(invalid("attenuate_regenerated", "Nonfinite displacement or weights"));
    }
    if weights.data().iter().any(|&v| !(0.0..=1.0).contains(&v)) {
        return Err(invalid("attenuate_regenerated", "Weights must be in [0, 1]"));
    }
    if flow.data().chunks_exact(2).any(|p| f64::from(p[0].hypot(p[1])) > 16.0001) {
        return Err(invalid("attenuate_regenerated", "Input displacement exceeds alignment bound"));
    }
    let mut weighted = flow.clone();
    for (px, &k) in weighted.data_mut().chunks_exact_mut(2).zip(weights.data()) {
        px[0] *= k;
        px[1] *= k;
    }
    bound_gradient(&mut weighted)?;
    Ok((displaced(donor, &weighted, Interpolation::Lanczos4)?, weighted))
}

/// `local_error_weights(reference, baseline, candidate, sigma)` on RGB `u8` images.
pub fn local_error_weights(reference: &Mat<u8>, baseline: &Mat<u8>, candidate: &Mat<u8>, sigma: f64) -> Result<Mat<f32>> {
    if reference.channels() != 3 || reference.is_empty() {
        return Err(invalid("local_error_weights", "Inputs must have nonempty RGB dimensions"));
    }
    if baseline.size() != reference.size() || candidate.size() != reference.size() || baseline.channels() != 3 || candidate.channels() != 3 {
        return Err(invalid("local_error_weights", "Input dimensions must match"));
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(invalid("local_error_weights", "Sigma must be positive and finite"));
    }
    let (width, height) = reference.size();
    let ratio = (256.0 / width.max(height) as f64).min(1.0);
    let size = (
        ((width as f64 * ratio).round_ties_even() as usize).max(1),
        ((height as f64 * ratio).round_ties_even() as usize).max(1),
    );
    let mut num = Mat::<f32>::zeros(width, height, 1);
    let mut den = Mat::<f32>::zeros(width, height, 1);
    for i in 0..width * height {
        let (mut n, mut d) = (0.0f32, 0.0f32);
        let mut terms_n = [0.0f32; 3];
        let mut terms_d = [0.0f32; 3];
        for c in 0..3 {
            let b = f32::from(baseline.data()[i * 3 + c]);
            let delta = f32::from(candidate.data()[i * 3 + c]) - b;
            let residual = f32::from(reference.data()[i * 3 + c]) - b;
            terms_n[c] = delta * residual;
            terms_d[c] = delta * delta;
        }
        n += (terms_n[0] + terms_n[1]) + terms_n[2];
        d += (terms_d[0] + terms_d[1]) + terms_d[2];
        num.data_mut()[i] = n / 3.0;
        den.data_mut()[i] = d / 3.0;
    }
    let num = resize(&num, size.0, size.1, Interpolation::Area)?;
    let den = resize(&den, size.0, size.1, Interpolation::Area)?;
    let (sx, sy) = (sigma * size.0 as f64 / width as f64, sigma * size.1 as f64 / height as f64);
    let smooth = |m: &Mat<f32>| gaussian_blur(m, (0, 0), sx, sy, Border::Reflect101);
    let sn = smooth(&num)?;
    let sd = smooth(&den)?;
    let weight = Mat::from_vec(
        sn.width(),
        sn.height(),
        1,
        sn.data().iter().zip(sd.data()).map(|(&a, &b)| (a / b.max(1.0)).clamp(0.0, 1.0)).collect(),
    )?;
    let out = resize(&smooth(&weight)?, width, height, Interpolation::Linear)?;
    Ok(out.map(|v: f32| v.clamp(0.0, 1.0)))
}

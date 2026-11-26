//! Self-calibrated YCrCb alpha mattes for stationary translucent marks
//! (`ClipAlphaMatte`, `estimate_clip_alpha_matte`, `unblend_ycbcr_roi`).

use raiw_cv::{
    Border, ColorConversion, Interpolation, LkParams, Mat, Point2f, Scalar, cvt_color, gaussian_blur,
    good_features_to_track, remap,
};

use super::affine::estimate_affine_partial_2d;
use super::pixel::{BoolPlane, VideoPixel, crop_mat, dilate_mask, ellipse, paste};
use crate::error::Result;

/// `ClipAlphaMatte`: per-pixel alpha and composite color over a fixed box.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipAlphaMatte {
    /// `(x0, y0, x1, y1)`.
    pub bounds: (usize, usize, usize, usize),
    pub alpha_y: Mat<f32>,
    pub alpha_c: Mat<f32>,
    pub c_y: Mat<f32>,
    pub c_cr: Mat<f32>,
    pub c_cb: Mat<f32>,
    pub calibrated: BoolPlane,
}

type Mat3 = [[f64; 3]; 3];

fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn inverse(m: &Mat3) -> Option<Mat3> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    Some([
        [c(1, 1, 2, 2) / det, -c(0, 1, 2, 2) / det, c(0, 1, 1, 2) / det],
        [-c(1, 0, 2, 2) / det, c(0, 0, 2, 2) / det, -c(0, 0, 1, 2) / det],
        [c(1, 0, 2, 1) / det, -c(0, 0, 2, 1) / det, c(0, 0, 1, 1) / det],
    ])
}

fn nan_median(values: &mut Vec<f32>) -> f32 {
    values.retain(|v| v.is_finite());
    crate::py::median_f32(values)
}

fn blur(alpha: &Mat<f32>) -> Result<Mat<f32>> {
    Ok(gaussian_blur(alpha, (3, 3), 0.7, 0.0, Border::default())?)
}

struct Warp {
    donor: usize,
    map_x: Mat<f32>,
    map_y: Mat<f32>,
    in_bounds: BoolPlane,
    context: BoolPlane,
}

fn ycrcb_crop<T: VideoPixel>(
    frame: &Mat<T>,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> Result<(Mat<f32>, Mat<u8>)> {
    let crop = crop_mat(frame, x0, y0, x1, y1)?;
    let ycrcb = cvt_color(&crop, ColorConversion::BgrToYCrCb)?;
    let scale = if T::PEAK > 255.0 { 255.0 / 65535.0 } else { 1.0 };
    let data: Vec<f32> = ycrcb.data().iter().map(|v| v.to_f32() * scale as f32).collect();
    let float = Mat::from_vec(crop.width(), crop.height(), 3, data)?;
    let gray: Vec<u8> = float
        .data()
        .chunks_exact(3)
        .map(|px| {
            if T::PEAK > 255.0 {
                px[0].round_ties_even().clamp(0.0, 255.0) as u8
            } else {
                px[0] as u8
            }
        })
        .collect();
    Ok((float, Mat::from_vec(crop.width(), crop.height(), 1, gray)?))
}

fn step_matrix(previous: &Mat<u8>, next: &Mat<u8>, feature_mask: &Mat<u8>) -> Result<Option<Mat3>> {
    let points = good_features_to_track(previous, 200, 0.01, 4.0, Some(feature_mask.view()))?;
    if points.len() < 8 {
        return Ok(None);
    }
    let params = LkParams {
        win_width: 15,
        win_height: 15,
        max_level: 3,
        ..LkParams::default()
    };
    let tracks = raiw_cv::calc_optical_flow_pyr_lk(previous, next, &points, params)?;
    let (from, to): (Vec<Point2f>, Vec<Point2f>) = points
        .iter()
        .zip(&tracks)
        .filter(|(_, t)| t.found)
        .map(|(p, t)| (*p, t.point))
        .unzip();
    if from.len() < 8 {
        return Ok(None);
    }
    let Some((affine, mask)) = estimate_affine_partial_2d(&from, &to, 1.5) else {
        return Ok(None);
    };
    if mask.iter().filter(|m| **m).count() < 6 {
        return Ok(None);
    }
    Ok(Some([affine[0], affine[1], [0.0, 0.0, 1.0]]))
}

fn chain(steps: &[Option<Mat3>], k_src: usize, k_dst: usize) -> Option<Mat3> {
    let mut m: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if k_dst > k_src {
        for step in &steps[k_src..k_dst] {
            m = mul(&(*step)?, &m);
        }
    } else {
        for j in (k_dst..k_src).rev() {
            m = mul(&inverse(&steps[j]?)?, &m);
        }
    }
    Some(m)
}

/// `estimate_clip_alpha_matte`: fit `(alpha, alpha*W)` per pixel across moving frames.
pub fn estimate_clip_alpha_matte<T: VideoPixel>(
    frames: &[Mat<T>],
    bounds: (i64, i64, i64, i64),
) -> Result<Option<ClipAlphaMatte>> {
    let n = frames.len();
    if n < 12 {
        return Ok(None);
    }
    let (width, height) = (frames[0].width() as i64, frames[0].height() as i64);
    let (x0, y0) = (bounds.0.max(0), bounds.1.max(0));
    let (x1, y1) = (bounds.2.min(width), bounds.3.min(height));
    let (rh, rw) = (y1 - y0, x1 - x0);
    if rh < 6 || rw < 6 {
        return Ok(None);
    }
    let pad = rh.max(rw).clamp(36, 72);
    let (cx0, cy0, cx1, cy1) = (
        (x0 - pad).max(0),
        (y0 - pad).max(0),
        width.min(x1 + pad),
        height.min(y1 + pad),
    );
    let (ch, cw) = ((cy1 - cy0) as usize, (cx1 - cx0) as usize);
    let (hx0, hy0, hx1, hy1) = (
        (x0 - cx0) as usize,
        (y0 - cy0) as usize,
        (x1 - cx0) as usize,
        (y1 - cy0) as usize,
    );
    let (rh, rw) = (rh as usize, rw as usize);
    let mut feature_mask = Mat::<u8>::filled(cw, ch, 1, 255);
    for y in hy0.saturating_sub(2)..ch.min(hy1 + 2) {
        for x in hx0.saturating_sub(2)..cw.min(hx1 + 2) {
            feature_mask.set(x, y, 0, 0);
        }
    }
    let context_bool = BoolPlane::from_mask(&feature_mask);
    if context_bool.count() < 256 {
        return Ok(None);
    }
    let mut crops = Vec::with_capacity(n);
    let mut grays = Vec::with_capacity(n);
    for frame in frames {
        let (float, gray) = ycrcb_crop(frame, cx0 as usize, cy0 as usize, cx1 as usize, cy1 as usize)?;
        crops.push(float);
        grays.push(gray);
    }
    let mut steps = Vec::with_capacity(n - 1);
    for k in 0..n - 1 {
        steps.push(step_matrix(&grays[k], &grays[k + 1], &feature_mask)?);
    }
    let mut warps: Vec<Vec<Warp>> = (0..n).map(|_| Vec::new()).collect();
    for (k, slot) in warps.iter_mut().enumerate() {
        for delta in [-12i64, -9, -6, -4, -2, 2, 4, 6, 9, 12] {
            let kd = k as i64 + delta;
            if kd < 0 || kd >= n as i64 {
                continue;
            }
            let Some(m) = chain(&steps, k, kd as usize) else {
                continue;
            };
            if m[0][2].hypot(m[1][2]) < 4.0 {
                continue;
            }
            let mut mx = Vec::with_capacity(cw * ch);
            let mut my = Vec::with_capacity(cw * ch);
            for y in 0..ch {
                for x in 0..cw {
                    let (gx, gy) = (f64::from(x as f32), f64::from(y as f32));
                    mx.push((m[0][0] * gx + m[0][1] * gy + m[0][2]) as f32);
                    my.push((m[1][0] * gx + m[1][1] * gy + m[1][2]) as f32);
                }
            }
            let in_bounds = BoolPlane {
                width: cw,
                height: ch,
                data: mx
                    .iter()
                    .zip(&my)
                    .map(|(x, y)| *x >= 1.0 && *x <= cw as f32 - 2.0 && *y >= 1.0 && *y <= ch as f32 - 2.0)
                    .collect(),
            };
            let context = context_bool.zip_with(&in_bounds, |a, b| a && b);
            if context.count() < 128 {
                continue;
            }
            slot.push(Warp {
                donor: kd as usize,
                map_x: Mat::from_vec(cw, ch, 1, mx)?,
                map_y: Mat::from_vec(cw, ch, 1, my)?,
                in_bounds,
                context,
            });
        }
    }

    let box_len = rh * rw;
    let mut alpha_hat = vec![0.0f32; box_len];
    let mut c_y = vec![0.0f32; box_len];
    let mut c_cr = vec![0.0f32; box_len];
    let mut c_cb = vec![0.0f32; box_len];
    let mut calibrated = vec![false; box_len];
    let mut support = BoolPlane::new(cw, ch);
    for y in hy0..hy1 {
        for x in hx0..hx1 {
            support.data[y * cw + x] = true;
        }
    }
    let box_index = |bi: usize| (hy0 + bi / rw) * cw + hx0 + bi % rw;
    for pass in 0..3 {
        let donor_pool: Vec<Mat<f32>> = if pass > 0 {
            let alpha_mat = Mat::from_vec(rw, rh, 1, alpha_hat.clone())?;
            let ac_d = blur(&alpha_mat)?;
            crops
                .iter()
                .map(|base| {
                    let mut unblended = base.clone();
                    let data = unblended.data_mut();
                    for bi in 0..box_len {
                        let (a, ad) = (alpha_hat[bi], ac_d.data()[bi]);
                        let i = box_index(bi) * 3;
                        if (0.02..=0.78).contains(&a) {
                            data[i] = (data[i] - c_y[bi]) / (1.0 - a).max(0.20);
                        }
                        if (0.02..=0.78).contains(&ad) {
                            let dc = (1.0 - ad).max(0.20);
                            data[i + 1] = (data[i + 1] - c_cr[bi]) / dc;
                            data[i + 2] = (data[i + 2] - c_cb[bi]) / dc;
                        }
                    }
                    unblended
                })
                .collect()
        } else {
            crops.clone()
        };
        let support_mask = support.to_mask(1);
        let mut observed: Vec<Vec<f32>> = Vec::new();
        let mut reference: Vec<Vec<f32>> = Vec::new();
        for (k, frame_warps) in warps.iter().enumerate() {
            let mut stack: Vec<(Vec<f32>, Vec<bool>)> = Vec::new();
            for warp in frame_warps {
                let warped = remap(
                    &donor_pool[warp.donor],
                    &warp.map_x,
                    &warp.map_y,
                    Interpolation::Linear,
                    Border::Reflect,
                )?;
                let (cur, wd) = (crops[k].data(), warped.data());
                let (mut total, mut count) = (0.0f64, 0usize);
                for (i, set) in warp.context.data.iter().enumerate() {
                    if *set {
                        total += f64::from((cur[i * 3] - wd[i * 3]).abs());
                        count += 1;
                    }
                }
                if total / count.max(1) as f64 > 9.0 {
                    continue;
                }
                let warped_support = remap(
                    &support_mask,
                    &warp.map_x,
                    &warp.map_y,
                    Interpolation::Nearest,
                    Border::Constant(Scalar::all(1.0)),
                )?;
                let mut values = Vec::with_capacity(box_len * 3);
                let mut valid = Vec::with_capacity(box_len);
                for bi in 0..box_len {
                    let i = box_index(bi);
                    values.extend_from_slice(&wd[i * 3..i * 3 + 3]);
                    valid.push(warp.in_bounds.data[i] && warped_support.data()[i] == 0);
                }
                stack.push((values, valid));
            }
            if stack.is_empty() {
                continue;
            }
            let mut obs = Vec::with_capacity(box_len * 3);
            let mut refs = Vec::with_capacity(box_len * 3);
            let cur = crops[k].data();
            for bi in 0..box_len {
                let i = box_index(bi);
                obs.extend_from_slice(&cur[i * 3..i * 3 + 3]);
                for c in 0..3 {
                    let mut column: Vec<f32> = stack
                        .iter()
                        .map(|(v, ok)| if ok[bi] { v[bi * 3 + c] } else { f32::NAN })
                        .collect();
                    refs.push(nan_median(&mut column));
                }
            }
            observed.push(obs);
            reference.push(refs);
        }
        if observed.len() < 8 {
            return Ok(None);
        }
        let mut calibrated_count = 0usize;
        let mut slope = vec![1.0f32; box_len];
        let mut a_raw = vec![0.0f32; box_len];
        for bi in 0..box_len {
            let samples: Vec<(f32, f32)> = observed
                .iter()
                .zip(&reference)
                .filter(|(_, r)| r[bi * 3].is_finite())
                .map(|(o, r)| (o[bi * 3], r[bi * 3]))
                .collect();
            let cnt = samples.len();
            let denom = cnt.max(1) as f32;
            let b_mean = samples.iter().map(|s| s.1).sum::<f32>() / denom;
            let i_mean = samples.iter().map(|s| s.0).sum::<f32>() / denom;
            let var_b = samples.iter().map(|s| (s.1 - b_mean).powi(2)).sum::<f32>() / denom;
            let cov = samples
                .iter()
                .map(|s| (s.1 - b_mean) * (s.0 - i_mean))
                .sum::<f32>()
                / denom;
            calibrated[bi] = cnt >= 6 && var_b >= 25.0;
            if calibrated[bi] {
                calibrated_count += 1;
                slope[bi] = (cov / var_b.max(1e-3)).clamp(0.0, 1.0);
            }
            a_raw[bi] = 1.0 - slope[bi];
        }
        if (calibrated_count as f64 / box_len as f64) < 0.35 {
            return Ok(None);
        }
        for bi in 0..box_len {
            let mut residual: Vec<f32> = Vec::new();
            let mut diffs: Vec<f32> = Vec::new();
            for (o, r) in observed.iter().zip(&reference) {
                if r[bi * 3].is_finite() {
                    residual.push(o[bi * 3] - slope[bi] * r[bi * 3]);
                    diffs.push((o[bi * 3] - r[bi * 3]).abs());
                }
            }
            let cy_raw = nan_median(&mut residual).clamp(0.0, 255.0);
            let cy_raw = if cy_raw.is_nan() { 0.0 } else { cy_raw };
            let diff_med = nan_median(&mut diffs);
            let diff_med = if diff_med.is_nan() { 0.0 } else { diff_med };
            let active = calibrated[bi] && a_raw[bi] >= 0.02 && diff_med >= 1.5;
            alpha_hat[bi] = if active { a_raw[bi] } else { 0.0 };
            c_y[bi] = if active { cy_raw } else { 0.0 };
        }
        let ac = blur(&Mat::from_vec(rw, rh, 1, alpha_hat.clone())?)?;
        for bi in 0..box_len {
            let a = ac.data()[bi];
            for (channel, out) in [(1, &mut c_cr), (2, &mut c_cb)] {
                let mut values: Vec<f32> = observed
                    .iter()
                    .zip(&reference)
                    .filter(|(_, r)| r[bi * 3].is_finite())
                    .map(|(o, r)| o[bi * 3 + channel] - (1.0 - a) * r[bi * 3 + channel])
                    .collect();
                let m = nan_median(&mut values);
                out[bi] = if a >= 0.02 {
                    if m.is_nan() { 0.0 } else { m }
                } else {
                    0.0
                };
            }
        }
        for bi in 0..box_len {
            support.data[box_index(bi)] = !calibrated[bi]
                || if pass == 1 {
                    alpha_hat[bi] > 0.78
                } else {
                    alpha_hat[bi] >= 0.04
                };
        }
    }
    if !alpha_hat.iter().any(|a| *a >= 0.02) {
        return Ok(None);
    }
    let alpha_y = Mat::from_vec(rw, rh, 1, alpha_hat)?;
    let alpha_c = blur(&alpha_y)?;
    Ok(Some(ClipAlphaMatte {
        bounds: (x0 as usize, y0 as usize, x1 as usize, y1 as usize),
        alpha_y,
        alpha_c,
        c_y: Mat::from_vec(rw, rh, 1, c_y)?,
        c_cr: Mat::from_vec(rw, rh, 1, c_cr)?,
        c_cb: Mat::from_vec(rw, rh, 1, c_cb)?,
        calibrated: BoolPlane {
            width: rw,
            height: rh,
            data: calibrated,
        },
    }))
}

/// `unblend_ycbcr_roi`: invert translucent pixels and shrink the mask to the opaque core.
pub fn unblend_ycbcr_roi<T: VideoPixel>(
    frame: &Mat<T>,
    mask: &Mat<u8>,
    matte: &ClipAlphaMatte,
) -> Result<(Mat<T>, Mat<u8>)> {
    const OPAQUE: f32 = 0.72;
    let (x0, y0, x1, y1) = matte.bounds;
    let mut out_frame = frame.clone();
    let mut out_mask = mask.clone();
    let roi_mask = BoolPlane::from_mask(&crop_mat(mask, x0, y0, x1, y1)?);
    if !roi_mask.any() {
        return Ok((out_frame, out_mask));
    }
    let c_scale = if T::PEAK > 255.0 { 65535.0 / 255.0 } else { 1.0 };
    let roi = crop_mat(frame, x0, y0, x1, y1)?;
    let ycrcb = cvt_color(&roi, ColorConversion::BgrToYCrCb)?;
    let mut values: Vec<f32> = ycrcb.data().iter().map(|v| v.to_f32()).collect();
    let floor = 0.15f32.max(1.0 - OPAQUE);
    for i in 0..roi_mask.data.len() {
        let calibrated = matte.calibrated.data[i];
        let (ay, ac) = (matte.alpha_y.data()[i], matte.alpha_c.data()[i]);
        if calibrated && (0.02..=OPAQUE).contains(&ay) {
            values[i * 3] = (values[i * 3] - matte.c_y.data()[i] * c_scale) / (1.0 - ay).max(floor);
        }
        if calibrated && (0.02..=OPAQUE).contains(&ac) {
            let denom = (1.0 - ac).max(floor);
            values[i * 3 + 1] = (values[i * 3 + 1] - matte.c_cr.data()[i] * c_scale) / denom;
            values[i * 3 + 2] = (values[i * 3 + 2] - matte.c_cb.data()[i] * c_scale) / denom;
        }
    }
    let quantized: Vec<T> = values
        .iter()
        .map(|v| T::from_f64(f64::from(v.round_ties_even()).clamp(0.0, T::PEAK)))
        .collect();
    let back = cvt_color(
        &Mat::from_vec(roi.width(), roi.height(), 3, quantized)?,
        ColorConversion::YCrCbToBgr,
    )?;
    paste(&mut out_frame, x0, y0, &back)?;
    let mut remaining = BoolPlane {
        width: roi_mask.width,
        height: roi_mask.height,
        data: (0..roi_mask.data.len())
            .map(|i| roi_mask.data[i] && !(matte.calibrated.data[i] && matte.alpha_y.data()[i] <= OPAQUE))
            .collect(),
    };
    if remaining.any() {
        let grown = BoolPlane::from_mask(&dilate_mask(&remaining.to_mask(255), &ellipse(3))?);
        remaining = grown.zip_with(&roi_mask, |g, r| g && r);
    }
    paste(&mut out_mask, x0, y0, &remaining.to_mask(255))?;
    Ok((out_frame, out_mask))
}

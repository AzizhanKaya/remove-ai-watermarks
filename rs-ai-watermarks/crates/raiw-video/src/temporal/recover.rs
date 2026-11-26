//! Donor-frame background recovery before single-frame inpainting (`recover_unobscured_background`).

use raiw_cv::{Affine, Border, InpaintMethod, Interpolation, Mat, inpaint, phase_correlate, warp_affine};

use super::flow::{backward_warp, displacement_to_maps, flow_field};
use super::motion::{forward_backward_error, maps_in_bounds, propagate_context_motion};
use super::pixel::{
    BoolPlane, LINEAR, NEAREST, VideoPixel, crop_mat, depth_bounds, dilate_mask, ellipse, paste,
};
use crate::error::{Result, VideoError};
use crate::py::round_int;

/// A donor frame and its own mask (`None` when the donor carries no mark).
pub type Donor<'a, T> = (&'a Mat<T>, Option<&'a Mat<u8>>);

/// Map x, map y, in-bounds, donor-masked and context planes of one projected flow.
type Projection = (Mat<f32>, Mat<f32>, BoolPlane, BoolPlane, BoolPlane);

fn mean_abs(a: &Mat<u8>, b: &Mat<u8>, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            total += f64::from((f32::from(a.at(x, y, 0)) - f32::from(b.at(x, y, 0))).abs());
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        total / count as f64
    }
}

/// `_coarse_shifted_flow`: dense flow with FFT coarse pre-alignment for pans over 4 px.
pub fn coarse_shifted_flow(current: &Mat<u8>, donor: &Mat<u8>) -> Result<(Mat<f32>, Vec<f32>)> {
    let (w, h) = current.size();
    let current_f: Mat<f32> = current.convert();
    let donor_f: Mat<f32> = donor.convert();
    let (shift, response) = phase_correlate(&current_f, &donor_f)?;
    let (cdx, cdy) = (round_int(shift.x), round_int(shift.y));
    let (wi, hi) = (w as i64, h as i64);
    if response >= 0.15 && cdx.abs().max(cdy.abs()) >= 4 && cdx.abs() < wi / 2 && cdy.abs() < hi / 2 {
        let matrix: Affine = [[1.0, 0.0, -(cdx as f64)], [0.0, 1.0, -(cdy as f64)]];
        let aligned = warp_affine(donor, &matrix, w, h, Interpolation::Linear, Border::Reflect)?;
        let (x_lo, x_hi) = ((-cdx).max(0) as usize, wi.min(wi - cdx).max(0) as usize);
        let (y_lo, y_hi) = ((-cdy).max(0) as usize, hi.min(hi - cdy).max(0) as usize);
        let raw_mae = mean_abs(current, donor, x_lo, y_lo, x_hi, y_hi);
        let aligned_mae = mean_abs(current, &aligned, x_lo, y_lo, x_hi, y_hi);
        if aligned_mae < raw_mae * 0.65 {
            let residual_to = flow_field(current, &aligned)?;
            let residual_from = flow_field(&aligned, current)?;
            let fb = forward_backward_error(&residual_to, &residual_from)?;
            let mut flow_to = residual_to;
            for pair in flow_to.data_mut().chunks_exact_mut(2) {
                pair[0] += cdx as f32;
                pair[1] += cdy as f32;
            }
            return Ok((flow_to, fb));
        }
    }
    let flow_to = flow_field(current, donor)?;
    let flow_from = flow_field(donor, current)?;
    let fb = forward_backward_error(&flow_to, &flow_from)?;
    Ok((flow_to, fb))
}

/// Options for [`recover_unobscured_background`].
#[derive(Clone, Copy, Debug)]
pub struct RecoverOptions {
    pub max_context_residual: f64,
    pub fb_tolerance: f32,
}

impl Default for RecoverOptions {
    fn default() -> Self {
        Self {
            max_context_residual: 12.0,
            fb_tolerance: 1.75,
        }
    }
}

/// `recover_unobscured_background`: warp genuine background from motion-aligned donors into the
/// hole and return the updated frame plus the remaining mask.
pub fn recover_unobscured_background<T: VideoPixel>(
    current_source: &Mat<T>,
    current_mask: &Mat<u8>,
    donors: &[Donor<'_, T>],
    options: RecoverOptions,
) -> Result<(Mat<T>, Mat<u8>)> {
    if options.max_context_residual <= 0.0 || options.fb_tolerance <= 0.0 {
        return Err(VideoError::value(
            "Context residual and flow consistency thresholds must be positive",
        ));
    }
    if current_mask.size() != current_source.size() {
        return Err(VideoError::value(
            "Current source and mask must share spatial geometry",
        ));
    }
    let mut remaining = current_mask.clone();
    if donors.is_empty() {
        return Ok((current_source.clone(), remaining));
    }
    let Some((xmin, ymin, xmax, ymax)) = BoolPlane::from_mask(current_mask).bounds() else {
        return Ok((current_source.clone(), remaining));
    };
    let (width, height) = current_mask.size();
    let (mask_w, mask_h) = (xmax - xmin + 1, ymax - ymin + 1);
    let padding = round_int(mask_w.max(mask_h) as f64).max(40) as usize;
    let (x0, y0) = (xmin.saturating_sub(padding), ymin.saturating_sub(padding));
    let (x1, y1) = (width.min(xmax + padding + 1), height.min(ymax + padding + 1));
    let (crop_w, crop_h) = (x1 - x0, y1 - y0);
    let (eff_residual, peak) = depth_bounds::<T>(options.max_context_residual);
    let source_crop = crop_mat(current_source, x0, y0, x1, y1)?;
    let source_f: Vec<f32> = source_crop.data().iter().map(|v| v.to_f32()).collect();
    let mask_crop = crop_mat(current_mask, x0, y0, x1, y1)?;
    let hole = BoolPlane::from_mask(&mask_crop);
    let guard_kernel = ellipse(7);
    let guard = dilate_mask(&hole.to_mask(255), &guard_kernel)?;
    let guard_plane = BoolPlane::from_mask(&guard);
    let gray_clean = inpaint(&T::gray_u8(&source_crop)?, &guard, 3.0, InpaintMethod::Telea)?;
    let ring = (round_int(mask_w.max(mask_h) as f64 * 0.30).max(9) | 1) as usize;
    let outer = BoolPlane::from_mask(&dilate_mask(&guard_plane.to_mask(1), &ellipse(ring))?);
    let channels = current_source.channels();
    let pixels = crop_w * crop_h;
    let mut accum = vec![0.0f32; pixels * channels];
    let mut weights = vec![0.0f32; pixels];

    for (rank, (donor_source, donor_mask)) in donors.iter().enumerate() {
        if donor_source.size() != current_source.size() {
            return Err(VideoError::value(
                "Donor frame must match current source geometry",
            ));
        }
        if donor_mask.is_some_and(|m| m.size() != current_mask.size()) {
            return Err(VideoError::value("Donor mask must match current mask geometry"));
        }
        let donor_crop = crop_mat(donor_source, x0, y0, x1, y1)?;
        let mut donor_gray = T::gray_u8(&donor_crop)?;
        let mut donor_guard = None;
        if let Some(mask) = donor_mask {
            let mask_crop = crop_mat(mask, x0, y0, x1, y1)?;
            let plane = BoolPlane::from_mask(&mask_crop);
            if plane.any() {
                let g = dilate_mask(&plane.to_mask(255), &guard_kernel)?;
                donor_gray = inpaint(&donor_gray, &g, 3.0, InpaintMethod::Telea)?;
                donor_guard = Some(g);
            }
        }
        let project = |flow: &Mat<f32>| -> Result<Projection> {
            let maps = displacement_to_maps(flow);
            let bounds = maps_in_bounds(&maps);
            let donor_masked = match &donor_guard {
                Some(g) => BoolPlane::from_mask(&backward_warp(g, &maps, NEAREST)?),
                None => BoolPlane::new(crop_w, crop_h),
            };
            let context = BoolPlane {
                width: crop_w,
                height: crop_h,
                data: (0..pixels)
                    .map(|i| {
                        outer.data[i] && !(guard_plane.data[i] || donor_masked.data[i]) && bounds.data[i]
                    })
                    .collect(),
            };
            Ok((maps.0, maps.1, bounds, donor_masked, context))
        };
        let (flow_to, fb) = coarse_shifted_flow(&gray_clean, &donor_gray)?;
        let (_, _, _, _, raw_context) = project(&flow_to)?;
        let consistent = BoolPlane {
            width: crop_w,
            height: crop_h,
            data: (0..pixels)
                .map(|i| raw_context.data[i] && fb[i] <= options.fb_tolerance)
                .collect(),
        };
        if consistent.count() < 48 {
            continue;
        }
        let keep = BoolPlane {
            width: crop_w,
            height: crop_h,
            data: (0..pixels)
                .map(|i| !guard_plane.data[i] && fb[i] <= options.fb_tolerance)
                .collect(),
        };
        let Some(propagated) = propagate_context_motion(&flow_to, &consistent, &keep, 32) else {
            continue;
        };
        let (map_x, map_y, in_bounds, donor_masked, context) = project(&propagated)?;
        let eligible: Vec<bool> = (0..pixels)
            .map(|i| hole.data[i] && !donor_masked.data[i] && in_bounds.data[i])
            .collect();
        let context_count = context.count();
        if !eligible.iter().any(|e| *e) || context_count < 48 {
            continue;
        }
        let donor_f: Mat<f32> = donor_crop.convert();
        let warped = backward_warp(&donor_f, &(map_x, map_y), LINEAR)?;
        let wd = warped.data();
        let mut total = 0.0;
        for (index, set) in context.data.iter().enumerate() {
            if *set {
                let mut channel_sum = 0.0f32;
                for c in 0..channels {
                    let i = index * channels + c;
                    channel_sum += (source_f[i] - wd[i]).abs();
                }
                total += f64::from(channel_sum / channels as f32);
            }
        }
        let residual = total / context_count as f64;
        if residual > eff_residual {
            continue;
        }
        let weight = ((1.0 - residual / eff_residual) / (1.0 + 0.25 * rank as f64)).max(0.05) as f32;
        for (index, e) in eligible.iter().enumerate() {
            if *e {
                for c in 0..channels {
                    accum[index * channels + c] += weight * wd[index * channels + c];
                }
                weights[index] += weight;
            }
        }
        if hole.data.iter().zip(&weights).all(|(h, w)| !*h || *w >= 1.1) {
            break;
        }
    }
    let recovered: Vec<bool> = hole
        .data
        .iter()
        .zip(&weights)
        .map(|(h, w)| *h && *w > 0.0)
        .collect();
    if !recovered.iter().any(|r| *r) {
        return Ok((current_source.clone(), remaining));
    }
    let mut result_crop = source_crop.clone();
    let out = result_crop.data_mut();
    for (index, r) in recovered.iter().enumerate() {
        if *r {
            for c in 0..channels {
                let i = index * channels + c;
                let value = f64::from((accum[i] / weights[index]).round_ties_even()).clamp(0.0, peak);
                out[i] = T::from_f64(value);
            }
        }
    }
    let mut result = current_source.clone();
    paste(&mut result, x0, y0, &result_crop)?;
    let mut remaining_crop = mask_crop.clone();
    for (value, r) in remaining_crop.data_mut().iter_mut().zip(&recovered) {
        if *r {
            *value = 0;
        }
    }
    if remaining_crop.data().iter().any(|v| *v > 0) {
        let grown = dilate_mask(&remaining_crop, &ellipse(5))?;
        for ((value, g), m) in remaining_crop
            .data_mut()
            .iter_mut()
            .zip(grown.data())
            .zip(mask_crop.data())
        {
            *value = g & m;
        }
    }
    paste(&mut remaining, x0, y0, &remaining_crop)?;
    Ok((result, remaining))
}

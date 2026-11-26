//! Background-aligned temporal blending of filled frames (`stabilize_filled_frame_bidirectional`).

use raiw_cv::{InpaintMethod, Mat, inpaint};

use super::flow::{Maps, backward_warp, displacement_to_maps, flow_field};
use super::motion::{forward_backward_error, propagate_context_motion};
use super::pixel::{
    BoolPlane, LINEAR, NEAREST, VideoPixel, crop_mat, depth_bounds, dilate_mask, ellipse, paste,
};
use crate::error::{Result, VideoError};
use crate::py::round_int;

/// One neighbor: its source frame, its cleaned frame, and its mask.
pub type Neighbor<'a, T> = (&'a Mat<T>, &'a Mat<T>, &'a Mat<u8>);

fn guarded_gray<T: VideoPixel>(source: &Mat<T>, guard: &Mat<u8>) -> Result<Mat<u8>> {
    let gray = T::gray_u8(source)?;
    if guard.data().iter().any(|v| *v > 0) {
        return Ok(inpaint(&gray, guard, 3.0, InpaintMethod::Telea)?);
    }
    Ok(gray)
}

/// `_background_aligned_maps`: flow inside occluded masks from a context affine fit.
pub fn background_aligned_maps<T: VideoPixel>(
    current_source: &Mat<T>,
    current_mask: &Mat<u8>,
    neighbor_source: &Mat<T>,
    neighbor_mask: &Mat<u8>,
    fb_tolerance: f32,
) -> Result<Maps> {
    let kernel = ellipse(5);
    let current_guard = dilate_mask(&BoolPlane::from_mask(current_mask).to_mask(255), &kernel)?;
    let current_guard_plane = BoolPlane::from_mask(&current_guard);
    let neighbor_guard = dilate_mask(&BoolPlane::from_mask(neighbor_mask).to_mask(255), &kernel)?;
    let current_gray = guarded_gray(current_source, &current_guard)?;
    let neighbor_gray = guarded_gray(neighbor_source, &neighbor_guard)?;
    let flow_to = flow_field(&current_gray, &neighbor_gray)?;
    let flow_from = flow_field(&neighbor_gray, &current_gray)?;
    let raw_maps = displacement_to_maps(&flow_to);
    let fb = forward_backward_error(&flow_to, &flow_from)?;
    let warped_guard = BoolPlane::from_mask(&backward_warp(&neighbor_guard, &raw_maps, NEAREST)?);
    let consistent = BoolPlane {
        width: fb.len(),
        height: 1,
        data: fb.iter().map(|e| *e <= fb_tolerance).collect(),
    };
    let context = BoolPlane {
        width: current_guard_plane.width,
        height: current_guard_plane.height,
        data: (0..fb.len())
            .map(|i| !(current_guard_plane.data[i] || warped_guard.data[i]) && consistent.data[i])
            .collect(),
    };
    if context.count() < 24 {
        return Ok(raw_maps);
    }
    let keep = BoolPlane {
        width: context.width,
        height: context.height,
        data: (0..fb.len())
            .map(|i| !current_guard_plane.data[i] && consistent.data[i])
            .collect(),
    };
    Ok(propagate_context_motion(&flow_to, &context, &keep, 24)
        .map_or(raw_maps, |prop| displacement_to_maps(&prop)))
}

/// Options for [`stabilize_filled_frame_bidirectional`].
#[derive(Clone, Copy, Debug)]
pub struct StabilizeOptions {
    pub blend: f64,
    pub max_context_residual: f64,
}

impl Default for StabilizeOptions {
    fn default() -> Self {
        Self {
            blend: 0.5,
            max_context_residual: 12.0,
        }
    }
}

fn mask_bounds(masks: &[&Mat<u8>]) -> Option<(usize, usize, usize, usize)> {
    let (w, h) = masks.first()?.size();
    let mut union = BoolPlane::new(w, h);
    for mask in masks {
        for (u, v) in union.data.iter_mut().zip(mask.data()) {
            *u |= *v > 0;
        }
    }
    union.bounds()
}

/// `stabilize_filled_frame_bidirectional`: blend motion-aligned prior and future fills where the
/// unmasked context agrees; only covered current-mask pixels change.
pub fn stabilize_filled_frame_bidirectional<T: VideoPixel>(
    current_source: &Mat<T>,
    current_cleaned: Mat<T>,
    current_mask: &Mat<u8>,
    previous: Option<Neighbor<'_, T>>,
    next: Option<Neighbor<'_, T>>,
    options: StabilizeOptions,
) -> Result<Mat<T>> {
    if !(0.0..=1.0).contains(&options.blend) {
        return Err(VideoError::value("Temporal blend must be between 0 and 1"));
    }
    if options.max_context_residual <= 0.0 {
        return Err(VideoError::value("Context residual threshold must be positive"));
    }
    let geometry_error = || VideoError::value("Temporal fill inputs must share frame and mask geometry");
    if current_source.size() != current_cleaned.size() || current_mask.size() != current_source.size() {
        return Err(geometry_error());
    }
    let neighbors: Vec<Neighbor<'_, T>> = [previous, next].into_iter().flatten().collect();
    if neighbors.is_empty() {
        return Ok(current_cleaned);
    }
    for (src, clean, mask) in &neighbors {
        if src.size() != current_source.size()
            || clean.size() != current_cleaned.size()
            || mask.size() != current_mask.size()
        {
            return Err(geometry_error());
        }
    }
    let mut all_masks = vec![current_mask];
    all_masks.extend(neighbors.iter().map(|n| n.2));
    let Some((xmin, ymin, xmax, ymax)) = mask_bounds(&all_masks) else {
        return Ok(current_cleaned);
    };
    let (width, height) = current_mask.size();
    let (mask_w, mask_h) = (xmax - xmin + 1, ymax - ymin + 1);
    let padding = round_int(mask_w.max(mask_h) as f64 * 0.75).max(24) as usize;
    let (x0, y0) = (xmin.saturating_sub(padding), ymin.saturating_sub(padding));
    let (x1, y1) = (width.min(xmax + padding + 1), height.min(ymax + padding + 1));
    let source_crop = crop_mat(current_source, x0, y0, x1, y1)?;
    let cleaned_crop = crop_mat(&current_cleaned, x0, y0, x1, y1)?;
    let mask_crop = crop_mat(current_mask, x0, y0, x1, y1)?;
    let hole = BoolPlane::from_mask(&mask_crop);
    let hole_count = hole.count();
    if hole_count == 0 {
        return Ok(current_cleaned);
    }
    let (eff_residual, peak) = depth_bounds::<T>(options.max_context_residual);
    let dilation = (round_int(mask_w.max(mask_h) as f64 * 0.25).max(7) | 1) as usize;
    let kernel = ellipse(dilation);
    let channels = current_source.channels();
    let mut accepted: Vec<(f64, BoolPlane, Vec<f32>)> = Vec::new();
    for (src, clean, mask) in &neighbors {
        let nb_src = crop_mat(src, x0, y0, x1, y1)?;
        let nb_clean = crop_mat(clean, x0, y0, x1, y1)?;
        let nb_mask = crop_mat(mask, x0, y0, x1, y1)?;
        let maps = background_aligned_maps(&source_crop, &mask_crop, &nb_src, &nb_mask, 1.75)?;
        let warped_src = backward_warp(&nb_src, &maps, LINEAR)?;
        let warped_clean = backward_warp(&nb_clean, &maps, LINEAR)?;
        let warped_mask = BoolPlane::from_mask(&backward_warp(&nb_mask, &maps, NEAREST)?);
        let covered = hole.zip_with(&warped_mask, |h, m| h && m);
        if (covered.count() as f64 / hole_count as f64) < 0.85 {
            continue;
        }
        let occupied = hole.zip_with(&warped_mask, |h, m| h || m);
        let grown = BoolPlane::from_mask(&dilate_mask(&occupied.to_mask(1), &kernel)?);
        let context = grown.zip_with(&occupied, |g, o| g && !o);
        let context_count = context.count();
        if context_count < 64 {
            continue;
        }
        let (s, w) = (source_crop.data(), warped_src.data());
        let mut total = 0.0;
        for (index, set) in context.data.iter().enumerate() {
            if *set {
                for c in 0..channels {
                    let i = index * channels + c;
                    total += f64::from((s[i].to_f32() - w[i].to_f32()).abs());
                }
            }
        }
        let residual = total / (context_count * channels) as f64;
        if residual > eff_residual {
            continue;
        }
        let conf = (1.0 - residual / eff_residual).max(0.0);
        accepted.push((
            conf,
            covered,
            warped_clean.data().iter().map(|v| v.to_f32()).collect(),
        ));
    }
    if accepted.is_empty() {
        return Ok(current_cleaned);
    }
    let pixels = hole.data.len();
    let mut accum = vec![0.0f32; pixels * channels];
    let mut weights = vec![0.0f32; pixels];
    if accepted.len() == 1 {
        let (conf, covered, warped) = &accepted[0];
        let w = (options.blend * conf) as f32;
        for (index, set) in covered.data.iter().enumerate() {
            if *set {
                for c in 0..channels {
                    accum[index * channels + c] = w * warped[index * channels + c];
                }
                weights[index] = w;
            }
        }
    } else {
        let weighted: Vec<f64> = accepted
            .iter()
            .enumerate()
            .map(|(i, (conf, _, _))| conf * if i == 0 && previous.is_some() { 1.35 } else { 1.0 })
            .collect();
        let total_conf: f64 = weighted.iter().sum();
        if total_conf <= 1e-6 {
            return Ok(current_cleaned);
        }
        let max_conf = accepted.iter().map(|a| a.0).fold(f64::NEG_INFINITY, f64::max);
        let total_blend = (options.blend * 1.25).min(0.75) * max_conf;
        for (weight, (_, covered, warped)) in weighted.iter().zip(&accepted) {
            let w = (total_blend * (weight / total_conf)) as f32;
            for (index, set) in covered.data.iter().enumerate() {
                if *set {
                    for c in 0..channels {
                        accum[index * channels + c] += w * warped[index * channels + c];
                    }
                    weights[index] += w;
                }
            }
        }
    }
    if !weights.iter().any(|w| *w > 0.0) {
        return Ok(current_cleaned);
    }
    let mut result_crop = cleaned_crop.clone();
    let current = cleaned_crop.data();
    let out = result_crop.data_mut();
    for (index, w) in weights.iter().enumerate() {
        if *w > 0.0 {
            for c in 0..channels {
                let i = index * channels + c;
                let blended = (1.0 - w) * current[i].to_f32() + accum[i];
                out[i] = T::from_f64(f64::from(blended.round_ties_even()).clamp(0.0, peak));
            }
        }
    }
    let mut result = current_cleaned;
    paste(&mut result, x0, y0, &result_crop)?;
    Ok(result)
}

//! Removal masks for stable boxes (`_mask_for_region`, `_masks_for_regions`, `_split_disjoint_masks`).

use raiw_cv::{
    Connectivity, Interpolation, Mat, MorphShape, StructuringElement, connected_components, dilate, resize,
};
use raiw_visible::eraser::boxes_to_mask;

use super::frame::{Region, region_parts};
use super::policy::MaskStyle;
use super::templates::template_source;
use crate::error::Result;
use crate::py::round_int;

/// `_mask_for_region`: a padded box, or the dilated Veo diamond for near-square Veo boxes.
pub fn mask_for_region(
    width: usize,
    height: usize,
    region: Region,
    padding_fraction: f64,
    style: MaskStyle,
) -> Result<Mat<u8>> {
    let (x, y, rw, rh) = region_parts(region);
    if style == MaskStyle::Veo && rh > 0 && (0.80..=1.25).contains(&(rw as f64 / rh as f64)) {
        let mut mask = Mat::<u8>::zeros(width, height, 1);
        let resized = resize(
            template_source("veo-diamond")?,
            rw as usize,
            rh as usize,
            Interpolation::Area,
        )?;
        let binary = resized.map(|v| if v >= 24 { 255 } else { 0 });
        let dilation = round_int(rh as f64 * 0.08).max(2) as usize;
        let side = dilation * 2 + 1;
        let diamond = dilate(
            &binary,
            &StructuringElement::new(MorphShape::Ellipse, side, side),
            1,
        )?;
        let (x0, y0) = (x.max(0) as usize, y.max(0) as usize);
        let (x1, y1) = (
            (x + rw).min(width as i64).max(0) as usize,
            (y + rh).min(height as i64).max(0) as usize,
        );
        for yy in y0..y1 {
            for xx in x0..x1 {
                mask.set(xx, yy, 0, diamond.at(xx - x0, yy - y0, 0));
            }
        }
        return Ok(mask);
    }
    let padding = round_int(rh as f64 * padding_fraction).max(4) as i32;
    let padded = Region::new(
        region.x - padding,
        region.y - padding,
        region.w + 2 * padding,
        region.h + 2 * padding,
    );
    Ok(boxes_to_mask(height, width, &[padded], 0)?)
}

/// Mask geometry for one frame's slots.
#[derive(Clone, Copy, Debug)]
pub struct MaskGeometry<'a> {
    pub padding_fraction: f64,
    pub mask_style: MaskStyle,
    pub slot_padding_fractions: Option<&'a [f64]>,
    pub slot_mask_styles: Option<&'a [MaskStyle]>,
    pub dilate_px: Option<i32>,
}

/// `_masks_for_regions`: the combined mask of one frame's slots, `None` when nothing is masked.
pub fn masks_for_regions(
    width: usize,
    height: usize,
    slots: &[Option<Region>],
    geometry: &MaskGeometry<'_>,
) -> Result<Option<Mat<u8>>> {
    let present: Vec<Region> = slots.iter().flatten().copied().collect();
    if present.is_empty() {
        return Ok(None);
    }
    if let Some(dilate_px) = geometry.dilate_px {
        return Ok(Some(boxes_to_mask(height, width, &present, dilate_px.max(0))?));
    }
    let mut combined = Mat::<u8>::zeros(width, height, 1);
    for (slot, region) in slots.iter().enumerate() {
        let Some(region) = region else { continue };
        let padding = geometry
            .slot_padding_fractions
            .and_then(|p| p.get(slot).copied())
            .unwrap_or(geometry.padding_fraction);
        let style = geometry
            .slot_mask_styles
            .and_then(|s| s.get(slot).copied())
            .unwrap_or(geometry.mask_style);
        let mask = mask_for_region(width, height, *region, padding, style)?;
        for (c, m) in combined.data_mut().iter_mut().zip(mask.data()) {
            *c = (*c).max(*m);
        }
    }
    Ok(Some(combined))
}

/// `_split_disjoint_masks`: separate spatially distant components so crops stay local.
pub fn split_disjoint_masks(mask: &Mat<u8>, multi_region: bool) -> Result<Vec<Mat<u8>>> {
    if !multi_region {
        return Ok(vec![mask.clone()]);
    }
    let binary = mask.map(|v| u8::from(v > 0));
    let bridged = dilate(&binary, &StructuringElement::rect(25, 25), 1)?;
    let (labels, map) = connected_components(&bridged, Connectivity::Eight)?;
    if labels <= 2 {
        return Ok(vec![mask.clone()]);
    }
    let mut parts = Vec::new();
    for label in 1..labels as i32 {
        let data: Vec<u8> = mask
            .data()
            .iter()
            .zip(map.data())
            .map(|(m, l)| if *l == label { *m } else { 0 })
            .collect();
        if data.iter().any(|v| *v > 0) {
            parts.push(Mat::from_vec(mask.width(), mask.height(), 1, data)?);
        }
    }
    Ok(if parts.is_empty() {
        vec![mask.clone()]
    } else {
        parts
    })
}

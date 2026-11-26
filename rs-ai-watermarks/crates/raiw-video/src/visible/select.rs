//! Stable-mark selection and multi-mark merging (`_visible_removal_plan`,
//! `_select_stable_visible_marks`, `_merge_selected_visible_marks` in `video.py`).

use raiw_metadata::AiMetadata;

use super::frame::{Region, region_parts};
use super::policy::{MaskStyle, policy, stabilize_localizations};
use super::scan::{Scans, VideoScan};
use crate::provenance::{confirms_video_provenance, contradicts_video_provenance, vendor_group};

/// The regions of one frame: one slot per selected mark (empty when nothing is masked).
pub type FrameRegions = Vec<Option<Region>>;

/// One accepted mark with its stable per-frame regions and fill geometry.
#[derive(Clone, Debug)]
pub struct SelectedMark {
    pub mark: String,
    pub scan: VideoScan,
    pub regions: Vec<Option<Region>>,
    pub padding_fraction: f64,
    pub mask_style: MaskStyle,
}

/// `_boxes_overlap`.
pub fn boxes_overlap(a: Region, b: Region) -> bool {
    let (ax, ay, aw, ah) = region_parts(a);
    let (bx, by, bw, bh) = region_parts(b);
    (ax + aw).min(bx + bw) > ax.max(bx) && (ay + ah).min(by + bh) > ay.max(by)
}

/// `_union_box`.
pub fn union_box(regions: &[Option<Region>]) -> Option<Region> {
    let present: Vec<Region> = regions.iter().flatten().copied().collect();
    if present.is_empty() {
        return None;
    }
    let x0 = present.iter().map(|r| r.x).min()?;
    let y0 = present.iter().map(|r| r.y).min()?;
    let x1 = present.iter().map(|r| r.x + r.w).max()?;
    let y1 = present.iter().map(|r| r.y + r.h).max()?;
    Some(Region::new(x0, y0, x1 - x0, y1 - y0))
}

/// `_visible_removal_plan`: one provider's stable regions and fill geometry.
pub fn visible_removal_plan(
    mark: &str,
    scan: &VideoScan,
    markers: &AiMetadata,
) -> (Vec<Option<Region>>, f64, MaskStyle) {
    let Some(row) = policy(mark) else {
        return (vec![None; scan.detections.len()], 0.12, MaskStyle::Box);
    };
    if contradicts_video_provenance(mark, markers) {
        return (
            vec![None; scan.detections.len()],
            row.padding_fraction,
            row.mask_style,
        );
    }
    let confirmed = confirms_video_provenance(mark, markers);
    (
        stabilize_localizations(mark, &scan.detections, confirmed),
        row.padding_fraction,
        row.mask_style,
    )
}

/// `_select_stable_visible_marks`: spatially disjoint stable marks without cross-vendor merges.
pub fn select_stable_visible_marks(
    scans: &Scans,
    markers: &AiMetadata,
    candidates: &[&str],
    explicit_multi: bool,
) -> Vec<SelectedMark> {
    let mut accepted: Vec<(SelectedMark, Region)> = Vec::new();
    for &candidate in candidates {
        if let Some((first, _)) = accepted.first()
            && !explicit_multi
        {
            let primary = vendor_group(&first.mark);
            if primary.is_none() || vendor_group(candidate) != primary {
                continue;
            }
        }
        let Some((_, scan)) = scans.iter().find(|(m, _)| m == candidate) else {
            continue;
        };
        let (regions, padding_fraction, mask_style) = visible_removal_plan(candidate, scan, markers);
        let Some(ubox) = union_box(&regions) else { continue };
        if !accepted.is_empty() {
            if accepted.iter().any(|(_, prev)| boxes_overlap(ubox, *prev)) {
                continue;
            }
            let frame_overlap = accepted.iter().any(|(prev, _)| {
                regions
                    .iter()
                    .zip(&prev.regions)
                    .any(|(cur, old)| matches!((cur, old), (Some(c), Some(o)) if boxes_overlap(*c, *o)))
            });
            if frame_overlap {
                continue;
            }
        }
        let selected = SelectedMark {
            mark: candidate.to_owned(),
            scan: scan.clone(),
            regions,
            padding_fraction,
            mask_style,
        };
        accepted.push((selected, ubox));
    }
    accepted.into_iter().map(|(s, _)| s).collect()
}

/// `_merge_selected_visible_marks`: one region list across all selected marks.
#[derive(Clone, Debug)]
pub struct MergedMarks {
    pub marks: Vec<String>,
    pub primary_mark: String,
    pub scan: VideoScan,
    pub regions: Vec<FrameRegions>,
    pub detected_frames: usize,
    pub padding_fraction: f64,
    pub mask_style: MaskStyle,
    pub slot_padding_fractions: Option<Vec<f64>>,
    pub slot_mask_styles: Option<Vec<MaskStyle>>,
}

pub fn merge_selected_visible_marks(selected: &[SelectedMark], total_frames: usize) -> Option<MergedMarks> {
    let first = selected.first()?;
    let marks: Vec<String> = selected.iter().map(|s| s.mark.clone()).collect();
    if selected.len() == 1 {
        let regions: Vec<FrameRegions> = first
            .regions
            .iter()
            .map(|r| r.map_or_else(Vec::new, |r| vec![Some(r)]))
            .collect();
        return Some(MergedMarks {
            marks,
            primary_mark: first.mark.clone(),
            scan: first.scan.clone(),
            detected_frames: first.regions.iter().filter(|r| r.is_some()).count(),
            regions,
            padding_fraction: first.padding_fraction,
            mask_style: first.mask_style,
            slot_padding_fractions: None,
            slot_mask_styles: None,
        });
    }
    let mut regions = Vec::with_capacity(total_frames);
    let mut detected = 0;
    for index in 0..total_frames {
        let slots: FrameRegions = selected
            .iter()
            .map(|s| s.regions.get(index).copied().flatten())
            .collect();
        if slots.iter().any(Option::is_some) {
            detected += 1;
            regions.push(slots);
        } else {
            regions.push(Vec::new());
        }
    }
    Some(MergedMarks {
        marks,
        primary_mark: first.mark.clone(),
        scan: first.scan.clone(),
        regions,
        detected_frames: detected,
        padding_fraction: first.padding_fraction,
        mask_style: first.mask_style,
        slot_padding_fractions: Some(selected.iter().map(|s| s.padding_fraction).collect()),
        slot_mask_styles: Some(selected.iter().map(|s| s.mask_style).collect()),
    })
}

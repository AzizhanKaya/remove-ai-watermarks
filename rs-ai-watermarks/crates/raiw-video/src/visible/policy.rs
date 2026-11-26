//! Provider temporal-arbiter tuning (`VISIBLE_MARK_POLICIES`) and the shared recurrence policy
//! (`stabilize_localizations`).

use super::frame::{FrameLocalization, Region, region_iou};

const MIN_STABLE_FRAMES: usize = 5;
const MIN_VEO_STABLE_FRAMES: usize = 12;
const MIN_FIXED_MARK_STABLE_FRAMES: usize = 12;
const MAX_STABLE_GAP: usize = 2;
pub(crate) const STABLE_IOU: f64 = 0.55;

/// How a removal mask is shaped around a stable box.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MaskStyle {
    Box,
    Veo,
}

/// `VisibleMarkPolicy`: one provider's measured arbiter floors plus its fill geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisibleMarkPolicy {
    pub weak_floor: f64,
    pub strong_floor: f64,
    pub transition_floor: f64,
    pub min_stable_frames: usize,
    pub cover_after_confirmation: bool,
    pub padding_fraction: f64,
    pub mask_style: MaskStyle,
    pub anchor_iou: Option<f64>,
    pub accepts_provenance: bool,
    pub provenance_weak_floor: Option<f64>,
}

const fn fixed(weak: f64, strong: f64, transition: f64, padding: f64) -> VisibleMarkPolicy {
    VisibleMarkPolicy {
        weak_floor: weak,
        strong_floor: strong,
        transition_floor: transition,
        min_stable_frames: MIN_FIXED_MARK_STABLE_FRAMES,
        cover_after_confirmation: true,
        padding_fraction: padding,
        mask_style: MaskStyle::Box,
        anchor_iou: Some(0.80),
        accepts_provenance: true,
        provenance_weak_floor: None,
    }
}

impl VisibleMarkPolicy {
    const fn provenance_floor(mut self, floor: f64) -> Self {
        self.provenance_weak_floor = Some(floor);
        self
    }

    const fn anchor(mut self, iou: f64) -> Self {
        self.anchor_iou = Some(iou);
        self
    }

    const fn visual_only(mut self) -> Self {
        self.accepts_provenance = false;
        self
    }
}

/// The policy row for `mark`.
pub fn policy(mark: &str) -> Option<VisibleMarkPolicy> {
    Some(match mark {
        "sora" => VisibleMarkPolicy {
            weak_floor: 0.60,
            strong_floor: 0.65,
            transition_floor: 0.45,
            min_stable_frames: MIN_STABLE_FRAMES,
            cover_after_confirmation: false,
            padding_fraction: 0.28,
            mask_style: MaskStyle::Box,
            anchor_iou: None,
            accepts_provenance: true,
            provenance_weak_floor: Some(0.58),
        },
        "veo" => VisibleMarkPolicy {
            weak_floor: 0.50,
            strong_floor: 0.55,
            transition_floor: 0.35,
            min_stable_frames: MIN_VEO_STABLE_FRAMES,
            cover_after_confirmation: true,
            padding_fraction: 0.18,
            mask_style: MaskStyle::Veo,
            anchor_iou: None,
            accepts_provenance: true,
            provenance_weak_floor: Some(0.45),
        },
        "dreamina" => fixed(0.40, 0.45, 0.34, 0.12),
        "seedance" => fixed(0.38, 0.43, 0.30, 0.0),
        "doubao" => fixed(0.35, 0.55, 0.30, 0.0),
        "dola" => fixed(0.50, 0.52, 0.40, 0.20).provenance_floor(0.48),
        "hailuo" => fixed(0.30, 0.34, 0.28, 0.12),
        "vidu" => fixed(0.52, 0.58, 0.45, 0.12),
        "wan" => fixed(0.50, 0.56, 0.42, 0.12),
        "jimeng" => fixed(0.45, 0.52, 0.38, 0.12),
        "jimeng_pill" => fixed(0.28, 0.31, 0.24, 0.24),
        "qwen" => fixed(0.40, 0.45, 0.35, 0.12),
        "yuanbao" => fixed(0.40, 0.46, 0.34, 0.12),
        "baidu" => fixed(0.35, 0.45, 0.30, 0.12).provenance_floor(0.33),
        "runninghub" => fixed(0.34, 0.40, 0.28, 0.12),
        "microsoft" => fixed(0.38, 0.45, 0.32, 0.14),
        "pixverse" => fixed(0.36, 0.42, 0.30, 0.16),
        "luma" => fixed(0.45, 0.52, 0.36, 0.14).visual_only(),
        "heygen" => fixed(0.34, 0.39, 0.29, 0.14).anchor(0.85).visual_only(),
        "sync_labs" => fixed(0.44, 0.50, 0.36, 0.16),
        "tiktok_ai" => fixed(0.42, 0.48, 0.35, 0.16),
        "generic_ai_label" => fixed(0.48, 0.55, 0.40, 0.12).visual_only(),
        "kling" => fixed(0.20, 0.24, 0.30, 0.12).anchor(0.95).visual_only(),
        _ => return None,
    })
}

/// `stabilize_localizations`: accept only spatially and temporally recurring candidates.
pub fn stabilize_localizations(
    mark: &str,
    detections: &[FrameLocalization],
    provenance: bool,
) -> Vec<Option<Region>> {
    let Some(policy) = policy(mark) else {
        return vec![None; detections.len()];
    };
    let trusted = provenance && policy.accepts_provenance;
    let weak_floor = if trusted {
        policy.provenance_weak_floor.unwrap_or(policy.weak_floor)
    } else {
        policy.weak_floor
    };
    stabilize_with(detections, trusted, weak_floor, &policy)
}

fn stabilize_with(
    detections: &[FrameLocalization],
    provenance: bool,
    weak_floor: f64,
    policy: &VisibleMarkPolicy,
) -> Vec<Option<Region>> {
    let mut accepted: Vec<Option<Region>> = vec![None; detections.len()];
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for (position, detection) in detections.iter().enumerate() {
        let Some(region) = detection.region.filter(|_| detection.confidence >= weak_floor) else {
            continue;
        };
        if let (Some(&last), Some(&first)) = (current.last(), current.first()) {
            let previous = &detections[last];
            let anchor = &detections[first];
            let gap = detection.frame_index as i64 - previous.frame_index as i64;
            let breaks = match (previous.region, anchor.region) {
                (Some(prev), Some(anchor_region)) => {
                    gap > (MAX_STABLE_GAP + 1) as i64
                        || region_iou(prev, region) < STABLE_IOU
                        || policy
                            .anchor_iou
                            .is_some_and(|iou| region_iou(anchor_region, region) < iou)
                }
                _ => true,
            };
            if breaks {
                runs.push(std::mem::take(&mut current));
            }
        }
        current.push(position);
    }
    if !current.is_empty() {
        runs.push(current);
    }
    for run in &runs {
        let strong = run
            .iter()
            .map(|&p| detections[p].confidence)
            .fold(f64::NEG_INFINITY, f64::max)
            >= policy.strong_floor;
        if run.len() < policy.min_stable_frames || (!provenance && !strong) {
            continue;
        }
        for &position in run {
            accepted[position] = detections[position].region;
        }
        for pair in run.windows(2) {
            let (left_pos, right_pos) = (pair[0], pair[1]);
            if right_pos - left_pos <= 1 {
                continue;
            }
            let (Some(left), Some(right)) = (detections[left_pos].region, detections[right_pos].region)
            else {
                continue;
            };
            if region_iou(left, right) < STABLE_IOU {
                continue;
            }
            for (offset, slot) in accepted[left_pos + 1..right_pos].iter_mut().enumerate() {
                let missing = left_pos + 1 + offset;
                *slot = Some(if missing - left_pos <= right_pos - missing {
                    left
                } else {
                    right
                });
            }
        }
    }
    let confirmed: Vec<usize> = accepted
        .iter()
        .enumerate()
        .filter_map(|(p, r)| r.map(|_| p))
        .collect();
    if (provenance || policy.cover_after_confirmation) && !confirmed.is_empty() {
        let confirmed_regions: Vec<Region> = confirmed.iter().filter_map(|&p| accepted[p]).collect();
        let mut carry = confirmed[0];
        let mut last_supported = confirmed[confirmed.len() - 1];
        for position in 0..accepted.len() {
            if accepted[position].is_some() {
                carry = position;
                continue;
            }
            let raw = &detections[position];
            if let Some(raw_region) = raw.region
                && raw.confidence >= policy.transition_floor
                && confirmed_regions
                    .iter()
                    .any(|c| region_iou(raw_region, *c) >= STABLE_IOU)
            {
                accepted[position] = Some(raw_region);
                last_supported = last_supported.max(position);
                continue;
            }
            if position > last_supported + policy.min_stable_frames {
                continue;
            }
            accepted[position] = accepted[carry];
        }
    }
    accepted
}

const COARSE_SCAN_STRIDE: usize = 4;

/// `_coarse_promotion_floor`.
pub(crate) fn coarse_promotion_floor(mark: &str) -> f64 {
    policy(mark).map_or(0.15, |p| {
        p.provenance_weak_floor
            .map_or(p.weak_floor, |f| p.weak_floor.min(f))
            * 0.85
    })
}

/// `_should_promote_coarse_mark`: whether coarse samples can join a valid stable run.
pub(crate) fn should_promote_coarse_mark(mark: &str, coarse_hits: &[(usize, FrameLocalization)]) -> bool {
    let floor = coarse_promotion_floor(mark);
    let qualified: Vec<(usize, Region)> = coarse_hits
        .iter()
        .filter_map(|(i, loc)| loc.region.filter(|_| loc.confidence >= floor).map(|r| (*i, r)))
        .collect();
    if qualified.is_empty() {
        return false;
    }
    let Some(policy) = policy(mark) else { return true };
    if policy.min_stable_frames < 12 || coarse_hits.len() < 3 {
        return true;
    }
    let min_pair_iou = policy.anchor_iou.map_or(STABLE_IOU, |a| STABLE_IOU.max(a * 0.92));
    let max_gap = COARSE_SCAN_STRIDE * 3;
    for (i, (a_idx, a_reg)) in qualified.iter().enumerate() {
        for (b_idx, b_reg) in &qualified[i + 1..] {
            if b_idx - a_idx > max_gap {
                break;
            }
            if region_iou(*a_reg, *b_reg) >= min_pair_iou {
                return true;
            }
        }
    }
    false
}

pub(crate) const fn coarse_stride() -> usize {
    COARSE_SCAN_STRIDE
}

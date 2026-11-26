//! Streaming per-frame removal (`iter_clean_video_frames`, `_precalibrate_stationary_mattes`).
//!
//! A bounded ±12-frame donor window recovers real background before the single-frame fill, and
//! a one-frame lookahead stabilizes the fill against its neighbors.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::path::Path;

use raiw_cv::Mat;
use raiw_visible::eraser::{Backend, FillBackends, FillKind};

use super::frame::region_iou;
use super::mask::{MaskGeometry, masks_for_regions, split_disjoint_masks};
use super::policy::MaskStyle;
use super::scan::VideoScan;
use super::select::FrameRegions;
use crate::decode::VideoReader;
use crate::error::{Result, VideoError};
use crate::py::{linspace_indices, median, round_int};
use crate::temporal::pixel::BoolPlane;
use crate::temporal::{
    ClipAlphaMatte, RecoverOptions, StabilizeOptions, VideoPixel, estimate_clip_alpha_matte,
    recover_unobscured_background, stabilize_filled_frame_bidirectional, unblend_ycbcr_roi,
};

/// Options for [`for_each_clean_frame`].
#[derive(Clone, Copy, Debug)]
pub struct CleanOptions<'a> {
    pub backend: Backend,
    pub padding_fraction: f64,
    pub mask_style: MaskStyle,
    pub slot_padding_fractions: Option<&'a [f64]>,
    pub slot_mask_styles: Option<&'a [MaskStyle]>,
    pub temporal_consistency: bool,
    pub dilate_px: Option<i32>,
    pub allow_unmasked_donors: bool,
    pub start_frame: usize,
    pub stop_frame: Option<usize>,
}

impl Default for CleanOptions<'_> {
    fn default() -> Self {
        Self {
            backend: Backend::Cv2,
            padding_fraction: 0.28,
            mask_style: MaskStyle::Box,
            slot_padding_fractions: None,
            slot_mask_styles: None,
            temporal_consistency: true,
            dilate_px: None,
            allow_unmasked_donors: true,
            start_frame: 0,
            stop_frame: None,
        }
    }
}

impl CleanOptions<'_> {
    fn geometry(&self) -> MaskGeometry<'_> {
        MaskGeometry {
            padding_fraction: self.padding_fraction,
            mask_style: self.mask_style,
            slot_padding_fractions: self.slot_padding_fractions,
            slot_mask_styles: self.slot_mask_styles,
            dilate_px: self.dilate_px,
        }
    }
}

type Bounds = (i64, i64, i64, i64);
type NextEntry<'a, T> = Option<(&'a Mat<T>, &'a Mat<T>, Option<&'a Mat<u8>>)>;
type Sink<'a, T> = dyn FnMut(Mat<T>, bool) -> Result<()> + 'a;

/// `_precalibrate_stationary_mattes`: alpha mattes for boxes anchored on at least 12 frames.
pub fn precalibrate_stationary_mattes<T: VideoPixel>(
    source: &Path,
    regions: &[FrameRegions],
    scan: &VideoScan,
    options: &CleanOptions<'_>,
) -> Result<Vec<ClipAlphaMatte>> {
    let max_slots = regions.iter().map(Vec::len).max().unwrap_or(0);
    let mut candidates: Vec<(Bounds, Vec<usize>)> = Vec::new();
    for slot in 0..max_slots {
        let pairs: Vec<(usize, super::Region)> = regions
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.get(slot).copied().flatten().map(|r| (i, r)))
            .collect();
        if pairs.len() < 12 {
            continue;
        }
        let first = pairs[0].1;
        let anchored: Vec<(usize, super::Region)> = pairs
            .into_iter()
            .filter(|(_, r)| region_iou(*r, first) >= 0.85)
            .collect();
        if anchored.len() < 12 {
            continue;
        }
        let mut heights: Vec<f64> = anchored.iter().map(|(_, r)| f64::from(r.h)).collect();
        let med_h = median(&mut heights) as i64;
        let slot_pad = options
            .slot_padding_fractions
            .and_then(|p| p.get(slot).copied())
            .unwrap_or(options.padding_fraction);
        let pad = options.dilate_px.map_or_else(
            || round_int(med_h as f64 * slot_pad).max(4),
            |d| i64::from(d.max(0)),
        );
        let x0 = anchored.iter().map(|(_, r)| i64::from(r.x)).min().unwrap_or(0);
        let y0 = anchored.iter().map(|(_, r)| i64::from(r.y)).min().unwrap_or(0);
        let x1 = anchored
            .iter()
            .map(|(_, r)| i64::from(r.x + r.w))
            .max()
            .unwrap_or(0);
        let y1 = anchored
            .iter()
            .map(|(_, r)| i64::from(r.y + r.h))
            .max()
            .unwrap_or(0);
        let bounds = (
            (x0 - pad).max(0),
            (y0 - pad).max(0),
            (scan.width as i64).min(x1 + pad),
            (scan.height as i64).min(y1 + pad),
        );
        candidates.push((bounds, anchored.into_iter().map(|(i, _)| i).collect()));
    }
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let samples: Vec<Vec<usize>> = candidates
        .iter()
        .map(|(_, anchored)| {
            if anchored.len() > 36 {
                linspace_indices(anchored.len(), 36)
                    .into_iter()
                    .map(|i| anchored[i])
                    .collect()
            } else {
                anchored.clone()
            }
        })
        .collect();
    let wanted: BTreeSet<usize> = samples.iter().flatten().copied().collect();
    let Some(&max_needed) = wanted.iter().next_back() else {
        return Ok(Vec::new());
    };
    let mut decoded: HashMap<usize, Mat<T>> = HashMap::new();
    let mut reader = VideoReader::<T>::open(source)?;
    for index in 0..=max_needed {
        if wanted.contains(&index) {
            let Some(frame) = reader.read()? else { break };
            decoded.insert(index, frame);
        } else if !reader.grab()? {
            break;
        }
    }
    let mut mattes = Vec::new();
    for ((bounds, _), picked) in candidates.iter().zip(&samples) {
        let sequence: Vec<Mat<T>> = picked.iter().filter_map(|i| decoded.get(i).cloned()).collect();
        if let Some(matte) = estimate_clip_alpha_matte(&sequence, *bounds)? {
            mattes.push(matte);
        }
    }
    Ok(mattes)
}

struct Pending<T> {
    source: Mat<T>,
    frame: Mat<T>,
    sub_masks: Vec<Mat<u8>>,
    stage_mask: Option<Mat<u8>>,
    removed: bool,
}

struct Prepared<T> {
    source: Mat<T>,
    sub_masks: Vec<Mat<u8>>,
    stage_mask: Option<Mat<u8>>,
    removed: bool,
}

fn crop_neighbor_mask(neighbor: Option<&Mat<u8>>, sub_mask: &Mat<u8>, single: bool) -> Option<Mat<u8>> {
    let neighbor = neighbor?;
    if single {
        return Some(neighbor.clone());
    }
    let (x0, y0, x1, y1) = BoolPlane::from_mask(sub_mask).bounds()?;
    let (w, h) = neighbor.size();
    let (x0, y0) = (x0.saturating_sub(48), y0.saturating_sub(48));
    let (x1, y1) = (w.min(x1 + 1 + 48), h.min(y1 + 1 + 48));
    let mut out = Mat::<u8>::zeros(w, h, 1);
    let mut any = false;
    for y in y0..y1 {
        for x in x0..x1 {
            let v = neighbor.at(x, y, 0);
            any |= v > 0;
            out.set(x, y, 0, v);
        }
    }
    any.then_some(out)
}

struct Previous<T> {
    source: Mat<T>,
    cleaned: Mat<T>,
    mask: Option<Mat<u8>>,
}

fn flush_pending<T: VideoPixel>(
    pending: &mut Option<Pending<T>>,
    previous: &mut Option<Previous<T>>,
    next: NextEntry<'_, T>,
    temporal: bool,
) -> Result<Option<(Mat<T>, bool)>> {
    let Some(p) = pending.take() else { return Ok(None) };
    let mut out = p.frame;
    if temporal && !p.sub_masks.is_empty() {
        let single = p.sub_masks.len() == 1;
        for sub in &p.sub_masks {
            let prev_sub = match previous.as_ref() {
                Some(prev) => crop_neighbor_mask(prev.mask.as_ref(), sub, single),
                None => None,
            };
            let next_sub = match next {
                Some((_, _, mask)) => crop_neighbor_mask(mask, sub, single),
                None => None,
            };
            let prev_arg = previous
                .as_ref()
                .zip(prev_sub.as_ref())
                .map(|(prev, m)| (&prev.source, &prev.cleaned, m));
            let next_arg = next.zip(next_sub.as_ref()).map(|((s, f, _), m)| (s, f, m));
            if prev_arg.is_some() || next_arg.is_some() {
                out = stabilize_filled_frame_bidirectional(
                    &p.source,
                    out,
                    sub,
                    prev_arg,
                    next_arg,
                    StabilizeOptions::default(),
                )?;
            }
        }
    }
    *previous = Some(Previous {
        source: p.source,
        cleaned: out.clone(),
        mask: p.stage_mask,
    });
    Ok(Some((out, p.removed)))
}

/// `iter_clean_video_frames` as a push stream: `sink(frame, was_removed)` for every frame in
/// `[start_frame, stop_frame)`.
pub fn for_each_clean_frame<T: VideoPixel>(
    source: &Path,
    scan: &VideoScan,
    regions: &[FrameRegions],
    fills: &FillBackends,
    options: &CleanOptions<'_>,
    mut sink: impl FnMut(Mat<T>, bool) -> Result<()>,
) -> Result<()> {
    let total = regions.len();
    let start = options.start_frame;
    let stop = options.stop_frame.map_or(total, |s| s.min(total));
    if start >= stop {
        return Ok(());
    }
    let dirty: Vec<usize> = (start..stop)
        .filter(|&i| regions[i].iter().any(Option::is_some))
        .collect();
    let mattes = if options.temporal_consistency && !dirty.is_empty() {
        precalibrate_stationary_mattes::<T>(source, regions, scan, options)?
    } else {
        Vec::new()
    };
    let mut reader = VideoReader::<T>::open(source).map_err(|_| {
        VideoError::runtime(format!(
            "OpenCV could not reopen video for removal: {}",
            source.display()
        ))
    })?;
    let kind: FillKind = fills.resolve(options.backend);
    let batch_size =
        if matches!(kind, FillKind::Lama | FillKind::Migan) && fills.prefers_batching(options.backend) {
            8
        } else {
            1
        };
    let radius = if options.temporal_consistency { 12 } else { 0 };
    let ended = |index: usize| {
        VideoError::runtime(format!(
            "Video ended while advancing to frame {index}: {}",
            source.display()
        ))
    };
    let mut next_decode = 0usize;
    while next_decode < start {
        if !reader.grab()? {
            return Err(ended(next_decode));
        }
        next_decode += 1;
    }
    let passthrough = |reader: &mut VideoReader<T>,
                       next_decode: &mut usize,
                       limit: usize,
                       sink: &mut Sink<'_, T>|
     -> Result<()> {
        while *next_decode < limit {
            let frame = reader.read()?.ok_or_else(|| {
                VideoError::runtime(format!(
                    "Video ended while reading frame {next_decode}: {}",
                    source.display()
                ))
            })?;
            *next_decode += 1;
            sink(frame, false)?;
        }
        Ok(())
    };
    let (Some(&first_dirty), Some(&last_dirty)) = (dirty.first(), dirty.last()) else {
        return passthrough(&mut reader, &mut next_decode, stop, &mut sink);
    };
    let active_start = start.max(first_dirty.saturating_sub(radius));
    let active_end = (stop - 1).min(last_dirty + radius);
    passthrough(&mut reader, &mut next_decode, active_start, &mut sink)?;

    let geometry = options.geometry();
    let (width, height) = (scan.width, scan.height);
    let mut window: VecDeque<(usize, Mat<T>, Option<Mat<u8>>)> = VecDeque::new();
    let mut prepared: Vec<Prepared<T>> = Vec::new();
    let mut pending: Option<Pending<T>> = None;
    let mut previous: Option<Previous<T>> = None;
    let mut clean_plate: Option<(Mat<T>, Option<Mat<u8>>)> = None;

    let drain = |prepared: &mut Vec<Prepared<T>>,
                 pending: &mut Option<Pending<T>>,
                 previous: &mut Option<Previous<T>>,
                 sink: &mut Sink<'_, T>|
     -> Result<()> {
        for entry in prepared.drain(..) {
            let mut filled = entry.source.clone();
            for sub in &entry.sub_masks {
                filled = fills.fill(&filled, sub, kind)?;
            }
            if options.temporal_consistency {
                if let Some((frame, removed)) = flush_pending(
                    pending,
                    previous,
                    Some((&entry.source, &filled, entry.stage_mask.as_ref())),
                    true,
                )? {
                    sink(frame, removed)?;
                }
                *pending = Some(Pending {
                    source: entry.source,
                    frame: filled,
                    sub_masks: entry.sub_masks,
                    stage_mask: entry.stage_mask,
                    removed: entry.removed,
                });
            } else {
                sink(filled, entry.removed)?;
            }
        }
        Ok(())
    };

    for frame_index in active_start..=active_end {
        while next_decode <= active_end && next_decode <= frame_index + radius {
            let frame = reader.read()?.ok_or_else(|| {
                VideoError::runtime(format!(
                    "Video ended while reading frame {next_decode}: {}",
                    source.display()
                ))
            })?;
            let mask = masks_for_regions(width, height, &regions[next_decode], &geometry)?;
            window.push_back((next_decode, frame, mask));
            next_decode += 1;
        }
        while window.front().is_some_and(|(i, _, _)| *i + radius < frame_index) {
            window.pop_front();
        }
        let Some(position) = window.iter().position(|(i, _, _)| *i == frame_index) else {
            return Err(VideoError::runtime(format!(
                "Frame {frame_index} left the donor window"
            )));
        };
        let (_, current_source, current_mask) = &window[position];
        let mut stage_source = current_source.clone();
        let mut stage_mask = None;
        let mut remaining_subs = Vec::new();
        let mut removed = false;
        if let Some(mask) = current_mask {
            let mut working = mask.clone();
            for matte in &mattes {
                (stage_source, working) = unblend_ycbcr_roi(&stage_source, &working, matte)?;
            }
            let current_regions: Vec<super::Region> =
                regions[frame_index].iter().flatten().copied().collect();
            let sub_masks = split_disjoint_masks(&working, current_regions.len() > 1)?;
            let mut donors: Vec<(&Mat<T>, Option<&Mat<u8>>)> = Vec::new();
            if options.temporal_consistency && working.data().iter().any(|v| *v > 0) {
                if window.len() > 1 {
                    let mut disjoint = Vec::new();
                    let mut overlap = Vec::new();
                    for (d_index, d_source, d_mask) in &window {
                        if *d_index == frame_index || !(options.allow_unmasked_donors || d_mask.is_some()) {
                            continue;
                        }
                        let is_disjoint = d_mask.is_none()
                            || current_regions.is_empty()
                            || regions[*d_index]
                                .iter()
                                .flatten()
                                .all(|dr| current_regions.iter().all(|cr| region_iou(*dr, *cr) < 0.25));
                        if is_disjoint {
                            disjoint.push((*d_index, d_source, d_mask.as_ref()));
                        } else {
                            overlap.push((*d_index, d_source, d_mask.as_ref()));
                        }
                    }
                    disjoint.sort_by_key(|(i, _, _)| i.abs_diff(frame_index));
                    overlap.sort_by_key(|(i, _, _)| (i.abs_diff(frame_index), *i > frame_index));
                    disjoint.truncate(8);
                    let slots = 8 - disjoint.len();
                    let picked_overlap: Vec<_> = if slots > 0 && !overlap.is_empty() {
                        if overlap.len() <= slots {
                            overlap
                        } else {
                            linspace_indices(overlap.len(), slots)
                                .into_iter()
                                .map(|i| overlap[i])
                                .collect()
                        }
                    } else {
                        Vec::new()
                    };
                    donors = disjoint
                        .into_iter()
                        .chain(picked_overlap)
                        .map(|(_, s, m)| (s, m))
                        .collect();
                }
                if let Some((plate, plate_mask)) = &clean_plate
                    && (options.allow_unmasked_donors || plate_mask.is_some())
                {
                    donors.push((plate, plate_mask.as_ref()));
                }
            }
            for mut sub in sub_masks {
                if !donors.is_empty() && sub.data().iter().any(|v| *v > 0) {
                    (stage_source, sub) = recover_unobscured_background(
                        &stage_source,
                        &sub,
                        &donors,
                        RecoverOptions::default(),
                    )?;
                }
                if sub.data().iter().any(|v| *v > 0) {
                    remaining_subs.push(sub);
                }
            }
            stage_mask = match remaining_subs.len() {
                0 => None,
                1 => Some(remaining_subs[0].clone()),
                _ => {
                    let mut combined = remaining_subs[0].clone();
                    for sub in &remaining_subs[1..] {
                        for (c, s) in combined.data_mut().iter_mut().zip(sub.data()) {
                            *c = (*c).max(*s);
                        }
                    }
                    Some(combined)
                }
            };
            removed = true;
        }
        clean_plate = options
            .temporal_consistency
            .then(|| (stage_source.clone(), stage_mask.clone()));
        prepared.push(Prepared {
            source: stage_source,
            sub_masks: remaining_subs,
            stage_mask,
            removed,
        });
        if prepared.len() >= batch_size {
            drain(&mut prepared, &mut pending, &mut previous, &mut sink)?;
        }
    }
    drain(&mut prepared, &mut pending, &mut previous, &mut sink)?;
    if let Some((frame, removed)) =
        flush_pending(&mut pending, &mut previous, None, options.temporal_consistency)?
    {
        sink(frame, removed)?;
    }
    window.clear();
    passthrough(&mut reader, &mut next_decode, stop, &mut sink)
}

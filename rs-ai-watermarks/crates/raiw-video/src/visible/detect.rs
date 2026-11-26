//! Per-frame provider-mark detectors (the `detect_*_frame` family of `video_visible.py`).
//!
//! Every result is an untrusted candidate; only the temporal arbiter can authorize removal.

use std::rc::Rc;

use raiw_cv::{ColorConversion, Interpolation, Mat, MatView, TemplateSearch, cvt_color, min_max_loc, resize};

use super::frame::{
    FeatureKey, FrameLocalization, PreparedFrame, Region, best_match, best_match_in, bounded_region, crop,
    region, region_parts, restore_region, single_score, slice, view_max, view_size,
};
use super::templates::{
    resized_raw_template, resized_sobel_template, resized_template_feature, resized_template_half_feature,
    template_source, top_hat,
};
use crate::error::Result;
use crate::py::{cpython_int_set_order, round_int};
use crate::temporal::sobel_magnitude;

const SORA_TEMPLATE_SIZE: (f64, f64) = (180.0, 64.0);
const SORA_RELATIVE_HEIGHTS: [f64; 5] = [0.065, 0.075, 0.085, 0.095, 0.105];
const SORA_ICON_SHAPE_FLOOR: f64 = 0.50;
const KLING_MIN_WHITE_FRACTION: f64 = 0.02;
const HAILUO_MIN_WHITE_FRACTION: f64 = 0.02;
const VEO_REFERENCE_SHORT_SIDE: f64 = 720.0;
const VEO_DIAMOND_PROFILES: [(f64, f64, f64); 3] =
    [(56.0, 92.0, 92.0), (48.0, 72.0, 72.0), (44.0, 29.0, 40.0)];

fn thousandths(range: impl Iterator<Item = i32>) -> Vec<f64> {
    range.map(|v| f64::from(v) / 1000.0).collect()
}

fn ri(value: f64) -> i64 {
    round_int(value)
}

fn odd_kernel(value: f64, floor: i64) -> usize {
    floor.max(ri(value) | 1) as usize
}

/// Which provider detector to run.
pub type Detector = fn(&Mat<u8>, usize, &mut PreparedFrame) -> Result<FrameLocalization>;

/// The detector registered for `mark` (`scan_video_marks`' table).
pub fn detector_for(mark: &str) -> Option<Detector> {
    Some(match mark {
        "sora" => detect_sora_frame,
        "veo" => detect_veo_frame,
        "dreamina" => detect_dreamina_frame,
        "jimeng" => detect_jimeng_frame,
        "jimeng_pill" => detect_jimeng_pill_frame,
        "seedance" => detect_seedance_frame,
        "doubao" => detect_doubao_frame,
        "dola" => detect_dola_frame,
        "hailuo" => detect_hailuo_frame,
        "vidu" => detect_vidu_frame,
        "wan" => detect_wan_frame,
        "qwen" => detect_qwen_frame,
        "yuanbao" => detect_yuanbao_frame,
        "baidu" => detect_baidu_frame,
        "runninghub" => detect_runninghub_frame,
        "microsoft" => detect_microsoft_frame,
        "pixverse" => detect_pixverse_frame,
        "luma" => detect_luma_frame,
        "heygen" => detect_heygen_frame,
        "sync_labs" => detect_sync_labs_frame,
        "tiktok_ai" => detect_tiktok_ai_frame,
        "generic_ai_label" => detect_generic_ai_label_frame,
        "kling" => detect_kling_frame,
        _ => return None,
    })
}

type Feature = Rc<TemplateSearch<'static, u8>>;

fn cached_u8(
    prep: &mut PreparedFrame,
    key: FeatureKey,
    build: impl FnOnce(&PreparedFrame) -> Result<Mat<u8>>,
) -> Result<Feature> {
    if let Some(found) = prep.u8_cache.get(&key) {
        return Ok(Rc::clone(found));
    }
    let value = Rc::new(TemplateSearch::owned(build(prep)?));
    prep.u8_cache.insert(key, Rc::clone(&value));
    Ok(value)
}

fn top_hat_view(view: &MatView<'_, u8>, kernel: usize) -> Result<Mat<u8>> {
    top_hat(&view.to_mat(), kernel)
}

// Sora

fn expanded_region(
    loc: (i64, i64),
    tw: usize,
    th: usize,
    icon_only: bool,
    scale: f64,
    fw: usize,
    fh: usize,
) -> Region {
    let x = ri(loc.0 as f64 / scale);
    let y = ri(loc.1 as f64 / scale);
    let mut height = ri(th as f64 / scale).max(1);
    let mut width = ri(tw as f64 / scale).max(1);
    if icon_only {
        width = ri(height as f64 * SORA_TEMPLATE_SIZE.0 / SORA_TEMPLATE_SIZE.1);
    }
    width = width.min(fw as i64 - x);
    height = height.min(fh as i64 - y);
    region(x, y, width.max(1), height.max(1))
}

struct PeakSearch {
    max_peaks: usize,
    first_floor: f64,
    later_floor: f64,
    min_nms_radius: i64,
}

/// `_refine_pyramid_peaks`: coarse 2x top-hat peaks refined in a full-resolution ±6 px window.
fn refine_pyramid_peaks(
    feature: &TemplateSearch<'_, u8>,
    feature_half: &TemplateSearch<'_, u8>,
    template_feature: &Mat<u8>,
    template_half: &Mat<u8>,
    search: &PeakSearch,
) -> Result<Vec<(f64, (i64, i64))>> {
    let (tw, th) = (template_feature.width() as i64, template_feature.height() as i64);
    let (tw2, th2) = (template_half.width() as i64, template_half.height() as i64);
    let feature = feature.image();
    let (fw, fh) = (feature.width() as i64, feature.height() as i64);
    let mut coarse = feature_half.match_template(template_half)?;
    let (cw, ch) = (coarse.width() as i64, coarse.height() as i64);
    let mut candidates = Vec::new();
    for peak in 0..search.max_peaks {
        let found = min_max_loc(&coarse)?;
        let floor = if candidates.is_empty() {
            search.first_floor
        } else {
            search.later_floor
        };
        if found.max < floor {
            break;
        }
        let (cx, cy) = (i64::from(found.max_loc.x), i64::from(found.max_loc.y));
        let (fx, fy) = (cx * 2, cy * 2);
        let (rx0, ry0) = ((fx - 6).max(0), (fy - 6).max(0));
        let (rx1, ry1) = ((fx + tw + 7).min(fw), (fy + th + 7).min(fh));
        let sub = slice(feature, rx0, ry0, rx1, ry1);
        if sub.height() as i64 >= th && sub.width() as i64 >= tw {
            let (conf, loc) = best_match(sub, template_feature)?;
            candidates.push((conf, (rx0 + loc.0, ry0 + loc.1)));
        }
        if peak + 1 < search.max_peaks {
            let x0 = (cx - search.min_nms_radius.max(tw2 / 2)).max(0);
            let y0 = (cy - search.min_nms_radius.max(th2 / 2)).max(0);
            let x1 = (cx + (search.min_nms_radius + 1).max(tw2 / 2)).min(cw);
            let y1 = (cy + (search.min_nms_radius + 1).max(th2 / 2)).min(ch);
            let width = coarse.width();
            let data = coarse.data_mut();
            for y in y0..y1 {
                for x in x0..x1 {
                    data[y as usize * width + x as usize] = -2.0;
                }
            }
        }
    }
    Ok(candidates)
}

/// `detect_sora_frame`: the strongest synthetic Sora wordmark or mascot match (full frame).
pub fn detect_sora_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let scale = prep.normalized_scale;
    let (nw, nh) = prep.normalized_gray.size();
    let feature = cached_u8(prep, FeatureKey::FullTopHat, |p| top_hat(&p.normalized_gray, 7))?;
    let feature_half = cached_u8(prep, FeatureKey::FullTopHatHalf, |_| {
        Ok(resize(
            feature.image(),
            (nw / 2).max(1),
            (nh / 2).max(1),
            Interpolation::Area,
        )?)
    })?;
    let mut best = 0.0;
    let mut best_region = None;
    let short = nw.min(nh);
    for (icon_only, key) in [(false, "sora-word"), (true, "sora-icon")] {
        let base = template_source(key)?;
        for rh in SORA_RELATIVE_HEIGHTS {
            let th = ri(short as f64 * rh).max(16) as usize;
            let tw = ri(base.width() as f64 * th as f64 / base.height() as f64).max(1) as usize;
            if th >= nh || tw >= nw {
                continue;
            }
            let pair = resized_template_feature(key, tw, th, 7)?;
            let (template, template_feature) = (&pair.0, &pair.1);
            let (th2, tw2) = ((th / 2).max(4), (tw / 2).max(4));
            let (hw, hh) = (feature_half.image().width(), feature_half.image().height());
            let candidates = if th2 < hh && tw2 < hw && (nh - th) * (nw - tw) >= 2500 {
                let floor = (best - 0.18f64).max(0.36);
                let half = resized_template_half_feature(key, tw, th, 7)?;
                let search = PeakSearch {
                    max_peaks: 3,
                    first_floor: floor,
                    later_floor: floor,
                    min_nms_radius: 4,
                };
                refine_pyramid_peaks(&feature, &feature_half, template_feature, &half, &search)?
            } else {
                vec![best_match_in(&feature, template_feature)?]
            };
            for (conf, loc) in candidates {
                if conf <= best {
                    continue;
                }
                if icon_only {
                    let candidate = crop(&prep.normalized_gray, loc.0, loc.1, tw as i64, th as i64);
                    if single_score(candidate, template)? < SORA_ICON_SHAPE_FLOOR {
                        continue;
                    }
                }
                best = conf;
                best_region = Some(expanded_region(loc, tw, th, icon_only, scale, fw, fh));
            }
        }
    }
    Ok(FrameLocalization {
        frame_index,
        confidence: best,
        region: best_region,
    })
}

// Shared bounded matchers

/// `_match_template`: one synthetic silhouette inside a bounded region of `gray`.
fn match_in_region(
    prep: &mut PreparedFrame,
    use_normalized: bool,
    template: &Mat<u8>,
    template_feature: &Mat<u8>,
    search: Region,
    kernel: usize,
    tag: &'static str,
) -> Result<(f64, Option<Region>)> {
    let (x, y, w, h) = region_parts(search);
    let gray = if use_normalized {
        &prep.normalized_gray
    } else {
        &prep.gray
    };
    let roi = crop(gray, x, y, w, h);
    let (tw, th) = template.size();
    if view_size(&roi) == 0 || th >= roi.height() || tw >= roi.width() {
        return Ok((0.0, None));
    }
    let key = FeatureKey::RoiTopHat {
        tag,
        region: search,
        kernel,
    };
    let feature = if let Some(found) = prep.u8_cache.get(&key) {
        Rc::clone(found)
    } else {
        let built = Rc::new(TemplateSearch::owned(top_hat_view(&roi, kernel)?));
        prep.u8_cache.insert(key, Rc::clone(&built));
        built
    };
    let (conf, loc) = best_match_in(&feature, template_feature)?;
    let patch = crop(feature.image(), loc.0, loc.1, tw as i64, th as i64);
    if view_size(&patch) == 0 || view_max(&patch) < 8 {
        return Ok((0.0, None));
    }
    Ok((conf, Some(region(x + loc.0, y + loc.1, tw as i64, th as i64))))
}

struct FixedMark<'a> {
    key: &'static str,
    relative_heights: &'a [f64],
    origin: (f64, f64),
    kernel_fraction: f64,
    prefer_larger_within: f64,
}

/// `_detect_fixed_mark`: one fixed silhouette over a normalized-frame search region.
fn detect_fixed_mark(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
    spec: &FixedMark<'_>,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let scale = prep.normalized_scale;
    let (nw, nh) = prep.normalized_gray.size();
    let short = nw.min(nh) as f64;
    let sx = ri(nw as f64 * spec.origin.0);
    let sy = ri(nh as f64 * spec.origin.1);
    let search = region(sx, sy, nw as i64 - sx, nh as i64 - sy);
    let base = template_source(spec.key)?;
    let mut matches: Vec<(f64, Region)> = Vec::new();
    for &rh in spec.relative_heights {
        let th = ri(short * rh).max(6) as usize;
        let tw = ri(base.width() as f64 * th as f64 / base.height() as f64).max(1) as usize;
        let kernel = odd_kernel(th as f64 * spec.kernel_fraction, 3);
        let pair = resized_template_feature(spec.key, tw, th, kernel)?;
        let (conf, candidate) = match_in_region(prep, true, &pair.0, &pair.1, search, kernel, "norm")?;
        if let Some(candidate) = candidate
            && conf > 0.0
        {
            matches.push((conf, candidate));
        }
    }
    let Some(mut best) = first_max_by(&matches, |m| m.0) else {
        return Ok(FrameLocalization::empty(frame_index));
    };
    if spec.prefer_larger_within > 0.0 {
        let floor = best.0 - spec.prefer_larger_within;
        let eligible: Vec<(f64, Region)> = matches.iter().copied().filter(|m| m.0 >= floor).collect();
        if let Some(larger) = first_max_by_key(&eligible, |m| (i64::from(m.1.w) * i64::from(m.1.h), m.0)) {
            best = larger;
        }
    }
    Ok(FrameLocalization {
        frame_index,
        confidence: best.0,
        region: Some(restore_region(best.1, scale, fw, fh)),
    })
}

fn first_max_by<T: Copy>(items: &[T], key: impl Fn(&T) -> f64) -> Option<T> {
    let mut best: Option<T> = None;
    for item in items {
        if best.as_ref().is_none_or(|b| key(item) > key(b)) {
            best = Some(*item);
        }
    }
    best
}

fn first_max_by_key<T: Copy>(items: &[T], key: impl Fn(&T) -> (i64, f64)) -> Option<T> {
    let mut best: Option<T> = None;
    for item in items {
        if best.as_ref().is_none_or(|b| {
            let (a, k) = (key(item), key(b));
            a.0 > k.0 || (a.0 == k.0 && a.1 > k.1)
        }) {
            best = Some(*item);
        }
    }
    best
}

// Corner badges

#[derive(Clone, Copy, PartialEq, Eq)]
enum Corner {
    BottomRight,
    TopLeft,
    TopRight,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Basis {
    Short,
    Width,
    Long,
}

struct CornerMark<'a> {
    key: &'static str,
    corner: Corner,
    basis: Basis,
    rel_sizes: &'a [f64],
    max_rx: f64,
    max_ry: f64,
    min_peak: u8,
    rival_keys: &'a [&'static str],
    extend_right_margin: Option<f64>,
    roi_origin_x: Option<f64>,
    kernel_factor: f64,
    min_kernel: i64,
}

impl<'a> CornerMark<'a> {
    const fn new(
        key: &'static str,
        corner: Corner,
        basis: Basis,
        rel_sizes: &'a [f64],
        max_rx: f64,
        max_ry: f64,
    ) -> Self {
        Self {
            key,
            corner,
            basis,
            rel_sizes,
            max_rx,
            max_ry,
            min_peak: 8,
            rival_keys: &[],
            extend_right_margin: None,
            roi_origin_x: None,
            kernel_factor: 0.22,
            min_kernel: 3,
        }
    }
}

fn masked_means(
    patch: &MatView<'_, u8>,
    template: &Mat<u8>,
    fg_above: u8,
    bg_below: u8,
) -> Option<(f64, f64)> {
    let (mut fg, mut fgn, mut bg, mut bgn) = (0.0, 0usize, 0.0, 0usize);
    for y in 0..patch.height().min(template.height()) {
        let (prow, trow) = (patch.row(y), template.row(y));
        for (p, t) in prow.iter().zip(trow) {
            if *t > fg_above {
                fg += f64::from(*p);
                fgn += 1;
            } else if *t < bg_below {
                bg += f64::from(*p);
                bgn += 1;
            }
        }
    }
    (fgn > 0 && bgn > 0).then(|| (fg / fgn as f64, bg / bgn as f64))
}

/// `_detect_corner_asset_mark`: the shared normalized-frame corner matcher.
fn detect_corner_mark(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
    spec: &CornerMark<'_>,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let scale = prep.normalized_scale;
    let (nw, nh) = prep.normalized_gray.size();
    let (nwf, nhf) = (nw as f64, nh as f64);
    let dim = match spec.basis {
        Basis::Short => nw.min(nh),
        Basis::Width => nw,
        Basis::Long => nw.max(nh),
    } as f64;
    let base = template_source(spec.key)?;
    let (corner_tag, sx, sy, roi_rect) = match spec.corner {
        Corner::BottomRight => {
            let (sx, sy) = (ri(nwf * 0.62), ri(nhf * 0.78));
            ("br", sx, sy, (sx, sy, nw as i64, nh as i64))
        }
        Corner::TopLeft => ("tl", 0, 0, (0, 0, ri(nwf * 0.45).max(32), ri(nhf * 0.18).max(20))),
        Corner::TopRight => {
            let sx = ri(nwf * spec.roi_origin_x.unwrap_or(0.72));
            let floor = if spec.roi_origin_x.is_some() { 24 } else { 20 };
            ("tr", sx, 0, (sx, 0, nw as i64, ri(nhf * 0.16).max(floor)))
        }
    };
    let roi = slice(
        &prep.normalized_gray,
        roi_rect.0,
        roi_rect.1,
        roi_rect.2,
        roi_rect.3,
    )
    .to_mat();
    let raw = TemplateSearch::new(&roi);
    let mut local: Vec<(usize, Feature)> = Vec::new();
    let mut best = 0.0;
    let mut best_region = None;
    for &rel in spec.rel_sizes {
        let (tw, th) = if matches!(spec.basis, Basis::Width | Basis::Long) {
            let tw = ri(dim * rel).max(14);
            (
                tw,
                ri(base.height() as f64 * tw as f64 / base.width() as f64).max(6),
            )
        } else {
            let th = ri(dim * rel).max(8);
            (
                ri(base.width() as f64 * th as f64 / base.height() as f64).max(12),
                th,
            )
        };
        if th as usize >= roi.height() || tw as usize >= roi.width() {
            continue;
        }
        let kernel = odd_kernel(th as f64 * spec.kernel_factor, spec.min_kernel);
        let (conf, (lx, ly), patch_feat) = if spec.key == "microsoft" {
            let t_res = resized_raw_template("microsoft", tw as usize, th as usize)?;
            let (conf, loc) = best_match_in(&raw, &*t_res)?;
            let patch = crop(&roi, loc.0, loc.1, tw, th);
            match masked_means(&patch, &t_res, 160, 80) {
                Some((fg, bg)) if fg - bg >= 22.0 => {}
                _ => continue,
            }
            (conf, loc, None)
        } else {
            let pair = resized_template_feature(spec.key, tw as usize, th as usize, kernel)?;
            let key = FeatureKey::CornerTopHat {
                corner: corner_tag,
                sx,
                kernel,
            };
            let feature = if let Some(found) = prep.u8_cache.get(&key) {
                Rc::clone(found)
            } else {
                let built = if let Some((_, f)) = local.iter().find(|(k, _)| *k == kernel) {
                    Rc::clone(f)
                } else {
                    let f = Rc::new(TemplateSearch::owned(top_hat(&roi, kernel)?));
                    local.push((kernel, Rc::clone(&f)));
                    f
                };
                prep.u8_cache.insert(key, Rc::clone(&built));
                built
            };
            let (conf, loc) = best_match_in(&feature, &pair.1)?;
            let patch = crop(feature.image(), loc.0, loc.1, tw, th).to_mat();
            if patch.is_empty() || patch.data().iter().copied().max().unwrap_or(0) < spec.min_peak {
                continue;
            }
            (conf, loc, Some(patch))
        };
        let (x, y) = (sx + lx, sy + ly);
        let rejected = match spec.corner {
            Corner::BottomRight => {
                (nw as i64 - (x + tw)) as f64 > nwf * spec.max_rx
                    || (nh as i64 - (y + th)) as f64 > nhf * spec.max_ry
            }
            Corner::TopLeft => x as f64 > nwf * spec.max_rx || y as f64 > nhf * spec.max_ry,
            Corner::TopRight => {
                (nw as i64 - (x + tw)) as f64 > nwf * spec.max_rx || y as f64 > nhf * spec.max_ry
            }
        };
        if rejected {
            continue;
        }
        if !spec.rival_keys.is_empty()
            && let Some(patch) = &patch_feat
            && conf >= 0.32
            && conf > best
        {
            let mut rival_won = false;
            for rival in spec.rival_keys {
                let pair = resized_template_feature(rival, tw as usize, th as usize, kernel)?;
                if single_score(patch, &pair.1)? >= conf - 0.02 {
                    rival_won = true;
                    break;
                }
            }
            if rival_won {
                continue;
            }
        }
        if conf > best {
            best = conf;
            let mut width = tw;
            if let Some(margin) = spec.extend_right_margin {
                let target = nw as i64 - ri(nwf * margin).max(4);
                if target > x + tw {
                    width = target - x;
                }
            }
            best_region = Some(restore_region(region(x, y, width, th), scale, fw, fh));
        }
    }
    Ok(FrameLocalization {
        frame_index,
        confidence: best,
        region: best_region,
    })
}

pub fn detect_dreamina_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let short = fw.min(fh) as f64;
    let roi_w = (ri(short * 0.12).max(24) as usize).min(fw);
    let roi_h = (ri(short * 0.10).max(20) as usize).min(fh);
    let roi = slice(&prep.gray, 0, 0, roi_w as i64, roi_h as i64).to_mat();
    let base = template_source("dreamina")?;
    let mut features: Vec<(usize, TemplateSearch<'static, u8>)> = Vec::new();
    let mut best = 0.0;
    let mut best_region = None;
    for rel in [0.030, 0.034, 0.038, 0.042, 0.046] {
        let th = ri(short * rel).max(12) as usize;
        let tw = ri(base.width() as f64 * th as f64 / base.height() as f64).max(14) as usize;
        if th >= roi_h || tw >= roi_w {
            continue;
        }
        let kernel = odd_kernel(th as f64 * 0.18, 3);
        let pair = resized_template_feature("dreamina", tw, th, kernel)?;
        if !features.iter().any(|(k, _)| *k == kernel) {
            features.push((kernel, TemplateSearch::owned(top_hat(&roi, kernel)?)));
        }
        let Some((_, feature)) = features.iter().find(|(k, _)| *k == kernel) else {
            continue;
        };
        let (conf, (x, y)) = best_match_in(feature, &pair.1)?;
        let patch = crop(feature.image(), x, y, tw as i64, th as i64);
        if view_size(&patch) == 0 || view_max(&patch) < 6 {
            continue;
        }
        let (lo, hi) = (short * 0.006, short * 0.035);
        if !((lo..=hi).contains(&(x as f64)) && (lo..=hi).contains(&(y as f64))) {
            continue;
        }
        if conf > best {
            best = conf;
            best_region = Some(region(x, y, tw as i64, th as i64));
        }
    }
    Ok(FrameLocalization {
        frame_index,
        confidence: best,
        region: best_region,
    })
}

pub fn detect_wan_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let scale = prep.normalized_scale;
    let (nw, nh) = prep.normalized_gray.size();
    let nshort = nw.min(nh) as f64;
    let (sx, sy) = (ri(nw as f64 * 0.72), ri(nh as f64 * 0.84));
    let roi = slice(&prep.normalized_gray, sx, sy, nw as i64, nh as i64).to_mat();
    let base = template_source("wan")?;
    let mut features: Vec<(usize, TemplateSearch<'static, u8>)> = Vec::new();
    let mut best = 0.0;
    let mut best_region = None;
    for rel in thousandths((36..65).step_by(4)) {
        let th = ri(nshort * rel).max(10) as usize;
        let tw = ri(base.width() as f64 * th as f64 / base.height() as f64).max(14) as usize;
        if th >= roi.height() || tw >= roi.width() {
            continue;
        }
        let kernel = odd_kernel(th as f64 * 0.22, 3);
        let pair = resized_template_feature("wan", tw, th, kernel)?;
        if !features.iter().any(|(k, _)| *k == kernel) {
            features.push((kernel, TemplateSearch::owned(top_hat(&roi, kernel)?)));
        }
        let Some((_, feature)) = features.iter().find(|(k, _)| *k == kernel) else {
            continue;
        };
        let (conf, (lx, ly)) = best_match_in(feature, &pair.1)?;
        let (x, y) = (sx + lx, sy + ly);
        let patch = crop(feature.image(), lx, ly, tw as i64, th as i64);
        if view_size(&patch) == 0 || view_max(&patch) < 8 {
            continue;
        }
        if (nw as i64 - (x + tw as i64)) as f64 > nshort * 0.025
            || (nh as i64 - (y + th as i64)) as f64 > nshort * 0.025
        {
            continue;
        }
        if conf > best {
            best = conf;
            best_region = Some(restore_region(region(x, y, tw as i64, th as i64), scale, fw, fh));
        }
    }
    Ok(FrameLocalization {
        frame_index,
        confidence: best,
        region: best_region,
    })
}

/// `_sobel_norm_u8`.
fn sobel_norm(gray: &Mat<u8>) -> Result<Mat<f32>> {
    let magnitude = sobel_magnitude(gray)?;
    let max = magnitude.data().iter().copied().fold(f32::MIN, f32::max);
    let denom = f64::from(max).max(48.0);
    let factor = (255.0 / denom) as f32;
    Ok(magnitude.map(|v| (v * factor).clamp(0.0, 255.0)))
}

pub fn detect_jimeng_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let mut spec = CornerMark::new(
        "jimeng",
        Corner::BottomRight,
        Basis::Width,
        &[0.16, 0.18, 0.20, 0.22, 0.25],
        0.035,
        0.035,
    );
    spec.rival_keys = &["doubao"];
    detect_corner_mark(image, frame_index, prep, &spec)
}

pub fn detect_jimeng_pill_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let scale = prep.normalized_scale;
    let (nw, nh) = prep.normalized_gray.size();
    let roi_h = ri(nh as f64 * 0.16).max(20);
    let roi_w = ri(nw as f64 * 0.36).max(36);
    let roi = slice(&prep.normalized_gray, 0, 0, roi_w, roi_h).to_mat();
    let gradient = if let Some(found) = prep.f32_cache.get(&FeatureKey::PillSobel) {
        found.clone()
    } else {
        let built = sobel_norm(&roi)?;
        prep.f32_cache.insert(FeatureKey::PillSobel, built.clone());
        built
    };
    let gradient = TemplateSearch::new(&gradient);
    let silhouette = template_source("jimeng_pill")?;
    let mut best = 0.0;
    let mut best_region = None;
    for wf in [0.135, 0.150, 0.161, 0.175, 0.190] {
        let tw = ri(nw as f64 * wf).max(20);
        let th = ri(tw as f64 * silhouette.height() as f64 / silhouette.width() as f64).max(10);
        if th as usize >= roi.height() || tw as usize >= roi.width() {
            continue;
        }
        let t_grad = resized_sobel_template("jimeng_pill", tw as usize, th as usize)?;
        let (conf, (lx, ly)) = best_match_in(&gradient, &*t_grad)?;
        if lx as f64 > nw as f64 * 0.04 || ly as f64 > nh as f64 * 0.04 {
            continue;
        }
        if conf > best {
            best = conf;
            best_region = Some(restore_region(region(lx, ly, tw, th), scale, fw, fh));
        }
    }
    Ok(FrameLocalization {
        frame_index,
        confidence: best,
        region: best_region,
    })
}

pub fn detect_yuanbao_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let spec = CornerMark::new(
        "yuanbao",
        Corner::BottomRight,
        Basis::Short,
        &[0.038, 0.044, 0.050, 0.056],
        0.040,
        0.040,
    );
    detect_corner_mark(image, frame_index, prep, &spec)
}

pub fn detect_qwen_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let mut cn = CornerMark::new(
        "qwen",
        Corner::BottomRight,
        Basis::Short,
        &[0.030, 0.036, 0.042, 0.048, 0.054],
        0.040,
        0.040,
    );
    cn.rival_keys = &["doubao", "kling-1"];
    let cn_loc = detect_corner_mark(image, frame_index, prep, &cn)?;
    let en = CornerMark::new(
        "qwen-en",
        Corner::BottomRight,
        Basis::Short,
        &[0.042, 0.048, 0.054, 0.060, 0.066],
        0.060,
        0.050,
    );
    let en_loc = detect_corner_mark(image, frame_index, prep, &en)?;
    Ok(
        if en_loc.confidence >= 0.48 && en_loc.confidence > cn_loc.confidence {
            en_loc
        } else {
            cn_loc
        },
    )
}

pub fn detect_pixverse_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let mut spec = CornerMark::new(
        "pixverse",
        Corner::TopRight,
        Basis::Short,
        &[0.022, 0.026, 0.030, 0.035, 0.040, 0.046],
        0.080,
        0.070,
    );
    spec.min_peak = 10;
    spec.roi_origin_x = Some(0.60);
    spec.kernel_factor = 0.25;
    detect_corner_mark(image, frame_index, prep, &spec)
}

pub fn detect_luma_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let scale = prep.normalized_scale;
    let (nw, nh) = prep.normalized_gray.size();
    let (nwf, nhf) = (nw as f64, nh as f64);
    let max_luma_w = (nw as i64).min(ri(nwf * 0.12) + ri(nhf * 0.13 * 4.5));
    let roi = slice(
        &prep.normalized_gray,
        0,
        0,
        max_luma_w.max(64),
        ri(nhf * 0.20).max(30),
    )
    .to_mat();
    let base = template_source("luma")?;
    let kernel = odd_kernel(nhf * 0.046, 5);
    let feature = top_hat(&roi, kernel)?;
    let search = TemplateSearch::new(&feature);
    let mut best = 0.0;
    let mut best_region = None;
    for rh in [0.095, 0.103, 0.110, 0.118, 0.126] {
        let th = ri(nhf * rh).max(18);
        let tw = ri(base.width() as f64 * th as f64 / base.height() as f64).max(40);
        if th as usize >= roi.height() || tw as usize >= roi.width() {
            continue;
        }
        let pair = resized_template_feature("luma", tw as usize, th as usize, kernel)?;
        let (conf, (lx, ly)) = best_match_in(&search, &pair.1)?;
        let patch = crop(&feature, lx, ly, tw, th);
        if view_size(&patch) == 0 || view_max(&patch) < 8 {
            continue;
        }
        let (lxf, lyf) = (lx as f64, ly as f64);
        if !(nwf * 0.03 <= lxf && lxf <= nwf * 0.11 && nhf * 0.015 <= lyf && lyf <= nhf * 0.075) {
            continue;
        }
        if conf > best {
            best = conf;
            let ext_w = (nw as i64 - lx).min(ri(tw as f64 * 1.48));
            best_region = Some(restore_region(region(lx, ly, ext_w, th), scale, fw, fh));
        }
    }
    Ok(FrameLocalization {
        frame_index,
        confidence: best,
        region: best_region,
    })
}

pub fn detect_heygen_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let mut spec = CornerMark::new(
        "heygen",
        Corner::BottomRight,
        Basis::Width,
        &[0.16, 0.19, 0.22, 0.25, 0.28],
        0.10,
        0.06,
    );
    spec.min_peak = 12;
    spec.kernel_factor = 0.28;
    spec.min_kernel = 5;
    detect_corner_mark(image, frame_index, prep, &spec)
}

pub fn detect_sync_labs_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let mut spec = CornerMark::new(
        "sync_labs",
        Corner::TopLeft,
        Basis::Short,
        &[0.028, 0.033, 0.038, 0.044, 0.050, 0.056],
        0.090,
        0.090,
    );
    spec.min_peak = 10;
    detect_corner_mark(image, frame_index, prep, &spec)
}

pub fn detect_tiktok_ai_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let scale = prep.normalized_scale;
    let (nw, nh) = prep.normalized_gray.size();
    let (nwf, nhf) = (nw as f64, nh as f64);
    let sy = ri(nhf * 0.86);
    let roi = slice(&prep.normalized_gray, 0, sy, ri(nwf * 0.42).max(48), nh as i64).to_mat();
    let base = template_source("tiktok_ai")?;
    let mut features: Vec<(usize, TemplateSearch<'static, u8>)> = Vec::new();
    let mut best = 0.0;
    let mut best_region = None;
    for rw in [0.14, 0.16, 0.18, 0.20, 0.22, 0.25] {
        let pw = ri(nwf * rw).max(28);
        let tw = ri(pw as f64 * 0.84).max(22);
        let th = ri(base.height() as f64 * tw as f64 / base.width() as f64).max(7);
        if th as usize >= roi.height() || tw as usize >= roi.width() {
            continue;
        }
        let kernel = odd_kernel(th as f64 * 0.30, 3);
        let pair = resized_template_feature("tiktok_ai", tw as usize, th as usize, kernel)?;
        if !features.iter().any(|(k, _)| *k == kernel) {
            features.push((kernel, TemplateSearch::owned(top_hat(&roi, kernel)?)));
        }
        let Some((_, feature)) = features.iter().find(|(k, _)| *k == kernel) else {
            continue;
        };
        let (conf, (lx, ly)) = best_match_in(feature, &pair.1)?;
        let patch = crop(feature.image(), lx, ly, tw, th);
        if view_size(&patch) == 0 || view_max(&patch) < 12 {
            continue;
        }
        let raw_patch = crop(&roi, lx, ly, tw, th);
        match masked_means(&raw_patch, &pair.0, 140, 40) {
            Some((fg, bg)) if fg - bg >= 12.0 => {}
            _ => continue,
        }
        let (x, y) = (lx, sy + ly);
        if x as f64 > nwf * 0.09 || (nh as i64 - (y + th)) as f64 > nhf * 0.06 {
            continue;
        }
        if conf > best {
            best = conf;
            let pad_x = ri((pw - tw) as f64 * 0.55).max(2);
            let pad_y = ri(th as f64 * 0.48).max(2);
            let (bx, by) = ((x - pad_x).max(0), (y - pad_y).max(0));
            let bw = (nw as i64 - bx).min(tw + 2 * pad_x);
            let bh = (nh as i64 - by).min(th + 2 * pad_y);
            best_region = Some(restore_region(region(bx, by, bw, bh), scale, fw, fh));
        }
    }
    Ok(FrameLocalization {
        frame_index,
        confidence: best,
        region: best_region,
    })
}

pub fn detect_baidu_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let mut spec = CornerMark::new(
        "baidu",
        Corner::BottomRight,
        Basis::Short,
        &[0.040, 0.046, 0.052],
        0.140,
        0.040,
    );
    spec.rival_keys = &["doubao", "qwen"];
    spec.extend_right_margin = Some(0.010);
    detect_corner_mark(image, frame_index, prep, &spec)
}

pub fn detect_runninghub_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let mut spec = CornerMark::new(
        "runninghub",
        Corner::TopLeft,
        Basis::Width,
        &[0.26, 0.29, 0.32, 0.35],
        0.025,
        0.020,
    );
    spec.min_peak = 6;
    detect_corner_mark(image, frame_index, prep, &spec)
}

pub fn detect_microsoft_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let spec = CornerMark::new(
        "microsoft",
        Corner::TopRight,
        Basis::Long,
        &[0.12, 0.136, 0.152, 0.17, 0.19],
        0.025,
        0.025,
    );
    detect_corner_mark(image, frame_index, prep, &spec)
}

pub fn detect_generic_ai_label_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let mut spec = CornerMark::new(
        "generic_ai_label",
        Corner::BottomRight,
        Basis::Width,
        &[0.135, 0.155, 0.172, 0.195, 0.215],
        0.030,
        0.030,
    );
    spec.rival_keys = &["jimeng", "doubao", "yuanbao", "qwen", "baidu"];
    detect_corner_mark(image, frame_index, prep, &spec)
}

pub fn detect_seedance_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let spec = FixedMark {
        key: "seedance",
        relative_heights: &[0.065, 0.075, 0.085, 0.095, 0.105],
        origin: (0.68, 0.72),
        kernel_fraction: 0.12,
        prefer_larger_within: 0.0,
    };
    detect_fixed_mark(image, frame_index, prep, &spec)
}

pub fn detect_doubao_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let heights = thousandths((32..61).step_by(4));
    let spec = FixedMark {
        key: "doubao",
        relative_heights: &heights,
        origin: (0.60, 0.76),
        kernel_fraction: 0.14,
        prefer_larger_within: 0.04,
    };
    detect_fixed_mark(image, frame_index, prep, &spec)
}

pub fn detect_dola_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let heights = thousandths(22..41);
    let spec = FixedMark {
        key: "dola",
        relative_heights: &heights,
        origin: (0.65, 0.85),
        kernel_fraction: 0.50,
        prefer_larger_within: 0.0,
    };
    detect_fixed_mark(image, frame_index, prep, &spec)
}

/// `_white_fraction`: share of bright low-saturation pixels, rejecting inverted light UI cards.
pub fn white_fraction(roi: &MatView<'_, u8>) -> Result<f64> {
    let size = roi.width() * roi.height();
    if size == 0 {
        return Ok(0.0);
    }
    let (values, white): (Vec<u8>, usize) = if roi.channels() == 1 {
        let v: Vec<u8> = roi.iter().collect();
        let w = v.iter().filter(|x| **x >= 180).count();
        (v, w)
    } else {
        let hsv = cvt_color(roi.to_mat().view(), ColorConversion::BgrToHsv)?;
        let mut v = Vec::with_capacity(size);
        let mut w = 0;
        for px in hsv.data().chunks_exact(3) {
            v.push(px[2]);
            if px[1] <= 55 && px[2] >= 180 {
                w += 1;
            }
        }
        (v, w)
    };
    let bright = values.iter().filter(|x| **x >= 180).count();
    if bright * 2 >= size {
        let mut sorted: Vec<f64> = values.iter().map(|v| f64::from(*v)).collect();
        let median = crate::py::median(&mut sorted);
        let mean = values.iter().map(|v| f64::from(*v)).sum::<f64>() / size as f64;
        if median >= 180.0 && mean < median {
            return Ok(0.0);
        }
    }
    Ok(white as f64 / size as f64)
}

fn white_fraction_at(image: &Mat<u8>, r: Region) -> Result<f64> {
    let (x, y, w, h) = region_parts(r);
    white_fraction(&crop(image, x, y, w, h))
}

/// `_reaches_kling_edge`.
fn reaches_kling_edge(r: Region, fw: usize, fh: usize) -> bool {
    let (x, y, w, h) = region_parts(r);
    (x + w) as f64 >= fw as f64 * 0.96 && (y + h) as f64 >= fh as f64 * 0.94
}

pub fn detect_hailuo_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let heights = thousandths((28..56).step_by(3));
    let spec = FixedMark {
        key: "hailuo",
        relative_heights: &heights,
        origin: (0.28, 0.76),
        kernel_fraction: 0.18,
        prefer_larger_within: 0.0,
    };
    let detection = detect_fixed_mark(image, frame_index, prep, &spec)?;
    let Some(found) = detection.region else {
        return Ok(detection);
    };
    if white_fraction_at(image, found)? < HAILUO_MIN_WHITE_FRACTION {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (x, y, w, h) = region_parts(found);
    let pad = ri(h as f64 * 1.25);
    let bounded = bounded_region(x - pad, y, w + pad * 2, h, image.width(), image.height());
    Ok(FrameLocalization {
        frame_index,
        confidence: detection.confidence,
        region: Some(bounded),
    })
}

pub fn detect_vidu_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    let heights = thousandths((34..61).step_by(3));
    let spec = FixedMark {
        key: "vidu",
        relative_heights: &heights,
        origin: (0.70, 0.84),
        kernel_fraction: 0.18,
        prefer_larger_within: 0.0,
    };
    detect_fixed_mark(image, frame_index, prep, &spec)
}

const KLING_KEYS: [&str; 6] = ["kling-0", "kling-1", "kling-2", "kling-3", "kling-4", "kling-5"];

pub fn detect_kling_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let heights = thousandths((24..49).step_by(3));
    let mut expanded = Vec::new();
    for key in KLING_KEYS {
        let spec = FixedMark {
            key,
            relative_heights: &heights,
            origin: (0.64, 0.84),
            kernel_fraction: 0.18,
            prefer_larger_within: 0.06,
        };
        let detection = detect_fixed_mark(image, frame_index, prep, &spec)?;
        let Some(found) = detection.region else { continue };
        let (x, y, w, h) = region_parts(found);
        let (left, right, vertical) = (ri(h as f64 * 1.8), ri(h as f64 * 4.0), ri(h as f64 * 0.4));
        let bounded = bounded_region(x - left, y - vertical, w + left + right, h + vertical * 2, fw, fh);
        expanded.push(FrameLocalization {
            frame_index,
            confidence: detection.confidence,
            region: Some(bounded),
        });
    }
    if expanded.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let edge: Vec<FrameLocalization> = expanded
        .into_iter()
        .filter(|c| c.region.is_some_and(|r| reaches_kling_edge(r, fw, fh)))
        .collect();
    let font = first_max_by(&edge, |c| c.confidence);
    let logo_heights = thousandths((20..61).step_by(3));
    let logo_spec = FixedMark {
        key: "kling-logo",
        relative_heights: &logo_heights,
        origin: (0.62, 0.90),
        kernel_fraction: 0.18,
        prefer_larger_within: 0.03,
    };
    let logo = detect_fixed_mark(image, frame_index, prep, &logo_spec)?;
    let mut logo_candidate = None;
    let mut logo_x = None;
    if let Some(found) = logo.region
        && logo.confidence >= 0.44
    {
        let (lx, ly, _, lh) = region_parts(found);
        logo_x = Some(lx);
        let lh = lh as f64;
        let bounded = bounded_region(
            lx - ri(lh * 0.2),
            ly - ri(lh * 0.25),
            ri(lh * 7.8),
            ri(lh * 1.5),
            fw,
            fh,
        );
        if reaches_kling_edge(bounded, fw, fh) {
            logo_candidate = Some(FrameLocalization {
                frame_index,
                confidence: logo.confidence,
                region: Some(bounded),
            });
        }
    }
    let mut best = font.or(logo_candidate);
    if let (Some(font), Some(logo_c), Some(lx)) = (font, logo_candidate, logo_x)
        && let Some(fr) = font.region
    {
        let (fx, fwid) = (i64::from(fr.x), f64::from(fr.w));
        if fx - ri(fwid * 0.08) <= lx && lx <= fx + ri(fwid * 0.50) {
            best = Some(logo_c);
        }
    }
    let Some(best) = best.filter(|b| b.region.is_some()) else {
        return Ok(FrameLocalization::empty(frame_index));
    };
    if let Some(r) = best.region
        && white_fraction_at(image, r)? < KLING_MIN_WHITE_FRACTION
    {
        return Ok(FrameLocalization::empty(frame_index));
    }
    Ok(best)
}

pub fn detect_veo_frame(
    image: &Mat<u8>,
    frame_index: usize,
    prep: &mut PreparedFrame,
) -> Result<FrameLocalization> {
    if image.is_empty() {
        return Ok(FrameLocalization::empty(frame_index));
    }
    let (fw, fh) = image.size();
    let (fwi, fhi) = (fw as i64, fh as i64);
    let ss = fw.min(fh) as f64 / VEO_REFERENCE_SHORT_SIDE;
    let text_base = template_source("veo-text")?;
    let mut best = 0.0;
    let mut best_region = None;
    let kernel = odd_kernel(7.0 * ss, 3);
    let mut sizes = Vec::new();
    for (base, right, bottom) in VEO_DIAMOND_PROFILES {
        let size = ri(base * ss).max(16);
        sizes.push(size);
        let pair = resized_template_feature("veo-diamond", size as usize, size as usize, kernel)?;
        let ex = ri(fw as f64 - (right + base) * ss);
        let ey = ri(fh as f64 - (bottom + base) * ss);
        let pad = ri(size as f64 * 0.25).max(6);
        let (sx, sy) = ((ex - pad).max(0), (ey - pad).max(0));
        let search = region(
            sx,
            sy,
            (fwi - sx).min(size + pad * 2),
            (fhi - sy).min(size + pad * 2),
        );
        let (conf, candidate) = match_in_region(prep, false, &pair.0, &pair.1, search, kernel, "raw")?;
        if conf > best {
            best = conf;
            best_region = candidate;
        }
    }
    let (cx, cy) = (ri(fw as f64 * 0.65), ri(fh as f64 * 0.65));
    let corner = region(cx, cy, fwi - cx, fhi - cy);
    for size in cpython_int_set_order(&sizes) {
        let pair = resized_template_feature("veo-diamond", size as usize, size as usize, kernel)?;
        let (conf, candidate) = match_in_region(prep, false, &pair.0, &pair.1, corner, kernel, "raw")?;
        if conf >= 0.70 && conf > best {
            best = conf;
            best_region = candidate;
        }
    }
    let trw = fwi.min(ri(180.0 * ss).max(32));
    let trh = fhi.min(ri(120.0 * ss).max(24));
    let text_region = region(fwi - trw, fhi - trh, trw, trh);
    let mut heights: Vec<i64> = (8..22).map(|h| ri(f64::from(h) * ss).max(5)).collect();
    heights.sort_unstable();
    heights.dedup();
    let text_kernel = odd_kernel(3.0 * ss, 3);
    for th in heights {
        let tw = ri(text_base.width() as f64 * th as f64 / text_base.height() as f64).max(1);
        let pair = resized_template_feature("veo-text", tw as usize, th as usize, text_kernel)?;
        let (conf, candidate) =
            match_in_region(prep, false, &pair.0, &pair.1, text_region, text_kernel, "raw")?;
        if conf > best {
            best = conf;
            best_region = candidate;
        }
    }
    Ok(FrameLocalization {
        frame_index,
        confidence: best,
        region: best_region,
    })
}

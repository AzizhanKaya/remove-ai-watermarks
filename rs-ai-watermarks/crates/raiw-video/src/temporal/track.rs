//! Moving-overlay tracking (`track_video_regions`) and outro-bumper detection
//! (`detect_ai_bumper_start`).

use raiw_cv::{
    ColorConversion, Interpolation, LkParams, Mat, Point2f, cvt_color, good_features_to_track, resize,
};

use super::sobel_magnitude;
use crate::error::{Result, VideoError};
use crate::py::{median, median_f32, round_int};
use crate::visible::Region;
use crate::visible::frame::{crop, region};

/// `_match_patch_in_roi`: combined edge and gray NCC (0.65 / 0.35).
fn match_patch(
    roi_gray: &Mat<u8>,
    roi_edge: &Mat<f32>,
    ref_gray: &Mat<f32>,
    ref_edge: &Mat<f32>,
) -> Result<(f64, i64, i64)> {
    let edge = raiw_cv::match_template_ccoeff_normed(roi_edge, ref_edge)?;
    let gray_f: Mat<f32> = roi_gray.convert();
    let gray = raiw_cv::match_template_ccoeff_normed(&gray_f, ref_gray)?;
    let combined: Vec<f32> = edge
        .data()
        .iter()
        .zip(gray.data())
        .map(|(e, g)| 0.65 * if e.is_nan() { 0.0 } else { *e } + 0.35 * if g.is_nan() { 0.0 } else { *g })
        .collect();
    let combined = Mat::from_vec(edge.width(), edge.height(), 1, combined)?;
    let found = raiw_cv::min_max_loc(&combined)?;
    Ok((found.max, i64::from(found.max_loc.x), i64::from(found.max_loc.y)))
}

struct TrackState {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
    ref_gray: Mat<f32>,
    ref_edge: Mat<f32>,
}

fn lk(prev: &Mat<u8>, next: &Mat<u8>, points: &[Point2f]) -> Result<Vec<raiw_cv::LkTrack>> {
    let params = LkParams {
        win_width: 15,
        win_height: 15,
        max_level: 2,
        ..LkParams::default()
    };
    Ok(raiw_cv::calc_optical_flow_pyr_lk(prev, next, points, params)?)
}

/// `track_video_regions`: per-frame boxes for user boxes, via PyrLK plus Sobel/NCC refinement.
pub fn track_video_regions<I>(
    frames: I,
    initial: &[Region],
    start_frame: usize,
    end_frame: Option<usize>,
) -> Result<Vec<Option<Vec<Region>>>>
where
    I: IntoIterator<Item = Result<Mat<u8>>>,
{
    if initial.is_empty() {
        return Err(VideoError::value("initial_regions must be non-empty"));
    }
    let mut results = Vec::new();
    let mut states: Vec<TrackState> = Vec::new();
    let mut previous_gray: Option<Mat<u8>> = None;
    for (index, frame) in frames.into_iter().enumerate() {
        let frame = frame?;
        let (fw, fh) = (frame.width() as i64, frame.height() as i64);
        if index < start_frame || end_frame.is_some_and(|end| index > end) {
            results.push(None);
            continue;
        }
        let gray = cvt_color(&frame, ColorConversion::BgrToGray)?;
        if index == start_frame || states.is_empty() {
            states.clear();
            let mut boxes = Vec::new();
            for r in initial {
                let cx = i64::from(r.x).min(fw - 1).max(0);
                let cy = i64::from(r.y).min(fh - 1).max(0);
                let cw = i64::from(r.w).min(fw - cx).max(1);
                let ch = i64::from(r.h).min(fh - cy).max(1);
                let (px0, py0) = ((cx - 1).max(0), (cy - 1).max(0));
                let (px1, py1) = (fw.min(cx + cw + 1), fh.min(cy + ch + 1));
                let padded_edge = sobel_magnitude(&crop(&gray, px0, py0, px1 - px0, py1 - py0).to_mat())?;
                states.push(TrackState {
                    x: cx,
                    y: cy,
                    w: cw,
                    h: ch,
                    ref_gray: crop(&gray, cx, cy, cw, ch).to_mat().convert(),
                    ref_edge: crop(&padded_edge, cx - px0, cy - py0, cw, ch).to_mat(),
                });
                boxes.push(region(cx, cy, cw, ch));
            }
            previous_gray = Some(gray);
            results.push(Some(boxes));
            continue;
        }
        let mut full_edge: Option<Mat<f32>> = None;
        let mut boxes = Vec::new();
        for state in &mut states {
            let (px, py, cw, ch) = (state.x, state.y, state.w, state.h);
            let (mut dx, mut dy) = (0.0f64, 0.0f64);
            if let Some(prev) = &previous_gray
                && cw >= 6
                && ch >= 6
            {
                let patch = crop(prev, px, py, cw, ch).to_mat();
                let corners = good_features_to_track(&patch, 32, 0.05, 3.0, None)?;
                if corners.len() >= 3 {
                    let absolute: Vec<Point2f> = corners
                        .iter()
                        .map(|p| Point2f::new(p.x + px as f32, p.y + py as f32))
                        .collect();
                    let forward = lk(prev, &gray, &absolute)?;
                    let moved: Vec<Point2f> = forward.iter().map(|t| t.point).collect();
                    let backward = lk(&gray, prev, &moved)?;
                    let mut flow_x = Vec::new();
                    let mut flow_y = Vec::new();
                    for ((p0, f), b) in absolute.iter().zip(&forward).zip(&backward) {
                        let (ex, ey) = (p0.x - b.point.x, p0.y - b.point.y);
                        if f.found && b.found && (ex * ex + ey * ey).sqrt() < 1.5 {
                            flow_x.push(f.point.x - p0.x);
                            flow_y.push(f.point.y - p0.y);
                        }
                    }
                    if flow_x.len() >= 3 {
                        dx = f64::from(median_f32(&mut flow_x));
                        dy = f64::from(median_f32(&mut flow_y));
                    }
                }
            }
            let pred_x = round_int(px as f64 + dx).min(fw - cw).max(0);
            let pred_y = round_int(py as f64 + dy).min(fh - ch).max(0);
            let rad = round_int(cw.max(ch) as f64 * 0.85).max(36);
            let (sx0, sy0) = ((pred_x - rad).max(0), (pred_y - rad).max(0));
            let (sx1, sy1) = (fw.min(pred_x + cw + rad), fh.min(pred_y + ch + rad));
            let (mut nx, mut ny) = (pred_x, pred_y);
            let mut best = 0.0;
            if sx1 - sx0 >= cw && sy1 - sy0 >= ch && cw >= 4 && ch >= 4 {
                let (ex0, ey0) = ((sx0 - 1).max(0), (sy0 - 1).max(0));
                let (ex1, ey1) = (fw.min(sx1 + 1), fh.min(sy1 + 1));
                let local_padded = sobel_magnitude(&crop(&gray, ex0, ey0, ex1 - ex0, ey1 - ey0).to_mat())?;
                let local_edge = crop(&local_padded, sx0 - ex0, sy0 - ey0, sx1 - sx0, sy1 - sy0).to_mat();
                let roi = crop(&gray, sx0, sy0, sx1 - sx0, sy1 - sy0).to_mat();
                let (score, lx, ly) = match_patch(&roi, &local_edge, &state.ref_gray, &state.ref_edge)?;
                best = score;
                (nx, ny) = (sx0 + lx, sy0 + ly);
            }
            if best < 0.42 && fw >= cw && fh >= ch && cw >= 8 && ch >= 8 {
                if full_edge.is_none() {
                    full_edge = Some(sobel_magnitude(&gray)?);
                }
                if let Some(edge) = &full_edge {
                    let (score, gx, gy) = match_patch(&gray, edge, &state.ref_gray, &state.ref_edge)?;
                    if score >= 0.58 {
                        (nx, ny) = (gx, gy);
                    }
                }
            }
            nx = nx.min(fw - cw).max(0);
            ny = ny.min(fh - ch).max(0);
            (state.x, state.y) = (nx, ny);
            boxes.push(region(nx, ny, cw, ch));
        }
        previous_gray = Some(gray);
        results.push(Some(boxes));
    }
    Ok(results)
}

fn mean_abs_diff(a: &[u8], b: &[u8], mask: Option<&[bool]>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if mask.is_none_or(|m| m[i]) {
            total += f64::from((f32::from(*x) - f32::from(*y)).abs());
            count += 1;
        }
    }
    total / count.max(1) as f64
}

fn std_masked(values: &[u8], mask: &[bool]) -> f64 {
    let selected: Vec<f64> = values
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(v, _)| f64::from(*v))
        .collect();
    let mean = selected.iter().sum::<f64>() / selected.len().max(1) as f64;
    (selected.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / selected.len().max(1) as f64).sqrt()
}

/// `detect_ai_bumper_start`: start frame of a trailing static full-screen outro card, if any.
pub fn detect_ai_bumper_start<I>(frames: I, fps: f64) -> Result<Option<usize>>
where
    I: IntoIterator<Item = Result<Mat<u8>>>,
{
    let fps = if fps > 0.0 { fps } else { 24.0 };
    let min_frames = round_int(0.5 * fps).max(6) as usize;
    let max_frames = (round_int(4.0 * fps) as usize).max(min_frames + 1);
    let keep_tail = max_frames + min_frames;
    let mut tail: std::collections::VecDeque<(Vec<u8>, Vec<f32>)> =
        std::collections::VecDeque::with_capacity(keep_tail + 1);
    let mut total = 0usize;
    for frame in frames {
        let frame = frame?;
        let small = resize(&frame, 160, 90, Interpolation::Area)?;
        let gray = cvt_color(&small, ColorConversion::BgrToGray)?.into_vec();
        let mut hist = vec![0.0f32; 64];
        for px in small.data().chunks_exact(3) {
            hist[usize::from(px[0] >> 6) * 16 + usize::from(px[1] >> 6) * 4 + usize::from(px[2] >> 6)] += 1.0;
        }
        let norm = hist
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for v in &mut hist {
                *v = (f64::from(*v) / norm) as f32;
            }
        }
        if tail.len() == keep_tail {
            tail.pop_front();
        }
        tail.push_back((gray, hist));
        total += 1;
    }
    if total < 18.max(min_frames * 2) {
        return Ok(None);
    }
    let offset = total - tail.len();
    let mut border = vec![true; 160 * 90];
    for y in 22..68 {
        for x in 32..128 {
            border[y * 160 + x] = false;
        }
    }
    let lo = min_frames.max(total.saturating_sub(max_frames)) as i64 - offset as i64;
    let hi = (total - min_frames) as i64 - offset as i64;
    if lo > hi || lo < 1 {
        return Ok(None);
    }
    let thumbs: Vec<&Vec<u8>> = tail.iter().map(|(g, _)| g).collect();
    let mut best_cut = None;
    let mut best_jump = 0.0;
    for cut in lo as usize..=hi as usize {
        let (h_prev, h_curr) = (&tail[cut - 1].1, &tail[cut].1);
        let dot: f64 = h_prev
            .iter()
            .zip(h_curr)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |h: &Vec<f32>| h.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        let denom = norm(h_prev) * norm(h_curr);
        let hist_corr = if denom > 1e-8 { dot / denom } else { 1.0 };
        let luma_jump = mean_abs_diff(thumbs[cut], thumbs[cut - 1], None);
        if hist_corr >= 0.65 && luma_jump < 26.0 {
            continue;
        }
        let card = &thumbs[cut..];
        let mut stds: Vec<f64> = card.iter().map(|t| std_masked(t, &border)).collect();
        if median(&mut stds) > 18.0 {
            continue;
        }
        let settle = &card[2.min(card.len() - 1)..];
        if settle.len() >= 2 {
            let mut diffs: Vec<f64> = (1..settle.len())
                .map(|i| mean_abs_diff(settle[i], settle[i - 1], Some(&border)))
                .collect();
            if median(&mut diffs) > 2.5 {
                continue;
            }
        }
        let pre = &thumbs[cut.saturating_sub(min_frames)..cut];
        let mut pre_stds: Vec<f64> = pre.iter().map(|t| std_masked(t, &border)).collect();
        let pre_std = median(&mut pre_stds);
        let pre_motion = if pre.len() >= 2 {
            (1..pre.len())
                .map(|i| mean_abs_diff(pre[i], pre[i - 1], None))
                .sum::<f64>()
                / (pre.len() - 1) as f64
        } else {
            0.0
        };
        let mean = |t: &[u8]| t.iter().map(|v| f64::from(*v)).sum::<f64>() / t.len() as f64;
        if pre_std < 18.0 && pre_motion < 1.5 && (mean(pre[pre.len() - 1]) - mean(card[0])).abs() < 20.0 {
            continue;
        }
        let strength = (1.0 - hist_corr) * 50.0 + luma_jump;
        if strength > best_jump {
            best_jump = strength;
            best_cut = Some(offset + cut);
        }
    }
    Ok(best_cut)
}

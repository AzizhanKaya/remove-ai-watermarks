//! Google Gemini sparkle: multi-scale shape search, false-positive gate, mask (`gemini_engine.py`).

use std::sync::OnceLock;

use raiw_cv::{
    Border, ColorConversion, Interpolation, Mat, MatView, MorphShape, Rect, StructuringElement, cvt_color, dilate,
    magnitude, match_template_ccoeff_normed, mean_std_dev, min_max_loc, resize, sobel,
};
use rayon::prelude::*;
use serde::Serialize;

use super::{EngineDetection, EngineOutput, MarkEngine};
use crate::assets;
use crate::error::Result;
use crate::geometry::{Region, Span};
use crate::np::{median_f32, percentile};
use crate::raster::{Image, Mask, crop_view, paste, to_bgr};

/// Provider size tier selected from the source dimensions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WatermarkSize {
    #[default]
    Small,
    Large,
}

/// Expected provider margins and logo size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WatermarkPosition {
    pub margin_right: i32,
    pub margin_bottom: i32,
    pub logo_size: i32,
}

impl WatermarkPosition {
    pub const fn position(&self, width: i32, height: i32) -> (i32, i32) {
        (width - self.margin_right - self.logo_size, height - self.margin_bottom - self.logo_size)
    }
}

/// The large tier only when both axes exceed 1024.
pub const fn watermark_size(width: usize, height: usize) -> WatermarkSize {
    if width > 1024 && height > 1024 { WatermarkSize::Large } else { WatermarkSize::Small }
}

/// The observed standard placement for the size tier.
pub const fn watermark_config(width: usize, height: usize) -> WatermarkPosition {
    match watermark_size(width, height) {
        WatermarkSize::Large => WatermarkPosition { margin_right: 64, margin_bottom: 64, logo_size: 96 },
        WatermarkSize::Small => WatermarkPosition { margin_right: 32, margin_bottom: 32, logo_size: 48 },
    }
}

/// Detection decision and its component scores (`DetectionResult`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparkleDetection {
    pub detected: bool,
    pub confidence: f64,
    pub region: Region,
    pub size: WatermarkSize,
    pub spatial_score: f64,
    pub gradient_score: f64,
    pub variance_score: f64,
}

impl EngineOutput for SparkleDetection {
    fn into_engine(self) -> EngineDetection {
        EngineDetection::Sparkle(self)
    }

    fn from_engine(detection: &EngineDetection) -> Option<&Self> {
        match detection {
            EngineDetection::Sparkle(d) => Some(d),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    scale: i32,
    x: i32,
    y: i32,
    spatial: f64,
    gradient: f64,
    variance: f64,
}

impl Candidate {
    const fn new(scale: i32, x: i32, y: i32, spatial: f64) -> Self {
        Self { scale, x, y, spatial, gradient: 0.0, variance: 0.0 }
    }

    fn fused(&self) -> f64 {
        if self.spatial < 0.25 {
            return f64::max(0.0, self.spatial * 0.5);
        }
        self.spatial * 0.50 + self.gradient * 0.30 + self.variance * 0.20
    }

    fn overlaps(&self, other: &Self) -> bool {
        let radius = 0.5 * f64::from(self.scale.max(other.scale));
        f64::from((self.x - other.x).abs()) < radius && f64::from((self.y - other.y).abs()) < radius
    }
}

/// The provenance-blind half of sparkle detection.
#[derive(Clone, Debug, Default)]
pub struct SparkleScan {
    source: Option<Image>,
    best: Option<Candidate>,
    base: SparkleDetection,
}

const CORE_ALPHA_FRAC: f64 = 0.8;
const SPARKLE_FP_CONF: f64 = 0.65;
const SPARKLE_FP_MARGIN: f64 = 5.0;
const SPARKLE_FP_GRAD: f64 = 0.55;
const SPARKLE_KEEP_CONF: f64 = 0.52;
const SPARKLE_WHITE_SAT: f64 = 0.20;
const FP_MAX_CORNER_AXIS_FRAC: f64 = 0.25;
const FP_OFF_CORNER_MARGIN_MULT: i32 = 2;
const FP_OFF_CORNER_MIN_SCALE_FRAC: f64 = 0.6;
const CORNER_PROMOTE_NCC: f64 = 0.85;
const CORNER_PROMOTE_FRAC: f64 = 0.20;
const CORNER_PROMOTE_MIN: i32 = 96;
const CORNER_PROMOTE_MAX: i32 = 384;
const SELECT_TOPK: usize = 3;
const MASK_ALPHA: f32 = 0.04;
const MASK_DILATE_FRAC: f64 = 0.18;

/// Project-native detector and mask builder for the white Gemini sparkle.
#[derive(Debug)]
pub struct GeminiEngine {
    alpha_small: Mat<f32>,
    alpha_large: Mat<f32>,
    templates: Vec<(i32, Mat<f32>)>,
}

/// Normalized opacity of a black-background capture: the max channel over 255.
fn alpha_map(capture: &Mat<u8>) -> Mat<f32> {
    let cn = capture.channels();
    let data = capture
        .data()
        .chunks_exact(cn)
        .map(|px| f32::from(px.iter().take(3).copied().max().unwrap_or(0)) / 255.0)
        .collect();
    Mat::from_vec(capture.width(), capture.height(), 1, data).unwrap_or_else(|_| Mat::zeros(0, 0, 1))
}

fn load_capture(name: &'static str, side: usize) -> Result<Mat<u8>> {
    let capture = assets::decode_bgr(name)?;
    if capture.size() != (side, side) {
        return Ok(resize(&capture, side, side, Interpolation::Area)?);
    }
    Ok(capture)
}

/// `cvtColor(BGR2GRAY) / 255` as `float32` (`_gray_float`).
fn gray_float(view: &MatView<'_, u8>) -> Result<Mat<f32>> {
    let gray = if view.channels() >= 3 { cvt_color(view, ColorConversion::BgrToGray)? } else { view.to_mat() };
    Ok(gray.map(|v| f32::from(v) / 255.0))
}

fn edges(gray: &Mat<f32>) -> Result<Mat<f32>> {
    let gx: Mat<f32> = sobel(gray, 1, 0, 3, 1.0, 0.0, Border::Reflect101)?;
    let gy: Mat<f32> = sobel(gray, 0, 1, 3, 1.0, 0.0, Border::Reflect101)?;
    Ok(magnitude(&gx, &gy)?)
}

impl GeminiEngine {
    pub fn new() -> Result<Self> {
        let alpha_small = alpha_map(&load_capture("gemini_bg_48.png", 48)?);
        let alpha_large = alpha_map(&load_capture("gemini_bg_96.png", 96)?);
        let templates = (16..120)
            .step_by(2)
            .map(|side| Ok((side, resize(&alpha_large, side as usize, side as usize, Interpolation::Area)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { alpha_small, alpha_large, templates })
    }

    /// The process-wide engine (its assets and template ladder are immutable).
    pub fn shared() -> Result<&'static Self> {
        static ENGINE: OnceLock<std::result::Result<GeminiEngine, String>> = OnceLock::new();
        ENGINE
            .get_or_init(|| Self::new().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|_| crate::error::VisibleError::Asset("gemini_bg_96.png"))
    }

    pub fn alpha_map(&self, size: WatermarkSize) -> &Mat<f32> {
        match size {
            WatermarkSize::Small => &self.alpha_small,
            WatermarkSize::Large => &self.alpha_large,
        }
    }

    /// The large alpha resized to `size` (linear up, area down).
    pub fn interpolated_alpha(&self, size: i32) -> Result<Mat<f32>> {
        let native = self.alpha_large.width() as i32;
        if size == native {
            return Ok(self.alpha_large.clone());
        }
        let method = if size > native { Interpolation::Linear } else { Interpolation::Area };
        Ok(resize(&self.alpha_large, size.max(1) as usize, size.max(1) as usize, method)?)
    }

    /// The strongest normalized match at every usable scale, in ladder order.
    fn scan_scales(&self, gray: &Mat<f32>) -> Result<Vec<(i32, f64, (i32, i32))>> {
        let (h, w) = (gray.height() as i32, gray.width() as i32);
        self.templates
            .par_iter()
            .filter(|(side, _)| *side <= h && *side <= w)
            .map(|(side, tmpl)| {
                let response = match_template_ccoeff_normed(gray, tmpl)?;
                let mm = min_max_loc(&response)?;
                Ok((*side, mm.max, (mm.max_loc.x, mm.max_loc.y)))
            })
            .collect()
    }

    fn global_candidates(&self, image: &Image) -> Result<Vec<Candidate>> {
        let (h, w) = (image.height() as i32, image.width() as i32);
        let side = h.min(w).min(512);
        let (ox, oy) = (w - side, h - side);
        let gray = gray_float(&crop_view(image, Span::new(ox, oy, w, h)))?;
        let mut ranked: Vec<(f64, Candidate)> = self
            .scan_scales(&gray)?
            .into_iter()
            .map(|(s, score, (lx, ly))| {
                (score * f64::min(1.0, (f64::from(s) / 96.0).powf(0.5)), Candidate::new(s, ox + lx, oy + ly, score))
            })
            .collect();
        ranked.sort_by(|a, b| {
            let key = |(wt, c): &(f64, Candidate)| (*wt, c.scale, c.spatial, c.x, c.y);
            let (ka, kb) = (key(a), key(b));
            kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut selected: Vec<Candidate> = Vec::new();
        for (_, c) in ranked {
            if selected.iter().any(|p| c.overlaps(p)) {
                continue;
            }
            selected.push(c);
            if selected.len() == SELECT_TOPK {
                break;
            }
        }
        Ok(selected)
    }

    fn corner_promote(&self, image: &Image, current_raw: f64) -> Result<Option<Candidate>> {
        let (h, w) = (image.height() as i32, image.width() as i32);
        let short = h.min(w);
        let desired = (f64::from(short) * CORNER_PROMOTE_FRAC).round_ties_even() as i32;
        let side = short.min(CORNER_PROMOTE_MIN.max(CORNER_PROMOTE_MAX.min(desired)));
        let (ox, oy) = (w - side, h - side);
        let gray = gray_float(&crop_view(image, Span::new(ox, oy, w, h)))?;
        let mut best: Option<(i32, f64, (i32, i32))> = None;
        for m in self.scan_scales(&gray)? {
            if best.is_none_or(|b| m.1 > b.1) {
                best = Some(m);
            }
        }
        Ok(best
            .filter(|b| b.1 >= CORNER_PROMOTE_NCC && b.1 > current_raw)
            .map(|(s, score, (lx, ly))| Candidate::new(s, ox + lx, oy + ly, score)))
    }

    fn score_candidate(&self, image: &Image, c: Candidate) -> Result<Candidate> {
        if c.spatial < 0.25 {
            return Ok(c);
        }
        let (gradient, variance) = self.grad_var_scores(image, c.scale, c.x, c.y)?;
        Ok(Candidate { gradient, variance, ..c })
    }

    fn grad_var_scores(&self, image: &Image, scale: i32, x: i32, y: i32) -> Result<(f64, f64)> {
        let (h, w) = (image.height() as i32, image.width() as i32);
        let (x2, y2) = (w.min(x + scale), h.min(y + scale));
        let region = crop_view(image, Span::new(x, y, x2, y2));
        let gray = gray_float(&region)?;
        let alpha = self.interpolated_alpha(scale)?.crop(Rect::new(0, 0, (x2 - x) as usize, (y2 - y) as usize))?;
        let response = match_template_ccoeff_normed(&edges(&gray)?, &edges(&alpha)?)?;
        let gradient = min_max_loc(&response)?.max;
        let mut variance = 0.0;
        let ref_h = y.min(scale);
        if ref_h > 8 {
            let reference = crop_view(image, Span::new(x, y - ref_h, x2, y));
            let ref_gray = if reference.channels() >= 3 {
                cvt_color(&reference, ColorConversion::BgrToGray)?
            } else {
                reference.to_mat()
            };
            let region_u8 = gray.map(|v| (v * 255.0) as u8);
            let (_, region_std) = mean_std_dev(&region_u8)?;
            let (_, ref_std) = mean_std_dev(&ref_gray)?;
            if ref_std[0] > 5.0 {
                variance = (1.0 - region_std[0] / ref_std[0]).clamp(0.0, 1.0);
            }
        }
        Ok((gradient, variance))
    }

    /// Everything in detection that does not depend on the trust level.
    pub fn sparkle_scan(&self, image: &Image, force_size: Option<WatermarkSize>) -> Result<SparkleScan> {
        if image.is_empty() {
            return Ok(SparkleScan::default());
        }
        let source = to_bgr(image)?.into_owned();
        let size = force_size.unwrap_or(watermark_size(source.width(), source.height()));
        let mut candidates = self.global_candidates(&source)?;
        let first = candidates.first().map_or(-1.0, |c| c.spatial);
        if let Some(promoted) = self.corner_promote(&source, first)? {
            candidates.push(promoted);
        }
        let mut base = SparkleDetection { size, ..SparkleDetection::default() };
        let mut best: Option<Candidate> = None;
        for c in candidates {
            let scored = self.score_candidate(&source, c)?;
            if best.is_none_or(|b| scored.fused() > b.fused()) {
                best = Some(scored);
            }
        }
        let Some(best) = best else { return Ok(SparkleScan { source: None, best: None, base }) };
        base.region = Region::new(best.x, best.y, best.scale, best.scale);
        base.spatial_score = best.spatial;
        base.gradient_score = best.gradient;
        base.variance_score = best.variance;
        Ok(SparkleScan { source: Some(source), best: Some(best), base })
    }

    /// The trust-level-dependent tail: the false-positive gate unless provenance is trusted.
    pub fn sparkle_verdict(&self, scan: &SparkleScan, trust_provenance: bool) -> Result<SparkleDetection> {
        let mut result = scan.base.clone();
        let (Some(best), Some(source)) = (scan.best, scan.source.as_ref()) else { return Ok(result) };
        let mut confidence = best.fused();
        if best.spatial >= 0.25 && confidence < SPARKLE_FP_CONF && !trust_provenance {
            confidence = self.false_positive_gate(source, best, confidence)?;
        }
        result.confidence = confidence.clamp(0.0, 1.0);
        result.detected = result.confidence >= 0.35;
        Ok(result)
    }

    pub fn detect_watermark(&self, image: &Image, trust_provenance: bool) -> Result<SparkleDetection> {
        let scan = self.sparkle_scan(image, None)?;
        self.sparkle_verdict(&scan, trust_provenance)
    }

    pub fn detect_watermark_both(&self, image: &Image) -> Result<(SparkleDetection, SparkleDetection)> {
        let scan = self.sparkle_scan(image, None)?;
        Ok((self.sparkle_verdict(&scan, false)?, self.sparkle_verdict(&scan, true)?))
    }

    fn corner_geometry(width: i32, height: i32, c: &Candidate) -> (bool, bool) {
        let max_margin = (width - c.x - c.scale).max(height - c.y - c.scale);
        if f64::from(max_margin) > f64::from(width.min(height)) * FP_MAX_CORNER_AXIS_FRAC {
            return (false, true);
        }
        let config = watermark_config(width as usize, height as usize);
        let off_corner = max_margin > config.margin_right * FP_OFF_CORNER_MARGIN_MULT;
        if off_corner && f64::from(c.scale) < f64::from(config.logo_size) * FP_OFF_CORNER_MIN_SCALE_FRAC {
            return (false, true);
        }
        (true, off_corner)
    }

    fn false_positive_gate(&self, image: &Image, c: Candidate, confidence: f64) -> Result<f64> {
        let (plausible, off_corner) = Self::corner_geometry(image.width() as i32, image.height() as i32, &c);
        if !plausible {
            return Ok(confidence.min(0.30));
        }
        let alpha = self.interpolated_alpha(c.scale)?;
        let margin = self.core_ring_margin(image, &alpha, c.x, c.y)?;
        let low_margin = margin.is_some_and(|m| m < SPARKLE_FP_MARGIN);
        let low_gradient = c.gradient < SPARKLE_FP_GRAD;
        if !low_margin && !low_gradient {
            return Ok(confidence);
        }
        if !low_margin
            && confidence >= SPARKLE_KEEP_CONF
            && !off_corner
            && self.core_saturation(image, &alpha, c.x, c.y)?.is_some_and(|s| f64::from(s) <= SPARKLE_WHITE_SAT)
        {
            return Ok(confidence);
        }
        Ok(confidence.min(0.30))
    }

    /// The alpha clipped to the frame and its `(y1, y2, x1, x2)` placement.
    fn footprint_indices(alpha: &Mat<f32>, x: i32, y: i32, width: i32, height: i32) -> Option<(Mat<f32>, [i32; 4])> {
        let (ah, aw) = (alpha.height() as i32, alpha.width() as i32);
        let (x1, y1) = (x.max(0), y.max(0));
        let (x2, y2) = (width.min(x + aw), height.min(y + ah));
        if x1 >= x2 || y1 >= y2 {
            return None;
        }
        let clipped = alpha
            .crop(Rect::new((x1 - x) as usize, (y1 - y) as usize, (x2 - x1) as usize, (y2 - y1) as usize))
            .ok()?;
        Some((clipped, [y1, y2, x1, x2]))
    }

    /// `(core, bounds, peak)` of the alpha's bright core placed at `(x, y)`.
    fn core_mask(&self, image: &Image, alpha: &Mat<f32>, x: i32, y: i32) -> Option<(Vec<bool>, [i32; 4], f32)> {
        let (clipped, bounds) = Self::footprint_indices(alpha, x, y, image.width() as i32, image.height() as i32)?;
        let peak = clipped.data().iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if f64::from(peak) < 0.2 {
            return None;
        }
        let thr = (f64::from(peak) * CORE_ALPHA_FRAC) as f32;
        let core: Vec<bool> = clipped.data().iter().map(|&v| v >= thr).collect();
        core.iter().any(|&c| c).then_some((core, bounds, peak))
    }

    fn core_ring_margin(&self, image: &Image, alpha: &Mat<f32>, x: i32, y: i32) -> Result<Option<f64>> {
        let Some((core, [y1, y2, x1, x2], _)) = self.core_mask(image, alpha, x, y) else { return Ok(None) };
        let (h, w) = (image.height() as i32, image.width() as i32);
        let padding = (f64::from(x2 - x1) * 0.7) as i32;
        let (ry1, ry2) = (0.max(y1 - padding), h.min(y2 + padding));
        let (rx1, rx2) = (0.max(x1 - padding), w.min(x2 + padding));
        let view = crop_view(image, Span::new(rx1, ry1, rx2, ry2));
        let cn = view.channels();
        let luminance = |px: &[u8]| -> f32 { px.iter().map(|&v| f32::from(v)).sum::<f32>() / cn as f32 };
        let (cw, inner_w) = (view.width() as i32, x2 - x1);
        let mut core_values = Vec::new();
        let mut background = Vec::new();
        for ry in 0..view.height() as i32 {
            let row = view.row(ry as usize);
            for rx in 0..cw {
                let l = luminance(&row[rx as usize * cn..(rx as usize + 1) * cn]);
                let (ax, ay) = (rx1 + rx, ry1 + ry);
                if ax >= x1 && ax < x2 && ay >= y1 && ay < y2 {
                    if core[((ay - y1) * inner_w + (ax - x1)) as usize] {
                        core_values.push(f64::from(l));
                    }
                } else {
                    background.push(l);
                }
            }
        }
        if background.len() < 10 {
            return Ok(None);
        }
        let core_value = f64::from(percentile(&mut core_values, 75.0) as f32);
        Ok(Some(core_value - f64::from(median_f32(&mut background))))
    }

    fn core_saturation(&self, image: &Image, alpha: &Mat<f32>, x: i32, y: i32) -> Result<Option<f32>> {
        let Some((core, [y1, y2, x1, x2], _)) = self.core_mask(image, alpha, x, y) else { return Ok(None) };
        let view = crop_view(image, Span::new(x1, y1, x2, y2));
        let cn = view.channels();
        let mut values = Vec::new();
        for yy in 0..view.height() {
            let row = view.row(yy);
            for xx in 0..view.width() {
                if !core[yy * view.width() + xx] {
                    continue;
                }
                let px = &row[xx * cn..(xx + 1) * cn];
                let bright = px.iter().map(|&v| f32::from(v)).fold(f32::NEG_INFINITY, f32::max);
                let dark = px.iter().map(|&v| f32::from(v)).fold(f32::INFINITY, f32::min);
                values.push((bright - dark) / (bright + 1.0));
            }
        }
        Ok(Some(median_f32(&mut values)))
    }

    /// Full-frame mask from a resolved `region`, or a strict detection, or the standard slot under `force`.
    pub fn footprint_mask(&self, image: &Image, force: bool, dilate_radius: Option<i32>, region: Option<Region>) -> Result<Option<Mask>> {
        if image.is_empty() {
            return Ok(None);
        }
        let source = to_bgr(image)?;
        let (h, w) = (source.height() as i32, source.width() as i32);
        let (x, y, scale) = if let Some(r) = region {
            (r.x, r.y, r.w)
        } else {
            let det = self.detect_watermark(&source, false)?;
            if det.detected {
                (det.region.x, det.region.y, det.region.w)
            } else if force {
                let config = watermark_config(w as usize, h as usize);
                let (px, py) = config.position(w, h);
                (px, py, config.logo_size)
            } else {
                return Ok(None);
            }
        };
        let alpha = self.interpolated_alpha(scale)?;
        let Some((clipped, [y1, _, x1, _])) = Self::footprint_indices(&alpha, x, y, w, h) else { return Ok(None) };
        let silhouette = clipped.map(|v| if v > MASK_ALPHA { 255u8 } else { 0 });
        if !silhouette.data().iter().any(|&v| v != 0) {
            return Ok(None);
        }
        let mut mask = Mask::zeros(w as usize, h as usize, 1);
        paste(&mut mask, x1 as usize, y1 as usize, &silhouette)?;
        let radius = dilate_radius.unwrap_or_else(|| 13.max((f64::from(scale) * MASK_DILATE_FRAC) as i32));
        if radius > 0 {
            let k = 2 * radius as usize + 1;
            mask = dilate(&mask, &StructuringElement::new(MorphShape::Ellipse, k, k), 1)?;
        }
        Ok(Some(mask))
    }
}

/// The local sparkle confidence of a decoded image (`detect_sparkle_confidence`).
pub fn detect_sparkle_confidence(image: &Image) -> Result<f64> {
    Ok(GeminiEngine::shared()?.detect_watermark(image, false)?.confidence)
}

/// The registry's Gemini trust gate (shared with `identify`'s sparkle threshold).
pub const GEMINI_SPARKLE_TRUST_CONF: f64 = 0.5;
/// The provenance-confirmed Gemini gate.
pub const GEMINI_PROVENANCE_MIN_CONF: f64 = 0.42;

/// The Gemini registry row's engine: the sparkle engine plus the registry's trust gate.
#[derive(Debug)]
pub struct GeminiMark {
    engine: &'static GeminiEngine,
}

impl GeminiMark {
    pub fn new() -> Result<Self> {
        Ok(Self { engine: GeminiEngine::shared()? })
    }

    fn gate(detection: SparkleDetection, provenance: bool) -> SparkleDetection {
        let gate = if provenance { GEMINI_PROVENANCE_MIN_CONF } else { GEMINI_SPARKLE_TRUST_CONF };
        SparkleDetection { detected: detection.detected && detection.confidence >= gate, ..detection }
    }
}

impl MarkEngine for GeminiMark {
    type Scan = SparkleScan;
    type Detection = SparkleDetection;

    fn scan(&self, image: &Image) -> Result<SparkleScan> {
        self.engine.sparkle_scan(image, None)
    }

    fn verdict(&self, scan: &SparkleScan, provenance: bool) -> Result<SparkleDetection> {
        Ok(Self::gate(self.engine.sparkle_verdict(scan, provenance)?, provenance))
    }

    fn footprint_mask(
        &self,
        image: &Image,
        force: bool,
        dilate: Option<i32>,
        detection: Option<&SparkleDetection>,
    ) -> Result<Option<Mask>> {
        let region = detection.filter(|d| d.detected).map(|d| d.region);
        self.engine.footprint_mask(image, force, dilate, region)
    }
}

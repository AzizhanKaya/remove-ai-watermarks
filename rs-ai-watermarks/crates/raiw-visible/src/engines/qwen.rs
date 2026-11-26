//! Alibaba Qwen "千问AI生成" text or Qwen Create's three-lobe symbol, bottom-right (`qwen_engine.py`).

use raiw_cv::{
    Border, Interpolation, Mat, MorphShape, Rect, StructuringElement, TemplateSearch, dilate, gaussian_blur,
    min_max_loc, resize,
};

use super::MarkEngine;
use crate::assets;
use crate::error::Result;
use crate::geometry::Region;
use crate::raster::{Image, Mask, crop_view, paste};
use crate::text_mark::{
    Corner, FootprintArgs, Frontend, ScaleBasis, TextMarkConfig, TextMarkDetection, TextMarkEngine, TextMarkScan,
    best_detection,
};
use crate::geometry::Span;

pub const DETECT_NCC_THRESHOLD: f64 = 0.45;

pub const CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Qwen",
    asset: "qwen_alpha.png",
    corner: Corner::Br,
    margin_floor: 4,
    width_frac: 0.231,
    height_frac: 0.074,
    margin_x_frac: 0.0203,
    margin_bottom_frac: 0.0218,
    max_saturation: 55.0,
    logo_min_luma: 150.0,
    tophat_delta: 12.0,
    morph_open_size: 5,
    detect_min_coverage: 0.04,
    detect_ncc_threshold: DETECT_NCC_THRESHOLD,
    frontend: Frontend::Tophat,
    scale_basis: ScaleBasis::Short,
    ladder: &[0.78, 1.27],
    alpha_width_frac: 0.160,
    alpha_height_frac: 0.0416,
    min_gw: 8,
    provenance_ncc_factor: 1.0,
    ..TextMarkConfig::DEFAULTS
};

const SYMBOL_ASSET: &str = "qwen_symbol_alpha.png";
const SYMBOL_SEARCH_FRAC: f64 = 0.11;
const SYMBOL_MARGIN_FRAC: f64 = 0.012;
const SYMBOL_SIZE_FRAC: f64 = 0.03125;
const SYMBOL_LADDER: [f64; 5] = [0.75, 0.875, 1.0, 1.125, 1.25];
const SYMBOL_NCC_THRESHOLD: f64 = 0.65;
const SYMBOL_SHAPE_IOU_THRESHOLD: f64 = 0.70;
const SYMBOL_TEMPLATE_MASK_FRAC: f64 = 0.25;
const SYMBOL_RESPONSE_MASK_FRAC: f64 = 0.35;
const SYMBOL_MIN_SHORT_SIDE: usize = 256;
const SYMBOL_MASK_ALPHA: f64 = 4.0 / 255.0;
const SYMBOL_MASK_DILATE_FRAC: f64 = 0.06;

/// The text engine plus the square symbol scan.
#[derive(Clone, Debug)]
pub struct QwenEngine {
    pub text: TextMarkEngine,
}

impl Default for QwenEngine {
    fn default() -> Self {
        Self { text: TextMarkEngine::new(CONFIG) }
    }
}

/// Overlap of the bright residual with the three-lobe silhouette.
fn symbol_shape_iou(response: &Mat<f32>, template: &Mat<f32>) -> f64 {
    let peak = |m: &Mat<f32>| m.data().iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (tp, rp) = (peak(template), peak(response));
    if response.is_empty() || template.is_empty() || tp <= 0.0 || rp <= 0.0 {
        return 0.0;
    }
    let tt = (f64::from(tp) * SYMBOL_TEMPLATE_MASK_FRAC) as f32;
    let rt = (f64::from(rp) * SYMBOL_RESPONSE_MASK_FRAC) as f32;
    let (mut union, mut inter) = (0usize, 0usize);
    for (t, r) in template.data().iter().zip(response.data()) {
        let (a, b) = (*t > tt, *r > rt);
        union += usize::from(a || b);
        inter += usize::from(a && b);
    }
    if union == 0 { 0.0 } else { inter as f64 / union as f64 }
}

impl QwenEngine {
    /// The strongest symbol-shaped candidate in the lower-right search box.
    pub fn symbol_scan(&self, image: &Image) -> Result<TextMarkDetection> {
        let mut det = TextMarkDetection { symbol: true, ..TextMarkDetection::default() };
        if image.is_empty() {
            return Ok(det);
        }
        let (h, w) = (image.height() as i32, image.width() as i32);
        let base = h.min(w);
        if (base as usize) < SYMBOL_MIN_SHORT_SIDE {
            return Ok(det);
        }
        let side = 80.max((f64::from(base) * SYMBOL_SEARCH_FRAC) as i32);
        let margin = (f64::from(base) * SYMBOL_MARGIN_FRAC) as i32;
        let ox = 0.max(w - margin - side);
        let oy = 0.max(h - margin - side);
        let roi = crop_view(image, Span::new(ox, oy, ox + side, oy + side));
        let cn = roi.channels();
        let mut luma = Mat::<f32>::zeros(roi.width(), roi.height(), 1);
        for y in 0..roi.height() {
            let row = roi.row(y);
            for (x, l) in luma.row_mut(y).iter_mut().enumerate() {
                let px = &row[x * cn..x * cn + cn];
                let (b, g, r) = if cn >= 3 { (px[0], px[1], px[2]) } else { (px[0], px[0], px[0]) };
                *l = (f32::from(b) + f32::from(g) + f32::from(r)) / 3.0;
            }
        }
        let sigma = f64::max(4.0, f64::from(side) * 0.20);
        let bg = gaussian_blur(&luma, (0, 0), sigma, sigma, Border::Reflect101)?;
        let mut response = luma.clone();
        for (r, b) in response.data_mut().iter_mut().zip(bg.data()) {
            *r = (*r - b).max(0.0);
        }
        let alpha = &assets::template(SYMBOL_ASSET)?.alpha;
        let (mut best_score, mut best_region, mut best_iou) = (0.0, Region::ZERO, 0.0);
        let search = TemplateSearch::new(&response);
        for rung in SYMBOL_LADDER {
            let s = 12.max((f64::from(base) * SYMBOL_SIZE_FRAC * rung) as i32);
            if s >= side {
                continue;
            }
            let template = resize(alpha, s as usize, s as usize, Interpolation::Area)?;
            let scores = search.match_template(&template)?;
            let mm = min_max_loc(&scores)?;
            if mm.max <= best_score {
                continue;
            }
            let (lx, ly) = (mm.max_loc.x as usize, mm.max_loc.y as usize);
            let crop = response.crop(Rect::new(lx, ly, s as usize, s as usize))?;
            best_score = mm.max;
            best_region = Region::new(ox + mm.max_loc.x, oy + mm.max_loc.y, s, s);
            best_iou = symbol_shape_iou(&crop, &template);
        }
        det.confidence = best_score;
        det.detected = best_score >= SYMBOL_NCC_THRESHOLD && best_iou >= SYMBOL_SHAPE_IOU_THRESHOLD;
        det.region = best_region;
        Ok(det)
    }

    fn symbol_mask(&self, image: &Image, region: Region, dilate_radius: Option<i32>) -> Result<Mask> {
        let alpha = &assets::template(SYMBOL_ASSET)?.alpha;
        let (sw, sh) = (region.w.max(1) as usize, region.h.max(1) as usize);
        let aligned = resize(alpha, sw, sh, Interpolation::Linear)?;
        let radius = dilate_radius.unwrap_or_else(|| 2.max((f64::from(region.w.max(region.h)) * SYMBOL_MASK_DILATE_FRAC) as i32));
        let r = radius.max(0) as usize;
        let mut local = Mask::zeros(sw + 2 * r, sh + 2 * r, 1);
        let thr = SYMBOL_MASK_ALPHA as f32;
        paste(&mut local, r, r, &aligned.map(|v| if v > thr { 255u8 } else { 0 }))?;
        if radius > 0 {
            let k = 2 * r + 1;
            local = dilate(&local, &StructuringElement::new(MorphShape::Ellipse, k, k), 1)?;
        }
        let (w, h) = (image.width() as i32, image.height() as i32);
        let (tx, ty) = (region.x - radius, region.y - radius);
        let (x1, y1) = (tx.max(0), ty.max(0));
        let (x2, y2) = (w.min(tx + local.width() as i32), h.min(ty + local.height() as i32));
        let mut mask = Mask::zeros(image.width(), image.height(), 1);
        if x2 > x1 && y2 > y1 {
            let part = local.crop(Rect::new((x1 - tx) as usize, (y1 - ty) as usize, (x2 - x1) as usize, (y2 - y1) as usize))?;
            paste(&mut mask, x1 as usize, y1 as usize, &part)?;
        }
        Ok(mask)
    }
}

impl MarkEngine for QwenEngine {
    type Scan = (TextMarkScan, TextMarkDetection);
    type Detection = TextMarkDetection;

    fn scan(&self, image: &Image) -> Result<Self::Scan> {
        Ok((self.text.scan(image)?, self.symbol_scan(image)?))
    }

    fn verdict(&self, (text, symbol): &Self::Scan, provenance: bool) -> Result<TextMarkDetection> {
        let text = self.text.verdict(text, provenance)?;
        Ok(best_detection([text, TextMarkDetection { provenance, ..symbol.clone() }]))
    }

    fn footprint_mask(
        &self,
        image: &Image,
        force: bool,
        dilate: Option<i32>,
        detection: Option<&TextMarkDetection>,
    ) -> Result<Option<Mask>> {
        if image.is_empty() {
            return Ok(None);
        }
        let resolved = match detection {
            Some(d) => d.clone(),
            None => MarkEngine::detect(self, image, false)?,
        };
        if resolved.symbol && resolved.detected {
            return Ok(Some(self.symbol_mask(image, resolved.region, dilate)?));
        }
        let redetect = |img: &Image| MarkEngine::detect(self, img, false);
        self.text.footprint_mask(image, FootprintArgs { force, dilate, detection: Some(&resolved) }, &redetect)
    }
}

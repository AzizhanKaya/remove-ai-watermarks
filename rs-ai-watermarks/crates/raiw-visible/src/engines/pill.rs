//! Top-left "AI生成" pills: Jimeng's outline or solid label (`pill_engine.py`) and LiblibAI's
//! corroboration-only compact pill (`liblib_engine.LibLibPillEngine`).

use raiw_cv::{
    Border, ColorConversion, Interpolation, Mat, MatView, cvt_color, magnitude, match_template_ccoeff_normed,
    min_max_loc, normalize_min_max, resize, sobel,
};

use super::{EngineDetection, EngineOutput, Features, MarkEngine};
use crate::assets;
use crate::eraser::boxes_to_mask;
use crate::error::Result;
use crate::geometry::{Region, Span};
use crate::np::median_f32;
use crate::raster::{Image, Mask, crop_view, view_to_bgr};
use crate::text_mark::{Corner, Frontend, ScaleBasis, TextMarkConfig, TextMarkDetection, TextMarkEngine, TextMarkScan};

const ASSET: &str = "jimeng_pill.png";
const WIDTH_FRAC: f64 = 0.161;
const ROI_W_FRAC: f64 = 0.34;
const ROI_H_FRAC: f64 = 0.14;
pub const DETECT_THRESHOLD: f64 = 0.22;
pub const PROVENANCE_DETECT_THRESHOLD: f64 = 0.195;
const MASK_X0: f64 = 0.012;
const MASK_Y0: f64 = 0.006;
const MASK_W: f64 = 0.205;
const MASK_H: f64 = 0.115;
/// Median gradient magnitude at or below which the footprint is flat enough for an invisible fill.
pub const FLAT_TEXTURE_MAX: f64 = 6.0;

/// The measured solid `AI生成` label.
pub const LABEL_CONFIG: TextMarkConfig = TextMarkConfig {
    name: "Jimeng AI生成 pill label",
    asset: "jimeng_label_alpha.png",
    corner: Corner::Tl,
    margin_floor: 2,
    width_frac: 0.34,
    height_frac: 0.14,
    margin_x_frac: 0.0,
    margin_bottom_frac: 0.0,
    max_saturation: 255.0,
    logo_min_luma: 100.0,
    tophat_delta: 8.0,
    morph_open_size: 3,
    detect_min_coverage: 0.01,
    detect_ncc_threshold: 0.55,
    frontend: Frontend::Contrast,
    scale_basis: ScaleBasis::Width,
    ladder: &[0.95, 1.0, 1.05],
    alpha_width_frac: 0.14,
    alpha_height_frac: 0.14 * 128.0 / 353.0,
    min_gw: 24,
    provenance_ncc_factor: 1.0,
    ..TextMarkConfig::DEFAULTS
};

/// A pill verdict; `region` is the matched pill (or verified label) box.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PillDetection {
    pub detected: bool,
    pub confidence: f64,
    pub region: Region,
    pub verified_label: bool,
}

impl EngineOutput for PillDetection {
    fn into_engine(self) -> EngineDetection {
        EngineDetection::Pill(self)
    }

    fn from_engine(detection: &EngineDetection) -> Option<&Self> {
        match detection {
            EngineDetection::Pill(d) => Some(d),
            _ => None,
        }
    }
}

/// `normalize(magnitude(Sobel x, Sobel y), 0, 255, NORM_MINMAX)`.
fn gradient(gray: &Mat<f32>) -> Result<Mat<f32>> {
    let gx: Mat<f32> = sobel(gray, 1, 0, 3, 1.0, 0.0, Border::Reflect101)?;
    let gy: Mat<f32> = sobel(gray, 0, 1, 3, 1.0, 0.0, Border::Reflect101)?;
    Ok(normalize_min_max(&magnitude(&gx, &gy)?, 0.0, 255.0)?)
}

fn gray_of(view: &MatView<'_, u8>) -> Result<Mat<u8>> {
    Ok(cvt_color(&view_to_bgr(view)?, ColorConversion::BgrToGray)?)
}

/// The outline match plus the optional label scan.
#[derive(Clone, Debug, Default)]
pub struct PillScan {
    matched: Option<(f64, Region)>,
    label: Option<TextMarkScan>,
}

/// Detect the top-left `AI生成` label or its synthetic pill outline.
#[derive(Clone, Debug)]
pub struct PillEngine {
    label: Option<TextMarkEngine>,
}

impl PillEngine {
    pub fn new(include_label: bool) -> Self {
        Self { label: include_label.then(|| TextMarkEngine::new(LABEL_CONFIG)) }
    }

    /// Edge-NCC of the synthetic silhouette against the top-left ROI.
    pub fn match_outline(&self, image: &Image) -> Result<Option<(f64, Region)>> {
        let sil = &assets::template(ASSET)?.gray;
        if image.is_empty() {
            return Ok(None);
        }
        let (h, w) = (image.height() as i32, image.width() as i32);
        if h < 64 || w < 64 {
            return Ok(None);
        }
        let (rh, rw) = ((f64::from(h) * ROI_H_FRAC) as i32, (f64::from(w) * ROI_W_FRAC) as i32);
        let tw = 24.max((WIDTH_FRAC * f64::from(w)) as i32);
        let th = 12.max((f64::from(tw) * sil.height() as f64 / sil.width() as f64) as i32);
        if th >= rh || tw >= rw {
            return Ok(None);
        }
        let roi = gray_of(&crop_view(image, Span::new(0, 0, rw, rh)))?;
        let tmpl = resize(sil, tw as usize, th as usize, Interpolation::Linear)?;
        let scores = match_template_ccoeff_normed(&gradient(&roi.convert())?, &gradient(&tmpl.convert())?)?;
        let mm = min_max_loc(&scores)?;
        Ok(Some((mm.max, Region::new(mm.max_loc.x, mm.max_loc.y, tw, th))))
    }

    fn label_box(label: &TextMarkDetection) -> Option<Region> {
        let b = label.match_box.filter(|_| label.detected)?;
        Some(Region::new(label.region.x + b.x0, label.region.y + b.y0, b.width(), b.height()))
    }

    fn combine(matched: Option<(f64, Region)>, label: &TextMarkDetection, provenance: bool) -> PillDetection {
        let label_box = Self::label_box(label);
        let Some((score, region)) = matched else {
            return PillDetection {
                detected: label.detected,
                confidence: label.confidence,
                region: label_box.unwrap_or(label.region),
                verified_label: label.detected,
            };
        };
        let threshold = if provenance { PROVENANCE_DETECT_THRESHOLD } else { DETECT_THRESHOLD };
        PillDetection {
            detected: score >= threshold || label.detected,
            confidence: score.max(label.confidence),
            region: label_box.unwrap_or(region),
            verified_label: label.detected,
        }
    }

    fn footprint_box(height: usize, width: usize) -> Option<Span> {
        let (w, h) = (width as f64, height as f64);
        let (x0, y0) = ((MASK_X0 * w) as i32, (MASK_Y0 * h) as i32);
        let x1 = (width as i32).min(x0 + (MASK_W * w) as i32);
        let y1 = (height as i32).min(y0 + (MASK_H * w) as i32);
        (x1 > x0 && y1 > y0).then_some(Span::new(x0, y0, x1, y1))
    }

    /// Median gradient magnitude over the fixed top-left footprint at a 220 px width.
    pub fn footprint_texture(&self, image: &Image) -> Result<f64> {
        if image.is_empty() {
            return Ok(0.0);
        }
        let Some(span) = Self::footprint_box(image.height(), image.width()) else { return Ok(0.0) };
        let gray = gray_of(&crop_view(image, span))?;
        let tw = 220usize;
        let th = 1.max((gray.height() as f64 * tw as f64 / gray.width() as f64) as usize);
        let gray: Mat<f32> = resize(&gray, tw, th, Interpolation::Linear)?.convert();
        let gx: Mat<f32> = sobel(&gray, 1, 0, 3, 1.0, 0.0, Border::Reflect101)?;
        let gy: Mat<f32> = sobel(&gray, 0, 1, 3, 1.0, 0.0, Border::Reflect101)?;
        let mut mag = magnitude(&gx, &gy)?.into_vec();
        Ok(f64::from(median_f32(&mut mag)))
    }

    pub fn footprint_is_flat(&self, image: &Image) -> Result<bool> {
        Ok(self.footprint_texture(image)? <= FLAT_TEXTURE_MAX)
    }

    /// The verified label's padded box, else the fixed top-left footprint.
    pub fn footprint_mask(&self, image: &Image, detection: Option<&PillDetection>) -> Result<Option<Mask>> {
        if image.is_empty() {
            return Ok(None);
        }
        let (h, w) = (image.height(), image.width());
        let span = match detection.filter(|d| d.verified_label) {
            Some(d) => {
                let r = d.region;
                let pad_x = (w as f64 * 0.025) as i32;
                let pad_y = (w as f64 * 0.014) as i32;
                Some(Span::new(
                    0.max(r.x - pad_x),
                    0.max(r.y - pad_y),
                    (w as i32).min(r.x + r.w + pad_x),
                    (h as i32).min(r.y + r.h + pad_y),
                ))
            }
            None => Self::footprint_box(h, w),
        };
        let Some(span) = span else { return Ok(None) };
        Ok(Some(boxes_to_mask(h, w, &[span.region()], 0)?))
    }
}

impl MarkEngine for PillEngine {
    type Scan = PillScan;
    type Detection = PillDetection;

    fn scan(&self, image: &Image) -> Result<PillScan> {
        let matched = self.match_outline(image)?;
        let label = self.label.as_ref().map(|l| l.scan(image)).transpose()?;
        Ok(PillScan { matched, label })
    }

    fn verdict(&self, scan: &PillScan, provenance: bool) -> Result<PillDetection> {
        let label = match (&self.label, &scan.label, provenance) {
            (Some(engine), Some(s), true) => engine.verdict(s, false)?,
            _ => TextMarkDetection::default(),
        };
        Ok(Self::combine(scan.matched, &label, provenance))
    }

    fn footprint_mask(
        &self,
        image: &Image,
        _force: bool,
        _dilate: Option<i32>,
        detection: Option<&PillDetection>,
    ) -> Result<Option<Mask>> {
        PillEngine::footprint_mask(self, image, detection)
    }

    fn features(&self, image: &Image) -> Result<Features> {
        let flat = self.footprint_is_flat(image)?;
        Ok(Features::from([("footprint_flat", if flat { 1.0 } else { 0.0 })]))
    }
}

const TOP_LEFT_PILL_MIN_CONFIDENCE: f64 = 0.20;
const TOP_LEFT_PILL_X_FRAC: f64 = 0.015;
const TOP_LEFT_PILL_Y_FRAC: f64 = 0.015;
const TOP_LEFT_PILL_WIDTH_FRAC: f64 = 0.17;
const TOP_LEFT_PILL_HEIGHT_FRAC: f64 = 0.08;

/// LiblibAI's compact top-left pill, actionable only with LiblibAI evidence.
#[derive(Clone, Debug)]
pub struct LibLibPillEngine {
    pill: PillEngine,
}

impl Default for LibLibPillEngine {
    fn default() -> Self {
        Self { pill: PillEngine::new(false) }
    }
}

/// The raw outline verdict and, when its score clears the corroborated gate, flatness.
#[derive(Clone, Debug, Default)]
pub struct LibLibPillScan {
    raw: PillDetection,
    flat: bool,
}

impl LibLibPillEngine {
    fn raw(&self, image: &Image) -> Result<PillDetection> {
        let matched = self.pill.match_outline(image)?;
        Ok(PillEngine::combine(matched, &TextMarkDetection::default(), false))
    }
}

impl MarkEngine for LibLibPillEngine {
    type Scan = LibLibPillScan;
    type Detection = PillDetection;

    fn scan(&self, image: &Image) -> Result<LibLibPillScan> {
        let raw = self.raw(image)?;
        let flat = raw.confidence >= TOP_LEFT_PILL_MIN_CONFIDENCE && self.pill.footprint_is_flat(image)?;
        Ok(LibLibPillScan { raw, flat })
    }

    fn verdict(&self, scan: &LibLibPillScan, provenance: bool) -> Result<PillDetection> {
        Ok(PillDetection {
            detected: provenance && scan.flat,
            confidence: scan.raw.confidence,
            region: scan.raw.region,
            verified_label: false,
        })
    }

    fn footprint_mask(
        &self,
        image: &Image,
        force: bool,
        _dilate: Option<i32>,
        detection: Option<&PillDetection>,
    ) -> Result<Option<Mask>> {
        if image.is_empty() {
            return Ok(None);
        }
        if detection.is_none() && !force && !self.raw(image)?.detected {
            return Ok(None);
        }
        let (h, w) = (image.height(), image.width());
        let (wf, hf) = (w as f64, h as f64);
        let x = (TOP_LEFT_PILL_X_FRAC * wf) as i32;
        let y = (TOP_LEFT_PILL_Y_FRAC * hf) as i32;
        let width = (TOP_LEFT_PILL_WIDTH_FRAC * wf) as i32;
        let height = (TOP_LEFT_PILL_HEIGHT_FRAC * wf) as i32;
        if width <= 0 || height <= 0 {
            return Ok(None);
        }
        Ok(Some(boxes_to_mask(h, w, &[Region::new(x, y, width, height)], 0)?))
    }
}

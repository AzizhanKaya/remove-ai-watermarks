//! Independently gated layouts under one registry key (`TextMarkVariantsMixin`).

use raiw_cv::{ColorConversion, MorphShape, StructuringElement, bitwise_or, cvt_color, dilate};

use super::MarkEngine;
use crate::assets;
use crate::cvx::any_nonzero;
use crate::eraser::boxes_to_mask;
use crate::error::Result;
use crate::geometry::{Region, Span};
use crate::np::percentile;
use crate::raster::{Image, Mask, crop_view, to_bgr, view_to_bgr};
use crate::text_mark::{FootprintArgs, TextMarkDetection, TextMarkEngine, TextMarkScan, best_detection};

/// The vendor-specific removal footprint of a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FamilyMask {
    /// The mixin's dispatch to the winning layout (Yuanbao).
    Layout,
    /// Aligned sparse glyphs, grown over a dark outline stroke when present.
    Doubao,
    /// The pixel footprint unioned with the aligned synthetic core.
    Kling,
    /// Aligned alpha, or a box over the Korean variant's leading sparkle.
    Samsung,
    /// The enclosing beige pill for the text-only layout.
    Microsoft,
}

/// A base layout plus variants, each with its own gate and mask geometry.
#[derive(Clone, Debug)]
pub struct TextMarkFamily {
    pub base: TextMarkEngine,
    pub variants: Vec<TextMarkEngine>,
    pub mask: FamilyMask,
}

const FOOTPRINT_ALPHA_FLOOR: f32 = 0.05;
const FOOTPRINT_DILATE: i32 = 1;

impl TextMarkFamily {
    pub fn new(base: TextMarkEngine, variants: Vec<TextMarkEngine>, mask: FamilyMask) -> Self {
        Self { base, variants, mask }
    }

    fn engines(&self) -> impl Iterator<Item = &TextMarkEngine> {
        std::iter::once(&self.base).chain(&self.variants)
    }

    /// The family's strict `detect` (best strict verdict over every layout).
    pub fn detect_strict(&self, image: &Image) -> Result<TextMarkDetection> {
        MarkEngine::detect(self, image, false)
    }

    fn is_base(&self, engine: &TextMarkEngine) -> bool {
        std::ptr::eq(engine, &self.base)
    }

    /// The layout's own `footprint_mask`, re-detecting through the family for the base layout.
    fn layout_footprint(&self, engine: &TextMarkEngine, image: &Image, args: FootprintArgs<'_>) -> Result<Option<Mask>> {
        if self.is_base(engine) {
            let redetect = |img: &Image| self.detect_strict(img);
            engine.footprint_mask(image, args, &redetect)
        } else {
            let redetect = |img: &Image| engine.detect(img, false);
            engine.footprint_mask(image, args, &redetect)
        }
    }

    /// `TextMarkVariantsMixin.footprint_mask`.
    fn mixin_footprint(&self, image: &Image, args: FootprintArgs<'_>) -> Result<Option<Mask>> {
        if image.is_empty() || args.force {
            return self.layout_footprint(&self.base, image, args);
        }
        let det = match args.detection {
            Some(d) => d.clone(),
            None => self.detect_strict(image)?,
        };
        let engine = self.base.variant_of(&self.variants, &det);
        self.layout_footprint(engine, image, args.unforced().with_detection(&det))
    }

    fn detection_or_detect(&self, image: &Image, args: FootprintArgs<'_>) -> Result<TextMarkDetection> {
        match args.detection {
            Some(d) => Ok(d.clone()),
            None => self.detect_strict(image),
        }
    }

    fn doubao_footprint(&self, image: &Image, args: FootprintArgs<'_>) -> Result<Option<Mask>> {
        if args.force {
            return self.mixin_footprint(image, args);
        }
        if image.is_empty() {
            return Ok(None);
        }
        let image = to_bgr(image)?;
        let det = self.detection_or_detect(&image, args)?;
        let Some(b) = det.match_box.filter(|_| det.detected) else {
            return self.mixin_footprint(&image, args.unforced().with_detection(&det));
        };
        let engine = self.base.variant_of(&self.variants, &det);
        if !self.is_base(engine) {
            return self.layout_footprint(engine, &image, args.unforced().with_detection(&det));
        }
        let alpha = &assets::template(engine.config.asset)?.alpha;
        let radius = args.dilate.map_or(FOOTPRINT_DILATE, |d| d.max(0));
        let base_mask = engine.aligned_alpha_mask(&image, &det, alpha, FOOTPRINT_ALPHA_FLOOR, radius)?;
        if args.dilate.is_some() {
            return Ok(base_mask);
        }
        let Some(base_mask) = base_mask else { return Ok(None) };

        // A dark border stroke around the glyphs grows the mask across the stroke.
        let loc = engine.locate(image.height(), image.width());
        let (h, w) = (image.height() as i32, image.width() as i32);
        let pad = 16;
        let span = Span::new(
            0.max(loc.x + b.x0 - pad),
            0.max(loc.y + b.y0 - pad),
            w.min(loc.x + b.x1 + 1 + pad),
            h.min(loc.y + b.y1 + 1 + pad),
        );
        let m_roi = crop_view(&base_mask, span).to_mat();
        if !any_nonzero(&m_roi) {
            return Ok(Some(base_mask));
        }
        let d_stroke = dilate(&m_roi, &StructuringElement::new(MorphShape::Ellipse, 9, 9), 1)?;
        let d_outer = dilate(&m_roi, &StructuringElement::new(MorphShape::Ellipse, 25, 25), 1)?;
        let gray = cvt_color(&view_to_bgr(&crop_view(&image, span))?, ColorConversion::BgrToGray)?;
        let (mut stroke, mut outer) = (Vec::new(), Vec::new());
        for i in 0..m_roi.data().len() {
            let (m, s, o) = (m_roi.data()[i], d_stroke.data()[i], d_outer.data()[i]);
            if s > 0 && m == 0 {
                stroke.push(f64::from(gray.data()[i]));
            }
            if o > 0 && s == 0 {
                outer.push(f64::from(gray.data()[i]));
            }
        }
        if stroke.len() >= 32 && outer.len() >= 32 {
            let (q10_out, q25_out) = (percentile(&mut outer, 10.0), percentile(&mut outer, 25.0));
            let (q10_str, q25_str) = (percentile(&mut stroke, 10.0), percentile(&mut stroke, 25.0));
            if q10_out - q10_str >= 14.0 && q25_out - q25_str >= 5.0 {
                let mh = f64::from(b.height());
                let stroke_radius = (FOOTPRINT_DILATE + 1).max((0.075 * mh).round_ties_even() as i32);
                return engine.aligned_alpha_mask(&image, &det, alpha, FOOTPRINT_ALPHA_FLOOR, stroke_radius);
            }
        }
        Ok(Some(base_mask))
    }

    fn kling_footprint(&self, image: &Image, args: FootprintArgs<'_>) -> Result<Option<Mask>> {
        if args.force || image.is_empty() {
            let redetect = |img: &Image| self.detect_strict(img);
            return self.base.rect_footprint(image, args, &redetect);
        }
        let det = self.detection_or_detect(image, args)?;
        if !det.detected {
            return Ok(None);
        }
        let engine = self.base.variant_of(&self.variants, &det);
        let unforced = args.unforced().with_detection(&det);
        let legacy = if self.is_base(engine) {
            engine.rect_footprint(image, unforced, &|img: &Image| self.detect_strict(img))?
        } else {
            engine.rect_footprint(image, unforced, &|img: &Image| engine.detect(img, false))?
        };
        let radius = args.dilate.map_or(FOOTPRINT_DILATE, |d| d.max(0));
        let alpha = &assets::template(engine.config.asset)?.alpha;
        let core = engine.aligned_alpha_mask(image, &det, alpha, FOOTPRINT_ALPHA_FLOOR, radius)?;
        Ok(match (legacy, core) {
            (Some(l), Some(c)) => Some(bitwise_or(&l, &c)?),
            (l, None) => l,
            (None, c) => c,
        })
    }

    fn samsung_footprint(&self, image: &Image, args: FootprintArgs<'_>) -> Result<Option<Mask>> {
        if args.force || image.is_empty() {
            return self.mixin_footprint(image, args);
        }
        let image = to_bgr(image)?;
        let det = self.detection_or_detect(&image, args)?;
        let engine = self.base.variant_of(&self.variants, &det);
        let radius = args.dilate.map_or(FOOTPRINT_DILATE, |d| d.max(0));
        if !self.is_base(engine)
            && let Some(b) = det.match_box.filter(|_| det.detected)
        {
            let loc = engine.locate(image.height(), image.width());
            let gh = b.height();
            let lead = (1.3 * f64::from(gh)).round_ties_even() as i32;
            let region = Region::new(0.max(loc.x + b.x0 - lead), loc.y + b.y0, b.width() + lead, gh);
            return Ok(Some(boxes_to_mask(image.height(), image.width(), &[region], radius)?));
        }
        let alpha = &assets::template(engine.config.asset)?.alpha;
        if let Some(aligned) = engine.aligned_alpha_mask(&image, &det, alpha, FOOTPRINT_ALPHA_FLOOR, radius)? {
            return Ok(Some(aligned));
        }
        self.mixin_footprint(&image, args.unforced().with_detection(&det))
    }

    fn microsoft_footprint(&self, image: &Image, args: FootprintArgs<'_>) -> Result<Option<Mask>> {
        if args.force || image.is_empty() {
            return self.mixin_footprint(image, args);
        }
        let det = self.detection_or_detect(image, args)?;
        let engine = self.base.variant_of(&self.variants, &det);
        let found = det.match_box.filter(|_| det.detected);
        let (Some(b), false) = (found, self.is_base(engine)) else {
            return self.mixin_footprint(image, args.unforced().with_detection(&det));
        };
        let loc = engine.locate(image.height(), image.width());
        let w = image.width() as f64;
        let (pad_x, pad_y) = ((w * 0.012) as i32, (w * 0.014) as i32);
        let region = Region::new(
            0.max(loc.x + b.x0 - pad_x),
            0.max(loc.y + b.y0 - pad_y),
            b.width() + 2 * pad_x,
            b.height() + 2 * pad_y,
        );
        let d = args.dilate.map_or(0, |d| d.max(0));
        Ok(Some(boxes_to_mask(image.height(), image.width(), &[region], d)?))
    }

    /// The family's footprint with an explicit dilation.
    pub fn footprint(&self, image: &Image, args: FootprintArgs<'_>) -> Result<Option<Mask>> {
        match self.mask {
            FamilyMask::Layout => self.mixin_footprint(image, args),
            FamilyMask::Doubao => self.doubao_footprint(image, args),
            FamilyMask::Kling => self.kling_footprint(image, args),
            FamilyMask::Samsung => self.samsung_footprint(image, args),
            FamilyMask::Microsoft => self.microsoft_footprint(image, args),
        }
    }
}

impl MarkEngine for TextMarkFamily {
    type Scan = Vec<TextMarkScan>;
    type Detection = TextMarkDetection;

    fn scan(&self, image: &Image) -> Result<Vec<TextMarkScan>> {
        self.engines().map(|e| e.scan(image)).collect()
    }

    fn verdict(&self, scans: &Vec<TextMarkScan>, provenance: bool) -> Result<TextMarkDetection> {
        let verdicts = self.engines().zip(scans).map(|(e, s)| e.verdict(s, provenance)).collect::<Result<Vec<_>>>()?;
        Ok(best_detection(verdicts))
    }

    fn footprint_mask(
        &self,
        image: &Image,
        force: bool,
        dilate: Option<i32>,
        detection: Option<&TextMarkDetection>,
    ) -> Result<Option<Mask>> {
        self.footprint(image, FootprintArgs { force, dilate, detection })
    }
}

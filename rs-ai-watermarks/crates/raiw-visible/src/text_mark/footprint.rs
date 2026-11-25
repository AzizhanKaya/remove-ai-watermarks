//! Removal footprints: the default blob/match-box rectangle, aligned alpha, and solid text boxes.

use raiw_cv::{Interpolation, Mat, StructuringElement, dilate, resize};

use super::{Location, TextMarkDetection, TextMarkEngine};
use crate::assets;
use crate::cvx::bounding_rect;
use crate::eraser::boxes_to_mask;
use crate::error::Result;
use crate::geometry::{MatchBox, Region, Span};
use crate::raster::{Image, Mask, paste, to_bgr};

/// Which rectangle the default footprint fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RectPolicy {
    /// The binary glyph blob, else the detector's match box, else the geometry box under `force`.
    Blob,
    /// The match box extended right to the locate box's corner end (Baidu's white tag).
    BaiduTag,
    /// The match box extended left over the triangle logo (LiblibAI).
    LiblibLogo,
}

/// How a detection becomes a removal mask.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskPolicy {
    /// `TextMarkEngine.footprint_mask`: the filled rectangle of [`RectPolicy`], dilated.
    Rect,
    /// `GlyphTextMarkEngine`: the asset aligned to the match box above a 0.05 alpha floor.
    Glyph,
    /// `BoxTextMarkEngine`: the solid match box padded left/right by fractions of its height.
    Box { pad_left: f64, pad_right: f64 },
    /// Aligned alpha with the rectangle as fallback (generic `AI生成` label).
    Aligned { alpha_floor: f32, dilate: i32 },
}

/// The arguments every `footprint_mask` takes.
#[derive(Clone, Copy, Debug, Default)]
pub struct FootprintArgs<'a> {
    pub force: bool,
    pub dilate: Option<i32>,
    pub detection: Option<&'a TextMarkDetection>,
}

impl<'a> FootprintArgs<'a> {
    pub const fn with_detection(self, detection: &'a TextMarkDetection) -> Self {
        Self { detection: Some(detection), ..self }
    }

    pub const fn unforced(self) -> Self {
        Self { force: false, ..self }
    }
}

/// The polymorphic strict `self.detect(image)` a footprint falls back on.
pub type Redetect<'r> = &'r dyn Fn(&Image) -> Result<TextMarkDetection>;

impl TextMarkEngine {
    /// The mark's own footprint policy.
    pub fn footprint_mask(&self, image: &Image, args: FootprintArgs<'_>, redetect: Redetect<'_>) -> Result<Option<Mask>> {
        match self.mask {
            MaskPolicy::Rect => self.rect_footprint(image, args, redetect),
            MaskPolicy::Glyph => self.glyph_footprint(image, args, redetect),
            MaskPolicy::Box { pad_left, pad_right } => self.box_footprint(image, args, pad_left, pad_right, redetect),
            MaskPolicy::Aligned { alpha_floor, dilate } => {
                if args.force {
                    return self.rect_footprint(image, args, redetect);
                }
                if image.is_empty() {
                    return Ok(None);
                }
                let bgr = to_bgr(image)?;
                let det = match args.detection {
                    Some(d) => d.clone(),
                    None => redetect(&bgr)?,
                };
                if !det.detected || det.match_box.is_none() {
                    return self.rect_footprint(&bgr, args.with_detection(&det), redetect);
                }
                let alpha = &assets::template(self.config.asset)?.alpha;
                let radius = args.dilate.map_or(dilate, |d| d.max(0));
                self.aligned_alpha_mask(&bgr, &det, alpha, alpha_floor, radius)
            }
        }
    }

    /// `TextMarkEngine.footprint_mask`: a solid, dilated rectangle.
    pub fn rect_footprint(&self, image: &Image, args: FootprintArgs<'_>, redetect: Redetect<'_>) -> Result<Option<Mask>> {
        if image.is_empty() {
            return Ok(None);
        }
        let bgr = to_bgr(image)?;
        let (h, w) = (bgr.height(), bgr.width());
        if h < 32 || w < 64 {
            return Ok(None);
        }
        let loc = self.locate(h, w);
        let Some(rect) = self.footprint_rect(&bgr, loc, args, redetect)? else { return Ok(None) };
        if rect.x0 >= rect.x1 || rect.y0 >= rect.y1 {
            return Ok(None);
        }
        let d = args.dilate.unwrap_or_else(|| 3.max((0.02 * f64::from(loc.w)) as i32));
        Ok(Some(boxes_to_mask(h, w, &[rect.region()], d)?))
    }

    /// `GlyphTextMarkEngine.footprint_mask`.
    pub fn glyph_footprint(&self, image: &Image, args: FootprintArgs<'_>, redetect: Redetect<'_>) -> Result<Option<Mask>> {
        if args.force {
            return self.rect_footprint(image, args, redetect);
        }
        if image.is_empty() {
            return Ok(None);
        }
        let det = match args.detection {
            Some(d) => d.clone(),
            None => redetect(image)?,
        };
        if !det.detected {
            return Ok(None);
        }
        let alpha = &assets::template(self.config.asset)?.alpha;
        self.aligned_alpha_mask(image, &det, alpha, 0.05, args.dilate.map_or(2, |d| d.max(0)))
    }

    /// `_box_footprint`: the full winning text rectangle for outlined or stacked layouts.
    pub fn box_footprint(
        &self,
        image: &Image,
        args: FootprintArgs<'_>,
        pad_left: f64,
        pad_right: f64,
        redetect: Redetect<'_>,
    ) -> Result<Option<Mask>> {
        if image.is_empty() || args.force {
            return self.glyph_footprint(image, args, redetect);
        }
        let det = match args.detection {
            Some(d) => d.clone(),
            None => redetect(image)?,
        };
        let Some(b) = det.match_box.filter(|_| det.detected) else { return Ok(None) };
        let loc = self.locate(image.height(), image.width());
        let height = f64::from(b.height());
        let extra_left = (height * pad_left).round_ties_even() as i32;
        let extra_right = (height * pad_right).round_ties_even() as i32;
        let pad = args.dilate.map_or(2, |d| d.max(0));
        let region = Region::new(loc.x + b.x0 - extra_left, loc.y + b.y0, b.width() + extra_left + extra_right, b.height());
        Ok(Some(boxes_to_mask(image.height(), image.width(), &[region], pad)?))
    }

    /// Place an alpha footprint on the detector's winning template box.
    pub fn aligned_alpha_mask(
        &self,
        image: &Image,
        det: &TextMarkDetection,
        alpha: &Mat<f32>,
        alpha_floor: f32,
        dilate_radius: i32,
    ) -> Result<Option<Mask>> {
        let Some(b) = det.match_box.filter(|_| det.detected) else { return Ok(None) };
        let (h, w) = (image.height(), image.width());
        let loc = self.locate(h, w);
        let (x0, y0) = (b.x0.max(0), b.y0.max(0));
        let (x1, y1) = ((loc.w - 1).min(b.x1), (loc.h - 1).min(b.y1));
        if x1 < x0 || y1 < y0 {
            return Ok(None);
        }
        let (gw, gh) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
        let aligned = resize(alpha, gw, gh, Interpolation::Linear)?;
        let mut local = Mask::zeros(loc.w as usize, loc.h as usize, 1);
        paste(&mut local, x0 as usize, y0 as usize, &aligned.map(|v| if v > alpha_floor { 255u8 } else { 0 }))?;
        if dilate_radius > 0 {
            let side = 2 * dilate_radius as usize + 1;
            local = dilate(&local, &StructuringElement::rect(side, side), 1)?;
        }
        let mut mask = Mask::zeros(w, h, 1);
        paste(&mut mask, loc.x as usize, loc.y as usize, &local)?;
        Ok(Some(mask))
    }

    /// The default footprint policy (`_footprint_rect`) or a match-box override.
    fn footprint_rect(
        &self,
        image: &Image,
        loc: Location,
        args: FootprintArgs<'_>,
        redetect: Redetect<'_>,
    ) -> Result<Option<Span>> {
        let frame = (image.height() as i32, image.width() as i32);
        if self.rect != RectPolicy::Blob {
            if args.force {
                return Ok(Some(geometry_rect(loc, frame)));
            }
            let det = self.strict_detection(image, args.detection, redetect)?;
            return Ok(det.match_box.filter(|_| det.detected).map(|b| self.extend_match_box(b, loc, frame)));
        }
        let glyphs = self.extract_mask(image, loc, None)?;
        let on = glyphs.data().iter().filter(|&&v| v > 0).count();
        let blob = (on >= MIN_GLYPH_PIXELS).then(|| {
            let r = bounding_rect(&glyphs);
            MatchBox::new(r.x, r.y, r.x + r.w - 1, r.y + r.h - 1)
        });
        use super::Frontend;
        let found = match self.config.frontend {
            Frontend::Gray | Frontend::Contrast => {
                let det = self.strict_detection(image, args.detection, redetect)?;
                if det.detected { det.match_box } else { blob }
            }
            _ if blob.is_some() => blob,
            Frontend::Tophat => {
                let det = self.strict_detection(image, args.detection, redetect)?;
                det.match_box.filter(|_| det.detected)
            }
            Frontend::Binary => None,
        };
        Ok(match found {
            Some(b) => Some(self.extend_match_box(b, loc, frame)),
            None => args.force.then(|| geometry_rect(loc, frame)),
        })
    }

    /// The strict detection the footprint is bounded by; a relaxed one is re-detected.
    fn strict_detection(
        &self,
        image: &Image,
        detection: Option<&TextMarkDetection>,
        redetect: Redetect<'_>,
    ) -> Result<TextMarkDetection> {
        match detection {
            Some(d) if !d.provenance => Ok(d.clone()),
            _ => redetect(image),
        }
    }

    /// Grow a locate-box-local match box into the absolute fill rectangle.
    fn extend_match_box(&self, b: MatchBox, loc: Location, (h, w): (i32, i32)) -> Span {
        match self.rect {
            RectPolicy::Blob => {
                let pad = 4.max((0.10 * f64::from(loc.h)) as i32);
                Span::new(
                    0.max(loc.x + b.x0 - pad),
                    0.max(loc.y + b.y0 - pad),
                    w.min(loc.x + b.x1 + 1 + pad),
                    h.min(loc.y + b.y1 + 1 + pad),
                )
            }
            RectPolicy::BaiduTag => {
                let pad = 4.max((0.15 * f64::from(loc.h)) as i32);
                Span::new(
                    0.max(loc.x + b.x0 - pad),
                    0.max(loc.y + b.y0 - pad),
                    w.min(loc.x + loc.w),
                    h.min(loc.y + b.y1 + 1 + pad),
                )
            }
            RectPolicy::LiblibLogo => {
                let gh = b.height();
                let pad = 3.max((0.25 * f64::from(gh)) as i32);
                Span::new(
                    0.max(loc.x + b.x0 - (1.3 * f64::from(gh)) as i32),
                    0.max(loc.y + b.y0 - pad),
                    w.min(loc.x + b.x1 + 1 + pad),
                    h.min(loc.y + b.y1 + 1 + pad),
                )
            }
        }
    }
}

/// Minimum glyph pixels for a template-free footprint.
const MIN_GLYPH_PIXELS: usize = 20;

/// The whole locate box clamped to the frame: the `force` footprint.
fn geometry_rect(loc: Location, (h, w): (i32, i32)) -> Span {
    Span::new(loc.x, loc.y, w.min(loc.x + loc.w), h.min(loc.y + loc.h))
}

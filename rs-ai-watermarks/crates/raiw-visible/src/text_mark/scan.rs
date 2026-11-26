//! Detection: locate, front-end response, ladder sweep, verdict and post-gates.

use raiw_cv::{
    Border, ColorConversion, Interpolation, Mat, MatView, MorphOp, StructuringElement, TemplateSearch, cvt_color,
    gaussian_blur, laplacian, match_template_ccoeff_normed, min_max_loc, morphology_ex, resize,
};

use super::{
    Frontend, Located, Location, MIN_DETECT_SHORT_SIDE, TextMarkConfig, TextMarkDetection, TextMarkEngine,
    TextMarkScan, rival_config,
};
use crate::assets;
use crate::error::Result;
use crate::geometry::{MatchBox, Region};
use crate::raster::{Image, region_view, view_to_bgr};

/// How the scale ladder is swept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ladder {
    /// Best match anywhere in the response.
    Standard,
    /// Only placements ending within the anchor window of the bottom-right corner.
    Anchored { max_right: f64, max_bottom: f64 },
    /// Standard, then capped by the row/column-projected glyph detail (Kling).
    GlyphDetail,
}

/// Per-mark demotion applied after the threshold, at both trust levels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PostGate {
    None,
    /// The match must end within these fractions of the short side from both edges.
    BottomRight { max_right: f64, max_bottom: f64 },
    /// The match must start within these fractions of the frame (RunningHub).
    TopLeft { max_x: f64, max_y: f64 },
    /// Bottom-right anchor plus both stacked lines and 2D glyph detail (Yuanbao).
    TwoLine { max_right: f64, max_bottom: f64, min_line: f64, min_detail: f64 },
}

/// A whole-scan precondition or rescan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScanHook {
    None,
    /// Skip the scan below this short side (LiblibAI).
    MinShortSide(usize),
    /// Scan a copy downscaled to this short side and map the result back (Genspark).
    Downscale(usize),
    /// Zero the score unless the winning box holds bright neutral strokes and edge detail.
    Outline { min_bright_neutral: f64, min_mean_laplacian: f64 },
    /// Rescan with a wider saturation filter on colorful corners below the strict gate (Doubao).
    Tinted { max_saturation: f32, min_colored_spread: u8, min_colored_fraction: f64 },
}

/// `(luma, saturation, local_background)` of a locate box, all `float32`.
#[derive(Clone, Debug)]
pub struct Fields {
    pub luma: Mat<f32>,
    pub sat: Mat<f32>,
    pub bg: Mat<f32>,
}

fn template_side(frac: f64, base: usize, scale: f64) -> i32 {
    (frac * base as f64 * scale) as i32
}

fn best_match(resp: &TemplateSearch<'_, u8>, tmpl: &Mat<u8>) -> Result<(f64, i32, i32)> {
    let scores = resp.match_template(tmpl)?;
    let mm = min_max_loc(&scores)?;
    Ok((mm.max, mm.max_loc.x, mm.max_loc.y))
}

/// Zero-mean correlation after removing row and column means (`glyph_detail_ncc`).
pub fn glyph_detail_ncc(patch: &Mat<u8>, template: &Mat<u8>) -> f64 {
    fn centred(m: &Mat<u8>) -> Vec<f64> {
        let (w, h) = (m.width(), m.height());
        let v: Vec<f64> = m.data().iter().map(|&x| f64::from(x)).collect();
        let mut col = vec![0.0; w];
        let mut row = vec![0.0; h];
        for y in 0..h {
            for x in 0..w {
                col[x] += v[y * w + x];
                row[y] += v[y * w + x];
            }
        }
        let total: f64 = row.iter().sum::<f64>() / (w * h).max(1) as f64;
        col.iter_mut().for_each(|c| *c /= h.max(1) as f64);
        row.iter_mut().for_each(|r| *r /= w.max(1) as f64);
        (0..h).flat_map(|y| (0..w).map(move |x| (y, x))).map(|(y, x)| v[y * w + x] - col[x] - row[y] + total).collect()
    }
    if patch.size() != template.size() || patch.is_empty() {
        return 0.0;
    }
    let (p, t) = (centred(patch), centred(template));
    let norm = |a: &[f64]| a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = norm(&p) * norm(&t);
    if denom > 1e-6 { p.iter().zip(&t).map(|(a, b)| a * b).sum::<f64>() / denom } else { 0.0 }
}

/// `cv2.matchTemplate(a, b)[0, 0]` for equal-sized images.
fn ncc_same_size(a: &MatView<'_, u8>, b: &MatView<'_, u8>) -> Result<f64> {
    let scores = match_template_ccoeff_normed(a, b)?;
    Ok(f64::from(scores.at(0, 0, 0)))
}

impl TextMarkEngine {
    /// Compute the locate box's luma, saturation and blurred background.
    pub fn roi_fields(&self, image: &Image, loc: Location) -> Result<Fields> {
        let view = region_view(image, loc.bbox());
        let (w, h, cn) = (view.width(), view.height(), view.channels());
        let mut luma = Mat::<f32>::zeros(w, h, 1);
        let mut sat = Mat::<f32>::zeros(w, h, 1);
        for y in 0..h {
            let row = view.row(y);
            let (lrow, srow) = (luma.row_mut(y), sat.row_mut(y));
            for x in 0..w {
                let px = &row[x * cn..x * cn + cn];
                let (b, g, r) = if cn >= 3 { (px[0], px[1], px[2]) } else { (px[0], px[0], px[0]) };
                let (bf, gf, rf) = (f32::from(b), f32::from(g), f32::from(r));
                lrow[x] = (bf + gf + rf) / 3.0;
                srow[x] = bf.max(gf).max(rf) - bf.min(gf).min(rf);
            }
        }
        let sigma = f64::max(4.0, f64::from(loc.h) * 0.4);
        let bg = gaussian_blur(&luma, (0, 0), sigma, sigma, Border::Reflect101)?;
        Ok(Fields { luma, sat, bg })
    }

    /// Box-sized binary glyph mask: bright, low-saturation strokes above the local background.
    pub fn extract_mask(&self, image: &Image, loc: Location, fields: Option<&Fields>) -> Result<Mat<u8>> {
        let c = &self.config;
        if loc.h < 16 || loc.w < 16 {
            return Ok(Mat::zeros(loc.w.max(0) as usize, loc.h.max(0) as usize, 1));
        }
        let owned;
        let f = match fields {
            Some(f) => f,
            None => {
                owned = self.roi_fields(image, loc)?;
                &owned
            }
        };
        let mut glyph = Mat::<u8>::zeros(f.luma.width(), f.luma.height(), 1);
        for (i, g) in glyph.data_mut().iter_mut().enumerate() {
            let luma = f.luma.data()[i];
            let tophat = luma - f.bg.data()[i];
            if f.sat.data()[i] < c.max_saturation && tophat > c.tophat_delta && luma > c.logo_min_luma {
                *g = 255;
            }
        }
        let glyph = morphology_ex(&glyph, MorphOp::Close, &StructuringElement::rect(5, 5), 1)?;
        let k = c.morph_open_size;
        Ok(morphology_ex(&glyph, MorphOp::Open, &StructuringElement::rect(k, k), 1)?)
    }

    /// Max-normalized local-luma residual, saturation-weighted; `absolute` keeps both polarities.
    pub fn residual_response(
        &self,
        image: &Image,
        loc: Location,
        absolute: bool,
        fields: Option<&Fields>,
    ) -> Result<Option<Mat<u8>>> {
        if loc.h < 16 || loc.w < 16 {
            return Ok(None);
        }
        let owned;
        let f = match fields {
            Some(f) => f,
            None => {
                owned = self.roi_fields(image, loc)?;
                &owned
            }
        };
        let max_sat = self.config.max_saturation;
        let resp: Vec<f32> = (0..f.luma.data().len())
            .map(|i| {
                let residual = f.luma.data()[i] - f.bg.data()[i];
                let v = if absolute { residual.abs() } else { residual.max(0.0) };
                if f.sat.data()[i] < max_sat { v } else { 0.0 }
            })
            .collect();
        let peak = resp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if f64::from(peak) <= 1e-6 {
            return Ok(None);
        }
        let out: Vec<u8> = resp.iter().map(|&v| (v / peak * 255.0) as u8).collect();
        Ok(Some(Mat::from_vec(f.luma.width(), f.luma.height(), 1, out)?))
    }

    /// The continuous white top-hat of the locate box (`tophat_response`).
    pub fn tophat_response(&self, image: &Image, loc: Location) -> Result<Option<Mat<u8>>> {
        self.residual_response(image, loc, false, None)
    }

    /// The uint8 image the ladder sweep correlates against, chosen by the front-end.
    pub fn detect_response(&self, image: &Image, loc: Location, fields: Option<&Fields>) -> Result<Option<Mat<u8>>> {
        match self.config.frontend {
            Frontend::Gray => {
                if loc.h < 16 || loc.w < 16 {
                    return Ok(None);
                }
                let bgr = view_to_bgr(&region_view(image, loc.bbox()))?;
                Ok(Some(cvt_color(&bgr, ColorConversion::BgrToGray)?))
            }
            Frontend::Contrast => self.residual_response(image, loc, true, fields),
            Frontend::Tophat => self.residual_response(image, loc, false, fields),
            Frontend::Binary => Ok(None),
        }
    }

    fn sized_silhouette(&self, gw: i32, gh: i32) -> Result<Mat<u8>> {
        let sil = &assets::template(self.config.asset)?.silhouette;
        let tmpl = resize(sil, gw as usize, gh as usize, Interpolation::Area)?;
        if self.config.frontend == Frontend::Tophat && self.config.template_blur > 0.0 {
            let s = self.config.template_blur;
            let blurred = gaussian_blur(&tmpl.convert::<f32>(), (0, 0), s, s, Border::Reflect101)?;
            return Ok(blurred.map(|v| v as u8));
        }
        Ok(tmpl)
    }

    /// Best `TM_CCOEFF_NORMED` over the ladder and its locate-box-local match box.
    pub fn ladder_best(&self, image: &Image, loc: Location, resp: Option<&Mat<u8>>) -> Result<(f64, Option<MatchBox>)> {
        let Some(resp) = resp else { return Ok((0.0, None)) };
        match self.ladder {
            Ladder::Standard => self.standard_ladder(image, resp),
            Ladder::Anchored { max_right, max_bottom } => self.anchored_ladder(image, loc, resp, max_right, max_bottom),
            Ladder::GlyphDetail => {
                let (score, found) = self.standard_ladder(image, resp)?;
                let Some(b) = found else { return Ok((score, None)) };
                let patch = resp.crop(raiw_cv::Rect::new(b.x0 as usize, b.y0 as usize, b.width() as usize, b.height() as usize))?;
                let sil = &assets::template(self.config.asset)?.silhouette;
                let tmpl = resize(sil, patch.width(), patch.height(), Interpolation::Area)?;
                let detail = glyph_detail_ncc(&patch, &tmpl);
                Ok((score.min(detail).max(0.0), Some(b)))
            }
        }
    }

    fn standard_ladder(&self, image: &Image, resp: &Mat<u8>) -> Result<(f64, Option<MatchBox>)> {
        let c = &self.config;
        let base = self.scale_base(image.height(), image.width());
        let search = TemplateSearch::new(resp);
        let mut best = (0.0, None);
        for &scale in c.ladder {
            let gw = c.min_gw.max(template_side(c.alpha_width_frac, base, scale));
            let gh = 4.max(template_side(c.alpha_height_frac, base, scale));
            if gw as usize >= resp.width() || gh as usize >= resp.height() {
                continue;
            }
            let tmpl = self.sized_silhouette(gw, gh)?;
            let (score, x, y) = best_match(&search, &tmpl)?;
            if score > best.0 {
                best = (score, Some(MatchBox::at(x, y, gw, gh)));
            }
        }
        Ok(best)
    }

    fn anchored_ladder(
        &self,
        image: &Image,
        loc: Location,
        resp: &Mat<u8>,
        max_right: f64,
        max_bottom: f64,
    ) -> Result<(f64, Option<MatchBox>)> {
        let c = &self.config;
        let (frame_h, frame_w) = (image.height() as i32, image.width() as i32);
        let anchor_base = f64::from(frame_h.min(frame_w));
        let base = self.scale_base(image.height(), image.width());
        let search = TemplateSearch::new(resp);
        let mut best = (0.0, None);
        for &scale in c.ladder {
            let gw = c.min_gw.max(template_side(c.alpha_width_frac, base, scale));
            let gh = 4.max(template_side(c.alpha_height_frac, base, scale));
            if gw as usize >= resp.width() || gh as usize >= resp.height() {
                continue;
            }
            let tmpl = resize(&assets::template(c.asset)?.silhouette, gw as usize, gh as usize, Interpolation::Area)?;
            let scores = search.match_template(&tmpl)?;
            let py_round = |v: f64| v.round_ties_even() as i32;
            let min_x = 0.max(py_round(f64::from(frame_w - loc.x - gw) - max_right * anchor_base));
            let max_x = (scores.width() as i32 - 1).min(frame_w - loc.x - gw);
            let min_y = 0.max(py_round(f64::from(frame_h - loc.y - gh) - max_bottom * anchor_base));
            let max_y = (scores.height() as i32 - 1).min(frame_h - loc.y - gh);
            if min_x > max_x || min_y > max_y {
                continue;
            }
            let window = scores.roi(raiw_cv::Rect::new(
                min_x as usize,
                min_y as usize,
                (max_x - min_x + 1) as usize,
                (max_y - min_y + 1) as usize,
            ))?;
            let mm = min_max_loc(&window)?;
            if mm.max > best.0 {
                let (x, y) = (min_x + mm.max_loc.x, min_y + mm.max_loc.y);
                best = (mm.max, Some(MatchBox::at(x, y, gw, gh)));
            }
        }
        Ok(best)
    }

    /// Everything in detection that does not depend on the trust level, with the scan hook.
    pub fn scan(&self, image: &Image) -> Result<TextMarkScan> {
        match self.hook {
            ScanHook::None => self.base_scan(image),
            ScanHook::MinShortSide(floor) => {
                if image.is_empty() || image.height().min(image.width()) < floor {
                    return Ok(TextMarkScan::default());
                }
                self.base_scan(image)
            }
            ScanHook::Downscale(short_side) => self.downscaled_scan(image, short_side),
            ScanHook::Outline { min_bright_neutral, min_mean_laplacian } => {
                let mut scan = self.base_scan(image)?;
                if let Some(l) = scan.located.as_mut()
                    && let (Some(b), Some(_)) = (l.match_box, l.score)
                {
                    let patch = region_view(
                        image,
                        Region::new(l.loc.x + b.x0, l.loc.y + b.y0, b.width(), b.height()),
                    );
                    if !outline_pixels_ok(&patch, min_bright_neutral, min_mean_laplacian)? {
                        l.score = Some(0.0);
                    }
                }
                Ok(scan)
            }
            ScanHook::Tinted { max_saturation, min_colored_spread, min_colored_fraction } => {
                let base = self.base_scan(image)?;
                let Some(l) = base.located.as_ref() else { return Ok(base) };
                let base_score = l.score.unwrap_or(0.0);
                if base_score >= self.config.detect_ncc_threshold {
                    return Ok(base);
                }
                let corner = region_view(image, l.loc.bbox());
                let (mut colored, mut total) = (0usize, 0usize);
                for y in 0..corner.height() {
                    for px in corner.row(y).chunks_exact(corner.channels()) {
                        let (b, g, r) = if px.len() >= 3 { (px[0], px[1], px[2]) } else { (px[0], px[0], px[0]) };
                        let spread = b.max(g).max(r) - b.min(g).min(r);
                        colored += usize::from(spread >= min_colored_spread);
                        total += 1;
                    }
                }
                if (colored as f64) < min_colored_fraction * total as f64 {
                    return Ok(base);
                }
                let tinted = Self::new(TextMarkConfig { max_saturation, ..self.config }).base_scan(image)?;
                let tinted_score = tinted.located.as_ref().and_then(|l| l.score).unwrap_or(0.0);
                Ok(if tinted_score > base_score { tinted } else { base })
            }
        }
    }

    /// The shared `_scan`: locate, binary blob, coverage, and the front-end score.
    pub fn base_scan(&self, image: &Image) -> Result<TextMarkScan> {
        let c = &self.config;
        if image.is_empty() || image.height().min(image.width()) < MIN_DETECT_SHORT_SIDE {
            return Ok(TextMarkScan::default());
        }
        let (h, w) = (image.height(), image.width());
        let loc = self.locate(h, w);
        let fields = if loc.h >= 16 && loc.w >= 16 && matches!(c.frontend, Frontend::Tophat | Frontend::Contrast) {
            Some(self.roi_fields(image, loc)?)
        } else {
            None
        };
        let glyphs = self.extract_mask(image, loc, fields.as_ref())?;
        let on = glyphs.data().iter().filter(|&&v| v > 0).count();
        let coverage = on as f64 / (loc.w * loc.h).max(1) as f64;
        let base = self.scale_base(h, w);
        let (score, match_box, match_patch) = if c.frontend == Frontend::Binary {
            if coverage >= c.detect_min_coverage {
                let (score, found) = template_match_best(&glyphs, base, c)?;
                let patch = match found {
                    Some(b) => Some(glyphs.crop(raiw_cv::Rect::new(
                        b.x0 as usize,
                        b.y0 as usize,
                        b.width() as usize,
                        b.height() as usize,
                    ))?),
                    None => None,
                };
                (Some(score), found, patch)
            } else {
                (None, None, None)
            }
        } else {
            let resp = self.detect_response(image, loc, fields.as_ref())?;
            let (score, found) = self.ladder_best(image, loc, resp.as_ref())?;
            let patch = match (&resp, found) {
                (Some(r), Some(b)) => Some(r.crop(raiw_cv::Rect::new(
                    b.x0 as usize,
                    b.y0 as usize,
                    b.width() as usize,
                    b.height() as usize,
                ))?),
                _ => None,
            };
            (Some(score), found, patch)
        };
        Ok(TextMarkScan {
            located: Some(Located { loc, glyphs, coverage, score, match_box, match_patch }),
            base,
            frame: (h, w),
        })
    }

    fn downscaled_scan(&self, image: &Image, short_side: usize) -> Result<TextMarkScan> {
        if image.is_empty() {
            return self.base_scan(image);
        }
        let (h, w) = (image.height(), image.width());
        let short = h.min(w);
        if short <= short_side {
            return self.base_scan(image);
        }
        let scale = short_side as f64 / short as f64;
        let sw = 200.max((w as f64 * scale).round_ties_even() as usize);
        let sh = 200.max((h as f64 * scale).round_ties_even() as usize);
        let small = resize(image, sw, sh, Interpolation::Area)?;
        let sub = self.base_scan(&small)?;
        let Some(sl) = sub.located else { return Ok(sub) };
        let loc = self.locate(h, w);
        let inv_x = f64::from(loc.w) / f64::from(sl.loc.w.max(1));
        let inv_y = f64::from(loc.h) / f64::from(sl.loc.h.max(1));
        let glyphs = resize(&sl.glyphs, loc.w as usize, loc.h as usize, Interpolation::Nearest)?;
        let match_box = sl.match_box.map(|m| {
            MatchBox::new(
                (f64::from(m.x0) * inv_x) as i32,
                (f64::from(m.y0) * inv_y) as i32,
                (loc.w - 1).min((f64::from(m.x1 + 1) * inv_x).round_ties_even() as i32 - 1),
                (loc.h - 1).min((f64::from(m.y1 + 1) * inv_y).round_ties_even() as i32 - 1),
            )
        });
        Ok(TextMarkScan {
            located: Some(Located {
                loc,
                glyphs,
                coverage: sl.coverage,
                score: sl.score,
                match_box,
                match_patch: sl.match_patch,
            }),
            base: self.scale_base(h, w),
            frame: (h, w),
        })
    }

    /// Apply the trust-level-dependent tail to a scan, as a fresh result.
    pub fn verdict(&self, scan: &TextMarkScan, provenance: bool) -> Result<TextMarkDetection> {
        let c = &self.config;
        let mut det = TextMarkDetection { provenance, template_asset: Some(c.asset), ..TextMarkDetection::default() };
        let Some(l) = scan.located.as_ref() else { return Ok(det) };
        det.region = l.loc.bbox();
        det.coverage = l.coverage;
        det.match_box = l.match_box;
        let Some(score) = l.score else { return Ok(det) };
        det.confidence = score;
        det.detected = score >= c.threshold(provenance) && self.rival_margin_ok(score, &l.glyphs, scan.base)?;
        self.post_gate(det, scan)
    }

    /// Whether this mark's template beats every same-corner rival's on the same glyph blob.
    fn rival_margin_ok(&self, score: f64, glyphs: &Mat<u8>, base: usize) -> Result<bool> {
        for asset in self.config.rivals {
            let Some(rival) = rival_config(asset) else { continue };
            if score - template_match_best(glyphs, base, rival)?.0 < self.config.rival_margin {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn post_gate(&self, mut det: TextMarkDetection, scan: &TextMarkScan) -> Result<TextMarkDetection> {
        let (max_right, max_bottom) = match self.gate {
            PostGate::None => return Ok(det),
            PostGate::TopLeft { max_x, max_y } => {
                let Some(l) = scan.located.as_ref().filter(|_| det.detected) else { return Ok(det) };
                let Some(b) = det.match_box else {
                    det.detected = false;
                    return Ok(det);
                };
                let (h, w) = scan.frame;
                let ax = f64::from(l.loc.x + b.x0) / w as f64;
                let ay = f64::from(l.loc.y + b.y0) / h as f64;
                if ax > max_x || ay > max_y {
                    det.detected = false;
                }
                return Ok(det);
            }
            PostGate::BottomRight { max_right, max_bottom } | PostGate::TwoLine { max_right, max_bottom, .. } => {
                (max_right, max_bottom)
            }
        };
        let Some(l) = scan.located.as_ref().filter(|_| det.detected) else { return Ok(det) };
        let Some(b) = det.match_box else {
            det.detected = false;
            return Ok(det);
        };
        let (h, w) = scan.frame;
        let base = h.min(w) as f64;
        let right = (w as f64 - f64::from(l.loc.x + b.x1 + 1)) / base;
        let bottom = (h as f64 - f64::from(l.loc.y + b.y1 + 1)) / base;
        if !((0.0..=max_right).contains(&right) && (0.0..=max_bottom).contains(&bottom)) {
            det.detected = false;
            return Ok(det);
        }
        if let PostGate::TwoLine { min_line, min_detail, .. } = self.gate
            && !self.two_line_glyph_ok(l.match_patch.as_ref(), min_line, min_detail)?
        {
            det.detected = false;
        }
        Ok(det)
    }

    fn two_line_glyph_ok(&self, patch: Option<&Mat<u8>>, min_line: f64, min_detail: f64) -> Result<bool> {
        let Some(patch) = patch.filter(|p| p.height() >= 4 && p.width() >= 4) else { return Ok(false) };
        let (gw, gh) = (patch.width(), patch.height());
        let tmpl = resize(&assets::template(self.config.asset)?.silhouette, gw, gh, Interpolation::Area)?;
        let mid = gh / 2;
        let halves = |m: &Mat<u8>| -> Result<(Mat<u8>, Mat<u8>)> {
            Ok((m.crop(raiw_cv::Rect::new(0, 0, gw, mid))?, m.crop(raiw_cv::Rect::new(0, mid, gw, gh - mid))?))
        };
        let (pt, pb) = halves(patch)?;
        let (tt, tb) = halves(&tmpl)?;
        let top = ncc_same_size(&pt.view(), &tt.view())?;
        let bot = ncc_same_size(&pb.view(), &tb.view())?;
        let line_min = top.min(bot);
        let detail = glyph_detail_ncc(patch, &tmpl);
        Ok(line_min >= min_line && detail >= min_detail)
    }

    /// One scan, then both verdicts.
    pub fn detect_both(&self, image: &Image) -> Result<(TextMarkDetection, TextMarkDetection)> {
        let scan = self.scan(image)?;
        Ok((self.verdict(&scan, false)?, self.verdict(&scan, true)?))
    }

    pub fn detect(&self, image: &Image, provenance: bool) -> Result<TextMarkDetection> {
        let scan = self.scan(image)?;
        self.verdict(&scan, provenance)
    }
}

/// Best nominal-size binary template score and its locate-box-local match box.
pub fn template_match_best(glyphs: &Mat<u8>, base: usize, c: &TextMarkConfig) -> Result<(f64, Option<MatchBox>)> {
    if glyphs.is_empty() {
        return Ok((0.0, None));
    }
    let sil = &assets::template(c.asset)?.silhouette;
    let gw = (glyphs.width() as i32 - 1).min(c.min_gw.max((c.alpha_width_frac * base as f64) as i32));
    let gh = (glyphs.height() as i32 - 1).min(4.max((c.alpha_height_frac * base as f64) as i32));
    if gw < c.min_gw || gh < 4 {
        return Ok((0.0, None));
    }
    let tmpl = resize(sil, gw as usize, gh as usize, Interpolation::Nearest)?;
    let (score, x, y) = best_match(&TemplateSearch::new(glyphs), &tmpl)?;
    Ok((score, Some(MatchBox::at(x, y, gw, gh))))
}

/// The outlined Doubao layout's pixel check on its winning box.
fn outline_pixels_ok(patch: &MatView<'_, u8>, min_bright_neutral: f64, min_mean_laplacian: f64) -> Result<bool> {
    if patch.is_empty() {
        return Ok(false);
    }
    let bgr = view_to_bgr(patch)?;
    let gray = cvt_color(&bgr, ColorConversion::BgrToGray)?;
    let hsv = cvt_color(&bgr, ColorConversion::BgrToHsv)?;
    let n = gray.data().len();
    let bright = gray
        .data()
        .iter()
        .zip(hsv.data().chunks_exact(3))
        .filter(|(g, px)| **g > 160 && px[1] < 60)
        .count();
    let lap: Mat<f32> = laplacian(&gray, 1, 1.0, 0.0, Border::Reflect101)?;
    let edge = lap.data().iter().map(|v| f64::from(v.abs())).sum::<f64>() / n as f64;
    Ok(bright as f64 / n as f64 >= min_bright_neutral && edge >= min_mean_laplacian)
}

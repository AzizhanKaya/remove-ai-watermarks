//! Per-frame detection types, shared frame representations, and numpy-style slicing.

use std::collections::HashMap;
use std::rc::Rc;

use raiw_cv::{
    ColorConversion, Element, Interpolation, Mat, MatView, Rect, TemplateSearch, cvt_color,
    match_template_ccoeff_normed, min_max_loc, resize,
};
use serde::Serialize;

pub use raiw_visible::Region;

use crate::error::Result;
use crate::py::round_int;

const NORMALIZED_SHORT_SIDE: f64 = 480.0;

/// `FrameLocalization`: the best untrusted candidate in one decoded frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FrameLocalization {
    pub frame_index: usize,
    pub confidence: f64,
    pub region: Option<Region>,
}

impl FrameLocalization {
    pub const fn empty(frame_index: usize) -> Self {
        Self {
            frame_index,
            confidence: 0.0,
            region: None,
        }
    }
}

/// Keys of the per-frame feature cache (`_PreparedFrame.feature_cache`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FeatureKey {
    FullTopHat,
    FullTopHatHalf,
    RoiTopHat {
        tag: &'static str,
        region: Region,
        kernel: usize,
    },
    CornerTopHat {
        corner: &'static str,
        sx: i64,
        kernel: usize,
    },
    PillSobel,
}

/// `_PreparedFrame`: grayscale forms shared by every detector in one scan pass.
#[derive(Debug)]
pub struct PreparedFrame {
    pub gray: Mat<u8>,
    pub normalized_gray: Mat<u8>,
    pub normalized_scale: f64,
    /// Feature images, each prepared once for every template matched against it.
    pub(crate) u8_cache: HashMap<FeatureKey, Rc<TemplateSearch<'static, u8>>>,
    pub(crate) f32_cache: HashMap<FeatureKey, Mat<f32>>,
}

/// `_normalized_gray`: area-downscale so the short side is at most 480.
pub fn normalized_gray(gray: &Mat<u8>) -> Result<(Mat<u8>, f64)> {
    let (width, height) = gray.size();
    let short = width.min(height);
    if short == 0 {
        return Ok((gray.clone(), 1.0));
    }
    let scale = (NORMALIZED_SHORT_SIDE / short as f64).min(1.0);
    if scale < 1.0 {
        let w = round_int(width as f64 * scale).max(1) as usize;
        let h = round_int(height as f64 * scale).max(1) as usize;
        return Ok((resize(gray, w, h, Interpolation::Area)?, scale));
    }
    Ok((gray.clone(), scale))
}

impl PreparedFrame {
    /// `_prepare_frame`.
    pub fn new(image_bgr: &Mat<u8>) -> Result<Self> {
        let gray = if image_bgr.channels() == 1 {
            image_bgr.clone()
        } else {
            cvt_color(image_bgr, ColorConversion::BgrToGray)?
        };
        let (normalized_gray, normalized_scale) = normalized_gray(&gray)?;
        Ok(Self {
            gray,
            normalized_gray,
            normalized_scale,
            u8_cache: HashMap::new(),
            f32_cache: HashMap::new(),
        })
    }
}

/// `array[y0:y1, x0:x1]` with numpy's clamping of non-negative bounds.
pub fn slice<'a, T: Element>(
    image: impl Into<MatView<'a, T>>,
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
) -> MatView<'a, T> {
    let image = image.into();
    let (w, h) = (image.width() as i64, image.height() as i64);
    let clamp = |v: i64, limit: i64| if v < 0 { (limit + v).max(0) } else { v.min(limit) };
    let (x0, x1) = (clamp(x0, w), clamp(x1, w));
    let (y0, y1) = (clamp(y0, h), clamp(y1, h));
    let rect = Rect::new(
        x0 as usize,
        y0 as usize,
        (x1 - x0).max(0) as usize,
        (y1 - y0).max(0) as usize,
    );
    image.roi(rect).unwrap_or_else(|_| empty_view(image.channels()))
}

fn empty_view<'a, T: Element>(channels: usize) -> MatView<'a, T> {
    MatView::new(&[], 0, 0, channels.max(1)).unwrap_or_else(|_| unreachable!("an empty view needs no data"))
}

/// `image[y:y+h, x:x+w]`.
pub fn crop<'a, T: Element>(
    image: impl Into<MatView<'a, T>>,
    x: i64,
    y: i64,
    w: i64,
    h: i64,
) -> MatView<'a, T> {
    slice(image, x, y, x + w, y + h)
}

/// `cv2.minMaxLoc(cv2.matchTemplate(image, template, TM_CCOEFF_NORMED))`: max value and location.
pub fn best_match<'a, 'b, T: Element>(
    image: impl Into<MatView<'a, T>>,
    template: impl Into<MatView<'b, T>>,
) -> Result<(f64, (i64, i64))> {
    peak(&match_template_ccoeff_normed(image, template)?)
}

/// [`best_match`] against a prepared image shared by several templates.
pub fn best_match_in<'b, T: Element>(
    search: &TemplateSearch<'_, T>,
    template: impl Into<MatView<'b, T>>,
) -> Result<(f64, (i64, i64))> {
    peak(&search.match_template(template)?)
}

fn peak(scores: &Mat<f32>) -> Result<(f64, (i64, i64))> {
    let found = min_max_loc(scores)?;
    Ok((
        found.max,
        (i64::from(found.max_loc.x), i64::from(found.max_loc.y)),
    ))
}

/// `float(cv2.matchTemplate(patch, template)[0, 0])` for equal-size inputs.
pub fn single_score<'a, 'b, T: Element>(
    patch: impl Into<MatView<'a, T>>,
    template: impl Into<MatView<'b, T>>,
) -> Result<f64> {
    let scores = match_template_ccoeff_normed(patch, template)?;
    Ok(scores.data().first().map_or(0.0, |v| f64::from(*v)))
}

pub fn view_max(view: &MatView<'_, u8>) -> u8 {
    view.iter().max().unwrap_or(0)
}

pub fn view_size<T: Element>(view: &MatView<'_, T>) -> usize {
    view.width() * view.height() * view.channels()
}

/// `Region` from a numpy-style `(x, y, w, h)`.
pub fn region(x: i64, y: i64, w: i64, h: i64) -> Region {
    Region::new(x as i32, y as i32, w as i32, h as i32)
}

pub fn region_parts(r: Region) -> (i64, i64, i64, i64) {
    (i64::from(r.x), i64::from(r.y), i64::from(r.w), i64::from(r.h))
}

/// `_region_intersection`.
pub fn region_intersection(a: Region, b: Region) -> i64 {
    let (ax, ay, aw, ah) = region_parts(a);
    let (bx, by, bw, bh) = region_parts(b);
    0.max((ax + aw).min(bx + bw) - ax.max(bx)) * 0.max((ay + ah).min(by + bh) - ay.max(by))
}

/// `_region_iou`.
pub fn region_iou(a: Region, b: Region) -> f64 {
    let inter = region_intersection(a, b);
    let union = i64::from(a.w) * i64::from(a.h) + i64::from(b.w) * i64::from(b.h) - inter;
    if union > 0 {
        inter as f64 / union as f64
    } else {
        0.0
    }
}

/// `_restore_region`: map a normalized-frame box back to source pixels.
pub fn restore_region(r: Region, scale: f64, frame_width: usize, frame_height: usize) -> Region {
    let (x, y, w, h) = region_parts(r);
    let sx = round_int(x as f64 / scale);
    let sy = round_int(y as f64 / scale);
    let sw = (frame_width as i64 - sx).min(round_int(w as f64 / scale).max(1));
    let sh = (frame_height as i64 - sy).min(round_int(h as f64 / scale).max(1));
    region(sx, sy, sw, sh)
}

/// `_bounded_region`: clip an expanded box to the frame without moving its anchor.
pub fn bounded_region(x: i64, y: i64, w: i64, h: i64, frame_width: usize, frame_height: usize) -> Region {
    let bx = x.max(0);
    let by = y.max(0);
    region(
        bx,
        by,
        (frame_width as i64 - bx).min(w + x - bx),
        (frame_height as i64 - by).min(h + y - by),
    )
}

//! The shared fill (`region_eraser.py`): erase boxes or a mask through a swappable backend.
//!
//! `cv2` is OpenCV's Telea / Navier-Stokes inpaint. `migan` and `lama` wrap a learned model
//! supplied by the caller (raiw-gpu); without one they report the backend unavailable.

mod learned;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use raiw_cv::color::ColorElement;
use raiw_cv::inpaint::InpaintElement;
use raiw_cv::resize::Resizable;
use raiw_cv::{InpaintMethod, Mat, MorphShape, Rect, StructuringElement, convert_scale_abs, dilate, inpaint};
use serde::{Deserialize, Serialize};

use crate::cvx::any_nonzero;
use crate::error::{Result, VisibleError};
use crate::geometry::Region;
use crate::raster::Mask;

pub use learned::{LamaFill, LamaModel, MiganFill, MiganModel, padded_crop_box};

/// The caller-facing fill choice (`--backend`); `auto` resolves best-first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Auto,
    Cv2,
    Migan,
    Lama,
}

/// A backend that can execute (`auto` resolved).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FillKind {
    Cv2,
    Migan,
    Lama,
}

/// Every fill backend name, plus `auto` (`region_eraser.FILL_BACKENDS`).
pub const FILL_BACKENDS: [&str; 4] = ["auto", "cv2", "migan", "lama"];

impl Backend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cv2 => "cv2",
            Self::Migan => "migan",
            Self::Lama => "lama",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(Self::Auto),
            "cv2" => Some(Self::Cv2),
            "migan" => Some(Self::Migan),
            "lama" => Some(Self::Lama),
            _ => None,
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FillKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cv2 => "cv2",
            Self::Migan => "migan",
            Self::Lama => "lama",
        }
    }
}

impl From<FillKind> for Backend {
    fn from(kind: FillKind) -> Self {
        match kind {
            FillKind::Cv2 => Self::Cv2,
            FillKind::Migan => Self::Migan,
            FillKind::Lama => Self::Lama,
        }
    }
}

/// Integer sample types the fill backends accept.
pub trait FillPixel: InpaintElement + Resizable + ColorElement {
    /// `np.iinfo(dtype).max`.
    const FULL_SCALE: f64;
    #[doc(hidden)]
    fn fill_with(backend: &dyn FillBackend, image: &Mat<Self>, mask: &Mask) -> Result<Mat<Self>>;
}

impl FillPixel for u8 {
    const FULL_SCALE: f64 = 255.0;
    fn fill_with(backend: &dyn FillBackend, image: &Mat<u8>, mask: &Mask) -> Result<Mat<u8>> {
        backend.fill_u8(image, mask)
    }
}

impl FillPixel for u16 {
    const FULL_SCALE: f64 = 65535.0;
    fn fill_with(backend: &dyn FillBackend, image: &Mat<u16>, mask: &Mask) -> Result<Mat<u16>> {
        backend.fill_u16(image, mask)
    }
}

/// One executable fill. Implementations paste back only masked pixels.
pub trait FillBackend: Send + Sync + fmt::Debug {
    fn kind(&self) -> FillKind;
    fn available(&self) -> bool;
    fn fill_u8(&self, image: &Mat<u8>, mask: &Mask) -> Result<Mat<u8>>;
    fn fill_u16(&self, image: &Mat<u16>, mask: &Mask) -> Result<Mat<u16>>;
    /// Whether batched inference is faster on this backend's device.
    fn prefers_batching(&self) -> bool {
        false
    }
}

/// `cv2.inpaint` (`erase_cv2`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cv2Fill {
    pub method: InpaintMethod,
    pub radius: f64,
}

impl Default for Cv2Fill {
    fn default() -> Self {
        Self { method: InpaintMethod::Telea, radius: 6.0 }
    }
}

impl FillBackend for Cv2Fill {
    fn kind(&self) -> FillKind {
        FillKind::Cv2
    }

    fn available(&self) -> bool {
        true
    }

    fn fill_u8(&self, image: &Mat<u8>, mask: &Mask) -> Result<Mat<u8>> {
        if image.channels() == 4 {
            let bgr = select_channels(image, 3);
            let filled = inpaint(&bgr, mask, self.radius, self.method)?;
            return Ok(with_channels_from(image, &filled));
        }
        Ok(inpaint(image, mask, self.radius, self.method)?)
    }

    fn fill_u16(&self, image: &Mat<u16>, mask: &Mask) -> Result<Mat<u16>> {
        if image.channels() == 1 {
            return Ok(inpaint(image, mask, self.radius, self.method)?);
        }
        erase_via_u8(image, mask, |work| self.fill_u8(work, mask))
    }
}

/// The first `n` channels of an interleaved image (`image[..., :n]`).
pub(crate) fn select_channels<T: raiw_cv::Element>(image: &Mat<T>, n: usize) -> Mat<T> {
    let cn = image.channels();
    let mut data = Vec::with_capacity(image.width() * image.height() * n);
    for px in image.data().chunks_exact(cn) {
        data.extend_from_slice(&px[..n]);
    }
    Mat::from_vec(image.width(), image.height(), n, data).unwrap_or_else(|_| image.clone())
}

/// `image` with its leading channels replaced by `head` (`np.dstack([head, image[..., n:]])`).
pub(crate) fn with_channels_from<T: raiw_cv::Element>(image: &Mat<T>, head: &Mat<T>) -> Mat<T> {
    let (cn, hn) = (image.channels(), head.channels());
    let mut out = image.clone();
    for (dst, src) in out.data_mut().chunks_exact_mut(cn).zip(head.data().chunks_exact(hn)) {
        dst[..hn].copy_from_slice(src);
    }
    out
}

/// Narrow a 16-bit image to 8 bits for an 8-bit-only fill, then scale only the filled pixels back.
fn erase_via_u8(
    image: &Mat<u16>,
    mask: &Mask,
    fill: impl FnOnce(&Mat<u8>) -> Result<Mat<u8>>,
) -> Result<Mat<u16>> {
    let source = if image.channels() == 4 { select_channels(image, 3) } else { image.clone() };
    let work = convert_scale_abs(&source, 255.0 / u16::FULL_SCALE, 0.0);
    let filled = fill(&work)?;
    let mut result = image.clone();
    paste_scaled_fill(&mut result, &filled, mask, 0, 0);
    Ok(result)
}

/// `dst[hole] = clip(filled[hole] * (scale / 255), 0, scale)` over `filled`'s channels, placing
/// `filled` (and the matching `mask` window) at `(x, y)` in `dst`.
pub(crate) fn paste_scaled_fill(dst: &mut Mat<u16>, filled: &Mat<u8>, mask: &Mask, x: usize, y: usize) {
    let factor = (u16::FULL_SCALE / 255.0) as f32;
    let (dc, fc) = (dst.channels(), filled.channels());
    for row in 0..filled.height() {
        for col in 0..filled.width() {
            if mask.at(x + col, y + row, 0) <= 127 {
                continue;
            }
            let src = filled.pixel(col, row);
            let out = dst.pixel_mut(x + col, y + row);
            for c in 0..fc.min(dc) {
                out[c] = (f32::from(src[c]) * factor).clamp(0.0, 65535.0) as u16;
            }
        }
    }
}

/// The configured fill backends: classical always, learned ones when a model is supplied.
#[derive(Clone, Debug, Default)]
pub struct FillBackends {
    pub cv2: Cv2Fill,
    pub lama: LamaFill,
    pub migan: MiganFill,
}

static WARNED_CV2_FALLBACK: AtomicBool = AtomicBool::new(false);

impl FillBackends {
    /// Only the classical fill (no learned model configured).
    pub fn classical() -> Self {
        Self::default()
    }

    pub fn with_lama(mut self, model: Arc<dyn LamaModel>) -> Self {
        self.lama = LamaFill::new(model);
        self
    }

    pub fn with_migan(mut self, model: Arc<dyn MiganModel>) -> Self {
        self.migan = MiganFill::new(model);
        self
    }

    /// Whether any learned inpaint model can run (`inpaint_model_available`).
    pub fn inpaint_model_available(&self) -> bool {
        self.lama.available() || self.migan.available()
    }

    /// What `auto` resolves to: LaMa, else MI-GAN, else cv2 with a one-time quality warning.
    pub fn preferred(&self) -> FillKind {
        if self.lama.available() {
            return FillKind::Lama;
        }
        if self.migan.available() {
            return FillKind::Migan;
        }
        if !WARNED_CV2_FALLBACK.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "No learned-inpaint backend available; falling back to the cv2 classical inpaint, which can \
                 smear textured or structured backgrounds. Enable the LaMa (best) or MI-GAN (lighter) backend \
                 for higher-quality fills."
            );
        }
        FillKind::Cv2
    }

    /// `resolve_backend`: `auto` to the preferred backend; the rest pass through.
    pub fn resolve(&self, backend: Backend) -> FillKind {
        match backend {
            Backend::Auto => self.preferred(),
            Backend::Cv2 => FillKind::Cv2,
            Backend::Migan => FillKind::Migan,
            Backend::Lama => FillKind::Lama,
        }
    }

    pub fn get(&self, kind: FillKind) -> &dyn FillBackend {
        match kind {
            FillKind::Cv2 => &self.cv2,
            FillKind::Migan => &self.migan,
            FillKind::Lama => &self.lama,
        }
    }

    /// Whether `backend` resolves to a device where batched inference is faster.
    pub fn prefers_batching(&self, backend: Backend) -> bool {
        let backend = self.get(self.resolve(backend));
        backend.available() && backend.prefers_batching()
    }

    /// The learned backend for `kind`, refusing when its model is missing.
    fn require(&self, kind: FillKind) -> Result<&dyn FillBackend> {
        let backend = self.get(kind);
        if !backend.available() {
            let label = match kind {
                FillKind::Lama => "LaMa",
                FillKind::Migan => "MI-GAN",
                FillKind::Cv2 => "cv2",
            };
            return Err(VisibleError::BackendUnavailable {
                backend: label,
                reason: format!("no {label} model is configured (it is provided by raiw-gpu)"),
            });
        }
        Ok(backend)
    }

    /// Fill `mask` with an already-resolved backend (`watermark_registry.fill`).
    pub fn fill<T: FillPixel>(&self, image: &Mat<T>, mask: &Mask, kind: FillKind) -> Result<Mat<T>> {
        erase(image, Selection::Mask(mask), &EraseOptions { backend: kind.into(), ..EraseOptions::default() }, self)
    }
}

/// What to erase: `(x, y, w, h)` boxes (dilated) or a precomputed mask.
#[derive(Clone, Copy, Debug)]
pub enum Selection<'a> {
    Boxes(&'a [Region]),
    Mask(&'a Mask),
}

/// `region_eraser.erase` knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EraseOptions {
    pub backend: Backend,
    pub dilate: i32,
    pub method: InpaintMethod,
    pub radius: f64,
}

impl Default for EraseOptions {
    fn default() -> Self {
        Self { backend: Backend::Cv2, dilate: 3, method: InpaintMethod::Telea, radius: 6.0 }
    }
}

/// Erase the selection; an empty image or selection returns an unmodified copy.
///
/// `auto` passed straight here degrades to the classical fill, as in Python; callers that want
/// best-first resolution go through [`FillBackends::resolve`] first.
pub fn erase<T: FillPixel>(
    image: &Mat<T>,
    selection: Selection<'_>,
    options: &EraseOptions,
    backends: &FillBackends,
) -> Result<Mat<T>> {
    if image.is_empty() {
        return Ok(image.clone());
    }
    let built;
    let mask = match selection {
        Selection::Mask(mask) => mask,
        Selection::Boxes(boxes) => {
            if boxes.is_empty() {
                return Ok(image.clone());
            }
            built = boxes_to_mask(image.height(), image.width(), boxes, options.dilate)?;
            &built
        }
    };
    if !any_nonzero(mask) {
        return Ok(image.clone());
    }
    match options.backend {
        Backend::Lama => T::fill_with(backends.require(FillKind::Lama)?, image, mask),
        Backend::Migan => T::fill_with(backends.require(FillKind::Migan)?, image, mask),
        Backend::Cv2 | Backend::Auto => {
            T::fill_with(&Cv2Fill { method: options.method, radius: options.radius }, image, mask)
        }
    }
}

/// [`erase`] over `(image, mask)` pairs, for multi-frame video fills.
pub fn erase_batch<T: FillPixel>(
    items: &[(&Mat<T>, &Mask)],
    backend: Backend,
    backends: &FillBackends,
) -> Result<Vec<Mat<T>>> {
    let options = EraseOptions { backend, ..EraseOptions::default() };
    items.iter().map(|(image, mask)| erase(image, Selection::Mask(mask), &options, backends)).collect()
}

/// A uint8 mask, 255 inside `(x, y, w, h)` boxes clipped to the frame, optionally dilated by an
/// elliptical kernel of radius `dilate`.
pub fn boxes_to_mask(height: usize, width: usize, boxes: &[Region], dilate_radius: i32) -> Result<Mask> {
    let mut mask = Mask::zeros(width, height, 1);
    let (w, h) = (width as i64, height as i64);
    for b in boxes {
        let (x, y) = (i64::from(b.x), i64::from(b.y));
        let (x0, y0) = (x.max(0), y.max(0));
        let (x1, y1) = ((x + i64::from(b.w)).min(w), (y + i64::from(b.h)).min(h));
        if x1 > x0 && y1 > y0 {
            let rect = Rect::new(x0 as usize, y0 as usize, (x1 - x0) as usize, (y1 - y0) as usize);
            mask.roi_mut(rect)?.fill(255);
        }
    }
    if dilate_radius > 0 && any_nonzero(&mask) {
        let side = 2 * dilate_radius as usize + 1;
        mask = dilate(&mask, &StructuringElement::new(MorphShape::Ellipse, side, side), 1)?;
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_round_trip() {
        for name in FILL_BACKENDS {
            assert_eq!(Backend::parse(name).map(Backend::as_str), Some(name));
        }
        assert_eq!(Backend::parse("nope"), None);
    }

    #[test]
    fn boxes_are_clipped_and_dilated() {
        let m0 = boxes_to_mask(100, 100, &[Region::new(40, 40, 10, 10)], 0).unwrap();
        let m5 = boxes_to_mask(100, 100, &[Region::new(40, 40, 10, 10)], 5).unwrap();
        let sum = |m: &Mask| m.data().iter().map(|&v| u64::from(v)).sum::<u64>();
        assert!(sum(&m5) > sum(&m0));
        let clipped = boxes_to_mask(50, 50, &[Region::new(40, 40, 100, 100)], 0).unwrap();
        assert_eq!(clipped.at(45, 45, 0), 255);
        let two = boxes_to_mask(100, 100, &[Region::new(0, 0, 10, 10), Region::new(90, 90, 10, 10)], 0).unwrap();
        assert_eq!((two.at(5, 5, 0), two.at(95, 95, 0), two.at(50, 50, 0)), (255, 255, 0));
        let inside = boxes_to_mask(100, 100, &[Region::new(10, 20, 30, 40)], 0).unwrap();
        assert_eq!((inside.at(15, 25, 0), inside.at(0, 0, 0)), (255, 0));
    }

    #[test]
    fn auto_without_models_resolves_to_cv2() {
        let backends = FillBackends::classical();
        assert_eq!(backends.resolve(Backend::Auto), FillKind::Cv2);
        assert_eq!(backends.resolve(Backend::Lama), FillKind::Lama);
        assert!(!backends.inpaint_model_available());
    }
}

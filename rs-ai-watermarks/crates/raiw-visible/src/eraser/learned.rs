//! Crop, run and paste for the learned fills; the networks themselves come from raiw-gpu.

use std::fmt;
use std::sync::Arc;

use raiw_cv::{ColorConversion, Interpolation, Mat, Rect, convert_scale_abs, cvt_color, resize};

use super::{FillBackend, FillKind, FillPixel, paste_scaled_fill, select_channels, with_channels_from};
use crate::error::{Result, VisibleError};
use crate::raster::Mask;

/// A big-LaMa session: RGB `1x3xSxS` in `[0, 1]` plus a binary `1x1xSxS` mask in, `1x3xSxS` out.
pub trait LamaModel: Send + Sync + fmt::Debug {
    /// The model's fixed square input side.
    fn input_size(&self) -> usize {
        512
    }
    fn infer(&self, image: &[f32], mask: &[f32], size: usize) -> Result<Vec<f32>>;
    fn prefers_batching(&self) -> bool {
        false
    }
}

/// An MI-GAN session: RGB `1x3xHxW` uint8 plus a `1x1xHxW` known-pixel mask (255 keep, 0 hole)
/// in; returns `(rgb_chw, width, height)` of the filled crop.
pub trait MiganModel: Send + Sync + fmt::Debug {
    fn infer(&self, image: &[u8], known: &[u8], width: usize, height: usize) -> Result<(Vec<u8>, usize, usize)>;
    fn prefers_batching(&self) -> bool {
        false
    }
}

/// Padded `(x0, y0, x1, y1)` bounding box of the set mask pixels, clamped to the image.
pub fn padded_crop_box(mask: &Mask, pad_frac: f64, pad_min: usize) -> Option<(usize, usize, usize, usize)> {
    let r = crate::cvx::bounding_rect(mask);
    if r.w <= 0 {
        return None;
    }
    let (h, w) = (mask.height(), mask.width());
    let span = r.w.max(r.h) as usize;
    let pad = pad_min.max((pad_frac * span as f64) as usize);
    let (xmin, ymin) = (r.x as usize, r.y as usize);
    let (xmax, ymax) = (xmin + r.w as usize - 1, ymin + r.h as usize - 1);
    Some((xmin.saturating_sub(pad), ymin.saturating_sub(pad), (xmax + 1 + pad).min(w), (ymax + 1 + pad).min(h)))
}

fn crop<T: raiw_cv::Element>(image: &Mat<T>, b: (usize, usize, usize, usize)) -> Result<Mat<T>> {
    Ok(image.crop(Rect::new(b.0, b.1, b.2 - b.0, b.3 - b.1))?)
}

fn chw<T: Copy>(hwc: &[T], channels: usize, plane: usize) -> Vec<T> {
    (0..channels).flat_map(|c| (0..plane).map(move |i| hwc[i * channels + c])).collect()
}

fn unavailable(backend: &'static str) -> VisibleError {
    VisibleError::BackendUnavailable {
        backend,
        reason: format!("no {backend} model is configured (it is provided by raiw-gpu)"),
    }
}

/// The big-LaMa fill: a padded crop resized to the model square, masked pixels pasted back.
#[derive(Clone, Debug, Default)]
pub struct LamaFill {
    model: Option<Arc<dyn LamaModel>>,
}

impl LamaFill {
    pub fn new(model: Arc<dyn LamaModel>) -> Self {
        Self { model: Some(model) }
    }

    fn run<T: FillPixel>(&self, image: &Mat<T>, mask: &Mask) -> Result<Mat<T>> {
        let model = self.model.as_ref().ok_or_else(|| unavailable("LaMa"))?;
        match image.channels() {
            1 => {
                let bgr = cvt_color(image, ColorConversion::GrayToBgr)?;
                return Ok(cvt_color(&self.run(&bgr, mask)?, ColorConversion::BgrToGray)?);
            }
            4 => return Ok(with_channels_from(image, &self.run(&select_channels(image, 3), mask)?)),
            _ => {}
        }
        let size = model.input_size();
        let Some(b) = padded_crop_box(mask, 0.4, 16) else { return Ok(image.clone()) };
        let scale = T::FULL_SCALE;
        let crop_img = crop(image, b)?;
        let crop_mask = crop(mask, b)?;
        let crop_rs = resize(&crop_img, size, size, Interpolation::Area)?;
        let mask_rs = resize(&crop_mask, size, size, Interpolation::Nearest)?;
        let rgb = cvt_color(&crop_rs, ColorConversion::BgrToRgb)?;
        let plane = size * size;
        let img_in: Vec<f32> = chw(rgb.data(), 3, plane).iter().map(|v| v.to_f32() / scale as f32).collect();
        let mask_in: Vec<f32> = mask_rs.data().iter().map(|&v| if v > 127 { 1.0 } else { 0.0 }).collect();
        let out = model.infer(&img_in, &mask_in, size)?;
        if out.len() != 3 * plane {
            return Err(VisibleError::Inference { backend: "LaMa", reason: format!("{} outputs", out.len()) });
        }
        let peak = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let factor = if peak <= 1.5 {
            scale as f32
        } else if scale != 255.0 {
            (scale / 255.0) as f32
        } else {
            1.0
        };
        let hwc: Vec<T> = (0..plane)
            .flat_map(|i| (0..3).map(move |c| (i, c)))
            .map(|(i, c)| T::from_f64(f64::from((out[c * plane + i] * factor).clamp(0.0, scale as f32)).trunc()))
            .collect();
        let out_rgb = Mat::from_vec(size, size, 3, hwc)?;
        let out_bgr = cvt_color(&out_rgb, ColorConversion::RgbToBgr)?;
        let out_crop = resize(&out_bgr, crop_mask.width(), crop_mask.height(), Interpolation::Linear)?;
        let mut result = image.clone();
        for y in 0..crop_mask.height() {
            for x in 0..crop_mask.width() {
                if crop_mask.at(x, y, 0) > 127 {
                    result.pixel_mut(b.0 + x, b.1 + y).copy_from_slice(out_crop.pixel(x, y));
                }
            }
        }
        Ok(result)
    }
}

impl FillBackend for LamaFill {
    fn kind(&self) -> FillKind {
        FillKind::Lama
    }

    fn available(&self) -> bool {
        self.model.is_some()
    }

    fn fill_u8(&self, image: &Mat<u8>, mask: &Mask) -> Result<Mat<u8>> {
        self.run(image, mask)
    }

    fn fill_u16(&self, image: &Mat<u16>, mask: &Mask) -> Result<Mat<u16>> {
        self.run(image, mask)
    }

    fn prefers_batching(&self) -> bool {
        self.model.as_ref().is_some_and(|m| m.prefers_batching())
    }
}

/// The MI-GAN fill: a native-resolution padded crop, inverse-polarity mask, masked paste-back.
#[derive(Clone, Debug, Default)]
pub struct MiganFill {
    model: Option<Arc<dyn MiganModel>>,
}

impl MiganFill {
    pub fn new(model: Arc<dyn MiganModel>) -> Self {
        Self { model: Some(model) }
    }

    fn infer_crop<T: FillPixel>(&self, image: &Mat<T>, mask: &Mask) -> Result<Option<(Image8, Bounds)>> {
        let model = self.model.as_ref().ok_or_else(|| unavailable("MI-GAN"))?;
        let Some(b) = padded_crop_box(mask, 2.0, 256) else { return Ok(None) };
        let crop_img = crop(image, b)?;
        let crop_mask = crop(mask, b)?;
        let crop_bgr = match crop_img.channels() {
            1 => cvt_color(&crop_img, ColorConversion::GrayToBgr)?,
            4 => cvt_color(&crop_img, ColorConversion::BgraToBgr)?,
            _ => crop_img,
        };
        let work = if T::FULL_SCALE == 255.0 {
            crop_bgr.convert::<u8>()
        } else {
            convert_scale_abs(&crop_bgr, 255.0 / T::FULL_SCALE, 0.0)
        };
        let rgb = cvt_color(&work, ColorConversion::BgrToRgb)?;
        let (cw, ch) = (crop_mask.width(), crop_mask.height());
        let known: Vec<u8> = crop_mask.data().iter().map(|&v| if v <= 127 { 255 } else { 0 }).collect();
        let (out, ow, oh) = model.infer(&chw(rgb.data(), 3, cw * ch), &known, cw, ch)?;
        if out.len() != 3 * ow * oh {
            return Err(VisibleError::Inference { backend: "MI-GAN", reason: format!("{} outputs", out.len()) });
        }
        let plane = ow * oh;
        let hwc: Vec<u8> = (0..plane).flat_map(|i| (0..3).map(move |c| out[c * plane + i])).collect();
        let mut res = Mat::from_vec(ow, oh, 3, hwc)?;
        if (ow, oh) != (cw, ch) {
            res = resize(&res, cw, ch, Interpolation::Linear)?;
        }
        Ok(Some((cvt_color(&res, ColorConversion::RgbToBgr)?, b)))
    }

    fn run_u8(&self, image: &Mat<u8>, mask: &Mask) -> Result<Mat<u8>> {
        let Some((filled_bgr, b)) = self.infer_crop(image, mask)? else { return Ok(image.clone()) };
        let filled = if image.channels() == 1 { cvt_color(&filled_bgr, ColorConversion::BgrToGray)? } else { filled_bgr };
        let mut result = image.clone();
        let n = filled.channels();
        for y in 0..filled.height() {
            for x in 0..filled.width() {
                if mask.at(b.0 + x, b.1 + y, 0) > 127 {
                    result.pixel_mut(b.0 + x, b.1 + y)[..n].copy_from_slice(filled.pixel(x, y));
                }
            }
        }
        Ok(result)
    }

    fn run_u16(&self, image: &Mat<u16>, mask: &Mask) -> Result<Mat<u16>> {
        let Some((filled_bgr, b)) = self.infer_crop(image, mask)? else { return Ok(image.clone()) };
        let filled = if image.channels() == 1 { cvt_color(&filled_bgr, ColorConversion::BgrToGray)? } else { filled_bgr };
        let mut result = image.clone();
        paste_scaled_fill(&mut result, &filled, mask, b.0, b.1);
        Ok(result)
    }
}

type Image8 = Mat<u8>;
type Bounds = (usize, usize, usize, usize);

impl FillBackend for MiganFill {
    fn kind(&self) -> FillKind {
        FillKind::Migan
    }

    fn available(&self) -> bool {
        self.model.is_some()
    }

    fn fill_u8(&self, image: &Mat<u8>, mask: &Mask) -> Result<Mat<u8>> {
        self.run_u8(image, mask)
    }

    fn fill_u16(&self, image: &Mat<u16>, mask: &Mask) -> Result<Mat<u16>> {
        self.run_u16(image, mask)
    }

    fn prefers_batching(&self) -> bool {
        self.model.as_ref().is_some_and(|m| m.prefers_batching())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_with(h: usize, w: usize, rect: Rect) -> Mask {
        let mut m = Mask::zeros(w, h, 1);
        m.roi_mut(rect).unwrap().fill(255);
        m
    }

    #[test]
    fn empty_mask_has_no_crop() {
        assert_eq!(padded_crop_box(&Mask::zeros(100, 100, 1), 0.1, 8), None);
    }

    #[test]
    fn pad_min_dominates_and_clamps() {
        let m = mask_with(100, 100, Rect::new(0, 0, 5, 5));
        assert_eq!(padded_crop_box(&m, 0.1, 8), Some((0, 0, 13, 13)));
        let big = mask_with(400, 400, Rect::new(100, 100, 200, 200));
        assert_eq!(padded_crop_box(&big, 0.2, 8), Some((60, 60, 340, 340)));
        let far = mask_with(50, 60, Rect::new(55, 45, 5, 5));
        let (_, _, x1, y1) = padded_crop_box(&far, 0.1, 8).unwrap();
        assert_eq!((x1, y1), (60, 50));
    }
}

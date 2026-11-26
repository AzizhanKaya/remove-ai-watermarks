//! Pixel-depth abstraction and small array helpers for the temporal stages.

use raiw_cv::warp::WarpElement;
use raiw_cv::{Interpolation, Mat, MorphShape, Rect, StructuringElement, dilate};
use raiw_visible::eraser::FillPixel;

use super::flow::{gray_u8, gray_u16_as_u8};
use crate::decode::FramePixel;
use crate::error::Result;

/// The 8-bit and 16-bit BGR frame types the temporal stages accept.
pub trait VideoPixel: FramePixel + WarpElement + FillPixel {
    /// `_depth_bounds` peak value.
    const PEAK: f64;
    /// `_depth_bounds` scale for 8-bit thresholds.
    const THRESHOLD_SCALE: f64;
    /// `_to_gray_u8`.
    fn gray_u8(frame: &Mat<Self>) -> Result<Mat<u8>>;
}

impl VideoPixel for u8 {
    const PEAK: f64 = 255.0;
    const THRESHOLD_SCALE: f64 = 1.0;
    fn gray_u8(frame: &Mat<Self>) -> Result<Mat<u8>> {
        gray_u8(frame)
    }
}

impl VideoPixel for u16 {
    const PEAK: f64 = 65535.0;
    const THRESHOLD_SCALE: f64 = 257.0;
    fn gray_u8(frame: &Mat<Self>) -> Result<Mat<u8>> {
        gray_u16_as_u8(frame)
    }
}

/// `_depth_bounds`: `(scaled_threshold, peak)`.
pub fn depth_bounds<T: VideoPixel>(threshold_u8: f64) -> (f64, f64) {
    (threshold_u8 * T::THRESHOLD_SCALE, T::PEAK)
}

/// A boolean plane, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoolPlane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<bool>,
}

impl BoolPlane {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![false; width * height],
        }
    }

    /// `mask > 0`.
    pub fn from_mask(mask: &Mat<u8>) -> Self {
        Self {
            width: mask.width(),
            height: mask.height(),
            data: mask.data().iter().map(|v| *v > 0).collect(),
        }
    }

    /// `plane.astype(uint8) * value`.
    pub fn to_mask(&self, value: u8) -> Mat<u8> {
        let data = self.data.iter().map(|b| if *b { value } else { 0 }).collect();
        Mat::from_vec(self.width, self.height, 1, data)
            .unwrap_or_else(|_| Mat::zeros(self.width, self.height, 1))
    }

    pub fn count(&self) -> usize {
        self.data.iter().filter(|b| **b).count()
    }

    pub fn any(&self) -> bool {
        self.data.iter().any(|b| *b)
    }

    #[must_use]
    pub fn zip_with(&self, other: &Self, f: impl Fn(bool, bool) -> bool) -> Self {
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| f(*a, *b))
            .collect();
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// `(min_x, min_y, max_x, max_y)` of the set pixels.
    pub fn bounds(&self) -> Option<(usize, usize, usize, usize)> {
        let mut bounds: Option<(usize, usize, usize, usize)> = None;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.data[y * self.width + x] {
                    bounds = Some(match bounds {
                        None => (x, y, x, y),
                        Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                    });
                }
            }
        }
        bounds
    }
}

/// An elliptical structuring element of `size` x `size`.
pub fn ellipse(size: usize) -> StructuringElement {
    StructuringElement::new(MorphShape::Ellipse, size, size)
}

/// `cv2.dilate(mask, kernel)` on a uint8 mask.
pub fn dilate_mask(mask: &Mat<u8>, kernel: &StructuringElement) -> Result<Mat<u8>> {
    Ok(dilate(mask, kernel, 1)?)
}

/// A contiguous crop `image[y0:y1, x0:x1]`.
pub fn crop_mat<T: raiw_cv::Element>(
    image: &Mat<T>,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> Result<Mat<T>> {
    Ok(image.crop(Rect::new(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)))?)
}

/// Paste `patch` into `image` at `(x0, y0)`.
pub fn paste<T: raiw_cv::Element>(image: &mut Mat<T>, x0: usize, y0: usize, patch: &Mat<T>) -> Result<()> {
    let mut view = image.roi_mut(Rect::new(x0, y0, patch.width(), patch.height()))?;
    view.copy_from(patch)?;
    Ok(())
}

/// `cv2.remap` with nearest-neighbor sampling and reflected borders.
pub const NEAREST: Interpolation = Interpolation::Nearest;
/// Bilinear sampling.
pub const LINEAR: Interpolation = Interpolation::Linear;

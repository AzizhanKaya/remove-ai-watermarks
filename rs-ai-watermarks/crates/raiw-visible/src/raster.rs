//! In-memory rasters: OpenCV-layout BGR(A) or grayscale pixels at 8 or 16 bits.

use std::borrow::Cow;

use raiw_cv::{ColorConversion, Mat, MatView, Rect, cvt_color};

use crate::error::{Result, unsupported};
use crate::geometry::{Region, Span};

/// An 8-bit image with 1 (gray), 3 (BGR) or 4 (BGRA) interleaved channels.
pub type Image = Mat<u8>;

/// A single-channel 8-bit mask, 255 where pixels are removed.
pub type Mask = Mat<u8>;

/// A decoded raster at the source's bit depth.
#[derive(Clone, Debug, PartialEq)]
pub enum Pixels {
    U8(Mat<u8>),
    U16(Mat<u16>),
}

impl Pixels {
    pub fn width(&self) -> usize {
        match self {
            Self::U8(m) => m.width(),
            Self::U16(m) => m.width(),
        }
    }

    pub fn height(&self) -> usize {
        match self {
            Self::U8(m) => m.height(),
            Self::U16(m) => m.height(),
        }
    }

    pub fn channels(&self) -> usize {
        match self {
            Self::U8(m) => m.channels(),
            Self::U16(m) => m.channels(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.width() == 0 || self.height() == 0
    }

    pub const fn depth_bits(&self) -> u32 {
        match self {
            Self::U8(_) => 8,
            Self::U16(_) => 16,
        }
    }

    /// The 8-bit raster, or an error naming the depth an 8-bit-only stage cannot take.
    pub fn as_u8(&self, op: &'static str) -> Result<&Mat<u8>> {
        match self {
            Self::U8(m) => Ok(m),
            Self::U16(_) => Err(unsupported(op, "16-bit pixels; detectors take 8-bit images")),
        }
    }

    /// Interleaved samples as native-endian bytes (`ndarray.tobytes()`).
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::U8(m) => m.as_bytes(),
            Self::U16(m) => m.as_bytes(),
        }
    }
}

/// A 3-channel BGR view of `image`, promoting grayscale and dropping alpha (`image_io.to_bgr`).
pub fn to_bgr(image: &Image) -> Result<Cow<'_, Image>> {
    match image.channels() {
        3 => Ok(Cow::Borrowed(image)),
        1 => Ok(Cow::Owned(cvt_color(image, ColorConversion::GrayToBgr)?)),
        4 => Ok(Cow::Owned(cvt_color(image, ColorConversion::BgraToBgr)?)),
        n => Err(unsupported("to_bgr", format!("{n} channels"))),
    }
}

/// [`to_bgr`] of a sub-view, always owned.
pub fn view_to_bgr(view: &MatView<'_, u8>) -> Result<Image> {
    match view.channels() {
        3 => Ok(view.to_mat()),
        1 => Ok(cvt_color(view, ColorConversion::GrayToBgr)?),
        4 => Ok(cvt_color(view, ColorConversion::BgraToBgr)?),
        n => Err(unsupported("to_bgr", format!("{n} channels"))),
    }
}

/// `image[y0:y1, x0:x1]` with numpy's clamping of out-of-range bounds.
pub fn crop_view<T: raiw_cv::Element>(image: &Mat<T>, span: Span) -> MatView<'_, T> {
    let (w, h) = (image.width() as i32, image.height() as i32);
    let x0 = span.x0.clamp(0, w);
    let y0 = span.y0.clamp(0, h);
    let x1 = span.x1.clamp(x0, w);
    let y1 = span.y1.clamp(y0, h);
    let rect = Rect::new(x0 as usize, y0 as usize, (x1 - x0) as usize, (y1 - y0) as usize);
    // The clamped rect always fits, so the fallback is unreachable.
    image.view().roi(rect).unwrap_or_else(|_| image.view())
}

/// `image[y:y+h, x:x+w]` for a region.
pub fn region_view<T: raiw_cv::Element>(image: &Mat<T>, r: Region) -> MatView<'_, T> {
    crop_view(image, Span::new(r.x, r.y, r.x + r.w, r.y + r.h))
}

/// `mask[y0:y1, x0:x1] = source` with both sides already clipped to the same shape.
pub fn paste(dst: &mut Mat<u8>, x: usize, y: usize, src: &Mat<u8>) -> Result<()> {
    let mut roi = dst.roi_mut(Rect::new(x, y, src.width(), src.height()))?;
    roi.copy_from(src)?;
    Ok(())
}

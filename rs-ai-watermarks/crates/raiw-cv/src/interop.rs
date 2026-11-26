//! Zero-copy conversions with `image` buffers, channel-order helpers, and `imencode`/`imdecode`
//! equivalents (PNG, JPEG, WebP through the `image` crate).

use std::io::Cursor;

use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, LumaA, Rgb, Rgba};

use crate::element::Element;
use crate::error::{CvError, Result, invalid};
use crate::mat::Mat;

macro_rules! image_interop {
    ($pixel:ident, $sub:ty, $cn:expr) => {
        impl From<ImageBuffer<$pixel<$sub>, Vec<$sub>>> for Mat<$sub> {
            /// Takes the buffer without copying; channel order stays as stored (RGB for color).
            fn from(img: ImageBuffer<$pixel<$sub>, Vec<$sub>>) -> Self {
                let (w, h) = img.dimensions();
                Mat::from_vec(w as usize, h as usize, $cn, img.into_raw())
                    .unwrap_or_else(|_| Mat::zeros(0, 0, $cn))
            }
        }

        impl TryFrom<Mat<$sub>> for ImageBuffer<$pixel<$sub>, Vec<$sub>> {
            type Error = CvError;

            /// Hands the buffer to `image` without copying.
            fn try_from(mat: Mat<$sub>) -> Result<Self> {
                if mat.channels() != $cn {
                    return Err(CvError::Channels {
                        op: "into image buffer",
                        channels: mat.channels(),
                    });
                }
                let (w, h) = (mat.width() as u32, mat.height() as u32);
                ImageBuffer::from_raw(w, h, mat.into_vec())
                    .ok_or_else(|| invalid("into image buffer", "size mismatch"))
            }
        }
    };
}

image_interop!(Luma, u8, 1);
image_interop!(LumaA, u8, 2);
image_interop!(Rgb, u8, 3);
image_interop!(Rgba, u8, 4);
image_interop!(Luma, u16, 1);
image_interop!(Rgb, u16, 3);
image_interop!(Rgba, u16, 4);
image_interop!(Rgb, f32, 3);
image_interop!(Rgba, f32, 4);

impl<T: Element> Mat<T> {
    /// Swap the first and third channel in place (BGR <-> RGB, BGRA <-> RGBA).
    pub fn swap_rb(&mut self) {
        let cn = self.channels();
        if cn >= 3 {
            for px in self.data_mut().chunks_exact_mut(cn) {
                px.swap(0, 2);
            }
        }
    }

    /// The same image with the first and third channel swapped.
    #[must_use]
    pub fn swapped_rb(mut self) -> Self {
        self.swap_rb();
        self
    }
}

/// What `imdecode` should return (`cv::ImreadModes`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ReadMode {
    /// 8-bit, 3-channel BGR.
    #[default]
    Color,
    /// 8-bit single channel.
    Grayscale,
    /// Native channels (gray, BGR or BGRA) at 8 bits.
    Unchanged,
}

/// A decoded image in OpenCV channel order.
#[derive(Clone, Debug)]
pub enum Decoded {
    U8(Mat<u8>),
    U16(Mat<u16>),
}

/// `cv::imdecode`: color images come back BGR(A) like OpenCV.
pub fn imdecode(bytes: &[u8], mode: ReadMode) -> Result<Decoded> {
    let img = image::load_from_memory(bytes)?;
    Ok(match mode {
        ReadMode::Color => Decoded::U8(Mat::from(img.to_rgb8()).swapped_rb()),
        ReadMode::Grayscale => {
            let bgr = Mat::from(img.to_rgb8()).swapped_rb();
            Decoded::U8(crate::color::cvt_color(
                &bgr,
                crate::color::ColorConversion::BgrToGray,
            )?)
        }
        ReadMode::Unchanged => match img {
            DynamicImage::ImageLuma8(g) => Decoded::U8(Mat::from(g)),
            DynamicImage::ImageLuma16(g) => Decoded::U16(Mat::from(g)),
            DynamicImage::ImageRgb16(c) => Decoded::U16(Mat::from(c).swapped_rb()),
            DynamicImage::ImageRgba16(c) => Decoded::U16(Mat::from(c).swapped_rb()),
            other if other.color().has_alpha() => Decoded::U8(Mat::from(other.to_rgba8()).swapped_rb()),
            other => Decoded::U8(Mat::from(other.to_rgb8()).swapped_rb()),
        },
    })
}

/// Output container of [`imencode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    Png,
    /// JPEG with `IMWRITE_JPEG_QUALITY` (0..=100).
    Jpeg(u8),
    /// Lossless WebP (the `image` crate has no lossy WebP encoder).
    WebpLossless,
}

/// `cv::imencode` for 8-bit gray, BGR or BGRA images.
pub fn imencode(mat: &Mat<u8>, encoding: Encoding) -> Result<Vec<u8>> {
    let rgb = mat.clone().swapped_rb();
    let dynamic = match mat.channels() {
        1 => DynamicImage::ImageLuma8(ImageBuffer::try_from(rgb)?),
        3 => DynamicImage::ImageRgb8(ImageBuffer::try_from(rgb)?),
        4 => DynamicImage::ImageRgba8(ImageBuffer::try_from(rgb)?),
        n => {
            return Err(CvError::Channels {
                op: "imencode",
                channels: n,
            });
        }
    };
    let mut out = Cursor::new(Vec::new());
    match encoding {
        Encoding::Png => dynamic.write_to(&mut out, ImageFormat::Png)?,
        Encoding::WebpLossless => dynamic.write_to(&mut out, ImageFormat::WebP)?,
        Encoding::Jpeg(quality) => {
            let rgb8 = if dynamic.color().has_alpha() {
                DynamicImage::ImageRgb8(dynamic.to_rgb8())
            } else {
                dynamic
            };
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality.clamp(1, 100));
            rgb8.write_with_encoder(encoder)?;
        }
    }
    Ok(out.into_inner())
}

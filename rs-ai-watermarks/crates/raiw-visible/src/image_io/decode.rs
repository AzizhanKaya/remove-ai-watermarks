//! Container decoders producing cv2's `IMREAD_UNCHANGED` layouts.

use std::io::Cursor;

use image::{DynamicImage, ImageFormat};
use raiw_cv::Mat;

use crate::error::{Result, VisibleError, unsupported};
use crate::raster::Pixels;

/// The container a file decoded from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Container {
    Png,
    Jpeg,
    WebP,
    Tiff,
    Bmp,
    Gif,
    Heif,
    Avif,
}

/// Sniff the container from magic bytes.
pub fn sniff(data: &[u8]) -> Option<Container> {
    if data.starts_with(raiw_core::constants::PNG_SIGNATURE) {
        return Some(Container::Png);
    }
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(Container::Jpeg);
    }
    if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        return Some(Container::WebP);
    }
    if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") {
        return Some(Container::Tiff);
    }
    if data.starts_with(b"BM") {
        return Some(Container::Bmp);
    }
    if data.starts_with(b"GIF8") {
        return Some(Container::Gif);
    }
    if data.len() >= 12 && &data[4..8] == b"ftyp" {
        let brand = &data[8..12];
        if brand == b"avif" || brand == b"avis" {
            return Some(Container::Avif);
        }
        if [b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"].iter().any(|b| brand == *b) {
            return Some(Container::Heif);
        }
    }
    None
}

fn bgr_from_rgb<T: Copy>(rgb: &[T], channels: usize) -> Vec<T> {
    let mut out = rgb.to_vec();
    for px in out.chunks_exact_mut(channels) {
        px.swap(0, 2);
    }
    out
}

fn gray_alpha_to_bgra<T: Copy>(ga: &[T]) -> Vec<T> {
    ga.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect()
}

/// Interleaved samples in cv2's channel order for an `image` color layout.
fn arrange<T: Copy + raiw_cv::Element>(w: usize, h: usize, samples: &[T], channels: usize) -> Result<Mat<T>> {
    let (data, cn) = match channels {
        1 => (samples.to_vec(), 1),
        2 => (gray_alpha_to_bgra(samples), 4),
        3 => (bgr_from_rgb(samples, 3), 3),
        4 => (bgr_from_rgb(samples, 4), 4),
        n => return Err(unsupported("decode", format!("{n} channels"))),
    };
    Ok(Mat::from_vec(w, h, cn, data)?)
}

/// `cv2` PNG decode: palette and low-bit expanded, tRNS to alpha except for grayscale, 16 bits kept.
fn decode_png(data: &[u8]) -> Result<Pixels> {
    let mut decoder = png::Decoder::new(Cursor::new(data));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(|e| unsupported("png", e.to_string()))?;
    let gray_trns = {
        let info = reader.info();
        info.trns.is_some() && info.color_type == png::ColorType::Grayscale
    };
    let size = reader.output_buffer_size().ok_or_else(|| unsupported("png", "image too large"))?;
    let mut buf = vec![0u8; size];
    let frame = reader.next_frame(&mut buf).map_err(|e| unsupported("png", e.to_string()))?;
    let (w, h) = (frame.width as usize, frame.height as usize);
    let mut channels = frame.color_type.samples();
    let wide = frame.bit_depth == png::BitDepth::Sixteen;
    buf.truncate(frame.buffer_size());
    if wide {
        let mut samples: Vec<u16> = buf.chunks_exact(2).map(|b| u16::from_be_bytes([b[0], b[1]])).collect();
        if gray_trns && channels == 2 {
            samples = samples.chunks_exact(2).map(|p| p[0]).collect();
            channels = 1;
        }
        return Ok(Pixels::U16(arrange(w, h, &samples, channels)?));
    }
    if gray_trns && channels == 2 {
        buf = buf.chunks_exact(2).map(|p| p[0]).collect();
        channels = 1;
    }
    Ok(Pixels::U8(arrange(w, h, &buf, channels)?))
}

#[cfg(feature = "turbojpeg")]
fn decode_jpeg(data: &[u8]) -> Result<Pixels> {
    let header = turbojpeg::read_header(data).map_err(|e| unsupported("jpeg", e.to_string()))?;
    let (format, cn) = if header.colorspace == turbojpeg::Colorspace::Gray {
        (turbojpeg::PixelFormat::GRAY, 1)
    } else {
        (turbojpeg::PixelFormat::BGR, 3)
    };
    match turbojpeg::decompress(data, format) {
        Ok(img) => {
            let (w, h) = (img.width, img.height);
            let mut out = Vec::with_capacity(w * h * cn);
            for row in img.pixels.chunks(img.pitch).take(h) {
                out.extend_from_slice(&row[..w * cn]);
            }
            Ok(Pixels::U8(Mat::from_vec(w, h, cn, out)?))
        }
        Err(_) => decode_generic(data, ImageFormat::Jpeg),
    }
}

#[cfg(not(feature = "turbojpeg"))]
fn decode_jpeg(data: &[u8]) -> Result<Pixels> {
    decode_generic(data, ImageFormat::Jpeg)
}

/// Decode through the `image` crate, keeping 8/16-bit depth and the alpha plane.
fn decode_generic(data: &[u8], format: ImageFormat) -> Result<Pixels> {
    let img = image::load_from_memory_with_format(data, format)?;
    from_dynamic(img)
}

pub(crate) fn from_dynamic(img: DynamicImage) -> Result<Pixels> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok(match img {
        DynamicImage::ImageLuma8(b) => Pixels::U8(arrange(w, h, b.as_raw(), 1)?),
        DynamicImage::ImageLumaA8(b) => Pixels::U8(arrange(w, h, b.as_raw(), 2)?),
        DynamicImage::ImageRgb8(b) => Pixels::U8(arrange(w, h, b.as_raw(), 3)?),
        DynamicImage::ImageRgba8(b) => Pixels::U8(arrange(w, h, b.as_raw(), 4)?),
        DynamicImage::ImageLuma16(b) => Pixels::U16(arrange(w, h, b.as_raw(), 1)?),
        DynamicImage::ImageLumaA16(b) => Pixels::U16(arrange(w, h, b.as_raw(), 2)?),
        DynamicImage::ImageRgb16(b) => Pixels::U16(arrange(w, h, b.as_raw(), 3)?),
        DynamicImage::ImageRgba16(b) => Pixels::U16(arrange(w, h, b.as_raw(), 4)?),
        other => Pixels::U8(arrange(w, h, other.to_rgba8().as_raw(), 4)?),
    })
}

/// Pillow's fallback layout for GIF: RGB unless any pixel is transparent.
fn decode_gif(data: &[u8]) -> Result<Pixels> {
    let rgba = image::load_from_memory_with_format(data, ImageFormat::Gif)?.to_rgba8();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    if rgba.pixels().any(|p| p[3] < 255) {
        return Ok(Pixels::U8(arrange(w, h, rgba.as_raw(), 4)?));
    }
    let rgb: Vec<u8> = rgba.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
    Ok(Pixels::U8(arrange(w, h, &rgb, 3)?))
}

#[cfg(feature = "heif")]
pub(crate) fn decode_heif(data: &[u8], apply_transforms: bool) -> Result<Pixels> {
    use libheif_rs::{ColorSpace, DecodingOptions, HeifContext, LibHeif, RgbChroma};
    let heif_err = |e: libheif_rs::HeifError| unsupported("heif", e.to_string());
    let lib = LibHeif::new();
    let ctx = HeifContext::read_from_bytes(data).map_err(heif_err)?;
    let handle = ctx.primary_image_handle().map_err(heif_err)?;
    let alpha = handle.has_alpha_channel();
    let mut options = DecodingOptions::new().ok_or_else(|| unsupported("heif", "decoding options"))?;
    options.set_ignore_transformations(!apply_transforms);
    let chroma = if alpha { RgbChroma::Rgba } else { RgbChroma::Rgb };
    let decoded = lib.decode(&handle, ColorSpace::Rgb(chroma), Some(options)).map_err(heif_err)?;
    let planes = decoded.planes();
    let plane = planes.interleaved.ok_or_else(|| unsupported("heif", "no interleaved plane"))?;
    let cn = if alpha { 4 } else { 3 };
    let (w, h) = (plane.width as usize, plane.height as usize);
    let mut rgb = Vec::with_capacity(w * h * cn);
    for row in plane.data.chunks(plane.stride).take(h) {
        rgb.extend_from_slice(&row[..w * cn]);
    }
    Ok(Pixels::U8(arrange(w, h, &rgb, cn)?))
}

#[cfg(not(feature = "heif"))]
pub(crate) fn decode_heif(_data: &[u8], _apply_transforms: bool) -> Result<Pixels> {
    Err(unsupported("heif", "HEIC/HEIF/AVIF decoding needs the `heif` feature"))
}

/// Decode `data` the way the Python pixel path does (cv2, then the Pillow fallback).
pub fn decode(data: &[u8]) -> Result<(Pixels, Container)> {
    let container = sniff(data).ok_or_else(|| unsupported("decode", "unrecognized image container"))?;
    let pixels = match container {
        Container::Png => decode_png(data)?,
        Container::Jpeg => decode_jpeg(data)?,
        Container::WebP => decode_generic(data, ImageFormat::WebP)?,
        Container::Tiff => decode_generic(data, ImageFormat::Tiff)?,
        Container::Bmp => decode_generic(data, ImageFormat::Bmp)?,
        Container::Gif => decode_gif(data)?,
        Container::Heif => decode_heif(data, true)?,
        Container::Avif => decode_heif(data, false)?,
    };
    if pixels.is_empty() {
        return Err(VisibleError::Unsupported { op: "decode", reason: "empty raster".into() });
    }
    Ok((pixels, container))
}

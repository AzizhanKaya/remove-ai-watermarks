//! The shared image codec boundary (`image_io.py`): cv2-layout decodes, quality-preserving
//! encodes, alpha handling, 16-bit pass-through, and display-tag carry with explicit decode state.

mod decode;
mod display;

use std::io::Cursor;
use std::path::Path;

use image::{ExtendedColorType, ImageEncoder};
use raiw_cv::{Mat, flip, transpose};

pub use decode::Container;
pub use display::{jpeg_splice_display_tags, png_splice_display_tags};

use crate::error::{Result, VisibleError, unsupported};
use crate::raster::{Image, Pixels};

/// Container extensions that carry an alpha channel.
pub const ALPHA_FORMATS: [&str; 5] = [".png", ".webp", ".heic", ".heif", ".avif"];

/// Extensions written through libheif instead of the cv2-equivalent encoders.
pub const HEIF_WRITE_EXTS: [&str; 3] = [".heic", ".heif", ".avif"];

/// cv2 read flags the package uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReadMode {
    /// `IMREAD_UNCHANGED`: native depth and channels, never rotated (TIFF/HEIF readers turn anyway).
    #[default]
    Unchanged,
    /// `IMREAD_COLOR`: 8-bit BGR with EXIF orientation applied for JPEG and PNG.
    Color,
}

/// Whether the decode turned the raster upright; dimensions alone cannot tell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeState {
    /// The stored raster.
    #[default]
    Raw,
    /// EXIF orientation (or a container transform) was applied.
    OrientationApplied,
}

/// A decoded file.
#[derive(Clone, Debug)]
pub struct Decoded {
    pub pixels: Pixels,
    pub container: Container,
    pub state: DecodeState,
}

/// `read_bgr_and_alpha`: BGR at source depth plus the separate alpha plane, if any.
#[derive(Clone, Debug)]
pub struct LoadedImage {
    pub bgr: Pixels,
    pub alpha: Option<Pixels>,
    pub container: Container,
    pub state: DecodeState,
}

fn orient_mat<T: raiw_cv::Element>(m: &Mat<T>, orientation: u16) -> Mat<T> {
    match orientation {
        2 => flip(m, true, false),
        3 => flip(m, true, true),
        4 => flip(m, false, true),
        5 => transpose(m),
        6 => flip(&transpose(m), true, false),
        7 => flip(&transpose(m), true, true),
        8 => flip(&transpose(m), false, true),
        _ => m.clone(),
    }
}

fn apply_orientation(pixels: Pixels, orientation: u16) -> Pixels {
    match pixels {
        Pixels::U8(m) => Pixels::U8(orient_mat(&m, orientation)),
        Pixels::U16(m) => Pixels::U16(orient_mat(&m, orientation)),
    }
}

/// `IMREAD_COLOR` narrowing: 16 bits by truncation (`png_set_strip_16`), then BGR.
fn to_color(pixels: Pixels) -> Result<Image> {
    let narrow = match pixels {
        Pixels::U8(m) => m,
        Pixels::U16(m) => m.map(|v| (v >> 8) as u8),
    };
    Ok(crate::raster::to_bgr(&narrow)?.into_owned())
}

/// Decode `data` with cv2's semantics for `mode`.
pub fn decode_bytes(data: &[u8], mode: ReadMode, tags_from: Option<&Path>) -> Result<Decoded> {
    let (mut pixels, container) = decode::decode(data)?;
    let mut state = match container {
        Container::Heif => DecodeState::OrientationApplied,
        _ => DecodeState::Raw,
    };
    let orientation = match (container, tags_from) {
        (Container::Tiff | Container::Jpeg | Container::Png, Some(path)) => display::source_tags(path).orientation,
        _ => None,
    };
    let rotate = match container {
        Container::Tiff => true,
        Container::Jpeg | Container::Png => mode == ReadMode::Color,
        _ => false,
    };
    if let Some(o) = orientation.filter(|&o| rotate && (2..=8).contains(&o)) {
        pixels = apply_orientation(pixels, o);
        state = DecodeState::OrientationApplied;
    }
    if mode == ReadMode::Color {
        pixels = Pixels::U8(to_color(pixels)?);
    }
    Ok(Decoded { pixels, container, state })
}

/// Unicode-safe `cv2.imread` with the Pillow fallbacks; errors replace Python's `None`.
pub fn imread(path: &Path, mode: ReadMode) -> Result<Decoded> {
    let data = std::fs::read(path).map_err(|_| VisibleError::Unreadable(path.to_path_buf()))?;
    if data.is_empty() {
        return Err(VisibleError::Unreadable(path.to_path_buf()));
    }
    decode_bytes(&data, mode, Some(path)).map_err(|_| VisibleError::Unreadable(path.to_path_buf()))
}

/// An 8-bit BGR read (`load_image_bgr`), failing loudly.
pub fn load_image_bgr(path: &Path) -> Result<Image> {
    match imread(path, ReadMode::Color)?.pixels {
        Pixels::U8(m) => Ok(m),
        Pixels::U16(_) => Err(VisibleError::Unreadable(path.to_path_buf())),
    }
}

fn split_alpha<T: raiw_cv::Element>(m: &Mat<T>) -> (Mat<T>, Option<Mat<T>>) {
    match m.channels() {
        1 => {
            let bgr = Mat::from_fn(m.width(), m.height(), 3, |x, y, _| m.at(x, y, 0));
            (bgr, None)
        }
        4 => {
            let bgr = Mat::from_fn(m.width(), m.height(), 3, |x, y, c| m.at(x, y, c));
            let alpha = Mat::from_fn(m.width(), m.height(), 1, |x, y, _| m.at(x, y, 3));
            (bgr, Some(alpha))
        }
        _ => (m.clone(), None),
    }
}

/// `read_bgr_and_alpha`: an `IMREAD_UNCHANGED` read with gray promoted and alpha split off.
pub fn load_image(path: &Path) -> Result<LoadedImage> {
    let decoded = imread(path, ReadMode::Unchanged)?;
    let (bgr, alpha) = match &decoded.pixels {
        Pixels::U8(m) => {
            let (b, a) = split_alpha(m);
            (Pixels::U8(b), a.map(Pixels::U8))
        }
        Pixels::U16(m) => {
            let (b, a) = split_alpha(m);
            (Pixels::U16(b), a.map(Pixels::U16))
        }
    };
    Ok(LoadedImage { bgr, alpha, container: decoded.container, state: decoded.state })
}

/// Where the pixels being written came from, for the display-tag carry.
#[derive(Clone, Copy, Debug, Default)]
pub struct WriteOptions<'a> {
    /// The file whose decode produced the pixels.
    pub display_tags_from: Option<&'a Path>,
    /// `Some(false)` for a raw decode, `Some(true)` for a transposed raster, `None` to infer.
    pub orientation_applied: Option<bool>,
}

impl<'a> WriteOptions<'a> {
    pub const fn from_source(source: &'a Path, state: DecodeState) -> Self {
        Self { display_tags_from: Some(source), orientation_applied: Some(matches!(state, DecodeState::OrientationApplied)) }
    }
}

/// `convertTo(CV_8U)` for encoders without 16-bit support (saturating, not scaled).
fn saturate_u8(pixels: &Pixels) -> Mat<u8> {
    match pixels {
        Pixels::U8(m) => m.clone(),
        Pixels::U16(m) => m.map(|v| v.min(255) as u8),
    }
}

fn rgb_order<T: raiw_cv::Element>(m: &Mat<T>) -> Vec<T> {
    let cn = m.channels();
    let mut out = m.data().to_vec();
    if cn >= 3 {
        for px in out.chunks_exact_mut(cn) {
            px.swap(0, 2);
        }
    }
    out
}

fn color_type(channels: usize, wide: bool) -> Result<ExtendedColorType> {
    Ok(match (channels, wide) {
        (1, false) => ExtendedColorType::L8,
        (3, false) => ExtendedColorType::Rgb8,
        (4, false) => ExtendedColorType::Rgba8,
        (1, true) => ExtendedColorType::L16,
        (3, true) => ExtendedColorType::Rgb16,
        (4, true) => ExtendedColorType::Rgba16,
        (n, _) => return Err(unsupported("imwrite", format!("{n} channels"))),
    })
}

fn encode_png(pixels: &Pixels) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let (w, h, cn) = (pixels.width() as u32, pixels.height() as u32, pixels.channels());
    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        &mut out,
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::Sub,
    );
    match pixels {
        Pixels::U8(m) => encoder.write_image(&rgb_order(m), w, h, color_type(cn, false)?)?,
        Pixels::U16(m) => {
            let bytes: Vec<u8> = rgb_order(m).iter().flat_map(|v| v.to_ne_bytes()).collect();
            encoder.write_image(&bytes, w, h, color_type(cn, true)?)?;
        }
    }
    Ok(out)
}

fn jpeg_input(pixels: &Pixels) -> Mat<u8> {
    let m = saturate_u8(pixels);
    if m.channels() == 4 { crate::eraser::select_channels(&m, 3) } else { m }
}

#[cfg(feature = "turbojpeg")]
fn encode_jpeg(pixels: &Pixels) -> Result<Vec<u8>> {
    let m = jpeg_input(pixels);
    let (format, subsamp) = if m.channels() == 1 {
        (turbojpeg::PixelFormat::GRAY, turbojpeg::Subsamp::Gray)
    } else {
        (turbojpeg::PixelFormat::BGR, turbojpeg::Subsamp::None)
    };
    let image = turbojpeg::Image { pixels: m.data(), width: m.width(), pitch: m.row_len(), height: m.height(), format };
    let buf = turbojpeg::compress(image, 100, subsamp).map_err(|e| unsupported("jpeg", e.to_string()))?;
    Ok(buf.to_vec())
}

#[cfg(not(feature = "turbojpeg"))]
fn encode_jpeg(pixels: &Pixels) -> Result<Vec<u8>> {
    let m = jpeg_input(pixels);
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 100).write_image(
        &rgb_order(&m),
        m.width() as u32,
        m.height() as u32,
        color_type(m.channels(), false)?,
    )?;
    Ok(out)
}

fn encode_webp(pixels: &Pixels, icc: Option<&[u8]>, orient: Option<u16>) -> Result<Vec<u8>> {
    let mut m = saturate_u8(pixels);
    if m.channels() == 1 {
        m = crate::raster::to_bgr(&m)?.into_owned();
    }
    let mut out = Vec::new();
    let mut encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut out);
    if let Some(profile) = icc {
        encoder.set_icc_profile(profile.to_vec()).map_err(|e| unsupported("webp", e.to_string()))?;
    }
    if let Some(exif) = display::orientation_exif(orient, false) {
        encoder.set_exif_metadata(exif).map_err(|e| unsupported("webp", e.to_string()))?;
    }
    encoder.write_image(&rgb_order(&m), m.width() as u32, m.height() as u32, color_type(m.channels(), false)?)?;
    Ok(out)
}

fn encode_with<E: ImageEncoder>(encoder: E, pixels: &Pixels) -> Result<()> {
    let (w, h, cn) = (pixels.width() as u32, pixels.height() as u32, pixels.channels());
    match pixels {
        Pixels::U8(m) => encoder.write_image(&rgb_order(m), w, h, color_type(cn, false)?)?,
        Pixels::U16(m) => {
            let bytes: Vec<u8> = rgb_order(m).iter().flat_map(|v| v.to_ne_bytes()).collect();
            encoder.write_image(&bytes, w, h, color_type(cn, true)?)?;
        }
    }
    Ok(())
}

#[cfg(feature = "heif")]
fn encode_heif(pixels: &Pixels, ext: &str, icc: Option<&[u8]>, orient: Option<u16>) -> Result<Vec<u8>> {
    use libheif_rs::{Channel, ColorSpace, CompressionFormat, EncoderQuality, HeifContext, Image as HeifImage, LibHeif, RgbChroma};
    let heif_err = |e: libheif_rs::HeifError| unsupported("heif", e.to_string());
    let m = saturate_u8(pixels);
    let m = match orient {
        Some(o) => orient_mat(&crate::raster::to_bgr(&m).map(std::borrow::Cow::into_owned).unwrap_or(m.clone()), o),
        None => m,
    };
    let m = if m.channels() == 1 { crate::raster::to_bgr(&m)?.into_owned() } else { m };
    let alpha = m.channels() == 4;
    let (w, h) = (m.width() as u32, m.height() as u32);
    let chroma = if alpha { RgbChroma::Rgba } else { RgbChroma::Rgb };
    let mut image = HeifImage::new(w, h, ColorSpace::Rgb(chroma)).map_err(heif_err)?;
    image.create_plane(Channel::Interleaved, w, h, 8).map_err(heif_err)?;
    {
        let planes = image.planes_mut();
        let plane = planes.interleaved.ok_or_else(|| unsupported("heif", "no interleaved plane"))?;
        let rgb = rgb_order(&m);
        let row = m.row_len();
        for (y, dst) in plane.data.chunks_mut(plane.stride).take(h as usize).enumerate() {
            dst[..row].copy_from_slice(&rgb[y * row..(y + 1) * row]);
        }
    }
    if let Some(profile) = icc {
        let _ = image.set_color_profile_raw(&libheif_rs::ColorProfileRaw::new(libheif_rs::color_profile_types::PROF, profile.to_vec()));
    }
    let lib = LibHeif::new();
    let format = if ext == ".avif" { CompressionFormat::Av1 } else { CompressionFormat::Hevc };
    let mut encoder = lib.encoder_for_format(format).map_err(heif_err)?;
    encoder.set_quality(EncoderQuality::Lossy(100)).map_err(heif_err)?;
    let mut ctx = HeifContext::new().map_err(heif_err)?;
    ctx.encode_image(&image, &mut encoder, None).map_err(heif_err)?;
    ctx.write_to_bytes().map_err(heif_err)
}

#[cfg(not(feature = "heif"))]
fn encode_heif(_pixels: &Pixels, _ext: &str, _icc: Option<&[u8]>, _orient: Option<u16>) -> Result<Vec<u8>> {
    Err(unsupported("imwrite", "HEIC/HEIF/AVIF encoding needs the `heif` feature"))
}

/// Encode `pixels` for `ext` at quality-preserving settings, with display tags attached.
pub fn encode(pixels: &Pixels, ext: &str, icc: Option<&[u8]>, orient: Option<u16>) -> Result<Vec<u8>> {
    if pixels.is_empty() {
        return Err(unsupported("imwrite", "empty image"));
    }
    if HEIF_WRITE_EXTS.contains(&ext) {
        return encode_heif(pixels, ext, icc, orient);
    }
    if raiw_core::constants::JPEG_SUFFIXES.contains(&ext) {
        return Ok(jpeg_splice_display_tags(&encode_jpeg(pixels)?, icc, orient));
    }
    match ext {
        ".png" => Ok(png_splice_display_tags(&encode_png(pixels)?, icc, orient)),
        ".webp" => encode_webp(pixels, icc, orient),
        ".bmp" => {
            let mut out = Vec::new();
            encode_with(image::codecs::bmp::BmpEncoder::new(&mut out), &Pixels::U8(saturate_u8(pixels)))?;
            Ok(out)
        }
        ".tif" | ".tiff" => {
            let mut out = Cursor::new(Vec::new());
            encode_with(image::codecs::tiff::TiffEncoder::new(&mut out), pixels)?;
            Ok(out.into_inner())
        }
        other => Err(unsupported("imwrite", format!("no encoder for {other:?}"))),
    }
}

/// Format-preserving, max-quality write (`imwrite`); the extension picks the format.
pub fn imwrite(path: &Path, pixels: &Pixels, options: &WriteOptions<'_>) -> Result<()> {
    let (icc, orient) = match options.display_tags_from {
        Some(source) => display::read_display_tags(source, pixels.height(), pixels.width(), options.orientation_applied),
        None => (None, None),
    };
    let ext = raiw_core::fs::suffix_lower(path);
    let ext = if ext.is_empty() { ".png".to_owned() } else { ext };
    let bytes = encode(pixels, &ext, icc.as_deref(), orient)?;
    std::fs::write(path, bytes).map_err(|_| VisibleError::WriteFailed(path.to_path_buf()))
}

fn join_alpha<T: raiw_cv::Element>(bgr: &Mat<T>, alpha: &Mat<T>) -> Result<Mat<T>> {
    if bgr.size() != alpha.size() || alpha.channels() != 1 || bgr.channels() != 3 {
        return Err(unsupported("save_image", "alpha plane does not match the BGR raster"));
    }
    Ok(Mat::from_fn(bgr.width(), bgr.height(), 4, |x, y, c| if c < 3 { bgr.at(x, y, c) } else { alpha.at(x, y, 0) }))
}

/// `write_bgr_with_alpha`: rejoin the original alpha plane when the container carries one.
pub fn save_image(path: &Path, bgr: &Pixels, alpha: Option<&Pixels>, options: &WriteOptions<'_>) -> Result<()> {
    let ext = raiw_core::fs::suffix_lower(path);
    let joined = match (alpha, bgr) {
        (Some(Pixels::U8(a)), Pixels::U8(b)) if ALPHA_FORMATS.contains(&ext.as_str()) => Some(Pixels::U8(join_alpha(b, a)?)),
        (Some(Pixels::U16(a)), Pixels::U16(b)) if ALPHA_FORMATS.contains(&ext.as_str()) => {
            Some(Pixels::U16(join_alpha(b, a)?))
        }
        (Some(_), _) if ALPHA_FORMATS.contains(&ext.as_str()) => {
            return Err(unsupported("save_image", "alpha depth differs from the BGR raster"));
        }
        _ => None,
    };
    imwrite(path, joined.as_ref().unwrap_or(bgr), options)
}

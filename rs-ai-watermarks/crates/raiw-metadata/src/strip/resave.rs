//! The Pillow open + save branch of `remove_ai_metadata` for 8-bit rasters.

use std::path::Path;

use super::png_lossless::keep_standard_text;
use super::png_writer::{PngExtras, Raster, encode as encode_png, text_chunk};
use super::{StripError, write_output};
use crate::exif;
use crate::exif_ai::scrub_ai_exif;
use crate::pil::png::{Ihdr, unfilter};
use crate::pil::{self, Image, Info, InfoValue, Plugin};

/// The decoded pixels in Pillow's mode layout (one byte per channel, palette indices for `P`).
#[derive(Debug, Clone)]
struct Pixels {
    mode: &'static str,
    width: u32,
    height: u32,
    data: Vec<u8>,
    palette: Option<Vec<u8>>,
}

fn channels(mode: &str) -> usize {
    match mode {
        "LA" => 2,
        "RGB" => 3,
        "RGBA" | "CMYK" => 4,
        _ => 1,
    }
}

fn expand_png(image: &Image, state: &pil::png::PngOpen, bytes: &[u8]) -> Result<Pixels, StripError> {
    let decoded = pil::png::decode(bytes, state, true);
    let raw = decoded.raw.ok_or(StripError::Decode)?;
    let ihdr: Ihdr = state.ihdr;
    let rows = unfilter(&ihdr, &raw).ok_or(StripError::Decode)?;
    let (w, h) = (ihdr.width as usize, ihdr.height as usize);
    let row_bytes = ihdr.row_bytes(w);
    let depth = ihdr.bit_depth as usize;
    let sample = |row: &[u8], index: usize| -> u16 {
        match depth {
            16 => u16::from_be_bytes([row[2 * index], row[2 * index + 1]]),
            8 => u16::from(row[index]),
            _ => {
                let per = 8 / depth;
                let shift = 8 - depth * (index % per + 1);
                u16::from((row[index / per] >> shift) & ((1u8 << depth) - 1))
            }
        }
    };
    let mode = image.mode;
    let mut data = Vec::with_capacity(w * h * channels(mode));
    for y in 0..h {
        let row = &rows[y * row_bytes..(y + 1) * row_bytes];
        match (ihdr.color_type, depth) {
            (0, 1) => (0..w).for_each(|x| data.push(if sample(row, x) != 0 { 255 } else { 0 })),
            (0, 2) => (0..w).for_each(|x| data.push((sample(row, x) * 85) as u8)),
            (0, 4) => (0..w).for_each(|x| data.push((sample(row, x) * 17) as u8)),
            (3, _) => (0..w).for_each(|x| data.push(sample(row, x) as u8)),
            (_, 8) => data.extend_from_slice(row),
            (0, 16) => (0..w).for_each(|x| data.extend_from_slice(&sample(row, x).to_be_bytes())),
            (4, 16) => (0..w).for_each(|x| {
                let gray = (sample(row, 2 * x) >> 8) as u8;
                data.extend_from_slice(&[gray, gray, gray, (sample(row, 2 * x + 1) >> 8) as u8]);
            }),
            (_, 16) => (0..w * ihdr.channels()).for_each(|i| data.push((sample(row, i) >> 8) as u8)),
            _ => return Err(StripError::Decode),
        }
    }
    Ok(Pixels {
        mode,
        width: ihdr.width,
        height: ihdr.height,
        data,
        palette: state.palette.clone(),
    })
}

#[cfg(feature = "turbojpeg")]
fn decode_jpeg(image: &Image, bytes: &[u8]) -> Result<Pixels, StripError> {
    let format = match image.mode {
        "L" => turbojpeg::PixelFormat::GRAY,
        "RGB" => turbojpeg::PixelFormat::RGB,
        _ => {
            return Err(StripError::Unsupported(format!(
                "cannot re-encode JPEG mode {}",
                image.mode
            )));
        }
    };
    let decoded = turbojpeg::decompress(bytes, format).map_err(|_| StripError::Decode)?;
    Ok(Pixels {
        mode: image.mode,
        width: decoded.width as u32,
        height: decoded.height as u32,
        data: decoded.pixels,
        palette: None,
    })
}

#[cfg(not(feature = "turbojpeg"))]
fn decode_jpeg(_image: &Image, _bytes: &[u8]) -> Result<Pixels, StripError> {
    Err(StripError::Unsupported(
        "JPEG decoding needs the turbojpeg feature".into(),
    ))
}

fn decode_generic(image: &Image, bytes: &[u8]) -> Result<Pixels, StripError> {
    let decoded = image::load_from_memory(bytes).map_err(|_| StripError::Decode)?;
    let (mode, data, (w, h)) = if image.mode == "RGBA" || image.mode == "LA" {
        let rgba = decoded.to_rgba8();
        ("RGBA", rgba.as_raw().clone(), rgba.dimensions())
    } else {
        let rgb = decoded.to_rgb8();
        ("RGB", rgb.as_raw().clone(), rgb.dimensions())
    };
    Ok(Pixels {
        mode,
        width: w,
        height: h,
        data,
        palette: None,
    })
}

fn decode(image: &Image, bytes: &[u8]) -> Result<Pixels, StripError> {
    match &image.plugin {
        Plugin::Png(state) => expand_png(image, state, bytes),
        Plugin::Jpeg(_) => decode_jpeg(image, bytes),
        _ => decode_generic(image, bytes),
    }
}

/// `img.convert("RGB")` for the modes that reach a JPEG or WebP save.
fn convert(pixels: &Pixels, target: &'static str, transparency: Option<&InfoValue>) -> Pixels {
    let n = channels(pixels.mode);
    let mut data = Vec::with_capacity(pixels.data.len() / n.max(1) * channels(target));
    let palette = pixels.palette.as_deref().unwrap_or_default();
    let alpha_of = |index: usize| -> u8 {
        match transparency {
            Some(InfoValue::Bytes(alpha)) => alpha.get(index).copied().unwrap_or(255),
            Some(InfoValue::Int(t)) if *t as usize == index => 0,
            _ => 255,
        }
    };
    for px in pixels.data.chunks_exact(n.max(1)) {
        let (rgb, alpha) = match pixels.mode {
            "P" => {
                let i = usize::from(px[0]);
                let rgb = palette
                    .get(3 * i..3 * i + 3)
                    .map_or([0, 0, 0], |c| [c[0], c[1], c[2]]);
                (rgb, alpha_of(i))
            }
            "L" | "1" => ([px[0]; 3], 255),
            "LA" => ([px[0]; 3], px[1]),
            "RGBA" => ([px[0], px[1], px[2]], px[3]),
            _ => ([px[0], px[1], px[2]], 255),
        };
        data.extend_from_slice(&rgb);
        if target == "RGBA" {
            data.push(alpha);
        }
    }
    Pixels {
        mode: target,
        width: pixels.width,
        height: pixels.height,
        data,
        palette: None,
    }
}

fn pack(pixels: &Pixels, bits: u8) -> Vec<u8> {
    if bits == 8 {
        return pixels.data.clone();
    }
    let w = pixels.width as usize;
    let per = 8 / bits as usize;
    let row_bytes = (w * bits as usize).div_ceil(8);
    let mut out = vec![0u8; row_bytes * pixels.height as usize];
    for (y, row) in pixels.data.chunks_exact(w.max(1)).enumerate() {
        for (x, &value) in row.iter().enumerate() {
            let value = if pixels.mode == "1" {
                u8::from(value != 0)
            } else {
                value
            };
            let shift = 8 - bits as usize * (x % per + 1);
            out[y * row_bytes + x / per] |= (value & ((1u16 << bits) - 1) as u8) << shift;
        }
    }
    out
}

fn dpi_of(info: &Info) -> Option<(f64, f64)> {
    let InfoValue::Tuple(items) = info.get("dpi")? else {
        return None;
    };
    let num = |v: &InfoValue| match v {
        InfoValue::Int(i) => Some(*i as f64),
        InfoValue::Float(f) => Some(*f),
        _ => None,
    };
    Some((num(items.first()?)?, num(items.get(1)?)?))
}

/// The PNG save of `remove_ai_metadata`'s Pillow branch.
fn save_png(pixels: &Pixels, info: &Info, extras: PngExtras) -> Result<Vec<u8>, StripError> {
    let mut extras = extras;
    let (bit_depth, color_type, rows) = match pixels.mode {
        "1" => (1, 0, pack(pixels, 1)),
        "L" => (8, 0, pixels.data.clone()),
        "I;16" => (16, 0, pixels.data.clone()),
        "LA" => (8, 4, pixels.data.clone()),
        "RGB" => (8, 2, pixels.data.clone()),
        "RGBA" => (8, 6, pixels.data.clone()),
        "P" => {
            let palette = pixels.palette.clone().unwrap_or_default();
            let colors = (palette.len() / 3).clamp(1, 256);
            let bits = match colors {
                0..=2 => 1,
                3..=4 => 2,
                5..=16 => 4,
                _ => 8,
            };
            let bytes = palette
                .get(..colors * 3)
                .map_or_else(|| b"\x00\x00\x00".to_vec(), <[u8]>::to_vec);
            extras.palette = Some(if bytes.is_empty() {
                b"\x00\x00\x00".to_vec()
            } else {
                bytes
            });
            match info.get("transparency") {
                Some(InfoValue::Bytes(alpha)) => {
                    extras.transparency = Some(alpha[..alpha.len().min(colors)].to_vec());
                }
                Some(InfoValue::Int(t)) => {
                    let t = (*t).clamp(0, 255) as usize;
                    let mut alpha = vec![0xFF; t];
                    alpha.push(0);
                    alpha.truncate(colors);
                    extras.transparency = Some(alpha);
                }
                _ => {}
            }
            (bits, 3, pack(pixels, bits))
        }
        other => {
            return Err(StripError::Unsupported(format!(
                "cannot write mode {other} as PNG"
            )));
        }
    };
    if pixels.mode != "P" {
        match (pixels.mode, info.get("transparency")) {
            ("1" | "L" | "I;16", Some(InfoValue::Int(t))) => {
                extras.transparency = Some(((*t).clamp(0, 65535) as u16).to_be_bytes().to_vec());
            }
            ("RGB", Some(InfoValue::Tuple(items))) if items.len() == 3 => {
                let mut trns = Vec::with_capacity(6);
                for item in items {
                    if let InfoValue::Int(v) = item {
                        trns.extend_from_slice(&(*v as u16).to_be_bytes());
                    }
                }
                extras.transparency = Some(trns);
            }
            _ => {}
        }
    }
    let raster = Raster {
        width: pixels.width,
        height: pixels.height,
        bit_depth,
        color_type,
        rows,
    };
    encode_png(&raster, &extras).map_err(|e| StripError::Unsupported(e.to_string()))
}

#[cfg(feature = "turbojpeg")]
fn save_jpeg(
    pixels: &Pixels,
    info: &Info,
    dpi: Option<(f64, f64)>,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, StripError> {
    let (format, subsamp) = match pixels.mode {
        "L" => (turbojpeg::PixelFormat::GRAY, turbojpeg::Subsamp::Gray),
        "RGB" => (turbojpeg::PixelFormat::RGB, turbojpeg::Subsamp::None),
        other => {
            return Err(StripError::Unsupported(format!(
                "cannot write mode {other} as JPEG"
            )));
        }
    };
    let image = turbojpeg::Image {
        pixels: pixels.data.as_slice(),
        width: pixels.width as usize,
        pitch: pixels.width as usize * format.size(),
        height: pixels.height as usize,
        format,
    };
    let mut compressor = turbojpeg::Compressor::new().map_err(|e| StripError::Unsupported(e.to_string()))?;
    compressor
        .set_quality(95)
        .map_err(|e| StripError::Unsupported(e.to_string()))?;
    compressor
        .set_subsamp(subsamp)
        .map_err(|e| StripError::Unsupported(e.to_string()))?;
    let mut encoded = compressor
        .compress_to_vec(image)
        .map_err(|e| StripError::Unsupported(e.to_string()))?;
    let comment = match info.get("comment") {
        Some(InfoValue::Bytes(b)) => Some(b.clone()),
        Some(v) => v.as_str().map(|s| s.as_bytes().to_vec()),
        None => None,
    };
    Ok(splice_jpeg_markers(
        &mut encoded,
        dpi,
        icc,
        exif,
        comment.as_deref(),
    ))
}

#[cfg(not(feature = "turbojpeg"))]
fn save_jpeg(
    _: &Pixels,
    _: &Info,
    _: Option<(f64, f64)>,
    _: Option<&[u8]>,
    _: Option<&[u8]>,
) -> Result<Vec<u8>, StripError> {
    Err(StripError::Unsupported(
        "JPEG encoding needs the turbojpeg feature".into(),
    ))
}

fn marker(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![0xFF, kind];
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// Pillow's marker layout after libjpeg's SOI + JFIF: EXIF, ICC (APP2), COM.
fn splice_jpeg_markers(
    encoded: &mut [u8],
    dpi: Option<(f64, f64)>,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    comment: Option<&[u8]>,
) -> Vec<u8> {
    let has_jfif = encoded.get(2..4) == Some(&[0xFF, 0xE0]) && encoded.get(6..11) == Some(b"JFIF\0");
    let jfif_end = if has_jfif {
        4 + usize::from(u16::from_be_bytes([encoded[4], encoded[5]]))
    } else {
        2
    };
    if has_jfif && let Some((x, y)) = dpi {
        let (x, y) = (crate::py::round_half_even(x), crate::py::round_half_even(y));
        if x > 0.0 && y > 0.0 {
            encoded[13] = 1;
            encoded[14..16].copy_from_slice(&(x as u16).to_be_bytes());
            encoded[16..18].copy_from_slice(&(y as u16).to_be_bytes());
        }
    }
    let mut out = encoded[..jfif_end].to_vec();
    if let Some(exif) = exif.filter(|e| !e.is_empty()) {
        out.extend(marker(0xE1, exif));
    }
    if let Some(icc) = icc.filter(|i| !i.is_empty()) {
        const MAX: usize = 65533 - 14;
        let parts: Vec<&[u8]> = icc.chunks(MAX).collect();
        for (i, part) in parts.iter().enumerate() {
            let mut payload = b"ICC_PROFILE\0".to_vec();
            payload.push((i + 1) as u8);
            payload.push(parts.len() as u8);
            payload.extend_from_slice(part);
            out.extend(marker(0xE2, &payload));
        }
    }
    if let Some(comment) = comment.filter(|c| !c.is_empty()) {
        out.extend(marker(0xFE, comment));
    }
    out.extend_from_slice(&encoded[jfif_end..]);
    out
}

fn save_webp(pixels: &Pixels, icc: Option<&[u8]>, exif: Option<&[u8]>) -> Result<Vec<u8>, StripError> {
    let mut encoded = Vec::new();
    let color = if pixels.mode == "RGBA" {
        image::ExtendedColorType::Rgba8
    } else {
        image::ExtendedColorType::Rgb8
    };
    image::codecs::webp::WebPEncoder::new_lossless(&mut encoded)
        .encode(&pixels.data, pixels.width, pixels.height, color)
        .map_err(|e| StripError::Unsupported(e.to_string()))?;
    let exif = exif
        .map(|e| e.strip_prefix(b"Exif\x00\x00").unwrap_or(e))
        .filter(|e| !e.is_empty());
    let icc = icc.filter(|i| !i.is_empty());
    if icc.is_none() && exif.is_none() {
        return Ok(encoded);
    }
    let body = encoded.get(12..).unwrap_or_default();
    let mut flags = 0u8;
    if icc.is_some() {
        flags |= 0x20;
    }
    if exif.is_some() {
        flags |= 0x08;
    }
    if pixels.mode == "RGBA" {
        flags |= 0x10;
    }
    let mut vp8x = vec![flags, 0, 0, 0];
    vp8x.extend_from_slice(&(pixels.width - 1).to_le_bytes()[..3]);
    vp8x.extend_from_slice(&(pixels.height - 1).to_le_bytes()[..3]);
    let riff_chunk = |kind: &[u8], data: &[u8]| {
        let mut out = kind.to_vec();
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        if data.len() % 2 == 1 {
            out.push(0);
        }
        out
    };
    let mut chunks = riff_chunk(b"VP8X", &vp8x);
    if let Some(icc) = icc {
        chunks.extend(riff_chunk(b"ICCP", icc));
    }
    chunks.extend_from_slice(body);
    if let Some(exif) = exif {
        chunks.extend(riff_chunk(b"EXIF", exif));
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&((chunks.len() + 4) as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend(chunks);
    Ok(out)
}

fn ext_format(suffix: &str) -> Option<&'static str> {
    match suffix {
        ".jpg" | ".jpeg" | ".jpe" | ".jfif" => Some("JPEG"),
        ".webp" => Some("WEBP"),
        ".png" => Some("PNG"),
        _ => None,
    }
}

/// The Pillow branch: decode, filter `info`, scrub EXIF and save in the chosen format.
pub fn resave(
    source: &Path,
    output: &Path,
    keep_standard: bool,
    true_fmt: Option<&'static str>,
) -> Result<(), StripError> {
    let probe = pil::probe(source);
    let image = probe.image().ok_or(StripError::Decode)?;
    let loaded = probe.loaded().ok_or(StripError::Decode)?;
    let source_exif = probe.exif_bytes();
    let xai_pair = crate::detect::xai_signature(source);
    let info = &loaded.info;
    let source_ext = ext_format(&raiw_core::fs::suffix_lower(source));
    let fmt = match true_fmt {
        Some(t) if Some(t) != source_ext => t,
        _ => ext_format(&raiw_core::fs::suffix_lower(output))
            .or(true_fmt)
            .unwrap_or("PNG"),
    };
    let mut pixels = decode(image, probe.bytes())?;
    match fmt {
        "JPEG" if matches!(pixels.mode, "RGBA" | "P") => pixels = convert(&pixels, "RGB", None),
        "WEBP" if pixels.mode == "P" => {
            let target = if info.contains_key("transparency") {
                "RGBA"
            } else {
                "RGB"
            };
            pixels = convert(&pixels, target, info.get("transparency"));
        }
        "WEBP" if !matches!(pixels.mode, "RGB" | "RGBA") => {
            let target = if matches!(pixels.mode, "LA" | "PA") {
                "RGBA"
            } else {
                "RGB"
            };
            pixels = convert(&pixels, target, None);
        }
        _ => {}
    }
    let mut kept: Vec<(String, InfoValue)> = Vec::new();
    let mut exif_dict = None;
    for (key, value) in info {
        if key == "exif" {
            if let Some(raw) = value.as_bytes()
                && let Ok(dict) = exif::load(raw)
            {
                exif_dict = Some(dict);
            }
            continue;
        }
        if key == "dpi" || key == "gamma" {
            continue;
        }
        if keep_standard_text(key, value.as_str(), keep_standard, xai_pair) {
            kept.push((key.clone(), value.clone()));
        }
    }
    if let Some(raw) = source_exif.as_deref().filter(|r| !r.is_empty())
        && let Ok(dict) = exif::load(raw)
    {
        exif_dict = Some(dict);
    }
    let dpi = dpi_of(info);
    let icc = match info.get("icc_profile") {
        Some(InfoValue::Bytes(icc)) if keep_standard && !icc.is_empty() => Some(icc.clone()),
        _ => None,
    };
    let exif_bytes = match exif_dict {
        Some(mut dict) if keep_standard => {
            scrub_ai_exif(&mut dict);
            exif::dump(&dict).ok()
        }
        _ => None,
    };
    let bytes = match fmt {
        "JPEG" => save_jpeg(&pixels, info, dpi, icc.as_deref(), exif_bytes.as_deref())?,
        "WEBP" => save_webp(&pixels, icc.as_deref(), exif_bytes.as_deref())?,
        _ => {
            let text = kept
                .iter()
                .map(|(key, value)| match value {
                    InfoValue::IText { text, lang, tkey } => text_chunk(key, text, Some((lang, tkey))),
                    other => text_chunk(key, &other.py_str(), None),
                })
                .collect();
            let extras = PngExtras {
                icc_profile: icc,
                text,
                dpi,
                exif: exif_bytes,
                ..PngExtras::default()
            };
            save_png(&pixels, info, extras)?
        }
    };
    write_output(output, &bytes)?;
    Ok(())
}

//! Pillow's PNG writer (`PngImagePlugin._save` + `ZipEncode.c`), byte for byte.

use crate::deflate::{DeflateError, PngFilter, Strategy, zip_encode, zlib_compress};

/// A raster in one of Pillow's PNG output raw modes, rows packed as PNG stores them.
#[derive(Debug, Clone)]
pub struct Raster {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: u8,
    /// Packed rows, `row_bytes` each.
    pub rows: Vec<u8>,
}

impl Raster {
    pub const fn bits_per_pixel(&self) -> usize {
        let channels = match self.color_type {
            2 => 3,
            4 => 2,
            6 => 4,
            _ => 1,
        };
        channels * self.bit_depth as usize
    }

    pub const fn row_bytes(&self) -> usize {
        (self.width as usize * self.bits_per_pixel()).div_ceil(8)
    }
}

/// One `PngInfo` text chunk.
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub kind: [u8; 4],
    pub data: Vec<u8>,
}

/// Everything `_save` writes besides the pixels, in the order it writes them.
#[derive(Debug, Clone, Default)]
pub struct PngExtras {
    pub icc_profile: Option<Vec<u8>>,
    pub text: Vec<TextChunk>,
    pub palette: Option<Vec<u8>>,
    pub transparency: Option<Vec<u8>>,
    pub dpi: Option<(f64, f64)>,
    pub exif: Option<Vec<u8>>,
}

fn chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    out.extend(raiw_container::png::serialize_chunk(&kind, data));
}

/// The IDAT payloads, one per `encoder.encode(bufsize)` call.
pub fn idat_chunks(raster: &Raster) -> Result<Vec<Vec<u8>>, DeflateError> {
    // Only the 8-bit `P` rawmode disables filtering; `P;1`/`P;2`/`P;4` are filtered.
    let palette = raster.color_type == 3 && raster.bit_depth == 8;
    let row_bytes = raster.row_bytes();
    let mut filter = PngFilter::new(row_bytes, raster.bits_per_pixel(), palette);
    let rows: Vec<Vec<u8>> = raster
        .rows
        .chunks_exact(row_bytes.max(1))
        .map(|row| filter.filter(row).to_vec())
        .collect();
    let strategy = if palette {
        Strategy::Default
    } else {
        Strategy::Filtered
    };
    zip_encode(rows, strategy, (raster.width as usize * 4).max(65536))
}

/// The complete file `img.save(path, "PNG", ...)` writes.
pub fn encode(raster: &Raster, extras: &PngExtras) -> Result<Vec<u8>, DeflateError> {
    let mut out = raiw_core::constants::PNG_SIGNATURE.to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&raster.width.to_be_bytes());
    ihdr.extend_from_slice(&raster.height.to_be_bytes());
    ihdr.extend_from_slice(&[raster.bit_depth, raster.color_type, 0, 0, 0]);
    chunk(&mut out, *b"IHDR", &ihdr);
    if let Some(icc) = extras.icc_profile.as_ref().filter(|icc| !icc.is_empty()) {
        let mut data = b"ICC Profile\0\0".to_vec();
        data.extend(zlib_compress(icc));
        chunk(&mut out, *b"iCCP", &data);
    }
    for text in &extras.text {
        chunk(&mut out, text.kind, &text.data);
    }
    if let Some(palette) = &extras.palette {
        chunk(&mut out, *b"PLTE", palette);
    }
    if let Some(trns) = &extras.transparency {
        chunk(&mut out, *b"tRNS", trns);
    }
    if let Some((x, y)) = extras.dpi {
        let mut phys = Vec::with_capacity(9);
        phys.extend_from_slice(&((x / 0.0254 + 0.5) as u32).to_be_bytes());
        phys.extend_from_slice(&((y / 0.0254 + 0.5) as u32).to_be_bytes());
        phys.push(1);
        chunk(&mut out, *b"pHYs", &phys);
    }
    if let Some(exif) = extras.exif.as_ref().filter(|e| !e.is_empty()) {
        let bare = exif.strip_prefix(b"Exif\x00\x00").unwrap_or(exif);
        chunk(&mut out, *b"eXIf", bare);
    }
    for part in idat_chunks(raster)? {
        chunk(&mut out, *b"IDAT", &part);
    }
    chunk(&mut out, *b"IEND", b"");
    Ok(out)
}

/// `PngInfo.add_text(key, value)` for a plain `str`, or `add_itxt` with the original tags.
pub fn text_chunk(key: &str, value: &str, itxt: Option<(&str, &str)>) -> TextChunk {
    let key_bytes = crate::py::latin1_encode_lossy(key);
    if let Some((lang, tkey)) = itxt {
        return itxt_chunk(&key_bytes, value, lang, tkey);
    }
    match crate::py::latin1_encode(value) {
        Some(latin) => {
            let mut data = key_bytes;
            data.push(0);
            data.extend(latin);
            TextChunk { kind: *b"tEXt", data }
        }
        None => itxt_chunk(&key_bytes, value, "", ""),
    }
}

fn itxt_chunk(key: &[u8], value: &str, lang: &str, tkey: &str) -> TextChunk {
    let mut data = key.to_vec();
    data.extend_from_slice(b"\0\0\0");
    data.extend_from_slice(lang.as_bytes());
    data.push(0);
    data.extend_from_slice(tkey.as_bytes());
    data.push(0);
    data.extend_from_slice(value.as_bytes());
    TextChunk { kind: *b"iTXt", data }
}

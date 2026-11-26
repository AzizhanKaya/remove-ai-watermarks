//! BMP and TIFF headers: enough of their plugins for `info`, mode and size.

use super::OpenError;
use super::ifd;
use super::value::{Info, InfoValue, float_pair};

#[derive(Debug, Clone)]
pub struct SimpleOpen {
    pub info: Info,
    pub mode: &'static str,
    pub size: (u32, u32),
    pub format: &'static str,
    /// TIFF only: the first IFD, for `getexif()` (`tag_v2`).
    pub tiff_ifd: Option<(ifd::RawIfd, bool)>,
}

fn le32(d: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(at..at + 4)?.try_into().ok()?))
}

fn le16(d: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(d.get(at..at + 2)?.try_into().ok()?))
}

/// `BmpImageFile._open` for the common DIB headers.
pub fn open_bmp(data: &[u8]) -> Result<SimpleOpen, OpenError> {
    if !data.starts_with(b"BM") {
        return Err(OpenError::Unidentified);
    }
    let header_size = le32(data, 14).ok_or(OpenError::Unidentified)?;
    let dib = data
        .get(18..14 + header_size as usize)
        .ok_or(OpenError::Unidentified)?;
    let mut info = Info::new();
    let (width, height, bits, compression) = match header_size {
        12 => (
            u32::from(le16(dib, 0).ok_or(OpenError::Unidentified)?),
            u32::from(le16(dib, 2).ok_or(OpenError::Unidentified)?),
            le16(dib, 6).ok_or(OpenError::Unidentified)?,
            0,
        ),
        40 | 52 | 56 | 64 | 108 | 124 => {
            let w = le32(dib, 0).ok_or(OpenError::Unidentified)?;
            let raw_h = le32(dib, 4).ok_or(OpenError::Unidentified)?;
            let h = if dib.get(7) == Some(&0xFF) {
                0u32.wrapping_sub(raw_h)
            } else {
                raw_h
            };
            let ppm = (le32(dib, 20).unwrap_or(0), le32(dib, 24).unwrap_or(0));
            if ppm != (0, 0) {
                info.insert(
                    "dpi".into(),
                    float_pair(f64::from(ppm.0) / 39.3701, f64::from(ppm.1) / 39.3701),
                );
            }
            (
                w,
                h,
                le16(dib, 10).ok_or(OpenError::Unidentified)?,
                le32(dib, 12).unwrap_or(0),
            )
        }
        _ => return Err(OpenError::Unidentified),
    };
    info.insert("compression".into(), InfoValue::Int(i64::from(compression)));
    let mode = match bits {
        1 => "1",
        4 | 8 => "P",
        16 | 24 | 32 => "RGB",
        _ => return Err(OpenError::Unidentified),
    };
    Ok(SimpleOpen {
        info,
        mode,
        size: (width, height),
        format: "BMP",
        tiff_ifd: None,
    })
}

const COMPRESSIONS: &[(i64, &str)] = &[
    (1, "raw"),
    (2, "tiff_ccitt"),
    (3, "group3"),
    (4, "group4"),
    (5, "tiff_lzw"),
    (6, "tiff_jpeg"),
    (7, "jpeg"),
    (8, "tiff_adobe_deflate"),
    (32771, "tiff_raw_16"),
    (32773, "packbits"),
    (32809, "tiff_thunderscan"),
    (32946, "tiff_deflate"),
    (34676, "tiff_sgilog"),
    (34677, "tiff_sgilog24"),
    (34925, "lzma"),
    (50000, "zstd"),
    (50001, "webp"),
];

/// `TiffImageFile._open` for classic TIFF: size, a coarse mode and the `info` keys.
pub fn open_tiff(data: &[u8]) -> Result<SimpleOpen, OpenError> {
    let (dir, little) = ifd::first_ifd(data).ok_or(OpenError::Unidentified)?;
    let int = |tag: u16| dir.get(&tag).and_then(|e| ifd::single_int(e, little));
    let width = int(256).ok_or(OpenError::Unidentified)? as u32;
    let height = int(257).ok_or(OpenError::Unidentified)? as u32;
    let samples = int(277).unwrap_or(1);
    let photometric = int(262).unwrap_or(1);
    let mode = match (photometric, samples) {
        (2, 3) => "RGB",
        (2, 4) => "RGBA",
        (5, _) => "CMYK",
        (3, _) => "P",
        _ => "L",
    };
    let mut info = Info::new();
    let compression = int(259).unwrap_or(1);
    let name = COMPRESSIONS
        .iter()
        .find(|(code, _)| *code == compression)
        .map_or("raw", |(_, name)| *name);
    info.insert("compression".into(), InfoValue::Str(name.into()));
    let resolution = |tag: u16| match dir.get(&tag).and_then(|e| ifd::single_rational(e, little)) {
        Some((n, d)) => InfoValue::Float(if d == 0 { f64::NAN } else { n as f64 / d as f64 }),
        None => InfoValue::Int(1),
    };
    let (x, y) = (resolution(282), resolution(283));
    let nonzero = |v: &InfoValue| !matches!(v, InfoValue::Float(f) if *f == 0.0);
    if nonzero(&x) && nonzero(&y) {
        let scaled = |v: &InfoValue| match v {
            InfoValue::Float(f) => InfoValue::Float(f * 2.54),
            InfoValue::Int(i) => InfoValue::Float(*i as f64 * 2.54),
            other => other.clone(),
        };
        let pair = |a: InfoValue, b: InfoValue| InfoValue::Tuple(vec![a, b]);
        match int(296) {
            Some(2) => {
                info.insert("dpi".into(), pair(x, y));
            }
            Some(3) => {
                info.insert("dpi".into(), pair(scaled(&x), scaled(&y)));
            }
            None => {
                info.insert("dpi".into(), pair(x.clone(), y.clone()));
                info.insert("resolution".into(), pair(x, y));
            }
            Some(_) => {
                info.insert("resolution".into(), pair(x, y));
            }
        }
    }
    if let Some((_, xmp)) = dir.get(&700) {
        info.insert("xmp".into(), InfoValue::Bytes(xmp.clone()));
    }
    if let Some((_, icc)) = dir.get(&34675) {
        info.insert("icc_profile".into(), InfoValue::Bytes(icc.clone()));
    }
    Ok(SimpleOpen {
        info,
        mode,
        size: (width, height),
        format: "TIFF",
        tiff_ifd: Some((dir, little)),
    })
}

//! `piexif.dump` and `piexif.insert` (JPEG), quirks included.

use super::{ExifDict, ExifValue, Ifd, IfdMap, tag, tags};

#[derive(Debug, thiserror::Error)]
pub enum DumpError {
    #[error("\"dump\" got wrong type of exif value: tag {0}")]
    WrongType(u16),
    #[error("struct.error packing tag {0}")]
    Pack(u16),
    #[error("Given thumbnail is too large. max 64kB")]
    ThumbnailTooLarge,
    #[error("invalid JPEG data")]
    Jpeg,
    #[error("Given data is not exif data")]
    NotExif,
}

type Result<T> = std::result::Result<T, DumpError>;

const TIFF_HEADER_LENGTH: usize = 8;

fn be16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// The integers of a value as `_pack_*(*raw_value)` sees them (`bytes` iterate as ints).
fn as_ints(value: &ExifValue, key: u16) -> Result<Vec<i64>> {
    match value {
        ExifValue::Int(v) => Ok(vec![*v]),
        ExifValue::Ints(v) => Ok(v.clone()),
        ExifValue::Bytes(b) => Ok(b.iter().map(|&x| i64::from(x)).collect()),
        ExifValue::Rational(n, d) => Ok(vec![*n, *d]),
        ExifValue::Rationals(_) | ExifValue::Float(_) | ExifValue::Floats(_) => Err(DumpError::Pack(key)),
    }
}

fn pack_ints(values: &[i64], width: usize, signed: bool, key: u16) -> Result<Vec<u8>> {
    let bits = 8 * width as u32;
    let (min, max) = if signed {
        (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
    } else {
        (0, (1i64 << bits) - 1)
    };
    let mut out = Vec::with_capacity(values.len() * width);
    for &v in values {
        if v < min || v > max {
            return Err(DumpError::Pack(key));
        }
        out.extend_from_slice(&(v as u64).to_be_bytes()[8 - width..]);
    }
    Ok(out)
}

fn as_floats(value: &ExifValue, key: u16) -> Result<Vec<f64>> {
    match value {
        ExifValue::Float(v) => Ok(vec![*v]),
        ExifValue::Floats(v) => Ok(v.clone()),
        ExifValue::Int(v) => Ok(vec![*v as f64]),
        ExifValue::Ints(v) => Ok(v.iter().map(|&x| x as f64).collect()),
        ExifValue::Bytes(b) => Ok(b.iter().map(|&x| f64::from(x)).collect()),
        ExifValue::Rational(n, d) => Ok(vec![*n as f64, *d as f64]),
        ExifValue::Rationals(_) => Err(DumpError::Pack(key)),
    }
}

fn rational_pairs(value: &ExifValue, key: u16) -> Result<Vec<(i64, i64)>> {
    match value {
        ExifValue::Rational(n, d) => Ok(vec![(*n, *d)]),
        ExifValue::Rationals(pairs) => Ok(pairs.clone()),
        ExifValue::Ints(v) if v.len() == 2 => Ok(vec![(v[0], v[1])]),
        ExifValue::Bytes(b) if b.len() == 2 => Ok(vec![(i64::from(b[0]), i64::from(b[1]))]),
        _ => Err(DumpError::WrongType(key)),
    }
}

/// `_value_to_bytes`: (count, inline field, out-of-line payload).
fn value_to_bytes(value: &ExifValue, kind: u16, offset: usize, key: u16) -> Result<(u32, Vec<u8>, Vec<u8>)> {
    let pointer = || be32(offset as u32).to_vec();
    let pad = |mut v: Vec<u8>, to: usize| {
        v.resize(to.max(v.len()), 0);
        v
    };
    let out = match kind {
        1 | 6 | 3 | 8 => {
            let width = if matches!(kind, 1 | 6) { 1 } else { 2 };
            let values = as_ints(value, key)?;
            let packed = pack_ints(&values, width, matches!(kind, 6 | 8), key)?;
            let n = values.len();
            if n * width <= 4 {
                (n as u32, pad(packed, 4), Vec::new())
            } else {
                (n as u32, pointer(), packed)
            }
        }
        4 | 9 => {
            let values = as_ints(value, key)?;
            let packed = pack_ints(&values, 4, kind == 9, key)?;
            if values.len() <= 1 {
                (values.len() as u32, packed, Vec::new())
            } else {
                (values.len() as u32, pointer(), packed)
            }
        }
        2 => {
            let ExifValue::Bytes(raw) = value else {
                return Err(DumpError::WrongType(key));
            };
            let mut terminated = raw.clone();
            terminated.push(0);
            let n = terminated.len();
            if n > 4 {
                (n as u32, pointer(), terminated)
            } else {
                (n as u32, pad(terminated, 4), Vec::new())
            }
        }
        5 | 10 => {
            let pairs = rational_pairs(value, key)?;
            let mut payload = Vec::with_capacity(pairs.len() * 8);
            for (num, den) in &pairs {
                payload.extend(pack_ints(&[*num, *den], 4, kind == 10, key)?);
            }
            (pairs.len() as u32, pointer(), payload)
        }
        7 => {
            let ExifValue::Bytes(raw) = value else {
                return Err(DumpError::WrongType(key));
            };
            let n = raw.len();
            if n > 4 {
                (n as u32, pointer(), raw.clone())
            } else {
                (n as u32, pad(raw.clone(), 4), Vec::new())
            }
        }
        11 => {
            let values = as_floats(value, key)?;
            let packed: Vec<u8> = values.iter().flat_map(|v| (*v as f32).to_be_bytes()).collect();
            if values.len() <= 1 {
                (values.len() as u32, packed, Vec::new())
            } else {
                (values.len() as u32, pointer(), packed)
            }
        }
        12 => {
            let values = as_floats(value, key)?;
            let packed: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
            (values.len() as u32, pointer(), packed)
        }
        _ => (0, Vec::new(), Vec::new()),
    };
    Ok(out)
}

/// `_dict_to_bytes`: (entry header + entries, out-of-line values).
fn dict_to_bytes(map: &IfdMap, ifd: Ifd, ifd_offset: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let tag_count = map.len();
    let entries_length = if matches!(ifd, Ifd::Zeroth | Ifd::First) {
        2 + tag_count * 12 + 4
    } else {
        2 + tag_count * 12
    };
    let mut entries = be16(tag_count as u16).to_vec();
    let mut values = Vec::new();
    let mut keys: Vec<u16> = map.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let skip = match ifd {
            Ifd::Zeroth => key == tag::EXIF_TAG || key == tag::GPS_TAG,
            Ifd::Exif => key == tag::INTEROPERABILITY_TAG,
            Ifd::First => key == tag::JPEG_INTERCHANGE_FORMAT || key == tag::JPEG_INTERCHANGE_FORMAT_LENGTH,
            _ => false,
        };
        if skip {
            continue;
        }
        let kind = tags::lookup(ifd, key).ok_or(DumpError::WrongType(key))?.2;
        let offset = TIFF_HEADER_LENGTH + entries_length + ifd_offset + values.len();
        let (count, inline, over) = value_to_bytes(&map[&key], kind, offset, key)?;
        entries.extend_from_slice(&be16(key));
        entries.extend_from_slice(&be16(kind));
        entries.extend_from_slice(&be32(count));
        entries.extend_from_slice(&inline);
        values.extend(over);
    }
    Ok((entries, values))
}

fn pointer_entry(key: u16, value: usize) -> Vec<u8> {
    [&be16(key)[..], &be16(4), &be32(1), &be32(value as u32)].concat()
}

fn thumbnail_bytes(jpeg: &[u8]) -> Result<Vec<u8>> {
    let mut segments = split_jpeg_segments(jpeg)?;
    while segments.len() > 1 && is_app_marker(&segments[1]) {
        segments.remove(1);
    }
    Ok(segments.concat())
}

fn is_app_marker(segment: &[u8]) -> bool {
    let head = &segment[..segment.len().min(2)];
    head >= &b"\xff\xe0"[..] && head <= &b"\xff\xef"[..]
}

/// `piexif.dump(exif_dict)`.
pub fn dump(dict: &ExifDict) -> Result<Vec<u8>> {
    let mut zeroth = dict.zeroth.clone();
    let mut exif = dict.exif.clone();
    let exif_is = !dict.exif.is_empty() || !dict.interop.is_empty();
    let mut interop_is = false;
    if exif_is {
        zeroth.insert(tag::EXIF_TAG, ExifValue::Int(1));
        if dict.interop.is_empty() {
            exif.shift_remove(&tag::INTEROPERABILITY_TAG);
        } else {
            exif.insert(tag::INTEROPERABILITY_TAG, ExifValue::Int(1));
            interop_is = true;
        }
    } else {
        zeroth.shift_remove(&tag::EXIF_TAG);
    }
    let gps_is = !dict.gps.is_empty();
    if gps_is {
        zeroth.insert(tag::GPS_TAG, ExifValue::Int(1));
    } else {
        zeroth.shift_remove(&tag::GPS_TAG);
    }
    let first_is = dict.thumbnail.is_some();
    let mut first = dict.first.clone();
    if first_is {
        first.insert(tag::JPEG_INTERCHANGE_FORMAT, ExifValue::Int(1));
        first.insert(tag::JPEG_INTERCHANGE_FORMAT_LENGTH, ExifValue::Int(1));
    }

    let zeroth_set = dict_to_bytes(&zeroth, Ifd::Zeroth, 0)?;
    let zeroth_length =
        zeroth_set.0.len() + usize::from(exif_is) * 12 + usize::from(gps_is) * 12 + 4 + zeroth_set.1.len();
    let (exif_set, exif_length) = if exif_is {
        let set = dict_to_bytes(&exif, Ifd::Exif, zeroth_length)?;
        let length = set.0.len() + usize::from(interop_is) * 12 + set.1.len();
        (Some(set), length)
    } else {
        (None, 0)
    };
    let gps_bytes = if gps_is {
        let set = dict_to_bytes(&dict.gps, Ifd::Gps, zeroth_length + exif_length)?;
        [set.0, set.1].concat()
    } else {
        Vec::new()
    };
    let interop_bytes = if interop_is {
        let set = dict_to_bytes(
            &dict.interop,
            Ifd::Interop,
            zeroth_length + exif_length + gps_bytes.len(),
        )?;
        [set.0, set.1].concat()
    } else {
        Vec::new()
    };
    let before_first = zeroth_length + exif_length + gps_bytes.len() + interop_bytes.len();
    let (first_bytes, first_pointer) = if first_is {
        let set = dict_to_bytes(&first, Ifd::First, before_first)?;
        let thumbnail = thumbnail_bytes(dict.thumbnail.as_deref().unwrap_or_default())?;
        if thumbnail.len() > 64000 {
            return Err(DumpError::ThumbnailTooLarge);
        }
        let pointer_value = TIFF_HEADER_LENGTH + before_first;
        let thumbnail_pointer = pointer_value + set.0.len() + 24 + 4 + set.1.len();
        let bytes = [
            &set.0[..],
            b"\x02\x01\x00\x04\x00\x00\x00\x01",
            &be32(thumbnail_pointer as u32),
            b"\x02\x02\x00\x04\x00\x00\x00\x01",
            &be32(thumbnail.len() as u32),
            b"\x00\x00\x00\x00",
            &set.1,
            &thumbnail,
        ]
        .concat();
        (bytes, be32(pointer_value as u32).to_vec())
    } else {
        (Vec::new(), vec![0; 4])
    };
    let exif_pointer = if exif_is {
        pointer_entry(tag::EXIF_TAG, TIFF_HEADER_LENGTH + zeroth_length)
    } else {
        Vec::new()
    };
    let gps_pointer = if gps_is {
        pointer_entry(tag::GPS_TAG, TIFF_HEADER_LENGTH + zeroth_length + exif_length)
    } else {
        Vec::new()
    };
    let interop_pointer = if interop_is {
        pointer_entry(
            tag::INTEROPERABILITY_TAG,
            TIFF_HEADER_LENGTH + zeroth_length + exif_length + gps_bytes.len(),
        )
    } else {
        Vec::new()
    };
    let mut out = b"Exif\x00\x00\x4d\x4d\x00\x2a\x00\x00\x00\x08".to_vec();
    out.extend(zeroth_set.0);
    out.extend(exif_pointer);
    out.extend(gps_pointer);
    out.extend(first_pointer);
    out.extend(zeroth_set.1);
    if let Some((entries, values)) = exif_set {
        out.extend(entries);
        out.extend(interop_pointer);
        out.extend(values);
    }
    out.extend(gps_bytes);
    out.extend(interop_bytes);
    out.extend(first_bytes);
    Ok(out)
}

/// `split_into_segments`: SOI, each marker segment, then SOS-to-end.
pub fn split_jpeg_segments(data: &[u8]) -> Result<Vec<Vec<u8>>> {
    if !data.starts_with(b"\xff\xd8") {
        return Err(DumpError::Jpeg);
    }
    let mut head = 2usize;
    let mut segments = vec![b"\xff\xd8".to_vec()];
    loop {
        if data.get(head..head + 2) == Some(b"\xff\xda") {
            segments.push(data[head..].to_vec());
            break;
        }
        let raw = data.get(head + 2..head + 4).ok_or(DumpError::Jpeg)?;
        let length = usize::from(u16::from_be_bytes([raw[0], raw[1]]));
        let end = head + length + 2;
        segments.push(data[head.min(data.len())..end.min(data.len())].to_vec());
        head = end;
        if head >= data.len() {
            return Err(DumpError::Jpeg);
        }
    }
    Ok(segments)
}

fn is_exif_app1(segment: &[u8]) -> bool {
    segment.starts_with(b"\xff\xe1") && segment.get(4..10) == Some(b"Exif\x00\x00")
}

/// `piexif.insert(exif, jpeg_bytes)`: the new JPEG bytes.
pub fn insert_jpeg(exif: &[u8], image: &[u8]) -> Result<Vec<u8>> {
    if !exif.starts_with(b"Exif\x00\x00") {
        return Err(DumpError::NotExif);
    }
    if !image.starts_with(b"\xff\xd8") {
        return Err(DumpError::Jpeg);
    }
    let length = u16::try_from(exif.len() + 2).map_err(|_| DumpError::Pack(0))?;
    let app1 = [&b"\xff\xe1"[..], &length.to_be_bytes(), exif].concat();
    let mut segments = split_jpeg_segments(image)?;
    let seg1_is_app0 = segments[1].starts_with(b"\xff\xe0");
    if seg1_is_app0 && segments.get(2).is_some_and(|s| is_exif_app1(s)) {
        segments[2] = app1;
        segments.remove(1);
    } else if seg1_is_app0 {
        if segments.len() < 3 && segments[1].starts_with(b"\xff\xe0") {
            // `segments[2]` is evaluated by piexif and raises on a two-segment list.
            return Err(DumpError::Jpeg);
        }
        segments[1] = app1;
    } else if is_exif_app1(&segments[1]) {
        segments[1] = app1;
    } else {
        segments.insert(1, app1);
    }
    Ok(segments.concat())
}

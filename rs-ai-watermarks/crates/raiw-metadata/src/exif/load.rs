//! `piexif.load` over bytes and over a JPEG file path.

use std::io::{self, Read};
use std::path::Path;

use super::{ExifDict, ExifValue, Ifd, IfdMap, tag, tags};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("Given file is neither JPEG nor TIFF.")]
    NotExif,
    #[error("Given data isn't JPEG.")]
    NotJpeg,
    #[error("Wrong JPEG data.")]
    WrongJpeg,
    #[error("not WebP or no exif chunk")]
    WebP,
    #[error("truncated EXIF structure")]
    Truncated,
    #[error("Exif might be wrong. Got incorrect value type to decode. type: {0}")]
    BadType(u16),
    #[error("non-integer IFD pointer")]
    BadPointer,
    #[error(transparent)]
    Io(#[from] io::Error),
}

type Result<T> = std::result::Result<T, LoadError>;

/// `piexif.load(data)` for in-memory bytes.
pub fn load(data: &[u8]) -> Result<ExifDict> {
    let tiftag = if data.starts_with(b"\xff\xd8") {
        let segments = super::split_jpeg_segments(data).map_err(|_| LoadError::WrongJpeg)?;
        segments
            .iter()
            .find(|seg| seg.starts_with(b"\xff\xe1") && seg.get(4..10) == Some(b"Exif\x00\x00"))
            .map(|seg| seg.get(10..).unwrap_or_default().to_vec())
    } else if data.starts_with(b"II") || data.starts_with(b"MM") {
        Some(data.to_vec())
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
        webp_exif(data)?
    } else if data.starts_with(b"Exif") {
        Some(data.get(6..).unwrap_or_default().to_vec())
    } else {
        return Err(LoadError::NotExif);
    };
    tiftag.map_or_else(|| Ok(ExifDict::default()), |tiftag| parse(&tiftag))
}

/// `piexif.load(str(path))` for a JPEG file (the streaming `read_exif_from_file`).
pub fn load_jpeg_file(path: &Path) -> Result<ExifDict> {
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0u8; 2];
    let got = read_up_to(&mut file, &mut magic)?;
    if got < 2 {
        return Err(LoadError::NotExif);
    }
    if magic != *b"\xff\xd8" {
        if magic == *b"II" || magic == *b"MM" {
            return parse(&std::fs::read(path)?);
        }
        let data = std::fs::read(path)?;
        if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
            return webp_exif(&data)?.map_or_else(|| Ok(ExifDict::default()), |t| parse(&t));
        }
        return Err(LoadError::NotExif);
    }
    let data = std::fs::read(path)?;
    match read_exif_segment(&data) {
        Some(app1) => parse(app1.get(10..).unwrap_or_default()),
        None => Ok(ExifDict::default()),
    }
}

fn read_up_to(file: &mut std::fs::File, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// `read_exif_from_file` over the file bytes: the first APP1 whose payload starts `Exif`.
fn read_exif_segment(data: &[u8]) -> Option<Vec<u8>> {
    let mut pos = 2usize;
    let take = |pos: usize, n: usize| &data[pos.min(data.len())..(pos + n).min(data.len())];
    loop {
        let head = take(pos, 4);
        if head.len() != 4 {
            return None;
        }
        pos += 4;
        let length = i64::from(u16::from_be_bytes([head[2], head[3]]));
        let body_len = if length - 2 < 0 {
            usize::MAX / 2
        } else {
            (length - 2) as usize
        };
        if head.starts_with(b"\xff\xe1") {
            let segment = take(pos, body_len);
            pos = pos.saturating_add(body_len);
            if !segment.starts_with(b"Exif") {
                continue;
            }
            return Some([head, segment].concat());
        } else if head[0] == 0xFF {
            pos = pos.saturating_add(body_len);
        } else {
            return None;
        }
    }
}

fn webp_exif(data: &[u8]) -> Result<Option<Vec<u8>>> {
    if data.get(12..16) != Some(b"VP8X") {
        return Err(LoadError::WebP);
    }
    let file_size = 8 + u64::from(u32_le(data, 4).ok_or(LoadError::Truncated)?);
    let mut pointer = 12usize;
    while (pointer as u64) < file_size {
        let fourcc = data.get(pointer..pointer + 4).unwrap_or_default().to_vec();
        pointer += 4;
        let mut length = u32_le(data, pointer).ok_or(LoadError::Truncated)? as usize;
        if length % 2 == 1 {
            length += 1;
        }
        pointer += 4;
        if fourcc == b"EXIF" {
            let end = pointer.saturating_add(length).min(data.len());
            return Ok(Some(data[pointer.min(end)..end].to_vec()));
        }
        pointer += length;
    }
    Ok(None)
}

fn u32_le(data: &[u8], at: usize) -> Option<u32> {
    let raw = data.get(at..at + 4)?;
    Some(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

struct Reader<'a> {
    data: &'a [u8],
    little: bool,
}

impl Reader<'_> {
    fn slice(&self, start: usize, len: usize) -> &[u8] {
        let start = start.min(self.data.len());
        &self.data[start..start.saturating_add(len).min(self.data.len())]
    }

    fn exact(&self, start: usize, len: usize) -> Result<&[u8]> {
        let slice = self.slice(start, len);
        if slice.len() == len {
            Ok(slice)
        } else {
            Err(LoadError::Truncated)
        }
    }

    fn u16_at(&self, at: usize) -> Result<u16> {
        let raw = self.exact(at, 2)?;
        Ok(if self.little {
            u16::from_le_bytes([raw[0], raw[1]])
        } else {
            u16::from_be_bytes([raw[0], raw[1]])
        })
    }

    fn u32_of(&self, raw: &[u8]) -> Result<u32> {
        if raw.len() != 4 {
            return Err(LoadError::Truncated);
        }
        Ok(if self.little {
            u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
        } else {
            u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]])
        })
    }

    fn ints(&self, raw: &[u8], width: usize, signed: bool) -> Vec<i64> {
        raw.chunks_exact(width)
            .map(|c| {
                let mut buf = [0u8; 8];
                let value = if self.little {
                    buf[..width].copy_from_slice(c);
                    u64::from_le_bytes(buf)
                } else {
                    buf[8 - width..].copy_from_slice(c);
                    u64::from_be_bytes(buf)
                };
                if signed {
                    let shift = 64 - 8 * width as u32;
                    ((value << shift) as i64) >> shift
                } else {
                    value as i64
                }
            })
            .collect()
    }

    /// The count-sized payload: inline when it fits in four bytes, else at the pointer.
    fn payload(&self, inline: &[u8], count: usize, width: usize, inline_limit: usize) -> Result<Vec<u8>> {
        let size = count.checked_mul(width).ok_or(LoadError::Truncated)?;
        if count > inline_limit {
            let pointer = self.u32_of(inline)? as usize;
            Ok(self.exact(pointer, size)?.to_vec())
        } else if width == 4 {
            // piexif unpacks the whole four-byte field for LONG/SLONG/FLOAT.
            if inline.len() == size {
                Ok(inline.to_vec())
            } else {
                Err(LoadError::Truncated)
            }
        } else {
            let slice = inline.get(..size).ok_or(LoadError::Truncated)?;
            Ok(slice.to_vec())
        }
    }

    fn convert(&self, kind: u16, count: u32, inline: &[u8]) -> Result<ExifValue> {
        let count = count as usize;
        let value = match kind {
            1 | 6 => {
                let raw = self.payload(inline, count, 1, 4)?;
                ints_value(self.ints(&raw, 1, kind == 6))
            }
            2 => {
                let end = |len: usize| {
                    if count == 0 {
                        len.saturating_sub(1)
                    } else {
                        count - 1
                    }
                };
                if count > 4 {
                    let pointer = self.u32_of(inline)? as usize;
                    ExifValue::Bytes(self.slice(pointer, count - 1).to_vec())
                } else {
                    ExifValue::Bytes(inline[..end(inline.len()).min(inline.len())].to_vec())
                }
            }
            3 | 8 => {
                let raw = self.payload(inline, count, 2, 2)?;
                ints_value(self.ints(&raw, 2, kind == 8))
            }
            4 | 9 => {
                let raw = self.payload(inline, count, 4, 1)?;
                ints_value(self.ints(&raw, 4, kind == 9))
            }
            5 | 10 => {
                let pointer = self.u32_of(inline)? as usize;
                let n = count.max(1);
                let raw = self.exact(pointer, n.checked_mul(8).ok_or(LoadError::Truncated)?)?;
                let values = self.ints(raw, 4, kind == 10);
                let pairs: Vec<(i64, i64)> = values.chunks_exact(2).map(|p| (p[0], p[1])).collect();
                if count > 1 {
                    ExifValue::Rationals(pairs)
                } else {
                    ExifValue::Rational(pairs[0].0, pairs[0].1)
                }
            }
            7 => {
                if count > 4 {
                    let pointer = self.u32_of(inline)? as usize;
                    ExifValue::Bytes(self.slice(pointer, count).to_vec())
                } else {
                    ExifValue::Bytes(inline[..count.min(inline.len())].to_vec())
                }
            }
            11 => {
                let raw = self.payload(inline, count, 4, 1)?;
                floats_value(
                    self.ints(&raw, 4, false)
                        .into_iter()
                        .map(|bits| f64::from(f32::from_bits(bits as u32)))
                        .collect(),
                )
            }
            12 => {
                let pointer = self.u32_of(inline)? as usize;
                let raw = self.exact(pointer, count.checked_mul(8).ok_or(LoadError::Truncated)?)?;
                floats_value(
                    raw.chunks_exact(8)
                        .map(|c| {
                            let mut buf = [0u8; 8];
                            buf.copy_from_slice(c);
                            if self.little {
                                f64::from_le_bytes(buf)
                            } else {
                                f64::from_be_bytes(buf)
                            }
                        })
                        .collect(),
                )
            }
            other => return Err(LoadError::BadType(other)),
        };
        Ok(value)
    }

    fn ifd(&self, pointer: usize, ifd: Ifd) -> Result<(IfdMap, Vec<u8>)> {
        let count = self.u16_at(pointer)? as usize;
        let offset = pointer + 2;
        let mut map = IfdMap::new();
        for x in 0..count {
            let at = offset + 12 * x;
            let number = self.u16_at(at)?;
            let kind = self.u16_at(at + 2)?;
            let value_num = self.u32_of(self.slice(at + 4, 4))?;
            let inline = self.slice(at + 8, 4);
            if tags::lookup(ifd, number).is_some() {
                let value = self.convert(kind, value_num, inline)?;
                map.insert(number, value);
            }
        }
        let next = if ifd == Ifd::Zeroth {
            self.slice(offset + 12 * count, 4).to_vec()
        } else {
            Vec::new()
        };
        Ok((map, next))
    }
}

fn ints_value(values: Vec<i64>) -> ExifValue {
    if values.len() == 1 {
        ExifValue::Int(values[0])
    } else {
        ExifValue::Ints(values)
    }
}

fn floats_value(values: Vec<f64>) -> ExifValue {
    if values.len() == 1 {
        ExifValue::Float(values[0])
    } else {
        ExifValue::Floats(values)
    }
}

fn pointer_of(map: &IfdMap, tag: u16) -> Result<Option<usize>> {
    match map.get(&tag) {
        None => Ok(None),
        Some(ExifValue::Int(value)) if *value >= 0 => Ok(Some(*value as usize)),
        Some(_) => Err(LoadError::BadPointer),
    }
}

/// The TIFF-structure half of `piexif.load` (`tiftag` already located).
fn parse(tiftag: &[u8]) -> Result<ExifDict> {
    let reader = Reader {
        data: tiftag,
        little: tiftag.starts_with(b"II"),
    };
    let mut out = ExifDict::default();
    let pointer = reader.u32_of(reader.slice(4, 4))? as usize;
    let (zeroth, first_pointer) = reader.ifd(pointer, Ifd::Zeroth)?;
    out.zeroth = zeroth;
    if let Some(at) = pointer_of(&out.zeroth, tag::EXIF_TAG)? {
        out.exif = reader.ifd(at, Ifd::Exif)?.0;
    }
    if let Some(at) = pointer_of(&out.zeroth, tag::GPS_TAG)? {
        out.gps = reader.ifd(at, Ifd::Gps)?.0;
    }
    if let Some(at) = pointer_of(&out.exif, tag::INTEROPERABILITY_TAG)? {
        out.interop = reader.ifd(at, Ifd::Interop)?.0;
    }
    if first_pointer != [0, 0, 0, 0] {
        let at = reader.u32_of(&first_pointer)? as usize;
        out.first = reader.ifd(at, Ifd::First)?.0;
        let start = out.first.get(&tag::JPEG_INTERCHANGE_FORMAT);
        let length = out.first.get(&tag::JPEG_INTERCHANGE_FORMAT_LENGTH);
        if let (Some(start), Some(length)) = (start, length) {
            let (Some(start), Some(length)) = (start.as_int(), length.as_int()) else {
                return Err(LoadError::BadPointer);
            };
            let end = start + length;
            let (s, e) = (start.max(0) as usize, end.max(0) as usize);
            let thumb = if s < e {
                reader.slice(s, e - s).to_vec()
            } else {
                Vec::new()
            };
            out.thumbnail = Some(thumb);
        }
    }
    Ok(out)
}

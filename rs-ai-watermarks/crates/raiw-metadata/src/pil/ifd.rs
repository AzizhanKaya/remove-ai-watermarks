//! Pillow's tolerant `ImageFileDirectory_v2.load` for the few tags the plugins read.

use indexmap::IndexMap;

/// Raw tag data of one directory: tag to (type, bytes).
pub type RawIfd = IndexMap<u16, (u16, Vec<u8>)>;

/// Unit sizes of the types Pillow decodes (`_load_dispatch`).
const fn unit_size(kind: u16) -> Option<usize> {
    match kind {
        1 | 2 | 6 | 7 => Some(1),
        3 | 8 => Some(2),
        4 | 9 | 11 | 13 => Some(4),
        5 | 10 | 12 | 16..=18 => Some(8),
        _ => None,
    }
}

/// A TIFF header and its byte order, or `None` (Pillow's `SyntaxError`).
pub fn header_endian(head: &[u8]) -> Option<bool> {
    match head.get(..4)? {
        b"II*\x00" | b"II+\x00" => Some(true),
        b"MM\x00*" | b"MM\x00+" => Some(false),
        _ => None,
    }
}

fn u16_at(data: &[u8], at: usize, little: bool) -> Option<u16> {
    let raw = data.get(at..at + 2)?;
    Some(if little {
        u16::from_le_bytes([raw[0], raw[1]])
    } else {
        u16::from_be_bytes([raw[0], raw[1]])
    })
}

pub fn u32_at(data: &[u8], at: usize, little: bool) -> Option<u32> {
    let raw = data.get(at..at + 4)?;
    let arr = [raw[0], raw[1], raw[2], raw[3]];
    Some(if little {
        u32::from_le_bytes(arr)
    } else {
        u32::from_be_bytes(arr)
    })
}

/// Load the directory at `offset`; a short read ends the walk with what was read.
pub fn load(data: &[u8], offset: usize, little: bool) -> RawIfd {
    let mut out = RawIfd::new();
    let Some(count) = u16_at(data, offset, little) else {
        return out;
    };
    for i in 0..usize::from(count) {
        let at = offset + 2 + 12 * i;
        if data.len() < at + 12 {
            break;
        }
        let (Some(tag), Some(kind), Some(n)) = (
            u16_at(data, at, little),
            u16_at(data, at + 2, little),
            u32_at(data, at + 4, little),
        ) else {
            break;
        };
        let Some(unit) = unit_size(kind) else {
            continue;
        };
        let size = n as usize * unit;
        let value = if size > 4 {
            let Some(ptr) = u32_at(data, at + 8, little) else {
                break;
            };
            match data.get(ptr as usize..ptr as usize + size) {
                Some(slice) => slice.to_vec(),
                None => break,
            }
        } else {
            data[at + 8..at + 8 + size].to_vec()
        };
        if value.is_empty() {
            continue;
        }
        out.insert(tag, (kind, value));
    }
    out
}

/// The first IFD of a TIFF blob (`Exif.load` without the `Exif\0\0` prefixes).
pub fn first_ifd(tiff: &[u8]) -> Option<(RawIfd, bool)> {
    let little = header_endian(tiff)?;
    let offset = u32_at(tiff, 4, little)? as usize;
    Some((load(tiff, offset, little), little))
}

/// One unsigned integer of a SHORT/LONG/BYTE tag with a single value.
pub fn single_int(entry: &(u16, Vec<u8>), little: bool) -> Option<i64> {
    let (kind, data) = entry;
    match (kind, data.len()) {
        (1 | 7, 1) => Some(i64::from(data[0])),
        (3, 2) => u16_at(data, 0, little).map(i64::from),
        (4, 4) => u32_at(data, 0, little).map(i64::from),
        _ => None,
    }
}

/// One RATIONAL as `(numerator, denominator)`.
pub fn single_rational(entry: &(u16, Vec<u8>), little: bool) -> Option<(i64, i64)> {
    let (kind, data) = entry;
    (*kind == 5 && data.len() == 8)
        .then(|| {
            Some((
                i64::from(u32_at(data, 0, little)?),
                i64::from(u32_at(data, 4, little)?),
            ))
        })
        .flatten()
}

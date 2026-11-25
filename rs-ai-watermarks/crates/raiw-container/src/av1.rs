//! AV1 Open Bitstream Unit scanning for AI `OBU_METADATA` and same-length padding rewrites.
//!
//! HDR (`HDR_CLL`, `HDR_MDCV`), scalability and timecode metadata are preserved; a matching
//! unit becomes `OBU_PADDING` of identical length so no container offset moves.

use std::io::{self, Seek, SeekFrom, Write};

use memchr::memmem;

use crate::markers::{C2PA_UUID, contains};
use crate::source::ReadAt;

const OBU_METADATA: u8 = 5;
const OBU_PADDING: u8 = 15;
const METADATA_TYPE_ITUT_T35: u64 = 4;
const PRESERVED_METADATA_TYPES: [u64; 4] = [1, 2, 3, 5];
const MAX_METADATA_PAYLOAD_BYTES: u64 = 4 * 1024 * 1024;
static ZERO_CHUNK: [u8; 65536] = [0; 65536];

/// One AI `OBU_METADATA` unit, located by absolute offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObuPatch {
    pub header_offset: u64,
    pub header_byte: u8,
    pub payload_start: u64,
    pub payload_end: u64,
}

/// Decode an unsigned LEB128 integer (at most 8 bytes, below 2^32) at `pos`.
pub fn read_leb128(data: &[u8], pos: usize) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for i in 0..8 {
        let byte = *data.get(pos + i)?;
        value |= u64::from(byte & 0x7F) << (7 * i);
        if byte & 0x80 == 0 {
            return (value < 1 << 32).then_some((value, pos + i + 1));
        }
    }
    None
}

/// Encode `value` as minimal unsigned LEB128.
pub fn encode_leb128(mut value: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            return out;
        }
    }
}

/// Whether an `OBU_METADATA` payload carries AI provenance rather than HDR or timing data.
pub fn is_ai_obu_metadata_payload(payload: &[u8]) -> bool {
    let Some((meta_type, mut pos)) = read_leb128(payload, 0) else {
        return false;
    };
    if PRESERVED_METADATA_TYPES.contains(&meta_type) {
        return false;
    }
    if meta_type == METADATA_TYPE_ITUT_T35 {
        let Some(&country) = payload.get(pos) else {
            return false;
        };
        pos += if country == 0xFF { 2 } else { 1 };
        if pos > payload.len() {
            return false;
        }
    }
    let body = &payload[pos..];
    if body.is_empty() {
        return false;
    }
    if memmem::find(body, &C2PA_UUID).is_some() {
        return true;
    }
    let lower = body.to_ascii_lowercase();
    contains(&lower, b"urn:c2pa:")
        || (contains(body, b"jumb") && contains(&lower, b"c2pa"))
        || contains(&lower, b"kling-ai")
        || contains(&lower, b"trainedalgorithmicmedia")
        || (contains(&lower, b"\"contentproducer\"") && contains(&lower, b"\"label\""))
        || (contains(&lower, b"\"serviceprovider\"") && contains(&lower, b"\"aigc\""))
        || contains(&lower, b"tc260:aigc")
}

struct ObuHeader {
    obu_type: u8,
    byte: u8,
    payload_start: usize,
    size: Option<u64>,
}

fn parse_header(header: &[u8]) -> Option<ObuHeader> {
    let byte = *header.first()?;
    if byte & 0x81 != 0 {
        return None;
    }
    let header_len = 1 + usize::from((byte >> 2) & 1);
    if header_len > header.len() {
        return None;
    }
    let (size, payload_start) = if (byte >> 1) & 1 == 1 {
        let (size, next) = read_leb128(header, header_len)?;
        (Some(size), next)
    } else {
        (None, header_len)
    };
    Some(ObuHeader {
        obu_type: (byte >> 3) & 0x0F,
        byte,
        payload_start,
        size,
    })
}

/// Walk OBU headers in `[start, end)` of `source`, reading only metadata payloads.
pub fn scan_stream_spans<R: ReadAt + ?Sized>(
    source: &R,
    start: u64,
    end: u64,
    stop_on_first: bool,
) -> io::Result<Vec<ObuPatch>> {
    let mut found = Vec::new();
    let mut pos = start;
    while pos < end {
        let header_bytes = source.read_vec(pos, (end - pos).min(10))?;
        let Some(header) = parse_header(&header_bytes) else {
            break;
        };
        let payload_start = pos + header.payload_start as u64;
        let size = header.size.unwrap_or_else(|| end.saturating_sub(payload_start));
        if payload_start + size > end {
            break;
        }
        if header.obu_type == OBU_METADATA && size > 0 && size <= MAX_METADATA_PAYLOAD_BYTES {
            let payload = source.read_vec(payload_start, size)?;
            if payload.len() as u64 == size && is_ai_obu_metadata_payload(&payload) {
                found.push(ObuPatch {
                    header_offset: pos,
                    header_byte: header.byte,
                    payload_start,
                    payload_end: payload_start + size,
                });
                if stop_on_first {
                    return Ok(found);
                }
            }
        }
        pos = payload_start + size;
    }
    Ok(found)
}

/// AI `OBU_METADATA` units in an in-memory OBU sequence whose first byte sits at `base`.
pub fn scan_sample_spans(sample: &[u8], base: u64) -> Vec<ObuPatch> {
    let mut found = Vec::new();
    let mut pos = 0usize;
    while pos < sample.len() {
        let Some(header) = parse_header(&sample[pos..]) else {
            break;
        };
        let payload_start = pos + header.payload_start;
        let size = header
            .size
            .map_or(sample.len() - payload_start, |size| size as usize);
        if payload_start + size > sample.len() {
            break;
        }
        if header.obu_type == OBU_METADATA
            && size > 0
            && is_ai_obu_metadata_payload(&sample[payload_start..payload_start + size])
        {
            found.push(ObuPatch {
                header_offset: base + pos as u64,
                header_byte: header.byte,
                payload_start: base + payload_start as u64,
                payload_end: base + (payload_start + size) as u64,
            });
        }
        pos = payload_start + size;
    }
    found
}

/// Rewrite one unit in place as same-length `OBU_PADDING` with valid trailing bits.
pub fn patch_to_padding<W: Write + Seek + ?Sized>(target: &mut W, patch: &ObuPatch) -> io::Result<()> {
    let new_header = (patch.header_byte & 0x87) | (OBU_PADDING << 3);
    target.seek(SeekFrom::Start(patch.header_offset))?;
    target.write_all(&[new_header])?;
    let length = patch.payload_end.saturating_sub(patch.payload_start);
    if length > 0 {
        target.seek(SeekFrom::Start(patch.payload_start))?;
        let mut zeros = length - 1;
        while zeros > 0 {
            let step = zeros.min(ZERO_CHUNK.len() as u64);
            target.write_all(&ZERO_CHUNK[..step as usize])?;
            zeros -= step;
        }
        target.write_all(&[0x80])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn padding_keeps_length_and_trailing_bits() {
        let mut data = vec![0x2A, 0x03, 0x06, b'a', b'b', 0x12, 0x00];
        let patch = scan_sample_spans(&[0x2A, 0x03, 0x06, b'k', b'l'], 0);
        assert!(patch.is_empty());
        let patch = ObuPatch {
            header_offset: 0,
            header_byte: 0x2A,
            payload_start: 2,
            payload_end: 5,
        };
        patch_to_padding(&mut Cursor::new(&mut data), &patch).unwrap();
        assert_eq!(data, [0x7A, 0x03, 0x00, 0x00, 0x80, 0x12, 0x00]);
    }

    #[test]
    fn leb128_round_trips() {
        for value in [0, 1, 127, 128, 300, u64::from(u32::MAX)] {
            let encoded = encode_leb128(value);
            assert_eq!(read_leb128(&encoded, 0), Some((value, encoded.len())));
        }
        assert_eq!(read_leb128(&encode_leb128(1 << 32), 0), None);
    }
}

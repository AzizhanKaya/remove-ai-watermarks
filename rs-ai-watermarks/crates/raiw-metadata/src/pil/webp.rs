//! `WebPImagePlugin`: the demuxer checks libwebp makes on open, and `info`.

use super::OpenError;
use super::value::{Info, InfoValue};

#[derive(Debug, Clone)]
pub struct WebpOpen {
    pub info: Info,
    pub mode: &'static str,
    pub size: (u32, u32),
    pub n_frames: usize,
}

fn le32(d: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(at..at + 4)?.try_into().ok()?))
}

fn le24(d: &[u8], at: usize) -> Option<u32> {
    let b = d.get(at..at + 3)?;
    Some(u32::from(b[0]) | (u32::from(b[1]) << 8) | (u32::from(b[2]) << 16))
}

/// Dimensions and alpha of a `VP8 ` or `VP8L` bitstream (`VP8GetInfo` / `VP8LGetInfo`).
fn bitstream_info(kind: &[u8], payload: &[u8]) -> Option<(u32, u32, bool)> {
    match kind {
        b"VP8 " => {
            if payload.len() < 10 || payload[3..6] != [0x9d, 0x01, 0x2a] {
                return None;
            }
            let bits = u32::from(payload[0]) | (u32::from(payload[1]) << 8) | (u32::from(payload[2]) << 16);
            if bits & 1 != 0 || ((bits >> 1) & 7) > 3 || (bits >> 4) & 1 == 0 {
                return None;
            }
            let w = u32::from(u16::from_le_bytes([payload[6], payload[7]]) & 0x3fff);
            let h = u32::from(u16::from_le_bytes([payload[8], payload[9]]) & 0x3fff);
            (w > 0 && h > 0).then_some((w, h, false))
        }
        b"VP8L" => {
            if payload.len() < 5 || payload[0] != 0x2f {
                return None;
            }
            let bits = le32(payload, 1)?;
            if bits >> 29 != 0 {
                return None;
            }
            Some((
                (bits & 0x3fff) + 1,
                ((bits >> 14) & 0x3fff) + 1,
                (bits >> 28) & 1 == 1,
            ))
        }
        _ => None,
    }
}

/// `WebPAnimDecoder(s)` plus `get_info` / `get_chunk`, as `WebPImageFile._open` uses them.
pub fn open(data: &[u8]) -> Result<WebpOpen, OpenError> {
    let ok = data.starts_with(b"RIFF")
        && data.get(8..12) == Some(b"WEBP")
        && matches!(data.get(12..16), Some(b"VP8 " | b"VP8X" | b"VP8L"));
    if !ok {
        return Err(OpenError::Unidentified);
    }
    let fail = || OpenError::Io("could not create decoder object".into());
    let riff_size = le32(data, 4).ok_or_else(fail)? as usize;
    if riff_size < 12 || riff_size + 8 > data.len() {
        return Err(fail());
    }
    let end = riff_size + 8;
    let mut pos = 12;
    let mut chunks: Vec<(&[u8], &[u8])> = Vec::new();
    while pos + 8 <= end {
        let kind = &data[pos..pos + 4];
        let len = le32(data, pos + 4).ok_or_else(fail)? as usize;
        let start = pos + 8;
        if start + len > end {
            return Err(fail());
        }
        chunks.push((kind, &data[start..start + len]));
        pos = start + len + (len & 1);
    }
    let (first_kind, first) = *chunks.first().ok_or_else(fail)?;
    let mut info = Info::new();
    let (size, has_alpha, n_frames, loop_count, bgcolor) = if first_kind == b"VP8X" {
        if first.len() < 10 {
            return Err(fail());
        }
        let flags = first[0];
        let size = (
            le24(first, 4).ok_or_else(fail)? + 1,
            le24(first, 7).ok_or_else(fail)? + 1,
        );
        let animated = flags & 0x02 != 0;
        let (frames, loop_count, bgcolor) = if animated {
            let anim = chunks.iter().find(|(k, _)| *k == b"ANIM").ok_or_else(fail)?.1;
            let frames = chunks.iter().filter(|(k, _)| *k == b"ANMF").count();
            if frames == 0 || anim.len() < 6 {
                return Err(fail());
            }
            (
                frames,
                u32::from(u16::from_le_bytes([anim[4], anim[5]])),
                le32(anim, 0).ok_or_else(fail)?,
            )
        } else {
            let image = chunks.iter().find_map(|(k, p)| bitstream_info(k, p));
            if image.is_none() {
                return Err(fail());
            }
            (1, 1, 0xFFFF_FFFF)
        };
        (size, flags & 0x10 != 0, frames, loop_count, bgcolor)
    } else {
        // libwebp's demuxer defaults a still image to one loop and a white background.
        let (w, h, alpha) = bitstream_info(first_kind, first).ok_or_else(fail)?;
        ((w, h), alpha, 1, 1, 0xFFFF_FFFF)
    };
    info.insert("loop".into(), InfoValue::Int(i64::from(loop_count)));
    let channel = |shift: u32| InfoValue::Int(i64::from((bgcolor >> shift) & 0xFF));
    info.insert(
        "background".into(),
        InfoValue::Tuple(vec![channel(16), channel(8), channel(0), channel(24)]),
    );
    if first_kind == b"VP8X" {
        for (key, fourcc) in [("icc_profile", b"ICCP"), ("exif", b"EXIF"), ("xmp", b"XMP ")] {
            if let Some((_, payload)) = chunks.iter().find(|(k, _)| k == fourcc)
                && !payload.is_empty()
            {
                info.insert(key.into(), InfoValue::Bytes(payload.to_vec()));
            }
        }
    }
    Ok(WebpOpen {
        info,
        mode: if has_alpha { "RGBA" } else { "RGB" },
        size,
        n_frames,
    })
}

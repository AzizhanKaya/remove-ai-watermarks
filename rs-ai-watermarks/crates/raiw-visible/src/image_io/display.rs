//! Display-tag carry across re-encodes: the source's ICC profile and EXIF orientation.

use std::io::{Read, Write};
use std::path::Path;

use super::decode::{Container, sniff};

/// Orientations that transpose width and height.
const TRANSPOSING: [u16; 4] = [5, 6, 7, 8];

/// What a source file declares for display.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SourceTags {
    pub container: Option<Container>,
    pub icc: Option<Vec<u8>>,
    pub orientation: Option<u16>,
    /// `(width, height)` the container stores.
    pub stored: Option<(usize, usize)>,
}

fn be16(d: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*d.get(at)?, *d.get(at + 1)?]))
}

fn be32(d: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes([*d.get(at)?, *d.get(at + 1)?, *d.get(at + 2)?, *d.get(at + 3)?]))
}

/// The IFD0 entries of a bare TIFF structure as `(tag, value)`, plus the raw ICC (34675) bytes.
fn tiff_ifd0(tiff: &[u8]) -> (Vec<(u16, u32)>, Option<Vec<u8>>) {
    let little = tiff.starts_with(b"II");
    if !(little || tiff.starts_with(b"MM")) || tiff.len() < 8 {
        return (Vec::new(), None);
    }
    let u16_at = |at: usize| -> Option<u16> {
        let b = [*tiff.get(at)?, *tiff.get(at + 1)?];
        Some(if little { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) })
    };
    let u32_at = |at: usize| -> Option<u32> {
        let b = [*tiff.get(at)?, *tiff.get(at + 1)?, *tiff.get(at + 2)?, *tiff.get(at + 3)?];
        Some(if little { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    };
    let Some(ifd) = u32_at(4) else { return (Vec::new(), None) };
    let ifd = ifd as usize;
    let Some(count) = u16_at(ifd) else { return (Vec::new(), None) };
    let mut entries = Vec::new();
    let mut icc = None;
    for i in 0..usize::from(count) {
        let at = ifd + 2 + i * 12;
        let (Some(tag), Some(kind), Some(n)) = (u16_at(at), u16_at(at + 2), u32_at(at + 4)) else { break };
        let value = match kind {
            3 => u16_at(at + 8).map(u32::from),
            4 => u32_at(at + 8),
            _ => None,
        };
        if tag == 34675 {
            let len = n as usize;
            let start = if len <= 4 { Some(at + 8) } else { u32_at(at + 8).map(|o| o as usize) };
            icc = start.and_then(|s| tiff.get(s..s + len)).map(<[u8]>::to_vec);
        }
        if let Some(v) = value {
            entries.push((tag, v));
        }
    }
    (entries, icc)
}

fn orientation_of(tiff: &[u8]) -> Option<u16> {
    let (entries, _) = tiff_ifd0(tiff);
    entries.iter().find(|(t, _)| *t == 0x0112).and_then(|(_, v)| u16::try_from(*v).ok())
}

fn strip_exif_prefix(payload: &[u8]) -> &[u8] {
    payload.strip_prefix(b"Exif\0\0").unwrap_or(payload)
}

fn png_tags(data: &[u8]) -> SourceTags {
    let mut tags = SourceTags { container: Some(Container::Png), ..SourceTags::default() };
    let mut pos = 8;
    while pos + 8 <= data.len() {
        let Some(len) = be32(data, pos) else { break };
        let kind = &data[pos + 4..pos + 8];
        let end = pos + 12 + len as usize;
        let Some(body) = data.get(pos + 8..pos + 8 + len as usize) else { break };
        match kind {
            b"IHDR" => {
                if let (Some(w), Some(h)) = (be32(body, 0), be32(body, 4)) {
                    tags.stored = Some((w as usize, h as usize));
                }
            }
            b"iCCP" => {
                if let Some(nul) = body.iter().position(|&b| b == 0) {
                    let mut out = Vec::new();
                    let compressed = body.get(nul + 2..).unwrap_or_default();
                    if flate2::read::ZlibDecoder::new(compressed).read_to_end(&mut out).is_ok() && !out.is_empty() {
                        tags.icc = Some(out);
                    }
                }
            }
            b"eXIf" => tags.orientation = orientation_of(strip_exif_prefix(body)),
            b"IEND" => break,
            _ => {}
        }
        pos = end;
    }
    tags
}

fn jpeg_tags(data: &[u8]) -> SourceTags {
    let mut tags = SourceTags { container: Some(Container::Jpeg), ..SourceTags::default() };
    let mut parts: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut pos = 2;
    while pos + 4 <= data.len() && data[pos] == 0xFF {
        let marker = data[pos + 1];
        if marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            pos += 2;
            continue;
        }
        let Some(len) = be16(data, pos + 2) else { break };
        let Some(body) = data.get(pos + 4..pos + 2 + len as usize) else { break };
        match marker {
            0xE1 if body.starts_with(b"Exif\0\0") => {
                tags.orientation = tags.orientation.or_else(|| orientation_of(&body[6..]));
            }
            0xE2 if body.starts_with(b"ICC_PROFILE\0") && body.len() > 14 => parts.push((body[12], body[14..].to_vec())),
            0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                if let (Some(h), Some(w)) = (be16(body, 1), be16(body, 3)) {
                    tags.stored = Some((usize::from(w), usize::from(h)));
                }
            }
            0xDA => break,
            _ => {}
        }
        pos += 2 + len as usize;
    }
    if !parts.is_empty() {
        parts.sort_by_key(|(seq, _)| *seq);
        tags.icc = Some(parts.into_iter().flat_map(|(_, p)| p).collect());
    }
    tags
}

fn webp_tags(data: &[u8]) -> SourceTags {
    let mut tags = SourceTags { container: Some(Container::WebP), ..SourceTags::default() };
    let mut pos = 12;
    while pos + 8 <= data.len() {
        let kind = &data[pos..pos + 4];
        let len = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        let Some(body) = data.get(pos + 8..pos + 8 + len) else { break };
        match kind {
            b"ICCP" if !body.is_empty() => tags.icc = Some(body.to_vec()),
            b"EXIF" => tags.orientation = orientation_of(strip_exif_prefix(body)),
            _ => {}
        }
        pos += 8 + len + (len & 1);
    }
    tags.stored = image::ImageReader::with_format(std::io::Cursor::new(data), image::ImageFormat::WebP)
        .into_dimensions()
        .ok()
        .map(|(w, h)| (w as usize, h as usize));
    tags
}

fn tiff_tags(data: &[u8]) -> SourceTags {
    let (entries, icc) = tiff_ifd0(data);
    let get = |tag: u16| entries.iter().find(|(t, _)| *t == tag).map(|(_, v)| *v);
    SourceTags {
        container: Some(Container::Tiff),
        icc: icc.filter(|b| !b.is_empty()),
        orientation: get(0x0112).and_then(|v| u16::try_from(v).ok()),
        stored: get(256).zip(get(257)).map(|(w, h)| (w as usize, h as usize)),
    }
}

#[cfg(feature = "heif")]
fn heif_tags(data: &[u8], container: Container) -> SourceTags {
    use libheif_rs::HeifContext;
    let mut tags = SourceTags { container: Some(container), ..SourceTags::default() };
    let Ok(ctx) = HeifContext::read_from_bytes(data) else { return tags };
    let Ok(handle) = ctx.primary_image_handle() else { return tags };
    tags.icc = handle.color_profile_raw().map(|p| p.data).filter(|d| !d.is_empty());
    let mut ids = vec![0; 4];
    let n = handle.metadata_block_ids(&mut ids, b"Exif");
    for id in ids.into_iter().take(n) {
        if let Ok(block) = handle.metadata(id)
            && block.len() > 4
        {
            let skip = u32::from_be_bytes([block[0], block[1], block[2], block[3]]) as usize;
            if let Some(tiff) = block.get(4 + skip..) {
                tags.orientation = orientation_of(strip_exif_prefix(tiff));
                break;
            }
        }
    }
    tags.stored = Some((handle.width() as usize, handle.height() as usize));
    tags
}

#[cfg(not(feature = "heif"))]
fn heif_tags(_data: &[u8], _container: Container) -> SourceTags {
    SourceTags::default()
}

/// Parse a file's display tags; unreadable files have none.
pub(crate) fn source_tags(path: &Path) -> SourceTags {
    let Ok(data) = std::fs::read(path) else { return SourceTags::default() };
    match sniff(&data) {
        Some(Container::Png) => png_tags(&data),
        Some(Container::Jpeg) => jpeg_tags(&data),
        Some(Container::WebP) => webp_tags(&data),
        Some(Container::Tiff) => tiff_tags(&data),
        Some(c @ (Container::Heif | Container::Avif)) => heif_tags(&data, c),
        Some(c @ (Container::Bmp | Container::Gif)) => SourceTags { container: Some(c), ..SourceTags::default() },
        None => SourceTags::default(),
    }
}

/// `(icc, orientation)` to carry from `source` onto a raster of `(rows, cols)` (`_read_display_tags`).
pub(crate) fn read_display_tags(
    source: &Path,
    rows: usize,
    cols: usize,
    orientation_applied: Option<bool>,
) -> (Option<Vec<u8>>, Option<u16>) {
    let tags = source_tags(source);
    let Some(container) = tags.container else { return (None, None) };
    let mut applied = orientation_applied;
    let mut orient = tags.orientation;
    match container {
        Container::Tiff => applied = None,
        Container::Heif => {
            applied = Some(raiw_container::isobmff::heif_transform_applied(source));
        }
        _ => {}
    }
    let orient = match applied {
        Some(true) => None,
        Some(false) => orient.take().filter(|o| (1..=8).contains(o)),
        None => orient.filter(|o| TRANSPOSING.contains(o) && tags.stored == Some((cols, rows))),
    };
    (tags.icc, orient)
}

/// EXIF bytes holding only the orientation tag (`piexif.dump`), `prefixed` with `Exif\0\0` for JPEG.
pub(crate) fn orientation_exif(orient: Option<u16>, prefixed: bool) -> Option<Vec<u8>> {
    let orient = orient.filter(|&o| o != 1)?;
    let mut out = Vec::with_capacity(32);
    if prefixed {
        out.extend_from_slice(b"Exif\0\0");
    }
    out.extend_from_slice(b"MM\0\x2a\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01");
    out.extend_from_slice(&orient.to_be_bytes());
    out.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    Some(out)
}

fn png_chunk(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut crc = crc32fast::Hasher::new();
    crc.update(kind);
    crc.update(payload);
    let mut out = Vec::with_capacity(payload.len() + 12);
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc.finalize().to_be_bytes());
    out
}

/// Insert `iCCP` / `eXIf` after IHDR, replacing same-kind chunks; IDAT is copied verbatim.
pub fn png_splice_display_tags(data: &[u8], icc: Option<&[u8]>, orient: Option<u16>) -> Vec<u8> {
    let iccp = icc.map(|profile| {
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
        let compressed = z.write_all(profile).and_then(|()| z.finish()).unwrap_or_default();
        let mut payload = b"ICC Profile\0\0".to_vec();
        payload.extend_from_slice(&compressed);
        png_chunk(b"iCCP", &payload)
    });
    let exif = orientation_exif(orient, false).map(|p| png_chunk(b"eXIf", &p));
    if iccp.is_none() && exif.is_none() {
        return data.to_vec();
    }
    let mut out = data[..8.min(data.len())].to_vec();
    let mut pos = 8;
    let mut inserted = false;
    while pos + 8 <= data.len() {
        let Some(len) = be32(data, pos) else { return data.to_vec() };
        let kind = &data[pos + 4..pos + 8];
        let end = pos + 12 + len as usize;
        if end > data.len() {
            return data.to_vec();
        }
        let replaced = (kind == b"iCCP" && iccp.is_some()) || (kind == b"eXIf" && exif.is_some());
        if !replaced {
            out.extend_from_slice(&data[pos..end]);
        }
        if kind == b"IHDR" && !inserted {
            out.extend(iccp.iter().flatten());
            out.extend(exif.iter().flatten());
            inserted = true;
        }
        if kind == b"IEND" {
            break;
        }
        pos = end;
    }
    if inserted { out } else { data.to_vec() }
}

/// Insert APP1 (EXIF) and APP2 (ICC) segments at the end of the APP block.
pub fn jpeg_splice_display_tags(data: &[u8], icc: Option<&[u8]>, orient: Option<u16>) -> Vec<u8> {
    let exif = orientation_exif(orient, true);
    let mut segments: Vec<u8> = Vec::new();
    if let Some(e) = &exif {
        segments.extend_from_slice(&[0xFF, 0xE1]);
        segments.extend_from_slice(&((e.len() + 2) as u16).to_be_bytes());
        segments.extend_from_slice(e);
    }
    if let Some(profile) = icc {
        let part_size = 65535 - 2 - 14;
        let parts: Vec<&[u8]> = profile.chunks(part_size).collect();
        for (seq, part) in parts.iter().enumerate() {
            let mut payload = b"ICC_PROFILE\0".to_vec();
            payload.extend_from_slice(&[(seq + 1) as u8, parts.len() as u8]);
            payload.extend_from_slice(part);
            segments.extend_from_slice(&[0xFF, 0xE2]);
            segments.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
            segments.extend_from_slice(&payload);
        }
    }
    if segments.is_empty() || data.len() < 2 {
        return data.to_vec();
    }
    let mut out = data[..2].to_vec();
    let mut pos = 2;
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            return data.to_vec();
        }
        let marker = data[pos + 1];
        if !(0xE0..0xF0).contains(&marker) {
            break;
        }
        let Some(len) = be16(data, pos + 2) else { return data.to_vec() };
        let end = pos + 2 + len as usize;
        if len < 2 || end > data.len() {
            return data.to_vec();
        }
        let payload = &data[pos + 4..end];
        let replaced = (marker == 0xE1 && exif.is_some() && payload.starts_with(b"Exif\0\0"))
            || (marker == 0xE2 && icc.is_some() && payload.starts_with(b"ICC_PROFILE\0"));
        if !replaced {
            out.extend_from_slice(&data[pos..end]);
        }
        pos = end;
    }
    out.extend_from_slice(&segments);
    out.extend_from_slice(&data[pos..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_exif_matches_piexif() {
        let six = orientation_exif(Some(6), true).unwrap();
        assert_eq!(hex::encode(&six), "4578696600004d4d002a00000008000101120003000000010006000000000000");
        assert_eq!(orientation_of(&six[6..]), Some(6));
        assert_eq!(orientation_exif(Some(1), true), None);
        assert_eq!(orientation_exif(None, false), None);
    }
}

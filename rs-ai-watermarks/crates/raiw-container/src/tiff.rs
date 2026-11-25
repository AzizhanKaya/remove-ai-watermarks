//! Lossless in-place IFD metadata stripping for classic TIFF and DNG.
//!
//! IFD tables shrink in place and removed out-of-line values are zeroed unless they
//! overlap a protected strip, tile, IFD or kept value, so no file offset moves.

use std::collections::{HashSet, VecDeque};
use std::io;
use std::path::Path;

use raiw_core::StagedSibling;

use crate::hooks::MetadataHooks;

const TAG_IMAGE_DESCRIPTION: u16 = 270;
const TAG_MAKE: u16 = 271;
const TAG_STRIP_OFFSETS: u16 = 273;
const TAG_STRIP_BYTE_COUNTS: u16 = 279;
const TAG_SOFTWARE: u16 = 305;
const TAG_ARTIST: u16 = 315;
const TAG_TILE_OFFSETS: u16 = 324;
const TAG_TILE_BYTE_COUNTS: u16 = 325;
const TAG_SUB_IFDS: u16 = 330;
const TAG_XMP: u16 = 700;
const TAG_COPYRIGHT: u16 = 33432;
const TAG_IPTC: u16 = 33723;
const TAG_PHOTOSHOP: u16 = 34377;
const TAG_EXIF_IFD: u16 = 34665;
const TAG_ICC_PROFILE: u16 = 34675;
const TAG_USER_COMMENT: u16 = 37510;
const XP_TEXT_TAGS: [u16; 5] = [40091, 40092, 40093, 40094, 40095];
const TEXT_METADATA_TAGS: [u16; 6] = [
    TAG_IMAGE_DESCRIPTION,
    TAG_MAKE,
    TAG_SOFTWARE,
    TAG_ARTIST,
    TAG_COPYRIGHT,
    TAG_USER_COMMENT,
];
const BLOB_METADATA_TAGS: [u16; 3] = [TAG_XMP, TAG_IPTC, TAG_PHOTOSHOP];

/// Whether `head` begins with a classic 32-bit TIFF/DNG header.
pub fn is_classic_tiff(head: &[u8]) -> bool {
    head.len() >= 4 && (&head[..4] == b"II*\x00" || &head[..4] == b"MM\x00*")
}

fn is_inspectable(tag: u16) -> bool {
    TEXT_METADATA_TAGS.contains(&tag) || XP_TEXT_TAGS.contains(&tag) || BLOB_METADATA_TAGS.contains(&tag)
}

fn type_size(field_type: u16) -> Option<u64> {
    match field_type {
        1 | 2 | 6 | 7 => Some(1),
        3 | 8 => Some(2),
        4 | 9 | 11 | 13 => Some(4),
        5 | 10 | 12 => Some(8),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn u16(self, buf: &[u8], offset: usize) -> u16 {
        let bytes = [buf[offset], buf[offset + 1]];
        match self {
            Self::Little => u16::from_le_bytes(bytes),
            Self::Big => u16::from_be_bytes(bytes),
        }
    }

    fn u32(self, buf: &[u8], offset: usize) -> u32 {
        let bytes = [buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]];
        match self {
            Self::Little => u32::from_le_bytes(bytes),
            Self::Big => u32::from_be_bytes(bytes),
        }
    }

    fn put_u16(self, buf: &mut [u8], offset: usize, value: u16) {
        let bytes = match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        };
        buf[offset..offset + 2].copy_from_slice(&bytes);
    }

    fn put_u32(self, buf: &mut [u8], offset: usize, value: u32) {
        let bytes = match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        };
        buf[offset..offset + 4].copy_from_slice(&bytes);
    }
}

/// One 12-byte IFD entry.
#[derive(Debug, Clone, Copy)]
struct Entry {
    raw: [u8; 12],
    tag: u16,
    field_type: u16,
    count: u32,
}

impl Entry {
    fn parse(buf: &[u8], offset: usize, endian: Endian) -> Self {
        let mut raw = [0u8; 12];
        raw.copy_from_slice(&buf[offset..offset + 12]);
        Self {
            raw,
            tag: endian.u16(&raw, 0),
            field_type: endian.u16(&raw, 2),
            count: endian.u32(&raw, 4),
        }
    }

    fn value_field(&self) -> &[u8] {
        &self.raw[8..12]
    }
}

/// A tag value out of bounds: the container is corrupt.
#[derive(Debug)]
struct OutOfBounds;

type Span = (usize, usize);

/// The out-of-line value span of `entry`, or `None` when the value fits in the entry.
fn external_span(size: usize, endian: Endian, entry: &Entry) -> Result<Option<Span>, OutOfBounds> {
    let Some(unit) = type_size(entry.field_type) else {
        return Ok(None);
    };
    let byte_len = unit * u64::from(entry.count);
    if byte_len <= 4 {
        return Ok(None);
    }
    let offset = u64::from(endian.u32(entry.value_field(), 0));
    if offset < 8 || offset + byte_len > size as u64 {
        return Err(OutOfBounds);
    }
    Ok(Some((offset as usize, (offset + byte_len) as usize)))
}

/// Decode a SHORT, LONG or IFD array, inline or out of line; empty when it does not fit.
fn int_array(data: &[u8], endian: Endian, entry: &Entry) -> Vec<u64> {
    let unit: usize = match entry.field_type {
        3 => 2,
        4 | 13 => 4,
        _ => return Vec::new(),
    };
    let count = entry.count as usize;
    if count == 0 {
        return Vec::new();
    }
    let total = unit as u64 * u64::from(entry.count);
    let (source, start) = if total <= 4 {
        (entry.value_field(), 0)
    } else {
        let offset = endian.u32(entry.value_field(), 0) as usize;
        if offset as u64 + total > data.len() as u64 {
            return Vec::new();
        }
        (data, offset)
    };
    (0..count)
        .map(|i| {
            let at = start + i * unit;
            if unit == 2 {
                u64::from(endian.u16(source, at))
            } else {
                u64::from(endian.u32(source, at))
            }
        })
        .collect()
}

/// Raw tag bytes, materialized only when a metadata predicate must inspect them.
fn tag_bytes(data: &[u8], entry: &Entry, span: Option<Span>) -> Vec<u8> {
    if let Some((start, end)) = span {
        return data[start..end].to_vec();
    }
    let byte_len = type_size(entry.field_type).unwrap_or(1) * u64::from(entry.count);
    entry.value_field()[..byte_len.min(4) as usize].to_vec()
}

/// Decode UTF-16 like Python's `decode(enc, errors="ignore")`, re-encoded as UTF-8.
fn utf16_ignore(body: &[u8], big_endian: bool) -> Vec<u8> {
    let units = body.chunks_exact(2).map(|pair| {
        if big_endian {
            u16::from_be_bytes([pair[0], pair[1]])
        } else {
            u16::from_le_bytes([pair[0], pair[1]])
        }
    });
    char::decode_utf16(units)
        .filter_map(Result::ok)
        .collect::<String>()
        .into_bytes()
}

fn payload_has_ai_markers(raw: &[u8], is_xp_utf16: bool, hooks: &impl MetadataHooks) -> bool {
    if raw.is_empty() {
        return false;
    }
    if hooks.carries_ai_metadata(raw) {
        return true;
    }
    let (body, encodings): (&[u8], &[bool]) = if is_xp_utf16 {
        (raw, &[false])
    } else if let Some(body) = raw.strip_prefix(b"UNICODE\x00") {
        (body, &[true, false])
    } else {
        (raw, &[])
    };
    encodings.iter().any(|&big_endian| {
        let decoded = utf16_ignore(body, big_endian);
        !decoded.is_empty() && hooks.carries_ai_metadata(&decoded)
    })
}

fn should_strip(tag: u16, raw: &[u8], keep_standard: bool, hooks: &impl MetadataHooks) -> bool {
    if tag == TAG_ICC_PROFILE {
        return !keep_standard;
    }
    is_inspectable(tag) && (!keep_standard || payload_has_ai_markers(raw, XP_TEXT_TAGS.contains(&tag), hooks))
}

struct IfdUpdate {
    offset: usize,
    kept: Vec<[u8; 12]>,
    next_ifd: u32,
    original_count: usize,
}

struct Plan {
    protected: Vec<Span>,
    updates: Vec<IfdUpdate>,
    zero: Vec<Span>,
}

/// Walk every IFD and decide removals; `None` for a truncated or corrupt structure.
fn plan(
    buf: &[u8],
    endian: Endian,
    first_ifd: usize,
    keep_standard: bool,
    hooks: &impl MetadataHooks,
) -> Option<Plan> {
    let size = buf.len();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::from([(first_ifd, false, false)]);
    let mut plan = Plan {
        protected: vec![(0, 8)],
        updates: Vec::new(),
        zero: Vec::new(),
    };
    let in_file = |offset: u64| (8..size as u64).contains(&offset);
    while let Some((ifd, is_exif, drop_whole)) = queue.pop_front() {
        if ifd == 0 || !visited.insert(ifd) {
            continue;
        }
        if ifd < 8 || ifd + 2 > size {
            return None;
        }
        let entry_count = usize::from(endian.u16(buf, ifd));
        let table_end = ifd + 2 + entry_count * 12 + 4;
        if table_end > size {
            return None;
        }
        let entries: Vec<Entry> = (0..entry_count)
            .map(|i| Entry::parse(buf, ifd + 2 + i * 12, endian))
            .collect();
        if drop_whole {
            plan.zero.push((ifd, table_end));
            for entry in &entries {
                if let Some(span) = external_span(size, endian, entry).ok()? {
                    plan.zero.push(span);
                }
            }
            continue;
        }
        plan.protected.push((ifd, table_end));
        let next_ifd = endian.u32(buf, ifd + 2 + entry_count * 12);
        if next_ifd != 0 {
            if !in_file(u64::from(next_ifd)) {
                return None;
            }
            queue.push_back((next_ifd as usize, is_exif, false));
        }
        let (mut strips, mut strip_counts, mut tiles, mut tile_counts) = (vec![], vec![], vec![], vec![]);
        let mut kept = Vec::new();
        let mut removed_any = false;
        for entry in &entries {
            let span = external_span(size, endian, entry).ok()?;
            if !is_exif {
                match entry.tag {
                    TAG_STRIP_OFFSETS => strips = int_array(buf, endian, entry),
                    TAG_STRIP_BYTE_COUNTS => strip_counts = int_array(buf, endian, entry),
                    TAG_TILE_OFFSETS => tiles = int_array(buf, endian, entry),
                    TAG_TILE_BYTE_COUNTS => tile_counts = int_array(buf, endian, entry),
                    TAG_SUB_IFDS => {
                        for sub in int_array(buf, endian, entry).into_iter().filter(|&o| in_file(o)) {
                            queue.push_back((sub as usize, false, false));
                        }
                    }
                    TAG_EXIF_IFD => {
                        let targets = int_array(buf, endian, entry);
                        let dropping = !keep_standard;
                        for target in targets.into_iter().filter(|&o| in_file(o)) {
                            queue.push_back((target as usize, true, dropping));
                        }
                        if dropping {
                            removed_any = true;
                            plan.zero.extend(span);
                        } else {
                            kept.push(entry.raw);
                            plan.protected.extend(span);
                        }
                        continue;
                    }
                    _ => {}
                }
            }
            let raw = if keep_standard && is_inspectable(entry.tag) {
                tag_bytes(buf, entry, span)
            } else {
                Vec::new()
            };
            if should_strip(entry.tag, &raw, keep_standard, hooks) {
                removed_any = true;
                plan.zero.extend(span);
                continue;
            }
            kept.push(entry.raw);
            plan.protected.extend(span);
        }
        for (offsets, counts) in [(&strips, &strip_counts), (&tiles, &tile_counts)] {
            for (&offset, &length) in offsets.iter().zip(counts.iter()) {
                if offset > 0 && length > 0 && offset + length <= size as u64 {
                    plan.protected.push((offset as usize, (offset + length) as usize));
                }
            }
        }
        if removed_any {
            plan.updates.push(IfdUpdate {
                offset: ifd,
                kept,
                next_ifd,
                original_count: entry_count,
            });
        }
    }
    Some(plan)
}

fn apply(buf: &mut [u8], endian: Endian, plan: &Plan) {
    for update in &plan.updates {
        endian.put_u16(buf, update.offset, update.kept.len() as u16);
        let mut cursor = update.offset + 2;
        for entry in &update.kept {
            buf[cursor..cursor + 12].copy_from_slice(entry);
            cursor += 12;
        }
        endian.put_u32(buf, cursor, update.next_ifd);
        cursor += 4;
        let original_end = update.offset + 2 + update.original_count * 12 + 4;
        if cursor < original_end {
            buf[cursor..original_end].fill(0);
        }
    }
    for &(start, end) in &plan.zero {
        if !plan
            .protected
            .iter()
            .any(|&(p_start, p_end)| start < p_end && p_start < end)
        {
            buf[start..end].fill(0);
        }
    }
}

/// Strip AI metadata from a classic TIFF/DNG file without altering any pixel offset.
///
/// Returns `Ok(false)` (writing nothing) for a truncated, corrupt or unsupported container
/// such as BigTIFF.
pub fn strip_tiff_metadata_lossless(
    source: &Path,
    output: &Path,
    keep_standard: bool,
    hooks: &impl MetadataHooks,
) -> io::Result<bool> {
    let mut buf = std::fs::read(source)?;
    if !strip_tiff_bytes(&mut buf, keep_standard, hooks) {
        return Ok(false);
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let stage = StagedSibling::new(output)?;
    std::fs::write(stage.path(), &buf)?;
    stage.commit()?;
    Ok(true)
}

/// In-memory form of [`strip_tiff_metadata_lossless`]; `false` leaves a malformed `buf` untouched.
pub fn strip_tiff_bytes(buf: &mut [u8], keep_standard: bool, hooks: &impl MetadataHooks) -> bool {
    if buf.len() < 8 || !is_classic_tiff(buf) {
        return false;
    }
    let endian = if &buf[..2] == b"II" {
        Endian::Little
    } else {
        Endian::Big
    };
    let first_ifd = endian.u32(buf, 4) as usize;
    if first_ifd < 8 || first_ifd >= buf.len() {
        return false;
    }
    let Some(plan) = plan(buf, endian, first_ifd, keep_standard, hooks) else {
        return false;
    };
    apply(buf, endian, &plan);
    true
}

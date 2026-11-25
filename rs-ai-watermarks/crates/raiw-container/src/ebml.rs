//! Matroska/WebM (EBML) walker: TC260 `SimpleTag`s, C2PA attachments and AV1 `OBU_METADATA`.
//!
//! Unrelated elements such as clusters are skipped by size; only bounded values are read.

use std::fs::OpenOptions;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::av1::{self, ObuPatch};
use crate::hooks::MetadataHooks;
use crate::markers::{MAX_TC260_VALUE_BYTES, strip_bytes};
use crate::source::{FileSource, ReadAt};

pub const EBML_MAGIC: &[u8; 4] = b"\x1aE\xdf\xa3";

const SEGMENT_ID: u32 = 0x1853_8067;
const TRACKS_ID: u32 = 0x1654_AE6B;
const TRACK_ENTRY_ID: u32 = 0xAE;
const TRACK_NUMBER_ID: u32 = 0xD7;
const CODEC_ID: u32 = 0x86;
const CODEC_PRIVATE_ID: u32 = 0x63A2;
const CONTENT_ENCODINGS_ID: u32 = 0x6D80;
const CLUSTER_ID: u32 = 0x1F43_B675;
const SIMPLE_BLOCK_ID: u32 = 0xA3;
const BLOCK_GROUP_ID: u32 = 0xA0;
const BLOCK_ID: u32 = 0xA1;
const CRC32_ID: u32 = 0xBF;
const TAGS_ID: u32 = 0x1254_C367;
const TAG_ID: u32 = 0x7373;
const SIMPLE_TAG_ID: u32 = 0x67C8;
const TAG_NAME_ID: u32 = 0x45A3;
const TAG_STRING_ID: u32 = 0x4487;
const ATTACHMENTS_ID: u32 = 0x1941_A469;
const ATTACHED_FILE_ID: u32 = 0x61A7;
const FILE_MEDIA_TYPE_ID: u32 = 0x4660;
const FILE_DATA_ID: u32 = 0x465C;
const MAX_TAG_NAME_BYTES: u64 = 256;
const MAX_C2PA_ATTACHMENT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_AV1_BLOCK_BYTES: u64 = 4 * 1024 * 1024;
const C2PA_ATTACHMENT_MIMES: [&[u8]; 2] = [b"application/c2pa", b"application/x-c2pa-manifest-store"];

/// One element: its id, payload start and end (an unknown size runs to the parent's end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Element {
    pub id: u32,
    pub payload: u64,
    pub end: u64,
}

fn vint_length(first: u8, maximum: u32) -> Option<u32> {
    let length = first.leading_zeros() + 1;
    (length <= maximum).then_some(length)
}

/// Read the element header at `pos` inside `limit`.
pub fn read_element_header<R: ReadAt + ?Sized>(
    source: &R,
    pos: u64,
    limit: u64,
) -> io::Result<Option<Element>> {
    if pos >= limit {
        return Ok(None);
    }
    let Some([first]) = source.read_array::<1>(pos)? else {
        return Ok(None);
    };
    let Some(id_length) = vint_length(first, 4).map(u64::from) else {
        return Ok(None);
    };
    if pos + id_length >= limit {
        return Ok(None);
    }
    let id_raw = source.read_vec(pos, id_length)?;
    if id_raw.len() as u64 != id_length {
        return Ok(None);
    }
    let id = id_raw.iter().fold(0u32, |acc, &b| (acc << 8) | u32::from(b));
    let size_pos = pos + id_length;
    let Some([size_first]) = source.read_array::<1>(size_pos)? else {
        return Ok(None);
    };
    let Some(size_length) = vint_length(size_first, 8) else {
        return Ok(None);
    };
    let rest = source.read_vec(size_pos + 1, u64::from(size_length) - 1)?;
    if rest.len() as u64 != u64::from(size_length) - 1 {
        return Ok(None);
    }
    let marker = 1u16 << (8 - size_length);
    let first_bits = u64::from(u16::from(size_first) & (marker - 1));
    let size = rest.iter().fold(first_bits, |acc, &b| (acc << 8) | u64::from(b));
    let payload = size_pos + u64::from(size_length);
    if payload > limit {
        return Ok(None);
    }
    let unknown_size = size == (1u64 << (7 * size_length)) - 1;
    let end = if unknown_size {
        Some(limit)
    } else {
        payload.checked_add(size)
    };
    match end {
        Some(end) if end <= limit => Ok(Some(Element { id, payload, end })),
        _ => Ok(None),
    }
}

/// Direct children of `[start, end)`; stops at the first invalid header.
pub fn elements<R: ReadAt + ?Sized>(source: &R, start: u64, end: u64) -> io::Result<Vec<Element>> {
    let mut found = Vec::new();
    let mut pos = start;
    while pos < end {
        let Some(element) = read_element_header(source, pos, end)? else {
            break;
        };
        found.push(element);
        if element.end <= pos {
            break;
        }
        pos = element.end;
    }
    Ok(found)
}

/// End offset of each valid top-level element; a truncated file ends short of its size.
pub fn top_level_ends<R: ReadAt + ?Sized>(source: &R) -> io::Result<Vec<u64>> {
    Ok(elements(source, 0, source.size())?
        .into_iter()
        .map(|element| element.end)
        .collect())
}

fn read_bounded<R: ReadAt + ?Sized>(
    source: &R,
    start: u64,
    end: u64,
    maximum: u64,
) -> io::Result<Option<Vec<u8>>> {
    let Some(size) = end.checked_sub(start).filter(|&size| size <= maximum) else {
        return Ok(None);
    };
    let value = source.read_vec(start, size)?;
    Ok((value.len() as u64 == size).then_some(value))
}

/// Children of every top-level Segment that carry `target`.
fn segment_children<R: ReadAt + ?Sized>(source: &R, target: u32) -> io::Result<Vec<Element>> {
    let mut found = Vec::new();
    for segment in elements(source, 0, source.size())? {
        if segment.id == SEGMENT_ID {
            found.extend(
                elements(source, segment.payload, segment.end)?
                    .into_iter()
                    .filter(|e| e.id == target),
            );
        }
    }
    Ok(found)
}

fn open_ebml(path: &Path) -> io::Result<Option<FileSource>> {
    let source = FileSource::open(path)?;
    Ok((source.read_vec(0, 4)? == EBML_MAGIC).then_some(source))
}

fn simple_tag_payloads<R: ReadAt + ?Sized>(
    source: &R,
    tag: &Element,
    hooks: &impl MetadataHooks,
) -> io::Result<Vec<Vec<u8>>> {
    let mut name = None;
    let mut values = Vec::new();
    for field in elements(source, tag.payload, tag.end)? {
        if field.id == TAG_NAME_ID {
            name = read_bounded(source, field.payload, field.end, MAX_TAG_NAME_BYTES)?;
        } else if field.id == TAG_STRING_ID
            && let Some(value) = read_bounded(source, field.payload, field.end, MAX_TC260_VALUE_BYTES)?
        {
            values.push(value);
        }
    }
    if name.as_deref() != Some(b"AIGC") {
        return Ok(Vec::new());
    }
    Ok(values
        .into_iter()
        .filter(|value| hooks.is_tc260_label(value))
        .collect())
}

/// Validated TC260 values from `SimpleTag` entries named `AIGC`; read errors yield nothing.
pub fn tc260_aigc_payloads(path: &Path, hooks: &impl MetadataHooks) -> Vec<Vec<u8>> {
    let scan = || -> io::Result<Vec<Vec<u8>>> {
        let Some(source) = open_ebml(path)? else {
            return Ok(Vec::new());
        };
        let mut found = Vec::new();
        for tags in segment_children(&source, TAGS_ID)? {
            for tag in elements(&source, tags.payload, tags.end)?
                .into_iter()
                .filter(|e| e.id == TAG_ID)
            {
                for simple in elements(&source, tag.payload, tag.end)? {
                    if simple.id == SIMPLE_TAG_ID {
                        found.extend(simple_tag_payloads(&source, &simple, hooks)?);
                    }
                }
            }
        }
        Ok(found)
    };
    scan().unwrap_or_default()
}

/// The `FileData` span of an `AttachedFile` whose media type is a C2PA manifest store.
fn attached_c2pa_span<R: ReadAt + ?Sized>(source: &R, file: &Element) -> io::Result<Option<(u64, u64)>> {
    let mut mime = None;
    let mut data = None;
    for field in elements(source, file.payload, file.end)? {
        if field.id == FILE_MEDIA_TYPE_ID {
            mime = read_bounded(source, field.payload, field.end, MAX_TAG_NAME_BYTES)?;
        } else if field.id == FILE_DATA_ID {
            data = Some((field.payload, field.end));
        }
    }
    let is_c2pa = mime.is_some_and(|mime| {
        let mime = strip_bytes(&mime, b"\x00").to_ascii_lowercase();
        C2PA_ATTACHMENT_MIMES.contains(&mime.as_slice())
    });
    Ok(if is_c2pa { data } else { None })
}

fn attached_files<R: ReadAt + ?Sized>(source: &R) -> io::Result<Vec<Element>> {
    let mut found = Vec::new();
    for attachments in segment_children(source, ATTACHMENTS_ID)? {
        found.extend(
            elements(source, attachments.payload, attachments.end)?
                .into_iter()
                .filter(|e| e.id == ATTACHED_FILE_ID),
        );
    }
    Ok(found)
}

/// C2PA manifest attachments (`application/c2pa`); read errors yield nothing.
pub fn c2pa_attachment_payloads(path: &Path) -> Vec<Vec<u8>> {
    let scan = || -> io::Result<Vec<Vec<u8>>> {
        let Some(source) = open_ebml(path)? else {
            return Ok(Vec::new());
        };
        let mut found = Vec::new();
        for file in attached_files(&source)? {
            if let Some((start, end)) = attached_c2pa_span(&source, &file)?
                && let Some(blob) = read_bounded(&source, start, end, MAX_C2PA_ATTACHMENT_BYTES)?
            {
                found.push(blob);
            }
        }
        Ok(found)
    };
    scan().unwrap_or_default()
}

/// `(total_attachments, c2pa_ordinals)` over `AttachedFile` elements; read errors give `(0, [])`.
pub fn attachment_ordinals(path: &Path) -> (usize, Vec<usize>) {
    let scan = || -> io::Result<(usize, Vec<usize>)> {
        let Some(source) = open_ebml(path)? else {
            return Ok((0, Vec::new()));
        };
        let files = attached_files(&source)?;
        let mut ordinals = Vec::new();
        for (ordinal, file) in files.iter().enumerate() {
            if attached_c2pa_span(&source, file)?.is_some() {
                ordinals.push(ordinal);
            }
        }
        Ok((files.len(), ordinals))
    };
    scan().unwrap_or_default()
}

fn vint_value(data: &[u8], pos: usize) -> Option<(u64, usize)> {
    let first = *data.get(pos)?;
    let length = vint_length(first, 8)? as usize;
    if pos + length > data.len() {
        return None;
    }
    let marker = 1u16 << (8 - length);
    let first_bits = u64::from(u16::from(first) & (marker - 1));
    let value = data[pos + 1..pos + length]
        .iter()
        .fold(first_bits, |acc, &b| (acc << 8) | u64::from(b));
    Some((value, pos + length))
}

fn signed_vint_value(data: &[u8], pos: usize) -> Option<(i64, usize)> {
    let (value, next) = vint_value(data, pos)?;
    let bias = (1i64 << (7 * (next - pos) - 1)) - 1;
    Some((value as i64 - bias, next))
}

/// File spans of each laced AV1 frame in a `SimpleBlock`/`Block` whose header prefix is `header`.
fn block_frame_spans(header: &[u8], block_start: u64, total_len: u64, av1_tracks: &[u64]) -> Vec<(u64, u64)> {
    let Some((track, mut pos)) = vint_value(header, 0) else {
        return Vec::new();
    };
    if !av1_tracks.contains(&track) || pos + 3 > header.len() || (pos + 3) as u64 > total_len {
        return Vec::new();
    }
    let flags = header[pos + 2];
    pos += 3;
    let lacing = (flags >> 1) & 0x03;
    let total = total_len as usize;
    if lacing == 0 {
        return if pos < total {
            vec![(block_start + pos as u64, block_start + total_len)]
        } else {
            Vec::new()
        };
    }
    let Some(&count_byte) = header.get(pos) else {
        return Vec::new();
    };
    let frame_count = usize::from(count_byte) + 1;
    pos += 1;
    let mut sizes: Vec<usize> = Vec::with_capacity(frame_count);
    match lacing {
        1 => {
            for _ in 0..frame_count - 1 {
                let mut size = 0;
                while pos < header.len() && header[pos] == 0xFF {
                    size += 255;
                    pos += 1;
                }
                let Some(&last) = header.get(pos) else {
                    return Vec::new();
                };
                size += usize::from(last);
                pos += 1;
                sizes.push(size);
            }
        }
        2 => {
            let Some(remaining) = total.checked_sub(pos).filter(|rem| rem % frame_count == 0) else {
                return Vec::new();
            };
            sizes.resize(frame_count - 1, remaining / frame_count);
        }
        _ => {
            let Some((first, next)) = vint_value(header, pos) else {
                return Vec::new();
            };
            pos = next;
            let mut current = first as i64;
            sizes.push(first as usize);
            for _ in 0..frame_count.saturating_sub(2) {
                let Some((delta, next)) = signed_vint_value(header, pos) else {
                    return Vec::new();
                };
                pos = next;
                current += delta;
                if current < 0 {
                    return Vec::new();
                }
                sizes.push(current as usize);
            }
        }
    }
    let used: usize = sizes.iter().sum();
    if pos + used > total {
        return Vec::new();
    }
    sizes.push(total - pos - used);
    let mut spans = Vec::new();
    let mut cursor = pos;
    for size in sizes {
        if size > 0 {
            spans.push((block_start + cursor as u64, block_start + (cursor + size) as u64));
        }
        cursor += size;
    }
    spans
}

/// Scan one block; `true` when a first-hit scan found a patch and may stop.
fn scan_av1_block<R: ReadAt + ?Sized>(
    source: &R,
    block: &Element,
    av1_tracks: &[u64],
    patches: &mut Vec<ObuPatch>,
    stop_on_first: bool,
) -> io::Result<bool> {
    let total = block.end - block.payload;
    if total <= 4 || total > MAX_AV1_BLOCK_BYTES {
        return Ok(false);
    }
    let prefix = source.read_vec(block.payload, total.min(1024))?;
    for (start, end) in block_frame_spans(&prefix, block.payload, total, av1_tracks) {
        let found = av1::scan_stream_spans(source, start, end, stop_on_first)?;
        if !found.is_empty() {
            patches.extend(found);
            if stop_on_first {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// A CRC-32 element to refresh: its 4-byte payload and the sibling bytes it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CrcParent {
    crc_payload: u64,
    body_start: u64,
    body_end: u64,
}

/// The leading CRC-32 element of a parent, when present.
fn leading_crc(children: &[Element], parent_end: u64) -> Option<CrcParent> {
    children
        .first()
        .filter(|first| first.id == CRC32_ID && first.end - first.payload == 4)
        .map(|crc| CrcParent {
            crc_payload: crc.payload,
            body_start: crc.end,
            body_end: parent_end,
        })
}

struct TrackFields {
    number: Option<u64>,
    codec: Option<Vec<u8>>,
    private: Option<(u64, u64)>,
    encoded: bool,
}

fn track_fields<R: ReadAt + ?Sized>(source: &R, entry: &Element) -> io::Result<TrackFields> {
    let mut fields = TrackFields {
        number: None,
        codec: None,
        private: None,
        encoded: false,
    };
    for field in elements(source, entry.payload, entry.end)? {
        match field.id {
            TRACK_NUMBER_ID if (1..=8).contains(&(field.end - field.payload)) => {
                if let Some(raw) = read_bounded(source, field.payload, field.end, 8)? {
                    fields.number = Some(raw.iter().fold(0u64, |acc, &b| (acc << 8) | u64::from(b)));
                }
            }
            CODEC_ID => fields.codec = read_bounded(source, field.payload, field.end, 64)?,
            CODEC_PRIVATE_ID => fields.private = Some((field.payload, field.end)),
            CONTENT_ENCODINGS_ID => fields.encoded = true,
            _ => {}
        }
    }
    Ok(fields)
}

/// `(patches, crc_parents)` for unencoded `V_AV1` tracks; a first-hit scan keeps one patch.
fn scan_av1_stream<R: ReadAt + ?Sized>(
    source: &R,
    stop_on_first: bool,
) -> io::Result<(Vec<ObuPatch>, Vec<CrcParent>)> {
    if source.read_vec(0, 4)? != EBML_MAGIC {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut av1_tracks: Vec<u64> = Vec::new();
    let mut patches = Vec::new();
    let mut crc_parents = Vec::new();
    for segment in elements(source, 0, source.size())?
        .into_iter()
        .filter(|e| e.id == SEGMENT_ID)
    {
        for child in elements(source, segment.payload, segment.end)? {
            let before = patches.len();
            let children = match child.id {
                TRACKS_ID => elements(source, child.payload, child.end)?,
                CLUSTER_ID if !av1_tracks.is_empty() => elements(source, child.payload, child.end)?,
                _ => continue,
            };
            let crc = leading_crc(&children, child.end);
            let body = &children[usize::from(crc.is_some())..];
            if child.id == TRACKS_ID {
                for entry in body.iter().filter(|e| e.id == TRACK_ENTRY_ID) {
                    let fields = track_fields(source, entry)?;
                    let is_av1 = fields
                        .codec
                        .as_deref()
                        .is_some_and(|codec| strip_bytes(codec, b"\x00") == b"V_AV1");
                    let Some(number) = fields.number.filter(|_| is_av1 && !fields.encoded) else {
                        continue;
                    };
                    if !av1_tracks.contains(&number) {
                        av1_tracks.push(number);
                    }
                    if let Some((start, end)) = fields.private.filter(|(start, end)| end - start > 4)
                        && let Some(config) = read_bounded(source, start + 4, end, MAX_AV1_BLOCK_BYTES)?
                    {
                        patches.extend(av1::scan_sample_spans(&config, start + 4));
                        if stop_on_first && !patches.is_empty() {
                            patches.truncate(1);
                            return Ok((patches, Vec::new()));
                        }
                    }
                }
            } else {
                for element in body {
                    let blocks = match element.id {
                        SIMPLE_BLOCK_ID => vec![*element],
                        BLOCK_GROUP_ID => elements(source, element.payload, element.end)?
                            .into_iter()
                            .filter(|e| e.id == BLOCK_ID)
                            .collect(),
                        _ => continue,
                    };
                    for block in &blocks {
                        if scan_av1_block(source, block, &av1_tracks, &mut patches, stop_on_first)? {
                            patches.truncate(1);
                            return Ok((patches, Vec::new()));
                        }
                    }
                }
            }
            if let Some(crc) = crc
                && patches.len() > before
            {
                crc_parents.push(crc);
            }
        }
    }
    Ok((patches, crc_parents))
}

/// AI `OBU_METADATA` units in Matroska/WebM `V_AV1` tracks; read errors yield nothing.
pub fn scan_av1_ai_obus(path: &Path, stop_on_first: bool) -> Vec<ObuPatch> {
    FileSource::open(path)
        .and_then(|source| scan_av1_stream(&source, stop_on_first))
        .map(|(p, _)| p)
        .unwrap_or_default()
}

/// Rewrite AI `OBU_METADATA` units to `OBU_PADDING` in place and refresh affected CRC-32s.
pub fn scrub_av1_ai_obus(path: &Path) -> io::Result<usize> {
    let (patches, crc_parents) = scan_av1_stream(&FileSource::open(path)?, false)?;
    if patches.is_empty() {
        return Ok(0);
    }
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    for patch in &patches {
        av1::patch_to_padding(&mut file, patch)?;
    }
    let mut buffer = vec![0u8; 1024 * 1024];
    for parent in &crc_parents {
        file.seek(SeekFrom::Start(parent.body_start))?;
        let mut hasher = crc32fast::Hasher::new();
        let mut remaining = parent.body_end - parent.body_start;
        while remaining > 0 {
            let want = remaining.min(buffer.len() as u64) as usize;
            let read = file.read(&mut buffer[..want])?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            remaining -= read as u64;
        }
        file.seek(SeekFrom::Start(parent.crc_payload))?;
        file.write_all(&hasher.finalize().to_le_bytes())?;
    }
    file.flush()?;
    Ok(patches.len())
}

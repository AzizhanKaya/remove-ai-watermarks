//! Structural H.264/HEVC SEI and AV1 `OBU_METADATA` scan across ISOBMFF video sample tables.

use std::collections::{BTreeSet, HashMap};
use std::io;
use std::path::Path;

use super::boxes::{BoxEntry, FourCc, Span, boxes, is_isobmff, iter_boxes, last_of_type};
use crate::av1::{self, ObuPatch};
use crate::markers::{C2PA_UUID, KLING_SEI_TAG, contains};
use crate::source::{FileSource, ReadAt};

const MAX_STBL_ENTRIES: u64 = 2_000_000;
const MAX_SEI_NAL_BYTES: u64 = 256 * 1024;
const X264_SEI_UUID: [u8; 16] = [
    0xdc, 0x45, 0xe9, 0xbd, 0xe6, 0xd9, 0x48, 0xb7, 0x96, 0x2c, 0xd8, 0x20, 0xd9, 0x23, 0xee, 0xef,
];

/// Result of [`scan_video_sei`]; every list is sorted and de-duplicated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VideoSeiScan {
    /// Every track was walked, none fragmented: the lists cover the whole file.
    pub has_parsed_tracks: bool,
    pub kling_tag_offsets: Vec<u64>,
    pub unregistered_uuids: Vec<String>,
    pub ai_sei_spans: Vec<Span>,
    pub av1_obu_patches: Vec<ObuPatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Avc,
    Hevc,
    Av1,
}

#[derive(Debug, Clone, Copy)]
struct Codec {
    family: Family,
    length_size: usize,
}

#[derive(Default)]
struct Findings {
    kling: BTreeSet<u64>,
    uuids: BTreeSet<String>,
    ai_spans: BTreeSet<Span>,
    av1: BTreeSet<ObuPatch>,
}

/// Remove `00 00 03` emulation-prevention bytes, keeping each kept byte's raw index.
fn unescape_rbsp(raw: &[u8]) -> Option<(Vec<u8>, Vec<usize>)> {
    memchr::memmem::find(raw, b"\x00\x00\x03")?;
    let mut rbsp = Vec::with_capacity(raw.len());
    let mut indices = Vec::with_capacity(raw.len());
    let mut zero_run = 0;
    for (index, &byte) in raw.iter().enumerate() {
        if zero_run >= 2 && byte == 0x03 {
            zero_run = 0;
            continue;
        }
        rbsp.push(byte);
        indices.push(index);
        zero_run = if byte == 0 { zero_run + 1 } else { 0 };
    }
    Some((rbsp, indices))
}

fn read_sei_uint(rbsp: &[u8], mut pos: usize) -> Option<(usize, usize)> {
    let mut value = 0;
    while pos < rbsp.len() && rbsp[pos] == 0xFF {
        value += 255;
        pos += 1;
    }
    Some((value + usize::from(*rbsp.get(pos)?), pos + 1))
}

/// Raw file span of `[start, end)` in RBSP coordinates, when no escape byte falls inside it.
fn raw_span(map: Option<&[usize]>, base: u64, start: usize, end: usize) -> Option<Span> {
    match map {
        None => Some(Span::new(base + start as u64, base + end as u64)),
        Some(map) => {
            let raw_start = map[start];
            let raw_end = map[end - 1] + 1;
            (raw_end - raw_start == end - start)
                .then(|| Span::new(base + raw_start as u64, base + raw_end as u64))
        }
    }
}

fn is_ai_sei_body(body: &[u8]) -> bool {
    let lower = body.to_ascii_lowercase();
    contains(&lower, b"urn:c2pa:")
        || (contains(&lower, b"jumb") && contains(&lower, b"c2pa"))
        || (contains(&lower, b"\"contentproducer\"") && contains(&lower, b"\"label\""))
}

/// Parse the SEI messages of one NAL body that starts at file offset `base`.
fn scan_sei_rbsp(raw: &[u8], base: u64, findings: &mut Findings) {
    let unescaped = unescape_rbsp(raw);
    let (rbsp, map) = match &unescaped {
        Some((rbsp, map)) => (rbsp.as_slice(), Some(map.as_slice())),
        None => (raw, None),
    };
    let mut pos = 0;
    while pos + 2 <= rbsp.len() && rbsp[pos] != 0x80 {
        let Some((payload_type, next)) = read_sei_uint(rbsp, pos) else {
            return;
        };
        let Some((payload_size, next)) = read_sei_uint(rbsp, next) else {
            return;
        };
        pos = next;
        if pos + payload_size > rbsp.len() {
            return;
        }
        if payload_type == 5 && payload_size >= 16 {
            let message = &rbsp[pos..pos + payload_size];
            let uuid = &message[..16];
            if uuid != X264_SEI_UUID && !message.starts_with(KLING_SEI_TAG) {
                findings.uuids.insert(hex::encode(uuid));
            }
            for found in memchr::memmem::find_iter(message, KLING_SEI_TAG) {
                let start = pos + found;
                if let Some(span) = raw_span(map, base, start, start + KLING_SEI_TAG.len()) {
                    findings.kling.insert(span.start);
                }
            }
            if payload_size > 16 {
                let is_ai = uuid == C2PA_UUID || (uuid != X264_SEI_UUID && is_ai_sei_body(&message[16..]));
                if is_ai && let Some(span) = raw_span(map, base, pos, pos + payload_size) {
                    findings.ai_spans.insert(span);
                }
            }
        }
        pos += payload_size;
    }
}

const NAL_CODECS: [(&FourCc, Family, &FourCc, u64); 4] = [
    (b"avc1", Family::Avc, b"avcC", 4),
    (b"avc3", Family::Avc, b"avcC", 4),
    (b"hvc1", Family::Hevc, b"hvcC", 21),
    (b"hev1", Family::Hevc, b"hvcC", 21),
];

/// Map 1-based `sample_description_index` to the codec family and NAL length size.
fn codec_map<R: ReadAt + ?Sized>(
    source: &R,
    stsd: &BoxEntry,
    findings: &mut Findings,
) -> io::Result<HashMap<u64, Codec>> {
    let mut map = HashMap::new();
    if stsd.payload + 8 > stsd.end {
        return Ok(map);
    }
    let Some(count) = source.read_array::<4>(stsd.payload + 4)? else {
        return Ok(map);
    };
    let entry_count = u64::from(u32::from_be_bytes(count));
    for (index, entry) in (1..).zip(boxes(source, stsd.payload + 8, stsd.end)?) {
        if index > entry_count {
            break;
        }
        let children_start = entry.payload + 78;
        if children_start > entry.end {
            continue;
        }
        if &entry.kind == b"av01" {
            map.insert(
                index,
                Codec {
                    family: Family::Av1,
                    length_size: 0,
                },
            );
            for child in iter_boxes(source, children_start, entry.end) {
                let child = child?;
                if &child.kind == b"av1C" && child.payload + 4 < child.end {
                    let length = (child.end - (child.payload + 4)).min(MAX_SEI_NAL_BYTES);
                    let config = source.read_vec(child.payload + 4, length)?;
                    findings
                        .av1
                        .extend(av1::scan_sample_spans(&config, child.payload + 4));
                    break;
                }
            }
            continue;
        }
        let Some(&(_, family, config_kind, length_offset)) =
            NAL_CODECS.iter().find(|spec| spec.0 == &entry.kind)
        else {
            continue;
        };
        for child in iter_boxes(source, children_start, entry.end) {
            let child = child?;
            if &child.kind == config_kind && child.payload + length_offset < child.end {
                let Some([byte]) = source.read_array::<1>(child.payload + length_offset)? else {
                    break;
                };
                map.insert(
                    index,
                    Codec {
                        family,
                        length_size: usize::from(byte & 0x03) + 1,
                    },
                );
                break;
            }
        }
    }
    Ok(map)
}

fn read_u32s<R: ReadAt + ?Sized>(source: &R, offset: u64, count: u64) -> io::Result<Vec<u32>> {
    let raw = source.read_vec(offset, count * 4)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_u32_at<R: ReadAt + ?Sized>(source: &R, offset: u64) -> io::Result<u64> {
    Ok(source
        .read_array::<4>(offset)?
        .map_or(0, |b| u64::from(u32::from_be_bytes(b))))
}

struct SampleTable {
    chunk_runs: Vec<[u32; 3]>,
    fixed_size: u64,
    sample_count: u64,
    sample_sizes: Vec<u32>,
    chunk_offsets: Vec<u64>,
}

/// Parse `stsc`, `stsz` and `stco`/`co64`, or `None` for a malformed or oversized table.
fn sample_table<R: ReadAt + ?Sized>(source: &R, children: &[BoxEntry]) -> io::Result<Option<SampleTable>> {
    let (Some(stsc), Some(stsz)) = (last_of_type(children, b"stsc"), last_of_type(children, b"stsz")) else {
        return Ok(None);
    };
    let (co, width) = match (last_of_type(children, b"stco"), last_of_type(children, b"co64")) {
        (Some(stco), _) => (stco, 4),
        (None, Some(co64)) => (co64, 8),
        (None, None) => return Ok(None),
    };
    if stsc.payload + 8 > stsc.end {
        return Ok(None);
    }
    let run_count = read_u32_at(source, stsc.payload + 4)?;
    if run_count == 0 || run_count > MAX_STBL_ENTRIES || stsc.payload + 8 + run_count * 12 > stsc.end {
        return Ok(None);
    }
    let runs = read_u32s(source, stsc.payload + 8, run_count * 3)?;
    let chunk_runs = runs.chunks_exact(3).map(|r| [r[0], r[1], r[2]]).collect();
    if stsz.payload + 12 > stsz.end {
        return Ok(None);
    }
    let fixed_size = read_u32_at(source, stsz.payload + 4)?;
    let sample_count = read_u32_at(source, stsz.payload + 8)?;
    if sample_count > MAX_STBL_ENTRIES {
        return Ok(None);
    }
    let mut sample_sizes = Vec::new();
    if fixed_size == 0 {
        if stsz.payload + 12 + sample_count * 4 > stsz.end {
            return Ok(None);
        }
        sample_sizes = read_u32s(source, stsz.payload + 12, sample_count)?;
    }
    if co.payload + 8 > co.end {
        return Ok(None);
    }
    let chunk_count = read_u32_at(source, co.payload + 4)?;
    if chunk_count > MAX_STBL_ENTRIES || co.payload + 8 + chunk_count * width > co.end {
        return Ok(None);
    }
    let raw = source.read_vec(co.payload + 8, chunk_count * width)?;
    let chunk_offsets = if width == 4 {
        raw.chunks_exact(4)
            .map(|c| u64::from(u32::from_be_bytes([c[0], c[1], c[2], c[3]])))
            .collect()
    } else {
        raw.chunks_exact(8)
            .map(|c| u64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    };
    Ok(Some(SampleTable {
        chunk_runs,
        fixed_size,
        sample_count,
        sample_sizes,
        chunk_offsets,
    }))
}

/// Walk one video track's samples; `false` when its sample table is unusable.
fn scan_stbl<R: ReadAt + ?Sized>(
    source: &R,
    stbl: &BoxEntry,
    findings: &mut Findings,
    stop_on_first_kling: bool,
) -> io::Result<bool> {
    let children = boxes(source, stbl.payload, stbl.end)?;
    let Some(stsd) = last_of_type(&children, b"stsd") else {
        return Ok(false);
    };
    let required = [b"stsc", b"stsz"]
        .iter()
        .all(|kind| last_of_type(&children, *kind).is_some());
    let has_offsets =
        last_of_type(&children, b"stco").is_some() || last_of_type(&children, b"co64").is_some();
    if !required || !has_offsets {
        return Ok(false);
    }
    let codecs = codec_map(source, &stsd, findings)?;
    if codecs.is_empty() {
        return Ok(true);
    }
    if stop_on_first_kling && !findings.av1.is_empty() {
        return Ok(true);
    }
    let Some(table) = sample_table(source, &children)? else {
        return Ok(false);
    };
    let file_size = source.size();
    let mut run_index = 0;
    let mut sample_index = 0u64;
    for (chunk_number, &chunk_offset) in (1u64..).zip(&table.chunk_offsets) {
        while run_index + 1 < table.chunk_runs.len()
            && chunk_number >= u64::from(table.chunk_runs[run_index + 1][0])
        {
            run_index += 1;
        }
        let [_, samples_per_chunk, description] = table.chunk_runs[run_index];
        let codec = codecs.get(&u64::from(description)).copied();
        let mut offset = chunk_offset;
        for _ in 0..samples_per_chunk {
            if sample_index >= table.sample_count {
                break;
            }
            let size = if table.fixed_size > 0 {
                table.fixed_size
            } else {
                u64::from(table.sample_sizes[sample_index as usize])
            };
            sample_index += 1;
            if let Some(codec) = codec
                && size > 0
                && offset + size <= file_size
            {
                let stop = scan_sample(
                    source,
                    codec,
                    offset,
                    offset + size,
                    findings,
                    stop_on_first_kling,
                )?;
                if stop {
                    return Ok(true);
                }
            }
            offset += size;
        }
    }
    Ok(true)
}

/// Scan one sample; `true` when a first-hit scan may stop.
fn scan_sample<R: ReadAt + ?Sized>(
    source: &R,
    codec: Codec,
    start: u64,
    end: u64,
    findings: &mut Findings,
    stop_on_first_kling: bool,
) -> io::Result<bool> {
    if codec.family == Family::Av1 {
        findings
            .av1
            .extend(av1::scan_stream_spans(source, start, end, stop_on_first_kling)?);
        return Ok(stop_on_first_kling && !findings.av1.is_empty());
    }
    let length_size = codec.length_size as u64;
    let mut nal_pos = start;
    while nal_pos + length_size < end {
        let header = source.read_vec(nal_pos, (length_size + 2).min(end - nal_pos))?;
        if header.len() <= codec.length_size {
            break;
        }
        let nal_len = header[..codec.length_size]
            .iter()
            .fold(0u64, |acc, &b| (acc << 8) | u64::from(b));
        let body_start = nal_pos + length_size;
        if nal_len == 0 || body_start + nal_len > end {
            break;
        }
        let nal_header = header[codec.length_size];
        let (header_len, is_sei) = match codec.family {
            Family::Avc => (1, nal_header & 0x1F == 6),
            _ => (
                2,
                header.len() >= codec.length_size + 2 && matches!((nal_header >> 1) & 0x3F, 39 | 40),
            ),
        };
        if is_sei && nal_len > header_len {
            let sei_start = body_start + header_len;
            let raw = source.read_vec(sei_start, (nal_len - header_len).min(MAX_SEI_NAL_BYTES))?;
            scan_sei_rbsp(&raw, sei_start, findings);
            if stop_on_first_kling && !findings.kling.is_empty() {
                return Ok(true);
            }
        }
        nal_pos = body_start + nal_len;
    }
    Ok(false)
}

/// Scan every track of an open ISOBMFF source.
pub fn scan_source_video_sei<R: ReadAt + ?Sized>(
    source: &R,
    stop_on_first_kling: bool,
) -> io::Result<VideoSeiScan> {
    let top = boxes(source, 0, source.size())?;
    let has_moof = top.iter().any(|entry| &entry.kind == b"moof");
    let mut parsed_tracks = 0;
    let mut all_parsed = true;
    let mut findings = Findings::default();
    for moov in top.iter().filter(|entry| &entry.kind == b"moov") {
        for trak in boxes(source, moov.payload, moov.end)?
            .into_iter()
            .filter(|e| &e.kind == b"trak")
        {
            match scan_track(source, &trak, &mut findings, stop_on_first_kling)? {
                Some(true) => parsed_tracks += 1,
                _ => all_parsed = false,
            }
        }
    }
    Ok(VideoSeiScan {
        has_parsed_tracks: parsed_tracks > 0 && all_parsed && !has_moof,
        kling_tag_offsets: findings.kling.into_iter().collect(),
        unregistered_uuids: findings.uuids.into_iter().collect(),
        ai_sei_spans: findings.ai_spans.into_iter().collect(),
        av1_obu_patches: findings.av1.into_iter().collect(),
    })
}

/// `Some(true)` for a parsed track, `Some(false)`/`None` for one that could not be walked.
fn scan_track<R: ReadAt + ?Sized>(
    source: &R,
    trak: &BoxEntry,
    findings: &mut Findings,
    stop_on_first_kling: bool,
) -> io::Result<Option<bool>> {
    let mdias: Vec<_> = boxes(source, trak.payload, trak.end)?
        .into_iter()
        .filter(|e| &e.kind == b"mdia")
        .collect();
    let [mdia] = mdias.as_slice() else { return Ok(None) };
    let mdia_children = boxes(source, mdia.payload, mdia.end)?;
    let (Some(hdlr), Some(minf)) = (
        last_of_type(&mdia_children, b"hdlr"),
        last_of_type(&mdia_children, b"minf"),
    ) else {
        return Ok(None);
    };
    if hdlr.payload + 12 > hdlr.end {
        return Ok(None);
    }
    let handler = source.read_vec(hdlr.payload + 8, 4)?;
    let Some(stbl) = last_of_type(&boxes(source, minf.payload, minf.end)?, b"stbl") else {
        return Ok(None);
    };
    if handler != b"vide" || (stop_on_first_kling && !findings.kling.is_empty()) {
        return Ok(Some(true));
    }
    Ok(Some(scan_stbl(source, &stbl, findings, stop_on_first_kling)?))
}

/// Walk ISOBMFF sample tables for H.264/HEVC SEI and AV1 metadata provenance.
///
/// A non-ISOBMFF file yields an empty, unparsed scan; `Err` is Python's `strict=True` raise.
pub fn scan_video_sei(path: &Path, stop_on_first_kling: bool) -> io::Result<VideoSeiScan> {
    let source = FileSource::open(path)?;
    if !is_isobmff(&source.read_vec(0, 8)?) {
        return Ok(VideoSeiScan::default());
    }
    scan_source_video_sei(&source, stop_on_first_kling)
}

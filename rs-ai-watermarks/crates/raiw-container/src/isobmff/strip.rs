//! Same-length provenance removal: boxes become `free`, labels are blanked in place.
//!
//! No byte moves, so `iloc`, `stco`/`co64` and every absolute media offset stay valid.

use std::collections::{BTreeSet, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::LazyLock;

use memchr::memmem;
use raiw_core::StagedSibling;
use regex::bytes::Regex;

use super::boxes::{Span, is_isobmff, read_box_header, top_level_boxes};
use super::items::metadata_items;
use super::labels::{
    LabelRegion, ai_metadata_regions, app_regions, audio_creation_regions, tc260_regions, xai_regions,
};
use super::region::has_c2pa_emsg_marker;
use super::sei::{VideoSeiScan, scan_source_video_sei};
use crate::av1;
use crate::hooks::MetadataHooks;
use crate::markers::{AI_LABEL_MARKERS, C2PA_UUID, KLING_SEI_TAG, contains_any};
use crate::source::{FileSource, ReadAt};

const STREAM_COPY_BYTES: usize = 1024 * 1024;
/// Default bound on how much of one metadata box is searched for an AI label.
pub const STREAM_SCAN_BYTES: u64 = 4 * 1024 * 1024;
const EXIF_WINDOW: usize = 256 * 1024;
const TIFF_HEADERS: [&[u8]; 2] = [b"II\x2a\x00", b"MM\x00\x2a"];

static XMP_PACKET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s-u)<\?xpacket begin=.*?<\?xpacket end=[^>]*?\?>").expect("static pattern")
});

/// Whether a metadata payload carries an AI-label marker or names a known AI generator.
pub fn has_ai_label_or_generator(payload: &[u8], hooks: &impl MetadataHooks) -> bool {
    contains_any(payload, AI_LABEL_MARKERS) || hooks.names_ai_generator(payload)
}

/// `data[start..end]` as a same-length `free` box with a zeroed payload.
fn blank_box(data: &[u8], start: usize, end: usize, payload: usize, out: &mut Vec<u8>) {
    out.extend_from_slice(&data[start..start + 4]);
    out.extend_from_slice(b"free");
    out.extend_from_slice(&data[start + 8..payload]);
    out.resize(out.len() + (end - payload), 0);
}

/// Blank C2PA `uuid`, AI-label `uuid`, `jumb` and C2PA `emsg` top-level boxes.
///
/// Returns the input unchanged (count 0) for non-ISOBMFF data or a walk that stops early.
pub fn strip_c2pa_boxes(data: &[u8], hooks: &impl MetadataHooks) -> (Vec<u8>, usize) {
    if !is_isobmff(data) {
        return (data.to_vec(), 0);
    }
    let mut out = Vec::with_capacity(data.len());
    let mut stripped = 0;
    let mut consumed = 0;
    for entry in top_level_boxes(data) {
        let (start, end, payload) = (entry.start as usize, entry.end as usize, entry.payload as usize);
        consumed = end;
        let provenance = match &entry.kind {
            b"jumb" => true,
            b"uuid" => {
                let body = &data[payload..end];
                (body.len() >= 16 && body[..16] == C2PA_UUID) || has_ai_label_or_generator(body, hooks)
            }
            b"emsg" => has_c2pa_emsg_marker(&data[payload..end.min(payload + 256)]),
            _ => false,
        };
        if provenance {
            stripped += 1;
            blank_box(data, start, end, payload, &mut out);
        } else {
            out.extend_from_slice(&data[start..end]);
        }
    }
    if consumed != data.len() {
        return (data.to_vec(), 0);
    }
    (out, stripped)
}

/// Overwrite every AI-label XMP packet with spaces, keeping plain XMP.
pub fn blank_ai_xmp_packets(data: &[u8], hooks: &impl MetadataHooks) -> (Vec<u8>, usize) {
    let mut out = data.to_vec();
    let mut blanked = 0;
    for found in XMP_PACKET.find_iter(data) {
        if has_ai_label_or_generator(found.as_bytes(), hooks) {
            blanked += 1;
            out[found.range()].fill(b' ');
        }
    }
    (out, blanked)
}

/// EXIF scan state carried across streaming windows.
struct ExifScan<'a> {
    seen: Option<&'a mut HashSet<u64>>,
    skip: &'a [Span],
}

impl ExifScan<'_> {
    fn skipped(&self, start: u64, end: u64) -> bool {
        self.skip.iter().any(|span| span.encloses(start, end))
    }

    /// Absolute spans of AI EXIF values in `window`, whose first byte is at `base`.
    fn find(&mut self, window: &[u8], base: u64, at_eof: bool, hooks: &impl MetadataHooks) -> Vec<Span> {
        let mut spans = Vec::new();
        for header in TIFF_HEADERS {
            for relative in memmem::find_iter(window, header) {
                let absolute = base + relative as u64;
                let unseen = self.seen.as_ref().is_none_or(|seen| !seen.contains(&absolute));
                if !(unseen
                    && !self.skipped(absolute, absolute + header.len() as u64)
                    && (at_eof || relative + EXIF_WINDOW <= window.len()))
                {
                    continue;
                }
                if let Some(seen) = self.seen.as_deref_mut() {
                    seen.insert(absolute);
                }
                let slice = &window[relative..(relative + EXIF_WINDOW).min(window.len())];
                for value in hooks.ai_exif_values(slice) {
                    if let Some(found) = memmem::find(slice, &value) {
                        let start = absolute + found as u64;
                        let end = start + value.len() as u64;
                        if !self.skipped(start, end) {
                            spans.push(Span::new(start, end));
                        }
                    }
                }
            }
        }
        spans
    }
}

/// Overwrite AI-generator EXIF values (a HEIF/AVIF `Exif` item in `mdat`/`idat`) with spaces.
pub fn blank_ai_exif_tokens(data: &[u8], hooks: &impl MetadataHooks) -> (Vec<u8>, usize) {
    let spans = ExifScan {
        seen: None,
        skip: &[],
    }
    .find(data, 0, true, hooks);
    let mut out = data.to_vec();
    for span in &spans {
        out[span.start as usize..span.end as usize].fill(b' ');
    }
    (out, spans.len())
}

/// Blank native TC260 values in place (key name becomes `free`, value becomes spaces).
pub fn blank_tc260_aigc_tags(data: &[u8], hooks: &impl MetadataHooks) -> (Vec<u8>, usize) {
    if !is_isobmff(data) {
        return (data.to_vec(), 0);
    }
    let regions = metadata_items(data)
        .map(|items| tc260_regions(&items, hooks))
        .unwrap_or_default();
    if regions.is_empty() {
        return (data.to_vec(), 0);
    }
    let mut out = data.to_vec();
    let mut keys = HashSet::new();
    for (region, _) in &regions {
        if let Some(key) = region.key_span {
            keys.insert(key);
            out.splice(key.start as usize..key.end as usize, *b"free");
        }
        out[region.value.start as usize..region.value.end as usize].fill(b' ');
    }
    (out, keys.len())
}

/// A top-level provenance box: `[start, end)` with its payload beginning at `payload`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvenanceBox {
    pub start: u64,
    pub payload: u64,
    pub end: u64,
}

/// Scan a bounded prefix of one metadata payload for an AI label, 1 MiB at a time.
fn payload_has_ai_label<R: ReadAt + ?Sized>(
    source: &R,
    start: u64,
    end: u64,
    max_scan: u64,
    hooks: &impl MetadataHooks,
) -> io::Result<bool> {
    let keep = AI_LABEL_MARKERS
        .iter()
        .map(|marker| marker.len())
        .max()
        .unwrap_or(1)
        - 1;
    let mut remaining = end.saturating_sub(start).min(max_scan);
    let mut overlap: Vec<u8> = Vec::new();
    let mut pos = start;
    while remaining > 0 {
        let chunk = source.read_vec(pos, remaining.min(STREAM_COPY_BYTES as u64))?;
        if chunk.is_empty() {
            return Ok(false);
        }
        let mut searchable = std::mem::take(&mut overlap);
        searchable.extend_from_slice(&chunk);
        if has_ai_label_or_generator(&searchable, hooks) {
            return Ok(true);
        }
        overlap = searchable[searchable.len().saturating_sub(keep)..].to_vec();
        remaining -= chunk.len() as u64;
        pos += chunk.len() as u64;
    }
    Ok(false)
}

/// Top-level provenance boxes, or `None` when the top-level walk is malformed.
pub fn streaming_provenance_boxes<R: ReadAt + ?Sized>(
    source: &R,
    max_scan: u64,
    hooks: &impl MetadataHooks,
) -> io::Result<Option<Vec<ProvenanceBox>>> {
    if !is_isobmff(&source.read_vec(0, 8)?) {
        return Ok(None);
    }
    let file_size = source.size();
    let mut targets = Vec::new();
    let mut pos = 0;
    while pos < file_size {
        let Some(entry) = read_box_header(source, pos, file_size)? else {
            return Ok(None);
        };
        let target = ProvenanceBox {
            start: pos,
            payload: entry.payload,
            end: entry.end,
        };
        match &entry.kind {
            b"uuid" => {
                let is_c2pa = entry.payload + 16 <= entry.end
                    && source.read_array::<16>(entry.payload)? == Some(C2PA_UUID);
                if is_c2pa || payload_has_ai_label(source, entry.payload, entry.end, max_scan, hooks)? {
                    targets.push(target);
                }
            }
            b"jumb" => targets.push(target),
            b"emsg" => {
                let prefix = source.read_vec(entry.payload, (entry.end - entry.payload).min(256))?;
                if has_c2pa_emsg_marker(&prefix) {
                    targets.push(target);
                }
            }
            _ => {}
        }
        pos = entry.end;
    }
    Ok(Some(targets))
}

fn overwrite_range<W: Write + Seek>(target: &mut W, start: u64, end: u64, byte: u8) -> io::Result<()> {
    if end <= start {
        return Ok(());
    }
    target.seek(SeekFrom::Start(start))?;
    let mut remaining = end - start;
    let block = vec![byte; remaining.min(STREAM_COPY_BYTES as u64) as usize];
    while remaining > 0 {
        let size = remaining.min(block.len() as u64) as usize;
        target.write_all(&block[..size])?;
        remaining -= size as u64;
    }
    Ok(())
}

fn apply_box_and_region_patches<W: Write + Seek>(
    target: &mut W,
    boxes: &[ProvenanceBox],
    regions: &[LabelRegion],
) -> io::Result<()> {
    for found in boxes {
        target.seek(SeekFrom::Start(found.start + 4))?;
        target.write_all(b"free")?;
        overwrite_range(target, found.payload, found.end, 0)?;
    }
    for region in regions {
        if let Some(key) = region.key_span {
            let length = key.len() as usize;
            let mut blank = b"free".to_vec();
            blank.resize(length.max(4), b' ');
            target.seek(SeekFrom::Start(key.start))?;
            target.write_all(&blank[..length])?;
        }
        overwrite_range(target, region.value.start, region.value.end, b' ')?;
    }
    Ok(())
}

/// Options of [`strip_media_file`].
#[derive(Debug, Clone, Copy)]
pub struct MediaStripOptions {
    pub max_box_scan: u64,
    /// Also blank Suno-style creation comments; the caller must confirm audio-only scope.
    pub strip_audio_comments: bool,
}

impl Default for MediaStripOptions {
    fn default() -> Self {
        Self {
            max_box_scan: STREAM_SCAN_BYTES,
            strip_audio_comments: false,
        }
    }
}

/// What [`strip_media_file`] removed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MediaStripCounts {
    pub provenance_boxes: usize,
    pub native_tags: usize,
    /// The top-level walk failed, so the input was copied unchanged.
    pub walk_failed: bool,
}

/// Copy `source` to `target`, returning each Kling SEI tag offset inside `mdat` ranges.
fn copy_finding_kling(source: &mut File, target: &mut File, mdat: &[Span]) -> io::Result<Vec<u64>> {
    let keep = KLING_SEI_TAG.len() - 1;
    let mut offsets = Vec::new();
    let mut window: Vec<u8> = Vec::with_capacity(STREAM_COPY_BYTES + keep);
    let mut chunk = vec![0u8; STREAM_COPY_BYTES];
    let mut pos = 0u64;
    loop {
        let read = read_full(source, &mut chunk)?;
        if read == 0 {
            break;
        }
        target.write_all(&chunk[..read])?;
        if !mdat.is_empty() {
            let tail = window.len();
            window.extend_from_slice(&chunk[..read]);
            let base = pos - tail as u64;
            for found in memmem::find_iter(&window, KLING_SEI_TAG) {
                let start = base + found as u64;
                if mdat
                    .iter()
                    .any(|span| span.encloses(start, start + KLING_SEI_TAG.len() as u64))
                {
                    offsets.push(start);
                }
            }
            window.drain(..window.len().saturating_sub(keep));
        }
        pos += read as u64;
    }
    Ok(offsets)
}

/// `read` until `buf` is full or the reader ends, like Python's buffered `read(n)`.
fn read_full(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
            Err(err) => return Err(err),
        }
    }
    Ok(filled)
}

fn publish(stage: StagedSibling, source: &Path, file: &File) -> io::Result<()> {
    file.sync_all()?;
    std::fs::set_permissions(stage.path(), std::fs::metadata(source)?.permissions())?;
    stage.commit()?;
    Ok(())
}

fn open_staged(stage: &StagedSibling) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(stage.path())
}

/// Stream-copy an MP4/MOV/M4A to `output`, removing supported AI metadata in place.
///
/// Every box size and byte offset is retained; the output is published atomically. A
/// malformed top-level walk copies the input unchanged.
pub fn strip_media_file(
    source: &Path,
    output: &Path,
    options: MediaStripOptions,
    hooks: &impl MetadataHooks,
) -> io::Result<MediaStripCounts> {
    let input = FileSource::open(source)?;
    let targets = streaming_provenance_boxes(&input, options.max_box_scan, hooks)?;
    let mut regions = Vec::new();
    let mut audio = Vec::new();
    let mut sei = VideoSeiScan::default();
    let mut mdat = Vec::new();
    if targets.is_some() {
        let items = metadata_items(&input)?;
        regions.extend(tc260_regions(&items, hooks).into_iter().map(|(region, _)| region));
        regions.extend(
            ai_metadata_regions(&items)
                .into_iter()
                .map(|(region, _, _)| region),
        );
        regions.extend(app_regions(&items, hooks));
        regions.extend(xai_regions(&items).into_iter().map(|(region, _)| region));
        if options.strip_audio_comments {
            audio = audio_creation_regions(&metadata_items(&input)?, hooks);
        }
        sei = scan_source_video_sei(&input, false)?;
        if !sei.has_parsed_tracks {
            mdat = super::boxes::boxes(&input, 0, input.size())?
                .into_iter()
                .filter(|entry| &entry.kind == b"mdat")
                .map(|entry| Span::new(entry.payload, entry.end))
                .collect();
        }
    }
    let blanked_keys: BTreeSet<Span> = regions.iter().filter_map(|region| region.key_span).collect();
    drop(input);

    let stage = StagedSibling::new(output)?;
    let mut staged = open_staged(&stage)?;
    let mut reader = File::open(source)?;
    let kling = if sei.has_parsed_tracks {
        io::copy(&mut reader, &mut staged)?;
        sei.kling_tag_offsets.clone()
    } else {
        copy_finding_kling(&mut reader, &mut staged, &mdat)?
    };
    let Some(targets) = targets else {
        publish(stage, source, &staged)?;
        return Ok(MediaStripCounts {
            walk_failed: true,
            ..MediaStripCounts::default()
        });
    };
    apply_box_and_region_patches(&mut staged, &targets, &regions)?;
    for span in &audio {
        overwrite_range(&mut staged, span.start, span.end, b' ')?;
    }
    // Spaces keep the SEI message well-formed and cannot form an emulation-prevention run.
    for &offset in &kling {
        overwrite_range(&mut staged, offset, offset + KLING_SEI_TAG.len() as u64, b' ')?;
    }
    for span in &sei.ai_sei_spans {
        overwrite_range(&mut staged, span.start, span.end, b' ')?;
    }
    for patch in &sei.av1_obu_patches {
        av1::patch_to_padding(&mut staged, patch)?;
    }
    staged.flush()?;
    publish(stage, source, &staged)?;
    let native_tags = blanked_keys.len()
        + usize::from(!kling.is_empty())
        + usize::from(!sei.ai_sei_spans.is_empty())
        + usize::from(!sei.av1_obu_patches.is_empty())
        + audio.len();
    Ok(MediaStripCounts {
        provenance_boxes: targets.len(),
        native_tags,
        walk_failed: false,
    })
}

/// What [`strip_image_file`] removed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageStripCounts {
    pub provenance_boxes: usize,
    pub tc260: usize,
    pub xmp: usize,
    pub exif: usize,
}

/// Stream-copy a HEIF/AVIF/JXL image, blanking AI boxes, TC260, XMP packets and EXIF values.
pub fn strip_image_file(
    source: &Path,
    output: &Path,
    max_box_scan: u64,
    hooks: &impl MetadataHooks,
) -> io::Result<ImageStripCounts> {
    let input = FileSource::open(source)?;
    let file_size = input.size();
    let targets = streaming_provenance_boxes(&input, max_box_scan, hooks)?;
    let tc260: Vec<LabelRegion> = match targets {
        Some(_) => tc260_regions(&metadata_items(&input)?, hooks)
            .into_iter()
            .map(|(region, _)| region)
            .collect(),
        None => Vec::new(),
    };
    drop(input);
    let target_ranges: Vec<Span> = targets
        .iter()
        .flatten()
        .map(|found| Span::new(found.start, found.end))
        .collect();
    let inside_target = |start: u64, end: u64| target_ranges.iter().any(|span| span.encloses(start, end));

    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let stage = StagedSibling::new(output)?;
    let mut staged = open_staged(&stage)?;
    let mut reader = File::open(source)?;
    let mut xmp_spans = BTreeSet::new();
    let mut exif_spans = BTreeSet::new();
    let mut seen = HashSet::new();
    let mut tail: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; STREAM_COPY_BYTES];
    let mut offset = 0u64;
    loop {
        let read = read_full(&mut reader, &mut chunk)?;
        if read == 0 {
            break;
        }
        staged.write_all(&chunk[..read])?;
        let tail_len = tail.len();
        let mut window = std::mem::take(&mut tail);
        window.extend_from_slice(&chunk[..read]);
        let base = offset - tail_len as u64;
        let at_eof = offset + read as u64 >= file_size;
        for found in XMP_PACKET.find_iter(&window) {
            if found.end() <= tail_len {
                continue;
            }
            let (start, end) = (base + found.start() as u64, base + found.end() as u64);
            if !inside_target(start, end) && has_ai_label_or_generator(found.as_bytes(), hooks) {
                xmp_spans.insert(Span::new(start, end));
            }
        }
        let mut scan = ExifScan {
            seen: Some(&mut seen),
            skip: &target_ranges,
        };
        exif_spans.extend(scan.find(&window, base, at_eof, hooks));
        offset += read as u64;
        tail = if window.len() > EXIF_WINDOW {
            window[window.len() - EXIF_WINDOW..].to_vec()
        } else {
            window
        };
    }
    if let Some(found) = &targets {
        apply_box_and_region_patches(&mut staged, found, &tc260)?;
    }
    for span in xmp_spans.union(&exif_spans) {
        overwrite_range(&mut staged, span.start, span.end, b' ')?;
    }
    staged.flush()?;
    publish(stage, source, &staged)?;
    Ok(ImageStripCounts {
        provenance_boxes: targets.map_or(0, |found| found.len()),
        tc260: tc260.len(),
        xmp: xmp_spans.len(),
        exif: exif_spans.len(),
    })
}

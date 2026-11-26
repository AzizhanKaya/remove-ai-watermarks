//! `collect_metadata_record`: the portable, JSON-safe record the provenance verdict reads.

use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use raiw_container::gif::{DEFAULT_LATE_BYTES as GIF_LATE, extension_metadata};
use raiw_container::isobmff::{DEFAULT_REGION_BYTES, is_isobmff, scan_c2pa_region};
use raiw_container::markers::SAMSUNG_EDITOR_MARKER;
use raiw_container::{gif, png, riff};
use raiw_core::constants::{PNG_SIGNATURE, RIFF_CODED_IMAGE_CHUNKS};
use serde_json::{Map, Value, json};

use crate::exif::{self, Ifd, tag};
use crate::hooks::Hooks;
use crate::pil::{self, Info, InfoValue, OpenError};
use crate::scan::{QUICK_SCAN_BYTES, read_file_tail, read_head};

pub const HEAD_WINDOW: usize = 1024 * 1024;
pub const TAIL_WINDOW: usize = QUICK_SCAN_BYTES;
pub const UNKNOWN_TRAILER_WINDOW: usize = 64 * 1024;
pub const METADATA_RECORD_SCHEMA_VERSION: i64 = 1;
pub const METADATA_RECORD_TYPE: &str = "provenance_metadata";
const GENERATOR_TEXT_KEYS: [&str; 4] = ["Software", "Source", "Title", "Description"];

fn u32_be(d: &[u8], at: usize) -> usize {
    u32::from_be_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]) as usize
}

fn u32_le(d: &[u8], at: usize) -> usize {
    u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]) as usize
}

/// `_jpeg_regions`: marker segments up to the coded scan.
pub fn jpeg_regions(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut index, size) = (2usize, data.len());
    while index + 1 < size {
        if data[index] != 0xFF {
            break;
        }
        let marker = data[index + 1];
        if matches!(marker, 0xDA | 0xD9) {
            break;
        }
        if matches!(marker, 0xD0..=0xD7 | 0x01) {
            index += 2;
            continue;
        }
        if index + 4 > size {
            break;
        }
        let length = usize::from(u16::from_be_bytes([data[index + 2], data[index + 3]]));
        let end = index + 2 + length;
        if length < 2 || end > size {
            break;
        }
        out.extend_from_slice(&data[index..end]);
        index = end;
    }
    out
}

/// `_png_regions`: every chunk but the IDAT payloads, plus bytes after IEND.
pub fn png_regions(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let size = data.len();
    let mut position = PNG_SIGNATURE.len();
    while position + 8 <= size {
        let length = u32_be(data, position);
        let kind = &data[position + 4..position + 8];
        let start = position + 8;
        let safe = length.min(size - start);
        if kind != b"IDAT" {
            out.extend_from_slice(kind);
            out.extend_from_slice(&data[start..start + safe]);
        }
        position = start + safe + 4;
        if kind == b"IEND" {
            out.extend_from_slice(&data[position.min(size)..]);
            break;
        }
    }
    out
}

/// `_riff_regions`: every RIFF chunk but the coded image.
pub fn riff_regions(data: &[u8]) -> Vec<u8> {
    let mut out = data[..data.len().min(12)].to_vec();
    let declared_end = if data.len() >= 12 {
        8 + u32_le(data, 4)
    } else {
        data.len()
    };
    let size = data.len().min(declared_end);
    let mut position = 12;
    while position + 8 <= size {
        let kind = &data[position..position + 4];
        let length = u32_le(data, position + 4);
        let start = position + 8;
        let safe = length.min(size - start);
        if !RIFF_CODED_IMAGE_CHUNKS.contains(&kind) {
            out.extend_from_slice(kind);
            out.extend_from_slice(&data[start..start + safe]);
        }
        position = start + safe + (safe & 1);
    }
    out
}

/// `_gif_regions`: the header plus comment and non-playback application extensions.
pub fn gif_regions(data: &[u8]) -> Vec<u8> {
    if data.len() < 13 {
        return data.to_vec();
    }
    let packed = data[10];
    let mut pos = 13
        + if packed & 0x80 != 0 {
            3 * (1 << ((packed & 7) + 1))
        } else {
            0
        };
    if pos > data.len() {
        return data[..13].to_vec();
    }
    let mut out = data[..13].to_vec();
    let size = data.len();
    let sub_blocks_end = |start: usize| {
        let mut cursor = start;
        while cursor < size {
            let step = usize::from(data[cursor]);
            cursor += 1 + step;
            if step == 0 {
                return (cursor <= size).then_some(cursor);
            }
        }
        None
    };
    while pos < size {
        match data[pos] {
            0x3B => {
                out.extend_from_slice(&data[pos..]);
                break;
            }
            0x2C => {
                if pos + 10 >= size {
                    break;
                }
                let local = data[pos + 9];
                let table = if local & 0x80 != 0 {
                    3 * (1 << ((local & 7) + 1))
                } else {
                    0
                };
                let Some(end) = sub_blocks_end(pos + 10 + table + 1) else {
                    break;
                };
                pos = end;
            }
            0x21 if pos + 2 <= size => {
                let Some(end) = sub_blocks_end(pos + 2) else { break };
                out.extend(extension_metadata(&data[pos..end]));
                pos = end;
            }
            _ => break,
        }
    }
    out
}

fn container_regions(path: &Path, head: &[u8]) -> (&'static str, std::io::Result<Vec<u8>>) {
    let window = HEAD_WINDOW as u64;
    if head.starts_with(b"\xff\xd8") {
        return ("jpeg", Ok(jpeg_regions(head)));
    }
    if head.starts_with(PNG_SIGNATURE) {
        let mut out = png_regions(head);
        return (
            "png",
            png::late_metadata_file(path, window).map(|late| {
                out.extend(late);
                out
            }),
        );
    }
    if head.starts_with(b"RIFF") && head.get(8..12) == Some(b"WEBP") {
        let mut out = riff_regions(head);
        return (
            "webp",
            riff::late_metadata_file(path, window, riff::DEFAULT_LATE_BYTES).map(|late| {
                out.extend(late);
                out
            }),
        );
    }
    if matches!(head.get(..6), Some(b"GIF87a" | b"GIF89a")) {
        let mut out = gif_regions(head);
        return (
            "gif",
            gif::late_metadata_file(path, window, GIF_LATE).map(|late| {
                out.extend(late);
                out
            }),
        );
    }
    if is_isobmff(head) {
        let regions = (|| {
            let mut out = head[..head.len().min(HEAD_WINDOW)].to_vec();
            out.extend(scan_c2pa_region(path, DEFAULT_REGION_BYTES)?);
            for payload in raiw_container::isobmff::tc260_aigc_payloads(path, &Hooks)? {
                out.extend_from_slice(b"AIGC");
                out.extend(payload);
            }
            Ok(out)
        })();
        return ("isobmff", regions);
    }
    ("unknown", Ok(head.to_vec()))
}

fn trailer(path: &Path, container: &str) -> std::io::Result<Vec<u8>> {
    match container {
        "webp" => {
            use std::io::{Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(path)?;
            let mut header = Vec::new();
            (&mut file).take(12).read_to_end(&mut header)?;
            if header.len() < 12 || !header.starts_with(b"RIFF") {
                return Ok(Vec::new());
            }
            let declared_end = 8 + u32_le(&header, 4) as u64;
            let file_size = file.metadata()?.len();
            if declared_end < 12 || declared_end >= file_size {
                return Ok(Vec::new());
            }
            file.seek(SeekFrom::Start(declared_end))?;
            let mut out = Vec::new();
            file.take((file_size - declared_end).min(UNKNOWN_TRAILER_WINDOW as u64))
                .read_to_end(&mut out)?;
            Ok(out)
        }
        "isobmff" | "gif" => Ok(Vec::new()),
        _ => {
            let tail = read_file_tail(path, TAIL_WINDOW)?;
            if raiw_container::markers::contains(&tail, SAMSUNG_EDITOR_MARKER) {
                return Ok(tail);
            }
            let marker: Option<&[u8]> = match container {
                "jpeg" => Some(b"\xff\xd9"),
                "png" => Some(b"IEND\xae\x42\x60\x82"),
                _ => None,
            };
            let unknown = |tail: &[u8]| tail[tail.len().saturating_sub(UNKNOWN_TRAILER_WINDOW)..].to_vec();
            Ok(
                match marker.and_then(|m| memchr::memmem::rfind(&tail, m).map(|i| i + m.len())) {
                    Some(after) => tail[after..].to_vec(),
                    None => unknown(&tail),
                },
            )
        }
    }
}

/// `_decoder_info`: PIL's `info` (plus the EXIF blob), or `Err` for a decoder failure.
fn decoder_info(path: &Path) -> Result<Info, OpenError> {
    let probe = pil::probe(path);
    match &probe.opened {
        Err(OpenError::Unidentified) => Ok(Info::new()),
        Err(err) => Err(err.clone()),
        Ok(image) => {
            let mut info = image.info.clone();
            if !info.contains_key("exif")
                && let Some(exif) = probe.exif_bytes()
            {
                info.insert("exif".into(), InfoValue::Bytes(exif));
            }
            Ok(info)
        }
    }
}

fn exif_pairs(info: &Info) -> Result<Map<String, Value>, exif::LoadError> {
    let mut pairs = Map::new();
    let raw = match info.get("exif") {
        Some(InfoValue::Bytes(raw)) if !raw.is_empty() => raw.clone(),
        Some(InfoValue::Str(text)) if !text.is_empty() => text.as_bytes().to_vec(),
        _ => return Ok(pairs),
    };
    let dict = exif::load(&raw)?;
    for (name, number) in [
        ("Software", tag::SOFTWARE),
        ("Make", tag::MAKE),
        ("Artist", tag::ARTIST),
        ("ImageDescription", tag::IMAGE_DESCRIPTION),
    ] {
        let text = dict.text(Ifd::Zeroth, number);
        if !text.is_empty() {
            pairs.insert(name.into(), Value::String(text));
        }
    }
    let comment = dict.text(Ifd::Exif, tag::USER_COMMENT);
    if !comment.is_empty() {
        pairs.insert("UserComment".into(), Value::String(comment));
    }
    Ok(pairs)
}

fn text_of(value: &InfoValue) -> String {
    match value {
        InfoValue::Bytes(bytes) => crate::py::utf8_replace(bytes),
        other => other.py_str(),
    }
}

fn pil_info(info: &Info) -> Map<String, Value> {
    let mut out = Map::new();
    for key in GENERATOR_TEXT_KEYS {
        if let Some(value) = info.get(key)
            && *value != InfoValue::None
            && !value.is_container()
        {
            out.insert(format!("info:{key}"), Value::String(text_of(value)));
        }
    }
    for (key, value) in info {
        if GENERATOR_TEXT_KEYS.contains(&key.as_str()) || value.is_container() {
            continue;
        }
        out.insert(format!("info:{key}"), Value::String(text_of(value)));
    }
    out
}

/// `collect_metadata_record(path)`: everything the verdict reads, as a JSON object.
pub fn collect_metadata_record(path: &Path) -> Value {
    let mut status = "complete";
    let mut issues: Vec<Value> = Vec::new();
    let issue = |stage: &str, code: &str| json!({"stage": stage, "code": code});
    let (mut container, mut regions, mut tail) = ("unknown", Vec::new(), Vec::new());
    let mut info = Info::new();
    let mut exif = Map::new();
    let mut store = None;
    if std::fs::metadata(path).is_err() {
        status = "error";
        issues.push(issue("source", "unavailable"));
    }
    if status == "complete" {
        match read_head(path, HEAD_WINDOW) {
            Err(_) => {
                status = "error";
                issues.push(issue("head", "read-failed"));
            }
            Ok(head) => {
                let (kind, collected) = container_regions(path, &head);
                container = kind;
                match collected {
                    Ok(bytes) => regions = bytes,
                    Err(_) => issues.push(issue("regions", "collection-failed")),
                }
                match trailer(path, container) {
                    Ok(bytes) => tail = bytes,
                    Err(_) => issues.push(issue("trailer", "collection-failed")),
                }
                let decoded = decoder_info(path).map_err(|_| ()).and_then(|mut decoded| {
                    if container == "isobmff" {
                        let tags = raiw_container::isobmff::ai_metadata_tags(path).map_err(|_| ())?;
                        for (key, value) in tags {
                            decoded.insert(key, InfoValue::Str(value));
                        }
                    }
                    let pairs = exif_pairs(&decoded).map_err(|_| ());
                    Ok((decoded, pairs))
                });
                match decoded {
                    Ok((decoded, Ok(pairs))) => {
                        info = decoded;
                        exif = pairs;
                    }
                    Ok((decoded, Err(()))) => {
                        info = decoded;
                        issues.push(issue("decoder", "collection-failed"));
                    }
                    Err(()) => issues.push(issue("decoder", "collection-failed")),
                }
                match crate::c2pa_store::manifest_store_json_strict(path) {
                    Ok(found) => store = found,
                    Err(_) => issues.push(issue("c2pa", "collection-failed")),
                }
                if !issues.is_empty() {
                    status = "partial";
                }
            }
        }
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut record = Map::new();
    record.insert("schema_version".into(), json!(METADATA_RECORD_SCHEMA_VERSION));
    record.insert("record_type".into(), json!(METADATA_RECORD_TYPE));
    record.insert("status".into(), json!(status));
    record.insert("issues".into(), Value::Array(issues));
    record.insert("container".into(), json!(container));
    record.insert("name".into(), json!(name));
    record.insert("metadata_base64".into(), json!(B64.encode(&regions)));
    record.insert("tail_base64".into(), json!(B64.encode(&tail)));
    record.insert("pil".into(), Value::Object(pil_info(&info)));
    record.insert("exif".into(), Value::Object(exif));
    if let Some(store) = store {
        record.insert("c2pa_store".into(), Value::String(store));
    }
    Value::Object(record)
}

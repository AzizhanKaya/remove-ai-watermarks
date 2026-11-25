//! The AI-only metadata view (port of `_internal/extractor.py`).
//!
//! Pillow exposes AI generation fields only through PNG text chunks read before the
//! first `IDAT`; every other format's info keys are fixed names that never match, so the
//! reader below mirrors Pillow's PNG text decoding and nothing else.

use std::io::{self, Read};
use std::path::Path;

use flate2::read::ZlibDecoder;
use indexmap::IndexMap;
use raiw_container::png::chunks;
use raiw_container::{FileSource, ReadAt};
use raiw_core::constants::{AI_KEYWORDS, AI_METADATA_KEYS};

use crate::info::C2paInfo;
use crate::png::{extract_c2pa_chunk, has_c2pa_metadata};
use crate::text::casefold;

/// Pillow's `MAX_TEXT_CHUNK`: a decompressed text value above this is refused.
const MAX_TEXT_CHUNK: u64 = 1024 * 1024;

/// One AI metadata value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataValue {
    Text(String),
    C2pa(Box<C2paInfo>),
    Chunk(Vec<u8>),
}

/// Why the image could not be opened, as Pillow's `Image.open` would refuse it.
#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("cannot identify image file {0}")]
    Unidentified(String),
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// `_is_ai_field`: a known AI key or a key containing an AI keyword, case-insensitively.
pub fn is_ai_field(key: &str) -> bool {
    let folded = casefold(key);
    AI_METADATA_KEYS.iter().any(|ai_key| casefold(ai_key) == folded)
        || AI_KEYWORDS.iter().any(|token| folded.contains(token))
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

fn inflate_text(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    ZlibDecoder::new(data)
        .take(MAX_TEXT_CHUNK + 1)
        .read_to_end(&mut out)
        .ok()?;
    (out.len() as u64 <= MAX_TEXT_CHUNK).then_some(out)
}

fn split_nul(data: &[u8]) -> (&[u8], &[u8]) {
    match memchr::memchr(0, data) {
        Some(index) => (&data[..index], &data[index + 1..]),
        None => (data, &[]),
    }
}

/// One PNG text chunk as Pillow stores it in `info`, or `None` when Pillow skips it.
fn png_text_entry(kind: [u8; 4], payload: &[u8]) -> Option<(String, String)> {
    match &kind {
        b"tEXt" => {
            let (key, value) = split_nul(payload);
            (!key.is_empty()).then(|| (latin1(key), latin1(value)))
        }
        b"zTXt" => {
            let (key, value) = split_nul(payload);
            let method = value.first().copied().unwrap_or(0);
            if method != 0 || key.is_empty() {
                return None;
            }
            let text = value.get(1..).map_or_else(Vec::new, |compressed| {
                inflate_text(compressed).unwrap_or_default()
            });
            Some((latin1(key), latin1(&text)))
        }
        b"iTXt" => {
            let separator = memchr::memchr(0, payload)?;
            let (key, rest) = (&payload[..separator], &payload[separator + 1..]);
            if rest.len() < 2 {
                return None;
            }
            let (flag, method) = (rest[0], rest[1]);
            let mut parts = rest[2..].splitn(3, |&b| b == 0);
            let (language, translated, body) = (parts.next()?, parts.next()?, parts.next()?);
            let body = if flag == 0 {
                body.to_vec()
            } else if method == 0 {
                inflate_text(body)?
            } else {
                return None;
            };
            std::str::from_utf8(language).ok()?;
            std::str::from_utf8(translated).ok()?;
            Some((latin1(key), String::from_utf8(body).ok()?))
        }
        _ => None,
    }
}

/// Whether Pillow would identify the raster: its magic plus, for GIF and JPEG, a walkable structure.
fn pillow_opens(path: &Path, head: &[u8]) -> io::Result<bool> {
    if head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a") {
        return Ok(raiw_container::gif::strip_metadata(&std::fs::read(path)?, true).is_some());
    }
    if head.starts_with(b"\xff\xd8") {
        return Ok(raiw_container::jpeg::strip_segments(&std::fs::read(path)?, |_, _| false).is_some());
    }
    Ok(head.starts_with(raiw_core::constants::PNG_SIGNATURE)
        || (head.starts_with(b"RIFF") && head.get(8..12) == Some(b"WEBP"))
        || head.starts_with(b"II*\x00")
        || head.starts_with(b"MM\x00*")
        || head.starts_with(b"BM"))
}

/// The string-keyed text fields Pillow's `Image.open(path).info` carries.
fn pillow_text_info(path: &Path) -> Result<IndexMap<String, String>, ExtractError> {
    let source = FileSource::open(path)?;
    let head = source.read_vec(0, 16)?;
    if !pillow_opens(path, &head)? {
        return Err(ExtractError::Unidentified(path.display().to_string()));
    }
    let mut info = IndexMap::new();
    if !head.starts_with(raiw_core::constants::PNG_SIGNATURE) {
        return Ok(info);
    }
    let mut walk = chunks(&source)?;
    for chunk in walk.by_ref() {
        let chunk = chunk?;
        if &chunk.kind == b"IDAT" {
            return Ok(info);
        }
        if matches!(&chunk.kind, b"tEXt" | b"zTXt" | b"iTXt") {
            let payload = source.read_vec(chunk.payload_offset(), chunk.length)?;
            if let Some((key, value)) = png_text_entry(chunk.kind, &payload) {
                info.insert(key, value);
            }
        }
    }
    if walk.complete() {
        return Ok(info);
    }
    Err(ExtractError::Io(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "image file is truncated",
    )))
}

fn attach_c2pa(path: &Path, metadata: &mut IndexMap<String, MetadataValue>) {
    if !has_c2pa_metadata(path) {
        return;
    }
    if let Some(info) = crate::extract_c2pa_info(path) {
        metadata.insert("c2pa".into(), MetadataValue::C2pa(Box::new(info)));
    }
    if let Some(chunk) = extract_c2pa_chunk(path) {
        metadata.insert("c2pa_chunk".into(), MetadataValue::Chunk(chunk));
    }
}

/// `extract_ai_metadata`: AI-recognized fields plus the C2PA evidence and raw chunk.
pub fn extract_ai_metadata(path: &Path) -> Result<IndexMap<String, MetadataValue>, ExtractError> {
    let mut metadata: IndexMap<String, MetadataValue> = pillow_text_info(path)?
        .into_iter()
        .filter(|(key, _)| is_ai_field(key))
        .map(|(key, value)| (key, MetadataValue::Text(value)))
        .collect();
    attach_c2pa(path, &mut metadata);
    Ok(metadata)
}

/// `has_ai_metadata`: an AI field or a recognizable C2PA chunk.
pub fn has_ai_metadata(path: &Path) -> Result<bool, ExtractError> {
    Ok(pillow_text_info(path)?.keys().any(|key| is_ai_field(key)) || has_c2pa_metadata(path))
}

fn summary_text(text: &str) -> String {
    if text.chars().count() <= 100 {
        text.to_owned()
    } else {
        format!("{}...", text.chars().take(100).collect::<String>())
    }
}

/// `get_ai_metadata_summary`: the command-line report of the AI-only view.
pub fn ai_metadata_summary(path: &Path) -> Result<String, ExtractError> {
    let metadata = extract_ai_metadata(path)?;
    if metadata.is_empty() {
        return Ok("No AI metadata found.".into());
    }
    let mut lines = vec!["AI Image Metadata:".to_owned(), "-".repeat(40)];
    for (key, value) in &metadata {
        match value {
            MetadataValue::Chunk(_) => {}
            MetadataValue::C2pa(info) => {
                lines.push("C2PA Metadata:".into());
                lines.extend(
                    info.string_entries()
                        .into_iter()
                        .map(|(name, text)| format!("  {name}: {}", summary_text(&text))),
                );
            }
            MetadataValue::Text(text) => lines.push(format!("{key}: {}", summary_text(text))),
        }
    }
    Ok(lines.join("\n"))
}

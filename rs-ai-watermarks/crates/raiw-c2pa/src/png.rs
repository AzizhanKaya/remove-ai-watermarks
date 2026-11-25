//! PNG `caBX` C2PA chunks: bounded discovery, extraction, injection and the app11 text fallback.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use flate2::read::ZlibDecoder;
use raiw_container::png::{PngChunk, chunks};
use raiw_container::{FileSource, ReadAt};
use raiw_core::constants::{C2PA_CHUNK_TYPE, PNG_SIGNATURE};
use raiw_core::fs::suffix_lower;

use crate::fallback::{
    MAX_RAW_APP11_PROFILE_BYTES, info_from_raw_app11_profile, looks_like_c2pa, parse_c2pa_chunk,
};
use crate::info::C2paInfo;

const APP11_KEYWORD: &[u8] = b"raw profile type app11";
const TEXT_CHUNKS: [&[u8; 4]; 3] = [b"tEXt", b"zTXt", b"iTXt"];

/// Why a chunk injection was refused.
#[derive(Debug, thiserror::Error)]
pub enum InjectError {
    #[error("C2PA chunk injection is only supported for PNG files")]
    NotPng,
    #[error("Target is not a well-formed PNG file")]
    Malformed,
    #[error(transparent)]
    Io(#[from] io::Error),
}

fn payload<R: ReadAt + ?Sized>(source: &R, chunk: &PngChunk) -> io::Result<Vec<u8>> {
    source.read_vec(chunk.payload_offset(), chunk.length)
}

/// The first recognizable C2PA chunk (payload, serialized chunk with CRC).
fn find_c2pa_chunk(path: &Path) -> Option<(Vec<u8>, Vec<u8>)> {
    let source = FileSource::open(path).ok()?;
    for chunk in chunks(&source).ok()? {
        let chunk = chunk.ok()?;
        if chunk.kind == *C2PA_CHUNK_TYPE {
            let body = payload(&source, &chunk).ok()?;
            if looks_like_c2pa(&body) {
                let serialized = source.read_vec(chunk.offset, chunk.end() - chunk.offset).ok()?;
                return Some((body, serialized));
            }
        }
    }
    None
}

fn is_png_path(path: &Path) -> bool {
    suffix_lower(path) == ".png"
}

/// The first complete C2PA chunk of a `.png` file, header and CRC included.
pub fn extract_c2pa_chunk(path: &Path) -> Option<Vec<u8>> {
    if !is_png_path(path) {
        return None;
    }
    find_c2pa_chunk(path).map(|(_, serialized)| serialized)
}

/// Whether a validly bounded `.png` holds a recognizable C2PA chunk.
pub fn has_c2pa_metadata(path: &Path) -> bool {
    extract_c2pa_chunk(path).is_some()
}

/// Whether every chunk fits the file up to `IEND`, checked with seeks only.
pub fn is_well_formed_png(path: &Path) -> bool {
    let Ok(source) = FileSource::open(path) else {
        return false;
    };
    let Ok(mut walk) = chunks(&source) else {
        return false;
    };
    for chunk in walk.by_ref() {
        if chunk.is_err() {
            return false;
        }
    }
    walk.complete()
}

/// Inflate at most `limit + 1` bytes, as `zlib.decompressobj().decompress(data, limit + 1)`:
/// a truncated stream yields what it holds, a corrupt one fails.
fn inflate_bounded(data: &[u8], limit: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    ZlibDecoder::new(data)
        .take(limit as u64 + 1)
        .read_to_end(&mut out)
        .ok()?;
    Some(out)
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

/// The text body of a `Raw profile type app11` text chunk, inflated when compressed.
fn decode_app11_text(kind: [u8; 4], payload: &[u8]) -> Option<String> {
    let separator = memchr::memchr(0, payload)?;
    let (keyword, rest) = (&payload[..separator], &payload[separator + 1..]);
    if raiw_container::markers::py_strip(keyword).to_ascii_lowercase() != APP11_KEYWORD {
        return None;
    }
    match &kind {
        b"tEXt" => Some(latin1(rest)),
        b"zTXt" => {
            let (&method, compressed) = rest.split_first()?;
            if method != 0 {
                return None;
            }
            let raw = inflate_bounded(compressed, MAX_RAW_APP11_PROFILE_BYTES)?;
            (raw.len() <= MAX_RAW_APP11_PROFILE_BYTES).then(|| latin1(&raw))
        }
        _ => {
            if rest.len() < 4 {
                return None;
            }
            let (flag, method) = (rest[0], rest[1]);
            let mut parts = rest[2..].splitn(3, |&b| b == 0);
            let (_language, _translated, body) = (parts.next()?, parts.next()?, parts.next()?);
            let body = if flag == 1 {
                if method != 0 {
                    return None;
                }
                let raw = inflate_bounded(body, MAX_RAW_APP11_PROFILE_BYTES)?;
                if raw.len() > MAX_RAW_APP11_PROFILE_BYTES {
                    return None;
                }
                raw
            } else {
                body.to_vec()
            };
            Some(String::from_utf8_lossy(&body).into_owned())
        }
    }
}

/// Single-pass PNG C2PA reader covering native `caBX` and ImageMagick `app11` text chunks.
pub fn extract_c2pa_info_png(path: &Path) -> Option<C2paInfo> {
    let source = FileSource::open(path).ok()?;
    let mut app11 = None;
    let Ok(walk) = chunks(&source) else { return None };
    for chunk in walk {
        let Ok(chunk) = chunk else { return app11 };
        if chunk.kind == *C2PA_CHUNK_TYPE {
            let Ok(body) = payload(&source, &chunk) else {
                return app11;
            };
            if looks_like_c2pa(&body) {
                return Some(parse_c2pa_chunk(&body));
            }
        } else if app11.is_none()
            && TEXT_CHUNKS.contains(&&chunk.kind)
            && (24..=MAX_RAW_APP11_PROFILE_BYTES as u64).contains(&chunk.length)
        {
            let Ok(peek) = source.read_vec(chunk.payload_offset(), chunk.length.min(32)) else {
                return app11;
            };
            let keyword = peek.split(|&b| b == 0).next().unwrap_or_default();
            if peek.contains(&0)
                && raiw_container::markers::py_strip(keyword).to_ascii_lowercase() == APP11_KEYWORD
            {
                let Ok(body) = payload(&source, &chunk) else {
                    return app11;
                };
                app11 = decode_app11_text(chunk.kind, &body)
                    .filter(|text| !text.is_empty())
                    .and_then(|text| info_from_raw_app11_profile(&text));
            }
        }
    }
    app11
}

/// Replace any C2PA chunks of a PNG with `c2pa_chunk`, inserted before the first `IDAT`.
pub fn inject_c2pa_chunk(target: &Path, output: &Path, c2pa_chunk: &[u8]) -> Result<(), InjectError> {
    if !is_png_path(target) || !is_png_path(output) {
        return Err(InjectError::NotPng);
    }
    if !is_well_formed_png(target) {
        return Err(InjectError::Malformed);
    }
    let source = FileSource::open(target)?;
    let mut layout = Vec::new();
    for chunk in chunks(&source)? {
        layout.push(chunk?);
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = io::BufWriter::new(File::create(output)?);
    writer.write_all(PNG_SIGNATURE)?;
    let mut inserted = false;
    for chunk in &layout {
        if &chunk.kind == b"IDAT" && !inserted {
            writer.write_all(c2pa_chunk)?;
            inserted = true;
        }
        if chunk.kind != *C2PA_CHUNK_TYPE {
            let mut remaining = chunk.end() - chunk.offset;
            let mut offset = chunk.offset;
            let mut buffer = vec![0u8; remaining.min(1024 * 1024) as usize];
            while remaining > 0 {
                let want = remaining.min(buffer.len() as u64) as usize;
                let read = source.read_at(offset, &mut buffer[..want])?;
                if read == 0 {
                    return Err(io::Error::other("PNG changed while it was being copied").into());
                }
                writer.write_all(&buffer[..read])?;
                remaining -= read as u64;
                offset += read as u64;
            }
        }
    }
    writer.flush()?;
    Ok(())
}

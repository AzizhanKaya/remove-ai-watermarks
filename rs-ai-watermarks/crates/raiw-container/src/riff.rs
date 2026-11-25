//! RIFF walkers: AVI `LIST/INFO/AIGC` labels and late metadata chunks (WebP, WAV, AVI).

use std::io;
use std::path::Path;

use crate::hooks::MetadataHooks;
use crate::markers::{MAX_TC260_VALUE_BYTES, rstrip_bytes};
use crate::source::{FileSource, ReadAt};

/// RIFF forms whose metadata may follow the media data (C2PA uses the last top-level chunk).
pub const LATE_SCAN_FORMS: [&[u8; 4]; 3] = [b"WEBP", b"WAVE", b"AVI "];
/// Default cap on what a late-metadata scan may pull into memory.
pub const DEFAULT_LATE_BYTES: u64 = 4 * 1024 * 1024;

fn u32_le(bytes: &[u8]) -> u64 {
    u64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn info_payloads<R: ReadAt + ?Sized>(
    source: &R,
    start: u64,
    end: u64,
    hooks: &impl MetadataHooks,
) -> io::Result<Vec<Vec<u8>>> {
    let mut found = Vec::new();
    let mut position = start;
    while position + 8 <= end {
        let Some(header) = source.read_array::<8>(position)? else {
            break;
        };
        let size = u32_le(&header[4..]);
        let payload_end = position + 8 + size;
        if payload_end > end {
            break;
        }
        if &header[..4] == b"AIGC" && size <= MAX_TC260_VALUE_BYTES {
            let value = source.read_vec(position + 8, size)?;
            if value.len() as u64 == size && hooks.is_tc260_label(&value) {
                found.push(rstrip_bytes(&value, b"\x00 ").to_vec());
            }
        }
        position = payload_end + (size & 1);
    }
    Ok(found)
}

/// Validated TC260 values from an AVI `LIST/INFO/AIGC` chunk; read errors yield nothing.
pub fn tc260_aigc_payloads(path: &Path, hooks: &impl MetadataHooks) -> Vec<Vec<u8>> {
    let scan = || -> io::Result<Vec<Vec<u8>>> {
        let source = FileSource::open(path)?;
        let Some(header) = source.read_array::<12>(0)? else {
            return Ok(Vec::new());
        };
        if &header[..4] != b"RIFF" || &header[8..] != b"AVI " {
            return Ok(Vec::new());
        }
        let declared_end = (8 + u32_le(&header[4..8])).min(source.size());
        let mut found = Vec::new();
        let mut position = 12;
        while position + 8 <= declared_end {
            let Some(chunk) = source.read_array::<8>(position)? else {
                break;
            };
            let size = u32_le(&chunk[4..]);
            let payload_start = position + 8;
            let payload_end = payload_start + size;
            if payload_end > declared_end {
                break;
            }
            if &chunk[..4] == b"LIST" && size >= 4 && source.read_vec(payload_start, 4)? == b"INFO" {
                found.extend(info_payloads(&source, payload_start + 4, payload_end, hooks)?);
            }
            position = payload_end + (size & 1);
        }
        Ok(found)
    };
    scan().unwrap_or_default()
}

/// Payloads of metadata chunks (`EXIF`, `XMP `, `ICCP`, `C2PA`) starting at or after `window`.
///
/// Each chunk is clamped to what remains and the total to `max_total`; `Err` is the strict raise.
pub fn late_metadata<R: ReadAt + ?Sized>(source: &R, window: u64, max_total: u64) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    if source.read_vec(0, 4)? != b"RIFF" {
        return Ok(out);
    }
    let Some(declared) = source.read_array::<4>(4)? else {
        return Ok(out);
    };
    let container_end = source.size().min(8 + u32_le(&declared));
    let mut position = 12;
    while position + 8 <= container_end && (out.len() as u64) < max_total {
        let Some(header) = source.read_array::<8>(position)? else {
            break;
        };
        let start = position + 8;
        let length = u32_le(&header[4..]).min(container_end.saturating_sub(start));
        if raiw_core::constants::RIFF_METADATA_CHUNKS.contains(&&header[..4]) && start >= window {
            out.extend(source.read_vec(start, length.min(max_total - out.len() as u64))?);
        }
        position = start + length + (length & 1);
    }
    Ok(out)
}

/// `metadata._riff_late_metadata` over a file.
pub fn late_metadata_file(path: &Path, window: u64, max_total: u64) -> io::Result<Vec<u8>> {
    late_metadata(&FileSource::open(path)?, window, max_total)
}

/// Why a RIFF file is unreadable: its header declares more bytes than the file has.
pub fn declared_size_reason(head: &[u8], size: u64) -> Option<String> {
    if !head.starts_with(b"RIFF") || head.len() < 8 {
        return None;
    }
    let declared = 8 + u32_le(&head[4..8]);
    (declared > size).then(|| format!("the RIFF header declares {declared} bytes but the file has {size}"))
}

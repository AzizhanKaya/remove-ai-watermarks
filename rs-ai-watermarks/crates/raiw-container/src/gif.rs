//! GIF block walkers: metadata extension extraction and lossless extension stripping.

use std::io;
use std::path::Path;

use crate::source::{FileSource, ReadAt};

/// Application extensions that carry playback behaviour, not provenance.
pub const PLAYBACK_APP_IDS: [&[u8; 11]; 2] = [b"NETSCAPE2.0", b"ANIMEXTS1.0"];
pub const ICC_APP_ID: &[u8; 11] = b"ICCRGBG1012";
/// Default cap on what a late-metadata scan may pull into memory.
pub const DEFAULT_LATE_BYTES: u64 = 4 * 1024 * 1024;

fn is_gif(head: &[u8]) -> bool {
    head.len() >= 6 && (&head[..6] == b"GIF87a" || &head[..6] == b"GIF89a")
}

const fn color_table_len(packed: u8) -> usize {
    if packed & 0x80 != 0 {
        3 * (1 << ((packed & 0x07) + 1))
    } else {
        0
    }
}

/// Concatenated data sub-blocks from `start`, or `None` when the terminator is missing.
pub fn sub_block_payload(data: &[u8], start: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut cursor = start;
    while cursor < data.len() {
        let length = usize::from(data[cursor]);
        cursor += 1;
        if length == 0 {
            return Some(out);
        }
        out.extend_from_slice(data.get(cursor..cursor + length)?);
        cursor += length;
    }
    None
}

/// Unpack one comment (`0xFE`) or non-playback application (`0xFF`) extension.
pub fn extension_metadata(ext: &[u8]) -> Vec<u8> {
    if ext.len() < 3 || ext[0] != 0x21 {
        return Vec::new();
    }
    match ext[1] {
        0xFF if ext.len() >= 14 && ext[2] == 11 => {
            let app_id = &ext[3..14];
            if PLAYBACK_APP_IDS.iter().any(|id| id.as_slice() == app_id) || app_id == ICC_APP_ID {
                return Vec::new();
            }
            let mut out = app_id.to_vec();
            out.extend(sub_block_payload(ext, 14).unwrap_or_else(|| ext[14..].to_vec()));
            out
        }
        0xFE => sub_block_payload(ext, 2).unwrap_or_else(|| ext[2..].to_vec()),
        _ => Vec::new(),
    }
}

/// End of the sub-block chain from `start`, or `None` when it runs past the file.
fn skip_sub_blocks<R: ReadAt + ?Sized>(source: &R, start: u64) -> io::Result<Option<u64>> {
    let size = source.size();
    let mut cursor = start;
    while cursor < size {
        let Some([step]) = source.read_array::<1>(cursor)? else {
            return Ok(None);
        };
        cursor += 1 + u64::from(step);
        if step == 0 {
            return Ok((cursor <= size).then_some(cursor));
        }
    }
    Ok(None)
}

/// Comment and non-playback application extensions at or past `window`, unpacked.
pub fn late_metadata<R: ReadAt + ?Sized>(source: &R, window: u64, max_total: u64) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    let Some(header) = source.read_array::<13>(0)? else {
        return Ok(out);
    };
    if !is_gif(&header) {
        return Ok(out);
    }
    let size = source.size();
    let mut pos = 13 + color_table_len(header[10]) as u64;
    while pos < size && (out.len() as u64) < max_total {
        let intro = source.read_vec(pos, 2)?;
        let Some(&introducer) = intro.first() else { break };
        if introducer == 0x3B {
            break;
        }
        if introducer == 0x2C {
            let Some(descriptor) = source.read_array::<10>(pos)? else {
                break;
            };
            let table = color_table_len(descriptor[9]) as u64;
            let Some(end) = skip_sub_blocks(source, pos + 10 + table + 1)? else {
                break;
            };
            pos = end;
            continue;
        }
        if introducer != 0x21 || intro.len() < 2 {
            break;
        }
        let label = intro[1];
        let Some(end) = skip_sub_blocks(source, pos + 2)? else {
            break;
        };
        if pos >= window && matches!(label, 0xFE | 0xFF) {
            let length = (end - pos).min(max_total - out.len() as u64);
            out.extend(extension_metadata(&source.read_vec(pos, length)?));
        }
        pos = end;
    }
    Ok(out)
}

/// `metadata._gif_late_metadata` over a file.
pub fn late_metadata_file(path: &Path, window: u64, max_total: u64) -> io::Result<Vec<u8>> {
    late_metadata(&FileSource::open(path)?, window, max_total)
}

fn sub_blocks_end(data: &[u8], start: usize) -> Option<usize> {
    let mut cursor = start;
    loop {
        let length = usize::from(*data.get(cursor)?);
        cursor += 1 + length;
        if length == 0 {
            return Some(cursor);
        }
    }
}

/// Remove metadata extensions from a GIF, copying frames, palettes and LZW data verbatim.
///
/// Graphic-control and plain-text extensions stay; application extensions survive only for
/// looping and, when `keep_standard`, the ICC profile. `None` when the bytes are not walkable.
pub fn strip_metadata(data: &[u8], keep_standard: bool) -> Option<Vec<u8>> {
    if !is_gif(data) || data.len() < 13 {
        return None;
    }
    let mut pos = 13 + color_table_len(data[10]);
    let mut out = data[..pos.min(data.len())].to_vec();
    while pos < data.len() {
        match data[pos] {
            0x3B => {
                out.push(0x3B);
                return Some(out);
            }
            0x2C => {
                let table = color_table_len(*data.get(pos + 9)?);
                let end = sub_blocks_end(data, pos + 10 + table + 1)?;
                out.extend_from_slice(data.get(pos..end)?);
                pos = end;
            }
            0x21 => {
                let label = *data.get(pos + 1)?;
                let end = sub_blocks_end(data, pos + 2)?;
                let mut keep = matches!(label, 0xF9 | 0x01);
                if label == 0xFF && data[pos + 2] == 11 {
                    let app_id = data
                        .get(pos + 3..pos + 14)
                        .unwrap_or(&data[(pos + 3).min(data.len())..]);
                    keep = PLAYBACK_APP_IDS.iter().any(|id| id.as_slice() == app_id)
                        || (keep_standard && app_id == ICC_APP_ID);
                }
                if keep {
                    out.extend_from_slice(&data[pos..end]);
                }
                pos = end;
            }
            _ => return None,
        }
    }
    None
}

//! FLV walker for the TC260 label: an AMF0 `AIGC` string in the `onMetaData` script tag.

use std::io;
use std::path::Path;

use crate::hooks::MetadataHooks;
use crate::markers::MAX_TC260_VALUE_BYTES;
use crate::source::{FileSource, ReadAt};

const SCRIPT_TAG: u8 = 18;
const MAX_SCRIPT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_AMF_DEPTH: usize = 8;

fn be(bytes: &[u8]) -> usize {
    bytes.iter().fold(0usize, |acc, &b| (acc << 8) | usize::from(b))
}

/// An AMF0 string (2-byte length, or 4-byte for a long string) at `pos`.
fn amf0_string(data: &[u8], pos: usize, long: bool) -> Option<(&[u8], usize)> {
    let width = if long { 4 } else { 2 };
    let length = be(data.get(pos..pos + width)?);
    let start = pos + width;
    let end = start.checked_add(length)?;
    Some((data.get(start..end)?, end))
}

/// Skip a name/value property list ending in the `00 00 09` object-end marker.
fn skip_properties(data: &[u8], mut pos: usize, depth: usize) -> Option<usize> {
    while pos + 3 <= data.len() {
        let name_length = be(&data[pos..pos + 2]);
        pos += 2;
        if name_length == 0 && data[pos] == 9 {
            return Some(pos + 1);
        }
        pos += name_length;
        if pos > data.len() {
            return None;
        }
        pos = skip_amf0(data, pos, depth + 1)?;
    }
    None
}

/// Position after the AMF0 value at `pos`, or `None` for an unsupported or truncated value.
fn skip_amf0(data: &[u8], pos: usize, depth: usize) -> Option<usize> {
    if pos >= data.len() || depth > MAX_AMF_DEPTH {
        return None;
    }
    let value_type = data[pos];
    let pos = pos + 1;
    let fixed = |width: usize| (pos + width <= data.len()).then_some(pos + width);
    match value_type {
        0 => fixed(8),
        1 => fixed(1),
        2 => amf0_string(data, pos, false).map(|(_, end)| end),
        5 | 6 => Some(pos),
        7 => fixed(2),
        11 => fixed(10),
        12 => amf0_string(data, pos, true).map(|(_, end)| end),
        10 => {
            let count = be(data.get(pos..pos + 4)?);
            let mut cursor = pos + 4;
            for _ in 0..count {
                cursor = skip_amf0(data, cursor, depth + 1)?;
            }
            Some(cursor)
        }
        3 => skip_properties(data, pos, depth),
        8 => {
            if pos + 4 > data.len() {
                return None;
            }
            skip_properties(data, pos + 4, depth)
        }
        _ => None,
    }
}

/// TC260 values in one script tag payload whose first value is the string `onMetaData`.
fn script_payloads(data: &[u8], hooks: &impl MetadataHooks) -> Vec<Vec<u8>> {
    let mut found = Vec::new();
    let first = if data.first() == Some(&2) {
        amf0_string(data, 1, false)
    } else {
        None
    };
    let Some((b"onMetaData", mut pos)) = first else {
        return found;
    };
    match data.get(pos) {
        Some(8) => pos += 5,
        Some(3) => pos += 1,
        _ => return found,
    }
    while pos + 3 <= data.len() {
        let name_length = be(&data[pos..pos + 2]);
        pos += 2;
        if name_length == 0 && data[pos] == 9 {
            break;
        }
        let name_end = pos + name_length;
        if name_end > data.len() {
            break;
        }
        let name = &data[pos..name_end];
        pos = name_end;
        if name == b"AIGC"
            && let Some(&kind @ (2 | 12)) = data.get(pos)
        {
            let Some((value, end)) = amf0_string(data, pos + 1, kind == 12) else {
                break;
            };
            pos = end;
            if value.len() as u64 <= MAX_TC260_VALUE_BYTES && hooks.is_tc260_label(value) {
                found.push(value.to_vec());
            }
            continue;
        }
        let Some(next) = skip_amf0(data, pos, 0) else {
            break;
        };
        pos = next;
    }
    found
}

/// Validated TC260 values from FLV `script.onMetaData.AIGC`; read errors yield nothing.
pub fn tc260_aigc_payloads(path: &Path, hooks: &impl MetadataHooks) -> Vec<Vec<u8>> {
    let scan = || -> io::Result<Vec<Vec<u8>>> {
        let source = FileSource::open(path)?;
        let Some(header) = source.read_array::<9>(0)? else {
            return Ok(Vec::new());
        };
        if &header[..3] != b"FLV" {
            return Ok(Vec::new());
        }
        let file_size = source.size();
        let mut position = be(&header[5..9]) as u64 + 4;
        while position + 11 <= file_size {
            let Some(tag) = source.read_array::<11>(position)? else {
                break;
            };
            let data_size = be(&tag[1..4]) as u64;
            let payload_end = position + 11 + data_size;
            if payload_end + 4 > file_size {
                break;
            }
            if tag[0] & 0x1F == SCRIPT_TAG && data_size <= MAX_SCRIPT_BYTES {
                let payload = source.read_vec(position + 11, data_size)?;
                if payload.len() as u64 == data_size {
                    let found = script_payloads(&payload, hooks);
                    if !found.is_empty() {
                        return Ok(found);
                    }
                }
            }
            position = payload_end + 4;
        }
        Ok(Vec::new())
    };
    scan().unwrap_or_default()
}

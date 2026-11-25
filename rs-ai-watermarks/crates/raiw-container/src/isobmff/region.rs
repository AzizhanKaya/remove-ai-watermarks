//! Provenance payload collection that seeks past `mdat`, for detectors that scan raw bytes.

use std::io;
use std::path::Path;

use super::boxes::{FourCc, read_box_header};
use super::items::moov_items;
use super::labels::{XAI_REGION_SENTINEL_END, XAI_REGION_SENTINEL_START, is_xai_signature};
use crate::markers::contains;
use crate::source::{FileSource, ReadAt};

/// Default cap on the collected bytes.
pub const DEFAULT_REGION_BYTES: u64 = 4 * 1024 * 1024;

/// Top-level box types that may carry C2PA (`uuid`) or JUMBF (`jumb`).
pub const C2PA_BOX_TYPES: [&FourCc; 2] = [b"uuid", b"jumb"];

/// Whether an `emsg` prefix names a C2PA manifest.
pub fn has_c2pa_emsg_marker(prefix: &[u8]) -> bool {
    let lowered = prefix.to_ascii_lowercase();
    contains(&lowered, b"urn:c2pa:") || (contains(&lowered, b"jumb") && contains(&lowered, b"c2pa"))
}

/// Concatenated `uuid`/`jumb`/C2PA `emsg` payloads and bounded `moov` metadata values.
///
/// Walks only box headers, so `mdat` is never loaded; a malformed box ends the walk.
/// xAI signature items are wrapped in the `©cmt\0 ... \0` sentinel so the region scan can
/// recognize them. A non-ISOBMFF source yields nothing.
pub fn scan_region<R: ReadAt + ?Sized>(source: &R, max_total: u64) -> io::Result<Vec<u8>> {
    let mut collected = Vec::new();
    let sniff = source.read_vec(0, 8)?;
    if sniff.len() < 8 || &sniff[4..8] != b"ftyp" {
        return Ok(collected);
    }
    let file_size = source.size();
    let mut pos = 0u64;
    while pos + 8 <= file_size && (collected.len() as u64) < max_total {
        let Some(entry) = read_box_header(source, pos, file_size)? else {
            break;
        };
        let remaining = |collected: &Vec<u8>| max_total.saturating_sub(collected.len() as u64);
        let payload_len = entry.end - entry.payload;
        if C2PA_BOX_TYPES.contains(&&entry.kind) {
            let to_read = payload_len.min(remaining(&collected));
            if to_read > 0 {
                collected.extend(source.read_vec(entry.payload, to_read)?);
            }
        } else if &entry.kind == b"emsg" {
            if has_c2pa_emsg_marker(&source.read_vec(entry.payload, payload_len.min(256))?) {
                let to_read = payload_len.min(remaining(&collected));
                if to_read > 0 {
                    collected.extend(source.read_vec(entry.payload, to_read)?);
                }
            }
        } else if &entry.kind == b"moov" {
            for item in moov_items(source, &entry)? {
                let wrapped;
                let value: &[u8] = if is_xai_signature(&item.item_type, item.key_name.as_deref(), &item.value)
                {
                    wrapped = [XAI_REGION_SENTINEL_START, &item.value, XAI_REGION_SENTINEL_END].concat();
                    &wrapped
                } else {
                    &item.value
                };
                let to_read = (value.len() as u64).min(remaining(&collected)) as usize;
                collected.extend_from_slice(&value[..to_read]);
            }
        }
        pos = entry.end;
    }
    Ok(collected)
}

/// `isobmff.scan_c2pa_region(path, max_total=...)`; `Err` is the `strict=True` raise.
pub fn scan_c2pa_region(path: &Path, max_total: u64) -> io::Result<Vec<u8>> {
    scan_region(&FileSource::open(path)?, max_total)
}

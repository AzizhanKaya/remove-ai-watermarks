//! `scan_head`: the bounded byte window every marker scan reads, plus late metadata.

use std::io::{self, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};

use raiw_c2pa::raw_app11_jumbf_bytes;
use raiw_c2pa::text::casefold;
use raiw_container::ebml::{EBML_MAGIC, c2pa_attachment_payloads};
use raiw_container::isobmff::{DEFAULT_REGION_BYTES, is_isobmff, scan_c2pa_region};
use raiw_container::markers::contains;
use raiw_container::{gif, png, riff};
use raiw_core::constants::PNG_SIGNATURE;

use crate::cache::ContentKey;
use crate::pil::{self, InfoValue};

/// `_QUICK_SCAN_BYTES`: the window of the cheap marker checks.
pub const QUICK_SCAN_BYTES: usize = 512 * 1024;
/// `scan_head`'s default window.
pub const HEAD_BYTES: usize = 1024 * 1024;
const DECODED_TEXT_LIMIT: usize = 512 * 1024;

type HeadCache = Vec<((ContentKey, usize), Arc<Vec<u8>>)>;
static CACHE: Mutex<HeadCache> = Mutex::new(Vec::new());
const CACHE_ENTRIES: usize = 8;

/// The first `size` bytes of a file.
pub fn read_head(path: &Path, size: usize) -> io::Result<Vec<u8>> {
    let mut head = Vec::with_capacity(size.min(16 * 1024 * 1024));
    std::fs::File::open(path)?
        .take(size as u64)
        .read_to_end(&mut head)?;
    Ok(head)
}

/// `scan_head(path, size)`, memoized per file content and window.
pub fn scan_head(path: &Path, size: usize) -> io::Result<Arc<Vec<u8>>> {
    let Some(key) = ContentKey::of(path) else {
        return scan_head_uncached(path, size).map(Arc::new);
    };
    let key = (key, size);
    if let Ok(cache) = CACHE.lock()
        && let Some((_, hit)) = cache.iter().find(|(k, _)| *k == key)
    {
        return Ok(Arc::clone(hit));
    }
    let head = Arc::new(scan_head_uncached(path, size)?);
    if let Ok(mut cache) = CACHE.lock() {
        cache.retain(|(k, _)| *k != key);
        if cache.len() >= CACHE_ENTRIES {
            cache.remove(0);
        }
        cache.push((key, Arc::clone(&head)));
    }
    Ok(head)
}

fn scan_head_uncached(path: &Path, size: usize) -> io::Result<Vec<u8>> {
    let mut head = read_head(path, size)?;
    let full = head.len() == size;
    if is_isobmff(&head) {
        if let Ok(region) = scan_c2pa_region(path, DEFAULT_REGION_BYTES) {
            head.extend(region);
        }
    } else if head.starts_with(PNG_SIGNATURE) && full {
        head.extend(png::late_metadata_file(path, size as u64).unwrap_or_default());
    } else if head.starts_with(b"RIFF")
        && head.len() >= 12
        && riff::LATE_SCAN_FORMS.iter().any(|f| head[8..12] == f[..])
        && full
    {
        head.extend(
            riff::late_metadata_file(path, size as u64, riff::DEFAULT_LATE_BYTES).unwrap_or_default(),
        );
    } else if matches!(head.get(..6), Some(b"GIF87a" | b"GIF89a")) {
        head.extend(gif::late_metadata_file(path, 0, gif::DEFAULT_LATE_BYTES).unwrap_or_default());
    } else if head.starts_with(EBML_MAGIC) {
        for blob in c2pa_attachment_payloads(path) {
            head.extend(blob);
        }
    }
    let extra = decoder_visible_text(path, &head);
    head.extend(extra);
    Ok(head)
}

/// `_decoder_visible_text`: decoder `info` text the raw window does not already spell.
pub fn decoder_visible_text(path: &Path, head: &[u8]) -> Vec<u8> {
    let probe = pil::probe(path);
    let Some(image) = probe.image() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (key, value) in &image.info {
        if key == "icc_profile" {
            continue;
        }
        if let Some(text) = value.as_str()
            && casefold(crate::py::strip(key)) == "raw profile type app11"
            && let Some(jumbf) = raw_app11_jumbf_bytes(text)
        {
            out.push(0);
            out.extend(jumbf);
            continue;
        }
        let encoded: &[u8] = match value {
            InfoValue::Str(text) | InfoValue::IText { text, .. } => text.as_bytes(),
            InfoValue::Bytes(bytes) => bytes,
            _ => continue,
        };
        if encoded.len() > DECODED_TEXT_LIMIT || encoded.is_empty() || contains(head, encoded) {
            continue;
        }
        out.push(0);
        out.extend_from_slice(encoded);
    }
    out
}

/// `_read_file_tail`: the last `size` bytes.
pub fn read_file_tail(path: &Path, size: usize) -> io::Result<Vec<u8>> {
    use std::io::{Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    if len > size as u64 {
        file.seek(SeekFrom::Start(len - size as u64))?;
    }
    let mut out = Vec::new();
    file.read_to_end(&mut out)?;
    Ok(out)
}

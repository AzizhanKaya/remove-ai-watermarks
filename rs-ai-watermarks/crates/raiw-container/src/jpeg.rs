//! JPEG marker-segment walkers: APP payload extraction and lossless segment stripping.

use memchr::memmem;

use crate::markers::{SAMSUNG_EDITOR_MARKER, contains};

const SOI: &[u8] = b"\xff\xd8";
const EOI: &[u8] = b"\xff\xd9";

const fn is_standalone(marker: u8) -> bool {
    matches!(marker, 0xD0..=0xD7 | 0x01)
}

/// Complete XMP (APP1) and IPTC (APP13) payloads from a JPEG or concatenated APP regions.
pub fn app_payloads(data: &[u8]) -> Vec<(u8, &[u8])> {
    let mut payloads = Vec::new();
    let mut pos = if data.starts_with(SOI) { 2 } else { 0 };
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            break;
        }
        let marker = data[pos + 1];
        if matches!(marker, 0xDA | 0xD9) {
            break;
        }
        if is_standalone(marker) {
            pos += 2;
            continue;
        }
        let length = usize::from(u16::from_be_bytes([data[pos + 2], data[pos + 3]]));
        let end = pos + 2 + length;
        if length < 2 || end > data.len() {
            break;
        }
        if matches!(marker, 0xE1 | 0xED) {
            payloads.push((marker, &data[pos + 4..end]));
        }
        pos = end;
    }
    payloads
}

/// Drop a Samsung Galaxy AI editing trailer that follows the final EOI.
pub fn strip_samsung_trailer(scan_and_tail: &[u8]) -> &[u8] {
    if !contains(scan_and_tail, SAMSUNG_EDITOR_MARKER) {
        return scan_and_tail;
    }
    match memmem::rfind(scan_and_tail, EOI) {
        Some(eoi) if contains(&scan_and_tail[eoi..], SAMSUNG_EDITOR_MARKER) => &scan_and_tail[..eoi + 2],
        _ => scan_and_tail,
    }
}

/// Rewrite a JPEG without the marker segments `drop(marker, payload)` selects.
///
/// The entropy-coded scan is copied verbatim (minus a Samsung trailer); `None` for bytes
/// that are not a parseable JPEG, so the caller can fall back to a re-encode.
pub fn strip_segments(data: &[u8], mut drop: impl FnMut(u8, &[u8]) -> bool) -> Option<Vec<u8>> {
    if !data.starts_with(SOI) {
        return None;
    }
    let mut out = SOI.to_vec();
    let n = data.len();
    let mut i = 2;
    while i + 1 < n {
        if data[i] != 0xFF {
            return None;
        }
        let marker = data[i + 1];
        if matches!(marker, 0xDA | 0xD9) {
            out.extend_from_slice(strip_samsung_trailer(&data[i..]));
            break;
        }
        if is_standalone(marker) {
            out.extend_from_slice(&data[i..i + 2]);
            i += 2;
            continue;
        }
        if i + 4 > n {
            return None;
        }
        let length = usize::from(u16::from_be_bytes([data[i + 2], data[i + 3]]));
        let end = i + 2 + length;
        if length < 2 || end > n {
            return None;
        }
        if !drop(marker, &data[i + 4..end]) {
            out.extend_from_slice(&data[i..end]);
        }
        i = end;
    }
    Some(out)
}

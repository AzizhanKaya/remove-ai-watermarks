//! Byte markers the container walkers match (defined in Python's `metadata.py`).

use memchr::memmem;

/// C2PA UUID of ISOBMFF `uuid` boxes and H.264/HEVC SEI messages.
pub const C2PA_UUID: [u8; 16] = [
    0xd8, 0xfe, 0xc3, 0xd6, 0x1b, 0x0e, 0x48, 0x3c, 0x92, 0x97, 0x58, 0x28, 0x87, 0x7e, 0xc4, 0x81,
];

/// Kling's `user_data_unregistered` SEI payload: a UUID followed by `kling-ai`.
pub const KLING_SEI_TAG: &[u8] = b"\x91\xca\x60\x61\x4a\xee\x38\x54\x86\x14\x2d\x5f\x73\xf4\xae\x2ekling-ai";

/// China TC260 AIGC XMP namespace markers.
pub const AIGC_MARKERS: &[&[u8]] = &[b"tc260.org.cn/ns/AIGC", b"TC260:AIGC"];

/// IPTC `digitalSourceType` values that flag AI provenance.
pub const IPTC_AI_MARKERS: &[&[u8]] = &[
    b"trainedAlgorithmicMedia",
    b"compositeSynthetic",
    b"compositeWithTrainedAlgorithmicMedia",
];

/// IPTC Photo Metadata 2025.1 AI-disclosure XMP properties.
pub const IPTC_AI_FIELD_MARKERS: &[&[u8]] = &[
    b"AISystemUsed",
    b"AISystemVersionUsed",
    b"AIPromptInformation",
    b"AIPromptWriterName",
];

/// Every AI-label marker an XMP payload is matched against (`isobmff._AI_LABEL_MARKERS`).
pub const AI_LABEL_MARKERS: &[&[u8]] = &[
    b"tc260.org.cn/ns/AIGC",
    b"TC260:AIGC",
    b"trainedAlgorithmicMedia",
    b"compositeSynthetic",
    b"compositeWithTrainedAlgorithmicMedia",
    b"AISystemUsed",
    b"AISystemVersionUsed",
    b"AIPromptInformation",
    b"AIPromptWriterName",
];

/// Lower-cased metadata keys that carry local generation parameters.
pub const LOCAL_GENERATION_METADATA_KEYS: &[&str] = &[
    "parameters",
    "prompt",
    "negative_prompt",
    "workflow",
    "comfyui",
    "invokeai_metadata",
    "dream",
];

/// Lower-cased Topaz Video AI metadata keys.
pub const TOPAZ_VIDEO_METADATA_KEYS: &[&str] = &[
    "topazcloud",
    "topazcloud_provider",
    "topazcloud_application",
    "topazcloud_timestamp",
    "videoai",
    "videoai_provider",
    "videoai_application",
    "videoai_timestamp",
];

pub const TOPAZ_AI_DESCRIPTION: &[u8] = b"AI-enhanced using Topaz Labs";

/// Upper bound on a native TC260 label value read from any container.
pub const MAX_TC260_VALUE_BYTES: u64 = 1024 * 1024;

/// Samsung Galaxy AI editing trailer marker appended after a JPEG EOI.
pub const SAMSUNG_EDITOR_MARKER: &[u8] = b"PhotoEditor_Re_Edit_Data";

/// `needle in haystack` for bytes.
pub fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    memmem::find(haystack, needle).is_some()
}

/// Whether any marker occurs in `haystack`.
pub fn contains_any(haystack: &[u8], markers: &[&[u8]]) -> bool {
    markers.iter().any(|marker| contains(haystack, marker))
}

/// Python's `bytes.isspace()` per byte, which unlike Rust's also counts `\x0b`.
pub const fn is_py_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

/// Python's `bytes.strip()`.
pub fn py_strip(data: &[u8]) -> &[u8] {
    let start = data.iter().position(|&b| !is_py_space(b)).unwrap_or(data.len());
    let end = data
        .iter()
        .rposition(|&b| !is_py_space(b))
        .map_or(start, |last| last + 1);
    &data[start..end]
}

/// Python's `bytes.rstrip(chars)`.
pub fn rstrip_bytes<'a>(data: &'a [u8], chars: &[u8]) -> &'a [u8] {
    let end = data
        .iter()
        .rposition(|b| !chars.contains(b))
        .map_or(0, |last| last + 1);
    &data[..end]
}

/// Python's `bytes.strip(chars)`.
pub fn strip_bytes<'a>(data: &'a [u8], chars: &[u8]) -> &'a [u8] {
    let trimmed = rstrip_bytes(data, chars);
    let start = trimmed
        .iter()
        .position(|b| !chars.contains(b))
        .unwrap_or(trimmed.len());
    &trimmed[start..]
}

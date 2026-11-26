//! Lossless JPEG strip: drop AI-bearing APP segments, keep the scan, scrub EXIF via piexif.

use std::path::Path;

use raiw_container::jpeg::strip_segments;
use raiw_container::markers::{AIGC_MARKERS, IPTC_AI_FIELD_MARKERS, IPTC_AI_MARKERS, contains, contains_any};

use super::{StripError, write_output};
use crate::exif;
use crate::exif_ai::scrub_ai_exif;
use crate::rules::app::{any, app_evidence_bytes};
use crate::rules::generator::names_ai_generator;
use crate::rules::markers::c2pa_marker_in;
use crate::rules::tc260::is_aigc_value_bytes;
use crate::rules::xai::xai_signature_in_metadata;

/// `_jpeg_app_carries_ai`: an APPn segment to drop wholesale (EXIF is scrubbed instead).
pub fn jpeg_app_carries_ai(marker: u8, payload: &[u8]) -> bool {
    if !(0xE0..=0xEF).contains(&marker) {
        return false;
    }
    if marker == 0xEB
        && (c2pa_marker_in(payload)
            || contains(&payload[..payload.len().min(256)].to_ascii_lowercase(), b"jumb"))
    {
        return true;
    }
    if marker == 0xE1
        && payload.starts_with(b"http://ns.adobe.com/xap/")
        && (c2pa_marker_in(payload)
            || contains_any(payload, AIGC_MARKERS)
            || contains_any(payload, IPTC_AI_MARKERS)
            || contains_any(payload, IPTC_AI_FIELD_MARKERS)
            || xai_signature_in_metadata(payload)
            || names_ai_generator(payload))
    {
        return true;
    }
    if marker == 0xED
        && (contains_any(payload, IPTC_AI_MARKERS)
            || contains_any(payload, IPTC_AI_FIELD_MARKERS)
            || xai_signature_in_metadata(payload))
    {
        return true;
    }
    if !(marker == 0xE1 && payload.starts_with(b"Exif")) {
        return is_aigc_value_bytes(payload) || any(app_evidence_bytes(payload));
    }
    false
}

/// `_strip_jpeg_metadata_lossless`; `Ok(false)` when the bytes are not a walkable JPEG.
pub fn strip_jpeg_lossless(source: &Path, output: &Path) -> Result<bool, StripError> {
    let data = std::fs::read(source)?;
    let Some(walked) = strip_segments(&data, jpeg_app_carries_ai) else {
        return Ok(false);
    };
    write_output(output, &walked)?;
    if let Ok(mut dict) = exif::load_jpeg_file(output)
        && !scrub_ai_exif(&mut dict).is_empty()
        && let Ok(dumped) = exif::dump(&dict)
        && let Ok(current) = std::fs::read(output)
        && let Ok(inserted) = exif::insert_jpeg(&dumped, &current)
    {
        std::fs::write(output, inserted)?;
    }
    Ok(true)
}

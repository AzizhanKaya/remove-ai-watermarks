//! Byte-scan provenance markers: C2PA, IPTC, TC260 XMP, Samsung, cloud manifests.

use std::sync::LazyLock;

use raiw_container::markers::{
    AIGC_MARKERS, C2PA_UUID, IPTC_AI_FIELD_MARKERS, IPTC_AI_MARKERS, SAMSUNG_EDITOR_MARKER, contains,
    contains_any,
};
use regex::Regex;
use regex::bytes::Regex as BytesRegex;

use super::app::{any, app_evidence_bytes};
use super::generator::names_ai_generator;
use super::keys::{has_generator_token, is_ai_value};
use super::tc260::is_aigc_value_bytes;
use super::xai::xai_signature_in_metadata;
use crate::py::{strip_chars, utf8_ignore, utf8_replace};

fn bytes_re(pattern: &str) -> BytesRegex {
    BytesRegex::new(pattern).unwrap_or_else(|err| unreachable!("static regex: {err}"))
}

static C2PA_CI: LazyLock<BytesRegex> = LazyLock::new(|| bytes_re(r"(?i-u)c2pa"));
static SAMSUNG_GENAI: LazyLock<BytesRegex> = LazyLock::new(|| bytes_re(r#"(?-u)genAIType"\s*:\s*(-?\d+)"#));
static AI_SYSTEM_USED: LazyLock<BytesRegex> =
    LazyLock::new(|| bytes_re(r#"(?-u)AISystemUsed[=:\s]*["'>]\s*([^<"']{1,120})"#));
static CLI_FLAG_GARBAGE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^-[A-Za-z0-9_:-]+=").unwrap_or_else(|err| unreachable!("{err}")));
pub static XMP_IPTC_SOURCE_TYPE: LazyLock<BytesRegex> = LazyLock::new(|| {
    bytes_re(concat!(
        r#"(?i-u)DigitalSourceType(?:\s*=\s*["']|\s*>)[^<"']*?"#,
        r"(compositeWithTrainedAlgorithmicMedia|trainedAlgorithmicMedia|compositeSynthetic)"
    ))
});

const C2PA_MANIFEST_REPOSITORIES: &[(&[u8], &str)] =
    &[(b"cai-manifests.adobe.com", "Adobe Content Authenticity")];

/// `c2pa_marker_in`: a JUMBF-wrapped manifest or the ISOBMFF C2PA UUID, not a bare `c2pa`.
pub fn c2pa_marker_in(data: &[u8]) -> bool {
    contains(data, &C2PA_UUID) || (contains(data, b"jumb") && C2PA_CI.is_match(data))
}

/// `samsung_genai_in`: the non-zero `genAIType` inside a Samsung editor record.
pub fn samsung_genai_in(data: &[u8]) -> Option<i64> {
    if !contains(data, SAMSUNG_EDITOR_MARKER) {
        return None;
    }
    let caps = SAMSUNG_GENAI.captures(data)?;
    let digits = std::str::from_utf8(&caps[1]).ok()?;
    let value: i64 = digits.parse().ok()?;
    (value != 0).then_some(value)
}

/// `iptc_ai_system_in`: the `AISystemUsed` value, or `"fields present"`.
pub fn iptc_ai_system_in(data: &[u8]) -> Option<String> {
    if !contains_any(data, IPTC_AI_FIELD_MARKERS) {
        return None;
    }
    if let Some(caps) = AI_SYSTEM_USED.captures(data) {
        let value = crate::py::strip(&utf8_replace(&caps[1])).to_owned();
        if !value.is_empty() && !CLI_FLAG_GARBAGE.is_match(&value) {
            return Some(value);
        }
    }
    Some("fields present".into())
}

/// `c2pa_cloud_manifest_in`: an XMP `dcterms:provenance` pointer to a known repository.
pub fn c2pa_cloud_manifest_in(data: &[u8]) -> Option<&'static str> {
    if !contains(data, b"dcterms:provenance") {
        return None;
    }
    C2PA_MANIFEST_REPOSITORIES
        .iter()
        .find(|(host, _)| contains(data, host))
        .map(|(_, vendor)| *vendor)
}

/// Any IPTC `digitalSourceType` / 2025.1 field, TC260 XMP or C2PA marker.
pub fn has_label_markers(data: &[u8]) -> bool {
    c2pa_marker_in(data)
        || contains_any(data, AIGC_MARKERS)
        || contains_any(data, IPTC_AI_MARKERS)
        || contains_any(data, IPTC_AI_FIELD_MARKERS)
}

/// `payload_carries_ai_metadata`.
pub fn payload_carries_ai_metadata(raw: &[u8]) -> bool {
    if has_label_markers(raw)
        || xai_signature_in_metadata(raw)
        || samsung_genai_in(raw).is_some()
        || names_ai_generator(raw)
        || is_aigc_value_bytes(raw)
        || any(app_evidence_bytes(raw))
    {
        return true;
    }
    let decoded = utf8_ignore(raw);
    let text = strip_chars(&decoded, &['\0', ' ']);
    if text.is_empty() {
        return false;
    }
    is_ai_value(text) || has_generator_token(&text.to_lowercase())
}

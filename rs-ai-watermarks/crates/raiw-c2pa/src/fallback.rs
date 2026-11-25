//! The hand-rolled JUMBF/CBOR scan for manifests the validating reader cannot open.
//!
//! It never validates: every dimension stays `unknown` and the evidence comes from
//! registry tokens in the raw bytes.

use std::sync::LazyLock;

use memchr::memmem;
use regex::bytes::Regex;

use crate::attribution::{ordered_matches, synthid_evidence_vendors_in, synthid_verdict};
use crate::info::{AiSourceKind, C2paInfo, Validation};
use crate::registry::{C2PA_ACTIONS, C2PA_AI_TOOLS, c2pa_issuers};
use crate::soft_binding::{self, SoftBindingKind};
use crate::store::{AI_DISCLOSURE_LEVELS, disclosure_kind};
use crate::text::is_printable;

static TIMESTAMP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?-u)[0-9]{14}Z").expect("static pattern"));

/// Upper bound on an ImageMagick `Raw profile type app11` text.
pub const MAX_RAW_APP11_PROFILE_BYTES: usize = 8 * 1024 * 1024;

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    memmem::find(haystack, needle).is_some()
}

/// Whether a payload is recognizably a C2PA/JUMBF store.
pub fn looks_like_c2pa(payload: &[u8]) -> bool {
    let lowered = payload.to_ascii_lowercase();
    raiw_core::constants::C2PA_SIGNATURES
        .iter()
        .any(|signature| contains(payload, signature))
        || contains(&lowered, b"c2pa")
        || contains(&lowered, b"jumb")
}

/// Decode the definite-length CBOR text string right after the first occurrence of `key`.
pub fn cbor_text_after(payload: &[u8], key: &[u8]) -> Option<String> {
    let mut cursor = memmem::find(payload, key)? + key.len();
    let initial = *payload.get(cursor)?;
    let length = match initial {
        0x60..=0x77 => {
            cursor += 1;
            usize::from(initial & 0x1F)
        }
        0x78 if cursor + 1 < payload.len() => {
            let length = usize::from(payload[cursor + 1]);
            cursor += 2;
            length
        }
        0x79 if cursor + 2 < payload.len() => {
            let length = usize::from(u16::from_be_bytes([payload[cursor + 1], payload[cursor + 2]]));
            cursor += 3;
            length
        }
        _ => return None,
    };
    let raw = payload.get(cursor..cursor + length)?;
    Some(match std::str::from_utf8(raw) {
        Ok(text) => text.to_owned(),
        Err(_) => raw.iter().map(|&b| char::from(b)).collect(),
    })
}

/// Registry evidence from raw manifest bytes; returns whether an AI source type was found.
fn populate_registry_fields(buffer: &[u8], info: &mut C2paInfo) -> bool {
    let issuers = ordered_matches(buffer, c2pa_issuers());
    let tools = ordered_matches(buffer, C2PA_AI_TOOLS.iter().copied());
    let actions = ordered_matches(buffer, C2PA_ACTIONS.iter().copied());
    info.issuer = (!issuers.is_empty()).then(|| issuers.join(", "));
    info.ai_tool = (!tools.is_empty()).then(|| tools.join(", "));
    info.actions = (!actions.is_empty()).then(|| actions.join(", "));

    let mut ai_source = true;
    if contains(buffer, b"trainedAlgorithmicMedia") {
        info.source_type = Some("trainedAlgorithmicMedia (AI-generated)".into());
        info.ai_source_kind = Some(AiSourceKind::Generated);
    } else if contains(buffer, b"compositeWithTrainedAlgorithmicMedia") {
        info.source_type = Some("compositeWithTrainedAlgorithmicMedia (AI-enhanced)".into());
        info.ai_source_kind = Some(AiSourceKind::Enhanced);
    } else if contains(buffer, b"c2pa.ai-disclosure") {
        let levels: Vec<String> = AI_DISCLOSURE_LEVELS
            .iter()
            .filter(|level| contains(buffer, level.as_bytes()))
            .map(|l| (*l).to_owned())
            .collect();
        let kind = disclosure_kind(&levels);
        info.source_type = Some(format!("c2pa.ai-disclosure ({})", kind.as_str()));
        info.ai_source_kind = Some(kind);
        info.ai_disclosure = Some("present".into());
    } else {
        ai_source = false;
        if contains(buffer, b"algorithmicMedia") {
            info.source_type = Some("algorithmicMedia".into());
        }
    }

    info.watermarked = contains(buffer, b"c2pa.watermarked");
    let entries = soft_binding::registry_entries_in(buffer);
    let labels = soft_binding::labels(entries.iter().copied(), None);
    let synthid = if entries
        .iter()
        .any(|entry| entry.kind == SoftBindingKind::Watermark)
    {
        Vec::new()
    } else {
        synthid_evidence_vendors_in(buffer, Some(info.watermarked), None)
    };
    if ai_source && !synthid.is_empty() {
        info.synthid_watermark = Some(synthid_verdict(&synthid.join(", ")));
        info.synthid_vendors = synthid;
    }
    if !labels.is_empty() {
        info.soft_binding = Some(labels.join(", "));
        info.soft_binding_vendors = labels;
        let algorithms: Vec<&str> = entries.iter().map(|entry| entry.algorithm).collect();
        info.soft_binding_algorithm = Some(algorithms.join(", "));
    }
    ai_source
}

/// `parse_c2pa_chunk`: unvalidated evidence from a raw JUMBF/CBOR manifest payload.
pub fn parse_c2pa_chunk(payload: &[u8]) -> C2paInfo {
    let mut info = C2paInfo::new(payload.len(), Validation::fallback());
    populate_registry_fields(payload, &mut info);
    if let Some(value) = cbor_text_after(payload, b"name").filter(|v| !v.is_empty() && is_printable(v)) {
        info.claim_generator = Some(value);
    }
    if let Some(value) = cbor_text_after(payload, b"specVersion").filter(|v| !v.is_empty() && is_printable(v))
    {
        info.c2pa_spec = Some(value);
    }
    let timestamps: Vec<String> = TIMESTAMP
        .find_iter(payload)
        .map(|m| String::from_utf8_lossy(m.as_bytes()).into_owned())
        .collect();
    if let Some(first) = timestamps.first() {
        info.timestamp = Some(first.clone());
        if timestamps.len() > 1 {
            info.timestamps = timestamps.into_iter().take(3).collect();
        }
    }
    info
}

/// Python's `bytes.fromhex`: hex pairs, ASCII whitespace allowed between pairs.
fn from_hex(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 2);
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if matches!(ch, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c') {
            continue;
        }
        let high = ch.to_digit(16)?;
        let low = chars.next()?.to_digit(16)?;
        out.push((high * 16 + low) as u8);
    }
    Some(out)
}

/// Raw C2PA JUMBF bytes from an ImageMagick `Raw profile type app11` string.
pub fn raw_app11_jumbf_bytes(profile_text: &str) -> Option<Vec<u8>> {
    if profile_text.is_empty() || profile_text.chars().count() > MAX_RAW_APP11_PROFILE_BYTES {
        return None;
    }
    let is_space = |c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c');
    let rest = profile_text.trim_start_matches(is_space).strip_prefix("app11")?;
    let digits_start = rest.trim_start_matches(is_space);
    if digits_start.len() == rest.len() {
        return None;
    }
    let after_digits = digits_start.trim_start_matches(|c: char| c.is_ascii_digit());
    if after_digits.len() == digits_start.len() {
        return None;
    }
    let hex = after_digits.trim_start_matches(is_space);
    if hex.len() == after_digits.len() {
        return None;
    }
    let raw = from_hex(hex)?;
    if raw.len() < 8 {
        return None;
    }
    let jumbf = match memmem::find(&raw, b"jumb") {
        Some(index) if index >= 4 => raw[index - 4..].to_vec(),
        _ => raw,
    };
    looks_like_c2pa(&jumbf).then_some(jumbf)
}

/// `c2pa_info_from_raw_app11_profile`.
pub fn info_from_raw_app11_profile(profile_text: &str) -> Option<C2paInfo> {
    raw_app11_jumbf_bytes(profile_text).map(|jumbf| parse_c2pa_chunk(&jumbf))
}

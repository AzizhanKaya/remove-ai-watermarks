//! The byte-scan C2PA helpers `text.py` borrows from `_internal/c2pa.py`.
//!
//! `parse_c2pa_chunk`, `claim_generator_platform` and `c2pa_vendor_platform` over a
//! snapshot of the registries in `_internal/constants.py` (`data/c2pa_registry.json`).

use std::collections::BTreeSet;
use std::sync::LazyLock;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::unicode;

const REGISTRY_JSON: &str = include_str!("../data/c2pa_registry.json");

#[derive(Debug, Deserialize)]
struct SoftBindingEntry {
    algorithm: String,
    kind: String,
    display_label: String,
}

#[derive(Debug, Deserialize)]
struct Vendor {
    issuer: String,
    org: String,
    platform: Option<String>,
    needle: Option<String>,
    synthid: bool,
    synthid_requires_watermark_action: bool,
}

#[derive(Debug, Deserialize)]
struct Registry {
    issuers: Vec<(String, String)>,
    ai_tools: Vec<(String, String)>,
    actions: Vec<(String, String)>,
    soft_bindings: Vec<(String, String)>,
    #[serde(rename = "soft_binding_registry")]
    soft_binding_algorithms: Vec<SoftBindingEntry>,
    vendor_soft_binding_prefixes: Vec<SoftBindingEntry>,
    vendors: Vec<Vendor>,
    synthid_edit_signers: Vec<String>,
    claim_generator_platforms: Vec<(String, String)>,
    ai_disclosure_levels: Vec<String>,
    ai_disclosure_enhanced_levels: Vec<String>,
}

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    serde_json::from_str(REGISTRY_JSON).expect("bundled C2PA registry snapshot is valid JSON")
});

static VENDOR_NEEDLES: LazyLock<Vec<(String, &'static str)>> = LazyLock::new(|| {
    REGISTRY
        .vendors
        .iter()
        .filter_map(
            |vendor| match (vendor.platform.as_deref(), vendor.needle.as_deref()) {
                (Some(platform), Some(needle)) if !platform.is_empty() && !needle.is_empty() => {
                    Some((unicode::casefold(needle), platform))
                }
                _ => None,
            },
        )
        .collect()
});

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|window| window == needle)
}

/// Every start offset of `needle` in `haystack`, overlapping occurrences included.
fn occurrences<'a>(haystack: &'a [u8], needle: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
    haystack
        .windows(needle.len().max(1))
        .enumerate()
        .filter(move |(_, window)| *window == needle)
        .map(|(at, _)| at)
}

fn is_soft_binding_boundary(byte: Option<&u8>) -> bool {
    !byte.is_some_and(|&byte| byte.is_ascii_alphanumeric() || byte == b'.' || byte == b'-')
}

fn exact_algorithm_in(buffer: &[u8], algorithm: &[u8]) -> bool {
    occurrences(buffer, algorithm).any(|at| {
        (at == 0 || is_soft_binding_boundary(buffer.get(at - 1)))
            && is_soft_binding_boundary(buffer.get(at + algorithm.len()))
    })
}

/// `prefix(?:\.[A-Za-z0-9-]+)*(?![A-Za-z0-9.-])` behind a `(?<![A-Za-z0-9.-])` boundary.
fn prefixed_algorithm_in(buffer: &[u8], prefix: &[u8]) -> bool {
    let label = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'-';
    occurrences(buffer, prefix).any(|at| {
        if at > 0 && !is_soft_binding_boundary(buffer.get(at - 1)) {
            return false;
        }
        let mut cursor = at + prefix.len();
        while buffer.get(cursor) == Some(&b'.') && buffer.get(cursor + 1).is_some_and(|&byte| label(byte)) {
            cursor += 1;
            while buffer.get(cursor).is_some_and(|&byte| label(byte)) {
                cursor += 1;
            }
        }
        is_soft_binding_boundary(buffer.get(cursor))
    })
}

fn soft_binding_entries_in(buffer: &[u8]) -> Vec<&'static SoftBindingEntry> {
    let registry = &*REGISTRY;
    let official = registry
        .soft_binding_algorithms
        .iter()
        .filter(|entry| exact_algorithm_in(buffer, entry.algorithm.as_bytes()));
    let supplemental = registry
        .vendor_soft_binding_prefixes
        .iter()
        .filter(|entry| prefixed_algorithm_in(buffer, entry.algorithm.as_bytes()));
    official.chain(supplemental).collect()
}

fn soft_binding_labels(entries: &[&SoftBindingEntry]) -> Vec<String> {
    let labels: BTreeSet<String> = entries
        .iter()
        .map(|entry| {
            REGISTRY
                .soft_bindings
                .iter()
                .find(|(algorithm, _)| *algorithm == entry.algorithm)
                .map_or_else(|| entry.display_label.clone(), |(_, label)| label.clone())
        })
        .collect();
    labels.into_iter().collect()
}

fn synthid_evidence_vendors_in(buffer: &[u8], has_watermark_action: bool) -> Vec<String> {
    let registry = &*REGISTRY;
    let records_synthid_action = contains(buffer, b"SynthID");
    let identity_suffices = records_synthid_action
        || !registry
            .synthid_edit_signers
            .iter()
            .any(|signer| contains(buffer, signer.as_bytes()));
    let vendors: BTreeSet<String> = registry
        .vendors
        .iter()
        .filter(|vendor| {
            vendor.synthid
                && contains(buffer, vendor.issuer.as_bytes())
                && ((has_watermark_action && vendor.synthid_requires_watermark_action)
                    || (identity_suffices && !vendor.synthid_requires_watermark_action))
        })
        .map(|vendor| vendor.org.clone())
        .collect();
    vendors.into_iter().collect()
}

fn ordered_matches(buffer: &[u8], registry: &[(String, String)]) -> Vec<String> {
    let mut labels: Vec<String> = Vec::new();
    for (token, label) in registry {
        if contains(buffer, token.as_bytes()) && !labels.contains(label) {
            labels.push(label.clone());
        }
    }
    labels
}

fn disclosure_kind(oversight: &[&String]) -> &'static str {
    let enhanced = &REGISTRY.ai_disclosure_enhanced_levels;
    if !oversight.is_empty() && oversight.iter().all(|level| enhanced.contains(level)) {
        "enhanced"
    } else {
        "generated"
    }
}

fn populate_registry_fields(buffer: &[u8], info: &mut Map<String, Value>) {
    let registry = &*REGISTRY;
    let issuers = ordered_matches(buffer, &registry.issuers);
    let tools = ordered_matches(buffer, &registry.ai_tools);
    let actions = ordered_matches(buffer, &registry.actions);
    if !issuers.is_empty() {
        info.insert("issuer".into(), json!(issuers.join(", ")));
    }
    if !tools.is_empty() {
        info.insert("ai_tool".into(), json!(tools.join(", ")));
    }
    if !actions.is_empty() {
        info.insert("actions".into(), json!(actions.join(", ")));
    }

    let mut ai_source = false;
    if contains(buffer, b"trainedAlgorithmicMedia") {
        info.insert(
            "source_type".into(),
            json!("trainedAlgorithmicMedia (AI-generated)"),
        );
        info.insert("ai_source_kind".into(), json!("generated"));
        ai_source = true;
    } else if contains(buffer, b"compositeWithTrainedAlgorithmicMedia") {
        info.insert(
            "source_type".into(),
            json!("compositeWithTrainedAlgorithmicMedia (AI-enhanced)"),
        );
        info.insert("ai_source_kind".into(), json!("enhanced"));
        ai_source = true;
    } else if contains(buffer, b"c2pa.ai-disclosure") {
        let levels: Vec<&String> = registry
            .ai_disclosure_levels
            .iter()
            .filter(|level| contains(buffer, level.as_bytes()))
            .collect();
        let kind = disclosure_kind(&levels);
        info.insert(
            "source_type".into(),
            json!(format!("c2pa.ai-disclosure ({kind})")),
        );
        info.insert("ai_source_kind".into(), json!(kind));
        info.insert("ai_disclosure".into(), json!("present"));
        ai_source = true;
    } else if contains(buffer, b"algorithmicMedia") {
        info.insert("source_type".into(), json!("algorithmicMedia"));
    }

    if contains(buffer, b"c2pa.watermarked") {
        info.insert("watermarked".into(), json!(true));
    }
    let entries = soft_binding_entries_in(buffer);
    let soft_bindings = soft_binding_labels(&entries);
    let synthid = if entries.iter().any(|entry| entry.kind == "watermark") {
        Vec::new()
    } else {
        let watermarked = info
            .get("watermarked")
            .is_some_and(|value| value.as_bool() == Some(true));
        synthid_evidence_vendors_in(buffer, watermarked)
    };
    if ai_source && !synthid.is_empty() {
        let verdict = format!("present according to {} provenance", synthid.join(", "));
        info.insert("synthid_vendors".into(), json!(synthid));
        info.insert("synthid_watermark".into(), json!(verdict));
    }
    if !soft_bindings.is_empty() {
        let algorithms: Vec<&str> = entries.iter().map(|entry| entry.algorithm.as_str()).collect();
        info.insert("soft_binding_vendors".into(), json!(soft_bindings));
        info.insert("soft_binding".into(), json!(soft_bindings.join(", ")));
        info.insert("soft_binding_algorithm".into(), json!(algorithms.join(", ")));
    }
}

/// `cbor_text_after`: the definite-length CBOR text value right after the first `key`.
fn cbor_text_after(payload: &[u8], key: &[u8]) -> Option<String> {
    let key_end = payload.windows(key.len()).position(|window| window == key)?;
    let mut cursor = key_end + key.len();
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
        Err(_) => raw.iter().map(|&byte| char::from(byte)).collect(),
    })
}

/// Non-overlapping `\d{14}Z` matches.
fn timestamps(payload: &[u8]) -> Vec<String> {
    let mut found = Vec::new();
    let mut index = 0;
    while index + 15 <= payload.len() {
        let window = &payload[index..index + 15];
        if window[..14].iter().all(u8::is_ascii_digit) && window[14] == b'Z' {
            found.push(String::from_utf8_lossy(window).into_owned());
            index += 15;
        } else {
            index += 1;
        }
    }
    found
}

/// `parse_c2pa_chunk(payload, info)` on a fresh `info`.
pub fn parse_c2pa_chunk(payload: &[u8]) -> Map<String, Value> {
    let mut info = Map::new();
    info.insert("has_c2pa".into(), json!(true));
    info.insert(
        "type".into(),
        json!("C2PA (Coalition for Content Provenance and Authenticity)"),
    );
    info.insert(
        "c2pa_manifest".into(),
        json!(format!("C2PA manifest ({} bytes)", payload.len())),
    );
    info.insert("c2pa_validation_source".into(), json!("fallback"));
    for key in [
        "c2pa_validation_state",
        "c2pa_integrity",
        "c2pa_signature",
        "c2pa_signer_trust",
        "c2pa_signer_validity",
    ] {
        info.insert(key.into(), json!("unknown"));
    }
    info.insert("c2pa_validation_codes".into(), json!([]));
    info.insert("c2pa_failed_codes".into(), json!([]));
    populate_registry_fields(payload, &mut info);
    for (key, output) in [
        (&b"name"[..], "claim_generator"),
        (&b"specVersion"[..], "c2pa_spec"),
    ] {
        if let Some(value) = cbor_text_after(payload, key)
            && !value.is_empty()
            && unicode::str_is_printable(&value)
        {
            info.insert(output.into(), json!(value));
        }
    }
    let stamps = timestamps(payload);
    if let Some(first) = stamps.first() {
        info.insert("timestamp".into(), json!(first));
        if stamps.len() > 1 {
            info.insert("timestamps".into(), json!(&stamps[..stamps.len().min(3)]));
        }
    }
    info
}

/// Whether `token` occurs in `text` delimited by non-ASCII-alphanumeric characters.
fn word_match(text: &str, token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let bytes = text.as_bytes();
    let mut from = 0;
    while let Some(found) = text[from..].find(token) {
        let at = from + found;
        let before = at.checked_sub(1).map(|index| bytes[index]);
        let after = bytes.get(at + token.len()).copied();
        if !before.is_some_and(|byte| byte.is_ascii_alphanumeric())
            && !after.is_some_and(|byte| byte.is_ascii_alphanumeric())
        {
            return true;
        }
        from = at + text[at..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// The product a claim-generator string names by a whole-word registry token.
pub fn claim_generator_platform(generator: &str) -> Option<String> {
    if generator.is_empty() {
        return None;
    }
    let folded = unicode::casefold(generator);
    REGISTRY
        .claim_generator_platforms
        .iter()
        .find(|(token, _)| word_match(&folded, token))
        .map(|(_, platform)| platform.clone())
}

/// Platform of the first C2PA AI vendor `text` names as a whole word, case-insensitively.
pub fn c2pa_vendor_platform(text: &str) -> Option<String> {
    let folded = unicode::casefold(text);
    VENDOR_NEEDLES
        .iter()
        .find(|(needle, _)| word_match(&folded, needle))
        .map(|(_, platform)| (*platform).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_manifest_fields() {
        let payload =
            b"jumbc2pa\x00\x00\x00\x01http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia\
OpenAI ChatGPT GPT-6 Astra com.cognitive-proof.vsrmark.120261109120000Z";
        let info = parse_c2pa_chunk(payload);
        assert_eq!(info["ai_source_kind"], "generated");
        assert_eq!(info["timestamp"], "20261109120000Z");
        assert!(info["issuer"].as_str().unwrap().contains("OpenAI"));
    }

    #[test]
    fn word_boundaries() {
        assert!(word_match("made by openai.", "openai"));
        assert!(!word_match("openaix", "openai"));
        assert!(word_match("xopenai openai", "openai"));
    }
}

//! Registry matching over manifest strings and bytes, SynthID scoping and platform names.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use memchr::memmem;

use crate::registry::{
    C2PA_AI_TOOLS, C2PA_AI_VENDORS, C2PA_CLAIM_GENERATOR_PLATFORMS, SYNTHID_EDIT_SIGNERS, c2pa_issuers,
    is_identity_ai_org,
};
use crate::text::casefold;

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    memmem::find(haystack, needle).is_some()
}

/// Whether `token` occurs in `text` with no ASCII letter or digit on either side.
pub fn contains_word(text: &str, token: &str) -> bool {
    if token.is_empty() {
        return true;
    }
    let bytes = text.as_bytes();
    let finder = memmem::Finder::new(token.as_bytes());
    let mut from = 0;
    while let Some(found) = bytes.get(from..).and_then(|rest| finder.find(rest)) {
        let start = from + found;
        let end = start + token.len();
        let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let after_ok = bytes.get(end).is_none_or(|b| !b.is_ascii_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

fn push_unique(labels: &mut Vec<String>, label: &str) {
    if !labels.iter().any(|existing| existing == label) {
        labels.push(label.to_owned());
    }
}

/// Labels whose token is a whole word of the decoded string `text`, in registry order.
pub fn registry_word_matches<'a>(
    text: &str,
    registry: impl IntoIterator<Item = (&'a [u8], &'a str)>,
) -> Vec<String> {
    let mut labels = Vec::new();
    for (token, label) in registry {
        if let Ok(token) = std::str::from_utf8(token)
            && contains_word(text, token)
        {
            push_unique(&mut labels, label);
        }
    }
    labels
}

/// Labels whose token is a raw substring of manifest bytes, in registry order.
pub fn ordered_matches<'a>(
    buffer: &[u8],
    registry: impl IntoIterator<Item = (&'a [u8], &'a str)>,
) -> Vec<String> {
    let mut labels = Vec::new();
    for (token, label) in registry {
        if contains(buffer, token) {
            push_unique(&mut labels, label);
        }
    }
    labels
}

pub fn issuer_word_matches(text: &str) -> Vec<String> {
    registry_word_matches(text, c2pa_issuers())
}

pub fn tool_word_matches(text: &str) -> Vec<String> {
    registry_word_matches(text, C2PA_AI_TOOLS.iter().copied())
}

/// `claim_generator_platform`: the product a claim generator names by a whole-word token.
pub fn claim_generator_platform(generator: Option<&str>) -> Option<&'static str> {
    let folded = casefold(generator.filter(|g| !g.is_empty())?);
    C2PA_CLAIM_GENERATOR_PLATFORMS
        .iter()
        .find(|(token, _)| contains_word(&folded, token))
        .map(|(_, platform)| *platform)
}

/// Case-folded vendor needles and platforms, first occurrence of each needle kept.
static VENDOR_PLATFORM_NEEDLES: LazyLock<Vec<(String, &'static str)>> = LazyLock::new(|| {
    let mut needles: Vec<(String, &'static str)> = Vec::new();
    for vendor in C2PA_AI_VENDORS {
        let (Some(needle), Some(platform)) = (vendor.needle, vendor.platform) else {
            continue;
        };
        let folded = casefold(needle);
        match needles.iter_mut().find(|(existing, _)| *existing == folded) {
            Some(entry) => entry.1 = platform,
            None => needles.push((folded, platform)),
        }
    }
    needles
});

/// `c2pa_vendor_platform`: platform of the first vendor `text` names as a whole word.
pub fn c2pa_vendor_platform(text: &str) -> Option<&'static str> {
    let folded = casefold(text);
    VENDOR_PLATFORM_NEEDLES
        .iter()
        .find(|(needle, _)| contains_word(&folded, needle))
        .map(|(_, platform)| *platform)
}

/// `synthid_verdict`.
pub fn synthid_verdict(vendors: &str) -> String {
    format!("present according to {vendors} provenance")
}

/// Issuers whose provenance establishes SynthID for this asset (`synthid_evidence_vendors_in`).
///
/// `None` for either flag derives it from the bytes, as the raw fallback does.
pub fn synthid_evidence_vendors_in(
    buffer: &[u8],
    has_watermark_action: Option<bool>,
    records_synthid_action: Option<bool>,
) -> Vec<String> {
    let has_watermark_action = has_watermark_action.unwrap_or_else(|| contains(buffer, b"c2pa.watermarked"));
    let records_synthid_action = records_synthid_action.unwrap_or_else(|| contains(buffer, b"SynthID"));
    let identity_suffices =
        records_synthid_action || !SYNTHID_EDIT_SIGNERS.iter().any(|signer| contains(buffer, signer));
    C2PA_AI_VENDORS
        .iter()
        .filter(|vendor| {
            vendor.synthid
                && contains(buffer, vendor.issuer)
                && ((has_watermark_action && vendor.synthid_requires_watermark_action)
                    || (identity_suffices && !vendor.synthid_requires_watermark_action))
        })
        .map(|vendor| vendor.org.to_owned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Whether any of `orgs` asserts AI by identity alone.
pub fn any_identity_ai_org<'a>(orgs: impl IntoIterator<Item = &'a String>) -> bool {
    orgs.into_iter().any(|org| is_identity_ai_org(org))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_words_only() {
        assert!(contains_word("Nova Canvas by Canva", "Canva"));
        assert!(!contains_word("Nova Canvas", "Canva"));
        assert!(contains_word("aCanva Canva", "Canva"));
        assert_eq!(claim_generator_platform(Some("Suno/1.0")), Some("Suno"));
        assert_eq!(claim_generator_platform(Some("")), None);
        assert_eq!(
            c2pa_vendor_platform("signed by Google LLC"),
            Some("Google (Gemini / Imagen)")
        );
    }
}

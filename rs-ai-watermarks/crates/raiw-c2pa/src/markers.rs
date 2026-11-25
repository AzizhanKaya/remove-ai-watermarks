//! Whether flattened container markers assert AI origin rather than neutral C2PA transport.

use crate::attribution::{claim_generator_platform, tool_word_matches};
use crate::registry::C2PA_IDENTITY_AI_ORGS;

/// C2PA fields that describe a manifest without asserting AI (re-signing platforms leave only these).
pub const C2PA_STRUCTURAL_MARKERS: &[&str] = &[
    "c2pa_manifest",
    "claim_generator",
    "ai_tool",
    "c2pa_spec",
    "issuer",
    "actions",
    "c2pa_validation_source",
    "c2pa_validation_state",
    "c2pa_integrity",
    "c2pa_signature",
    "c2pa_signer_trust",
    "c2pa_signer_validity",
    "soft_binding",
    "soft_binding_algorithm",
    "soft_binding_value",
];

/// `markers_claim_ai` over `(key, value)` marker pairs.
pub fn markers_claim_ai<'a>(markers: impl IntoIterator<Item = (&'a str, &'a str)>) -> bool {
    let markers: Vec<(&str, &str)> = markers.into_iter().collect();
    if markers.is_empty() {
        return false;
    }
    let get = |name: &str| {
        markers
            .iter()
            .rev()
            .find(|(key, _)| *key == name)
            .map_or("", |(_, value)| *value)
    };
    if markers
        .iter()
        .any(|(key, _)| !C2PA_STRUCTURAL_MARKERS.contains(key) && *key != "source_type")
    {
        return true;
    }
    let source_type = get("source_type");
    let folded = crate::text::casefold(source_type);
    if folded.contains("trainedalgorithmicmedia")
        || folded.contains("compositesynthetic")
        || source_type.starts_with("c2pa.ai-disclosure")
    {
        return true;
    }
    let issuer = get("issuer");
    C2PA_IDENTITY_AI_ORGS.iter().any(|org| issuer.contains(org))
        || [get("claim_generator"), get("ai_tool")].iter().any(|value| {
            !value.is_empty()
                && (claim_generator_platform(Some(value)).is_some() || !tool_word_matches(value).is_empty())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_markers_alone_do_not_claim_ai() {
        assert!(!markers_claim_ai([
            ("issuer", "Google LLC"),
            ("actions", "opened, transcoded")
        ]));
        assert!(markers_claim_ai([("issuer", "Ideogram")]));
        assert!(markers_claim_ai([(
            "source_type",
            "trainedAlgorithmicMedia (AI-generated)"
        )]));
        assert!(markers_claim_ai([("ai_generator", "x")]));
        assert!(markers_claim_ai([("claim_generator", "Suno/2")]));
        assert!(!markers_claim_ai(std::iter::empty()));
    }
}

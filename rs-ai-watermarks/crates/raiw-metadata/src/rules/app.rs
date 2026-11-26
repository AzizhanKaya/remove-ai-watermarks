//! ByteDance-family app-export JSON and XMP AI disclosures (`_app_metadata_evidence`).

use std::sync::LazyLock;

use regex::Regex;
use regex::bytes::Regex as BytesRegex;

use crate::py::latin1;

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|err| unreachable!("static regex: {err}"))
}

static HINT: LazyLock<BytesRegex> = LazyLock::new(|| {
    BytesRegex::new(r"(?i-u)product|aigc_info|ContainsAiGeneratedContent|dc:publisher")
        .unwrap_or_else(|err| unreachable!("static regex: {err}"))
});
static PRODUCT: LazyLock<Regex> =
    LazyLock::new(|| re(r#"(?i)"product"[\s\x1c-\x1f]*:[\s\x1c-\x1f]*"([a-z0-9_]+)""#));
static EXPORT_TYPE: LazyLock<Regex> =
    LazyLock::new(|| re(r#"(?i)"exportType"[\s\x1c-\x1f]*:[\s\x1c-\x1f]*"([a-z0-9_]+)""#));
static AIGC_LABEL: LazyLock<Regex> =
    LazyLock::new(|| re(r#"(?i)"aigc_label_type"[\s\x1c-\x1f]*:[\s\x1c-\x1f]*[12][\s\x1c-\x1f]*[,}]"#));
static AIGC_TYPE: LazyLock<Regex> =
    LazyLock::new(|| re(r#"(?i)"aigc_type"[\s\x1c-\x1f]*:[\s\x1c-\x1f]*1[\s\x1c-\x1f]*[,}]"#));
static CAPCUT_AI_FLAG: LazyLock<Regex> = LazyLock::new(|| {
    re(concat!(
        r#"(?i)"(?:is_ai_lyric|is_aimusic_mv|is_use_ai_image_generation|is_use_ai_video_generation|"#,
        r#"is_use_aimusic_bgm|is_use_aimusic_vocal|is_use_jichuang_mode_in_ai_writer|is_use_voice_clone)""#,
        r#"[\s\x1c-\x1f]*:[\s\x1c-\x1f]*(?:1|"1"|true)[\s\x1c-\x1f]*[,}]"#
    ))
});
static CANVA_XMP_AIGC: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r#"(?i)ContainsAiGeneratedContent(?::ContainsAiGeneratedContent)?[\s\x1c-\x1f]*=[\s\x1c-\x1f]*["'](?:Yes|True)["']"#,
    )
});
static META_XMP_PUBLISHER: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"<dc:publisher>[\s\x1c-\x1f]*<rdf:(?:Bag|Seq)>[\s\x1c-\x1f]*<rdf:li>[\s\x1c-\x1f]*Meta[\s\x1c-\x1f]*</rdf:li>",
    )
});

const PROVENANCE_PRODUCTS: &[(&str, &str)] = &[
    ("doubao", "ByteDance Doubao"),
    ("xinghui", "ByteDance Xinghui"),
    ("dreamina", "ByteDance Dreamina"),
    ("dreamina_oversea", "ByteDance Dreamina"),
];

/// Removable product provenance and stronger AI-origin evidence, as `(provenance, generator)`.
pub type AppEvidence = (Option<&'static str>, Option<&'static str>);

/// `_app_metadata_evidence(value)` for a Python `str` value.
pub fn app_evidence_str(value: &str) -> AppEvidence {
    if value.is_empty() {
        return (None, None);
    }
    evidence(&value.replace("\\\"", "\""))
}

/// `_app_metadata_evidence(value)` for a Python `bytes` value.
pub fn app_evidence_bytes(value: &[u8]) -> AppEvidence {
    if value.is_empty() || !HINT.is_match(value) {
        return (None, None);
    }
    evidence(&latin1(value).replace("\\\"", "\""))
}

/// Either half of the evidence is present.
pub fn any(evidence: AppEvidence) -> bool {
    evidence.0.is_some() || evidence.1.is_some()
}

fn evidence(normalized: &str) -> AppEvidence {
    let products: Vec<String> = PRODUCT
        .captures_iter(normalized)
        .map(|c| c[1].to_lowercase())
        .collect();
    let provenance = products.iter().find_map(|product| {
        PROVENANCE_PRODUCTS
            .iter()
            .find(|(name, _)| name == product)
            .map(|(_, label)| *label)
    });
    let has_product = |names: &[&str]| products.iter().any(|p| names.contains(&p.as_str()));
    let generator = if has_product(&["dreamina", "dreamina_oversea"])
        && EXPORT_TYPE
            .captures_iter(normalized)
            .any(|c| c[1].to_lowercase() == "generation")
    {
        Some("ByteDance Dreamina")
    } else if has_product(&["vicut", "lv"]) && CAPCUT_AI_FLAG.is_match(normalized) {
        Some("ByteDance CapCut AI")
    } else if normalized.to_lowercase().contains("\"aigc_info\"") {
        if has_product(&["aweme"]) && AIGC_TYPE.is_match(normalized) {
            Some("ByteDance Aweme AI")
        } else if AIGC_LABEL.is_match(normalized) {
            Some("Embedded app AIGC disclosure")
        } else {
            None
        }
    } else if normalized.contains("ContainsAiGeneratedContent") && CANVA_XMP_AIGC.is_match(normalized) {
        Some("Canva")
    } else if normalized.contains("dc:publisher")
        && META_XMP_PUBLISHER.is_match(normalized)
        && [
            "trainedAlgorithmicMedia",
            "compositeWithTrainedAlgorithmicMedia",
            "compositeSynthetic",
        ]
        .iter()
        .any(|marker| normalized.contains(marker))
    {
        Some("Meta AI")
    } else {
        None
    };
    (provenance, generator)
}

/// `app_provenance_from_metadata` for bytes.
pub fn app_provenance_bytes(value: &[u8]) -> Option<&'static str> {
    app_evidence_bytes(value).0
}

/// `app_generator_from_metadata` for bytes.
pub fn app_generator_bytes(value: &[u8]) -> Option<&'static str> {
    app_evidence_bytes(value).1
}

/// `app_generator_from_metadata` for a `str`.
pub fn app_generator_str(value: &str) -> Option<&'static str> {
    app_evidence_str(value).1
}

//! Deterministic text provenance identification and metadata/steganography removal.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use raiw_core::{JsonStyle, to_py_json};
use serde::{Serialize, Serializer};
use serde_json::{Map, Value};

use crate::c2pa::{c2pa_vendor_platform, claim_generator_platform, parse_c2pa_chunk};
use crate::error::TextError;
use crate::io::{TextInput, prepare_text_io, write_output_text};
use crate::provenance::armor::strip_c2pa_ascii_armor;
use crate::provenance::c2patxt::{C2PATXT_VERSION, strip_c2patxt_suffix};
use crate::provenance::html::strip_html_c2pa;
use crate::provenance::vsrmark::{VSRMARK_ALGORITHM, detect_vsrmark};
use crate::spans::split_protected_text_spans;
use crate::stego::decode::{
    detect_innamark_payload, detect_stegcloak_stream, detect_variation_selector_stego_payload,
    detect_zero_width_binary_payload,
};
use crate::stego::prose::{
    decode_invisible_html_entities, normalize_exotic_spaces, normalize_mixed_script_homoglyphs,
    strip_assistant_citations,
};
use crate::stego::strip::extract_and_strip_stego_codepoints;
use crate::unicode;

/// The published Windows AI text-watermark sequence.
pub const WINDOWS_AI_TEXT_WATERMARK: &str = "\u{200b}\u{034f}\u{2060}\u{200b}";

const STATISTICAL_SAMPLING_CAVEAT: &str = "Token-level statistical LLM watermarks (Google SynthID-Text, Anthropic Claude Fable 5.1/5.2 and Mythos 5.1/5.2, OpenAI GPT-6 Astra textGrain, KGW, Unigram) are keyed in the token-sampling distribution and cannot be verified locally without the provider's secret key.";

/// A deterministic text provenance or steganography signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextSignal {
    WindowsAiTextWatermark,
    C2paTextManifestWrapper,
    C2paAsciiArmor,
    C2paHtmlHead,
    Vsrmark,
    InnamarkWhitespace,
    Stegcloak,
    GroundtruthMcp,
    AssistantCitationArtifact,
    HomoglyphSteganography,
    WhitespaceSteganography,
    UnicodeTags,
    ZeroWidthSteganography,
}

impl TextSignal {
    /// Signals that establish provenance with high confidence.
    const fn is_high_confidence(self) -> bool {
        matches!(
            self,
            Self::C2paTextManifestWrapper
                | Self::C2paAsciiArmor
                | Self::C2paHtmlHead
                | Self::Vsrmark
                | Self::InnamarkWhitespace
        )
    }
}

/// Verdict strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    High,
    Medium,
    Unknown,
}

#[allow(clippy::ref_option)]
fn serialize_path<S: Serializer>(path: &Option<PathBuf>, serializer: S) -> Result<S::Ok, S::Error> {
    match path {
        Some(path) => serializer.serialize_str(&path.to_string_lossy()),
        None => serializer.serialize_none(),
    }
}

/// Locally verifiable provenance and character-steganography signals in text.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextProvenanceReport {
    #[serde(serialize_with = "serialize_path")]
    pub source: Option<PathBuf>,
    pub is_ai_generated: Option<bool>,
    pub confidence: Confidence,
    pub platform: Option<String>,
    pub has_ai_metadata: bool,
    pub signals: Vec<TextSignal>,
    pub metadata_markers: IndexMap<String, String>,
    pub c2pa_info: Map<String, Value>,
    pub vsrmark_payload: Option<String>,
    pub unicode_tags_payload: Option<String>,
    pub stego_codepoint_count: usize,
    pub caveats: Vec<String>,
}

impl TextProvenanceReport {
    /// `json.dumps(asdict(report), default=str, indent=2)`, the `text identify --json` output.
    pub fn to_json(&self) -> String {
        to_py_json(self, JsonStyle::PRETTY)
    }
}

/// Result of stripping deterministic C2PA wrappers and character steganography.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextMetadataResult {
    #[serde(serialize_with = "serialize_path")]
    pub source: Option<PathBuf>,
    #[serde(serialize_with = "serialize_path")]
    pub output: Option<PathBuf>,
    pub cleaned_text: String,
    pub removed_signals: Vec<TextSignal>,
    pub detected_markers: IndexMap<String, String>,
    pub remaining_markers: IndexMap<String, String>,
    pub removed_codepoints: usize,
}

fn platform_from_c2pa_info(info: &Map<String, Value>) -> Option<String> {
    if let Some(Value::String(generator)) = info.get("claim_generator")
        && let Some(platform) =
            claim_generator_platform(generator).or_else(|| c2pa_vendor_platform(generator))
    {
        return Some(platform);
    }
    ["ai_tool", "issuer", "claim_generator", "soft_binding"]
        .iter()
        .find_map(|key| match info.get(*key) {
            Some(Value::String(value)) if !unicode::strip(value).is_empty() => {
                Some(c2pa_vendor_platform(value).unwrap_or_else(|| unicode::strip(value).to_owned()))
            }
            _ => None,
        })
}

fn inspect_jumbf(payload: &[u8]) -> Map<String, Value> {
    if payload.is_empty() {
        Map::new()
    } else {
        parse_c2pa_chunk(payload)
    }
}

/// Per-pass counts from prose preprocessing.
#[derive(Debug, Default)]
struct ProseCounts {
    entities: usize,
    citations: usize,
    homoglyphs: usize,
    spaces: usize,
}

fn preprocess_unlocked_prose_segments(text: &str, force_innamark_clean: bool) -> (String, ProseCounts) {
    let mut counts = ProseCounts::default();
    if text.is_empty() {
        return (String::new(), counts);
    }
    let spans = split_protected_text_spans(text);
    let last = spans.len().saturating_sub(1);
    let mut out = String::with_capacity(text.len());
    for (index, span) in spans.iter().enumerate() {
        if span.kind.is_code_locked() {
            out.push_str(&span.text);
            continue;
        }
        let (segment, entities) = decode_invisible_html_entities(&span.text);
        let (segment, citations) = strip_assistant_citations(&segment, index == last);
        let (segment, homoglyphs) = normalize_mixed_script_homoglyphs(&segment);
        let (segment, spaces) = normalize_exotic_spaces(&segment, force_innamark_clean);
        counts.entities += entities;
        counts.citations += citations;
        counts.homoglyphs += homoglyphs;
        counts.spaces += spaces;
        out.push_str(&segment);
    }
    (out, counts)
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Run every deterministic inspection pass and return the report and the cleaned text.
pub(crate) fn inspect_and_strip_text(text: &str, source: Option<&Path>) -> (TextProvenanceReport, String) {
    let mut signals = Vec::new();
    let mut markers: IndexMap<String, String> = IndexMap::new();
    let mut c2pa_info: Map<String, Value> = Map::new();
    let mut caveats: Vec<String> = Vec::new();

    let windows_count = text.matches(WINDOWS_AI_TEXT_WATERMARK).count();
    let owned;
    let text = if windows_count > 0 {
        // Remove the complete sequence before classifying neighboring joiners.
        owned = text.replace(WINDOWS_AI_TEXT_WATERMARK, "");
        signals.push(TextSignal::WindowsAiTextWatermark);
        markers.insert(
            "windows_ai_text_watermark".into(),
            format!("{windows_count} published Windows AI text sequence(s)"),
        );
        caveats.push(
            "The Windows AI text sequence is forgeable and does not establish source attribution.".into(),
        );
        owned.as_str()
    } else {
        text
    };

    let (after_c2patxt, c2patxt) = strip_c2patxt_suffix(text);
    if let Some(found) = &c2patxt {
        signals.push(TextSignal::C2paTextManifestWrapper);
        let padding = if found.padding_bytes > 0 {
            format!(", {} bytes padding", found.padding_bytes)
        } else {
            String::new()
        };
        markers.insert(
            "c2pa_text_wrapper".into(),
            format!(
                "C2PATXT v{C2PATXT_VERSION} ({} bytes JUMBF{padding})",
                found.manifest_bytes.len()
            ),
        );
        c2pa_info.extend(inspect_jumbf(&found.manifest_bytes));
        if !found.had_bom_prefix {
            caveats.push(
                "C2PATXT wrapper lacked the leading U+FEFF sentinel required by C2PA Appendix A.8.".into(),
            );
        }
    }

    let (after_armor, armor) = strip_c2pa_ascii_armor(&after_c2patxt);
    if let Some(found) = &armor {
        signals.push(TextSignal::C2paAsciiArmor);
        let marker = match (&found.manifest_bytes, &found.manifest_uri) {
            (Some(bytes), _) => format!("Inline JUMBF ({} bytes)", bytes.len()),
            (None, Some(uri)) => format!("External manifest URI ({uri})"),
            (None, None) => format!("{} block(s)", found.count),
        };
        markers.insert("c2pa_ascii_armor".into(), marker);
        for (key, value) in found
            .manifest_bytes
            .as_deref()
            .map(inspect_jumbf)
            .unwrap_or_default()
        {
            c2pa_info.entry(key).or_insert(value);
        }
    }

    let (after_html, html) = strip_html_c2pa(&after_armor);
    if let Some(found) = &html {
        signals.push(TextSignal::C2paHtmlHead);
        let marker = match (&found.manifest_bytes, &found.manifest_uri) {
            (Some(bytes), _) => format!("Inline HTML C2PA JUMBF ({} bytes)", bytes.len()),
            (None, Some(uri)) => format!("External HTML C2PA link ({uri})"),
            (None, None) => format!("{} HTML C2PA element(s)", found.element_count),
        };
        markers.insert("c2pa_html_manifest".into(), marker);
        for (key, value) in found
            .manifest_bytes
            .as_deref()
            .map(inspect_jumbf)
            .unwrap_or_default()
        {
            c2pa_info.entry(key).or_insert(value);
        }
        if found.element_count > 1 {
            caveats.push(
                "C2PA Appendix A.7 violation: HTML <head> contains multiple C2PA manifest elements.".into(),
            );
        }
        if found.outside_head {
            caveats.push("C2PA Appendix A.7 warning: C2PA HTML element found outside <head>.".into());
        }
    }

    let chars: Vec<char> = after_html.chars().collect();
    let vsrmark = detect_vsrmark(&chars);
    let vsrmark_indices: BTreeSet<usize> = vsrmark
        .as_ref()
        .map(|found| found.indices.clone())
        .unwrap_or_default();
    if let Some(found) = &vsrmark {
        signals.push(TextSignal::Vsrmark);
        markers.insert(
            "vsrmark".into(),
            format!(
                "{VSRMARK_ALGORITHM} ({} bytes, crc32=0x{:08x})",
                found.raw_payload.len(),
                found.crc32
            ),
        );
    }

    let innamark = detect_innamark_payload(&after_html);
    if let Some(payload) = &innamark {
        signals.push(TextSignal::InnamarkWhitespace);
        markers.insert("innamark_payload".into(), payload.clone());
    }

    let stegcloak = detect_stegcloak_stream(&chars);
    if let Some(info) = &stegcloak {
        signals.push(TextSignal::Stegcloak);
        markers.insert("stegcloak".into(), info.clone());
    }

    let zero_width = detect_zero_width_binary_payload(&chars);
    if let Some(payload) = zero_width.get("groundtruth_mcp") {
        signals.push(TextSignal::GroundtruthMcp);
        markers.insert("groundtruth_mcp_watermark".into(), payload.clone());
    }
    if let Some(payload) = zero_width.get("zero_width_binary") {
        markers.insert("zero_width_payload".into(), payload.clone());
    }
    for (key, value) in detect_variation_selector_stego_payload(&chars, &vsrmark_indices) {
        markers.insert(format!("{key}_payload"), value);
    }

    // Restore neighboring characters before classifying spaces and other contextual controls.
    let after_vsrmark: String = if vsrmark_indices.is_empty() {
        after_html
    } else {
        chars
            .iter()
            .enumerate()
            .filter(|(index, _)| !vsrmark_indices.contains(index))
            .map(|(_, &ch)| ch)
            .collect()
    };
    let (preprocessed, counts) = preprocess_unlocked_prose_segments(&after_vsrmark, innamark.is_some());
    if counts.entities > 0 {
        markers.insert(
            "invisible_html_entities".into(),
            format!("{} invisible HTML entity(ies)", counts.entities),
        );
    }
    if counts.citations > 0 {
        signals.push(TextSignal::AssistantCitationArtifact);
        markers.insert(
            "assistant_citation_artifact".into(),
            format!("{} assistant citation artifact(s)", counts.citations),
        );
    }
    if counts.homoglyphs > 0 {
        signals.push(TextSignal::HomoglyphSteganography);
        markers.insert(
            "homoglyph_steganography".into(),
            format!("{} mixed-script confusable homoglyph(s)", counts.homoglyphs),
        );
    }
    if counts.spaces > 0 && !signals.contains(&TextSignal::InnamarkWhitespace) {
        signals.push(TextSignal::WhitespaceSteganography);
        markers.insert(
            "whitespace_steganography".into(),
            format!("{} exotic space codepoint(s)", counts.spaces),
        );
    }

    let preserve_contextual = stegcloak.as_deref().is_none_or(str::is_empty) && zero_width.is_empty();
    let strip = extract_and_strip_stego_codepoints(&preprocessed, preserve_contextual);
    if strip.tag_codepoints > 0 {
        signals.push(TextSignal::UnicodeTags);
        let chars = strip
            .tags_payload
            .as_deref()
            .map_or(0, |payload| payload.chars().count());
        markers.insert(
            "unicode_tags".into(),
            format!("Unicode Tags hidden ASCII ({chars} chars)"),
        );
    }
    if strip.anomalous > 0 {
        signals.push(TextSignal::ZeroWidthSteganography);
        markers.insert(
            "zero_width_steganography".into(),
            format!("{} steganographic codepoint(s)", strip.anomalous),
        );
    }

    let total = windows_count * WINDOWS_AI_TEXT_WATERMARK.chars().count()
        + c2patxt.as_ref().map_or(0, |found| found.removed_codepoints)
        + vsrmark_indices.len()
        + strip.removed
        + counts.citations
        + counts.homoglyphs
        + counts.spaces;

    for key in [
        "issuer",
        "ai_tool",
        "source_type",
        "claim_generator",
        "soft_binding",
        "synthid_watermark",
    ] {
        if let Some(value) = c2pa_info.get(key) {
            markers.insert(format!("c2pa_{key}"), value_text(value));
        }
    }

    let has_high = signals.iter().any(|signal| signal.is_high_confidence());
    let has_any = !signals.is_empty();
    let platform = platform_from_c2pa_info(&c2pa_info).or_else(|| {
        if vsrmark.is_some() {
            Some("VSRMark (com.cognitive-proof.vsrmark.1)".to_owned())
        } else {
            innamark
                .as_ref()
                .map(|_| "Innamark (arXiv:2502.12710)".to_owned())
        }
    });
    let (is_ai_generated, confidence) = match (has_high, has_any) {
        (true, _) => (Some(true), Confidence::High),
        (false, true) => (Some(true), Confidence::Medium),
        (false, false) => (None, Confidence::Unknown),
    };
    caveats.push(STATISTICAL_SAMPLING_CAVEAT.into());

    let report = TextProvenanceReport {
        source: source.map(Path::to_path_buf),
        is_ai_generated,
        confidence,
        platform,
        has_ai_metadata: has_any,
        signals,
        metadata_markers: markers,
        c2pa_info,
        vsrmark_payload: vsrmark.map(|found| found.payload),
        unicode_tags_payload: strip.tags_payload,
        stego_codepoint_count: total,
        caveats,
    };
    (report, strip.cleaned)
}

/// Inspect a text string or file for C2PA wrappers and character steganography.
pub fn identify_text<'a>(input: impl Into<TextInput<'a>>) -> Result<TextProvenanceReport, TextError> {
    let input = input.into();
    let text = input.read()?;
    Ok(inspect_and_strip_text(&text, input.source()).0)
}

/// Strip C2PA text/HTML wrappers, VSRMark soft bindings, Unicode Tags and zero-width steganography.
///
/// File input without `output` writes `<stem>_clean<suffix>` next to the source.
pub fn remove_text_metadata<'a>(
    input: impl Into<TextInput<'a>>,
    output: Option<&Path>,
) -> Result<TextMetadataResult, TextError> {
    let input = input.into();
    let (resolved, text) = prepare_text_io(input, output)?;
    let (before, cleaned) = inspect_and_strip_text(&text, input.source());
    let remaining = if cleaned == text && !before.has_ai_metadata {
        before.metadata_markers.clone()
    } else {
        inspect_and_strip_text(&cleaned, None).0.metadata_markers
    };
    write_output_text(resolved.as_deref(), &cleaned)?;
    Ok(TextMetadataResult {
        source: input.source().map(Path::to_path_buf),
        output: resolved,
        cleaned_text: cleaned,
        removed_signals: before.signals,
        detected_markers: before.metadata_markers,
        remaining_markers: remaining,
        removed_codepoints: before.stego_codepoint_count,
    })
}

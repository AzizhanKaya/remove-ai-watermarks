//! File-level detectors: `has_ai_metadata`, `get_ai_metadata` and the per-signal probes.

use std::path::Path;
use std::sync::{LazyLock, Mutex};

use indexmap::IndexMap;
use raiw_c2pa::soft_binding::{self, SoftBindingKind};
use raiw_c2pa::{
    C2paInfo, ValidationSource, extract_c2pa_info, parse_c2pa_chunk, synthid_evidence_vendors_in,
    synthid_verdict,
};
use raiw_container::ebml::{self, EBML_MAGIC, c2pa_attachment_payloads};
use raiw_container::isobmff::{self, is_isobmff};
use raiw_container::markers::{
    AIGC_MARKERS, IPTC_AI_FIELD_MARKERS, IPTC_AI_MARKERS, KLING_SEI_TAG, SAMSUNG_EDITOR_MARKER, contains,
    contains_any,
};
use raiw_container::{flv, riff};
use raiw_core::fs::suffix_lower;
use regex::bytes::Regex;

use crate::audio_creation::audio_creation_tool;
use crate::cache::{LruCache, memoize};
use crate::exif::{Ifd, tag};
use crate::exif_ai::exif_generator_candidates;
use crate::hooks::Hooks;
use crate::pil::{self, InfoValue};
use crate::rules::app::{any, app_evidence_bytes};
use crate::rules::generator_from_metadata;
use crate::rules::keys::{HF_JOB_KEY, is_ai_key};
use crate::rules::markers::{
    XMP_IPTC_SOURCE_TYPE, c2pa_cloud_manifest_in, c2pa_marker_in, iptc_ai_system_in, samsung_genai_in,
};
use crate::rules::tc260::{Tc260Label, aigc_label_from_metadata};
use crate::rules::xai::{decoded_pair, xai_signature_in_metadata, xai_signature_pair};
use crate::scan::{HEAD_BYTES, QUICK_SCAN_BYTES, read_file_tail, scan_head};
use crate::tc260_signature::check_tc260_signature;

/// `get_ai_metadata` output: key to display value, in Python's insertion order.
pub type AiMetadata = IndexMap<String, String>;

/// Suffixes routed to the media strippers (`_NON_IMAGE_MEDIA_EXTS`).
pub const NON_IMAGE_MEDIA_EXTS: &[&str] = &[
    ".mp4", ".mov", ".m4v", ".m4a", ".webm", ".mkv", ".mka", ".avi", ".flv", ".mp3", ".wav", ".flac", ".ogg",
    ".oga", ".opus", ".aac",
];

static URN_C2PA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i-u)urn:c2pa:").unwrap_or_else(|err| unreachable!("static regex: {err}"))
});

fn head_or_empty(path: &Path, size: usize) -> std::sync::Arc<Vec<u8>> {
    scan_head(path, size).unwrap_or_default()
}

fn truncate200(value: &str) -> String {
    if value.chars().count() > 200 {
        let mut out: String = value.chars().take(200).collect();
        out.push('\u{2026}');
        out
    } else {
        value.to_owned()
    }
}

// ── TC260 label ──

fn aigc_label_impl(path: &Path) -> Option<Tc260Label> {
    let probe = pil::probe(path);
    if let Some(value) = probe
        .image()
        .and_then(|img| img.info.get("AIGC"))
        .and_then(InfoValue::as_str)
        && let Some(label) = aigc_label_from_metadata(b"", &[value])
    {
        return Some(label);
    }
    let readers: [&dyn Fn() -> Vec<Vec<u8>>; 4] = [
        &|| isobmff::tc260_aigc_payloads(path, &Hooks).unwrap_or_default(),
        &|| ebml::tc260_aigc_payloads(path, &Hooks),
        &|| riff::tc260_aigc_payloads(path, &Hooks),
        &|| flv::tc260_aigc_payloads(path, &Hooks),
    ];
    for reader in readers {
        let candidates: Vec<String> = reader().iter().map(|p| crate::py::utf8_replace(p)).collect();
        if let Some(label) = aigc_label_from_metadata(b"", &candidates) {
            return Some(label);
        }
    }
    aigc_label_from_metadata::<&str>(&head_or_empty(path, HEAD_BYTES), &[])
}

static AIGC: Mutex<LruCache<Option<Tc260Label>>> = Mutex::new(LruCache::new(4));
static HF_JOB: Mutex<LruCache<Option<String>>> = Mutex::new(LruCache::new(4));
static SAMSUNG: Mutex<LruCache<Option<i64>>> = Mutex::new(LruCache::new(4));
static IPTC_SYSTEM: Mutex<LruCache<Option<String>>> = Mutex::new(LruCache::new(4));
static XAI: Mutex<LruCache<bool>> = Mutex::new(LruCache::new(4));

/// `aigc_label(path)`: the China TC260 AI-labeling block, if present.
pub fn aigc_label(path: &Path) -> Option<Tc260Label> {
    memoize(&AIGC, path, || aigc_label_impl(path))
}

/// `huggingface_job(path)`: Higgsfield's `hf-job-id` PNG text value.
pub fn huggingface_job(path: &Path) -> Option<String> {
    memoize(&HF_JOB, path, || {
        let probe = pil::probe(path);
        let value = probe.image()?.info.get(HF_JOB_KEY)?.as_str()?;
        let stripped = crate::py::strip(value);
        (!stripped.is_empty()).then(|| stripped.to_owned())
    })
}

/// `samsung_genai(path)`: the non-zero Galaxy AI `genAIType`.
pub fn samsung_genai(path: &Path) -> Option<i64> {
    memoize(&SAMSUNG, path, || {
        let mut data = head_or_empty(path, QUICK_SCAN_BYTES).to_vec();
        if !contains(&data, SAMSUNG_EDITOR_MARKER) {
            let oversize = std::fs::metadata(path).is_ok_and(|m| m.len() > QUICK_SCAN_BYTES as u64);
            if oversize {
                data = read_file_tail(path, QUICK_SCAN_BYTES).unwrap_or_default();
            }
        }
        samsung_genai_in(&data)
    })
}

/// `iptc_ai_system(path)`: the IPTC 2025.1 AI-disclosure note.
pub fn iptc_ai_system(path: &Path) -> Option<String> {
    memoize(&IPTC_SYSTEM, path, || {
        iptc_ai_system_in(&head_or_empty(path, HEAD_BYTES))
    })
}

/// `c2pa_cloud_manifest(path)`: a Durable Content Credentials repository pointer.
pub fn c2pa_cloud_manifest(path: &Path) -> Option<&'static str> {
    c2pa_cloud_manifest_in(&head_or_empty(path, QUICK_SCAN_BYTES))
}

fn xai_signature_impl(path: &Path) -> bool {
    let probe = pil::probe(path);
    let mut pair = false;
    if let Some(image) = probe.image() {
        let exif = probe.exif_bytes();
        let info = probe.info_after_exif().unwrap_or(&image.info);
        let texts = |keys: &[&str]| -> Vec<String> {
            keys.iter()
                .filter_map(|k| info.get(*k).and_then(InfoValue::as_str).map(str::to_owned))
                .collect()
        };
        let descriptions = texts(&["Description", "ImageDescription"]);
        let creators = texts(&["Author", "Artist", "Creator"]);
        pair = descriptions
            .iter()
            .any(|d| creators.iter().any(|c| xai_signature_pair(d, c)));
        let mut aborted = false;
        match pil::iptc::getiptcinfo(image) {
            Ok(Some(iptc)) if !iptc.is_empty() => {
                let field = |key| match iptc.get(&key) {
                    None => Some(Vec::new()),
                    Some(pil::iptc::IptcValue::Bytes(b)) => Some(b.clone()),
                    Some(_) => None,
                };
                if let (Some(caption), Some(byline)) = (field((2, 120)), field((2, 80))) {
                    pair = pair || decoded_pair(&caption, &byline);
                }
            }
            Ok(_) => {}
            Err(_) => aborted = true,
        }
        if !aborted
            && let Some(exif) = exif
            && let Ok(dict) = crate::exif::load(&exif)
        {
            pair = pair
                || xai_signature_pair(
                    &dict.text(Ifd::Zeroth, tag::IMAGE_DESCRIPTION),
                    &dict.text(Ifd::Zeroth, tag::ARTIST),
                );
        }
    }
    if pair {
        return true;
    }
    scan_head(path, HEAD_BYTES).is_ok_and(|head| xai_signature_in_metadata(&head))
}

/// `xai_signature(path)`: xAI / Grok's signature + UUID pair in any supported field.
pub fn xai_signature(path: &Path) -> bool {
    memoize(&XAI, path, || xai_signature_impl(path))
}

/// `exif_generator(path)`: a known AI generator named in EXIF, PNG text or XMP.
pub fn exif_generator(path: &Path) -> Option<String> {
    let probe = pil::probe(path);
    let candidates = probe.image().map_or_else(Vec::new, |image| {
        let exif = probe.exif_bytes();
        let info = probe.info_after_exif().unwrap_or(&image.info);
        exif_generator_candidates(info, exif.as_deref())
    });
    generator_from_metadata(&candidates, &head_or_empty(path, HEAD_BYTES))
}

/// `synthid_source(path, c2pa_info=...)`: vendors whose provenance establishes SynthID.
pub fn synthid_source(path: &Path, c2pa: Option<&C2paInfo>) -> Option<String> {
    let owned;
    let c2pa = if let Some(info) = c2pa {
        Some(info)
    } else {
        owned = extract_c2pa_info(path);
        owned.as_ref()
    };
    if let Some(info) = c2pa {
        if info.has_invalid_credential() {
            return None;
        }
        if !info.synthid_vendors.is_empty() {
            return Some(info.synthid_vendors.join(", "));
        }
        if info.validation.source == ValidationSource::Reader {
            return None;
        }
        if let Some(algorithm) = &info.soft_binding_algorithm {
            let entries = soft_binding::registry_entries_in(algorithm.as_bytes());
            if entries.is_empty() || entries.iter().any(|e| e.kind == SoftBindingKind::Watermark) {
                return None;
            }
        }
    }
    let data = head_or_empty(path, HEAD_BYTES);
    let ai_source =
        contains(&data, b"trainedAlgorithmicMedia") || contains(&data, b"TrainedAlgorithmicMedia");
    if !(c2pa_marker_in(&data) && ai_source) {
        return None;
    }
    if soft_binding::registry_entries_in(&data)
        .iter()
        .any(|e| e.kind == SoftBindingKind::Watermark)
    {
        return None;
    }
    let matched = synthid_evidence_vendors_in(&data, None, None);
    (!matched.is_empty()).then(|| matched.join(", "))
}

fn reader_has_manifest(path: &Path) -> bool {
    crate::c2pa_store::manifest_store_json(path).is_some()
}

/// `has_ai_metadata(path)`: whether any supported AI declaration is present.
pub fn has_ai_metadata(path: &Path) -> std::io::Result<bool> {
    if audio_creation_tool(path).is_some() {
        return Ok(true);
    }
    let probe = pil::probe(path);
    let mut candidates = Vec::new();
    if let Some(image) = probe.image() {
        if image.info.keys().any(|key| is_ai_key(key)) {
            return Ok(true);
        }
        let exif = probe.exif_bytes();
        let info = probe.info_after_exif().unwrap_or(&image.info);
        candidates = exif_generator_candidates(info, exif.as_deref());
        if exif
            .as_deref()
            .is_some_and(|e| !e.is_empty() && any(app_evidence_bytes(e)))
        {
            return Ok(true);
        }
    }
    if reader_has_manifest(path) {
        return Ok(true);
    }
    let data = scan_head(path, QUICK_SCAN_BYTES)?;
    if c2pa_marker_in(&data)
        || contains_any(&data, AIGC_MARKERS)
        || contains_any(&data, IPTC_AI_MARKERS)
        || contains_any(&data, IPTC_AI_FIELD_MARKERS)
        || any(app_evidence_bytes(&data))
        || aigc_label(path).is_some()
        || huggingface_job(path).is_some()
    {
        return Ok(true);
    }
    if is_isobmff(&data) {
        if !isobmff::ai_metadata_tags(path).unwrap_or_default().is_empty() {
            return Ok(true);
        }
        let sei = isobmff::scan_video_sei(path, true).unwrap_or_default();
        if !sei.kling_tag_offsets.is_empty()
            || !sei.ai_sei_spans.is_empty()
            || !sei.av1_obu_patches.is_empty()
        {
            return Ok(true);
        }
    } else if data.starts_with(EBML_MAGIC)
        && (!c2pa_attachment_payloads(path).is_empty() || !ebml::scan_av1_ai_obus(path, true).is_empty())
    {
        return Ok(true);
    }
    if !NON_IMAGE_MEDIA_EXTS.contains(&suffix_lower(path).as_str()) {
        let head = scan_head(path, HEAD_BYTES)?;
        if generator_from_metadata(&candidates, &head).is_some() {
            return Ok(true);
        }
    }
    Ok(xai_signature(path))
}

const C2PA_KEYS: &[&str] = &[
    "c2pa_manifest",
    "claim_generator",
    "ai_tool",
    "c2pa_spec",
    "issuer",
    "source_type",
    "actions",
    "synthid_watermark",
    "soft_binding",
    "soft_binding_algorithm",
    "soft_binding_value",
    "c2pa_validation_source",
    "c2pa_validation_state",
    "c2pa_integrity",
    "c2pa_signature",
    "c2pa_signer_trust",
    "c2pa_signer_validity",
];

fn set_default(result: &mut AiMetadata, key: &str, value: impl Into<String>) {
    if !result.contains_key(key) {
        result.insert(key.to_owned(), value.into());
    }
}

/// `get_ai_metadata(path)`: every supported AI declaration, as display strings.
pub fn get_ai_metadata(path: &Path) -> AiMetadata {
    let mut result = AiMetadata::new();
    if let Some(tool) = audio_creation_tool(path) {
        result.insert("audio_generator".into(), tool.into());
    }
    let probe = pil::probe(path);
    if let Some(image) = probe.image() {
        for (key, value) in &image.info {
            if !is_ai_key(key) {
                continue;
            }
            let rendered = match value {
                InfoValue::Bytes(bytes) => format!("<binary {} bytes>", bytes.len()),
                InfoValue::Str(text) | InfoValue::IText { text, .. } if text.chars().count() > 200 => {
                    truncate200(text)
                }
                other => other.py_str(),
            };
            result.insert(key.clone(), rendered);
        }
    }
    for (key, value) in isobmff::ai_metadata_tags(path).unwrap_or_default() {
        set_default(&mut result, &key, truncate200(&value));
    }
    let c2pa = extract_c2pa_info(path);
    if let Some(info) = &c2pa {
        let entries = info.string_entries();
        for key in C2PA_KEYS {
            if let Some((_, value)) = entries.iter().find(|(k, _)| k == key) {
                set_default(&mut result, key, value.clone());
            }
        }
    }
    if !result.contains_key("synthid_watermark")
        && let Some(vendor) = synthid_source(path, c2pa.as_ref())
    {
        set_default(&mut result, "synthid_watermark", synthid_verdict(&vendor));
    }
    if !result.contains_key("soft_binding") {
        let vendors = soft_binding::vendors_in(&head_or_empty(path, HEAD_BYTES));
        if !vendors.is_empty() {
            result.insert("soft_binding".into(), vendors.join(", "));
        }
    }
    if let Some(aigc) = aigc_label(path) {
        let producer = aigc.get("ContentProducer").cloned().unwrap_or_default();
        let suffix = if producer.is_empty() {
            String::new()
        } else {
            format!("; producer {producer}")
        };
        result.insert("aigc_label".into(), format!("China AIGC label (TC260){suffix}"));
        if !producer.is_empty() {
            result.insert("aigc_producer".into(), producer);
        }
        if let Some(signature) = check_tc260_signature(|key| aigc.get(key).map(String::as_str)) {
            result.insert("aigc_signature".into(), signature.describe());
        }
    }
    let app_scan = head_or_empty(path, HEAD_BYTES);
    let (provenance, generator) = app_evidence_bytes(&app_scan);
    if let Some(p) = provenance {
        set_default(
            &mut result,
            "app_provenance",
            format!("App export provenance ({p})"),
        );
    }
    if let Some(g) = generator {
        set_default(&mut result, "app_aigc", format!("App AIGC disclosure ({g})"));
    }
    let isobmff_head = is_isobmff(&app_scan[..app_scan.len().min(8)]);
    let sei = isobmff_head.then(|| isobmff::scan_video_sei(path, true).unwrap_or_default());
    let kling_in_scan = contains(&app_scan, KLING_SEI_TAG);
    if sei.as_ref().is_some_and(|s| !s.kling_tag_offsets.is_empty())
        || (sei.as_ref().is_none_or(|s| !s.has_parsed_tracks) && kling_in_scan)
    {
        set_default(
            &mut result,
            "kling_sei",
            "Kling AI encoder tag (H.264 SEI kling-ai)",
        );
    }
    if sei.as_ref().is_some_and(|s| !s.ai_sei_spans.is_empty()) {
        set_default(&mut result, "ai_sei", "AI provenance payload in H.264/HEVC SEI");
    }
    if sei.as_ref().is_some_and(|s| !s.av1_obu_patches.is_empty()) {
        set_default(&mut result, "ai_obu", "AI provenance payload in AV1 OBU_METADATA");
    }
    if isobmff_head && URN_C2PA.is_match(&app_scan) {
        set_default(
            &mut result,
            "c2pa_manifest",
            "fMP4 C2PA emsg segment provenance (urn:c2pa:)",
        );
    }
    if app_scan.starts_with(EBML_MAGIC) {
        for blob in c2pa_attachment_payloads(path) {
            for (key, value) in parse_c2pa_chunk(&blob).string_entries() {
                if key != "timestamps" && key != "timestamp" {
                    set_default(&mut result, key, value);
                }
            }
        }
        if !ebml::scan_av1_ai_obus(path, true).is_empty() {
            set_default(&mut result, "ai_obu", "AI provenance payload in AV1 OBU_METADATA");
        }
    }
    if xai_signature(path) {
        set_default(
            &mut result,
            "xai_signature",
            "xAI/Grok signature (UUID + Signature blob)",
        );
    }
    if let Some(system) = iptc_ai_system(path) {
        set_default(
            &mut result,
            "ai_system",
            format!("IPTC 2025.1 AI disclosure ({system})"),
        );
    }
    let xmp_packet = contains(&app_scan, b"<x:xmpmeta")
        || contains(&app_scan, b"<?xpacket")
        || contains(&app_scan, b"Iptc4xmpExt:");
    if xmp_packet && let Some(caps) = XMP_IPTC_SOURCE_TYPE.captures(&app_scan) {
        let token = caps[1].to_ascii_lowercase();
        let label = match token.as_slice() {
            b"compositewithtrainedalgorithmicmedia" => "compositeWithTrainedAlgorithmicMedia (AI-enhanced)",
            b"compositesynthetic" => "compositeSynthetic (AI-generated)",
            _ => "trainedAlgorithmicMedia (AI-generated)",
        };
        set_default(&mut result, "source_type", label);
    }
    if let Some(job) = huggingface_job(path) {
        set_default(&mut result, "huggingface_job", format!("Higgsfield job ({job})"));
    }
    if let Some(genai) = samsung_genai(path) {
        set_default(
            &mut result,
            "samsung_genai",
            format!("Samsung Galaxy AI editing marker (genAIType={genai})"),
        );
    }
    if !NON_IMAGE_MEDIA_EXTS.contains(&suffix_lower(path).as_str())
        && let Some(generator) = exif_generator(path)
    {
        set_default(&mut result, "exif_generator", truncate200(&generator));
    }
    result
}

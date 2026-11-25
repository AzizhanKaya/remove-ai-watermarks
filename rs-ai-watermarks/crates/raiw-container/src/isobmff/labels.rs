//! Label regions inside `moov` metadata lists: TC260, xAI signatures, generation keys, app exports.

use std::io;
use std::path::Path;
use std::sync::LazyLock;

use base64::Engine;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use indexmap::IndexMap;
use regex::bytes::Regex;

use super::boxes::{Span, is_isobmff};
use super::items::{MetadataItem, metadata_items};
use crate::hooks::MetadataHooks;
use crate::markers::{
    LOCAL_GENERATION_METADATA_KEYS, TOPAZ_AI_DESCRIPTION, TOPAZ_VIDEO_METADATA_KEYS, py_strip,
};
use crate::source::{FileSource, ReadAt};

pub(crate) const XAI_REGION_SENTINEL_START: &[u8] = b"\xa9cmt\x00";
pub(crate) const XAI_REGION_SENTINEL_END: &[u8] = b"\x00";

static XAI_SIGNATURE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?-u)\ASignature:[\t\n\x0B\x0C\r ]*([A-Za-z0-9+/=]{64,})[\t\n\x0B\x0C\r ]*\z")
        .expect("static pattern")
});

/// Python's `base64.b64decode(..., validate=True)`: canonical padding, lenient trailing bits.
const STRICT_BASE64: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    GeneralPurposeConfig::new()
        .with_decode_allow_trailing_bits(true)
        .with_decode_padding_mode(DecodePaddingMode::RequireCanonical),
);

/// A located value with the optional key name span blanked alongside it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelRegion {
    pub key_span: Option<Span>,
    pub value: Span,
}

/// Whether one comment item holds an xAI Grok Imagine `Signature: <base64>` blob.
pub fn is_xai_signature(item_type: &[u8], key_name: Option<&[u8]>, value: &[u8]) -> bool {
    let is_comment = match key_name {
        None => item_type == b"\xa9cmt",
        Some(name) => {
            let lowered = name.to_ascii_lowercase();
            lowered == b"comment" || lowered == b"com.apple.quicktime.comment"
        }
    };
    if !is_comment {
        return false;
    }
    let Some(blob) = XAI_SIGNATURE.captures(value).and_then(|caps| caps.get(1)) else {
        return false;
    };
    blob.len() % 4 == 0 && STRICT_BASE64.decode(blob.as_bytes()).is_ok()
}

/// Validated native TC260 entries: keyed `AIGC` items and keyless QuickTime values.
pub fn tc260_regions(items: &[MetadataItem], hooks: &impl MetadataHooks) -> Vec<(LabelRegion, Vec<u8>)> {
    items
        .iter()
        .filter(|item| {
            item.key_name.as_deref().is_none_or(|name| name == b"AIGC") && hooks.is_tc260_label(&item.value)
        })
        .map(|item| {
            (
                LabelRegion {
                    key_span: item.key_span,
                    value: item.value_span(),
                },
                item.value.clone(),
            )
        })
        .collect()
}

/// xAI signature comment items; a keyless item blanks its own type instead of a key name.
pub fn xai_regions(items: &[MetadataItem]) -> Vec<(LabelRegion, Vec<u8>)> {
    items
        .iter()
        .filter(|item| is_xai_signature(&item.item_type, item.key_name.as_deref(), &item.value))
        .map(|item| {
            let key_span = Some(item.key_span.unwrap_or(item.item_type_span));
            (
                LabelRegion {
                    key_span,
                    value: item.value_span(),
                },
                item.value.clone(),
            )
        })
        .collect()
}

/// Keyed generation parameters (prompt, workflow), Topaz keys and the Topaz description.
pub fn ai_metadata_regions(items: &[MetadataItem]) -> Vec<(LabelRegion, String, Vec<u8>)> {
    let mut regions = Vec::new();
    for item in items {
        let (Some(name_bytes), Some(key_span)) = (&item.key_name, item.key_span) else {
            continue;
        };
        let Ok(name) = std::str::from_utf8(name_bytes) else {
            continue;
        };
        let lowered = name.to_lowercase();
        let topaz_description = (lowered == "description" || lowered == "com.apple.quicktime.description")
            && py_strip(&item.value) == TOPAZ_AI_DESCRIPTION;
        if LOCAL_GENERATION_METADATA_KEYS.contains(&lowered.as_str())
            || TOPAZ_VIDEO_METADATA_KEYS.contains(&lowered.as_str())
            || topaz_description
        {
            let region = LabelRegion {
                key_span: Some(key_span),
                value: item.value_span(),
            };
            regions.push((region, name.to_owned(), item.value.clone()));
        }
    }
    regions
}

/// Values carrying an app-export provenance object (Dreamina `LvMetaInfo` and similar).
pub fn app_regions(items: &[MetadataItem], hooks: &impl MetadataHooks) -> Vec<LabelRegion> {
    items
        .iter()
        .filter(|item| hooks.is_app_export(&item.value))
        .map(|item| LabelRegion {
            key_span: item.key_span,
            value: item.value_span(),
        })
        .collect()
}

/// Exact audio creation declarations (`Made with Suno; created=...`).
pub fn audio_creation_regions(items: &[MetadataItem], hooks: &impl MetadataHooks) -> Vec<Span> {
    items
        .iter()
        .filter(|item| hooks.is_audio_creation(&item.value))
        .map(MetadataItem::value_span)
        .collect()
}

/// Open `path` and run `extract` over its metadata items, or nothing for a non-ISOBMFF file.
fn with_items<T: Default>(path: &Path, extract: impl FnOnce(&[MetadataItem]) -> T) -> io::Result<T> {
    let source = FileSource::open(path)?;
    let head = source.read_vec(0, 8)?;
    if !is_isobmff(&head) {
        return Ok(T::default());
    }
    Ok(extract(&metadata_items(&source)?))
}

/// Native TC260 `AIGC` values (`isobmff.tc260_aigc_payloads`); `Err` is the `strict=True` raise.
pub fn tc260_aigc_payloads(path: &Path, hooks: &impl MetadataHooks) -> io::Result<Vec<Vec<u8>>> {
    with_items(path, |items| {
        tc260_regions(items, hooks)
            .into_iter()
            .map(|(_, value)| value)
            .collect()
    })
}

/// Validated xAI Grok Imagine comment signatures (`isobmff.isobmff_xai_signature_payloads`).
pub fn xai_signature_payloads(path: &Path) -> io::Result<Vec<Vec<u8>>> {
    with_items(path, |items| {
        xai_regions(items).into_iter().map(|(_, value)| value).collect()
    })
}

pub fn has_xai_signature(path: &Path) -> io::Result<bool> {
    Ok(!xai_signature_payloads(path)?.is_empty())
}

/// Keyed AI-generation metadata (`isobmff.ai_metadata_tags`), values decoded lossily.
pub fn ai_metadata_tags(path: &Path) -> io::Result<IndexMap<String, String>> {
    with_items(path, |items| {
        ai_metadata_regions(items)
            .into_iter()
            .map(|(_, name, value)| (name, String::from_utf8_lossy(&value).into_owned()))
            .collect()
    })
}

/// Whether metadata lists hold a creation declaration, before the audio-only stream probe.
pub fn has_audio_creation_comment(path: &Path, hooks: &impl MetadataHooks) -> io::Result<bool> {
    let source = FileSource::open(path)?;
    Ok(!audio_creation_regions(&metadata_items(&source)?, hooks).is_empty())
}

/// Whether a metadata region (such as `scan_c2pa_region` output) carries an xAI signature.
pub fn xai_signature_in_region(data: &[u8]) -> bool {
    if !is_isobmff(data) {
        return false;
    }
    if metadata_items(data).is_ok_and(|items| !xai_regions(&items).is_empty()) {
        return true;
    }
    let finder = memchr::memmem::Finder::new(XAI_REGION_SENTINEL_START);
    let mut cursor = finder.find(data);
    while let Some(start) = cursor {
        let value_start = start + XAI_REGION_SENTINEL_START.len();
        if let Some(length) = memchr::memchr(XAI_REGION_SENTINEL_END[0], &data[value_start..])
            && is_xai_signature(b"\xa9cmt", None, &data[value_start..value_start + length])
        {
            return true;
        }
        cursor = finder.find(&data[start + 1..]).map(|next| start + 1 + next);
    }
    false
}

//! The single AI-EXIF rule set (`_ai_exif_targets`) and the EXIF generator candidates.

use crate::exif::{ExifDict, Ifd, tag};
use crate::pil::{Info, InfoValue};
use crate::py::latin1;
use crate::rules::app::{any, app_evidence_bytes};
use crate::rules::generator::structured_generator;
use crate::rules::keys::has_generator_token;
use crate::rules::tc260::is_aigc_value_bytes;
use crate::rules::xai::xai_signature_pair;

/// One AI-provenance EXIF value: `(ifd, tag, value bytes, tag name)`.
pub type ExifTarget = (Ifd, u16, Vec<u8>, &'static str);

/// `_ai_exif_targets(loaded)`, de-duplicated by `(ifd, tag)` in rule order.
pub fn ai_exif_targets(dict: &ExifDict) -> Vec<ExifTarget> {
    let mut targets: Vec<ExifTarget> = Vec::new();
    let mut add = |ifd: Ifd, number: u16, name: &'static str| {
        if let Some(value) = dict.bytes(ifd, number)
            && !targets.iter().any(|(i, t, _, _)| *i == ifd && *t == number)
        {
            targets.push((ifd, number, value.to_vec(), name));
        }
    };
    if xai_signature_pair(
        &dict.text(Ifd::Zeroth, tag::IMAGE_DESCRIPTION),
        &dict.text(Ifd::Zeroth, tag::ARTIST),
    ) {
        add(Ifd::Zeroth, tag::IMAGE_DESCRIPTION, "ImageDescription");
        add(Ifd::Zeroth, tag::ARTIST, "Artist");
    }
    let text_tags = [
        (Ifd::Zeroth, tag::SOFTWARE, "Software"),
        (Ifd::Zeroth, tag::MAKE, "Make"),
        (Ifd::Zeroth, tag::ARTIST, "Artist"),
        (Ifd::Zeroth, tag::IMAGE_DESCRIPTION, "ImageDescription"),
        (Ifd::Exif, tag::USER_COMMENT, "UserComment"),
    ];
    for (ifd, number, name) in text_tags {
        let text = dict.bytes(ifd, number).map(latin1).unwrap_or_default();
        if has_generator_token(&text.to_lowercase()) || structured_generator(&text).is_some() {
            add(ifd, number, name);
        }
    }
    if dict
        .bytes(Ifd::Zeroth, tag::IMAGE_DESCRIPTION)
        .is_some_and(is_aigc_value_bytes)
    {
        add(Ifd::Zeroth, tag::IMAGE_DESCRIPTION, "ImageDescription");
    }
    if dict
        .bytes(Ifd::Exif, tag::USER_COMMENT)
        .is_some_and(is_aigc_value_bytes)
    {
        add(Ifd::Exif, tag::USER_COMMENT, "UserComment");
    }
    for (ifd, number, name) in [
        (Ifd::Zeroth, tag::IMAGE_DESCRIPTION, "ImageDescription"),
        (Ifd::Exif, tag::USER_COMMENT, "UserComment"),
    ] {
        if dict
            .bytes(ifd, number)
            .is_some_and(|value| any(app_evidence_bytes(value)))
        {
            add(ifd, number, name);
        }
    }
    targets
}

/// `_scrub_ai_exif`: drop the targets in place, returning the removed tag names.
pub fn scrub_ai_exif(dict: &mut ExifDict) -> Vec<&'static str> {
    let targets = ai_exif_targets(dict);
    for (ifd, number, _, _) in &targets {
        dict.ifd_mut(*ifd).shift_remove(number);
    }
    targets.into_iter().map(|(_, _, _, name)| name).collect()
}

/// The value bytes `isobmff.blank_ai_exif_tokens` blanks (`{}` when the blob does not load).
pub fn ai_exif_values(exif: &[u8]) -> Vec<Vec<u8>> {
    crate::exif::load(exif)
        .map(|dict| {
            ai_exif_targets(&dict)
                .into_iter()
                .map(|(_, _, value, _)| value)
                .collect()
        })
        .unwrap_or_default()
}

/// `_exif_generator_candidates(info, exif_bytes)`.
pub fn exif_generator_candidates(info: &Info, exif_bytes: Option<&[u8]>) -> Vec<String> {
    let mut candidates: Vec<String> = ["Software", "Source", "Title", "Description"]
        .iter()
        .filter_map(|key| info.get(*key).and_then(InfoValue::as_str))
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect();
    if let Some(exif) = exif_bytes.filter(|b| !b.is_empty())
        && let Ok(dict) = crate::exif::load(exif)
    {
        for number in [tag::SOFTWARE, tag::MAKE, tag::ARTIST, tag::IMAGE_DESCRIPTION] {
            if let Some(value) = dict.bytes(Ifd::Zeroth, number) {
                candidates.push(latin1(value));
            }
        }
        if let Some(value) = dict.bytes(Ifd::Exif, tag::USER_COMMENT) {
            candidates.push(latin1(value));
        }
    }
    candidates
}

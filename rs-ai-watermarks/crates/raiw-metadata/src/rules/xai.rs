//! xAI / Grok signature pairs in EXIF, XMP, IPTC and PNG text.

use std::sync::LazyLock;

use raiw_container::isobmff::xai_signature_in_region;
use raiw_container::jpeg::app_payloads;
use raiw_container::markers::py_strip;
use regex::Regex;
use regex::bytes::Regex as BytesRegex;

use crate::py::{strip, utf8_replace};

fn bytes_re(pattern: &str) -> BytesRegex {
    BytesRegex::new(pattern).unwrap_or_else(|err| unreachable!("static regex: {err}"))
}

pub static XAI_SIGNATURE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^Signature:[\s\x1c-\x1f]*[A-Za-z0-9+/=]{64,}").unwrap_or_else(|err| unreachable!("{err}"))
});
pub static XAI_SIGNATURE_ANYWHERE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"Signature:[\s\x1c-\x1f]*[A-Za-z0-9+/=]{64,}").unwrap_or_else(|err| unreachable!("{err}"))
});
pub static UUID_FULL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
        .unwrap_or_else(|err| unreachable!("{err}"))
});
static TAG: LazyLock<BytesRegex> = LazyLock::new(|| bytes_re(r"(?-u)<[^>]+>"));
static RDF_DESCRIPTION: LazyLock<BytesRegex> =
    LazyLock::new(|| bytes_re(r"(?is-u)<rdf:Description\b[^>]*>.*?</rdf:Description>"));
static XMP_DESCRIPTION: LazyLock<BytesRegex> = LazyLock::new(|| {
    bytes_re(r#"(?is-u)(?:<dc:description\b[^>]*>(.*?)</dc:description>|\bdc:description=["']([^"']*)["'])"#)
});
static XMP_CREATOR: LazyLock<BytesRegex> = LazyLock::new(|| {
    bytes_re(r#"(?is-u)(?:<dc:creator\b[^>]*>(.*?)</dc:creator>|\bdc:creator=["']([^"']*)["'])"#)
});
static CAPTION_LABEL: LazyLock<BytesRegex> =
    LazyLock::new(|| bytes_re(r"(?is-u)Caption-Abstract\x00([^\x00]{1,4096})"));
static BYLINE_LABEL: LazyLock<BytesRegex> =
    LazyLock::new(|| bytes_re(r"(?is-u)By-line\x00([^\x00]{1,4096})"));

/// `xai_signature_pair(description, artist)`.
pub fn xai_signature_pair(description: &str, artist: &str) -> bool {
    XAI_SIGNATURE.is_match(description) && UUID_FULL.is_match(artist)
}

/// `_decoded_xai_signature_pair`: tags removed, both sides stripped and decoded.
pub fn decoded_pair(description: &[u8], creator: &[u8]) -> bool {
    let clean = |raw: &[u8]| {
        let untagged = TAG.replace_all(raw, &b""[..]);
        strip(&utf8_replace(py_strip(&untagged))).to_owned()
    };
    xai_signature_pair(&clean(description), &clean(creator))
}

fn first_group<'a>(caps: &regex::bytes::Captures<'a>) -> &'a [u8] {
    caps.get(1)
        .or_else(|| caps.get(2))
        .map_or(&[][..], |m| m.as_bytes())
}

fn xmp_pair(data: &[u8]) -> bool {
    RDF_DESCRIPTION.find_iter(data).any(|block| {
        let block = block.as_bytes();
        let descriptions: Vec<&[u8]> = XMP_DESCRIPTION
            .captures_iter(block)
            .map(|c| first_group(&c))
            .collect();
        if descriptions.is_empty() {
            return false;
        }
        let creators: Vec<&[u8]> = XMP_CREATOR
            .captures_iter(block)
            .map(|c| first_group(&c))
            .collect();
        descriptions
            .iter()
            .any(|d| creators.iter().any(|c| decoded_pair(d, c)))
    })
}

/// `_iptc_dataset_values`: bounded IPTC application-record values for one dataset.
pub fn iptc_dataset_values(data: &[u8], dataset: u8) -> Vec<&[u8]> {
    let marker = [0x1C, 0x02, dataset];
    let finder = memchr::memmem::Finder::new(&marker);
    let mut values = Vec::new();
    let mut start = 0;
    while let Some(found) = finder.find(&data[start..]) {
        let index = start + found;
        let length_start = index + 3;
        if length_start + 2 > data.len() {
            break;
        }
        let length = usize::from(u16::from_be_bytes([data[length_start], data[length_start + 1]]));
        let value_start = length_start + 2;
        let value_end = value_start + length;
        if value_end <= data.len() && length <= 4096 {
            values.push(&data[value_start..value_end]);
        }
        start = length_start;
    }
    values
}

fn iptc_pair(data: &[u8]) -> bool {
    let mut descriptions = iptc_dataset_values(data, 120);
    let mut creators = iptc_dataset_values(data, 80);
    descriptions.extend(
        CAPTION_LABEL
            .captures_iter(data)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_bytes()),
    );
    creators.extend(
        BYLINE_LABEL
            .captures_iter(data)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_bytes()),
    );
    descriptions
        .iter()
        .any(|d| creators.iter().any(|c| decoded_pair(d, c)))
}

/// `xai_signature_in_metadata`: a paired record inside one APP segment or RDF description.
pub fn xai_signature_in_metadata(data: &[u8]) -> bool {
    if xai_signature_in_region(data) {
        return true;
    }
    let payloads = app_payloads(data);
    if !payloads.is_empty() {
        return payloads.iter().any(|(marker, payload)| {
            if *marker == 0xE1 {
                xmp_pair(payload)
            } else {
                iptc_pair(payload)
            }
        });
    }
    xmp_pair(data) || iptc_pair(data)
}

/// `_xai_text_pair_member`: one PNG text item of an already proven pair.
pub fn text_pair_member(key: &str, value: &str) -> bool {
    let normalized: String = key
        .to_lowercase()
        .chars()
        .filter(|c| !matches!(c, '-' | '_'))
        .collect();
    match normalized.as_str() {
        "description" | "imagedescription" | "captionabstract" => XAI_SIGNATURE_ANYWHERE.is_match(value),
        "author" | "artist" | "creator" | "byline" => UUID_FULL.is_match(strip(value)),
        _ => false,
    }
}

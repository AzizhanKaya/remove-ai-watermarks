//! China TC260 AIGC label parsing (`parse_tc260_aigc_json`, `aigc_label_from_metadata`).

use std::sync::LazyLock;

use indexmap::IndexMap;
use memchr::memmem;
use raiw_c2pa::text::casefold;
use raiw_container::markers::{MAX_TC260_VALUE_BYTES, rstrip_bytes};
use regex::bytes::Regex;

use super::html::unescape;
use crate::py::{latin1, utf8_replace};
use crate::pyjson::{self, PyValue};

/// A decoded TC260 label: canonical field name to its string rendering.
pub type Tc260Label = IndexMap<String, String>;

pub const TC260_AIGC_FIELDS: &[&str] = &[
    "Label",
    "ContentProducer",
    "ProduceID",
    "ContentPropagator",
    "PropagateID",
    "ReservedCode1",
    "ReservedCode2",
    "ServiceProvider",
    "ServiceUser",
];

static XMP_LABEL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s-u)<TC260:AIGC>(.*?)</TC260:AIGC>|TC260:AIGC\s*=\s*"(.*?)""#)
        .unwrap_or_else(|err| unreachable!("static regex: {err}"))
});
static AUDIO_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s-u)AIGC(?:=|\x00|[\x00-\xff]{4})\{")
        .unwrap_or_else(|err| unreachable!("static regex: {err}"))
});

/// `_tc260_fields`: canonical keys, nested values as compact JSON.
pub fn tc260_fields(parsed: &[(String, PyValue)]) -> Tc260Label {
    let mut out = IndexMap::new();
    for (key, item) in parsed {
        let folded = casefold(key);
        let canonical = TC260_AIGC_FIELDS
            .iter()
            .find(|field| casefold(field) == folded)
            .map_or_else(|| key.clone(), |field| (*field).to_owned());
        let value = if item.is_container() {
            item.dumps_compact()
        } else {
            item.py_str()
        };
        out.insert(canonical, value);
    }
    out
}

fn has_tc260_field(fields: &Tc260Label) -> bool {
    TC260_AIGC_FIELDS.iter().any(|field| fields.contains_key(*field))
}

/// `parse_tc260_aigc_json`: a bounded JSON object carrying a normative field.
pub fn parse_tc260_aigc_json(value: &[u8]) -> Option<Tc260Label> {
    if value.len() as u64 > MAX_TC260_VALUE_BYTES {
        return None;
    }
    let text = std::str::from_utf8(rstrip_bytes(value, b"\x00 ")).ok()?;
    let parsed = pyjson::loads(text)?;
    let fields = tc260_fields(parsed.as_dict()?);
    has_tc260_field(&fields).then_some(fields)
}

fn parse_loose(text: &str) -> Option<Tc260Label> {
    let parsed = pyjson::loads(text)?;
    Some(tc260_fields(parsed.as_dict()?))
}

/// `aigc_label_from_metadata(data, candidates)`.
pub fn aigc_label_from_metadata<S: AsRef<str>>(data: &[u8], candidates: &[S]) -> Option<Tc260Label> {
    for candidate in candidates {
        if let Some(label) = parse_tc260_aigc_json(candidate.as_ref().as_bytes()) {
            return Some(label);
        }
    }
    if let Some(caps) = XMP_LABEL.captures(data) {
        let body = caps
            .get(1)
            .or_else(|| caps.get(2))
            .map_or(&[][..], |m| m.as_bytes());
        return parse_loose(&unescape(&utf8_replace(body)));
    }
    for needle in [&b"\"AIGC\""[..], b"AIGC{"] {
        let Some(start) = memmem::find(data, needle) else {
            continue;
        };
        let Some(brace) = memchr::memchr(b'{', &data[start..]).map(|i| start + i) else {
            continue;
        };
        let Some((_, end)) = pyjson::raw_decode_latin1(data, brace) else {
            continue;
        };
        if let Some(label) = parse_tc260_aigc_json(latin1(&data[brace..end]).as_bytes()) {
            return Some(label);
        }
    }
    for found in AUDIO_PREFIX.find_iter(data) {
        let brace = found.end() - 1;
        let Some((_, end)) = pyjson::raw_decode_latin1(data, brace) else {
            continue;
        };
        if let Some(label) = parse_tc260_aigc_json(utf8_replace(&data[brace..end]).as_bytes()) {
            return Some(label);
        }
    }
    None
}

/// `_is_aigc_exif_value` for bytes: the `AIGC` marker and a parsable TC260 block.
pub fn is_aigc_value_bytes(raw: &[u8]) -> bool {
    memmem::find(raw, b"AIGC").is_some() && aigc_label_from_metadata::<&str>(raw, &[]).is_some()
}

/// `_is_aigc_exif_value` for a `str` (latin-1 encoded, unencodable characters dropped).
pub fn is_aigc_value_str(raw: &str) -> bool {
    is_aigc_value_bytes(&crate::py::latin1_encode_lossy(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_forms() {
        let label = aigc_label_from_metadata::<&str>(b"x<TC260:AIGC>{&quot;Label&quot;:1}</TC260:AIGC>", &[])
            .unwrap();
        assert_eq!(label["Label"], "1");
        let raw = br#"junk "AIGC":{"label":"1","ContentProducer":"p","ReservedCode1":{"a":[1,2]}} tail"#;
        let label = aigc_label_from_metadata::<&str>(raw, &[]).unwrap();
        assert_eq!(label["Label"], "1");
        assert_eq!(label["ReservedCode1"], r#"{"a":[1,2]}"#);
        assert!(aigc_label_from_metadata::<&str>(br#""AIGC":{"x":1}"#, &[]).is_none());
        let wav = b"AIGC\x10\x00\x00\x00{\"Label\":\"1\"}";
        assert!(aigc_label_from_metadata::<&str>(wav, &[]).is_some());
    }
}

//! Validation and rendering behind `sync-c2pa-soft-bindings` (port of `scripts/sync_c2pa_soft_bindings.py`).
//!
//! The binary fetches the upstream list; parsing and rendering stay pure so they are tested here.

use std::fmt::Write as _;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use super::SoftBindingKind;
use crate::text::is_printable;

pub const DEFAULT_SOURCE_URL: &str = "https://raw.githubusercontent.com/c2pa-org/softbinding-algorithm-list/main/softbinding-algorithm-list.json";
pub const PINNED_SOURCE_URL: &str = "https://raw.githubusercontent.com/c2pa-org/softbinding-algorithm-list/{revision}/softbinding-algorithm-list.json";
pub const DEFAULT_REVISION_URL: &str = "https://api.github.com/repos/c2pa-org/softbinding-algorithm-list/commits?path=softbinding-algorithm-list.json&per_page=1";
pub const SOURCE_LICENSE: &str = "CC BY 4.0";
pub const SOURCE_LICENSE_URL: &str = "https://creativecommons.org/licenses/by/4.0/";

const DECODED_MEDIA_TYPES: [&str; 6] = ["application", "audio", "image", "model", "text", "video"];

static ALGORITHM: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\A[A-Za-z]{2,63}(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+\.?\z")
        .expect("static pattern")
});
static REVISION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\A[0-9a-f]{40}\z").expect("static pattern"));

/// An upstream entry the registry contract rejects.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct RegistryError(pub String);

fn invalid(message: impl Into<String>) -> RegistryError {
    RegistryError(message.into())
}

/// The runtime subset of one validated upstream entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryRow {
    pub identifier: u16,
    pub algorithm: String,
    pub kind: SoftBindingKind,
    pub decoded_media_types: Vec<String>,
    pub encoded_media_types: Vec<String>,
    pub display_label: String,
    pub date_entered: String,
    pub resolution_apis: Vec<String>,
    pub deprecated: bool,
}

fn require_text<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a str, RegistryError> {
    match value.and_then(Value::as_str) {
        Some(text) if !text.is_empty() && is_printable(text) => Ok(text),
        _ => Err(invalid(format!("{field} must be a non-empty printable string"))),
    }
}

fn string_array(value: Option<&Value>, field: &str) -> Result<Vec<String>, RegistryError> {
    let Some(value) = value else { return Ok(Vec::new()) };
    let items = value
        .as_array()
        .ok_or_else(|| invalid(format!("{field} must be a string array")))?;
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid(format!("{field} must be a string array")))
        })
        .collect()
}

/// `urlparse(value)` has an http(s) scheme and a network location.
pub fn is_http_url(value: &str) -> bool {
    let Some((scheme, rest)) = value.split_once(':') else {
        return false;
    };
    let scheme = scheme.to_ascii_lowercase();
    let Some(after) = rest.strip_prefix("//") else {
        return false;
    };
    let netloc = after.split(['/', '?', '#']).next().unwrap_or_default();
    (scheme == "http" || scheme == "https") && !netloc.is_empty()
}

fn two_digits(text: &str, max: u32) -> bool {
    text.len() == 2 && text.bytes().all(|b| b.is_ascii_digit()) && text.parse::<u32>().is_ok_and(|v| v <= max)
}

/// The `datetime.fromisoformat` shapes the registry uses: a date, optional time and offset.
pub fn is_iso_datetime(value: &str) -> bool {
    let value = value.replace('Z', "+00:00");
    let (date, rest) = value.split_at(value.len().min(10));
    let parts: Vec<&str> = date.split('-').collect();
    let [year, month, day] = parts.as_slice() else {
        return false;
    };
    let numeric = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    if year.len() != 4 || !numeric(year) || !two_digits(month, 12) || !two_digits(day, 31) {
        return false;
    }
    let (Ok(year), Ok(month), Ok(day)) = (year.parse::<u32>(), month.parse::<u32>(), day.parse::<u32>())
    else {
        return false;
    };
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if month == 0 || day == 0 || day > days[month as usize - 1] {
        return false;
    }
    let mut chars = rest.chars();
    if chars.next().is_none() {
        return true;
    }
    let time = chars.as_str();
    let (clock, offset) = match time.find(['+', '-']) {
        Some(index) => (&time[..index], Some(&time[index + 1..])),
        None => (time, None),
    };
    let (clock, fraction) = clock.split_once('.').map_or((clock, None), |(c, f)| (c, Some(f)));
    let fields: Vec<&str> = clock.split(':').collect();
    let limits = [23, 59, 59];
    if fields.is_empty() || fields.len() > 3 || fields.iter().zip(limits).any(|(f, max)| !two_digits(f, max))
    {
        return false;
    }
    if let Some(fraction) = fraction
        && (fields.len() != 3 || !(1..=6).contains(&fraction.len()) || !numeric(fraction))
    {
        return false;
    }
    offset.is_none_or(|offset| {
        let parts: Vec<&str> = offset.split(':').collect();
        matches!(parts.as_slice(), [h, m] if two_digits(h, 23) && two_digits(m, 59))
    })
}

fn media_types(
    entry: &Map<String, Value>,
    identifier: u16,
) -> Result<(Vec<String>, Vec<String>), RegistryError> {
    let decoded = string_array(
        entry.get("decodedMediaTypes"),
        &format!("entry {identifier}: decodedMediaTypes"),
    )?;
    let encoded = string_array(
        entry.get("encodedMediaTypes"),
        &format!("entry {identifier}: encodedMediaTypes"),
    )?;
    if decoded.is_empty() && encoded.is_empty() {
        return Err(invalid(format!(
            "entry {identifier}: one media-type array is required"
        )));
    }
    let mut unknown: Vec<&String> = decoded
        .iter()
        .filter(|m| !DECODED_MEDIA_TYPES.contains(&m.as_str()))
        .collect();
    if !unknown.is_empty() {
        unknown.sort();
        unknown.dedup();
        let listed = unknown
            .iter()
            .map(|m| format!("'{m}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(invalid(format!(
            "entry {identifier}: unsupported decoded media types: [{listed}]"
        )));
    }
    if encoded
        .iter()
        .any(|m| !m.contains('/') || m.chars().any(char::is_whitespace))
    {
        return Err(invalid(format!("entry {identifier}: invalid encoded media type")));
    }
    Ok((decoded, encoded))
}

/// The official description's compact first sentence.
pub fn display_label(description: &str) -> String {
    let first = description.split_once(". ").map_or(description, |(head, _)| head);
    first.trim_end_matches('.').to_owned()
}

fn parse_entry(raw: &Value) -> Result<RegistryRow, RegistryError> {
    let entry = raw
        .as_object()
        .ok_or_else(|| invalid("every registry entry must be an object"))?;
    let identifier = entry
        .get("identifier")
        .and_then(Value::as_u64)
        .and_then(|id| u16::try_from(id).ok())
        .ok_or_else(|| invalid("identifier must be an integer from 0 through 65535"))?;
    let algorithm = require_text(entry.get("alg"), &format!("entry {identifier}: alg"))?;
    if !ALGORITHM.is_match(algorithm) {
        return Err(invalid(format!("entry {identifier}: invalid alg '{algorithm}'")));
    }
    let kind = match entry.get("type").and_then(Value::as_str) {
        Some("watermark") => SoftBindingKind::Watermark,
        Some("fingerprint") => SoftBindingKind::Fingerprint,
        other => return Err(invalid(format!("entry {identifier}: invalid type {other:?}"))),
    };
    let (decoded, encoded) = media_types(entry, identifier)?;
    let metadata = entry
        .get("entryMetadata")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(format!("entry {identifier}: entryMetadata must be an object")))?;
    let description = require_text(
        metadata.get("description"),
        &format!("entry {identifier}: description"),
    )?;
    let date_entered = require_text(
        metadata.get("dateEntered"),
        &format!("entry {identifier}: dateEntered"),
    )?;
    if !is_iso_datetime(date_entered) {
        return Err(invalid(format!(
            "entry {identifier}: invalid dateEntered '{date_entered}'"
        )));
    }
    let contact = require_text(metadata.get("contact"), &format!("entry {identifier}: contact"))?;
    if !contact.contains('@') || contact.chars().any(char::is_whitespace) {
        return Err(invalid(format!("entry {identifier}: invalid contact")));
    }
    let info = require_text(
        metadata.get("informationalUrl"),
        &format!("entry {identifier}: informationalUrl"),
    )?;
    if !is_http_url(info) {
        return Err(invalid(format!("entry {identifier}: invalid informationalUrl")));
    }
    let apis_field = format!("entry {identifier}: softBindingResolutionApis");
    let resolution_apis = string_array(entry.get("softBindingResolutionApis"), &apis_field)?;
    if resolution_apis.iter().any(|url| !is_http_url(url)) {
        return Err(invalid(format!("{apis_field} contains an invalid URL")));
    }
    let deprecated = match entry.get("deprecated") {
        None => false,
        Some(Value::Bool(flag)) => *flag,
        Some(_) => {
            return Err(invalid(format!(
                "entry {identifier}: deprecated must be a boolean"
            )));
        }
    };
    Ok(RegistryRow {
        identifier,
        algorithm: algorithm.to_owned(),
        kind,
        decoded_media_types: decoded,
        encoded_media_types: encoded,
        display_label: display_label(description),
        date_entered: date_entered.to_owned(),
        resolution_apis,
        deprecated,
    })
}

/// Validate the upstream shape and return its normalized runtime rows.
pub fn parse_registry(payload: &Value) -> Result<Vec<RegistryRow>, RegistryError> {
    let entries = payload
        .as_array()
        .ok_or_else(|| invalid("registry root must be an array"))?;
    let rows = entries.iter().map(parse_entry).collect::<Result<Vec<_>, _>>()?;
    if rows
        .windows(2)
        .any(|pair| pair[0].identifier > pair[1].identifier)
    {
        return Err(invalid("registry entries must be ordered by identifier"));
    }
    if rows
        .windows(2)
        .any(|pair| pair[0].identifier == pair[1].identifier)
    {
        return Err(invalid("registry identifiers must be unique"));
    }
    let mut algorithms: Vec<&str> = rows.iter().map(|row| row.algorithm.as_str()).collect();
    algorithms.sort_unstable();
    if algorithms.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid("registry algorithms must be unique"));
    }
    Ok(rows)
}

/// Whether `revision` is a 40-character commit SHA.
pub fn is_commit_sha(revision: &str) -> bool {
    REVISION.is_match(revision)
}

fn string_slice(values: &[String]) -> String {
    let items: Vec<String> = values.iter().map(|value| format!("{value:?}")).collect();
    format!("&[{}]", items.join(", "))
}

/// Render the deterministic generated Rust table.
pub fn render_module(rows: &[RegistryRow], source: &str, revision: &str) -> String {
    let mut out = String::new();
    let header = [
        "//! Generated C2PA soft-binding registry; do not edit by hand.".to_owned(),
        "//!".to_owned(),
        format!("//! Source: {source}"),
        format!("//! Revision: {revision}"),
        format!("//! Source license: {SOURCE_LICENSE} ({SOURCE_LICENSE_URL})"),
        "//! Changes: validated, ordered, and reduced to fields used at runtime; descriptions".to_owned(),
        "//! are shortened to their first sentence for display.".to_owned(),
        "//! Refresh: `cargo run -p raiw-c2pa --features sync --bin sync-c2pa-soft-bindings`.".to_owned(),
    ];
    for line in header {
        out.push_str(&line);
        out.push('\n');
    }
    out.push('\n');
    out.push_str(
        "use super::SoftBindingAlgorithm;\nuse super::SoftBindingKind::{Fingerprint, Watermark};\n\n",
    );
    let _ = writeln!(out, "pub const SOURCE_URL: &str = {source:?};");
    let _ = writeln!(out, "pub const SOURCE_REVISION: &str = {revision:?};");
    let _ = writeln!(out, "pub const SOURCE_LICENSE: &str = {SOURCE_LICENSE:?};\n");
    out.push_str("pub const REGISTRY: &[SoftBindingAlgorithm] = &[\n");
    for row in rows {
        let kind = match row.kind {
            SoftBindingKind::Watermark => "Watermark",
            SoftBindingKind::Fingerprint => "Fingerprint",
        };
        out.push_str("    SoftBindingAlgorithm {\n");
        let _ = writeln!(out, "        identifier: {},", row.identifier);
        let _ = writeln!(out, "        algorithm: {:?},", row.algorithm);
        let _ = writeln!(out, "        kind: {kind},");
        let _ = writeln!(
            out,
            "        decoded_media_types: {},",
            string_slice(&row.decoded_media_types)
        );
        let _ = writeln!(
            out,
            "        encoded_media_types: {},",
            string_slice(&row.encoded_media_types)
        );
        let _ = writeln!(out, "        display_label: {:?},", row.display_label);
        let _ = writeln!(out, "        date_entered: {:?},", row.date_entered);
        let _ = writeln!(
            out,
            "        resolution_apis: {},",
            string_slice(&row.resolution_apis)
        );
        let _ = writeln!(out, "        deprecated: {},", row.deprecated);
        out.push_str("    },\n");
    }
    out.push_str("];\n");
    out
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn entry(identifier: u64, alg: &str) -> Value {
        json!({
            "identifier": identifier,
            "alg": alg,
            "type": "watermark",
            "decodedMediaTypes": ["image"],
            "encodedMediaTypes": ["image/png"],
            "entryMetadata": {
                "description": "Example watermark. Second sentence.",
                "dateEntered": "2024-05-17T17:00:00.000Z",
                "contact": "a@example.com",
                "informationalUrl": "https://example.com",
            },
        })
    }

    #[test]
    fn accepts_encoded_media_and_shortens_the_label() {
        let rows = parse_registry(&json!([entry(1, "com.example.mark")])).unwrap();
        assert_eq!(rows[0].display_label, "Example watermark");
        assert_eq!(rows[0].encoded_media_types, ["image/png"]);
        let rendered = render_module(&rows, DEFAULT_SOURCE_URL, "abc");
        assert!(rendered.contains("Source license: CC BY 4.0"));
        assert!(rendered.contains("algorithm: \"com.example.mark\","));
    }

    #[test]
    fn rejects_duplicates_order_and_bad_fields() {
        let duplicate = parse_registry(&json!([
            entry(1, "com.example.mark"),
            entry(2, "com.example.mark")
        ]));
        assert_eq!(
            duplicate.unwrap_err().to_string(),
            "registry algorithms must be unique"
        );
        let unordered = parse_registry(&json!([entry(2, "com.a.b"), entry(1, "com.c.d")]));
        assert_eq!(
            unordered.unwrap_err().to_string(),
            "registry entries must be ordered by identifier"
        );
        assert!(parse_registry(&json!([entry(1, "nodots")])).is_err());
        assert!(parse_registry(&json!([entry(70000, "com.a.b")])).is_err());
        assert!(parse_registry(&json!({})).is_err());
    }

    #[test]
    fn iso_dates_and_urls() {
        assert!(is_iso_datetime("2024-05-17T17:00:00.000Z"));
        assert!(is_iso_datetime("2024-05-17T16:00:00Z"));
        assert!(is_iso_datetime("2024-02-29"));
        assert!(!is_iso_datetime("2023-02-29"));
        assert!(!is_iso_datetime("2024-13-01T00:00:00Z"));
        assert!(is_http_url("https://example.com/x"));
        assert!(!is_http_url("ftp://example.com"));
        assert!(!is_http_url("https:///path"));
    }
}

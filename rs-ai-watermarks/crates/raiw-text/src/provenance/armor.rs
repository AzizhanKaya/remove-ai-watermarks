//! C2PA 2.3/2.4 Appendix A.9: `-----BEGIN C2PA MANIFEST-----` ASCII armor.

use base64::Engine as _;
use base64::engine::DecodePaddingMode;
use base64::engine::general_purpose::{GeneralPurpose, GeneralPurposeConfig};
use fancy_regex::Captures;

use crate::pyre::{Flags, PyRegex, group, py_regex, whole};
use crate::spans::{SpanKind, split_protected_text_spans};
use crate::unicode;

pub const C2PA_ASCII_ARMOR_BEGIN: &str = "-----BEGIN C2PA MANIFEST-----";
pub const C2PA_ASCII_ARMOR_END: &str = "-----END C2PA MANIFEST-----";

/// The body of an armored block: inline JUMBF bytes or an external URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestSource<'a> {
    Bytes(&'a [u8]),
    Uri(&'a str),
}

/// Host comment syntax wrapped around the armored block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CommentStyle {
    #[default]
    None,
    Html,
    Hash,
}

/// Python's `base64.b64decode(..., validate=True)`.
pub(crate) static STRICT_BASE64: GeneralPurpose = GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    GeneralPurposeConfig::new()
        .with_decode_allow_trailing_bits(true)
        .with_decode_padding_mode(DecodePaddingMode::RequireCanonical),
);

/// Format an Appendix A.9 armored manifest block.
pub fn encode_c2pa_ascii_armor(manifest: &ManifestSource<'_>, comment_style: CommentStyle) -> String {
    let body = match manifest {
        ManifestSource::Bytes(bytes) => format!(
            "data:application/c2pa;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        ),
        ManifestSource::Uri(uri) => unicode::strip(uri).to_owned(),
    };
    let block = format!("{C2PA_ASCII_ARMOR_BEGIN}\n{body}\n{C2PA_ASCII_ARMOR_END}");
    match comment_style {
        CommentStyle::Html => format!("<!--\n{block}\n-->\n"),
        CommentStyle::Hash => {
            let lines: Vec<String> = unicode::splitlines(&block)
                .into_iter()
                .map(|line| format!("# {line}"))
                .collect();
            lines.join("\n") + "\n"
        }
        CommentStyle::None => block + "\n",
    }
}

/// Inline payload of an armored body: `(bytes, uri)`.
pub(crate) fn decode_inline_c2pa_payload(raw_body: &str) -> (Option<Vec<u8>>, Option<String>) {
    let mut joined = String::new();
    for line in unicode::splitlines(raw_body) {
        let mut stripped = unicode::strip(line);
        if stripped.starts_with('#') {
            stripped = unicode::strip(stripped.trim_start_matches('#'));
        } else if let Some(rest) = stripped.strip_prefix("//") {
            stripped = unicode::strip(rest);
        } else if stripped.starts_with('*') {
            stripped = unicode::strip(stripped.trim_start_matches('*'));
        }
        joined.push_str(stripped);
    }
    if joined.is_empty() {
        return (None, None);
    }
    if joined.starts_with("http://") || joined.starts_with("https://") || joined.starts_with("urn:") {
        return (None, Some(joined));
    }
    let prefix = "data:application/c2pa;base64,";
    let has_prefix = joined
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix));
    let encoded = if has_prefix {
        &joined[prefix.len()..]
    } else {
        joined.as_str()
    };
    match STRICT_BASE64.decode(encoded) {
        Ok(decoded) if decoded.is_empty() => (None, None),
        Ok(decoded) => (Some(decoded), None),
        Err(_) => (None, Some(joined.clone())),
    }
}

py_regex!(
    COMMENT_RE,
    r"(?:<!--[ \t]*\r?\n?)?(?:^[ \t]*(?:#|//|\*)?[ \t]*)?-----BEGIN C2PA MANIFEST-----[ \t]*\r?\n(?P<body>.*?)(?:^[ \t]*(?:#|//|\*)?[ \t]*)?-----END C2PA MANIFEST-----[ \t]*(?:\r?\n?[ \t]*-->)?[ \t]*(?:\r?\n)?",
    Flags {
        ignore_case: false,
        multiline: true,
        dotall: true
    }
);

const URI_BODY: &str = r"(?:https?://|urn:|data:application/c2pa;base64,)[^\s<>]+?";

// The exclusion range includes the preceding line ending for a manifest at EOF.
py_regex!(
    SINGLE_LINE_RE,
    format!(
        r"(?:(?:\r?\n)(?=[^\r\n]*(?:\r?\n)?\Z)|^)[ \t]*(?:<!--[ \t]+-----BEGIN C2PA MANIFEST-----[ \t]+(?P<html>{URI_BODY})[ \t]+-----END C2PA MANIFEST-----[ \t]+-->|/\*[ \t]+-----BEGIN C2PA MANIFEST-----[ \t]+(?P<css>{URI_BODY})[ \t]+-----END C2PA MANIFEST-----[ \t]+\*/|(?://|--|[#;%]|NOTE)[ \t]+-----BEGIN C2PA MANIFEST-----[ \t]+(?P<plain>{URI_BODY})[ \t]+-----END C2PA MANIFEST-----)[ \t]*(?:\r?\n|\Z)"
    ),
    Flags::M
);

/// What the stripped blocks declared.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AsciiArmorMatch {
    pub manifest_bytes: Option<Vec<u8>>,
    pub manifest_uri: Option<String>,
    pub count: usize,
}

impl AsciiArmorMatch {
    fn record(&mut self, body: &str) {
        self.count += 1;
        let (decoded, uri) = decode_inline_c2pa_payload(body);
        if decoded.is_some() && self.manifest_bytes.is_none() {
            self.manifest_bytes = decoded;
        }
        if uri.is_some() && self.manifest_uri.is_none() {
            self.manifest_uri = uri;
        }
    }
}

/// Substitute matches that do not overlap a fenced code span.
fn sub_outside_fences(
    pattern: &PyRegex,
    content: &str,
    mut replace: impl FnMut(&Captures<'_, str>) -> String,
) -> String {
    if !content.contains("```") && !content.contains("~~~") {
        return pattern.sub_with(content, replace).into_owned();
    }
    let mut fences = Vec::new();
    let mut start = 0;
    for span in split_protected_text_spans(content) {
        let end = start + span.text.len();
        if span.kind == SpanKind::FencedCode {
            fences.push((start, end));
        }
        start = end;
    }
    let mut remaining = fences.into_iter();
    let mut fence = remaining.next();
    pattern
        .sub_with(content, |caps| {
            let span = whole(caps);
            while fence.is_some_and(|(_, end)| end <= span.start) {
                fence = remaining.next();
            }
            if fence.is_some_and(|(start, _)| start < span.end) {
                content[span.range()].to_owned()
            } else {
                replace(caps)
            }
        })
        .into_owned()
}

/// Detect and strip Appendix A.9 blocks outside fenced code examples.
pub fn strip_c2pa_ascii_armor(text: &str) -> (String, Option<AsciiArmorMatch>) {
    if !text.contains(C2PA_ASCII_ARMOR_BEGIN) {
        return (text.to_owned(), None);
    }
    let mut found = AsciiArmorMatch::default();
    let cleaned = sub_outside_fences(&SINGLE_LINE_RE, text, |caps| {
        let body = ["html", "css", "plain"]
            .iter()
            .find_map(|name| group(caps, name))
            .unwrap_or("");
        if body.starts_with("data:") && decode_inline_c2pa_payload(body).0.is_none() {
            return caps.get(0).map_or_else(String::new, |m| m.as_str().to_owned());
        }
        found.record(body);
        String::new()
    });
    let cleaned = sub_outside_fences(&COMMENT_RE, &cleaned, |caps| {
        found.record(group(caps, "body").unwrap_or(""));
        String::new()
    });
    if found.count == 0 {
        return (text.to_owned(), None);
    }
    (cleaned, Some(found))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_line_forms_are_removed_with_their_line_ending() {
        let comment =
            "# -----BEGIN C2PA MANIFEST----- https://example.com/m.c2pa -----END C2PA MANIFEST-----";
        let (clean, found) = strip_c2pa_ascii_armor(&format!("{comment}\nHello\n"));
        assert_eq!(clean, "Hello\n");
        assert_eq!(
            found.unwrap().manifest_uri.as_deref(),
            Some("https://example.com/m.c2pa")
        );
        let html =
            "<!-- -----BEGIN C2PA MANIFEST----- https://example.com/m.c2pa -----END C2PA MANIFEST----- -->";
        assert_eq!(
            strip_c2pa_ascii_armor(&format!("<root/>\r\n\r\n{html}")).0,
            "<root/>\r\n"
        );
    }

    #[test]
    fn strict_base64() {
        assert_eq!(
            decode_inline_c2pa_payload("data:application/c2pa;base64,3q2+7w=="),
            (Some(vec![0xde, 0xad, 0xbe, 0xef]), None)
        );
        assert_eq!(decode_inline_c2pa_payload("abc").1.as_deref(), Some("abc"));
    }
}

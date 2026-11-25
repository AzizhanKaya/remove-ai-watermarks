//! C2PA 2.3/2.4 Appendix A.7: `<script type="application/c2pa">` and `<link rel="c2pa-manifest">`.

use crate::provenance::armor::decode_inline_c2pa_payload;
use crate::pyre::{Flags, Span, group, py_regex, whole};

py_regex!(
    HEAD_RE,
    r"<head\b[^>]*>(?P<head>.*?)</head\s*>",
    Flags {
        ignore_case: true,
        multiline: false,
        dotall: true
    }
);
py_regex!(
    SCRIPT_RE,
    r#"[ \t]*<script\b[^>]*\btype\s*=\s*['"]application/c2pa['"][^>]*>(?P<payload>.*?)</script\s*>[ \t]*\r?\n?"#,
    Flags {
        ignore_case: true,
        multiline: false,
        dotall: true
    }
);
py_regex!(
    LINK_RE,
    r#"[ \t]*<link\b[^>]*\brel\s*=\s*['"]c2pa-manifest['"][^>]*>[ \t]*\r?\n?"#,
    Flags {
        ignore_case: true,
        multiline: false,
        dotall: true
    }
);
py_regex!(HREF_RE, r#"\bhref\s*=\s*['"](?P<href>[^'"]+)['"]"#, Flags::I);

/// What the stripped HTML elements declared.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HtmlC2paMatch {
    pub manifest_bytes: Option<Vec<u8>>,
    pub manifest_uri: Option<String>,
    pub element_count: usize,
    pub outside_head: bool,
}

fn char_span(text: &str, span: Span) -> (usize, usize) {
    let start = text[..span.start].chars().count();
    (start, start + text[span.range()].chars().count())
}

/// Detect and strip the Appendix A.7 elements, flagging any outside `<head>`.
pub fn strip_html_c2pa(text: &str) -> (String, Option<HtmlC2paMatch>) {
    let lowered = text.to_ascii_lowercase();
    if !lowered.contains("application/c2pa") && !lowered.contains("c2pa-manifest") {
        return (text.to_owned(), None);
    }
    let head = HEAD_RE.search(text).map(|span| char_span(text, span));
    let mut found = HtmlC2paMatch::default();
    let mark_element = |found: &mut HtmlC2paMatch, haystack: &str, span: Span| {
        found.element_count += 1;
        let (start, end) = char_span(haystack, span);
        if !head.is_some_and(|(head_start, head_end)| head_start <= start && end <= head_end) {
            found.outside_head = true;
        }
    };
    let cleaned = SCRIPT_RE
        .sub_with(text, |caps| {
            mark_element(&mut found, text, whole(caps));
            let (decoded, uri) = decode_inline_c2pa_payload(group(caps, "payload").unwrap_or(""));
            if decoded.is_some() && found.manifest_bytes.is_none() {
                found.manifest_bytes = decoded;
            }
            if uri.is_some() && found.manifest_uri.is_none() {
                found.manifest_uri = uri;
            }
            String::new()
        })
        .into_owned();
    let cleaned = LINK_RE
        .sub_with(&cleaned, |caps| {
            mark_element(&mut found, &cleaned, whole(caps));
            let element = caps.get(0).map_or("", |m| m.as_str());
            if found.manifest_uri.is_none()
                && let Some(href) = HREF_RE
                    .captures(element)
                    .and_then(|href| group(&href, "href").map(str::to_owned))
            {
                found.manifest_uri = Some(href);
            }
            String::new()
        })
        .into_owned();
    if found.element_count == 0 {
        return (text.to_owned(), None);
    }
    (cleaned, Some(found))
}

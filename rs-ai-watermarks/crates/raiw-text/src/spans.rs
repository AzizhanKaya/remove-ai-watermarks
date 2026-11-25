//! Span-protected segmentation: locked technical/structural spans and rewritable prose.

use std::sync::LazyLock;

use serde::Serialize;

use crate::pyre::{Flags, PyRegex, last_group, whole};
use crate::unicode;

/// What a segment of a document is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Prose,
    FencedCode,
    InlineCode,
    MathBlock,
    MathInline,
    HtmlBlock,
    HtmlTag,
    MarkdownStructure,
    ListMarker,
    Url,
    Path,
    Whitespace,
}

impl SpanKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prose => "prose",
            Self::FencedCode => "fenced_code",
            Self::InlineCode => "inline_code",
            Self::MathBlock => "math_block",
            Self::MathInline => "math_inline",
            Self::HtmlBlock => "html_block",
            Self::HtmlTag => "html_tag",
            Self::MarkdownStructure => "markdown_structure",
            Self::ListMarker => "list_marker",
            Self::Url => "url",
            Self::Path => "path",
            Self::Whitespace => "whitespace",
        }
    }

    fn from_group(name: &str) -> Self {
        match name {
            "math_block" => Self::MathBlock,
            "html_block" => Self::HtmlBlock,
            "markdown_structure" => Self::MarkdownStructure,
            "inline_code" => Self::InlineCode,
            "math_inline" => Self::MathInline,
            "url" => Self::Url,
            "html_tag" => Self::HtmlTag,
            "path" => Self::Path,
            "list_marker" => Self::ListMarker,
            _ => Self::FencedCode,
        }
    }

    /// Spans whose content is code-like and never preprocessed as prose.
    pub const fn is_code_locked(self) -> bool {
        matches!(
            self,
            Self::FencedCode
                | Self::InlineCode
                | Self::MathBlock
                | Self::MathInline
                | Self::HtmlBlock
                | Self::HtmlTag
                | Self::Url
                | Self::Path
        )
    }

    /// Spans that end a paragraph cluster during regeneration.
    pub const fn is_block(self) -> bool {
        matches!(
            self,
            Self::FencedCode | Self::MathBlock | Self::HtmlBlock | Self::MarkdownStructure | Self::Whitespace
        )
    }
}

/// A contiguous segment classified as locked or rewritable prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TextSpan {
    pub kind: SpanKind,
    pub text: String,
    pub locked: bool,
}

const HTML_NAMES: &str = "(?:script|style|pre|code|head|svg|math)";

// `(?=\n|$)` under MULTILINE is the `$` assertion; `<name\b[^>]*>` is `<name(?:>|[^\w>][^>]*>)`.
static BLOCK_SPAN_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        &format!(
            r"(?P<fenced_code>^[ \t]*(?:```|~~~)[^\n]*\n[\s\S]*?\n[ \t]*(?:```|~~~)[ \t]*$)|(?P<math_block>\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\])|(?P<html_block><{HTML_NAMES}(?:>|[^\w>][^>]*>)[\s\S]*?</{HTML_NAMES}\s*>)|(?P<markdown_structure>^(?:[ \t]*#{{1,6}}[ \t]+[^\n]+|[ \t]*\|[^\n]+\|[ \t]*$|[ \t]*\[[^\]]+\]:[ \t]+\S+[^\n]*))"
        ),
        Flags {
            ignore_case: true,
            multiline: true,
            dotall: false,
        },
    )
});

static INLINE_SPAN_RE: LazyLock<PyRegex> = LazyLock::new(|| {
    PyRegex::new(
        r"(?P<inline_code>`[^`\n]+`)|(?P<math_inline>(?<!\$)\$(?![\$\s])[^\$\n]+?(?<!\s)\$(?![\$\d])|\\\([\s\S]+?\\\))|(?P<url>https?://[^\s<>)\]]+)|(?P<html_tag></?[A-Za-z][A-Za-z0-9:_-]*(?:\s+[^<>]*?)?>)|(?P<path>(?<![A-Za-z0-9_])(?:\.{0,2}/)[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8}\b)|(?P<list_marker>^[ \t]*(?:[-+*]|\d+[.)])[ \t]+)",
        Flags::M,
    )
});

static PARAGRAPH_BREAK_RE: LazyLock<PyRegex> = LazyLock::new(|| PyRegex::new(r"(\n\s*\n)", Flags::NONE));

const BLOCK_GROUPS: [&str; 4] = ["fenced_code", "math_block", "html_block", "markdown_structure"];
const INLINE_GROUPS: [&str; 6] = [
    "inline_code",
    "math_inline",
    "url",
    "html_tag",
    "path",
    "list_marker",
];

fn prose_or_whitespace(piece: &str) -> TextSpan {
    let is_ws = unicode::str_is_space(piece);
    TextSpan {
        kind: if is_ws {
            SpanKind::Whitespace
        } else {
            SpanKind::Prose
        },
        text: piece.to_owned(),
        locked: is_ws,
    }
}

fn emit_prose_and_inline(segment: &str, spans: &mut Vec<TextSpan>) {
    for part in PARAGRAPH_BREAK_RE.split_keep(segment) {
        if part.is_empty() {
            continue;
        }
        if unicode::str_is_space(part) {
            spans.push(TextSpan {
                kind: SpanKind::Whitespace,
                text: part.to_owned(),
                locked: true,
            });
            continue;
        }
        let mut cursor = 0;
        for caps in INLINE_SPAN_RE.captures_iter(part) {
            let span = whole(&caps);
            if span.start > cursor {
                spans.push(prose_or_whitespace(&part[cursor..span.start]));
            }
            let kind = last_group(&caps, &INLINE_GROUPS).map_or(SpanKind::InlineCode, SpanKind::from_group);
            spans.push(TextSpan {
                kind,
                text: part[span.range()].to_owned(),
                locked: true,
            });
            cursor = span.end;
        }
        if cursor < part.len() {
            spans.push(prose_or_whitespace(&part[cursor..]));
        }
    }
}

fn split_lf(text: &str) -> Vec<TextSpan> {
    let mut spans = Vec::new();
    let mut cursor = 0;
    for caps in BLOCK_SPAN_RE.captures_iter(text) {
        let span = whole(&caps);
        if span.start > cursor {
            emit_prose_and_inline(&text[cursor..span.start], &mut spans);
        }
        let kind = last_group(&caps, &BLOCK_GROUPS).map_or(SpanKind::FencedCode, SpanKind::from_group);
        spans.push(TextSpan {
            kind,
            text: text[span.range()].to_owned(),
            locked: true,
        });
        cursor = span.end;
    }
    if cursor < text.len() {
        emit_prose_and_inline(&text[cursor..], &mut spans);
    }
    spans
}

/// Segment `text` into locked technical/structural spans and unlocked prose spans.
///
/// CR and CRLF line endings are matched as LF and every span maps back to the original bytes.
pub fn split_protected_text_spans(text: &str) -> Vec<TextSpan> {
    if text.is_empty() {
        return Vec::new();
    }
    if !text.contains('\r') {
        return split_lf(text);
    }
    // Original byte offset reached after each normalized byte.
    let mut normalized = String::with_capacity(text.len());
    let mut boundaries = Vec::with_capacity(text.len() + 1);
    boundaries.push(0);
    let mut chars = text.char_indices().peekable();
    while let Some((at, ch)) = chars.next() {
        if ch == '\r' {
            let mut end = at + 1;
            if chars.next_if(|&(_, next)| next == '\n').is_some() {
                end += 1;
            }
            normalized.push('\n');
            boundaries.push(end);
        } else {
            normalized.push(ch);
            boundaries.extend((1..=ch.len_utf8()).map(|width| at + width));
        }
    }
    let mut offset = 0;
    split_lf(&normalized)
        .into_iter()
        .map(|span| {
            let end = offset + span.text.len();
            let restored = TextSpan {
                kind: span.kind,
                text: text[boundaries[offset]..boundaries[end]].to_owned(),
                locked: span.locked,
            };
            offset = end;
            restored
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crlf_spans_map_back_to_original_bytes() {
        let text = "Original prose.\r\n\r\n```python\r\nvalue = \"é\"\r\n```\r\n";
        let spans = split_protected_text_spans(text);
        let joined: String = spans.iter().map(|span| span.text.as_str()).collect();
        assert_eq!(joined, text);
        assert!(
            spans
                .iter()
                .any(|span| span.kind == SpanKind::FencedCode && span.text.contains("value = \"é\"\r\n"))
        );
    }

    #[test]
    fn list_markers_are_locked() {
        let spans = split_protected_text_spans(
            "1. Libraries help readers.\n2) Libraries host workshops.\n- Libraries lend books.",
        );
        let markers: Vec<&str> = spans
            .iter()
            .filter(|span| span.kind == SpanKind::ListMarker)
            .map(|span| span.text.as_str())
            .collect();
        assert_eq!(markers, ["1. ", "2) ", "- "]);
    }
}

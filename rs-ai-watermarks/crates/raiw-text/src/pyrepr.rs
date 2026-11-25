//! `repr(str)` for error messages that quote user text as CPython does.

use std::fmt::Write as _;

use crate::unicode;

/// CPython's `repr()` of a string.
pub fn repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for ch in text.chars() {
        match ch {
            _ if ch == quote || ch == '\\' => {
                out.push('\\');
                out.push(ch);
            }
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if (ch as u32) < 0x20 || ch == '\x7f' => {
                let _ = write!(out, "\\x{:02x}", ch as u32);
            }
            _ if ch.is_ascii() || unicode::is_printable(ch) => out.push(ch),
            _ if (ch as u32) <= 0xFF => {
                let _ = write!(out, "\\x{:02x}", ch as u32);
            }
            _ if (ch as u32) <= 0xFFFF => {
                let _ = write!(out, "\\u{:04x}", ch as u32);
            }
            _ => {
                let _ = write!(out, "\\U{:08x}", ch as u32);
            }
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython() {
        assert_eq!(repr_str("a'b"), "\"a'b\"");
        assert_eq!(repr_str("a'\"b"), "'a\\'\"b'");
        assert_eq!(
            repr_str("\u{e9}\u{200b}\n\u{e0001}"),
            "'\u{e9}\\u200b\\n\\U000e0001'"
        );
        assert_eq!(repr_str("\u{a0}\u{85}"), "'\\xa0\\x85'");
    }
}

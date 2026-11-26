//! CPython string and bytes semantics the scanners depend on.

use std::fmt::Write as _;

use raiw_c2pa::text::is_printable;

/// `bytes.decode("latin-1")`.
pub fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

/// `str.encode("latin-1", "ignore")`.
pub fn latin1_encode_lossy(text: &str) -> Vec<u8> {
    text.chars()
        .filter_map(|c| u8::try_from(u32::from(c)).ok())
        .collect()
}

/// `str.encode("latin-1", "strict")`, `None` on the first unencodable character.
pub fn latin1_encode(text: &str) -> Option<Vec<u8>> {
    text.chars().map(|c| u8::try_from(u32::from(c)).ok()).collect()
}

/// `bytes.decode("utf-8", "replace")`.
pub fn utf8_replace(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `bytes.decode("utf-8", "ignore")`.
pub fn utf8_ignore(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for chunk in bytes.utf8_chunks() {
        out.push_str(chunk.valid());
    }
    out
}

/// `bytes.decode("utf-8", "strict")`.
pub fn utf8_strict(bytes: &[u8]) -> Option<&str> {
    std::str::from_utf8(bytes).ok()
}

/// `str.isspace()` for one character.
pub fn is_space(c: char) -> bool {
    matches!(c, '\t'..='\r' | '\x1c'..='\x1f' | ' ' | '\u{85}' | '\u{a0}') || c.is_whitespace()
}

/// `str.strip()`.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// `str.strip(chars)`.
pub fn strip_chars<'a>(text: &'a str, chars: &[char]) -> &'a str {
    text.trim_matches(|c| chars.contains(&c))
}

/// `str.lower()`.
pub fn lower(text: &str) -> String {
    text.to_lowercase()
}

/// `repr(bytes)`.
pub fn bytes_repr(data: &[u8]) -> String {
    let quote = if data.contains(&b'\'') && !data.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut out = String::with_capacity(data.len() + 3);
    out.push('b');
    out.push(char::from(quote));
    for &byte in data {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b if b == quote => {
                out.push('\\');
                out.push(char::from(b));
            }
            0x20..=0x7e => out.push(char::from(byte)),
            _ => {
                let _ = write!(out, "\\x{byte:02x}");
            }
        }
    }
    out.push(char::from(quote));
    out
}

/// `repr(str)`, escaping what `str.isprintable()` rejects.
pub fn str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    let mut buf = [0u8; 4];
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c == ' ' || is_printable(c.encode_utf8(&mut buf)) => out.push(c),
            c => {
                let code = u32::from(c);
                let _ = if code < 0x100 {
                    write!(out, "\\x{code:02x}")
                } else if code < 0x10000 {
                    write!(out, "\\u{code:04x}")
                } else {
                    write!(out, "\\U{code:08x}")
                };
            }
        }
    }
    out.push(quote);
    out
}

/// `str(float)`.
pub fn float_str(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    raiw_core::pyjson::py_float_repr(value)
}

/// `round(value)` with banker's rounding, as Python's built-in.
pub fn round_half_even(value: f64) -> f64 {
    let rounded = value.round();
    if (value - value.trunc()).abs() == 0.5 {
        2.0 * (value / 2.0).round()
    } else {
        rounded
    }
}

/// `round(value, 2)` as CPython computes it (correctly rounded through the decimal string).
pub fn round2(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let text = format!("{value:.2}");
    text.parse().unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reprs_match_python() {
        assert_eq!(bytes_repr(b"a'b\x00\xff"), "b\"a'b\\x00\\xff\"");
        assert_eq!(str_repr("a\u{a0}b\u{200b}"), "'a\\xa0b\\u200b'");
        assert_eq!(float_str(0.45455), "0.45455");
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(utf8_ignore(b"a\xffb"), "ab");
        assert_eq!(strip("\x1c x \u{3000}"), "x");
    }
}

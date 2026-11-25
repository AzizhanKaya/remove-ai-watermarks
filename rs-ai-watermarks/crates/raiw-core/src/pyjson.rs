//! JSON output that matches CPython's `json.dumps` byte for byte.

use std::fmt::Write as _;
use std::io;

use serde::Serialize;
use serde_json::ser::{CharEscape, Formatter};

/// Formatting choices mirroring the `json.dumps` keyword arguments.
#[derive(Debug, Clone, Copy)]
pub struct JsonStyle {
    pub indent: Option<usize>,
    pub ensure_ascii: bool,
    pub item_separator: &'static str,
    pub key_separator: &'static str,
}

impl JsonStyle {
    /// `json.dumps(value)`.
    pub const COMPACT: Self = Self {
        indent: None,
        ensure_ascii: true,
        item_separator: ", ",
        key_separator: ": ",
    };
    /// `json.dumps(value, indent=2)`, the CLI report format.
    pub const PRETTY: Self = Self {
        indent: Some(2),
        ensure_ascii: true,
        item_separator: ",",
        key_separator: ": ",
    };
    /// `json.dumps(value, ensure_ascii=False)`, the MCP wire format.
    pub const UNICODE: Self = Self {
        indent: None,
        ensure_ascii: false,
        item_separator: ", ",
        key_separator: ": ",
    };
    /// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`.
    pub const TIGHT_UNICODE: Self = Self {
        indent: None,
        ensure_ascii: false,
        item_separator: ",",
        key_separator: ":",
    };

    #[must_use]
    pub const fn ascii(mut self, ensure_ascii: bool) -> Self {
        self.ensure_ascii = ensure_ascii;
        self
    }
}

/// Serialize `value` exactly as Python's `json.dumps` would with `style`.
pub fn to_py_json<T: Serialize + ?Sized>(value: &T, style: JsonStyle) -> String {
    let mut out = Vec::with_capacity(256);
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFormatter::new(style));
    value
        .serialize(&mut ser)
        .expect("serializing into memory cannot fail");
    String::from_utf8(out).expect("formatter emits UTF-8")
}

/// Format `value` the way `repr(float)` does.
pub fn py_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".into();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let mut buffer = ryu_like_shortest(value);
    let negative = buffer.starts_with('-');
    if negative {
        buffer.remove(0);
    }
    let (digits, decpt) = split_digits(&buffer);
    let mut out = String::with_capacity(24);
    if negative {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let exp = decpt - 1;
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let magnitude = exp.unsigned_abs();
        if magnitude < 10 {
            out.push('0');
        }
        out.push_str(&magnitude.to_string());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        let point = decpt as usize;
        if digits.len() <= point {
            out.push_str(&digits);
            for _ in digits.len()..point {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point]);
            out.push('.');
            out.push_str(&digits[point..]);
        }
    }
    out
}

/// Python `repr()` of a JSON-decoded value (lists, dicts, strings, numbers, booleans, `None`).
pub fn py_repr(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(number) => py_number(number),
        Value::String(text) => py_str_repr(text),
        Value::Array(items) => format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => {
            let items: Vec<_> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
    }
}

/// Python `str()` of a JSON-decoded scalar; containers fall back to `repr()`.
pub fn py_str(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => py_repr(other),
    }
}

fn py_number(number: &serde_json::Number) -> String {
    if number.is_f64() {
        number.as_f64().map_or_else(|| number.to_string(), py_float_repr)
    } else {
        number.to_string()
    }
}

/// Python `repr(str)`: single quotes unless the text holds a single quote and no double quote.
pub fn py_str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
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
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn ryu_like_shortest(value: f64) -> String {
    format!("{value:e}")
}

/// Split Rust's shortest `{:e}` output into significant digits and a decimal-point position.
fn split_digits(scientific: &str) -> (String, i32) {
    let (mantissa, exponent) = scientific.split_once('e').expect("`{:e}` always has an exponent");
    let exponent: i32 = exponent.parse().expect("integer exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    (digits.to_owned(), exponent + 1)
}

struct PyFormatter {
    style: JsonStyle,
    depth: usize,
    has_value: bool,
}

impl PyFormatter {
    const fn new(style: JsonStyle) -> Self {
        Self {
            style,
            depth: 0,
            has_value: false,
        }
    }

    fn newline<W: ?Sized + io::Write>(&self, writer: &mut W) -> io::Result<()> {
        if let Some(width) = self.style.indent {
            writer.write_all(b"\n")?;
            for _ in 0..self.depth * width {
                writer.write_all(b" ")?;
            }
        }
        Ok(())
    }

    fn open<W: ?Sized + io::Write>(&mut self, writer: &mut W, bracket: &[u8]) -> io::Result<()> {
        self.depth += 1;
        self.has_value = false;
        writer.write_all(bracket)
    }

    fn close<W: ?Sized + io::Write>(&mut self, writer: &mut W, bracket: &[u8]) -> io::Result<()> {
        self.depth -= 1;
        if self.has_value {
            self.newline(writer)?;
        }
        writer.write_all(bracket)
    }

    fn begin_item<W: ?Sized + io::Write>(&self, writer: &mut W, first: bool) -> io::Result<()> {
        if !first {
            writer.write_all(self.style.item_separator.as_bytes())?;
        }
        self.newline(writer)
    }
}

impl Formatter for PyFormatter {
    fn write_f64<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(py_float_repr(value).as_bytes())
    }

    fn write_f32<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        self.write_f64(writer, f64::from(value))
    }

    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        if !self.style.ensure_ascii || (fragment.is_ascii() && !fragment.contains('\x7f')) {
            return writer.write_all(fragment.as_bytes());
        }
        let mut units = [0u16; 2];
        for ch in fragment.chars() {
            if ch.is_ascii() && ch != '\x7f' {
                writer.write_all(&[ch as u8])?;
            } else {
                for unit in ch.encode_utf16(&mut units) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }

    fn write_char_escape<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        escape: CharEscape,
    ) -> io::Result<()> {
        let text: &[u8] = match escape {
            CharEscape::Quote => b"\\\"",
            CharEscape::ReverseSolidus => b"\\\\",
            CharEscape::Solidus => b"\\/",
            CharEscape::Backspace => b"\\b",
            CharEscape::FormFeed => b"\\f",
            CharEscape::LineFeed => b"\\n",
            CharEscape::CarriageReturn => b"\\r",
            CharEscape::Tab => b"\\t",
            CharEscape::AsciiControl(byte) => return write!(writer, "\\u{byte:04x}"),
        };
        writer.write_all(text)
    }

    fn begin_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.open(writer, b"[")
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b"]")
    }

    fn begin_array_value<W: ?Sized + io::Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        self.begin_item(writer, first)
    }

    fn end_array_value<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.open(writer, b"{")
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b"}")
    }

    fn begin_object_key<W: ?Sized + io::Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        self.begin_item(writer, first)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(self.style.key_separator.as_bytes())
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn floats_follow_python_repr() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-2.5e-7, "-2.5e-07"),
            (123_456.789, "123456.789"),
            (1.5e300, "1.5e+300"),
            (-0.0, "-0.0"),
            (1.0 / 3.0, "0.3333333333333333"),
        ];
        for (value, expected) in cases {
            assert_eq!(py_float_repr(value), expected, "{value}");
        }
    }

    #[test]
    fn pretty_matches_json_dumps_indent_2() {
        let value = json!({"a": [1, 2.5, {}], "b": [], "c": {"d": null, "é": "ü"}});
        let expected = "{\n  \"a\": [\n    1,\n    2.5,\n    {}\n  ],\n  \"b\": [],\n  \"c\": {\n    \"d\": null,\n    \"\\u00e9\": \"\\u00fc\"\n  }\n}";
        assert_eq!(to_py_json(&value, JsonStyle::PRETTY), expected);
    }

    #[test]
    fn compact_uses_spaced_separators() {
        let value = json!({"a": [1, true], "s": "\u{1F600}\n"});
        assert_eq!(
            to_py_json(&value, JsonStyle::COMPACT),
            "{\"a\": [1, true], \"s\": \"\\ud83d\\ude00\\n\"}"
        );
        assert_eq!(
            to_py_json(&value, JsonStyle::UNICODE),
            "{\"a\": [1, true], \"s\": \"\u{1F600}\\n\"}"
        );
        assert_eq!(
            to_py_json(&value, JsonStyle::TIGHT_UNICODE),
            "{\"a\":[1,true],\"s\":\"\u{1F600}\\n\"}"
        );
    }

    #[test]
    fn repr_matches_python() {
        assert_eq!(
            py_repr(&json!(["auto", "cv2", 1, 2.5, true, null])),
            "['auto', 'cv2', 1, 2.5, True, None]"
        );
        assert_eq!(py_repr(&json!("it's")), "\"it's\"");
        assert_eq!(py_repr(&json!({"a": "b"})), "{'a': 'b'}");
        assert_eq!(py_str(&json!(0.15)), "0.15");
        assert_eq!(py_str(&json!("x")), "x");
    }

    #[test]
    fn control_characters_use_lowercase_unicode_escapes() {
        assert_eq!(
            to_py_json(&"\u{1}\u{1f}", JsonStyle::COMPACT),
            "\"\\u0001\\u001f\""
        );
    }
}

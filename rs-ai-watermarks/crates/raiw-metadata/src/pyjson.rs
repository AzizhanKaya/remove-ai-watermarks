//! CPython `json` decoding and `str()` / compact `json.dumps` of the decoded values.

use std::fmt::Write as _;

/// A value decoded by `json.loads`, with Python's integer and dict semantics.
#[derive(Debug, Clone, PartialEq)]
pub enum PyValue {
    Null,
    Bool(bool),
    /// Arbitrary-precision integer, kept as its decimal text.
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<PyValue>),
    /// Insertion-ordered; a repeated key keeps its first position and its last value.
    Dict(Vec<(String, PyValue)>),
}

impl PyValue {
    pub fn as_dict(&self) -> Option<&[(String, PyValue)]> {
        match self {
            Self::Dict(items) => Some(items),
            _ => None,
        }
    }

    /// `dict.get(key)`.
    pub fn get(&self, key: &str) -> Option<&PyValue> {
        self.as_dict()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(text) => Some(text),
            _ => None,
        }
    }

    pub const fn is_container(&self) -> bool {
        matches!(self, Self::List(_) | Self::Dict(_))
    }

    /// `str(value)` for a scalar; containers render as compact JSON (never needed as repr).
    pub fn py_str(&self) -> String {
        match self {
            Self::Null => "None".into(),
            Self::Bool(true) => "True".into(),
            Self::Bool(false) => "False".into(),
            Self::Int(digits) => digits.clone(),
            Self::Float(value) => crate::py::float_str(*value),
            Self::Str(text) => text.clone(),
            Self::List(_) | Self::Dict(_) => self.dumps_compact(),
        }
    }

    /// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`.
    pub fn dumps_compact(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Int(digits) => out.push_str(digits),
            Self::Float(value) => {
                if value.is_nan() {
                    out.push_str("NaN");
                } else if value.is_infinite() {
                    out.push_str(if *value > 0.0 { "Infinity" } else { "-Infinity" });
                } else {
                    out.push_str(&raiw_core::pyjson::py_float_repr(*value));
                }
            }
            Self::Str(text) => write_json_str(text, out),
            Self::List(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_compact(out);
                }
                out.push(']');
            }
            Self::Dict(items) => {
                out.push('{');
                for (i, (key, item)) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_str(key, out);
                    out.push(':');
                    item.write_compact(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_json_str(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

const MAX_DEPTH: usize = 900;

struct Scanner<'a> {
    chars: Vec<char>,
    _text: &'a str,
}

impl Scanner<'_> {
    fn skip_ws(&self, mut pos: usize) -> usize {
        while pos < self.chars.len() && matches!(self.chars[pos], ' ' | '\t' | '\n' | '\r') {
            pos += 1;
        }
        pos
    }

    fn value(&self, pos: usize, depth: usize) -> Option<(PyValue, usize)> {
        if depth > MAX_DEPTH {
            return None;
        }
        let c = *self.chars.get(pos)?;
        match c {
            '"' => self.string(pos + 1).map(|(s, end)| (PyValue::Str(s), end)),
            '{' => self.object(pos + 1, depth),
            '[' => self.array(pos + 1, depth),
            'n' if self.literal(pos, "null") => Some((PyValue::Null, pos + 4)),
            't' if self.literal(pos, "true") => Some((PyValue::Bool(true), pos + 4)),
            'f' if self.literal(pos, "false") => Some((PyValue::Bool(false), pos + 5)),
            'N' if self.literal(pos, "NaN") => Some((PyValue::Float(f64::NAN), pos + 3)),
            'I' if self.literal(pos, "Infinity") => Some((PyValue::Float(f64::INFINITY), pos + 8)),
            '-' if self.literal(pos, "-Infinity") => Some((PyValue::Float(f64::NEG_INFINITY), pos + 9)),
            _ => self.number(pos),
        }
    }

    fn literal(&self, pos: usize, word: &str) -> bool {
        let n = word.chars().count();
        pos + n <= self.chars.len() && self.chars[pos..pos + n].iter().copied().eq(word.chars())
    }

    fn number(&self, start: usize) -> Option<(PyValue, usize)> {
        let digit = |p: usize| self.chars.get(p).is_some_and(char::is_ascii_digit);
        let mut pos = start;
        if self.chars.get(pos) == Some(&'-') {
            pos += 1;
        }
        if self.chars.get(pos) == Some(&'0') {
            pos += 1;
        } else if digit(pos) {
            while digit(pos) {
                pos += 1;
            }
        } else {
            return None;
        }
        let int_end = pos;
        let mut is_float = false;
        if self.chars.get(pos) == Some(&'.') && digit(pos + 1) {
            pos += 1;
            while digit(pos) {
                pos += 1;
            }
            is_float = true;
        }
        if matches!(self.chars.get(pos), Some('e' | 'E')) {
            let mut exp = pos + 1;
            if matches!(self.chars.get(exp), Some('+' | '-')) {
                exp += 1;
            }
            if digit(exp) {
                while digit(exp) {
                    exp += 1;
                }
                pos = exp;
                is_float = true;
            }
        }
        let text: String = self.chars[start..pos].iter().collect();
        if is_float {
            Some((PyValue::Float(text.parse().ok()?), pos))
        } else {
            let digits: String = self.chars[start..int_end].iter().collect();
            let normalized = if digits == "-0" { "0".to_owned() } else { digits };
            Some((PyValue::Int(normalized), pos))
        }
    }

    fn string(&self, mut pos: usize) -> Option<(String, usize)> {
        let mut out = String::new();
        loop {
            let c = *self.chars.get(pos)?;
            pos += 1;
            match c {
                '"' => return Some((out, pos)),
                '\\' => {
                    let esc = *self.chars.get(pos)?;
                    pos += 1;
                    match esc {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let high = self.hex4(pos)?;
                            pos += 4;
                            if (0xD800..0xDC00).contains(&high)
                                && self.chars.get(pos) == Some(&'\\')
                                && self.chars.get(pos + 1) == Some(&'u')
                                && let Some(low) = self.hex4(pos + 2)
                                && (0xDC00..0xE000).contains(&low)
                            {
                                pos += 6;
                                let code = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                                out.push(char::from_u32(code)?);
                            } else {
                                out.push(char::from_u32(high).unwrap_or('\u{fffd}'));
                            }
                        }
                        _ => return None,
                    }
                }
                c if (c as u32) < 0x20 => return None,
                c => out.push(c),
            }
        }
    }

    fn hex4(&self, pos: usize) -> Option<u32> {
        if pos + 4 > self.chars.len() {
            return None;
        }
        let mut value = 0;
        for c in &self.chars[pos..pos + 4] {
            value = value * 16 + c.to_digit(16)?;
        }
        Some(value)
    }

    fn object(&self, pos: usize, depth: usize) -> Option<(PyValue, usize)> {
        let mut items: Vec<(String, PyValue)> = Vec::new();
        let mut pos = self.skip_ws(pos);
        if self.chars.get(pos) == Some(&'}') {
            return Some((PyValue::Dict(items), pos + 1));
        }
        loop {
            if self.chars.get(pos) != Some(&'"') {
                return None;
            }
            let (key, end) = self.string(pos + 1)?;
            pos = self.skip_ws(end);
            if self.chars.get(pos) != Some(&':') {
                return None;
            }
            pos = self.skip_ws(pos + 1);
            let (value, end) = self.value(pos, depth + 1)?;
            if let Some(slot) = items.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = value;
            } else {
                items.push((key, value));
            }
            pos = self.skip_ws(end);
            match self.chars.get(pos) {
                Some('}') => return Some((PyValue::Dict(items), pos + 1)),
                Some(',') => pos = self.skip_ws(pos + 1),
                _ => return None,
            }
        }
    }

    fn array(&self, pos: usize, depth: usize) -> Option<(PyValue, usize)> {
        let mut items = Vec::new();
        let mut pos = self.skip_ws(pos);
        if self.chars.get(pos) == Some(&']') {
            return Some((PyValue::List(items), pos + 1));
        }
        loop {
            let (value, end) = self.value(pos, depth + 1)?;
            items.push(value);
            pos = self.skip_ws(end);
            match self.chars.get(pos) {
                Some(']') => return Some((PyValue::List(items), pos + 1)),
                Some(',') => pos = self.skip_ws(pos + 1),
                _ => return None,
            }
        }
    }
}

/// `json.loads(text)`; `None` where Python raises `ValueError`.
pub fn loads(text: &str) -> Option<PyValue> {
    let scanner = Scanner {
        chars: text.chars().collect(),
        _text: text,
    };
    let start = scanner.skip_ws(0);
    let (value, end) = scanner.value(start, 0)?;
    (scanner.skip_ws(end) == scanner.chars.len()).then_some(value)
}

/// `json.JSONDecoder().raw_decode(chars, index)` over a char slice: the value and its end index.
pub fn raw_decode_chars(chars: &[char], index: usize) -> Option<(PyValue, usize)> {
    let scanner = Scanner {
        chars: chars.to_vec(),
        _text: "",
    };
    scanner.value(index, 0)
}

/// `raw_decode` over latin-1 decoded bytes, with byte offsets (one char per byte).
pub fn raw_decode_latin1(data: &[u8], index: usize) -> Option<(PyValue, usize)> {
    let scanner = Scanner {
        chars: data.iter().map(|&b| char::from(b)).collect(),
        _text: "",
    };
    scanner.value(index, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_semantics() {
        let value = loads(r#" {"a": 1, "b": [1.5, true, null], "a": "x"} "#).unwrap();
        assert_eq!(value.dumps_compact(), r#"{"a":"x","b":[1.5,true,null]}"#);
        assert_eq!(loads("NaN").unwrap().py_str(), "nan");
        assert_eq!(loads("1e400").unwrap().py_str(), "inf");
        assert_eq!(
            loads("12345678901234567890123").unwrap().py_str(),
            "12345678901234567890123"
        );
        assert!(loads("01").is_none());
        assert!(loads("\"a\u{1}\"").is_none());
        assert!(loads("[1,]").is_none());
        assert_eq!(loads(r#""😀""#).unwrap().py_str(), "\u{1f600}");
        assert_eq!(loads("1.0").unwrap().py_str(), "1.0");
        assert_eq!(loads("-0").unwrap().py_str(), "0");
    }
}

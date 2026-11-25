//! A 128-bit UUID with Python `uuid.UUID` parsing and formatting.

use std::fmt;

/// A UUID stored in RFC 4122 (big-endian) byte order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Uuid([u8; 16]);

impl Uuid {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// `uuid.UUID(bytes_le=...)`: the first three fields are little-endian.
    pub fn from_bytes_le(le: [u8; 16]) -> Self {
        let mut bytes = le;
        bytes[..4].reverse();
        bytes[4..6].reverse();
        bytes[6..8].reverse();
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// `uuid.UUID(...).bytes_le`.
    pub fn to_bytes_le(self) -> [u8; 16] {
        Self::from_bytes_le(self.0).0
    }

    pub fn as_u128(self) -> u128 {
        u128::from_be_bytes(self.0)
    }

    pub fn is_nil(self) -> bool {
        self.0 == [0; 16]
    }

    /// Parse exactly as `uuid.UUID(text)` does, `None` where Python raises `ValueError`.
    ///
    /// Python strips `urn:`, `uuid:`, surrounding braces and every hyphen, requires 32
    /// characters, then reads them with `int(text, 16)` (whitespace, sign, `0x` and digit
    /// underscores included) and rejects negative values.
    pub fn parse_python(text: &str) -> Option<Self> {
        let stripped = text.replace("urn:", "").replace("uuid:", "");
        let hex = stripped.trim_matches(['{', '}']).replace('-', "");
        if hex.chars().count() != 32 {
            return None;
        }
        python_hex_int(&hex).map(|value| Self(value.to_be_bytes()))
    }
}

impl fmt::Display for Uuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, byte) in self.0.iter().enumerate() {
            if matches!(i, 4 | 6 | 8 | 10) {
                f.write_str("-")?;
            }
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Python's `str.isspace` for the characters `int()` strips.
fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// `int(text, 16)` restricted to values that fit 128 bits and are non-negative.
fn python_hex_int(text: &str) -> Option<u128> {
    let body = text.trim_matches(is_python_space);
    let (negative, body) = match body.as_bytes().first() {
        Some(b'+') => (false, &body[1..]),
        Some(b'-') => (true, &body[1..]),
        _ => (false, body),
    };
    let (body, underscore_ok) = match body.get(..2) {
        Some("0x" | "0X") => (&body[2..], true),
        _ => (body, false),
    };
    let mut value: u128 = 0;
    let mut digits = 0;
    let mut previous_underscore = false;
    for (i, c) in body.chars().enumerate() {
        if c == '_' {
            if previous_underscore || (i == 0 && !underscore_ok) {
                return None;
            }
            previous_underscore = true;
            continue;
        }
        let digit = c.to_digit(16)?;
        value = value.checked_mul(16)?.checked_add(u128::from(digit))?;
        digits += 1;
        previous_underscore = false;
    }
    if digits == 0 || previous_underscore || (negative && value != 0) {
        return None;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_and_parses_round_trip() {
        let text = "83424621-03cb-40e3-9808-a9fae837156d";
        let id = Uuid::parse_python(text).expect("parse");
        assert_eq!(id.to_string(), text);
        assert_eq!(Uuid::from_bytes_le(id.to_bytes_le()), id);
        assert_eq!(id.to_bytes_le()[..4], [0x21, 0x46, 0x42, 0x83]);
    }

    #[test]
    fn python_int_grammar() {
        assert_eq!(python_hex_int("0x_ff"), Some(255));
        assert_eq!(python_hex_int("f_f"), Some(255));
        assert_eq!(python_hex_int("_ff"), None);
        assert_eq!(python_hex_int("ff_"), None);
        assert_eq!(python_hex_int("f__f"), None);
        assert_eq!(python_hex_int(" +ff "), Some(255));
        assert_eq!(python_hex_int("-0"), Some(0));
        assert_eq!(python_hex_int("-1"), None);
        assert_eq!(python_hex_int("0x"), None);
    }
}

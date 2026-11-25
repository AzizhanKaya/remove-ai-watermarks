//! The byte-to-Variation-Selector bijection shared by C2PATXT and VSRMark.

/// `VS1..VS16` (`U+FE00..U+FE0F`) carry bytes `0..15`; `VS17..VS256` carry `16..255`.
pub const fn byte_to_variation_selector(byte: u8) -> char {
    let cp = if byte < 16 {
        0xFE00 + byte as u32
    } else {
        0xE0100 + (byte as u32 - 16)
    };
    match char::from_u32(cp) {
        Some(ch) => ch,
        None => '\u{FFFD}',
    }
}

/// Inverse of [`byte_to_variation_selector`].
pub const fn variation_selector_to_byte(ch: char) -> Option<u8> {
    let cp = ch as u32;
    if cp >= 0xFE00 && cp <= 0xFE0F {
        Some((cp - 0xFE00) as u8)
    } else if cp >= 0xE0100 && cp <= 0xE01EF {
        Some((cp - 0xE0100 + 16) as u8)
    } else {
        None
    }
}

pub const fn is_variation_selector(ch: char) -> bool {
    variation_selector_to_byte(ch).is_some()
}

/// Encode bytes as a contiguous run of Variation Selectors.
pub fn encode_variation_selectors(payload: &[u8]) -> String {
    payload
        .iter()
        .map(|&byte| byte_to_variation_selector(byte))
        .collect()
}

/// Decode every Variation Selector in `text`, ignoring other characters.
pub fn decode_variation_selectors(text: &str) -> Vec<u8> {
    text.chars().filter_map(variation_selector_to_byte).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bijection_covers_every_byte() {
        let raw: Vec<u8> = (0..=255).collect();
        let encoded = encode_variation_selectors(&raw);
        let chars: Vec<char> = encoded.chars().collect();
        assert_eq!(chars.len(), 256);
        assert_eq!(chars[0] as u32, 0xFE00);
        assert_eq!(chars[15] as u32, 0xFE0F);
        assert_eq!(chars[16] as u32, 0xE0100);
        assert_eq!(chars[255] as u32, 0xE01EF);
        assert_eq!(decode_variation_selectors(&encoded), raw);
        assert_eq!(variation_selector_to_byte('A'), None);
    }
}

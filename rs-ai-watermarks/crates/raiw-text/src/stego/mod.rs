//! Character steganography: decoders, contextual preservation rules and the codepoint scrubber.

pub mod context;
pub mod decode;
pub mod prose;
pub mod strip;

/// Zero-width, format, filler and default-ignorable codepoints treated as steganography.
const ZERO_WIDTH_CODEPOINTS: [u32; 58] = [
    0x00AD, 0x034F, 0x061C, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x180B, 0x180C, 0x180D, 0x180E, 0x180F, 0x200B,
    0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2060, 0x2061, 0x2062, 0x2063,
    0x2064, 0x2065, 0x2066, 0x2067, 0x2068, 0x2069, 0x206A, 0x206B, 0x206C, 0x206D, 0x206E, 0x206F, 0x3164,
    0xFEFF, 0xFFA0, 0xFFF9, 0xFFFA, 0xFFFB, 0x1BCA0, 0x1BCA1, 0x1BCA2, 0x1BCA3, 0x1D159, 0x1D173, 0x1D174,
    0x1D175, 0x1D176, 0x1D177, 0x1D178, 0x1D179, 0x1D17A, 0xE0000,
];

pub fn is_zero_width(cp: u32) -> bool {
    ZERO_WIDTH_CODEPOINTS.binary_search(&cp).is_ok()
}

/// Unicode Private Use Areas (ChatGPT citation markers `U+E200..U+E203` included).
pub const fn is_pua(cp: u32) -> bool {
    matches!(cp, 0xE000..=0xF8FF | 0xF0000..=0xFFFFD | 0x10_0000..=0x10_FFFD)
}

/// Unassigned default-ignorable codepoints and Unicode noncharacters.
pub const fn is_reserved_ignorable_or_noncharacter(cp: u32) -> bool {
    matches!(cp, 0xFFF0..=0xFFF8 | 0xE0080..=0xE00FF | 0xE01F0..=0xE0FFF | 0xFDD0..=0xFDEF)
        || ((cp & 0xFFFE) == 0xFFFE && (cp >> 16) <= 0x10)
}

/// Invisible, Variation Selector, tag, PUA or reserved codepoints.
pub fn is_invisible_or_stego(cp: u32) -> bool {
    is_zero_width(cp)
        || matches!(cp, 0xFE00..=0xFE0F | 0xE0000..=0xE007F | 0xE0100..=0xE01EF)
        || is_pua(cp)
        || is_reserved_ignorable_or_noncharacter(cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_width_table_is_sorted() {
        assert!(ZERO_WIDTH_CODEPOINTS.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(is_zero_width(0x200B) && is_zero_width(0xE0000) && !is_zero_width(0x20));
    }
}

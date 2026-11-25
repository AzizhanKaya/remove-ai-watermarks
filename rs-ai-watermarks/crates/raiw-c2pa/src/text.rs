//! Python string predicates the manifest normalization relies on.

use unicode_general_category::{GeneralCategory, get_general_category};

/// Python's `str.isprintable()`: no "Other" or "Separator" character except the ASCII space.
pub fn is_printable(text: &str) -> bool {
    text.chars().all(|ch| {
        ch == ' '
            || !matches!(
                get_general_category(ch),
                GeneralCategory::Control
                    | GeneralCategory::Format
                    | GeneralCategory::Surrogate
                    | GeneralCategory::PrivateUse
                    | GeneralCategory::Unassigned
                    | GeneralCategory::LineSeparator
                    | GeneralCategory::ParagraphSeparator
                    | GeneralCategory::SpaceSeparator
            )
    })
}

/// Python's `str.casefold()` for matching ASCII registry tokens.
///
/// Beyond lower-casing, every full case folding that produces ASCII letters is applied
/// (sharp s, long s, Latin ligatures, and the precomposed letters with combining marks).
pub fn casefold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            'ß' | 'ẞ' => out.push_str("ss"),
            'ſ' => out.push('s'),
            'ŉ' => out.push_str("\u{2bc}n"),
            'ǰ' => out.push_str("j\u{30c}"),
            'ẖ' => out.push_str("h\u{331}"),
            'ẗ' => out.push_str("t\u{308}"),
            'ẘ' => out.push_str("w\u{30a}"),
            'ẙ' => out.push_str("y\u{30a}"),
            'ẚ' => out.push_str("a\u{2be}"),
            'ﬀ' => out.push_str("ff"),
            'ﬁ' => out.push_str("fi"),
            'ﬂ' => out.push_str("fl"),
            'ﬃ' => out.push_str("ffi"),
            'ﬄ' => out.push_str("ffl"),
            'ﬅ' | 'ﬆ' => out.push_str("st"),
            _ => out.extend(ch.to_lowercase()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printable_matches_python() {
        assert!(is_printable("Adobe Firefly 1.0"));
        assert!(is_printable(""));
        assert!(!is_printable("a\u{a0}b"));
        assert!(!is_printable("tab\t"));
        assert!(!is_printable("zw\u{200b}"));
        assert!(is_printable("DALL·E 北京"));
    }

    #[test]
    fn casefold_reaches_ascii_tokens() {
        assert_eq!(casefold("Straße ﬁrefly"), "strasse firefly");
        assert_eq!(casefold("SUNO"), "suno");
    }
}

//! Registered variation sequences and emoji flag tag codes (generated from the Python tables).
//!
//! Variation data: Unicode 17.0 StandardizedVariants, emoji-variation-sequences and the
//! 2026-08-03 IVD. Flag codes: CLDR 48.2 validity files. Unicode License V3.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

const VARIANTS: &str = include_str!("../data/variants.txt");
const FLAG_CODES: &str = include_str!("../data/flag_codes.txt");

fn parse_hex(hex: &str) -> Option<u32> {
    u32::from_str_radix(hex, 16).ok()
}

static VARIATION_BASES: LazyLock<HashMap<u32, HashSet<u32>>> = LazyLock::new(|| {
    VARIANTS
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(' ').map(parse_hex);
            let selector = fields.next()??;
            Some((selector, fields.collect::<Option<HashSet<u32>>>()?))
        })
        .collect()
});

static FLAGS: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| FLAG_CODES.lines().filter(|line| !line.is_empty()).collect());

/// Whether `base` followed by `selector` is a registered variation pair.
pub fn is_registered_variation(selector: char, base: char) -> bool {
    VARIATION_BASES
        .get(&(selector as u32))
        .is_some_and(|bases| bases.contains(&(base as u32)))
}

/// Whether `selector` has any registered base at all.
pub fn has_registered_bases(selector: char) -> bool {
    VARIATION_BASES.contains_key(&(selector as u32))
}

/// Valid CLDR subdivision or numeric region code for an emoji tag sequence.
pub fn is_emoji_flag_code(code: &str) -> bool {
    FLAGS.contains(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_load() {
        assert!(is_registered_variation('\u{FE0F}', '\u{2764}'));
        assert!(is_registered_variation('\u{E0100}', '\u{845B}'));
        assert!(!is_registered_variation('\u{FE0F}', 'a'));
        assert!(is_emoji_flag_code("gbeng") && is_emoji_flag_code("001") && !is_emoji_flag_code("us"));
    }
}

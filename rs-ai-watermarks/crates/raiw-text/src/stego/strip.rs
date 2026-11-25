//! The emoji- and script-aware scrubber for tags, PUA markers and zero-width/VS codepoints.

use std::collections::HashSet;

use crate::selectors::variation_selector_to_byte;
use crate::stego::context::{
    blocks_canonical_mark_reordering, contextual_zwsp_indices, has_discretionary_hyphen_context,
    is_emoji_base, is_explicit_virama_join, is_joining_script_char, math_word_joiner_indices,
    neighbor_skipping_modifiers, subdivision_flag_tag_indices, valid_rtl_bidi_indices,
};
use crate::stego::{is_pua, is_reserved_ignorable_or_noncharacter, is_zero_width};
use crate::tables::is_registered_variation;

const NON_JOINING_ARABIC: [char; 11] = [
    '\u{0622}', '\u{0623}', '\u{0624}', '\u{0625}', '\u{0627}', '\u{062F}', '\u{0630}', '\u{0631}',
    '\u{0632}', '\u{0648}', '\u{0698}',
];

/// Result of [`extract_and_strip_stego_codepoints`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StegoStrip {
    pub cleaned: String,
    pub tags_payload: Option<String>,
    pub removed: usize,
    pub tag_codepoints: usize,
    pub anomalous: usize,
}

const fn is_tag(cp: u32) -> bool {
    cp >= 0xE0000 && cp <= 0xE007F
}

/// Remove non-flag Unicode Tags and extract their printable ASCII.
pub fn strip_unicode_tags(chars: &[char]) -> (Vec<char>, Option<String>) {
    if !chars.iter().any(|&ch| is_tag(ch as u32)) {
        return (chars.to_vec(), None);
    }
    let flags = subdivision_flag_tag_indices(chars);
    let mut out = Vec::with_capacity(chars.len());
    let mut payload = String::new();
    for (index, &ch) in chars.iter().enumerate() {
        let cp = ch as u32;
        if is_tag(cp) && !flags.contains(&index) {
            if (0xE0020..=0xE007E).contains(&cp)
                && let Some(ascii) = char::from_u32(cp - 0xE0000)
            {
                payload.push(ascii);
            }
            continue;
        }
        out.push(ch);
    }
    (out, (!payload.is_empty()).then_some(payload))
}

fn keeps_zero_width(chars: &[char], index: usize, valid_bidi: &HashSet<usize>, contextual: bool) -> bool {
    let ch = chars[index];
    let prev = index.checked_sub(1).map_or(0, |at| chars[at] as u32);
    let next = chars.get(index + 1).map_or(0, |&next| next as u32);
    match ch {
        '\u{034F}' => blocks_canonical_mark_reordering(chars, index),
        '\u{00AD}' => contextual && has_discretionary_hyphen_context(chars, index),
        '\u{202A}' | '\u{202B}' | '\u{202C}' | '\u{2066}' | '\u{2067}' | '\u{2068}' | '\u{2069}'
        | '\u{200E}' | '\u{200F}' | '\u{061C}' => valid_bidi.contains(&index),
        '\u{180B}'..='\u{180F}' => (0x1800..=0x18AF).contains(&prev) && !(0x180B..=0x180F).contains(&next),
        '\u{17B4}' | '\u{17B5}' => (0x1780..=0x17FF).contains(&prev),
        '\u{115F}' | '\u{1160}' => {
            (0x1100..=0x11FF).contains(&prev)
                || (0xA960..=0xA97C).contains(&prev)
                || (0xD7B0..=0xD7C6).contains(&prev)
        }
        '\u{200D}' => {
            let left = neighbor_skipping_modifiers(chars, index, false);
            let right = neighbor_skipping_modifiers(chars, index, true);
            is_explicit_virama_join(chars, index)
                || left.zip(right).is_some_and(|(left, right)| {
                    (is_emoji_base(left) && is_emoji_base(right))
                        || (is_joining_script_char(left) && is_joining_script_char(right))
                })
        }
        '\u{200C}' => {
            let left = index.checked_sub(1).map(|at| chars[at]);
            let right = chars.get(index + 1).copied();
            left.zip(right).is_some_and(|(left, right)| {
                is_joining_script_char(left)
                    && is_joining_script_char(right)
                    && !NON_JOINING_ARABIC.contains(&left)
            })
        }
        _ => false,
    }
}

/// Strip Unicode Tags, PUA markers and steganographic zero-width/VS codepoints.
///
/// Preserves registered variation pairs, valid Mongolian/Khmer/Hangul formatters, balanced RTL
/// Bidi controls, valid subdivision flags, emoji and joining-script joiners, explicit virama
/// joins, contextual ZWSP, soft hyphens, math word joiners and reordering-blocking CGJ.
pub fn extract_and_strip_stego_codepoints(text: &str, preserve_contextual_controls: bool) -> StegoStrip {
    let original: Vec<char> = text.chars().collect();
    let (chars, tags_payload) = strip_unicode_tags(&original);
    let tag_codepoints = original.len() - chars.len();
    let valid_bidi = valid_rtl_bidi_indices(&chars);
    let mut contextual = if preserve_contextual_controls {
        contextual_zwsp_indices(&chars)
    } else {
        HashSet::default()
    };
    if preserve_contextual_controls && chars.contains(&'\u{2060}') {
        contextual.extend(math_word_joiner_indices(&chars));
    }
    let mut anomalous = 0;
    let mut out = String::with_capacity(text.len());
    for (index, &ch) in chars.iter().enumerate() {
        let cp = ch as u32;
        if is_pua(cp) || is_reserved_ignorable_or_noncharacter(cp) {
            anomalous += 1;
            continue;
        }
        if variation_selector_to_byte(ch).is_some() {
            let prev = index.checked_sub(1).map(|at| chars[at]);
            let next_is_vs = chars
                .get(index + 1)
                .is_some_and(|&next| variation_selector_to_byte(next).is_some());
            if prev.is_some_and(|prev| is_registered_variation(ch, prev)) && !next_is_vs {
                out.push(ch);
            } else {
                anomalous += 1;
            }
            continue;
        }
        if contextual.contains(&index) {
            out.push(ch);
            continue;
        }
        if is_zero_width(cp) {
            if keeps_zero_width(&chars, index, &valid_bidi, preserve_contextual_controls) {
                out.push(ch);
            } else {
                anomalous += 1;
            }
            continue;
        }
        out.push(ch);
    }
    StegoStrip {
        cleaned: out,
        tags_payload,
        removed: tag_codepoints + anomalous,
        tag_codepoints,
        anomalous,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emoji_and_scripts_survive() {
        let clean = " Shipping \u{2764}\u{fe0f} and \u{1f469}\u{200d}\u{1f4bb} with zero steganography. \u{0645}\u{06cc}\u{200c}\u{0634}\u{0648}\u{062f}";
        let strip = extract_and_strip_stego_codepoints(clean, true);
        assert_eq!(strip.cleaned, clean);
        assert_eq!(strip.removed, 0);
        let dirty = "Hello\u{200b}\u{200c} world";
        assert_eq!(
            extract_and_strip_stego_codepoints(dirty, true).cleaned,
            "Hello world"
        );
    }
}

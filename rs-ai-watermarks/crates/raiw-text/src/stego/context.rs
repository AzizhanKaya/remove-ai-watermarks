//! Contextual rules that keep legitimate joiners, selectors, Bidi controls and flag tags.

use std::collections::{BTreeSet, HashSet};

use crate::spans::{SpanKind, split_protected_text_spans};
use crate::stego::is_zero_width;
use crate::tables::is_emoji_flag_code;
use crate::unicode::{self, Category};

const EMOJI_BASE_EXTRA: [u32; 28] = [
    0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x2194, 0x2195, 0x2196, 0x2197, 0x2198, 0x2199, 0x21A9,
    0x21AA, 0x25AA, 0x25AB, 0x25B6, 0x25C0, 0x25FB, 0x25FC, 0x25FD, 0x25FE, 0x2934, 0x2935, 0x3030, 0x303D,
    0x3297, 0x3299,
];

/// Whether `ch` can serve as an emoji base around a ZWJ.
pub fn is_emoji_base(ch: char) -> bool {
    let cp = ch as u32;
    matches!(cp, 0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0x2300..=0x23FF | 0x2B00..=0x2BFF)
        || EMOJI_BASE_EXTRA.contains(&cp)
        || unicode::category(ch) == Category::So
}

/// Emoji presentation selector, keycap or skin-tone modifier.
pub fn is_emoji_modifier(ch: char) -> bool {
    matches!(ch as u32, 0xFE0E | 0xFE0F | 0x20E3 | 0x1F3FB..=0x1F3FF)
}

/// The nearest neighbor in direction `step`, skipping emoji modifiers.
pub fn neighbor_skipping_modifiers(chars: &[char], start: usize, forward: bool) -> Option<char> {
    let mut index = start;
    loop {
        index = if forward {
            index.checked_add(1)?
        } else {
            index.checked_sub(1)?
        };
        let &ch = chars.get(index)?;
        if !is_emoji_modifier(ch) {
            return Some(ch);
        }
    }
}

/// Orthographic scripts that require `U+200C`/`U+200D`.
pub fn is_joining_script_char(ch: char) -> bool {
    let in_block = matches!(
        ch as u32,
        0x0600..=0x08FF | 0x0900..=0x0DFF | 0x0F00..=0x109F | 0x1780..=0x18AF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFC
    );
    in_block && {
        let category = unicode::category(ch);
        category.is_letter() || category.is_mark()
    }
}

/// ZWJ shaping sequences missed by the two-neighbor join check (Unicode 17, Chapters 12-13).
pub fn is_explicit_virama_join(chars: &[char], index: usize) -> bool {
    if index < 2 {
        return false;
    }
    let virama = chars[index - 1];
    let mut base = chars[index - 2];
    let right = chars.get(index + 1).copied();
    if virama == '\u{A8C4}' {
        let saurashtra = |ch: char| ('\u{A892}'..='\u{A8B3}').contains(&ch);
        return saurashtra(base) && right.is_some_and(saurashtra);
    }
    if let Some(right) = right
        && !(unicode::is_space(right) || unicode::category(right).is_punctuation())
    {
        return false;
    }
    if virama == '\u{094D}' {
        if base == '\u{093C}' && index >= 3 {
            base = chars[index - 3];
        }
        return matches!(base, '\u{0915}'..='\u{0939}' | '\u{0958}'..='\u{095F}' | '\u{0978}'..='\u{097F}');
    }
    virama == '\u{0D4D}'
        && matches!(
            base,
            '\u{0D23}' | '\u{0D28}' | '\u{0D30}' | '\u{0D32}' | '\u{0D33}'
        )
}

const BIDI_CONTROLS: [char; 10] = [
    '\u{202A}', '\u{202B}', '\u{202C}', '\u{2066}', '\u{2067}', '\u{2068}', '\u{2069}', '\u{200E}',
    '\u{200F}', '\u{061C}',
];

/// Balanced Bidi controls on lines with RTL characters; `U+202D`/`U+202E` overrides never qualify.
pub fn valid_rtl_bidi_indices(chars: &[char]) -> HashSet<usize> {
    let mut valid = HashSet::new();
    if !chars.iter().any(|ch| BIDI_CONTROLS.contains(ch)) {
        return valid;
    }
    let text: String = chars.iter().collect();
    let mut offset = 0;
    for line in unicode::splitlines_keepends(&text) {
        let line: Vec<char> = line.chars().collect();
        if line.iter().any(|ch| BIDI_CONTROLS.contains(ch)) && line.iter().any(|&ch| unicode::is_rtl(ch)) {
            let mut embeddings = Vec::new();
            let mut isolates = Vec::new();
            for (index, &ch) in line.iter().enumerate() {
                match ch {
                    '\u{202A}' | '\u{202B}' => embeddings.push(offset + index),
                    '\u{202C}' => {
                        if let Some(open) = embeddings.pop() {
                            valid.extend([open, offset + index]);
                        }
                    }
                    '\u{2066}' | '\u{2067}' | '\u{2068}' => isolates.push(offset + index),
                    '\u{2069}' => {
                        if let Some(open) = isolates.pop() {
                            valid.extend([open, offset + index]);
                        }
                    }
                    '\u{200E}' | '\u{200F}' | '\u{061C}' => {
                        let left = index.checked_sub(1).map(|at| line[at]);
                        let right = line.get(index + 1).copied();
                        if left.is_some_and(unicode::is_rtl) || right.is_some_and(unicode::is_rtl) {
                            valid.insert(offset + index);
                        }
                    }
                    _ => {}
                }
            }
        }
        offset += line.len();
    }
    valid
}

const fn is_flag_tag_char(ch: char) -> bool {
    matches!(ch as u32, 0xE0030..=0xE0039 | 0xE0061..=0xE007A)
}

/// Indices of tag characters (and the cancel tag) in valid CLDR subdivision or region flags.
pub fn subdivision_flag_tag_indices(chars: &[char]) -> BTreeSet<usize> {
    let mut preserved = BTreeSet::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '\u{1F3F4}' {
            index += 1;
            continue;
        }
        let tags = chars[index + 1..]
            .iter()
            .take_while(|&&ch| is_flag_tag_char(ch))
            .count();
        let cancel = index + 1 + tags;
        if (1..=30).contains(&tags) && chars.get(cancel) == Some(&'\u{E007F}') {
            let code: String = chars[index + 1..cancel]
                .iter()
                .filter_map(|&ch| char::from_u32(ch as u32 - 0xE0000))
                .collect();
            if is_emoji_flag_code(&code) {
                preserved.extend(index + 1..=cancel);
            }
            index = cancel + 1;
        } else {
            index += 1;
        }
    }
    preserved
}

const fn is_thai_word_char(ch: char) -> bool {
    matches!(ch, '\u{0E01}'..='\u{0E3A}' | '\u{0E40}'..='\u{0E4E}')
}

/// Isolated word breaks in Thai and literal zero-width spaces inside code.
pub fn contextual_zwsp_indices(chars: &[char]) -> HashSet<usize> {
    let mut indices = HashSet::new();
    if !chars.contains(&'\u{200B}') {
        return indices;
    }
    for index in 1..chars.len().saturating_sub(1) {
        if chars[index] == '\u{200B}'
            && is_thai_word_char(chars[index - 1])
            && is_thai_word_char(chars[index + 1])
        {
            indices.insert(index);
        }
    }
    let text: String = chars.iter().collect();
    let mut offset = 0;
    for span in split_protected_text_spans(&text) {
        let length = span.text.chars().count();
        if matches!(span.kind, SpanKind::InlineCode | SpanKind::FencedCode) {
            for (local, ch) in span.text.chars().enumerate() {
                if ch != '\u{200B}' {
                    continue;
                }
                let index = offset + local;
                let left = index.checked_sub(1).map(|at| chars[at]);
                let right = chars.get(index + 1).copied();
                if [left, right]
                    .into_iter()
                    .flatten()
                    .all(|neighbor| !is_zero_width(neighbor as u32))
                {
                    indices.insert(index);
                }
            }
        }
        offset += length;
    }
    indices
}

/// Paired `U+2060` around `U+00D7`/`U+00F7` between decimal digits.
pub fn math_word_joiner_indices(chars: &[char]) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut index = 1;
    while index + 3 < chars.len() {
        if chars[index] == '\u{2060}'
            && matches!(chars[index + 1], '\u{00D7}' | '\u{00F7}')
            && chars[index + 2] == '\u{2060}'
            && unicode::is_decimal(chars[index - 1])
            && unicode::is_decimal(chars[index + 3])
        {
            indices.extend([index, index + 2]);
            index += 3;
        } else {
            index += 1;
        }
    }
    indices
}

/// An intra-word discretionary hyphen between letter clusters.
pub fn has_discretionary_hyphen_context(chars: &[char], index: usize) -> bool {
    let is_mark = |ch: char| unicode::category(ch).is_mark();
    let mut right = index + 1;
    if chars.get(right) == Some(&'\u{2011}') {
        right += 1;
    }
    let mut left = index.checked_sub(1);
    while let Some(at) = left {
        if !is_mark(chars[at]) {
            break;
        }
        if is_zero_width(chars[at] as u32) {
            return false;
        }
        left = at.checked_sub(1);
    }
    let Some(left) = left else { return false };
    if right >= chars.len() || !(unicode::is_alpha(chars[left]) && unicode::is_alpha(chars[right])) {
        return false;
    }
    let mut end = right + 1;
    while end < chars.len() && is_mark(chars[end]) {
        if is_zero_width(chars[end] as u32) {
            return false;
        }
        end += 1;
    }
    // A neighboring non-hyphen control can be part of a mixed hidden payload.
    let before = left.checked_sub(1).map(|at| chars[at]);
    [before, chars.get(end).copied()]
        .into_iter()
        .flatten()
        .all(|ch| !is_zero_width(ch as u32) || ch == '\u{00AD}')
}

/// A CGJ barrier between marks that would otherwise cross it in NFD.
pub fn blocks_canonical_mark_reordering(chars: &[char], index: usize) -> bool {
    let mut highest_left = 0;
    let mut left = index;
    while left > 0 {
        let class = unicode::combining(chars[left - 1]);
        if class == 0 {
            break;
        }
        highest_left = highest_left.max(class);
        left -= 1;
    }
    if highest_left == 0 {
        return false;
    }
    chars[index + 1..]
        .iter()
        .map(|&ch| unicode::combining(ch))
        .take_while(|&class| class != 0)
        .any(|class| class < highest_left)
}

//! Prose-only cleanup: invisible HTML entities, assistant citation artifacts, exotic spaces, homoglyphs.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::pyre::{group, py_regex, whole};
use crate::selectors::variation_selector_to_byte;
use crate::stego::is_invisible_or_stego;
use crate::unicode;

const INVISIBLE_NAMED_ENTITIES: [(&str, char); 14] = [
    ("zerowidthspace", '\u{200B}'),
    ("zwnj", '\u{200C}'),
    ("zwj", '\u{200D}'),
    ("lrm", '\u{200E}'),
    ("rlm", '\u{200F}'),
    ("shy", '\u{00AD}'),
    ("nobreak", '\u{2060}'),
    ("wordjoiner", '\u{2060}'),
    ("applyfunction", '\u{2061}'),
    ("af", '\u{2061}'),
    ("invisibletimes", '\u{2062}'),
    ("it", '\u{2062}'),
    ("invisiblecomma", '\u{2063}'),
    ("ic", '\u{2063}'),
];

py_regex!(
    HTML_ENTITY_RE,
    r"&(?:#(?P<dec>\d{1,7})|#x(?P<hex>[0-9A-Fa-f]{1,6})|(?P<name>[A-Za-z]{2,24}));"
);
py_regex!(
    ASSISTANT_CITATION_RE,
    "\u{e200}[^\u{e200}\u{e201}]{0,160}\u{e201}|\\[cite(?:_start|_end|:\\s*\\d+(?:\\s*,\\s*\\d+)*)\\]|\\[span_\\d+\\](?:\\((?:start|end)_span\\))?|\\bgrok_render_citation_card_json\\b(?:=\\{[^}\\n]*\\})?"
);
const PLAIN_CHATGPT_CITATION: &str = r"\bcite(?:turn\d+(?:search|news|view|image|file|fetch|forecast|sports|finance|academia|product|map)\d+)+\b";
py_regex!(
    PLAIN_CHATGPT_CITATION_RUN_RE,
    format!(
        r"(?<![ \t])(?P<before>[ \t]*)(?P<tokens>{PLAIN_CHATGPT_CITATION}(?:[ \t]+{PLAIN_CHATGPT_CITATION})*)(?P<after>[ \t]*)"
    )
);

/// Decode HTML entities that resolve to zero-width, format or Variation Selector codepoints.
pub fn decode_invisible_html_entities(text: &str) -> (String, usize) {
    if !text.contains('&') {
        return (text.to_owned(), 0);
    }
    let mut count = 0;
    let decoded = HTML_ENTITY_RE.sub_with(text, |caps| {
        let original = || caps.get(0).map_or_else(String::new, |m| m.as_str().to_owned());
        if let Some(name) = group(caps, "name") {
            let lowered = unicode::lower(name);
            return match INVISIBLE_NAMED_ENTITIES
                .iter()
                .find(|(entity, _)| *entity == lowered)
            {
                Some(&(_, ch)) => {
                    count += 1;
                    ch.to_string()
                }
                None => original(),
            };
        }
        let cp = match (group(caps, "dec"), group(caps, "hex")) {
            (Some(dec), _) => dec
                .chars()
                .try_fold(0u32, |acc, ch| Some(acc * 10 + unicode::decimal_value(ch)?)),
            (None, Some(hex)) => u32::from_str_radix(hex, 16).ok(),
            (None, None) => None,
        };
        match cp
            .filter(|&cp| cp <= 0x10_FFFF && is_invisible_or_stego(cp))
            .and_then(char::from_u32)
        {
            Some(ch) => {
                count += 1;
                ch.to_string()
            }
            None => original(),
        }
    });
    (decoded.into_owned(), count)
}

/// Strip ChatGPT citation tokens and PUA blocks, Gemini markers and Grok citation cards.
pub fn strip_assistant_citations(text: &str, at_document_end: bool) -> (String, usize) {
    let mut count = 0;
    let cleaned = if text.contains('\u{E200}')
        || text.contains("[cite")
        || text.contains("[span_")
        || text.contains("grok_render_citation_card_json")
    {
        let (cleaned, removed) = ASSISTANT_CITATION_RE.subn("", text);
        count = removed;
        cleaned
    } else {
        text.to_owned()
    };
    if !cleaned.contains("citeturn") {
        return (cleaned, count);
    }
    let result = PLAIN_CHATGPT_CITATION_RUN_RE.sub_with(&cleaned, |caps| {
        count += unicode::split_whitespace(group(caps, "tokens").unwrap_or("")).count();
        let end = whole(caps).end;
        let following = cleaned[end..].chars().next();
        let after = group(caps, "after").unwrap_or("");
        let closes = following.is_some_and(|ch| ",.;:!?\u{2026})]}\r\n".contains(ch));
        if (following.is_none() && at_document_end) || closes {
            return after.to_owned();
        }
        // A protected span may follow this prose fragment. Preserve its separator.
        let before = group(caps, "before").unwrap_or("");
        if before.is_empty() {
            after.to_owned()
        } else {
            before.to_owned()
        }
    });
    (result.into_owned(), count)
}

const fn is_exotic_space(cp: u32) -> bool {
    matches!(cp, 0x00A0 | 0x1680 | 0x2000..=0x200A | 0x202F | 0x205F | 0x3000)
}

fn is_legitimate_typographic_nbsp(chars: &[char], index: usize) -> bool {
    let cp = chars[index] as u32;
    if cp == 0x00A0 {
        return true;
    }
    if !matches!(cp, 0x202F | 0x2007 | 0x2009) {
        return false;
    }
    let left = index.checked_sub(1).map(|at| chars[at]);
    let right = chars.get(index + 1).copied();
    if left.is_some_and(unicode::is_digit) || right.is_some_and(unicode::is_digit) {
        return true;
    }
    left.is_some_and(|ch| matches!(ch, '«' | '\u{2039}'))
        || right.is_some_and(|ch| matches!(ch, '»' | '\u{203A}' | ':' | ';' | '!' | '?'))
}

/// Normalize exotic `Zs` space-substitution watermarks to ASCII spaces.
pub fn normalize_exotic_spaces(text: &str, force_innamark_clean: bool) -> (String, usize) {
    if !text.chars().any(|ch| is_exotic_space(ch as u32)) {
        return (text.to_owned(), 0);
    }
    let chars: Vec<char> = text.chars().collect();
    let mut count = 0;
    let mut out = String::with_capacity(text.len());
    for (index, &ch) in chars.iter().enumerate() {
        if !is_exotic_space(ch as u32) {
            out.push(ch);
            continue;
        }
        let between_selectors = ch == '\u{200A}'
            && index > 0
            && index + 1 < chars.len()
            && variation_selector_to_byte(chars[index - 1]).is_some()
            && variation_selector_to_byte(chars[index + 1]).is_some();
        if between_selectors {
            count += 1;
        } else if !force_innamark_clean && is_legitimate_typographic_nbsp(&chars, index) {
            out.push(ch);
        } else {
            out.push(' ');
            count += 1;
        }
    }
    (out, count)
}

static CONFUSABLE_TO_LATIN: LazyLock<HashMap<char, char>> = LazyLock::new(|| {
    let pairs = [
        ('\u{0410}', 'A'),
        ('\u{0412}', 'B'),
        ('\u{0421}', 'C'),
        ('\u{0415}', 'E'),
        ('\u{041d}', 'H'),
        ('\u{0406}', 'I'),
        ('\u{0408}', 'J'),
        ('\u{041a}', 'K'),
        ('\u{041c}', 'M'),
        ('\u{041e}', 'O'),
        ('\u{0420}', 'P'),
        ('\u{0405}', 'S'),
        ('\u{0422}', 'T'),
        ('\u{0425}', 'X'),
        ('\u{04ae}', 'Y'),
        ('\u{0430}', 'a'),
        ('\u{0441}', 'c'),
        ('\u{0501}', 'd'),
        ('\u{0435}', 'e'),
        ('\u{04bb}', 'h'),
        ('\u{0456}', 'i'),
        ('\u{0458}', 'j'),
        ('\u{043e}', 'o'),
        ('\u{0440}', 'p'),
        ('\u{0455}', 's'),
        ('\u{0443}', 'y'),
        ('\u{0445}', 'x'),
        ('\u{0391}', 'A'),
        ('\u{0392}', 'B'),
        ('\u{0395}', 'E'),
        ('\u{0396}', 'Z'),
        ('\u{0397}', 'H'),
        ('\u{0399}', 'I'),
        ('\u{039a}', 'K'),
        ('\u{039c}', 'M'),
        ('\u{039d}', 'N'),
        ('\u{039f}', 'O'),
        ('\u{03a1}', 'P'),
        ('\u{03a4}', 'T'),
        ('\u{03a5}', 'Y'),
        ('\u{03a7}', 'X'),
        ('\u{03b1}', 'a'),
        ('\u{03b9}', 'i'),
        ('\u{03bf}', 'o'),
        ('\u{03c1}', 'p'),
        ('\u{03bd}', 'v'),
        ('\u{0578}', 'n'),
        ('\u{0585}', 'o'),
        ('\u{13a0}', 'D'),
        ('\u{13a1}', 'R'),
        ('\u{13a2}', 'T'),
        ('\u{13aa}', 'A'),
        ('\u{13ab}', 'J'),
        ('\u{13ac}', 'E'),
        ('\u{13b3}', 'W'),
        ('\u{13b7}', 'M'),
        ('\u{13bb}', 'H'),
        ('\u{13c0}', 'G'),
        ('\u{13c2}', 'h'),
        ('\u{13c3}', 'Z'),
        ('\u{13cf}', 'b'),
        ('\u{13d9}', 'V'),
        ('\u{13de}', 'L'),
        ('\u{13df}', 'C'),
        ('\u{13e2}', 'P'),
        ('\u{13e6}', 'K'),
        ('\u{13f4}', 'B'),
        ('\u{0251}', 'a'),
        ('\u{0261}', 'g'),
        ('\u{1d0f}', 'o'),
        ('\u{212a}', 'K'),
        ('\u{0423}', 'Y'),
        ('\u{043a}', 'k'),
    ];
    let fullwidth = (0..26u8).flat_map(|offset| {
        [
            (
                char::from_u32(0xFF21 + u32::from(offset)).unwrap_or('A'),
                char::from(b'A' + offset),
            ),
            (
                char::from_u32(0xFF41 + u32::from(offset)).unwrap_or('a'),
                char::from(b'a' + offset),
            ),
        ]
    });
    pairs.into_iter().chain(fullwidth).collect()
});

static CONFUSABLE_TO_CYRILLIC: LazyLock<HashMap<char, char>> = LazyLock::new(|| {
    [
        ('A', '\u{0410}'),
        ('B', '\u{0412}'),
        ('C', '\u{0421}'),
        ('E', '\u{0415}'),
        ('H', '\u{041d}'),
        ('I', '\u{0406}'),
        ('J', '\u{0408}'),
        ('K', '\u{041a}'),
        ('M', '\u{041c}'),
        ('O', '\u{041e}'),
        ('P', '\u{0420}'),
        ('S', '\u{0405}'),
        ('T', '\u{0422}'),
        ('X', '\u{0425}'),
        ('Y', '\u{0423}'),
        ('a', '\u{0430}'),
        ('c', '\u{0441}'),
        ('e', '\u{0435}'),
        ('i', '\u{0456}'),
        ('j', '\u{0458}'),
        ('k', '\u{043a}'),
        ('o', '\u{043e}'),
        ('p', '\u{0440}'),
        ('s', '\u{0455}'),
        ('x', '\u{0445}'),
        ('y', '\u{0443}'),
        ('\u{0391}', '\u{0410}'),
        ('\u{0392}', '\u{0412}'),
        ('\u{0395}', '\u{0415}'),
        ('\u{0397}', '\u{041d}'),
        ('\u{0399}', '\u{0406}'),
        ('\u{039a}', '\u{041a}'),
        ('\u{039c}', '\u{041c}'),
        ('\u{039f}', '\u{041e}'),
        ('\u{03a1}', '\u{0420}'),
        ('\u{03a4}', '\u{0422}'),
        ('\u{03a5}', '\u{0423}'),
        ('\u{03a7}', '\u{0425}'),
        ('\u{03b1}', '\u{0430}'),
        ('\u{03b9}', '\u{0456}'),
        ('\u{03bf}', '\u{043e}'),
        ('\u{03c1}', '\u{0440}'),
    ]
    .into_iter()
    .collect()
});

fn is_native_latin_letter(ch: char) -> bool {
    if ch.is_ascii_alphabetic() {
        return true;
    }
    !CONFUSABLE_TO_LATIN.contains_key(&ch) && unicode::name_has_latin(ch)
}

fn is_native_cyrillic_letter(ch: char) -> bool {
    if ('\u{0400}'..='\u{052F}').contains(&ch) {
        return true;
    }
    !CONFUSABLE_TO_CYRILLIC.contains_key(&ch) && unicode::name_has_cyrillic(ch)
}

fn is_transparent_in_word(ch: char) -> bool {
    is_invisible_or_stego(ch as u32)
}

/// Normalize confusable homoglyphs inside predominantly Latin or Cyrillic words.
pub fn normalize_mixed_script_homoglyphs(text: &str) -> (String, usize) {
    let chars: Vec<char> = text.chars().collect();
    let mut out = chars.clone();
    let mut replaced = 0;
    let mut span_counts: Option<(usize, usize)> = None;
    let mut index = 0;
    while index < chars.len() {
        let ch = chars[index];
        if !(unicode::is_alpha(ch) || is_transparent_in_word(ch)) {
            index += 1;
            continue;
        }
        let mut letters = Vec::new();
        while index < chars.len() && (unicode::is_alpha(chars[index]) || is_transparent_in_word(chars[index]))
        {
            if unicode::is_alpha(chars[index]) {
                letters.push(index);
            }
            index += 1;
        }
        if letters.len() < 2 {
            continue;
        }
        let latin_conf: Vec<usize> = letters
            .iter()
            .copied()
            .filter(|&at| CONFUSABLE_TO_LATIN.contains_key(&chars[at]))
            .collect();
        let cyrillic_conf: Vec<usize> = letters
            .iter()
            .copied()
            .filter(|&at| CONFUSABLE_TO_CYRILLIC.contains_key(&chars[at]))
            .collect();
        if latin_conf.is_empty() && cyrillic_conf.is_empty() {
            continue;
        }
        let latin_native = if latin_conf.is_empty() {
            0
        } else {
            letters
                .iter()
                .filter(|&&at| is_native_latin_letter(chars[at]))
                .count()
        };
        let can_latin =
            latin_native > 0 && !latin_conf.is_empty() && latin_native + latin_conf.len() == letters.len();
        let cyrillic_native = if cyrillic_conf.is_empty() {
            0
        } else {
            letters
                .iter()
                .filter(|&&at| is_native_cyrillic_letter(chars[at]))
                .count()
        };
        let can_cyrillic = cyrillic_native > 0
            && !cyrillic_conf.is_empty()
            && cyrillic_native + cyrillic_conf.len() == letters.len();
        let prefer_latin = if can_latin && can_cyrillic && latin_native == cyrillic_native {
            let (latin, cyrillic) = *span_counts.get_or_insert_with(|| {
                (
                    chars
                        .iter()
                        .filter(|&&c| unicode::is_alpha(c) && is_native_latin_letter(c))
                        .count(),
                    chars
                        .iter()
                        .filter(|&&c| unicode::is_alpha(c) && is_native_cyrillic_letter(c))
                        .count(),
                )
            });
            latin >= cyrillic
        } else {
            !can_cyrillic || latin_native > cyrillic_native
        };
        if can_latin && prefer_latin {
            for at in latin_conf {
                out[at] = CONFUSABLE_TO_LATIN[&chars[at]];
                replaced += 1;
            }
        } else if can_cyrillic {
            for at in cyrillic_conf {
                out[at] = CONFUSABLE_TO_CYRILLIC[&chars[at]];
                replaced += 1;
            }
        }
    }
    (out.into_iter().collect(), replaced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homoglyphs_follow_the_majority_script() {
        let spoofed = "\u{043f}a\u{0440}o\u{043b}\u{044c} c\u{043e}\u{0443}\u{0441} p\u{ff41}ssword";
        assert_eq!(
            normalize_mixed_script_homoglyphs(spoofed).0,
            "пароль \u{0441}\u{043e}\u{0443}\u{0441} password"
        );
    }

    #[test]
    fn plain_citations_keep_local_spacing() {
        assert_eq!(
            strip_assistant_citations("It is ready citeturn0search0.", true).0,
            "It is ready."
        );
        assert_eq!(
            strip_assistant_citations("See the result citeturn0search0 and continue.", true).0,
            "See the result and continue."
        );
    }
}

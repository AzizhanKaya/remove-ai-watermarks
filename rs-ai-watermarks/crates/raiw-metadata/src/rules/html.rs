//! `html.unescape` for the character references XMP packets carry.

use std::sync::LazyLock;

use regex::{Captures, Regex};

static CHARREF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\x0c <&#;]{1,32};?)")
        .unwrap_or_else(|err| unreachable!("static regex: {err}"))
});

/// Named references with their `;`; the legacy subset also matches without it.
const NAMED: &[(&str, &str, bool)] = &[
    ("amp", "&", true),
    ("lt", "<", true),
    ("gt", ">", true),
    ("quot", "\"", true),
    ("apos", "'", false),
    ("nbsp", "\u{a0}", true),
    ("copy", "\u{a9}", true),
    ("reg", "\u{ae}", true),
    ("deg", "\u{b0}", true),
    ("shy", "\u{ad}", true),
    ("laquo", "\u{ab}", true),
    ("raquo", "\u{bb}", true),
    ("middot", "\u{b7}", true),
    ("times", "\u{d7}", true),
    ("hellip", "\u{2026}", false),
    ("mdash", "\u{2014}", false),
    ("ndash", "\u{2013}", false),
    ("lsquo", "\u{2018}", false),
    ("rsquo", "\u{2019}", false),
    ("ldquo", "\u{201c}", false),
    ("rdquo", "\u{201d}", false),
    ("trade", "\u{2122}", false),
    ("euro", "\u{20ac}", false),
];

/// Windows-1252 remaps and the other `_invalid_charrefs` of the `html` module.
fn invalid_charref(num: u32) -> Option<char> {
    let mapped = match num {
        0x00 => '\u{fffd}',
        0x0d => '\r',
        0x80 => '\u{20ac}',
        0x81 => '\u{81}',
        0x82 => '\u{201a}',
        0x83 => '\u{192}',
        0x84 => '\u{201e}',
        0x85 => '\u{2026}',
        0x86 => '\u{2020}',
        0x87 => '\u{2021}',
        0x88 => '\u{2c6}',
        0x89 => '\u{2030}',
        0x8a => '\u{160}',
        0x8b => '\u{2039}',
        0x8c => '\u{152}',
        0x8d => '\u{8d}',
        0x8e => '\u{17d}',
        0x8f => '\u{8f}',
        0x90 => '\u{90}',
        0x91 => '\u{2018}',
        0x92 => '\u{2019}',
        0x93 => '\u{201c}',
        0x94 => '\u{201d}',
        0x95 => '\u{2022}',
        0x96 => '\u{2013}',
        0x97 => '\u{2014}',
        0x98 => '\u{2dc}',
        0x99 => '\u{2122}',
        0x9a => '\u{161}',
        0x9b => '\u{203a}',
        0x9c => '\u{153}',
        0x9d => '\u{9d}',
        0x9e => '\u{17e}',
        0x9f => '\u{178}',
        _ => return None,
    };
    Some(mapped)
}

fn invalid_codepoint(num: u32) -> bool {
    matches!(num, 0x1..=0x8 | 0xb | 0xe..=0x1f | 0x7f..=0x9f | 0xfdd0..=0xfdef) || (num & 0xfffe) == 0xfffe
}

fn numeric(body: &str) -> String {
    let trimmed = body.trim_end_matches(';');
    let parsed = if let Some(hex) = trimmed.strip_prefix(['x', 'X']) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u64>().ok()
    };
    let Some(num) = parsed else {
        return "\u{fffd}".into();
    };
    let Ok(num) = u32::try_from(num) else {
        return "\u{fffd}".into();
    };
    if let Some(mapped) = invalid_charref(num) {
        return mapped.to_string();
    }
    if (0xD800..=0xDFFF).contains(&num) || num > 0x0010_FFFF {
        return "\u{fffd}".into();
    }
    if invalid_codepoint(num) {
        return String::new();
    }
    char::from_u32(num).map_or_else(|| "\u{fffd}".into(), |c| c.to_string())
}

fn lookup(name: &str) -> Option<&'static str> {
    if let Some(stem) = name.strip_suffix(';') {
        NAMED.iter().find(|(n, _, _)| *n == stem).map(|(_, v, _)| *v)
    } else {
        NAMED
            .iter()
            .find(|(n, _, legacy)| *legacy && *n == name)
            .map(|(_, v, _)| *v)
    }
}

fn named(body: &str) -> String {
    if let Some(value) = lookup(body) {
        return value.to_owned();
    }
    let chars: Vec<(usize, char)> = body.char_indices().collect();
    for x in (2..chars.len()).rev() {
        let split = chars[x].0;
        if let Some(value) = lookup(&body[..split]) {
            return format!("{value}{}", &body[split..]);
        }
    }
    format!("&{body}")
}

/// `html.unescape(text)`.
pub fn unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    CHARREF
        .replace_all(text, |caps: &Captures<'_>| {
            let body = &caps[1];
            body.strip_prefix('#').map_or_else(|| named(body), numeric)
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::unescape;

    #[test]
    fn references() {
        assert_eq!(unescape("{&quot;Label&quot;:&#34;1&#x22;}"), r#"{"Label":"1"}"#);
        assert_eq!(unescape("a &amp b &unknown; &#0;"), "a & b &unknown; \u{fffd}");
        assert_eq!(unescape("&#150;"), "\u{2013}");
    }
}

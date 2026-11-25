//! CPython 3.12 (Unicode 15.0) character properties and `str` predicates.
//!
//! The tables are generated from the reference interpreter by `tools/gen_data.py`, so
//! `isalpha`, `isspace`, `casefold`, regex shorthands and friends agree with Python exactly.

use std::collections::HashMap;
use std::sync::LazyLock;

use unicode_normalization::UnicodeNormalization;

const TABLE_BYTES: &[u8] = include_bytes!("../data/unicode.bin");
const CASEFOLD: &str = include_str!("../data/casefold.txt");

/// Unicode general category, in the generator's index order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Category {
    Cn,
    Lu,
    Ll,
    Lt,
    Lm,
    Lo,
    Mn,
    Mc,
    Me,
    Nd,
    Nl,
    No,
    Pc,
    Pd,
    Ps,
    Pe,
    Pi,
    Pf,
    Po,
    Sm,
    Sc,
    Sk,
    So,
    Zs,
    Zl,
    Zp,
    Cc,
    Cf,
    Cs,
    Co,
}

impl Category {
    const ALL: [Self; 30] = [
        Self::Cn,
        Self::Lu,
        Self::Ll,
        Self::Lt,
        Self::Lm,
        Self::Lo,
        Self::Mn,
        Self::Mc,
        Self::Me,
        Self::Nd,
        Self::Nl,
        Self::No,
        Self::Pc,
        Self::Pd,
        Self::Ps,
        Self::Pe,
        Self::Pi,
        Self::Pf,
        Self::Po,
        Self::Sm,
        Self::Sc,
        Self::Sk,
        Self::So,
        Self::Zs,
        Self::Zl,
        Self::Zp,
        Self::Cc,
        Self::Cf,
        Self::Cs,
        Self::Co,
    ];

    pub const fn is_letter(self) -> bool {
        matches!(self, Self::Lu | Self::Ll | Self::Lt | Self::Lm | Self::Lo)
    }

    pub const fn is_mark(self) -> bool {
        matches!(self, Self::Mn | Self::Mc | Self::Me)
    }

    pub const fn is_number(self) -> bool {
        matches!(self, Self::Nd | Self::Nl | Self::No)
    }

    pub const fn is_punctuation(self) -> bool {
        matches!(
            self,
            Self::Pc | Self::Pd | Self::Ps | Self::Pe | Self::Pi | Self::Pf | Self::Po
        )
    }
}

const RTL: u8 = 1;
const LATIN: u8 = 1 << 1;
const CYRILLIC: u8 = 1 << 2;
const UPPER: u8 = 1 << 3;
const LOWER: u8 = 1 << 4;
const DIGIT: u8 = 1 << 5;
const SPACE: u8 = 1 << 6;

#[derive(Debug, Clone, Copy)]
struct Props {
    category: Category,
    flags: u8,
    combining: u8,
    word: u16,
}

#[derive(Debug)]
struct Table {
    starts: Vec<u32>,
    props: Vec<Props>,
    words: Vec<&'static str>,
}

static TABLE: LazyLock<Table> = LazyLock::new(load_table);

fn load_table() -> Table {
    let read_u32 = |at: usize| {
        u32::from_le_bytes([
            TABLE_BYTES[at],
            TABLE_BYTES[at + 1],
            TABLE_BYTES[at + 2],
            TABLE_BYTES[at + 3],
        ])
    };
    assert_eq!(&TABLE_BYTES[..4], b"RUP1", "corrupt Unicode table");
    let count = read_u32(4) as usize;
    let mut starts = Vec::with_capacity(count);
    let mut props = Vec::with_capacity(count);
    for index in 0..count {
        let at = 12 + index * 10;
        starts.push(read_u32(at));
        props.push(Props {
            category: Category::ALL[TABLE_BYTES[at + 4] as usize],
            flags: TABLE_BYTES[at + 5],
            combining: TABLE_BYTES[at + 6],
            word: u16::from_le_bytes([TABLE_BYTES[at + 8], TABLE_BYTES[at + 9]]),
        });
    }
    let words_text =
        std::str::from_utf8(&TABLE_BYTES[12 + count * 10..]).expect("Unicode table words are UTF-8");
    Table {
        starts,
        props,
        words: words_text.split('\n').collect(),
    }
}

fn props(ch: char) -> Props {
    let table = &*TABLE;
    let index = table.starts.partition_point(|&start| start <= ch as u32) - 1;
    table.props[index]
}

/// `unicodedata.category(ch)`.
pub fn category(ch: char) -> Category {
    props(ch).category
}

/// `str.isalpha()` for one character (categories `L*`).
pub fn is_alpha(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_alphabetic();
    }
    category(ch).is_letter()
}

/// `str.isspace()` for one character.
pub fn is_space(ch: char) -> bool {
    if ch.is_ascii() {
        return matches!(ch, '\t'..='\r' | '\x1c'..='\x1f' | ' ');
    }
    props(ch).flags & SPACE != 0
}

/// `str.isdigit()` for one character (Numeric_Type Decimal or Digit).
pub fn is_digit(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_digit();
    }
    props(ch).flags & DIGIT != 0
}

/// Python `\d` / `str.isdecimal()` for one character (category `Nd`).
pub fn is_decimal(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_digit();
    }
    category(ch) == Category::Nd
}

/// Python `\w`: letters, numbers and underscore.
pub fn is_word(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_alphanumeric() || ch == '_';
    }
    let category = category(ch);
    category.is_letter() || category.is_number()
}

pub fn is_upper(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_uppercase();
    }
    props(ch).flags & UPPER != 0
}

pub fn is_lower(ch: char) -> bool {
    if ch.is_ascii() {
        return ch.is_ascii_lowercase();
    }
    props(ch).flags & LOWER != 0
}

/// `str.isprintable()` for one character.
pub fn is_printable(ch: char) -> bool {
    if ch == ' ' {
        return true;
    }
    !matches!(
        category(ch),
        Category::Cc
            | Category::Cf
            | Category::Cs
            | Category::Co
            | Category::Cn
            | Category::Zl
            | Category::Zp
            | Category::Zs
    )
}

/// Bidirectional class `R`, `AL` or `AN`.
pub fn is_rtl(ch: char) -> bool {
    props(ch).flags & RTL != 0
}

/// Whether the character name contains `LATIN`.
pub fn name_has_latin(ch: char) -> bool {
    props(ch).flags & LATIN != 0
}

/// Whether the character name contains `CYRILLIC`.
pub fn name_has_cyrillic(ch: char) -> bool {
    props(ch).flags & CYRILLIC != 0
}

/// First word of `unicodedata.name(ch, "UNKNOWN")` for alphabetic characters, else `""`.
pub fn script_word(ch: char) -> &'static str {
    let word = props(ch).word;
    TABLE.words.get(word as usize).copied().unwrap_or("")
}

/// `unicodedata.combining(ch)`.
pub fn combining(ch: char) -> u8 {
    if ch.is_ascii() {
        return 0;
    }
    props(ch).combining
}

static FOLDS: LazyLock<HashMap<char, Vec<char>>> = LazyLock::new(|| {
    let parse = |hex: &str| u32::from_str_radix(hex, 16).ok().and_then(char::from_u32);
    CASEFOLD
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(' ').map(parse);
            let source = fields.next()??;
            let target: Option<Vec<char>> = fields.collect();
            Some((source, target?))
        })
        .collect()
});

/// `str.casefold()`.
pub fn casefold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            out.push(ch.to_ascii_lowercase());
        } else if let Some(target) = FOLDS.get(&ch) {
            out.extend(target);
        } else {
            out.push(ch);
        }
    }
    out
}

/// `str.lower()`.
pub fn lower(text: &str) -> String {
    if text.is_ascii() {
        return text.to_ascii_lowercase();
    }
    text.to_lowercase()
}

/// `str.isupper()`: at least one cased character and no lowercase or titlecase ones.
pub fn str_is_upper(text: &str) -> bool {
    let mut cased = false;
    for ch in text.chars() {
        if is_lower(ch) || category(ch) == Category::Lt {
            return false;
        }
        cased |= is_upper(ch);
    }
    cased
}

/// `str.isspace()`.
pub fn str_is_space(text: &str) -> bool {
    !text.is_empty() && text.chars().all(is_space)
}

/// `str.isprintable()`.
pub fn str_is_printable(text: &str) -> bool {
    text.chars().all(is_printable)
}

/// `str.strip()` with Python's whitespace definition.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// `str.lstrip()`.
pub fn lstrip(text: &str) -> &str {
    text.trim_start_matches(is_space)
}

/// `str.rstrip()`.
pub fn rstrip(text: &str) -> &str {
    text.trim_end_matches(is_space)
}

/// `str.split()` without arguments.
pub fn split_whitespace(text: &str) -> impl Iterator<Item = &str> {
    text.split(is_space).filter(|piece| !piece.is_empty())
}

/// `" ".join(text.split())`.
pub fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for piece in split_whitespace(text) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(piece);
    }
    out
}

/// `" ".join(text.casefold().split())`.
pub fn fold_words(text: &str) -> String {
    collapse_whitespace(&casefold(text))
}

/// `unicodedata.normalize("NFC", text)`.
pub fn nfc(text: &str) -> String {
    if text.is_ascii() {
        return text.to_owned();
    }
    text.nfc().collect()
}

/// `str.splitlines(keepends=True)`.
pub fn splitlines_keepends(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        let is_break = matches!(
            ch,
            '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}'
        );
        if !is_break {
            continue;
        }
        let mut end = index + ch.len_utf8();
        if ch == '\r' && chars.peek().is_some_and(|&(_, next)| next == '\n') {
            chars.next();
            end += 1;
        }
        lines.push(&text[start..end]);
        start = end;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// `str.splitlines()` without line endings.
pub fn splitlines(text: &str) -> Vec<&str> {
    splitlines_keepends(text)
        .into_iter()
        .map(|line| {
            line.trim_end_matches([
                '\n', '\r', '\x0b', '\x0c', '\x1c', '\x1d', '\x1e', '\u{85}', '\u{2028}', '\u{2029}',
            ])
        })
        .collect()
}

/// Decimal value of a Unicode `Nd` character (`int(ch)`).
pub fn decimal_value(ch: char) -> Option<u32> {
    if ch.is_ascii_digit() {
        return Some(ch as u32 - '0' as u32);
    }
    if !is_decimal(ch) {
        return None;
    }
    let mut zero = ch as u32;
    while zero > 0 && char::from_u32(zero - 1).is_some_and(is_decimal) {
        zero -= 1;
    }
    Some((ch as u32 - zero) % 10)
}

/// Inclusive codepoint ranges where `predicate` holds, for building regex classes.
pub(crate) fn ranges_where(predicate: impl Fn(Category, u8) -> bool) -> Vec<(u32, u32)> {
    let table = &*TABLE;
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (index, (&start, props)) in table.starts.iter().zip(&table.props).enumerate() {
        if !predicate(props.category, props.flags) {
            continue;
        }
        let end = table.starts.get(index + 1).map_or(0x10_FFFF, |next| next - 1);
        match out.last_mut() {
            Some(last) if last.1 + 1 == start => last.1 = end,
            _ => out.push((start, end)),
        }
    }
    out
}

pub(crate) const SPACE_FLAG: u8 = SPACE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicates_follow_python() {
        assert!(is_alpha('é') && !is_alpha('\u{301}') && is_alpha('ж'));
        assert!(is_space('\x1c') && is_space('\u{3000}') && !is_space('\u{200b}'));
        assert!(is_digit('²') && !is_decimal('²') && is_decimal('٣'));
        assert_eq!(decimal_value('٣'), Some(3));
        assert_eq!(decimal_value('\u{1d7d9}'), Some(1));
        assert!(is_word('_') && is_word('²') && !is_word('\u{301}'));
        assert_eq!(casefold("Straße ﬁ"), "strasse fi");
        assert!(str_is_upper("NASA-1") && !str_is_upper("Nasa") && !str_is_upper("123"));
        assert_eq!(script_word('ж'), "CYRILLIC");
        assert_eq!(script_word('漢'), "CJK");
        assert_eq!(script_word('1'), "");
        assert!(name_has_latin('ａ') && !name_has_latin('а'));
        assert_eq!(combining('\u{301}'), 230);
        assert!(is_rtl('ب') && !is_rtl('a'));
        assert!(!is_printable('\u{200b}') && is_printable(' ') && !is_printable('\u{a0}'));
    }

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(
            splitlines_keepends("a\r\nb\rc\nd\u{2028}e"),
            ["a\r\n", "b\r", "c\n", "d\u{2028}", "e"]
        );
        assert_eq!(splitlines("a\n\nb\n"), ["a", "", "b"]);
    }
}

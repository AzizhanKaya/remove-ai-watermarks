//! Python `re` patterns on top of `fancy-regex`.
//!
//! Patterns are written in Python syntax and translated so that `\w`, `\d`, `\s`, `\b`,
//! `$`, `\Z` and character-class literals keep CPython's semantics (Unicode 15.0 tables).

use std::borrow::Cow;
use std::fmt::Write as _;
use std::ops::Range;
use std::sync::{LazyLock, OnceLock};

use fancy_regex::{Captures, Regex, RegexBuilder, RegexInput};

use crate::unicode;

/// Python compile flags.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flags {
    pub ignore_case: bool,
    pub multiline: bool,
    pub dotall: bool,
}

impl Flags {
    pub const NONE: Self = Self {
        ignore_case: false,
        multiline: false,
        dotall: false,
    };
    pub const I: Self = Self {
        ignore_case: true,
        multiline: false,
        dotall: false,
    };
    pub const M: Self = Self {
        ignore_case: false,
        multiline: true,
        dotall: false,
    };
}

fn class_body(ranges: &[(u32, u32)]) -> String {
    let mut out = String::new();
    for &(start, end) in ranges {
        push_class_char(&mut out, start);
        if end > start {
            out.push('-');
            push_class_char(&mut out, end);
        }
    }
    out
}

fn push_class_char(out: &mut String, cp: u32) {
    let _ = write!(out, "\\x{{{cp:X}}}");
}

// CPython's `\w` is exactly `L|N|_` and `\d` is `Nd` (checked over Unicode 15.0). Property
// classes compile far faster than explicit range lists in `regex-syntax`.
static WORD: LazyLock<String> = LazyLock::new(|| r"\p{L}\p{N}_".to_owned());
static DIGIT: LazyLock<String> = LazyLock::new(|| r"\p{Nd}".to_owned());
static SPACE: LazyLock<String> = LazyLock::new(|| {
    let mut ranges = unicode::ranges_where(|_, flags| flags & unicode::SPACE_FLAG != 0);
    ranges.extend([(0x09, 0x0D), (0x1C, 0x20)]);
    ranges.sort_unstable();
    class_body(&ranges)
});

fn shorthand(letter: char) -> Option<(&'static str, bool)> {
    match letter {
        'w' => Some((WORD.as_str(), false)),
        'W' => Some((WORD.as_str(), true)),
        'd' => Some((DIGIT.as_str(), false)),
        'D' => Some((DIGIT.as_str(), true)),
        's' => Some((SPACE.as_str(), false)),
        'S' => Some((SPACE.as_str(), true)),
        _ => None,
    }
}

/// Translate a Python pattern into `fancy-regex` syntax.
pub fn translate(pattern: &str, flags: Flags) -> String {
    let word = WORD.as_str();
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() * 2);
    if flags.ignore_case {
        out.push_str("(?i)");
    }
    if flags.dotall {
        out.push_str("(?s)");
    }
    let mut index = 0;
    let mut in_class = false;
    let mut class_start = 0;
    while index < chars.len() {
        let ch = chars[index];
        if ch == '\\' && index + 1 < chars.len() {
            let next = chars[index + 1];
            index += 2;
            if let Some((body, negated)) = shorthand(next) {
                match (in_class, negated) {
                    (false, false) => {
                        let _ = write!(out, "[{body}]");
                    }
                    (_, true) => {
                        let _ = write!(out, "[^{body}]");
                    }
                    (true, false) => out.push_str(body),
                }
                continue;
            }
            match next {
                'b' if !in_class => {
                    let _ = write!(out, "(?:(?<=[{word}])(?![{word}])|(?<![{word}])(?=[{word}]))");
                }
                'B' if !in_class => {
                    let _ = write!(out, "(?:(?<=[{word}])(?=[{word}])|(?<![{word}])(?![{word}]))");
                }
                'b' => out.push_str("\\x08"),
                'Z' => out.push_str("\\z"),
                'u' | 'U' => {
                    let width = if next == 'u' { 4 } else { 8 };
                    let hex: String = chars[index..(index + width).min(chars.len())].iter().collect();
                    index += hex.len();
                    let _ = write!(out, "\\x{{{hex}}}");
                }
                _ => {
                    out.push('\\');
                    out.push(next);
                }
            }
            continue;
        }
        index += 1;
        if in_class {
            match ch {
                ']' if index - 1 > class_start => {
                    in_class = false;
                    out.push(']');
                }
                '[' | '&' | '~' => {
                    out.push('\\');
                    out.push(ch);
                }
                '-' => {
                    let at_edge = index - 1 == class_start || chars.get(index) == Some(&']');
                    out.push_str(if at_edge { "\\-" } else { "-" });
                }
                _ => out.push(ch),
            }
            continue;
        }
        match ch {
            '[' => {
                in_class = true;
                out.push('[');
                if chars.get(index) == Some(&'^') {
                    out.push('^');
                    index += 1;
                }
                class_start = index;
                if chars.get(index) == Some(&']') {
                    out.push_str("\\]");
                    index += 1;
                }
            }
            '$' if flags.multiline => out.push_str("(?m:$)"),
            '$' => out.push_str("(?=\\n?\\z)"),
            '^' if flags.multiline => out.push_str("(?m:^)"),
            _ => out.push(ch),
        }
    }
    out
}

/// Python `re.escape` restricted to the characters that matter in our translated syntax.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        if matches!(
            ch,
            '\\' | '.'
                | '+'
                | '*'
                | '?'
                | '('
                | ')'
                | '|'
                | '['
                | ']'
                | '{'
                | '}'
                | '^'
                | '$'
                | '-'
                | '#'
                | '&'
                | '~'
        ) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

/// Index just past the character class starting at `chars[start] == '['`.
fn skip_class(chars: &[char], start: usize) -> usize {
    let mut index = start + 1;
    if chars.get(index) == Some(&'^') {
        index += 1;
    }
    if chars.get(index) == Some(&']') {
        index += 1;
    }
    let mut depth = 1;
    while index < chars.len() && depth > 0 {
        match chars[index] {
            '\\' => index += 1,
            '[' => depth += 1,
            ']' => depth -= 1,
            _ => {}
        }
        index += 1;
    }
    index
}

/// Index just past the group opened at `chars[start] == '('`.
fn skip_group(chars: &[char], start: usize) -> usize {
    let mut index = start + 1;
    let mut depth = 1;
    while index < chars.len() && depth > 0 {
        match chars[index] {
            '\\' => index += 2,
            '[' => index = skip_class(chars, index),
            '(' => {
                depth += 1;
                index += 1;
            }
            ')' => {
                depth -= 1;
                index += 1;
            }
            _ => index += 1,
        }
    }
    index
}

/// Replace every look-around group of a translated pattern with `(?:)`.
///
/// Look-arounds only restrict where a match may occur, so the result matches wherever the
/// original can: it is a sound prefilter for candidate start positions. `None` when the pattern
/// has no look-arounds.
fn strip_lookarounds(pattern: &str) -> Option<String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len());
    let mut found = false;
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            '\\' => {
                out.extend(&chars[index..(index + 2).min(chars.len())]);
                index += 2;
            }
            '[' => {
                let end = skip_class(&chars, index);
                out.extend(&chars[index..end.min(chars.len())]);
                index = end;
            }
            '(' => {
                let head: String = chars[index + 1..(index + 4).min(chars.len())].iter().collect();
                if head.starts_with("?=") || head.starts_with("?!") || head == "?<=" || head == "?<!" {
                    found = true;
                    out.push_str("(?:)");
                    index = skip_group(&chars, index);
                } else {
                    out.push('(');
                    index += 1;
                }
            }
            ch => {
                out.push(ch);
                index += 1;
            }
        }
    }
    found.then_some(out)
}

fn build_prefilter(translated: &str) -> Option<regex::Regex> {
    let stripped = strip_lookarounds(translated)?;
    regex::RegexBuilder::new(&stripped).size_limit(64 << 20).dfa_size_limit(64 << 20).build().ok()
}

/// A compiled Python pattern.
#[derive(Debug)]
pub struct PyRegex {
    regex: Regex,
    /// Look-around-free superset pattern that locates candidate match starts quickly.
    prefilter: Option<regex::Regex>,
    source: String,
    flags: Flags,
    full: OnceLock<Regex>,
}

/// Leftmost matches of a [`PyRegex`], with Python's iteration rules.
#[derive(Debug)]
pub struct Matches<'r, 't> {
    regex: &'r PyRegex,
    text: &'t str,
    position: usize,
    done: bool,
}

impl<'t> Iterator for Matches<'_, 't> {
    type Item = Captures<'t, str>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let Some(caps) = self.regex.captures_from(self.text, self.position) else {
            self.done = true;
            return None;
        };
        let span = whole(&caps);
        self.position = if span.end > span.start {
            span.end
        } else {
            match self.text[span.end..].chars().next() {
                Some(ch) => span.end + ch.len_utf8(),
                None => {
                    self.done = true;
                    span.end
                }
            }
        };
        Some(caps)
    }
}

/// One match with byte offsets into the searched text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn range(self) -> Range<usize> {
        self.start..self.end
    }
}

fn build(pattern: &str) -> Result<Regex, fancy_regex::Error> {
    RegexBuilder::new(pattern).backtrack_limit(50_000_000).build()
}

impl PyRegex {
    /// Compile a Python pattern; static patterns are trusted source code.
    pub fn new(pattern: &str, flags: Flags) -> Self {
        Self::try_new(pattern, flags)
            .unwrap_or_else(|err| panic!("invalid built-in pattern {pattern:?}: {err}"))
    }

    pub fn try_new(pattern: &str, flags: Flags) -> Result<Self, fancy_regex::Error> {
        let translated = translate(pattern, flags);
        Ok(Self {
            regex: build(&translated)?,
            prefilter: build_prefilter(&translated),
            source: pattern.to_owned(),
            flags,
            full: OnceLock::new(),
        })
    }

    /// Next candidate start at or after `from` (every position without a prefilter).
    fn candidates(&self, text: &str, from: usize) -> Option<usize> {
        match &self.prefilter {
            Some(prefilter) => prefilter.find_at(text, from).map(|m| m.start()),
            None => (from <= text.len()).then_some(from),
        }
    }

    fn next_candidate(text: &str, start: usize) -> Option<usize> {
        text[start..].chars().next().map(|ch| start + ch.len_utf8())
    }

    /// Leftmost-first captures starting at or after `from`.
    fn captures_from<'t>(&self, text: &'t str, from: usize) -> Option<Captures<'t, str>> {
        if self.prefilter.is_none() {
            return self.regex.captures_from_pos(text, from).ok().flatten();
        }
        let mut position = from;
        loop {
            let start = self.candidates(text, position)?;
            if let Some(caps) = self.captures_at(text, start) {
                return Some(caps);
            }
            position = Self::next_candidate(text, start)?;
        }
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    fn full_regex(&self) -> &Regex {
        self.full.get_or_init(|| {
            let translated = translate(&format!("(?:{})", self.source), self.flags);
            build(&format!("\\A(?:{translated})\\z"))
                .unwrap_or_else(|err| panic!("invalid pattern {:?}: {err}", self.source))
        })
    }

    /// `pattern.search(text, pos)`.
    pub fn search_from(&self, text: &str, pos: usize) -> Option<Span> {
        if self.prefilter.is_none() {
            return self.regex.find_from_pos(text, pos).ok().flatten().map(|m| Span {
                start: m.start(),
                end: m.end(),
            });
        }
        let mut position = pos;
        loop {
            let start = self.candidates(text, position)?;
            if let Some(span) = self.match_at(text, start) {
                return Some(span);
            }
            position = Self::next_candidate(text, start)?;
        }
    }

    /// `pattern.search(text)`.
    pub fn search(&self, text: &str) -> Option<Span> {
        self.search_from(text, 0)
    }

    pub fn is_match(&self, text: &str) -> bool {
        self.search(text).is_some()
    }

    /// `pattern.match(text, pos)`: anchored at `pos`, lookbehind sees the whole text.
    pub fn match_at(&self, text: &str, pos: usize) -> Option<Span> {
        let input = RegexInput::new(text).from_pos(pos).anchored(true);
        self.regex.find_input(input).ok().flatten().map(|m| Span {
            start: m.start(),
            end: m.end(),
        })
    }

    pub fn captures_at<'t>(&self, text: &'t str, pos: usize) -> Option<Captures<'t, str>> {
        let input = RegexInput::new(text).from_pos(pos).anchored(true);
        self.regex.captures_input(input).ok().flatten()
    }

    /// `pattern.fullmatch(text)`.
    pub fn fullmatch(&self, text: &str) -> bool {
        self.full_regex().is_match(text).unwrap_or(false)
    }

    pub fn full_captures<'t>(&self, text: &'t str) -> Option<Captures<'t, str>> {
        self.full_regex().captures(text).ok().flatten()
    }

    /// `pattern.finditer(text)` as spans.
    pub fn find_iter<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Span> + 'a {
        self.captures_iter(text).map(|caps| whole(&caps))
    }

    /// `pattern.finditer(text)` with groups.
    pub fn captures_iter<'r, 't>(&'r self, text: &'t str) -> Matches<'r, 't> {
        Matches {
            regex: self,
            text,
            position: 0,
            done: false,
        }
    }

    pub fn captures<'t>(&self, text: &'t str) -> Option<Captures<'t, str>> {
        self.captures_from(text, 0)
    }

    /// `pattern.findall(text)` for patterns without groups.
    pub fn findall<'a>(&'a self, text: &'a str) -> Vec<&'a str> {
        self.find_iter(text).map(|span| &text[span.range()]).collect()
    }

    pub fn count(&self, text: &str) -> usize {
        self.find_iter(text).count()
    }

    /// `pattern.sub(replacement, text)` with a literal replacement.
    pub fn sub<'t>(&self, replacement: &str, text: &'t str) -> Cow<'t, str> {
        self.sub_with(text, |_| replacement.to_owned())
    }

    /// `pattern.subn(func, text)`.
    pub fn sub_with<'t>(
        &self,
        text: &'t str,
        mut replace: impl FnMut(&Captures<'_, str>) -> String,
    ) -> Cow<'t, str> {
        let mut out: Option<String> = None;
        let mut cursor = 0;
        for caps in self.captures_iter(text) {
            let Some(whole) = caps.get(0) else { continue };
            let buffer = out.get_or_insert_with(|| String::with_capacity(text.len()));
            buffer.push_str(&text[cursor..whole.start()]);
            buffer.push_str(&replace(&caps));
            cursor = whole.end();
        }
        match out {
            Some(mut buffer) => {
                buffer.push_str(&text[cursor..]);
                Cow::Owned(buffer)
            }
            None => Cow::Borrowed(text),
        }
    }

    /// `pattern.subn(replacement, text)`.
    pub fn subn(&self, replacement: &str, text: &str) -> (String, usize) {
        let mut count = 0;
        let out = self.sub_with(text, |_| {
            count += 1;
            replacement.to_owned()
        });
        (out.into_owned(), count)
    }

    /// `pattern.split(text)`; capturing groups are kept between the pieces.
    pub fn split_keep<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = Vec::new();
        let mut cursor = 0;
        for caps in self.captures_iter(text) {
            let Some(whole) = caps.get(0) else { continue };
            pieces.push(&text[cursor..whole.start()]);
            for group in 1..caps.len() {
                if let Some(m) = caps.get(group) {
                    pieces.push(&text[m.start()..m.end()]);
                }
            }
            cursor = whole.end();
        }
        pieces.push(&text[cursor..]);
        pieces
    }
}

/// Group text or `None`, mirroring `match[name]`.
pub fn group<'t>(caps: &Captures<'t, str>, name: &str) -> Option<&'t str> {
    caps.name(name).map(|m| m.as_str())
}

/// Group span (`match.span(name)`).
pub fn group_span(caps: &Captures<'_, str>, name: &str) -> Option<Span> {
    caps.name(name).map(|m| Span {
        start: m.start(),
        end: m.end(),
    })
}

/// Whole-match span.
pub fn whole(caps: &Captures<'_, str>) -> Span {
    caps.get(0).map_or(Span { start: 0, end: 0 }, |m| Span {
        start: m.start(),
        end: m.end(),
    })
}

/// Name of the last closed top-level group that matched (`match.lastgroup`).
pub fn last_group<'n>(caps: &Captures<'_, str>, names: &[&'n str]) -> Option<&'n str> {
    names
        .iter()
        .copied()
        .filter_map(|name| caps.name(name).map(|m| (name, m.end())))
        .max_by_key(|&(_, end)| end)
        .map(|(name, _)| name)
}

/// Define a lazily compiled Python pattern.
macro_rules! py_regex {
    ($name:ident, $pattern:expr) => {
        py_regex!($name, $pattern, $crate::pyre::Flags::NONE);
    };
    ($name:ident, $pattern:expr, $flags:expr) => {
        pub(crate) static $name: std::sync::LazyLock<$crate::pyre::PyRegex> =
            std::sync::LazyLock::new(|| $crate::pyre::PyRegex::new(&$pattern, $flags));
    };
}
pub(crate) use py_regex;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shorthands_follow_python() {
        let word = PyRegex::new(r"\b\w+\b", Flags::NONE);
        assert_eq!(word.findall("cafe\u{301} x_1"), ["cafe", "x_1"]);
        let space = PyRegex::new(r"[^\S\r\n]+", Flags::NONE);
        assert_eq!(space.findall("a \t\nb\x1cc"), [" \t", "\x1c"]);
        let class = PyRegex::new(r"[^\W\d_]+", Flags::NONE);
        assert_eq!(class.findall("ab1_cd"), ["ab", "cd"]);
        let literal = PyRegex::new(r#"["'(\[]?x[a-z'-]"#, Flags::NONE);
        assert!(literal.is_match("[x-"));
    }

    #[test]
    fn dollar_and_fullmatch_follow_python() {
        let end = PyRegex::new("a$", Flags::NONE);
        assert!(end.is_match("a\n") && !end.is_match("a\n\n"));
        let alternation = PyRegex::new("a|ab", Flags::NONE);
        assert!(alternation.fullmatch("ab"));
        assert_eq!(alternation.match_at("xab", 1), Some(Span { start: 1, end: 2 }));
        assert_eq!(alternation.match_at("xab", 0), None);
    }

    #[test]
    fn split_keeps_groups() {
        let para = PyRegex::new(r"(\n\s*\n)", Flags::NONE);
        assert_eq!(para.split_keep("a\n\nb"), ["a", "\n\n", "b"]);
    }
}

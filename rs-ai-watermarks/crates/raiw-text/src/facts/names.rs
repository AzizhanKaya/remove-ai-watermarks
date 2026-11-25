//! Heuristic name readers: mid-sentence proper nouns, honorific and appositive name phrases.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use crate::facts::numbers::{ENGLISH_WORD_NUMERAL_RE, MEASUREMENT_RE, without_placeholders};
use crate::pyre::{Flags, PyRegex, escape, group, py_regex, whole};
use crate::unicode;

py_regex!(CAPITALIZED_RE, r"\b[A-Z][a-z]+");
py_regex!(FOLLOWING_CAPITALIZED_RE, r"[^\S\r\n]+[A-Z][a-z]+");
py_regex!(TERM_RE, r"\b[^\W\d_][\w-]*\b");
py_regex!(SENTENCE_SPLIT_RE, r"[.!?]+");
py_regex!(SENTENCE_WORD_RE, r#"["'(\[]?[^\W\d_][\w-]*"#);
py_regex!(HONORIFIC_RE, r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.[ \t]+", Flags::I);
py_regex!(NAME_WORD_RE, r"(?:[^\W\d_]\.)+|[^\W\d_][\w'’-]*");
py_regex!(GAP_RE, r"[ \t]+");
py_regex!(
    APPOSITIVE_LEADING_WORD_RE,
    r"a|an|the|in|on|at|by|to|from|for|of|with|without|after|before|during|since|until|as|and|or|but|if|when|while|although|because|this|that|these|those|all|both|each|every|some|several|many|most|more|less|few|fewer|no|not|only|also|however|first|second|third|next|last|finally|meanwhile|later|today|yesterday|tomorrow|early|late|my|our|your|their",
    Flags::I
);
py_regex!(
    APPOSITIVE_RE,
    r"(?:\A|(?<=[.!?\r\n]))[ \t]*(?P<name>[A-Z][A-Za-z'’-]+(?:[ \t]+[A-Z][A-Za-z'’-]+)+),[ \t]+(?:the|an?)[ \t]+[a-z]"
);

/// Heuristic mid-sentence names plus all-caps terms anywhere, lowercased.
pub fn extract_mid_sentence_proper_nouns(text: &str) -> BTreeSet<String> {
    let prose = without_placeholders(text);
    // Keep title-like name runs such as One Hundred Years Cafe available.
    let prose = MEASUREMENT_RE.sub_with(&prose, |caps| {
        let amount = group(caps, "amount").unwrap_or("");
        let end = whole(caps).end;
        if CAPITALIZED_RE.count(amount) > 1 || FOLLOWING_CAPITALIZED_RE.match_at(&prose, end).is_some() {
            return caps.get(0).map_or_else(String::new, |m| m.as_str().to_owned());
        }
        " ".to_owned()
    });
    // Acronyms and literal uppercase terms carry facts even at sentence start.
    let mut entities: BTreeSet<String> = TERM_RE
        .findall(&prose)
        .into_iter()
        .filter(|word| word.chars().count() > 1 && unicode::str_is_upper(word))
        .map(unicode::lower)
        .collect();
    for sentence in SENTENCE_SPLIT_RE.split_keep(&prose) {
        let stripped = unicode::strip(sentence);
        for (index, raw) in SENTENCE_WORD_RE.findall(stripped).into_iter().enumerate() {
            let mut chars = raw.chars();
            let Some(first) = chars.next() else { continue };
            if index == 0 || matches!(first, '"' | '\'' | '(' | '[') {
                continue;
            }
            if unicode::is_upper(first) && chars.any(unicode::is_lower) {
                entities.insert(unicode::lower(raw));
            }
        }
    }
    entities
}

/// Selected honorifics with capitalized name words, including initials.
pub fn titled_name_phrases(text: &str) -> Vec<String> {
    let prose = without_placeholders(text);
    let mut phrases = Vec::new();
    for prefix in HONORIFIC_RE.find_iter(&prose) {
        let mut cursor = prefix.end;
        let mut end = prefix.end;
        loop {
            let rest = &prose[cursor..];
            let Some(word) = NAME_WORD_RE.match_at(rest, 0) else {
                break;
            };
            let literal = &rest[word.range()];
            if !literal.chars().next().is_some_and(unicode::is_upper) {
                break;
            }
            end = cursor + word.end;
            if let Some(suffix) = ["'s", "\u{2019}s"]
                .iter()
                .find(|suffix| literal.ends_with(**suffix))
            {
                end -= suffix.len();
                break;
            }
            let Some(gap) = GAP_RE.match_at(&prose[end..], 0) else {
                break;
            };
            cursor = end + gap.end;
        }
        if end > prefix.end {
            phrases.push(prose[prefix.start..end].to_owned());
        }
    }
    phrases
}

/// Selected sentence-initial ASCII name runs before an appositive.
pub fn appositive_name_phrases(text: &str) -> Vec<String> {
    let prose = without_placeholders(text);
    APPOSITIVE_RE
        .captures_iter(&prose)
        .filter_map(|caps| {
            let phrase = group(&caps, "name")?;
            let first = unicode::split_whitespace(phrase).next()?;
            (!APPOSITIVE_LEADING_WORD_RE.fullmatch(first) && !ENGLISH_WORD_NUMERAL_RE.fullmatch(first))
                .then(|| phrase.to_owned())
        })
        .collect()
}

/// Honorific and appositive name phrases without duplicates, in reading order per reader.
pub fn selected_name_phrases(text: &str) -> Vec<String> {
    let mut phrases: Vec<String> = Vec::new();
    for phrase in titled_name_phrases(text)
        .into_iter()
        .chain(appositive_name_phrases(text))
    {
        if !phrases.contains(&phrase) {
            phrases.push(phrase);
        }
    }
    phrases
}

/// `\bWORD(?:[ \t]+WORD)*\b` over recognized name words, longest first.
pub fn proper_name_run_pattern(entities: &BTreeSet<String>) -> Option<String> {
    if entities.is_empty() {
        return None;
    }
    let mut names: Vec<&String> = entities.iter().collect();
    names.sort_by(|left, right| {
        right
            .chars()
            .count()
            .cmp(&left.chars().count())
            .then_with(|| left.cmp(right))
    });
    let word = format!(
        "(?:{})",
        names
            .iter()
            .map(|name| escape(name))
            .collect::<Vec<_>>()
            .join("|")
    );
    Some(format!(r"\b{word}(?:[ \t]+{word})*\b"))
}

type RegexCache = std::sync::Mutex<std::collections::HashMap<(String, bool), std::sync::Arc<PyRegex>>>;

/// Compile a dynamic pattern, remembering recent ones like Python's `re` cache.
pub(crate) fn cached_regex(pattern: &str, flags: Flags) -> std::sync::Arc<PyRegex> {
    use std::sync::Arc;
    static CACHE: LazyLock<RegexCache> = LazyLock::new(RegexCache::default);
    let key = (pattern.to_owned(), flags.ignore_case);
    if let Ok(cache) = CACHE.lock()
        && let Some(regex) = cache.get(&key)
    {
        return Arc::clone(regex);
    }
    let regex = Arc::new(PyRegex::new(pattern, flags));
    if let Ok(mut cache) = CACHE.lock() {
        if cache.len() >= 512 {
            cache.clear();
        }
        cache.insert(key, Arc::clone(&regex));
    }
    regex
}

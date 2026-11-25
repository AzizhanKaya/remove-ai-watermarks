//! Selected factual-preservation heuristics for rewritten prose (`verify_text_rewrite_facts`).
//!
//! These lexical checks reject observed failure modes (lost numerals, flipped polarity, moved
//! names, changed qualifiers); a pass is not a semantic-equivalence certificate.

pub mod names;
pub mod numbers;
pub mod polarity;
pub mod relations;

use std::collections::BTreeSet;

use crate::pyre::{Flags, escape, py_regex};
use crate::unicode;

pub use names::{
    appositive_name_phrases, extract_mid_sentence_proper_nouns, selected_name_phrases, titled_name_phrases,
};
pub use numbers::{
    extract_explicit_percent_change_roles, extract_frequencies, extract_measurements,
    extract_normalized_numerals, extract_numeral_occurrences, extract_numeric_bounds,
    extract_vague_durations, extract_weekend_timing, normalize_english_integer, normalize_numeral_token,
    normalize_russian_integer,
};
pub use polarity::{
    Constraint, constraint_texts_for_capability_frames, extract_rewrite_constraints, sentence_content_words,
};
pub use relations::{
    explicit_seasons, explicit_type_coverage_count, extract_coordinated_outcomes, extract_event_phases,
    extract_explicit_membership_stances, extract_explicit_record_distribution,
    extract_explicit_signal_directions, extract_handover_status, extract_influence_qualifications,
};

use names::cached_regex;
use numbers::without_placeholders;

py_regex!(
    SCENARIO_OPENING_RE,
    r"(?:\A\s*|(?<=[.!?])\s+)in (one scenario|the second scenario|a particular situation|another scenario),",
    Flags::I
);
py_regex!(SENTENCE_BREAK_RE, r"[.!?]+");
py_regex!(HORIZONTAL_SPACE_RE, r"[ \t]+");

/// Numerals compared as occurrences, ignoring the illustrative one in matching scenario openings.
pub fn preserves_rewrite_numerals(source: &str, candidate: &str) -> bool {
    let openings = |text: &str| -> Vec<String> {
        SCENARIO_OPENING_RE
            .captures_iter(text)
            .filter_map(|caps| caps.get(1).map(|m| unicode::casefold(m.as_str())))
            .collect()
    };
    let (left, right) = (openings(source), openings(candidate));
    let first = ["one scenario", "the second scenario"];
    let second = ["a particular situation", "another scenario"];
    let is = |value: &[String], expected: &[&str; 2]| {
        value.len() == 2 && value[0] == expected[0] && value[1] == expected[1]
    };
    if (is(&left, &first) && is(&right, &second)) || (is(&left, &second) && is(&right, &first)) {
        let source = SCENARIO_OPENING_RE.sub(" ", source);
        let candidate = SCENARIO_OPENING_RE.sub(" ", candidate);
        return extract_numeral_occurrences(&source) == extract_numeral_occurrences(&candidate);
    }
    extract_numeral_occurrences(source) == extract_numeral_occurrences(candidate)
}

fn singleton_conflict<K: Ord, V: Ord>(
    before: &std::collections::BTreeMap<K, BTreeSet<V>>,
    after: &std::collections::BTreeMap<K, BTreeSet<V>>,
) -> bool {
    before.iter().any(|(key, left)| {
        after
            .get(key)
            .is_some_and(|right| left.len() == 1 && right.len() == 1 && left != right)
    })
}

fn sentences(text: &str) -> Vec<&str> {
    SENTENCE_BREAK_RE
        .split_keep(text)
        .into_iter()
        .map(unicode::strip)
        .filter(|sentence| !sentence.is_empty())
        .collect()
}

/// Check selected quantity, coverage, name, polarity and relation heuristics.
///
/// This can miss category narrowing, changed participant roles and altered quantity-to-entity
/// or season-to-event bindings. A `true` result is not a fidelity certificate.
pub fn verify_text_rewrite_facts(source_text: &str, candidate_text: &str) -> bool {
    if explicit_type_coverage_count(source_text) != explicit_type_coverage_count(candidate_text)
        || relations::changes_actor_apposition(source_text, candidate_text)
        || relations::changes_influence_qualification(source_text, candidate_text)
        || relations::changes_coordinated_outcome_qualification(source_text, candidate_text)
        || polarity::changes_explicit_expected_action(source_text, candidate_text)
        || singleton_conflict(
            &extract_event_phases(source_text),
            &extract_event_phases(candidate_text),
        )
    {
        return false;
    }
    let (constraint_source, constraint_candidate) =
        constraint_texts_for_capability_frames(source_text, candidate_text);
    if extract_rewrite_constraints(&constraint_source) != extract_rewrite_constraints(&constraint_candidate) {
        return false;
    }
    let (source_records, candidate_records) = (
        extract_explicit_record_distribution(source_text),
        extract_explicit_record_distribution(candidate_text),
    );
    if source_records.len() == 1 && candidate_records.len() == 1 && source_records != candidate_records {
        return false;
    }
    if singleton_conflict(
        &extract_explicit_signal_directions(source_text),
        &extract_explicit_signal_directions(candidate_text),
    ) || singleton_conflict(
        &extract_explicit_membership_stances(source_text),
        &extract_explicit_membership_stances(candidate_text),
    ) || singleton_conflict(
        &extract_handover_status(source_text),
        &extract_handover_status(candidate_text),
    ) || extract_frequencies(source_text) != extract_frequencies(candidate_text)
        || extract_vague_durations(source_text) != extract_vague_durations(candidate_text)
        || extract_weekend_timing(source_text) != extract_weekend_timing(candidate_text)
        || extract_measurements(source_text) != extract_measurements(candidate_text)
        || extract_numeric_bounds(source_text) != extract_numeric_bounds(candidate_text)
        || !numbers::preserves_explicit_percent_change_roles(source_text, candidate_text)
        || !relations::preserves_explicit_seasons(source_text, candidate_text)
        || !preserves_rewrite_numerals(source_text, candidate_text)
    {
        return false;
    }

    let candidate_prose = without_placeholders(candidate_text);
    for phrase in selected_name_phrases(source_text) {
        let words: Vec<String> = unicode::split_whitespace(&phrase).map(escape).collect();
        let pattern = format!(r"\b{}(?!\w)", words.join(r"\s+"));
        if !cached_regex(&pattern, Flags::I).is_match(&candidate_prose) {
            return false;
        }
    }
    if let Some(name_pattern) =
        names::proper_name_run_pattern(&extract_mid_sentence_proper_nouns(source_text))
    {
        let source_prose = without_placeholders(source_text);
        let runs = cached_regex(&name_pattern, Flags::I);
        for span in runs.find_iter(&source_prose) {
            let parts: Vec<String> = HORIZONTAL_SPACE_RE
                .split_keep(&source_prose[span.range()])
                .into_iter()
                .map(escape)
                .collect();
            let pattern = format!(r"\b{}\b", parts.join(r"\s+"));
            if !cached_regex(&pattern, Flags::I).is_match(&candidate_prose) {
                return false;
            }
        }
    }

    let source_sentences = sentences(source_text);
    let candidate_sentences = sentences(candidate_text);
    if source_sentences.len() == 1 && candidate_sentences.len() == 1 {
        let (source, candidate) = (source_sentences[0], candidate_sentences[0]);
        if !polarity::preserves_rewrite_negation(source, candidate) {
            return false;
        }
        return !polarity::has_antonym_pole_inversion(
            &sentence_content_words(source),
            &sentence_content_words(candidate),
        );
    }
    let candidate_words: Vec<BTreeSet<String>> = candidate_sentences
        .iter()
        .map(|sentence| sentence_content_words(sentence))
        .collect();
    let mut used = BTreeSet::new();
    for source in source_sentences {
        let source_words = sentence_content_words(source);
        if source_words.is_empty() {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        for (index, words) in candidate_words.iter().enumerate() {
            if used.contains(&index) || words.is_empty() {
                continue;
            }
            let shared = source_words.intersection(words).count() as f64;
            let similarity = shared / ((source_words.len() * words.len()) as f64).sqrt();
            if similarity > best.map_or(0.0, |(_, best)| best) {
                best = Some((index, similarity));
            }
        }
        if let Some((index, similarity)) = best
            && similarity >= 0.45
        {
            used.insert(index);
            if !polarity::preserves_rewrite_negation(source, candidate_sentences[index])
                || polarity::has_antonym_pole_inversion(&source_words, &candidate_words[index])
            {
                return false;
            }
        }
    }
    true
}

/// Python-syntax sources of the module-level patterns, for parity checks.
#[doc(hidden)]
pub fn pattern_sources() -> Vec<(&'static str, String)> {
    let mut sources = numbers::pattern_sources();
    sources.push((
        "_TYPE_COVERAGE_PATTERN",
        relations::TYPE_COVERAGE_PATTERN.to_owned(),
    ));
    sources.push(("_SCENARIO_OPENING_RE", SCENARIO_OPENING_RE.source().to_owned()));
    sources
}

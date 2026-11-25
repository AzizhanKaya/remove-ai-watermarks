//! Fact-guard goldens: every extractor, pair check, fact mask and number normalizer.

use std::collections::{BTreeMap, BTreeSet};

use raiw_core::testing::golden;
use raiw_text::facts::{self, numbers, polarity, relations};
use raiw_text::guard;
use serde::Serialize;
use serde_json::{Value, json};

fn to_value<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("serializable")
}

fn canonical_set(mut items: Vec<Value>) -> Value {
    items.sort_by_key(ToString::to_string);
    Value::Array(items)
}

fn set_value<T: Serialize>(items: impl IntoIterator<Item = T>) -> Value {
    canonical_set(items.into_iter().map(|item| to_value(&item)).collect())
}

fn keyed_sets<K: Serialize, V: Serialize>(map: &BTreeMap<K, BTreeSet<V>>) -> Value {
    canonical_set(
        map.iter()
            .map(|(key, values)| json!([to_value(key), set_value(values)]))
            .collect(),
    )
}

fn extract(name: &str, text: &str) -> Value {
    match name {
        "_extract_frequencies" => to_value(&facts::extract_frequencies(text)),
        "_extract_numeral_occurrences" => to_value(&facts::extract_numeral_occurrences(text)),
        "_extract_measurements" => to_value(&facts::extract_measurements(text)),
        "_extract_numeric_bounds" => set_value(facts::extract_numeric_bounds(text)),
        "_extract_mid_sentence_proper_nouns" => set_value(facts::extract_mid_sentence_proper_nouns(text)),
        "_selected_name_phrases" => to_value(&facts::selected_name_phrases(text)),
        "_extract_rewrite_constraints" => set_value(facts::extract_rewrite_constraints(text)),
        "_extract_explicit_record_distribution" => {
            set_value(facts::extract_explicit_record_distribution(text))
        }
        "_extract_explicit_signal_directions" => keyed_sets(&facts::extract_explicit_signal_directions(text)),
        "_extract_explicit_membership_stances" => {
            keyed_sets(&facts::extract_explicit_membership_stances(text))
        }
        "_explicit_type_coverage_count" => json!(facts::explicit_type_coverage_count(text)),
        "_extract_handover_status" => keyed_sets(&facts::extract_handover_status(text)),
        "_extract_event_phases" => keyed_sets(&facts::extract_event_phases(text)),
        "_extract_influence_qualifications" => keyed_sets(&facts::extract_influence_qualifications(text)),
        "_extract_coordinated_outcomes" => keyed_sets(&facts::extract_coordinated_outcomes(text)),
        "_extract_vague_durations" => to_value(&facts::extract_vague_durations(text)),
        "_extract_weekend_timing" => to_value(&facts::extract_weekend_timing(text)),
        "_explicit_seasons" => set_value(facts::explicit_seasons(text)),
        "_has_active_negation" => json!(polarity::has_active_negation(text)),
        "_extract_explicit_percent_change_roles" => {
            let roles = facts::extract_explicit_percent_change_roles(text);
            Value::Object(
                roles
                    .iter()
                    .map(|(value, kinds)| (value.clone(), set_value(kinds)))
                    .collect(),
            )
        }
        "_sentence_content_words" => set_value(facts::sentence_content_words(text)),
        other => panic!("unknown extractor {other}"),
    }
}

fn check(name: &str, source: &str, candidate: &str) -> Value {
    match name {
        "verify_text_rewrite_facts" => json!(facts::verify_text_rewrite_facts(source, candidate)),
        "_preserves_rewrite_numerals" => json!(facts::preserves_rewrite_numerals(source, candidate)),
        "_preserves_rewrite_negation" => json!(polarity::preserves_rewrite_negation(source, candidate)),
        "_constraint_texts_for_capability_frames" => {
            to_value(&facts::constraint_texts_for_capability_frames(source, candidate))
        }
        "_changes_explicit_expected_action" => {
            json!(polarity::changes_explicit_expected_action(source, candidate))
        }
        "_changes_actor_apposition" => json!(relations::changes_actor_apposition(source, candidate)),
        "_changes_influence_qualification" => {
            json!(relations::changes_influence_qualification(source, candidate))
        }
        "_changes_coordinated_outcome_qualification" => {
            json!(relations::changes_coordinated_outcome_qualification(
                source, candidate
            ))
        }
        "_preserves_explicit_seasons" => json!(relations::preserves_explicit_seasons(source, candidate)),
        "_preserves_explicit_percent_change_roles" => json!(
            numbers::preserves_explicit_percent_change_roles(source, candidate)
        ),
        "_preserves_dominant_script" => json!(guard::preserves_dominant_script(source, candidate)),
        "_preserves_rewrite_extent" => json!(guard::preserves_rewrite_extent(source, candidate)),
        other => panic!("unknown check {other}"),
    }
}

fn assert_no_failures(failures: &[String], total: usize) {
    let shown: Vec<&String> = failures.iter().take(40).collect();
    assert!(
        failures.is_empty(),
        "{} of {total} comparisons differ:\n{}",
        failures.len(),
        shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
    );
}

/// Decode `\uXXXX` regex escapes, which denote the same character as the literal.
fn decode_u_escapes(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '\\' && chars.get(index + 1) == Some(&'u') && index + 6 <= chars.len() {
            let hex: String = chars[index + 2..index + 6].iter().collect();
            if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                out.push(ch);
                index += 6;
                continue;
            }
        }
        if chars[index] == '\\' {
            out.push(chars[index]);
            index += 1;
            if let Some(&next) = chars.get(index) {
                out.push(next);
                index += 1;
            }
            continue;
        }
        out.push(chars[index]);
        index += 1;
    }
    out
}

#[test]
fn pattern_sources_match_python() {
    let patterns = golden("text", "patterns");
    for (name, source) in facts::pattern_sources() {
        let expected = patterns[name].as_str().map(decode_u_escapes);
        assert_eq!(Some(decode_u_escapes(&source)), expected, "{name}");
    }
}

#[test]
fn extractors_match_python() {
    let cases = golden("text", "facts");
    let mut failures = Vec::new();
    let mut total = 0;
    for case in cases["extracts"].as_array().expect("array") {
        let text = case["text"].as_str().expect("text");
        for (name, expected) in case.as_object().expect("object") {
            if name == "text" {
                continue;
            }
            total += 1;
            let actual = extract(name, text);
            let expected = match name.as_str() {
                "_extract_numeric_bounds"
                | "_extract_mid_sentence_proper_nouns"
                | "_extract_rewrite_constraints"
                | "_extract_explicit_record_distribution"
                | "_explicit_seasons"
                | "_sentence_content_words" => {
                    canonical_set(expected.as_array().cloned().unwrap_or_default())
                }
                "_extract_explicit_signal_directions"
                | "_extract_explicit_membership_stances"
                | "_extract_handover_status"
                | "_extract_event_phases"
                | "_extract_influence_qualifications"
                | "_extract_coordinated_outcomes" => canonical_set(
                    expected
                        .as_array()
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|pair| {
                            json!([
                                pair[0],
                                canonical_set(pair[1].as_array().cloned().unwrap_or_default())
                            ])
                        })
                        .collect(),
                ),
                _ => expected.clone(),
            };
            if actual != expected {
                failures.push(format!(
                    "{name}({text:?}):\n  rust   {actual}\n  python {expected}"
                ));
            }
        }
    }
    assert_no_failures(&failures, total);
}

#[test]
fn pair_checks_match_python() {
    let cases = golden("text", "facts");
    let mut failures = Vec::new();
    let mut total = 0;
    for case in cases["checks"].as_array().expect("array") {
        let (source, candidate) = (
            case["source"].as_str().expect("source"),
            case["candidate"].as_str().expect("candidate"),
        );
        for (name, expected) in case.as_object().expect("object") {
            if name == "source" || name == "candidate" {
                continue;
            }
            total += 1;
            let actual = check(name, source, candidate);
            if &actual != expected {
                failures.push(format!(
                    "{name}({source:?}, {candidate:?}):\n  rust   {actual}\n  python {expected}"
                ));
            }
        }
    }
    assert_no_failures(&failures, total);
}

#[test]
fn fact_masks_match_python() {
    let cases = golden("text", "facts");
    let mut failures = Vec::new();
    let masks = cases["masks"].as_array().expect("array");
    for case in masks {
        let text = case["text"].as_str().expect("text");
        let (masked, replacements) = guard::mask_local_rewrite_facts(text);
        let actual = json!({"masked": masked, "replacements": replacements});
        let expected = json!({"masked": case["masked"], "replacements": case["replacements"]});
        if actual != expected {
            failures.push(format!("{text:?}:\n  rust   {actual}\n  python {expected}"));
        }
    }
    assert_no_failures(&failures, masks.len());
}

#[test]
fn number_normalizers_match_python() {
    let cases = golden("text", "facts");
    let mut failures = Vec::new();
    let mut total = 0;
    for (name, rows) in cases["normalizers"].as_object().expect("object") {
        for row in rows.as_array().expect("array") {
            total += 1;
            let token = row["token"].as_str().expect("token");
            let actual = match name.as_str() {
                "_normalize_numeral_token" => facts::normalize_numeral_token(token),
                "_normalize_english_integer" => facts::normalize_english_integer(token),
                _ => facts::normalize_russian_integer(token),
            };
            if Some(actual.as_str()) != row["out"].as_str() {
                failures.push(format!(
                    "{name}({token:?}): rust {actual:?} python {}",
                    row["out"]
                ));
            }
        }
    }
    assert_no_failures(&failures, total);
}

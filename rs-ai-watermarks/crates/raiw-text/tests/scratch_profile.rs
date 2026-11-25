use std::time::Instant;

use raiw_core::testing::repo_root;
use raiw_text::facts::{self, polarity, relations};
use serde_json::Value;

#[test]
#[ignore = "profiling"]
fn profile_extractors() {
    let path = repo_root().join("data/evaluations/text-watermark/fidelity-cases.json");
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = cases["cases"].as_array().unwrap();
    let mut texts: Vec<String> = cases.iter().map(|case| case["source"].as_str().unwrap().to_owned()).collect();
    texts.extend(cases.iter().map(|case| case["candidate"].as_str().unwrap().to_owned()));
    let run = |name: &str, f: &dyn Fn(&str)| {
        for text in &texts {
            f(text);
        }
        let start = Instant::now();
        for text in &texts {
            f(text);
        }
        eprintln!("{name:45} {:8.2} ms", start.elapsed().as_secs_f64() * 1000.0);
    };
    run("type_coverage", &|t| drop(facts::explicit_type_coverage_count(t)));
    run("event_phases", &|t| drop(facts::extract_event_phases(t)));
    run("influence", &|t| drop(facts::extract_influence_qualifications(t)));
    run("coordinated", &|t| drop(facts::extract_coordinated_outcomes(t)));
    run("constraints", &|t| drop(facts::extract_rewrite_constraints(t)));
    run("record", &|t| drop(facts::extract_explicit_record_distribution(t)));
    run("signals", &|t| drop(facts::extract_explicit_signal_directions(t)));
    run("membership", &|t| drop(facts::extract_explicit_membership_stances(t)));
    run("handover", &|t| drop(facts::extract_handover_status(t)));
    run("frequencies", &|t| drop(facts::extract_frequencies(t)));
    run("vague", &|t| drop(facts::extract_vague_durations(t)));
    run("weekend", &|t| drop(facts::extract_weekend_timing(t)));
    run("measurements", &|t| drop(facts::extract_measurements(t)));
    run("bounds", &|t| drop(facts::extract_numeric_bounds(t)));
    run("seasons", &|t| drop(facts::explicit_seasons(t)));
    run("numerals", &|t| drop(facts::extract_numeral_occurrences(t)));
    run("name_phrases", &|t| drop(facts::selected_name_phrases(t)));
    run("proper_nouns", &|t| drop(facts::extract_mid_sentence_proper_nouns(t)));
    run("negation", &|t| drop(polarity::has_active_negation(t)));
    run("content_words", &|t| drop(facts::sentence_content_words(t)));
    run("percent", &|t| drop(facts::extract_explicit_percent_change_roles(t)));
    let _ = relations::explicit_seasons;
}

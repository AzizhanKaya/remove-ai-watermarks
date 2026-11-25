//! py3langid parity and the shared candidate guard, against the pinned model.
#![cfg(feature = "langid")]

use std::path::PathBuf;
use std::sync::OnceLock;

use indexmap::IndexMap;
use raiw_core::testing::{golden, repo_root};
use raiw_text::guard::{self, LanguageClassifier};
use raiw_text::langid::LanguageIdentifier;

/// The model the Python reference uses, when the shared environment has it installed.
fn identifier() -> Option<&'static LanguageIdentifier> {
    static MODEL: OnceLock<Option<LanguageIdentifier>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let path = std::env::var_os("RAIW_LANGID_MODEL").map_or_else(
                || repo_root().join(".venv/lib/python3.12/site-packages/py3langid/data/model.npz.xz"),
                PathBuf::from,
            );
            path.exists()
                .then(|| LanguageIdentifier::from_path(&path).expect("pinned model loads"))
        })
        .as_ref()
}

#[test]
fn classifier_matches_py3langid() {
    let Some(model) = identifier() else {
        eprintln!("skipping: py3langid model not available");
        return;
    };
    let cases = golden("text", "language");
    let rows = cases["classify"].as_array().expect("array");
    let mut failures = Vec::new();
    for row in rows {
        let text = row["text"].as_str().expect("text");
        let (label, score) = model.classify_text(text);
        let expected_score = row["score"].as_f64().expect("score");
        if label != row["label"].as_str().expect("label") || (score - expected_score).abs() > 1e-4 {
            failures.push(format!(
                "{text:?}: rust ({label}, {score}) python ({}, {expected_score})",
                row["label"]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} differ:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

#[test]
fn language_guard_and_candidate_guard_match_python() {
    let Some(model) = identifier() else {
        eprintln!("skipping: py3langid model not available");
        return;
    };
    let classifier: &dyn LanguageClassifier = model;
    let cases = golden("text", "language");
    let mut failures = Vec::new();
    for row in cases["preserves"].as_array().expect("array") {
        let (source, candidate) = (
            row["source"].as_str().unwrap(),
            row["candidate"].as_str().unwrap(),
        );
        let actual = guard::preserves_rewrite_language(source, candidate, classifier).unwrap();
        if Some(actual) != row["out"].as_bool() {
            failures.push(format!("language({source:?}, {candidate:?}): rust {actual}"));
        }
    }
    for row in cases["guards"].as_array().expect("array") {
        let facts: IndexMap<String, String> = serde_json::from_value(row["facts"].clone()).unwrap();
        let actual = guard::guard_local_rewrite_candidate(
            row["passage"].as_str().unwrap(),
            row["masked"].as_str().unwrap(),
            &facts,
            row["candidate"].as_str().unwrap(),
            None,
            classifier,
        )
        .unwrap();
        if actual.as_deref() != row["out"].as_str() {
            failures.push(format!(
                "guard({:?}, {:?}): rust {actual:?} python {}",
                row["passage"], row["candidate"], row["out"]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn invalid_classifier_results_are_failures() {
    for (label, score) in [("", 0.99), ("en", f64::NAN), ("en", -1.0), ("en", 1.1)] {
        let classifier = move |_: &str| Ok((label.to_owned(), score));
        let error = guard::classify_text_language(&classifier, "text").unwrap_err();
        assert!(error.to_string().contains("language identifier"));
    }
}

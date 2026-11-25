//! Keyed scorer goldens (BLAKE2 simulations, Threefry SynthID), sentence slots and loan roles.

use raiw_core::testing::{assert_json_close, fixture, golden};
use raiw_text::roles::{DependencyParser, ParsedDoc, verify_text_rewrite_loan_roles};
use raiw_text::scores::threefry::threefry2x32;
use raiw_text::{
    DEFAULT_ALPHA_001_Z, TextError, ThreefryOptions, score_gumbel_text, score_kgw_text,
    score_synthid_threefry_text, score_textgrain_text, score_textseal2_text, score_unigram_text,
};
use serde_json::Value;

fn tokens(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .expect("tokens")
        .iter()
        .map(|token| token.as_i64().expect("int"))
        .collect()
}

#[test]
fn blake2_scorers_match_python() {
    let cases = golden("text", "scores");
    for row in cases["blake2"].as_array().expect("array") {
        let ids = tokens(&row["tokens"]);
        let key = row["key"].as_i64().unwrap();
        let z = DEFAULT_ALPHA_001_Z;
        let results = [
            ("gumbel", score_gumbel_text(&ids, key, 3, z)),
            ("gumbel_ctx1", score_gumbel_text(&ids, key, 1, z)),
            ("textgrain", score_textgrain_text(&ids, key, 3, 256, 256, z)),
            ("textgrain_small", score_textgrain_text(&ids, key, 3, 3, 5, z)),
            ("textseal2", score_textseal2_text(&ids, key, None, 3, z)),
            (
                "textseal2_b",
                score_textseal2_text(&ids, key, Some(key + 1), 3, z),
            ),
            ("kgw", score_kgw_text(&ids, key, 1, 0.25, z)),
            ("kgw_wide", score_kgw_text(&ids, key, 3, 0.5, z)),
            ("unigram", score_unigram_text(&ids, key, 0.25, z)),
        ];
        for (name, result) in results {
            let actual = serde_json::to_value(result.unwrap()).unwrap();
            assert_json_close(
                &actual,
                &row[name],
                1e-9,
                &format!("{name} key={key} tokens={}", ids.len()),
            );
        }
    }
}

#[test]
fn threefry_scores_match_python() {
    assert_eq!(threefry2x32((0, 0), 0, 0), (0x6B20_0159, 0x99BA_4EFE));
    let cases = golden("text", "scores");
    for row in cases["threefry"].as_array().expect("array") {
        let options = ThreefryOptions {
            ngram_len: row["ngram_len"].as_u64().unwrap() as usize,
            depth: row["depth"].as_u64().unwrap() as usize,
            context_history_size: row["history"].as_u64().unwrap() as usize,
            eos_token_ids: tokens(&row["eos"]),
            alpha: 0.5,
        };
        let result =
            score_synthid_threefry_text(&tokens(&row["ids"]), row["key"].as_u64().unwrap(), &options)
                .unwrap();
        let actual = serde_json::to_value(result).unwrap();
        assert_eq!(actual["num_ones"], row["result"]["num_ones"]);
        assert_eq!(actual["num_tokens"], row["result"]["num_tokens"]);
        assert_json_close(&actual, &row["result"], 1e-13, "threefry");
    }
}

#[test]
fn threefry_reference_vectors_and_errors() {
    let repeated = score_synthid_threefry_text(
        &[1, 2, 3, 1, 2, 9, 4],
        654,
        &ThreefryOptions {
            ngram_len: 3,
            ..ThreefryOptions::default()
        },
    )
    .unwrap();
    assert_eq!(repeated.num_tokens, 4);
    let empty = score_synthid_threefry_text(&[1, 2], 654, &ThreefryOptions::default()).unwrap();
    assert_eq!((empty.p_value, empty.num_tokens, empty.detected), (1.0, 0, false));
    assert!(score_synthid_threefry_text(&[1], 1 << 63, &ThreefryOptions::default()).is_err());
    assert!(score_synthid_threefry_text(&[1 << 33], 1, &ThreefryOptions::default()).is_err());
    assert!(
        score_synthid_threefry_text(
            &[1],
            1,
            &ThreefryOptions {
                alpha: 1.0,
                ..ThreefryOptions::default()
            }
        )
        .is_err()
    );
}

#[cfg(feature = "api")]
#[test]
fn sentence_slots_match_python() {
    let cases = golden("text", "sentence_slots");
    for row in cases.as_array().expect("array") {
        let text = row["text"].as_str().unwrap();
        let (sentences, separators) = raiw_text::api::sentence_slots(text);
        assert_eq!(serde_json::json!([sentences, separators]), row["out"], "{text:?}");
    }
}

struct FixtureParser {
    pairs: Vec<(String, String)>,
    lang: &'static str,
}

impl DependencyParser for FixtureParser {
    fn lang(&self) -> &str {
        self.lang
    }

    fn parse(&self, texts: &[&str]) -> Result<Vec<ParsedDoc>, TextError> {
        texts
            .iter()
            .map(|text| {
                let annotation = self
                    .pairs
                    .iter()
                    .find(|(source, _)| source == text)
                    .map_or("", |(_, parse)| parse.as_str());
                ParsedDoc::from_annotation(text, annotation)
            })
            .collect()
    }
}

fn parser_for(row: &Value) -> FixtureParser {
    let field = |name: &str| row[name].as_str().unwrap().to_owned();
    FixtureParser {
        pairs: vec![
            (field("source"), field("source_parse")),
            (field("candidate"), field("candidate_parse")),
        ],
        lang: "en",
    }
}

#[test]
fn loan_roles_replay_frozen_parses() {
    let text = std::fs::read_to_string(fixture("synthetic/text-loan-role-parses.json")).unwrap();
    let cases: Value = serde_json::from_str(&text).unwrap();
    let rows = cases["cases"].as_array().unwrap();
    for row in rows {
        let parser = parser_for(row);
        let (source, candidate) = (
            row["source"].as_str().unwrap(),
            row["candidate"].as_str().unwrap(),
        );
        let expected = row["no_selected_reversal"].as_bool().unwrap();
        assert_eq!(
            verify_text_rewrite_loan_roles(source, candidate, &parser).unwrap(),
            expected,
            "{}",
            row["id"]
        );
        assert_eq!(
            verify_text_rewrite_loan_roles(candidate, source, &parser).unwrap(),
            expected,
            "{} reversed",
            row["id"]
        );
    }
    let first = &rows[0];
    let (source, candidate) = (
        first["source"].as_str().unwrap(),
        first["candidate"].as_str().unwrap(),
    );
    let english = parser_for(first);
    let russian = FixtureParser {
        lang: "ru",
        ..parser_for(first)
    };
    assert!(
        verify_text_rewrite_loan_roles(source, candidate, &russian)
            .unwrap_err()
            .to_string()
            .contains("English parser")
    );
    assert!(
        verify_text_rewrite_loan_roles("", candidate, &english)
            .unwrap_err()
            .to_string()
            .contains("nonempty")
    );
    assert!(
        verify_text_rewrite_loan_roles(source, "  ", &english)
            .unwrap_err()
            .to_string()
            .contains("nonempty")
    );
    let empty = FixtureParser {
        pairs: vec![],
        lang: "en",
    };
    assert!(
        verify_text_rewrite_loan_roles(source, candidate, &empty)
            .unwrap_err()
            .to_string()
            .contains("complete dependency")
    );
}

struct BrokenParser(usize);

impl DependencyParser for BrokenParser {
    fn lang(&self) -> &'static str {
        "en"
    }

    fn parse(&self, texts: &[&str]) -> Result<Vec<ParsedDoc>, TextError> {
        match self.0 {
            0 => Err(TextError::runtime("Parser unavailable")),
            1 => Ok(vec![
                ParsedDoc::from_annotation("Truncated.", "x/NOUN/ROOT/0")?;
                2
            ]),
            count => Ok(vec![
                ParsedDoc::from_annotation(texts[0], "x/NOUN/ROOT/0")?;
                count
            ]),
        }
    }
}

#[test]
fn loan_role_parser_contract() {
    let check = |parser: &BrokenParser| {
        verify_text_rewrite_loan_roles("Residents borrow tools.", "Volunteers lend tools.", parser)
            .unwrap_err()
            .to_string()
    };
    assert!(check(&BrokenParser(0)).contains("Parser unavailable"));
    assert!(check(&BrokenParser(1)).contains("exact input text"));
    assert!(check(&BrokenParser(3)).contains("one complete document per input"));
}

//! Deterministic provenance goldens: identify/metadata reports, stego payloads, spans, encoders.

use raiw_core::testing::golden;
use raiw_text::{
    CommentStyle, ManifestSource, decode_stego_payloads, encode_c2pa_ascii_armor, encode_c2patxt,
    encode_innamark, encode_unicode_tags, encode_vsrmark, identify_text, remove_text_metadata,
    split_protected_text_spans,
};
use serde_json::{Value, json};

fn report_failures(failures: &[String], total: usize) {
    assert!(
        failures.is_empty(),
        "{} of {total} cases differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn identify_and_metadata_match_python() {
    let cases = golden("text", "identify");
    let cases = cases.as_array().expect("array");
    let mut failures = Vec::new();
    for case in cases {
        let text = case["text"].as_str().expect("text");
        let report = identify_text(text).expect("identify");
        if report.to_json() != case["json"].as_str().expect("json") {
            failures.push(format!(
                "identify {text:?}:\n  rust   {}\n  python {}",
                report.to_json(),
                case["json"]
            ));
        }
        let metadata =
            serde_json::to_value(remove_text_metadata(text, None).expect("metadata")).expect("serialize");
        if metadata != case["metadata"] {
            failures.push(format!(
                "metadata {text:?}:\n  rust   {metadata}\n  python {}",
                case["metadata"]
            ));
        }
        let stego = serde_json::to_value(decode_stego_payloads(text)).expect("serialize");
        if stego != case["stego"] {
            failures.push(format!(
                "stego {text:?}:\n  rust   {stego}\n  python {}",
                case["stego"]
            ));
        }
    }
    report_failures(&failures, cases.len());
}

#[test]
fn spans_match_python() {
    let cases = golden("text", "spans");
    let cases = cases.as_array().expect("array");
    let mut failures = Vec::new();
    for case in cases {
        let text = case["text"].as_str().expect("text");
        let spans: Vec<Value> = split_protected_text_spans(text)
            .into_iter()
            .map(|span| json!([span.kind.as_str(), span.text, span.locked]))
            .collect();
        if Value::Array(spans.clone()) != case["spans"] {
            failures.push(format!(
                "{text:?}:\n  rust   {}\n  python {}",
                Value::Array(spans),
                case["spans"]
            ));
        }
    }
    report_failures(&failures, cases.len());
}

fn hex(value: &Value) -> Vec<u8> {
    hex::decode(value.as_str().expect("hex")).expect("valid hex")
}

#[test]
fn encoders_match_python() {
    let cases = golden("text", "encoders");
    for case in cases["c2patxt"].as_array().expect("array") {
        let out = encode_c2patxt(
            case["text"].as_str().unwrap(),
            &hex(&case["manifest"]),
            case["padding"].as_u64().unwrap() as usize,
        );
        assert_eq!(out.unwrap(), case["out"].as_str().unwrap());
    }
    for case in cases["armor"].as_array().expect("array") {
        let bytes;
        let source = if case["kind"] == "bytes" {
            bytes = hex(&case["value"]);
            ManifestSource::Bytes(&bytes)
        } else {
            ManifestSource::Uri(case["value"].as_str().unwrap())
        };
        let style = match case["style"].as_str().unwrap() {
            "html" => CommentStyle::Html,
            "hash" => CommentStyle::Hash,
            _ => CommentStyle::None,
        };
        assert_eq!(
            encode_c2pa_ascii_armor(&source, style),
            case["out"].as_str().unwrap()
        );
    }
    for case in cases["vsrmark"].as_array().expect("array") {
        let out = encode_vsrmark(
            case["text"].as_str().unwrap(),
            case["payload"].as_str().unwrap().as_bytes(),
        );
        assert_eq!(out.unwrap(), case["out"].as_str().unwrap());
    }
    for case in cases["tags"].as_array().expect("array") {
        assert_eq!(
            encode_unicode_tags(case["text"].as_str().unwrap(), case["hidden"].as_str().unwrap()).unwrap(),
            case["out"].as_str().unwrap()
        );
    }
    for case in cases["innamark"].as_array().expect("array") {
        let out = encode_innamark(
            case["text"].as_str().unwrap(),
            case["payload"].as_str().unwrap().as_bytes(),
            false,
            case["crc32"].as_bool().unwrap(),
        );
        assert_eq!(out.unwrap(), case["out"].as_str().unwrap());
    }
    let errors: Vec<(String, String)> = cases["errors"]
        .as_array()
        .expect("array")
        .iter()
        .map(|case| {
            (
                case["case"].as_str().unwrap().to_owned(),
                case["message"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    for (label, message) in errors {
        let error = match label.as_str() {
            "vsrmark_short" => encode_vsrmark("short", b"payload").unwrap_err(),
            "tags_non_ascii" => encode_unicode_tags("x", "é").unwrap_err(),
            "tags_control" => encode_unicode_tags("x", "\n").unwrap_err(),
            "innamark_short" => encode_innamark("a b", b"payload", false, true).unwrap_err(),
            other => panic!("unknown error case {other}"),
        };
        assert_eq!(error.to_string(), message, "{label}");
    }
}

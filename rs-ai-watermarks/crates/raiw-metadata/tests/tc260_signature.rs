//! TC260 label signatures against the Python reference verdicts on the tracked fixtures.

use raiw_core::testing::golden;
use raiw_metadata::check_tc260_signature;
use serde_json::Value;

#[test]
fn verdicts_match_python_on_every_fixture_label() {
    let cases = golden("metadata", "tc260_signatures");
    let cases = cases.as_array().expect("golden is a list");
    assert!(
        cases.iter().any(|c| c["verdict"]["label"] == "verified"),
        "golden must hold a verified label"
    );
    for case in cases {
        let label = case["label"].as_object().expect("label object");
        let field = |key: &str| label.get(key).and_then(Value::as_str);
        let actual = check_tc260_signature(field).map(
            |s| serde_json::json!({"label": s.label.as_str(), "signer": s.signer, "describe": s.describe()}),
        );
        assert_eq!(actual.unwrap_or(Value::Null), case["verdict"], "{}", case["name"]);
    }
}

//! Direct-function goldens (`units.json`): LEB128, AV1 metadata payloads, GIF and JPEG helpers.

use raiw_container::{av1, gif, isobmff, jpeg};
use raiw_core::testing::golden;
use serde_json::{Value, json};

fn hex_of(value: &Value) -> Vec<u8> {
    hex::decode(value.as_str().expect("hex string")).expect("valid hex")
}

fn cases<'a>(units: &'a Value, name: &str) -> &'a Vec<Value> {
    units[name]
        .as_array()
        .unwrap_or_else(|| panic!("missing unit {name}"))
}

fn patch_json(found: &[av1::ObuPatch]) -> Value {
    json!(
        found
            .iter()
            .map(|p| json!([p.header_offset, p.header_byte, p.payload_start, p.payload_end]))
            .collect::<Vec<_>>()
    )
}

#[test]
fn leb128_matches_python() {
    let units = golden("container", "units");
    for case in cases(&units, "read_leb128") {
        let parsed = av1::read_leb128(&hex_of(&case[0]), 0).map(|(value, next)| json!([value, next]));
        assert_eq!(parsed.unwrap_or(Value::Null), case[1], "read_leb128({})", case[0]);
    }
    for case in cases(&units, "encode_leb128") {
        assert_eq!(
            hex::encode(av1::encode_leb128(case[0].as_u64().unwrap())),
            case[1].as_str().unwrap()
        );
    }
}

#[test]
fn av1_metadata_classification_matches_python() {
    let units = golden("container", "units");
    for case in cases(&units, "is_ai_obu_metadata_payload") {
        assert_eq!(
            json!(av1::is_ai_obu_metadata_payload(&hex_of(&case[0]))),
            case[1],
            "{}",
            case[0]
        );
    }
    let sample = &units["scan_av1_obu_ai_spans"];
    let stream = hex_of(&sample[0]);
    assert_eq!(patch_json(&av1::scan_sample_spans(&stream, 100)), sample[1]);
    let expected = &units["scan_av1_obu_stream_spans"];
    let all = av1::scan_stream_spans(stream.as_slice(), 0, stream.len() as u64, false).unwrap();
    let first = av1::scan_stream_spans(stream.as_slice(), 0, stream.len() as u64, true).unwrap();
    assert_eq!(patch_json(&all), expected[0]);
    assert_eq!(patch_json(&first), expected[1]);
}

#[test]
fn gif_helpers_match_python() {
    let units = golden("container", "units");
    for case in cases(&units, "gif_sub_block_payload") {
        let got = gif::sub_block_payload(&hex_of(&case[0]), 0).map(hex::encode);
        assert_eq!(json!(got), case[1], "{}", case[0]);
    }
    for case in cases(&units, "gif_extension_metadata") {
        assert_eq!(
            hex::encode(gif::extension_metadata(&hex_of(&case[0]))),
            case[1].as_str().unwrap()
        );
    }
}

#[test]
fn jpeg_app_payloads_match_python() {
    let units = golden("container", "units");
    for case in cases(&units, "jpeg_app_payloads") {
        let data = hex_of(&case[0]);
        let got: Vec<Value> = jpeg::app_payloads(&data)
            .iter()
            .map(|(m, p)| json!([m, hex::encode(p)]))
            .collect();
        assert_eq!(json!(got), case[1], "{}", case[0]);
    }
}

#[test]
fn xai_region_scan_matches_python() {
    let units = golden("container", "units");
    for case in cases(&units, "xai_in_region") {
        assert_eq!(
            json!(isobmff::xai_signature_in_region(&hex_of(&case[0]))),
            case[1],
            "{}",
            case[0]
        );
    }
}

//! Differential tests for the DWT-DCT decoder and the open-watermark verdict (`pixel_dwt` goldens).

mod common;

use common::embed::{embed, message_bits};
use common::{base_image, bit_string, decode_bgr, fixture_bgr, mat_sha256, unpack_bits};
use raiw_core::testing::golden;
use raiw_cv::Mat;
use raiw_pixel::invisible_watermark::{
    BITS_48, MATCH_48, MATCH_SD1_FRAC, SD1_STRING, bits_match, bytes_match_frac,
};
use raiw_pixel::{DwtDctError, block_bits, decode_dwt_dct, decode_dwt_dct_lengths, detect_invisible_watermark};
use serde_json::Value;

fn cases(name: &str) -> Vec<Value> {
    golden("pixel_dwt", name)["cases"].as_array().expect("cases").clone()
}

fn python_error(error: &DwtDctError) -> &'static str {
    match error {
        DwtDctError::TooSmall => "RuntimeError",
        DwtDctError::InvalidLengths | DwtDctError::EmptyPlane { .. } => "ValueError",
        DwtDctError::Channels(_) => "error",
    }
}

/// Compare every recorded output for one decoded image.
fn check_case(img: &Mat<u8>, case: &Value) {
    let name = case["name"].as_str().unwrap_or("?");
    let shape: Vec<usize> = case["shape"].as_array().expect("shape").iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
    assert_eq!(vec![img.height(), img.width(), img.channels()], shape, "{name}: shape");
    let decoded = decode_dwt_dct_lengths(img.view(), &[48, 136]);
    match case.get("error").and_then(Value::as_str) {
        Some(expected) => {
            let err = decoded.expect_err(name);
            assert_eq!(python_error(&err), expected, "{name}: error class");
            if let Some(message) = case["message"].as_str().filter(|_| err == DwtDctError::TooSmall) {
                assert_eq!(err.to_string(), message, "{name}: message");
            }
        }
        None => {
            let decoded = decoded.unwrap_or_else(|err| panic!("{name}: {err}"));
            let blocks = block_bits(img.view()).expect("blocks");
            let count = case["blocks"].as_u64().expect("blocks") as usize;
            assert_eq!(blocks.len(), count, "{name}: block count");
            let expected = unpack_bits(case["block_bits"].as_str().expect("block bits"), count);
            let wrong = blocks.iter().zip(&expected).filter(|(a, b)| a != b).count();
            assert_eq!(wrong, 0, "{name}: {wrong} of {count} block bits differ");
            for length in [48, 136] {
                let bits = decoded.get(length).expect("length");
                assert_eq!(bit_string(bits), case["bits"][length.to_string()].as_str().expect("bits"), "{name}: {length}");
            }
        }
    }
    let scheme = detect_invisible_watermark(img.view()).map(|scheme| scheme.name());
    assert_eq!(scheme, case["scheme"].as_str(), "{name}: scheme");
}

#[test]
fn tracked_fixtures_match_python_bit_for_bit() {
    let mut exact = 0;
    for case in cases("fixtures") {
        let name = case["name"].as_str().expect("name");
        let img = fixture_bgr(case["fixture"].as_str().expect("fixture")).unwrap_or_else(|| panic!("{name}: decode"));
        if mat_sha256(&img) != case["sha256"].as_str().expect("sha") {
            // OpenCV's libjpeg-turbo and the image crate disagree on lossy decodes.
            eprintln!("{name}: decoder pixels differ from OpenCV, checked on the verdict only");
            assert_eq!(detect_invisible_watermark(img.view()).map(|s| s.name()), case["scheme"].as_str(), "{name}");
            continue;
        }
        check_case(&img, &case);
        exact += 1;
    }
    assert!(exact >= 20, "only {exact} fixtures decoded to OpenCV's pixels");
}

#[test]
fn sdxl_fixture_is_detected() {
    let img = fixture_bgr("provenance/sdxl-dwt-dct.png").expect("fixture");
    assert_eq!(detect_invisible_watermark(img.view()).map(|s| s.name()), Some("Stable Diffusion XL"));
}

#[test]
fn imwatermark_embeds_match_python_bit_for_bit() {
    for case in cases("embeds") {
        let name = case["name"].as_str().expect("name");
        let base = base_image(&case["base"]);
        let img = match case.get("message").filter(|m| !m.is_null()) {
            Some(message) => embed(&base, &message_bits(message)),
            None => base,
        };
        assert_eq!(mat_sha256(&img), case["sha256"].as_str().expect("sha"), "{name}: encoder port");
        check_case(&img, &case);
    }
}

#[test]
fn jpeg_recompressed_embeds_are_not_detected() {
    for case in cases("jpeg") {
        let name = case["name"].as_str().expect("name");
        let img = decode_bgr(&common::b64(case["jpeg"].as_str().expect("jpeg"))).expect("jpeg decode");
        assert!(case["scheme"].is_null(), "{name}: golden says the mark survived");
        if mat_sha256(&img) == case["sha256"].as_str().expect("sha") {
            check_case(&img, &case);
        } else {
            assert_eq!(detect_invisible_watermark(img.view()), None, "{name}");
        }
    }
}

#[test]
fn degenerate_shapes_match_python() {
    for case in cases("degenerate") {
        let img = base_image(&case["base"]);
        check_case(&img, &case);
    }
}

#[test]
fn single_length_decode_equals_multi_length() {
    let img = base_image(&serde_json::json!({"kind": "noise", "width": 300, "height": 260, "seed": 99}));
    let multi = decode_dwt_dct_lengths(img.view(), &[48, 136]).expect("decode");
    assert_eq!(decode_dwt_dct(img.view(), 48).expect("decode"), multi.get(48).expect("48"));
    assert_eq!(decode_dwt_dct(img.view(), 136).expect("decode"), multi.get(136).expect("136"));
}

#[test]
fn bgra_input_ignores_alpha() {
    let bgr = base_image(&serde_json::json!({"kind": "noise", "width": 260, "height": 300, "seed": 5}));
    let bgra = Mat::from_fn(bgr.width(), bgr.height(), 4, |x, y, c| if c < 3 { bgr.at(x, y, c) } else { 7 });
    assert_eq!(block_bits(bgr.view()).expect("bgr"), block_bits(bgra.view()).expect("bgra"));
}

#[test]
fn pattern_helpers_match_python() {
    let doc = golden("pixel_dwt", "patterns");
    assert_eq!(doc["thresholds"]["match48"].as_i64(), Some(MATCH_48));
    assert_eq!(doc["thresholds"]["sd1_fraction"].as_f64(), Some(MATCH_SD1_FRAC));
    assert_eq!(doc["sd1_hex"].as_str().map(str::to_owned), Some(SD1_STRING.iter().map(|b| format!("{b:02x}")).collect()));
    for (scheme, value) in BITS_48 {
        assert_eq!(doc["patterns48"][scheme.name()].as_u64(), Some(value), "{scheme}");
    }
    for row in doc["match48"].as_array().expect("match48") {
        let (value, reference) = (row["value"].as_u64().expect("v"), row["ref"].as_u64().expect("r"));
        assert_eq!(bits_match(value, reference, 48), row["matches"].as_i64().expect("m"));
    }
    for row in doc["sd1"].as_array().expect("sd1") {
        let hex = row["hex"].as_str().expect("hex");
        let raw: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex")).collect();
        assert_eq!(bytes_match_frac(&raw, SD1_STRING), row["fraction"].as_f64().expect("fraction"));
    }
}

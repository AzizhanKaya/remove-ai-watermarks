//! Differential tests for the local Microsoft InvisMark reader and remover (`pixel_invismark`).

mod common;

use common::{base_image, fixture_rgb, inflate, mat_sha256, unpack_bits};
use raiw_core::testing::golden;
use raiw_cv::Mat;
use raiw_pixel::invismark::{self, InvisMarkAnalysis, analyze, validate};
use raiw_pixel::{InvisMarkError, detect_local_invismark, disrupt_local_invismark, invismark_id_from_c2pa_info};
use serde_json::Value;

const SIGMA_TOLERANCE: f64 = 1e-9;

fn cases(name: &str) -> Vec<Value> {
    golden("pixel_invismark", name)["cases"].as_array().expect("cases").clone()
}

fn shape(value: &Value) -> Vec<usize> {
    value.as_array().expect("shape").iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect()
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= SIGMA_TOLERANCE * expected.abs().max(1.0)
}

/// Compare one analysis with its golden, skipping blocks Python itself placed on a boundary.
fn check_analysis(name: &str, analysis: &InvisMarkAnalysis, case: &Value) {
    let (rows, cols) = analysis.blocks();
    assert_eq!(vec![analysis.ll_shape().0, analysis.ll_shape().1], shape(&case["ll_shape"]), "{name}: ll shape");
    assert_eq!(vec![rows, cols], shape(&case["blocks"]), "{name}: blocks");
    let ll_sum: f64 = analysis.ll().iter().sum();
    assert!(close(ll_sum, case["ll_sum"].as_f64().expect("ll_sum")), "{name}: ll sum {ll_sum}");
    let near: Vec<usize> = shape(&case["near_boundary"]);
    let count = rows * cols;
    let eligible = unpack_bits(case["eligible"].as_str().expect("eligible"), count);
    let eligible_count = case["eligible_count"].as_u64().expect("count") as usize;
    let eligible_bits = unpack_bits(case["eligible_bits"].as_str().expect("bits"), eligible_count);
    let mut cursor = 0;
    for index in 0..count {
        let skip = near.contains(&index);
        if !skip {
            assert_eq!(analysis.eligible()[index], eligible[index], "{name}: eligibility of block {index}");
        }
        if eligible[index] {
            if !skip {
                assert_eq!(analysis.bits()[index] == 1, eligible_bits[cursor], "{name}: bit of block {index}");
            }
            cursor += 1;
        }
    }
    for (i, expected) in case["sigma_sample"].as_array().expect("sample").iter().enumerate() {
        let expected = expected.as_f64().expect("sigma");
        assert!(close(analysis.sigma()[i], expected), "{name}: sigma[{i}] {} != {expected}", analysis.sigma()[i]);
    }
    let sigma_sum: f64 = analysis.sigma().iter().sum();
    assert!(close(sigma_sum, case["sigma_sum"].as_f64().expect("sum")), "{name}: sigma sum");
    let expected: Vec<[u32; 2]> = case["support"]
        .as_array()
        .expect("support")
        .iter()
        .map(|pair| [pair[0].as_u64().unwrap_or(0) as u32, pair[1].as_u64().unwrap_or(0) as u32])
        .collect();
    let drift: u32 = analysis
        .support()
        .iter()
        .zip(&expected)
        .map(|(a, e)| a[0].abs_diff(e[0]) + a[1].abs_diff(e[1]))
        .sum();
    assert!(drift as usize <= 2 * near.len(), "{name}: support drift {drift} beyond {} boundary blocks", near.len());
    let detection = validate(analysis);
    match case["detection"].as_object() {
        None => assert!(detection.is_none(), "{name}: unexpected detection"),
        Some(record) => {
            let detection = detection.unwrap_or_else(|| panic!("{name}: no detection"));
            assert_eq!(Some(detection.watermark_id.to_string().as_str()), record["watermark_id"].as_str());
            let payload: String = detection.payload.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(Some(payload.as_str()), record["payload"].as_str(), "{name}: payload");
            assert_eq!(u64::from(detection.minimum_support), record["minimum_support"].as_u64().expect("support"));
        }
    }
}

fn apply_delta(base: &Mat<u8>, delta: &str) -> Mat<u8> {
    let delta = inflate(delta);
    assert_eq!(delta.len(), base.data().len());
    let data = base.data().iter().zip(delta).map(|(&b, d)| (i16::from(b) + i16::from(d as i8)) as u8).collect();
    Mat::from_vec(base.width(), base.height(), 3, data).expect("mat")
}

#[test]
fn tracked_fixtures_match_python() {
    let mut exact = 0;
    for case in cases("fixtures") {
        let name = case["name"].as_str().expect("name");
        let rgb = fixture_rgb(case["fixture"].as_str().expect("fixture")).expect("decode");
        let analysis = analyze(rgb.view());
        if case.get("error").is_some() {
            assert!(analysis.is_err(), "{name}: expected an error");
            continue;
        }
        let analysis = analysis.unwrap_or_else(|err| panic!("{name}: {err}"));
        if mat_sha256(&rgb) == case["sha256"].as_str().expect("sha") {
            check_analysis(name, &analysis, &case);
            exact += 1;
        } else {
            assert!(validate(&analysis).is_none(), "{name}: lossy decode produced a payload");
        }
    }
    assert!(exact >= 20, "only {exact} fixtures decoded to OpenCV's pixels");
}

#[test]
fn paint_fixture_is_not_the_local_dll_format() {
    let rgb = fixture_rgb("provenance/microsoft-paint-invismark.png").expect("fixture");
    assert_eq!(detect_local_invismark(rgb.view()).expect("analysis"), None);
}

#[test]
fn synthetic_carriers_detect_and_disrupt_like_python() {
    for case in cases("synthetic") {
        let name = case["name"].as_str().expect("name");
        let base = base_image(&case["base"]);
        let Some(delta) = case["carrier_delta"].as_str() else {
            let analysis = analyze(base.view());
            match case.get("error") {
                Some(_) => assert!(matches!(analysis, Err(InvisMarkError::NoBlocks { .. })), "{name}"),
                None => check_analysis(name, &analysis.expect("analysis"), &case),
            }
            continue;
        };
        let carrier = apply_delta(&base, delta);
        assert_eq!(mat_sha256(&carrier), case["carrier_sha256"].as_str().expect("sha"), "{name}: carrier");
        check_analysis(name, &analyze(carrier.view()).expect("analysis"), &case);
        let Some(disruption) = case.get("disruption") else {
            assert_eq!(disrupt_local_invismark(carrier.view(), None).expect("disrupt"), None, "{name}");
            continue;
        };
        let id = case["watermark_id"].as_str().expect("id");
        let other = "11111111-2222-3333-4444-555555555555";
        assert_eq!(disrupt_local_invismark(carrier.view(), Some(other)).expect("disrupt"), None, "{name}: wrong id");
        assert_eq!(disrupt_local_invismark(carrier.view(), Some("not a uuid")).expect("disrupt"), None);
        let removal = disrupt_local_invismark(carrier.view(), Some(id)).expect("disrupt").expect("removal");
        assert_eq!(removal.detection.watermark_id.to_string(), id);
        assert_eq!(removal.changed_blocks as u64, disruption["changed_blocks"].as_u64().expect("blocks"), "{name}");
        let expected = apply_delta(&carrier, disruption["delta"].as_str().expect("delta"));
        let differing = removal.rgb.data().iter().zip(expected.data()).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "{name}: {differing} rewritten samples differ from Python");
        assert_eq!(mat_sha256(&removal.rgb), disruption["sha256"].as_str().expect("sha"));
        assert_eq!(detect_local_invismark(removal.rgb.view()).expect("analysis"), None, "{name}: payload survived");
        let mean_change = removal
            .rgb
            .data()
            .iter()
            .zip(base.data())
            .map(|(&a, &b)| f64::from(a.abs_diff(b)))
            .sum::<f64>()
            / base.data().len() as f64;
        assert!(mean_change < 2.0, "{name}: mean change {mean_change}");
    }
}

#[test]
fn c2pa_soft_binding_ids_match_python() {
    for case in cases("c2pa") {
        let got = invismark_id_from_c2pa_info(case["algorithm"].as_str(), case["value"].as_str());
        assert_eq!(got.map(|id| id.to_string()).as_deref(), case["watermark_id"].as_str(), "{case}");
    }
    assert_eq!(invismark::INVISMARK_ALGORITHM, "com.microsoft.invismark.1");
}

#[test]
fn non_rgb_input_is_rejected() {
    let gray = Mat::<u8>::zeros(64, 64, 1);
    assert!(matches!(analyze(gray.view()), Err(InvisMarkError::Channels(1))));
}

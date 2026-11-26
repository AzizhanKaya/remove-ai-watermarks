//! Phase 3 goldens: frozen templates, per-mark scans, and the temporal arbiter.

mod common;

use common::sha256_hex;
use raiw_core::testing::{fixture, golden, workspace_root};
use raiw_video::provenance::VIDEO_VISIBLE_MARKS;
use raiw_video::visible::templates::{template_keys, template_source};
use raiw_video::visible::{FrameLocalization, Region, scan_video_marks, stabilize_localizations};
use serde_json::{Value, json};

#[test]
fn templates_match_the_python_arrays() {
    let manifest_path = workspace_root().join("crates/raiw-video/assets/video_templates/manifest.json");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(manifest_path).expect("manifest")).expect("json");
    let keys: Vec<&str> = template_keys().collect();
    assert_eq!(keys.len(), manifest.as_object().expect("object").len());
    for key in keys {
        let mat = template_source(key).expect("template");
        let want = &manifest[key];
        assert_eq!(json!([mat.height(), mat.width()]), want["shape"], "{key}");
        assert_eq!(
            sha256_hex(mat.as_bytes()),
            want["sha256"].as_str().expect("sha"),
            "{key}"
        );
    }
}

#[test]
fn stabilization_matches_python() {
    let golden = golden("video_identify", "stabilize");
    let patterns = golden["patterns"].as_object().expect("patterns");
    for mark in VIDEO_VISIBLE_MARKS {
        for (pattern, rows) in patterns {
            let detections: Vec<FrameLocalization> = rows
                .as_array()
                .expect("rows")
                .iter()
                .map(|row| FrameLocalization {
                    frame_index: row[0].as_u64().expect("index") as usize,
                    confidence: row[1].as_f64().expect("conf"),
                    region: row[2].as_array().map(|r| {
                        let v: Vec<i32> = r.iter().map(|x| x.as_i64().expect("int") as i32).collect();
                        Region::new(v[0], v[1], v[2], v[3])
                    }),
                })
                .collect();
            for provenance in [false, true] {
                let accepted = stabilize_localizations(mark, &detections, provenance);
                let xs: Vec<Option<i32>> = accepted.iter().map(|r| r.map(|r| r.x)).collect();
                let key = format!("{mark}/{pattern}/{}", i32::from(provenance));
                assert_eq!(json!(xs), golden["accepted_x"][&key], "{key}");
            }
        }
    }
}

fn check_scan(key: &str, want: &Value) {
    let scans = scan_video_marks(&fixture(key), &VIDEO_VISIBLE_MARKS, false).expect("scan");
    for (mark, scan) in scans {
        let expected = &want[&mark];
        let boxed: Vec<(usize, Region)> = scan
            .detections
            .iter()
            .filter_map(|d| d.region.map(|r| (d.frame_index, r)))
            .collect();
        let boxed_json = raiw_core::to_py_json(&boxed, raiw_core::JsonStyle::COMPACT);
        let conf_sum: f64 = scan.detections.iter().map(|d| d.confidence).sum();
        let conf_max = scan.detections.iter().map(|d| d.confidence).fold(0.0, f64::max);
        let context = format!("{key} {mark}");
        assert_eq!(
            json!(scan.detections.len()),
            expected["frames"],
            "{context} frames"
        );
        assert_eq!(json!(boxed.len()), expected["boxed"], "{context} boxed");
        assert_eq!(
            sha256_hex(boxed_json.as_bytes()),
            expected["regions_sha256"].as_str().expect("sha"),
            "{context} regions"
        );
        let n = scan.detections.len() as f64;
        assert!(
            (conf_sum - expected["conf_sum"].as_f64().expect("sum")).abs() <= 1e-4 * n,
            "{context} conf sum {conf_sum}"
        );
        assert!(
            (conf_max - expected["conf_max"].as_f64().expect("max")).abs() <= 1e-4,
            "{context} conf max {conf_max}"
        );
    }
}

#[test]
fn gallery_scans_match_python() {
    common::ffmpeg_threads();
    let scans = golden("video_identify", "scans");
    for (key, want) in scans.as_object().expect("scans") {
        if key.ends_with("example.mp4") {
            check_scan(key, want);
        }
    }
}

#[test]
#[ignore = "slow: scans every tracked provider and provenance clip"]
fn provider_scans_match_python() {
    common::ffmpeg_threads();
    let scans = golden("video_identify", "scans");
    for (key, want) in scans.as_object().expect("scans") {
        if !key.ends_with("example.mp4") {
            check_scan(key, want);
        }
    }
}

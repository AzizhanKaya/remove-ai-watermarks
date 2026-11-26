//! Phase 2 goldens and behavior: provenance rules, identification, stream-copy metadata removal,
//! source/output validation, and the media-command exit contract.

mod common;

use std::path::Path;

use common::{clip, sha256_hex};
use raiw_core::testing::{fixture, golden};
use raiw_metadata::AiMetadata;
use raiw_video::api::recommend_video_recipe;
use raiw_video::cli;
use raiw_video::provenance::{
    VIDEO_VISIBLE_MARKS, confirms_video_provenance, contradicts_video_provenance, invisible_watermark_hints,
    is_bytedance_video_provenance, markers_claim_ai, platform_from_video_metadata,
    video_invisible_target_cohort,
};
use raiw_video::{
    IdentifyVideoOptions, VideoError, VideoRuntime, identify_video, inspect_video_metadata,
    parse_mark_selector, remove_video_metadata, video_output, video_source,
};
use serde_json::{Value, json};

fn canonical(key: &str) -> std::path::PathBuf {
    std::fs::canonicalize(fixture(key)).expect("fixture")
}

fn markers_of(value: &Value) -> AiMetadata {
    value
        .as_object()
        .expect("markers")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().expect("string").to_owned()))
        .collect()
}

#[test]
fn provenance_rules_match_python() {
    for case in golden("video_identify", "rules").as_array().expect("cases") {
        let markers = markers_of(&case["markers"]);
        let context = format!("{markers:?}");
        assert_eq!(
            json!(markers_claim_ai(&markers)),
            case["claims_ai"],
            "{context} claims_ai"
        );
        assert_eq!(
            json!(platform_from_video_metadata(&markers)),
            case["platform"],
            "{context} platform"
        );
        let hints = invisible_watermark_hints(&markers);
        assert_eq!(json!(hints), case["hints"], "{context} hints");
        assert_eq!(
            json!(is_bytedance_video_provenance(&markers, None)),
            case["bytedance"],
            "{context} bytedance"
        );
        assert_eq!(
            json!(video_invisible_target_cohort(&markers, None, None)),
            case["cohort"],
            "{context} cohort"
        );
        assert_eq!(
            json!(video_invisible_target_cohort(&markers, Some("veo"), None)),
            case["cohort_veo"],
            "{context} cohort veo"
        );
        let contradicts: Vec<&str> = VIDEO_VISIBLE_MARKS
            .iter()
            .copied()
            .filter(|m| contradicts_video_provenance(m, &markers))
            .collect();
        assert_eq!(json!(contradicts), case["contradicts"], "{context} contradicts");
        let confirms: Vec<&str> = VIDEO_VISIBLE_MARKS
            .iter()
            .copied()
            .filter(|m| confirms_video_provenance(m, &markers))
            .collect();
        assert_eq!(json!(confirms), case["confirms"], "{context} confirms");
        for (mark, want) in case["recipes"].as_object().expect("recipes") {
            let selected = (mark != "None").then_some(mark.as_str());
            let recipe = recommend_video_recipe(&markers, selected, markers_claim_ai(&markers), &hints, &[]);
            assert_eq!(json!(recipe), *want, "{context} recipe {mark}");
        }
    }
}

#[test]
fn metadata_only_reports_match_python() {
    let runtime = VideoRuntime::discover();
    let options = IdentifyVideoOptions {
        check_visible: false,
        ..IdentifyVideoOptions::default()
    };
    for (key, want) in golden("video_identify", "reports").as_object().expect("reports") {
        let report = identify_video(&runtime, &canonical(key), &options).expect("identify");
        let got: Value = serde_json::from_str(&cli::to_json(&report)).expect("json");
        assert_eq!(got, want["metadata_only"], "{key}");
    }
}

#[test]
fn stream_copy_metadata_removal_is_byte_identical() {
    let dir = tempfile::tempdir().expect("dir");
    for (key, want) in golden("video_identify", "metadata_removal")
        .as_object()
        .expect("removals")
    {
        let source = fixture(key);
        let out = dir.path().join(format!(
            "{}_clean.mp4",
            source.file_stem().expect("stem").to_string_lossy()
        ));
        let result = remove_video_metadata(&source, Some(&out), true, None).expect("remove");
        assert_eq!(
            serde_json::to_value(&result.detected).expect("json"),
            want["detected"],
            "{key} detected"
        );
        assert_eq!(
            serde_json::to_value(&result.remaining).expect("json"),
            want["remaining"],
            "{key} remaining"
        );
        assert_eq!(
            sha256_hex(&std::fs::read(&out).expect("read")),
            want["output_sha256"].as_str().expect("sha"),
            "{key} bytes"
        );
    }
}

#[test]
#[ignore = "slow: scans every tracked clip with all 23 detectors"]
fn visible_reports_match_python() {
    common::ffmpeg_threads();
    let runtime = VideoRuntime::discover();
    for (key, want) in golden("video_identify", "reports").as_object().expect("reports") {
        let report =
            identify_video(&runtime, &canonical(key), &IdentifyVideoOptions::default()).expect("identify");
        let got: Value = serde_json::from_str(&cli::to_json(&report)).expect("json");
        assert_eq!(got, want["visible"], "{key}");
    }
}

#[test]
fn gallery_reports_match_python_for_a_sample() {
    common::ffmpeg_threads();
    let runtime = VideoRuntime::discover();
    let reports = golden("video_identify", "reports");
    for key in [
        "visible/veo/example.mp4",
        "provenance/higgsfield-genjutsu-motion-control.mp4",
    ] {
        let report =
            identify_video(&runtime, &canonical(key), &IdentifyVideoOptions::default()).expect("identify");
        let got: Value = serde_json::from_str(&cli::to_json(&report)).expect("json");
        assert_eq!(got, reports[key]["visible"], "{key}");
    }
}

#[allow(clippy::needless_pass_by_value)]
fn message(error: VideoError) -> String {
    error.to_string()
}

#[test]
fn sources_are_validated_like_python() {
    let dir = tempfile::tempdir().expect("dir");
    let missing = dir.path().join("missing.mp4");
    assert!(
        matches!(video_source(&missing), Err(VideoError::NotFound(m)) if m.starts_with("Video does not exist"))
    );
    let as_dir = dir.path().join("folder.mp4");
    std::fs::create_dir(&as_dir).expect("mkdir");
    assert!(message(video_source(&as_dir).unwrap_err()).starts_with("Video source must be a file"));
    let bad_ext = dir.path().join("clip.gif");
    std::fs::write(&bad_ext, b"GIF89a").expect("write");
    assert_eq!(
        message(video_source(&bad_ext).unwrap_err()),
        "Unsupported video format .gif; expected one of: .avi, .flv, .m4v, .mkv, .mov, .mp4, .webm"
    );
    let wrong = dir.path().join("wrong.mkv");
    std::fs::copy(clip("cfr_audio"), &wrong).expect("copy");
    assert!(
        message(video_source(&wrong).unwrap_err())
            .starts_with("Video content does not match its .mkv extension")
    );
    let truncated = clip("truncated");
    assert!(message(video_source(truncated).unwrap_err()).starts_with("Cannot read video"));
    let source = clip("cfr_audio");
    assert!(
        message(video_output(source, Some(Path::new("x.mkv")), "metadata removal").unwrap_err())
            .contains("does not change containers to .mkv")
    );
    assert!(
        message(video_output(source, Some(source), "metadata removal").unwrap_err())
            .contains("requires a distinct output path")
    );
    assert!(
        video_output(source, None, "metadata removal")
            .expect("default")
            .to_string_lossy()
            .ends_with("cfr_audio_clean.mp4")
    );
}

#[test]
fn mark_selectors_follow_python() {
    assert_eq!(parse_mark_selector("auto").expect("auto").0.len(), 23);
    assert_eq!(parse_mark_selector("sora").expect("sora"), (vec!["sora"], false));
    assert_eq!(
        parse_mark_selector("sora, veo,sora").expect("multi"),
        (vec!["sora", "veo"], true)
    );
    let error = message(parse_mark_selector("nope").unwrap_err());
    assert!(
        error.starts_with("Unsupported visible video mark; expected auto, sora, veo"),
        "{error}"
    );
    assert!(error.ends_with(", generic_ai_label, or kling"), "{error}");
}

#[test]
fn metadata_inspection_reports_markers() {
    let report = inspect_video_metadata(&fixture("provenance/xai-grok-imagine-video.mp4")).expect("inspect");
    assert!(report.has_ai_metadata);
    let json = cli::metadata_check_json(&report);
    assert_eq!(json.exit_code, 0);
    assert!(json.lines[0].contains("\"has_ai_metadata\": true"));
    let (lines, rows) = cli::metadata_report(&report.source, report.has_ai_metadata, &report.markers);
    assert!(lines[0].starts_with("  Warning: AI metadata detected in xai-grok-imagine-video.mp4"));
    assert_eq!(rows.len(), report.markers.len());
}

#[test]
fn media_command_exit_contract() {
    use raiw_video::{VideoMetadataResult, VideoVisibleResult};
    let mut leftover = AiMetadata::new();
    leftover.insert("c2pa_manifest".into(), "present".into());
    let visible_none = VideoVisibleResult {
        source: "a.mp4".into(),
        output: None,
        mark: "auto".into(),
        total_frames: 10,
        detected_frames: 0,
        removed_frames: 0,
        remaining_metadata: leftover.clone(),
        marks: Vec::new(),
        trimmed_bumper_frames: 0,
    };
    for as_json in [false, true] {
        assert_eq!(
            cli::visible_output(&visible_none, "auto", as_json).exit_code,
            cli::EXIT_NO_SIGNAL,
            "json={as_json}"
        );
        let wrote = VideoVisibleResult {
            output: Some("b.mp4".into()),
            ..visible_none.clone()
        };
        assert_eq!(
            cli::visible_output(&wrote, "auto", as_json).exit_code,
            1,
            "survived json={as_json}"
        );
        let clean = VideoVisibleResult {
            remaining_metadata: AiMetadata::new(),
            ..wrote.clone()
        };
        assert_eq!(
            cli::visible_output(&clean, "auto", as_json).exit_code,
            0,
            "clean json={as_json}"
        );
        let metadata = VideoMetadataResult {
            source: "a.mp4".into(),
            output: "b.mp4".into(),
            detected: leftover.clone(),
            remaining: leftover.clone(),
        };
        assert_eq!(cli::metadata_remove_output(&metadata, as_json).exit_code, 1);
    }
    let text = cli::visible_output(
        &VideoVisibleResult {
            output: Some("b.mp4".into()),
            remaining_metadata: AiMetadata::new(),
            ..visible_none
        },
        "auto",
        false,
    );
    assert_eq!(
        text.lines.last().map(String::as_str),
        Some("  Audio watermark: UNVERIFIED; source audio copied unchanged if present")
    );
}

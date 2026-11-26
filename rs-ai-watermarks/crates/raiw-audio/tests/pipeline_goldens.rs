//! Inspection, verification and removal orchestration against `tools/goldens/audio_pipeline.py`.

mod common;
mod fakes;

use std::path::Path;
use std::sync::Arc;

use common::{encode_raw, golden, synth};
use raiw_audio::io::FfmpegTools;
use raiw_audio::options::{Device, Method, Scheme, Vendor};
use raiw_audio::policy::{Permissions, ProcessingArgs, adaptive_auto_profiles, should_auto_scrub};
use raiw_audio::report::Markers;
use raiw_audio::{
    AudioError, AudioRuntime, IdentifyOptions, RemovalRequest, identify_audio, remove_audio_invisible,
};
use raiw_core::testing::assert_json_close;
use raiw_core::{JsonStyle, to_py_json};
use serde::Serialize;
use serde_json::Value;

/// Replace the random 16-hex staging token in `.<stem>-<token><suffix>` names.
fn normalize_tokens(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'-'
            && i + 17 <= bytes.len()
            && bytes[i + 1..i + 17].iter().all(u8::is_ascii_hexdigit)
            && text[..i].rfind("/.").is_some_and(|p| !text[p..i].contains('"'))
        {
            out.push_str("-<TOKEN>");
            i += 17;
            continue;
        }
        out.push(char::from(bytes[i]));
        i += 1;
    }
    out
}

fn scrubbed<T: Serialize>(value: &T, root: &Path) -> Value {
    let text = to_py_json(value, JsonStyle::COMPACT).replace(&root.display().to_string(), "<ROOT0>");
    serde_json::from_str(&normalize_tokens(&text)).unwrap()
}

fn normalized(value: &Value) -> Value {
    serde_json::from_str(&normalize_tokens(&to_py_json(value, JsonStyle::COMPACT))).unwrap()
}

/// Same values within `tolerance`, and the same key order everywhere.
fn assert_same(actual: &Value, expected: &Value, tolerance: f64, context: &str) {
    fn keys(value: &Value, at: &str, out: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                out.push(format!(
                    "{at}:{}",
                    map.keys().cloned().collect::<Vec<_>>().join(",")
                ));
                for (k, v) in map {
                    keys(v, &format!("{at}.{k}"), out);
                }
            }
            Value::Array(items) => {
                for (i, v) in items.iter().enumerate() {
                    keys(v, &format!("{at}[{i}]"), out);
                }
            }
            _ => {}
        }
    }
    assert_json_close(actual, expected, tolerance, context);
    let (mut a, mut e) = (Vec::new(), Vec::new());
    keys(actual, "$", &mut a);
    keys(expected, "$", &mut e);
    assert_eq!(a, e, "{context}: key order");
}

fn runtime(case: &Value, source: &Path) -> AudioRuntime {
    let markers: Markers = case["markers"]
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
                .collect()
        })
        .unwrap_or_default();
    AudioRuntime::bare()
        .with_decoders(fakes::decoders(&case["fakes"]))
        .with_codecs(fakes::codecs(&case["fakes"]))
        .with_metadata(Arc::new(fakes::FakeMetadata {
            source: source.canonicalize().unwrap(),
            markers,
            claims_ai: case["claims_ai"].as_bool().unwrap(),
        }))
}

fn request(kwargs: &Value) -> RemovalRequest {
    let get = |k: &str| kwargs.get(k);
    let flag = |k: &str| get(k).and_then(Value::as_bool).unwrap_or(false);
    let defaults = ProcessingArgs::default();
    RemovalRequest {
        args: ProcessingArgs {
            method: get("method").map_or(Method::Auto, |v| v.as_str().unwrap().parse().unwrap()),
            scheme: get("scheme").map_or(Scheme::AudioSeal, |v| v.as_str().unwrap().parse().unwrap()),
            verify: get("verify").and_then(Value::as_bool).unwrap_or(true),
            bandwidth_kbps: get("bandwidth_kbps")
                .and_then(Value::as_f64)
                .unwrap_or(defaults.bandwidth_kbps),
            vocoder_device: get("vocoder_device")
                .map_or(Device::Auto, |v| v.as_str().unwrap().parse().unwrap()),
            vendor: None,
        },
        permissions: Permissions {
            force: flag("force"),
            allow_unverified: flag("allow_unverified"),
            allow_positive: flag("allow_positive"),
            allow_quality_override: flag("allow_quality_override"),
        },
        ..RemovalRequest::default()
    }
}

fn error_type(err: &AudioError) -> &'static str {
    match err {
        AudioError::Invalid(_) => "ValueError",
        AudioError::Failed(_) | AudioError::Io(_) => "RuntimeError",
        AudioError::Unavailable(_) => "ImportError",
        AudioError::NotFound(_) => "FileNotFoundError",
        AudioError::Verification(_) => "AudioVerificationIncomplete",
    }
}

fn run_scenario(case: &Value) {
    let name = case["name"].as_str().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let source = root.join("src.wav");
    let spec = &case["spec"];
    let args: Vec<&str> = case["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    encode_raw(
        &synth(spec),
        spec["rate"].as_u64().unwrap() as u32,
        &source,
        &args,
        case["layout"].as_str(),
    );
    let runtime = runtime(case, &source);
    let kwargs = &case["kwargs"];
    let output = root.join(kwargs.get("output").and_then(Value::as_str).unwrap_or("out.wav"));
    let outcome: Result<Value, AudioError> = if case["kind"] == "identify" {
        let options = IdentifyOptions {
            check_invisible: kwargs
                .get("check_invisible")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            scheme: kwargs
                .get("scheme")
                .map_or(Scheme::AudioSeal, |v| v.as_str().unwrap().parse().unwrap()),
            ..IdentifyOptions::default()
        };
        identify_audio(&runtime, &source, &options).map(|report| {
            check_adaptive(case, &runtime, &report);
            scrubbed(&report, &root)
        })
    } else {
        remove_audio_invisible(&runtime, &source, &output, &request(kwargs)).map(|r| scrubbed(&r, &root))
    };
    match (&outcome, case.get("result"), case.get("error")) {
        // Delivered files are bit-identical, so `after.sha256` must match too.
        (Ok(actual), Some(expected), _) => assert_same(actual, &normalized(expected), 1e-6, name),
        (Err(err), _, Some(expected)) => {
            let message = err.to_string().replace(&root.display().to_string(), "<ROOT0>");
            assert_eq!(
                message,
                expected["message"].as_str().unwrap(),
                "{name}: error message"
            );
            let kind = expected["type"].as_str().unwrap();
            assert_eq!(error_type(err), kind, "{name}: error type");
            if let AudioError::Verification(details) = err {
                assert_same(
                    &scrubbed(&details.before, &root),
                    &normalized(&expected["before"]),
                    1e-6,
                    &format!("{name}: before"),
                );
                assert_same(
                    &scrubbed(&details.after, &root),
                    &normalized(&expected["after"]),
                    1e-6,
                    &format!("{name}: after"),
                );
            }
        }
        (Ok(_), None, Some(expected)) => panic!("{name}: expected error {expected}"),
        (Err(err), Some(_), _) => panic!("{name}: unexpected error {err}"),
        _ => unreachable!(),
    }
    assert_eq!(
        output.exists(),
        case["output_exists"].as_bool().unwrap(),
        "{name}: publication"
    );
    let staged: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with('.'))
        .collect();
    assert!(staged.is_empty(), "{name}: leftover staged files {staged:?}");
}

fn check_adaptive(case: &Value, runtime: &AudioRuntime, report: &raiw_audio::AudioProvenanceReport) {
    let name = case["name"].as_str().unwrap();
    for (scheme, by_vendor) in case["adaptive"].as_object().unwrap() {
        let scheme: Scheme = scheme.parse().unwrap();
        for (vendor, expected) in by_vendor.as_object().unwrap() {
            let vendor: Option<Vendor> = (vendor != "None").then(|| vendor.parse().unwrap());
            let profiles: Vec<&str> = adaptive_auto_profiles(runtime, scheme, report, vendor)
                .iter()
                .map(|p| p.as_str())
                .collect();
            let expected: Vec<&str> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(profiles, expected, "{name}: adaptive {scheme} {vendor:?}");
        }
    }
    assert_eq!(
        should_auto_scrub(runtime, report, Method::Auto, None),
        case["auto_scrub"].as_bool().unwrap(),
        "{name}: auto scrub"
    );
}

fn scenarios(filter: impl Fn(&str) -> bool) {
    assert!(FfmpegTools::is_available(), "ffmpeg and ffprobe are required");
    let golden = golden("pipeline");
    let cases: Vec<&Value> = golden["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| filter(c["name"].as_str().unwrap()))
        .collect();
    assert!(!cases.is_empty());
    for case in cases {
        run_scenario(case);
    }
}

#[test]
fn identify_scenarios_match_python() {
    scenarios(|n| n.starts_with("identify_") || n.starts_with("recipe_"));
}

#[test]
fn spectral_and_policy_removal_scenarios_match_python() {
    scenarios(|n| {
        n.starts_with("remove_")
            && !n.contains("dac")
            && !n.contains("snac")
            && !n.contains("encodec")
            && !n.contains("fallback")
            && !n.contains("elevenlabs")
    });
}

#[test]
fn codec_removal_scenarios_match_python() {
    scenarios(|n| {
        n.starts_with("remove_")
            && (n.contains("dac")
                || n.contains("snac")
                || n.contains("encodec")
                || n.contains("fallback")
                || n.contains("elevenlabs"))
    });
}

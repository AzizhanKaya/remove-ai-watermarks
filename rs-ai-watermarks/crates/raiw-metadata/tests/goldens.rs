//! Differential goldens from `tools/goldens/metadata.py`: scans, strips and records per input.

use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use raiw_core::sha256_bytes;
use raiw_core::testing::{fixture, golden};
use raiw_metadata::record::collect_metadata_record;
use raiw_metadata::{
    aigc_label, get_ai_metadata, has_ai_metadata, remove_ai_metadata, strip_and_verify, unreadable_reason,
};
use serde_json::{Map, Value, json};

fn error_entry(kind: &str, message: impl Into<String>) -> Value {
    let message: String = message.into();
    json!({"error": kind, "message": message})
}

fn record_summary(path: &Path) -> Value {
    let Value::Object(mut record) = collect_metadata_record(path) else {
        unreachable!("records are objects");
    };
    let store = record.shift_remove("c2pa_store");
    for key in ["metadata_base64", "tail_base64"] {
        let raw = B64
            .decode(
                record
                    .shift_remove(key)
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .unwrap_or_default(),
            )
            .unwrap_or_default();
        let stem = key.trim_end_matches("_base64");
        record.insert(format!("{stem}_sha256"), json!(sha256_bytes(&raw)));
        record.insert(format!("{stem}_len"), json!(raw.len()));
    }
    record.insert("has_c2pa_store".into(), json!(store.is_some()));
    json!({"ok": Value::Object(record)})
}

fn strip_error(err: &raiw_metadata::StripError) -> Value {
    match err {
        raiw_metadata::StripError::Value(_) => error_entry("ValueError", err.to_string()),
        raiw_metadata::StripError::Ffmpeg(_) => {
            let message = err.to_string();
            let head = message.split(": ").next().unwrap_or_default().to_owned();
            error_entry("RuntimeError", head)
        }
        _ => error_entry("OSError", err.to_string()),
    }
}

const RANDOM_ID_EXTS: &[&str] = &[".mkv", ".webm", ".mka", ".ogg", ".oga", ".opus"];

/// Packets and tags of an output whose muxer writes random IDs.
fn media_digest(path: &Path) -> Value {
    let run = |args: &[&str]| {
        std::process::Command::new(args[0])
            .args(&args[1..])
            .arg(path)
            .args(if args[0] == "ffmpeg" {
                &["-map", "0", "-c", "copy", "-f", "framemd5", "-"][..]
            } else {
                &[][..]
            })
            .output()
            .unwrap()
            .stdout
    };
    let packets = run(&["ffmpeg", "-v", "error", "-i"]);
    let tags = run(&[
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format_tags:stream_tags",
        "-of",
        "json",
    ]);
    let tags: Value = serde_json::from_slice(&tags).unwrap();
    json!({"framemd5_sha256": sha256_bytes(&packets), "tags": tags, "size": std::fs::metadata(path).unwrap().len()})
}

fn run_case(path: &Path, work: &Path) -> Map<String, Value> {
    let out_dir = work.join("out");
    std::fs::create_dir_all(&out_dir).unwrap();
    let mut entry = Map::new();
    entry.insert(
        "has_ai_metadata".into(),
        match has_ai_metadata(path) {
            Ok(flag) => json!({"ok": flag}),
            Err(err) => error_entry("OSError", err.to_string()),
        },
    );
    entry.insert("get_ai_metadata".into(), json!({"ok": get_ai_metadata(path)}));
    entry.insert("aigc_label".into(), json!({"ok": aigc_label(path)}));
    entry.insert("unreadable_reason".into(), json!(unreadable_reason(path)));
    entry.insert("record".into(), record_summary(path));
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    for keep in [true, false] {
        let tag = if keep { "keep" } else { "all" };
        let target = out_dir.join(format!("{tag}-{name}"));
        let removed = match remove_ai_metadata(path, Some(&target), keep) {
            Ok(written) if RANDOM_ID_EXTS.contains(&raiw_core::fs::suffix_lower(&written).as_str()) => {
                json!({"ok": media_digest(&written)})
            }
            Ok(written) => {
                let bytes = std::fs::read(&written).unwrap();
                json!({"ok": {"sha256": sha256_bytes(&bytes), "size": bytes.len()}})
            }
            Err(err) => strip_error(&err),
        };
        entry.insert(
            format!("remove_keep_{}", if keep { "True" } else { "False" }),
            removed,
        );
        let verified = out_dir.join(format!("v{tag}-{name}"));
        let outcome = match strip_and_verify(path, Some(&verified), keep) {
            Ok(outcome) => json!({"ok": outcome.remaining}),
            Err(raiw_metadata::VerifyError::Strip(err)) => strip_error(&err),
            Err(err) => error_entry("ValueError", err.to_string()),
        };
        entry.insert(
            format!("verify_keep_{}", if keep { "True" } else { "False" }),
            outcome,
        );
    }
    entry
}

fn normalize(value: &Value, work: &Path) -> Value {
    let text = serde_json::to_string(value).unwrap();
    let fixtures = fixture("").display().to_string();
    let fixtures = fixtures.trim_end_matches('/');
    let resolved = std::fs::canonicalize(fixtures).unwrap().display().to_string();
    let text = text
        .replace(&work.display().to_string(), "{dir}")
        .replace(fixtures, "{fixtures}")
        .replace(&resolved, "{fixtures}");
    serde_json::from_str(&text).unwrap()
}

fn shrink(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(map.into_iter().map(|(k, v)| (k, shrink(v))).collect()),
        Value::Array(items) => Value::Array(items.into_iter().map(shrink).collect()),
        Value::String(text) if text.chars().count() > 160 => {
            json!(format!(
                "sha256:{}:{}",
                sha256_bytes(text.as_bytes()),
                text.chars().count()
            ))
        }
        other => other,
    }
}

fn forensic_summary(path: &Path) -> Value {
    match raiw_metadata::forensic::collect_forensic_metadata(path) {
        Ok(Value::Object(mut record)) => {
            for key in ["mtime", "birthtime", "file"] {
                record.shift_remove(key);
            }
            let store = record
                .get("c2pa_store")
                .is_some_and(|s| s.as_object().is_some_and(|m| !m.is_empty()));
            record.insert("c2pa_store".into(), json!(store));
            json!({"ok": shrink(Value::Object(record))})
        }
        Ok(_) => unreachable!("records are objects"),
        Err(err) => error_entry("OSError", err.to_string()),
    }
}

fn detectors(path: &Path) -> Value {
    json!({"ok": {
        "xai_signature": raiw_metadata::xai_signature(path),
        "iptc_ai_system": raiw_metadata::iptc_ai_system(path),
        "samsung_genai": raiw_metadata::samsung_genai(path),
        "huggingface_job": raiw_metadata::huggingface_job(path),
        "c2pa_cloud_manifest": raiw_metadata::c2pa_cloud_manifest(path),
        "synthid_source": raiw_metadata::synthid_source(path, None),
        "exif_generator": raiw_metadata::exif_generator(path),
    }})
}

fn input_path(case: &Value, synth: &Path) -> PathBuf {
    if let Some(rel) = case["fixture"].as_str() {
        return fixture(rel);
    }
    let name = case["name"].as_str().unwrap();
    let path = synth.join(name);
    std::fs::write(&path, B64.decode(case["input_base64"].as_str().unwrap()).unwrap()).unwrap();
    path
}

/// Known, documented divergences: (case name suffix, field prefix, reason).
const KNOWN_DIVERGENCES: &[(&str, &str, &str)] = &[
    (
        "recraft-v3.webp",
        "remove_keep_",
        "WebP lossless re-encode is not libwebp 1.6.0",
    ),
    (
        "exif.webp",
        "remove_keep_",
        "WebP lossless re-encode is not libwebp 1.6.0",
    ),
    (
        "software.tif",
        "record",
        "Pillow Exif.tobytes re-serialization is approximated by piexif.dump",
    ),
    (
        "raw_profile_exif.png",
        "record",
        "Pillow Exif.tobytes re-serialization is approximated by piexif.dump",
    ),
];

fn known_divergence(name: &str, field: &str) -> bool {
    KNOWN_DIVERGENCES
        .iter()
        .any(|(case, prefix, _)| name.ends_with(case) && field.starts_with(prefix))
}

fn check_suite(suite: &str) {
    let cases = golden("metadata", suite);
    let work = tempfile::tempdir().unwrap();
    let synth = work.path().join("synth");
    std::fs::create_dir_all(&synth).unwrap();
    let mut failures = Vec::new();
    let only = std::env::var("RAIW_ONLY").ok();
    for case in cases.as_array().unwrap() {
        if only
            .as_deref()
            .is_some_and(|filter| !case["name"].as_str().unwrap_or_default().contains(filter))
        {
            continue;
        }
        let path = input_path(case, &synth);
        let mut entry = run_case(&path, work.path());
        if case.get("detectors").is_some() {
            entry.insert("detectors".into(), detectors(&path));
            entry.insert("forensic".into(), forensic_summary(&path));
        }
        let actual = normalize(&Value::Object(entry), work.path());
        for (key, expected) in case.as_object().unwrap() {
            if matches!(key.as_str(), "name" | "fixture" | "input_base64") {
                continue;
            }
            if actual[key] != *expected && !known_divergence(case["name"].as_str().unwrap_or_default(), key) {
                failures.push(format!(
                    "{} {key}:\n  rust:   {}\n  python: {expected}",
                    case["name"], actual[key]
                ));
            }
        }
    }
    if let Some(keep) = std::env::var_os("RAIW_KEEP_OUT") {
        for item in std::fs::read_dir(work.path().join("out")).unwrap().flatten() {
            let _ = std::fs::copy(item.path(), Path::new(&keep).join(item.file_name()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn media_containers_match_python() {
    check_suite("media");
}

#[test]
fn images_match_python() {
    check_suite("images");
}

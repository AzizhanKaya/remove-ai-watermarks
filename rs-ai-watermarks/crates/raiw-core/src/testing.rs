//! Golden-file helpers for differential tests against the Python reference implementation.
//!
//! Goldens live in `rs-ai-watermarks/goldens/<suite>/<name>.json` and are produced by
//! `tools/goldens/<suite>.py` run with `uv run python` from the repository root.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// The Python repository root (`/opt/remove-ai-watermarks`), which holds `data/fixtures`.
pub fn repo_root() -> PathBuf {
    if let Some(root) = std::env::var_os("RAIW_REPO_ROOT") {
        return PathBuf::from(root);
    }
    workspace_root()
        .parent()
        .expect("workspace lives inside the repository")
        .to_path_buf()
}

/// The Rust workspace root (`rs-ai-watermarks/`).
pub fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(2)
        .expect("crates/<name> layout")
        .to_path_buf()
}

/// Absolute path of a tracked fixture, given relative to `data/fixtures`.
pub fn fixture(relative: impl AsRef<Path>) -> PathBuf {
    repo_root().join("data").join("fixtures").join(relative)
}

pub fn golden_path(suite: &str, name: &str) -> PathBuf {
    workspace_root()
        .join("goldens")
        .join(suite)
        .join(format!("{name}.json"))
}

/// Load a golden JSON document, panicking with the regeneration hint when it is missing.
pub fn golden(suite: &str, name: &str) -> Value {
    let path = golden_path(suite, name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "missing golden {}: {err}; run `uv run python rs-ai-watermarks/tools/goldens/{suite}.py`",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("invalid golden {}: {err}", path.display()))
}

/// Compare two JSON values, treating numbers as equal within `tolerance` relative error.
pub fn assert_json_close(actual: &Value, expected: &Value, tolerance: f64, context: &str) {
    if let Some(diff) = json_diff(actual, expected, tolerance, "$") {
        panic!("{context}: {diff}");
    }
}

fn json_diff(actual: &Value, expected: &Value, tolerance: f64, at: &str) -> Option<String> {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64()?, e.as_f64()?);
            let scale = a.abs().max(e.abs()).max(1.0);
            ((a - e).abs() > tolerance * scale).then(|| format!("{at}: {a} != {e}"))
        }
        (Value::Array(a), Value::Array(e)) => {
            if a.len() != e.len() {
                return Some(format!("{at}: length {} != {}", a.len(), e.len()));
            }
            a.iter()
                .zip(e)
                .enumerate()
                .find_map(|(i, (a, e))| json_diff(a, e, tolerance, &format!("{at}[{i}]")))
        }
        (Value::Object(a), Value::Object(e)) => {
            let missing: Vec<_> = e.keys().filter(|k| !a.contains_key(*k)).collect();
            let extra: Vec<_> = a.keys().filter(|k| !e.contains_key(*k)).collect();
            if !missing.is_empty() || !extra.is_empty() {
                return Some(format!("{at}: missing keys {missing:?}, extra keys {extra:?}"));
            }
            e.iter()
                .find_map(|(k, e)| json_diff(&a[k], e, tolerance, &format!("{at}.{k}")))
        }
        _ => (actual != expected).then(|| format!("{at}: {actual} != {expected}")),
    }
}

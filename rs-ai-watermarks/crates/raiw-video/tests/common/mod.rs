//! Shared test support: rebuild the golden synthetic clips with the system ffmpeg.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use raiw_core::testing::{fixture, golden};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub struct Clips {
    _dir: tempfile::TempDir,
    pub paths: HashMap<String, PathBuf>,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn ffmpeg_threads() {
    raiw_video::set_ffmpeg_threads(4);
}

fn build() -> Clips {
    ffmpeg_threads();
    let manifest = golden("video", "clips");
    let dir = tempfile::tempdir().expect("temp dir");
    let base: Vec<String> = serde_json::from_value(manifest["base_args"].clone()).expect("base args");
    let mut paths: HashMap<String, PathBuf> = HashMap::new();
    for clip in manifest["clips"].as_array().expect("clips") {
        let name = clip["name"].as_str().expect("name").to_owned();
        let out = dir
            .path()
            .join(format!("{name}{}", clip["suffix"].as_str().expect("suffix")));
        let args: Vec<String> = clip["args"]
            .as_array()
            .expect("args")
            .iter()
            .map(|a| {
                let a = a.as_str().expect("arg");
                a.strip_prefix("{clip:")
                    .and_then(|rest| rest.strip_suffix('}'))
                    .map_or_else(|| a.to_owned(), |key| paths[key].to_string_lossy().into_owned())
            })
            .collect::<Vec<_>>();
        let status = Command::new("ffmpeg")
            .args(&base)
            .args(&args)
            .arg(&out)
            .status()
            .expect("ffmpeg");
        assert!(status.success(), "building clip {name}");
        paths.insert(name, out);
    }
    for (name, spec) in manifest["truncated"].as_object().expect("truncated") {
        let base_name = spec[0].as_str().expect("base");
        let fraction = spec[1].as_f64().expect("fraction");
        let data = std::fs::read(&paths[base_name]).expect("read");
        let suffix = paths[base_name]
            .extension()
            .expect("ext")
            .to_string_lossy()
            .into_owned();
        let out = dir.path().join(format!("{name}.{suffix}"));
        std::fs::write(&out, &data[..(data.len() as f64 * fraction) as usize]).expect("write");
        paths.insert(name.clone(), out);
    }
    for (name, digest) in manifest["sha256"].as_object().expect("sha") {
        let actual = sha256_hex(&std::fs::read(&paths[name]).expect("read clip"));
        assert_eq!(
            Some(actual.as_str()),
            digest.as_str(),
            "clip {name} differs from the golden build"
        );
    }
    Clips { _dir: dir, paths }
}

pub fn clips() -> &'static Clips {
    static CLIPS: OnceLock<Clips> = OnceLock::new();
    CLIPS.get_or_init(build)
}

/// Resolve a golden key (`clip:<name>` or a fixture path) to a file.
pub fn resolve(key: &str) -> PathBuf {
    key.strip_prefix("clip:")
        .map_or_else(|| fixture(key), |name| clips().paths[name].clone())
}

pub fn clip(name: &str) -> &'static Path {
    &clips().paths[name]
}

/// A golden number as f64 (handles ints and floats).
pub fn num(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

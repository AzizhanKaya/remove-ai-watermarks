//! Release timing against `tools/bench/metadata.py` (same sample, one cold pass per file).

use std::path::{Path, PathBuf};
use std::time::Instant;

use raiw_core::testing::fixture;

const SAMPLE: &[&str] = &[
    "provenance/chatgpt-1.png",
    "provenance/doubao-1.png",
    "provenance/mj-1.png",
    "provenance/higgsfield-kling-o1.png",
    "provenance/qwen-image.png",
    "provenance/flux-1.jpg",
    "provenance/grok-1.jpg",
    "provenance/google-photos-ios-edit.jpg",
    "provenance/recraft-v3.webp",
    "provenance/google-gemini-omni-app.mp4",
    "provenance/higgsfield-minimax-h3.mp4",
    "visible/veo/provider-original.mp4",
];

fn timed(label: &str, mut run: impl FnMut(&Path)) {
    let files: Vec<PathBuf> = SAMPLE.iter().map(fixture).collect();
    let start = Instant::now();
    for path in &files {
        run(path);
    }
    let per = start.elapsed().as_secs_f64() * 1000.0 / files.len() as f64;
    println!("{label}: {per:.1} ms/file");
}

#[test]
#[ignore = "timing run: cargo test --release --test bench -- --ignored --nocapture"]
fn metadata_timings() {
    timed("has_ai_metadata", |p| {
        let _ = raiw_metadata::has_ai_metadata(p);
    });
    let copies = tempfile::tempdir().unwrap();
    let cold: Vec<PathBuf> = SAMPLE
        .iter()
        .map(|rel| {
            let target = copies.path().join(rel.replace('/', "_"));
            std::fs::copy(fixture(rel), &target).unwrap();
            target
        })
        .collect();
    let start = Instant::now();
    for path in &cold {
        let _ = raiw_metadata::get_ai_metadata(path);
    }
    println!(
        "get_ai_metadata: {:.1} ms/file",
        start.elapsed().as_secs_f64() * 1000.0 / cold.len() as f64
    );
    let out = tempfile::tempdir().unwrap();
    timed("remove_ai_metadata", |p| {
        let _ = raiw_metadata::remove_ai_metadata(p, Some(&out.path().join(p.file_name().unwrap())), true);
    });
}

//! Timing against `tools/bench/c2pa.py`:
//! `cargo test -p raiw-c2pa --release --test bench -- --ignored --nocapture`.

use std::path::PathBuf;
use std::time::Instant;

use raiw_c2pa::{png, reader, soft_binding, store};
use raiw_container::isobmff;
use raiw_core::testing::repo_root;

const ROUNDS: u32 = 5;

fn fixtures() -> Vec<PathBuf> {
    let suffixes = [".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov"];
    let mut found = Vec::new();
    let mut stack = vec![repo_root().join("data").join("fixtures")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if suffixes.contains(&raiw_core::fs::suffix_lower(&path).as_str()) {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

fn timed(label: &str, mut work: impl FnMut()) {
    work();
    let start = Instant::now();
    for _ in 0..ROUNDS {
        work();
    }
    println!(
        "{label:<30} {:9.2} ms/round",
        start.elapsed().as_secs_f64() * 1000.0 / f64::from(ROUNDS)
    );
}

#[test]
#[ignore = "timing run; use --release --ignored --nocapture"]
fn bench_c2pa() {
    let files = fixtures();
    let stores: Vec<String> = files
        .iter()
        .filter_map(|p| reader::read_manifest_store_json(p).ok().flatten())
        .collect();
    let raws: Vec<Vec<u8>> = files
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e == "mp4"))
        .map(|p| isobmff::scan_c2pa_region(p, isobmff::DEFAULT_REGION_BYTES).unwrap())
        .filter(|r| !r.is_empty())
        .collect();
    println!(
        "{} fixtures, {} manifest stores; {ROUNDS} rounds",
        files.len(),
        stores.len()
    );
    timed("extract_c2pa_info (uncached)", || {
        for p in &files {
            std::hint::black_box(raiw_c2pa::extract_c2pa_info_uncached(p));
        }
    });
    timed("read_manifest_store_json", || {
        for p in &files {
            std::hint::black_box(reader::read_manifest_store_json(p).ok());
        }
    });
    timed("c2pa_info_from_manifest_store", || {
        for s in &stores {
            std::hint::black_box(store::info_from_manifest_json(s));
        }
    });
    timed("parse_c2pa_chunk (fallback)", || {
        for r in &raws {
            std::hint::black_box(raiw_c2pa::parse_c2pa_chunk(r));
        }
    });
    timed("soft_binding_vendors_in", || {
        for r in &raws {
            std::hint::black_box(soft_binding::vendors_in(r));
        }
    });
    timed("_extract_c2pa_info_png", || {
        for p in &files {
            std::hint::black_box(png::extract_c2pa_info_png(p));
        }
    });
}

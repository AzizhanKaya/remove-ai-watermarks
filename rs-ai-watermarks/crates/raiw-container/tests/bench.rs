//! Timing against `tools/bench/container.py`:
//! `cargo test -p raiw-container --release --test bench -- --ignored --nocapture`.

use std::path::PathBuf;
use std::time::Instant;

use raiw_container::isobmff::{self, MediaStripOptions, STREAM_SCAN_BYTES};
use raiw_container::{MetadataHooks, jpeg, png};
use raiw_core::testing::repo_root;

const ROUNDS: u32 = 5;

/// Cheap hooks with the same call pattern; the timed work is the walking.
struct NoHooks;

impl MetadataHooks for NoHooks {
    fn is_tc260_label(&self, _: &[u8]) -> bool {
        false
    }
    fn names_ai_generator(&self, _: &[u8]) -> bool {
        false
    }
    fn is_app_export(&self, _: &[u8]) -> bool {
        false
    }
    fn is_audio_creation(&self, _: &[u8]) -> bool {
        false
    }
    fn carries_ai_metadata(&self, _: &[u8]) -> bool {
        false
    }
    fn ai_exif_values(&self, _: &[u8]) -> Vec<Vec<u8>> {
        Vec::new()
    }
}

fn fixtures(suffixes: &[&str]) -> Vec<PathBuf> {
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
        "{label:<28} {:9.2} ms/round",
        start.elapsed().as_secs_f64() * 1000.0 / f64::from(ROUNDS)
    );
}

#[test]
#[ignore = "timing run; use --release --ignored --nocapture"]
fn bench_container() {
    let media = fixtures(&[".mp4", ".mov"]);
    let pngs = fixtures(&[".png"]);
    let jpegs = fixtures(&[".jpg", ".jpeg"]);
    let media_bytes: Vec<Vec<u8>> = media.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let jpeg_bytes: Vec<Vec<u8>> = jpegs.iter().map(|p| std::fs::read(p).unwrap()).collect();
    println!(
        "{} ISOBMFF, {} PNG, {} JPEG fixtures; {ROUNDS} rounds",
        media.len(),
        pngs.len(),
        jpegs.len()
    );
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.mp4");
    timed("scan_c2pa_region", || {
        for p in &media {
            std::hint::black_box(isobmff::scan_c2pa_region(p, isobmff::DEFAULT_REGION_BYTES).unwrap());
        }
    });
    timed("scan_isobmff_video_sei", || {
        for p in &media {
            std::hint::black_box(isobmff::scan_video_sei(p, false).unwrap());
        }
    });
    timed("strip_c2pa_boxes", || {
        for d in &media_bytes {
            std::hint::black_box(isobmff::strip_c2pa_boxes(d, &NoHooks));
        }
    });
    timed("blank_ai_xmp_packets", || {
        for d in &media_bytes {
            std::hint::black_box(isobmff::blank_ai_xmp_packets(d, &NoHooks));
        }
    });
    timed("strip_isobmff_media_file", || {
        for p in &media {
            isobmff::strip_media_file(p, &out, MediaStripOptions::default(), &NoHooks).unwrap();
        }
    });
    timed("strip_isobmff_image_file", || {
        for p in &media {
            isobmff::strip_image_file(p, &out, STREAM_SCAN_BYTES, &NoHooks).unwrap();
        }
    });
    timed("png_late_metadata", || {
        for p in &pngs {
            std::hint::black_box(png::late_metadata_file(p, 0).unwrap());
        }
    });
    timed("jpeg_app_payloads", || {
        for d in &jpeg_bytes {
            std::hint::black_box(jpeg::app_payloads(d));
        }
    });
}

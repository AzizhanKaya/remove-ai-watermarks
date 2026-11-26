//! Release timings for the `video` commands; compare with `tools/bench/video.py`.
//!
//! `cargo test --release -p raiw-video --test bench -- --ignored --nocapture`

use std::time::Instant;

use raiw_core::testing::fixture;
use raiw_video::api::{EraseVideoOptions, IdentifyVideoOptions, VisibleVideoOptions};
use raiw_video::visible::Region;
use raiw_video::{
    VideoRuntime, identify_video, remove_video_metadata, remove_video_region, remove_video_visible,
};
use raiw_visible::eraser::Backend;

fn best(mut run: impl FnMut()) -> f64 {
    (0..2)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore = "timing run; use --release"]
fn detector_kernel_timings() {
    use raiw_cv::{Mat, MorphOp, StructuringElement, match_template_ccoeff_normed, morphology_ex};
    let frame = Mat::<u8>::from_fn(480, 270, 1, |x, y, _| ((x * 7 + y * 13) % 251) as u8);
    let corner = Mat::<u8>::from_fn(180, 60, 1, |x, y, _| ((x * 3 + y * 5) % 241) as u8);
    let template = Mat::<u8>::from_fn(60, 22, 1, |x, y, _| ((x * 11 + y) % 199) as u8);
    let small_template = Mat::<u8>::from_fn(40, 14, 1, |x, y, _| ((x * 11 + y) % 199) as u8);
    let kernel = StructuringElement::rect(7, 7);
    let n = 200;
    let tophat = best(|| {
        for _ in 0..n {
            morphology_ex(&frame, MorphOp::TopHat, &kernel, 1).expect("tophat");
        }
    }) / f64::from(n);
    let full = best(|| {
        for _ in 0..n {
            match_template_ccoeff_normed(&frame, &template).expect("match");
        }
    }) / f64::from(n);
    let roi = best(|| {
        for _ in 0..n {
            match_template_ccoeff_normed(&corner, &small_template).expect("match");
        }
    }) / f64::from(n);
    println!("top-hat 480x270 k7: {:.3} ms", tophat * 1e3);
    println!("match_template 480x270 / 60x22: {:.3} ms", full * 1e3);
    println!("match_template 180x60 / 40x14: {:.3} ms", roi * 1e3);
}

#[test]
#[ignore = "timing run; use --release"]
fn video_command_timings() {
    raiw_video::set_ffmpeg_threads(4);
    let runtime = VideoRuntime::discover();
    let dir = tempfile::tempdir().expect("dir");
    let out = dir.path();
    let metadata = fixture("provenance/xai-grok-imagine-video.mp4");
    let identify = fixture("visible/veo/example.mp4");
    let visible = fixture("visible/sora/example.mp4");
    let erase = fixture("visible/kling/example.mp4");
    let results = [
        (
            "video_metadata_remove",
            best(|| {
                remove_video_metadata(&metadata, Some(&out.join("m.mp4")), true, None).expect("metadata");
            }),
        ),
        (
            "video_identify_visible",
            best(|| {
                identify_video(&runtime, &identify, &IdentifyVideoOptions::default()).expect("identify");
            }),
        ),
        (
            "video_visible_auto",
            best(|| {
                remove_video_visible(
                    &runtime,
                    &visible,
                    Some(&out.join("v.mp4")),
                    &VisibleVideoOptions::default(),
                )
                .expect("visible");
            }),
        ),
        (
            "video_erase_box",
            best(|| {
                let options = EraseVideoOptions {
                    regions: vec![Region::new(700, 450, 220, 70)],
                    frame_range: None,
                    backend: Backend::Cv2,
                    dilate: 3,
                    strip_metadata: true,
                    temporal_consistency: true,
                    track: false,
                };
                remove_video_region(&runtime, &erase, Some(&out.join("e.mp4")), &options).expect("erase");
            }),
        ),
    ];
    for (name, seconds) in results {
        println!("{name}: {seconds:.3}s");
    }
}

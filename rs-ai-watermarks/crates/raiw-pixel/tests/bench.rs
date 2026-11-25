//! Release timings matching `tools/bench/pixel.py`.
//!
//! `cargo test -p raiw-pixel --release --test bench -- --ignored --nocapture`.

mod common;

use std::time::Instant;

use common::base_image;
use raiw_cv::Mat;
use raiw_pixel::{decode_dwt_dct_lengths, detect_local_invismark};

const SIZES: [usize; 3] = [1024, 2048, 4096];
const REPEATS: usize = 5;

fn timed(mut run: impl FnMut()) -> f64 {
    run();
    let mut samples: Vec<f64> = (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[REPEATS / 2] * 1000.0
}

#[test]
#[ignore = "timing run; use --release --ignored --nocapture"]
fn pixel_hot_paths() {
    for side in SIZES {
        let bgr = base_image(&serde_json::json!({"kind": "noise", "width": side, "height": side, "seed": side}));
        let ms = timed(|| {
            decode_dwt_dct_lengths(bgr.view(), &[48, 136]).expect("decode");
        });
        println!("dwt_dct_{side}: {ms:.2} ms");
        let rgb = Mat::from_fn(side, side, 3, |x, y, c| bgr.at(x, y, 2 - c));
        let ms = timed(|| {
            detect_local_invismark(rgb.view()).expect("analysis");
        });
        println!("invismark_{side}: {ms:.2} ms");
    }
}

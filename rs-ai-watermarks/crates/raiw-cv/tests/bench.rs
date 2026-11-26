//! Release timing of the kernels video uses at 1080p/4K; mirrors `tools/bench/cv.py`.
//! `RAYON_NUM_THREADS=6 cargo test -p raiw-cv --release --test bench -- --ignored --nocapture`

use std::time::Instant;

use raiw_cv::*;

type Op<'a> = (&'static str, Box<dyn Fn() + 'a>);

fn best(f: &dyn Fn(), repeats: usize) -> f64 {
    f();
    (0..repeats)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn frame(w: usize, h: usize, seed: u64) -> Mat<u8> {
    let mut state = seed;
    let raw = Mat::from_fn(w, h, 3, |_, _, _| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 56) as u8
    });
    gaussian_blur(&raw, (0, 0), 3.0, 0.0, Border::Reflect101).expect("blur")
}

/// Loop one kernel (`RAIW_PROFILE=remap|warp|gauss|resize|cvt|tophat|lanczos`) for `perf record`.
#[test]
#[ignore = "profiling loop"]
fn profile_kernel() {
    let Ok(which) = std::env::var("RAIW_PROFILE") else {
        return;
    };
    let (w, h) = if std::env::var("RAIW_PROFILE_4K").is_ok() {
        (3840, 2160)
    } else {
        (1920, 1080)
    };
    let img = frame(w, h, 3);
    let gray = cvt_color(&img, ColorConversion::BgrToGray).expect("gray");
    let mx = Mat::from_fn(w, h, 1, |x, _, _| x as f32 + 0.37);
    let my = Mat::from_fn(w, h, 1, |_, y, _| y as f32 - 0.81);
    let rot = get_rotation_matrix_2d(Point2f::new(w as f32 / 2.0, h as f32 / 2.0), 7.0, 1.0);
    let rect7 = StructuringElement::rect(7, 7);
    let nxt = Mat::from_fn(w, h, 1, |x, y, _| gray.at((x + w - 2) % w, (y + h - 1) % h, 0));
    let pts = good_features_to_track(&gray, 200, 0.01, 4.0, None).expect("gftt");
    let lk = LkParams {
        win_width: 15,
        win_height: 15,
        ..LkParams::default()
    };
    for _ in 0..60 {
        match which.as_str() {
            "remap" => drop(remap(&img, &mx, &my, Interpolation::Linear, Border::Reflect101)),
            "warp" => drop(warp_affine(
                &gray,
                &rot,
                w,
                h,
                Interpolation::Linear,
                Border::ZERO,
            )),
            "gauss" => drop(gaussian_blur(&img, (0, 0), 2.0, 0.0, Border::Reflect101)),
            "resize" => drop(resize(&img, w * 2 / 3, h * 2 / 3, Interpolation::Linear)),
            "lanczos" => drop(resize(&img, w * 2 / 3, h * 2 / 3, Interpolation::Lanczos4)),
            "cvt" => drop(cvt_color(&img, ColorConversion::BgrToYCrCb)),
            "tophat" => drop(morphology_ex(&gray, MorphOp::TopHat, &rect7, 1)),
            "lk" => drop(calc_optical_flow_pyr_lk(&gray, &nxt, &pts, lk)),
            _ => return,
        }
    }
}

#[test]
#[ignore = "timing run; use --release --ignored --nocapture"]
fn kernel_timings() {
    println!("rayon threads {}", rayon::current_num_threads());
    for (w, h) in [(1920usize, 1080usize), (3840, 2160)] {
        let repeats = if w == 1920 { 5 } else { 3 };
        let img = frame(w, h, 7);
        let gray = cvt_color(&img, ColorConversion::BgrToGray).expect("gray");
        let mut mask = Mat::<u8>::zeros(w, h, 1);
        let r = (h / 12) as i32;
        fill_ellipse(
            &mut mask,
            Point::new((w / 2) as i32, (h / 2) as i32),
            (r, r),
            0.0,
            255.0,
            LineType::Eight,
        )
        .expect("mask");
        let tmpl = gray.crop(Rect::new(w / 4, h / 3, 96, 64)).expect("tmpl");
        let roi = gray
            .crop(Rect::new(w / 4 - 150, h / 3 - 100, 450, 300))
            .expect("roi");
        let ell = StructuringElement::ellipse(9);
        let rect7 = StructuringElement::rect(7, 7);
        let mx = Mat::from_fn(w, h, 1, |x, _, _| x as f32 + 0.37);
        let my = Mat::from_fn(w, h, 1, |_, y, _| y as f32 - 0.81);
        let rot = get_rotation_matrix_2d(Point2f::new(w as f32 / 2.0, h as f32 / 2.0), 7.0, 1.0);
        let (w23, h23) = (w * 2 / 3, h * 2 / 3);
        let small = resize(&gray, w / 2, h / 2, Interpolation::Area).expect("small");
        let shifted = Mat::from_fn(w / 2, h / 2, 1, |x, y, _| {
            small.at((x + w / 2 - 2) % (w / 2), y, 0)
        });
        let nxt = Mat::from_fn(w, h, 1, |x, y, _| gray.at((x + w - 2) % w, (y + h - 1) % h, 0));
        let pts = good_features_to_track(&gray, 200, 0.01, 4.0, None).expect("gftt");
        let mut ops: Vec<Op<'_>> = vec![
            (
                "resize_linear_720p",
                Box::new(|| drop(resize(&img, w23, h23, Interpolation::Linear))),
            ),
            (
                "resize_area_half",
                Box::new(|| drop(resize(&img, w / 2, h / 2, Interpolation::Area))),
            ),
            (
                "resize_area_0.37",
                Box::new(|| drop(resize(&img, w * 37 / 100, h * 37 / 100, Interpolation::Area))),
            ),
            (
                "resize_lanczos_2/3",
                Box::new(|| drop(resize(&img, w23, h23, Interpolation::Lanczos4))),
            ),
            (
                "cvt_bgr2gray",
                Box::new(|| drop(cvt_color(&img, ColorConversion::BgrToGray))),
            ),
            (
                "cvt_bgr2ycrcb",
                Box::new(|| drop(cvt_color(&img, ColorConversion::BgrToYCrCb))),
            ),
            (
                "gaussian_u8_s2",
                Box::new(|| drop(gaussian_blur(&img, (0, 0), 2.0, 0.0, Border::Reflect101))),
            ),
            (
                "sobel_f32",
                Box::new(|| drop(sobel::<u8, f32>(&gray, 1, 0, 3, 1.0, 0.0, Border::Reflect101))),
            ),
            ("dilate_ellipse9", Box::new(|| drop(dilate(&mask, &ell, 1)))),
            (
                "tophat_rect7",
                Box::new(|| drop(morphology_ex(&gray, MorphOp::TopHat, &rect7, 1))),
            ),
            (
                "match_template_roi",
                Box::new(|| drop(match_template_ccoeff_normed(&roi, &tmpl))),
            ),
            (
                "remap_linear",
                Box::new(|| drop(remap(&img, &mx, &my, Interpolation::Linear, Border::Reflect101))),
            ),
            (
                "warp_affine",
                Box::new(|| {
                    drop(warp_affine(
                        &gray,
                        &rot,
                        w,
                        h,
                        Interpolation::Linear,
                        Border::ZERO,
                    ));
                }),
            ),
        ];
        if w == 1920 {
            ops.push((
                "match_template_full",
                Box::new(|| drop(match_template_ccoeff_normed(&gray, &tmpl))),
            ));
            ops.push((
                "inpaint_telea",
                Box::new(|| drop(inpaint(&img, &mask, 3.0, InpaintMethod::Telea))),
            ));
            ops.push((
                "farneback_540p",
                Box::new(|| {
                    drop(calc_optical_flow_farneback(
                        &small,
                        &shifted,
                        FarnebackParams::default(),
                    ));
                }),
            ));
            let params = LkParams {
                win_width: 15,
                win_height: 15,
                ..LkParams::default()
            };
            let (g, n, p) = (&gray, &nxt, &pts);
            ops.push((
                "pyr_lk_200",
                Box::new(move || drop(calc_optical_flow_pyr_lk(g, n, p, params))),
            ));
        }
        for (name, f) in &ops {
            println!("{w}x{h} {name:<20} {:9.2} ms", best(f.as_ref(), repeats));
        }
    }
}

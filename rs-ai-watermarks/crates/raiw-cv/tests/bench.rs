//! Release timing of the hot kernels; mirrors `tools/bench/cv.py`.
//! `cargo test -p raiw-cv --release --test bench -- --ignored --nocapture`

use std::time::Instant;

use raiw_cv::*;

fn best(mut f: impl FnMut(), repeats: usize) -> f64 {
    f();
    (0..repeats)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn noise(side: usize, seed: u64) -> Mat<u8> {
    let mut state = seed;
    let raw = Mat::from_fn(side, side, 3, |_, _, _| {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 56) as u8
    });
    gaussian_blur(&raw, (0, 0), 3.0, 0.0, Border::Reflect101).expect("blur")
}

#[test]
#[ignore = "timing run; use --release --ignored --nocapture"]
fn kernel_timings() {
    println!("rayon threads {}", rayon::current_num_threads());
    for side in [1024usize, 4096] {
        let repeats = if side == 1024 { 7 } else { 3 };
        let img = noise(side, 7);
        let gray = cvt_color(&img, ColorConversion::BgrToGray).expect("gray");
        let gray_f = gray.convert::<f32>();
        let mut mask = Mat::<u8>::zeros(side, side, 1);
        let c = (side / 2) as i32;
        fill_ellipse(&mut mask, Point::new(c, c), ((side / 12) as i32, (side / 12) as i32), 0.0, 255.0, LineType::Eight).expect("mask");
        let tmpl = gray.crop(Rect::new(side / 4, side / 3, 96, 64)).expect("tmpl");
        let ell = StructuringElement::ellipse(9);
        let rect7 = StructuringElement::rect(7, 7);
        let mx = Mat::from_fn(side, side, 1, |x, _, _| x as f32 + 0.37);
        let my = Mat::from_fn(side, side, 1, |_, y, _| y as f32 - 0.81);
        let rot = get_rotation_matrix_2d(Point2f::new(side as f32 / 2.0, side as f32 / 2.0), 7.0, 1.0);
        let shifted = Mat::from_fn(side, side, 1, |x, y, _| gray.at((x + side - 2) % side, y, 0));
        let mut ops: Vec<(&str, Box<dyn Fn()>)> = vec![
            ("resize_linear_half", Box::new(|| drop(resize(&img, side / 2 + 1, side / 2 + 1, Interpolation::Linear)))),
            ("resize_area_0.37", Box::new(|| drop(resize(&img, side * 37 / 100, side * 37 / 100, Interpolation::Area)))),
            ("resize_lanczos_up", Box::new(|| drop(resize(&img, side * 5 / 4, side * 5 / 4, Interpolation::Lanczos4)))),
            ("cvt_bgr2gray", Box::new(|| drop(cvt_color(&img, ColorConversion::BgrToGray)))),
            ("cvt_bgr2ycrcb", Box::new(|| drop(cvt_color(&img, ColorConversion::BgrToYCrCb)))),
            ("gaussian_u8_s2", Box::new(|| drop(gaussian_blur(&img, (0, 0), 2.0, 0.0, Border::Reflect101)))),
            ("gaussian_f32_s2", Box::new(|| drop(gaussian_blur(&gray_f, (0, 0), 2.0, 0.0, Border::Reflect101)))),
            ("sobel_f32", Box::new(|| drop(sobel::<u8, f32>(&gray, 1, 0, 3, 1.0, 0.0, Border::Reflect101)))),
            ("dilate_ellipse9", Box::new(|| drop(dilate(&mask, &ell, 1)))),
            ("tophat_rect7", Box::new(|| drop(morphology_ex(&gray, MorphOp::TopHat, &rect7, 1)))),
            ("match_template", Box::new(|| drop(match_template_ccoeff_normed(&gray, &tmpl)))),
            ("remap_linear", Box::new(|| drop(remap(&img, &mx, &my, Interpolation::Linear, Border::Reflect101)))),
            ("warp_affine", Box::new(|| drop(warp_affine(&gray, &rot, side, side, Interpolation::Linear, Border::ZERO)))),
        ];
        if side == 1024 {
            ops.push(("inpaint_telea", Box::new(|| drop(inpaint(&img, &mask, 3.0, InpaintMethod::Telea)))));
            ops.push(("farneback", Box::new(|| drop(calc_optical_flow_farneback(&gray, &shifted, FarnebackParams::default())))));
        }
        for (name, f) in &ops {
            println!("{side:>5} {name:<20} {:9.2} ms", best(f, repeats));
        }
    }
}

//! Differential tests against OpenCV 4.10 outputs recorded by `tools/goldens/cv.py`.

mod common;

use common::{Array, Report, cases, decode, input};
use raiw_cv::*;
use serde_json::Value;

macro_rules! each_array {
    ($arr:expr, $m:ident => $body:expr) => {
        match $arr {
            Array::U8($m) => $body,
            Array::U16($m) => $body,
            Array::F32($m) => $body,
            Array::F64($m) => $body,
        }
    };
}

fn interp(name: &str) -> Interpolation {
    match name {
        "nearest" => Interpolation::Nearest,
        "linear" => Interpolation::Linear,
        "cubic" => Interpolation::Cubic,
        "area" => Interpolation::Area,
        "lanczos4" => Interpolation::Lanczos4,
        other => panic!("interp {other}"),
    }
}

fn border(name: &str) -> Border {
    match name {
        "reflect101" => Border::Reflect101,
        "reflect" => Border::Reflect,
        "replicate" => Border::Replicate,
        "constant" => Border::ZERO,
        "wrap" => Border::Wrap,
        other => panic!("border {other}"),
    }
}

fn p_usize(c: &Value, k: &str) -> usize {
    c["params"][k].as_u64().unwrap_or_else(|| panic!("param {k}")) as usize
}

fn p_f64(c: &Value, k: &str) -> f64 {
    c["params"][k].as_f64().unwrap_or_else(|| panic!("param {k}"))
}

fn p_str<'a>(c: &'a Value, k: &str) -> &'a str {
    c["params"][k].as_str().unwrap_or_else(|| panic!("param {k}"))
}

fn tol_for(dtype: &str, ipp: bool) -> f64 {
    match (dtype, ipp) {
        ("uint8" | "uint16", false) => 0.0,
        ("uint8" | "uint16", true) => 1.0,
        _ => 2e-5,
    }
}

#[test]
fn resize_matches_opencv() {
    let mut report = Report::default();
    let mut ref_report = Report::default();
    for c in cases("resize") {
        let name = c["name"].as_str().unwrap_or_default();
        let src = input(c["input"].as_str().unwrap_or_default());
        let method = interp(p_str(&c, "interp"));
        let expected = &c["output"][0];
        let ipp = c.get("noipp").is_some();
        let tol = tol_for(expected["dtype"].as_str().unwrap_or(""), ipp);
        each_array!(src, m => {
            let out = if c["params"].get("fx").is_some() {
                resize_by(m, p_f64(&c, "fx"), p_f64(&c, "fy"), method).expect("resize_by")
            } else {
                resize(m, p_usize(&c, "width"), p_usize(&c, "height"), method).expect("resize")
            };
            let has_values = expected.get("zlib").is_some() || expected.get("sample").is_some();
            if !ipp || has_values {
                report.check(name, &out, expected, tol);
            }
            if let Some(reference) = c.get("noipp") {
                ref_report.check(name, &out, &reference[0], tol_for(expected["dtype"].as_str().unwrap_or(""), false));
            }
        });
    }
    report.finish("resize");
    ref_report.finish("resize(no-ipp reference)");
}

fn color_code(name: &str) -> ColorConversion {
    use ColorConversion as C;
    match name {
        "bgr2rgb" => C::BgrToRgb,
        "bgr2gray" => C::BgrToGray,
        "rgb2gray" => C::RgbToGray,
        "gray2bgr" => C::GrayToBgr,
        "bgr2hsv" => C::BgrToHsv,
        "bgr2ycrcb" => C::BgrToYCrCb,
        "ycrcb2bgr" => C::YCrCbToBgr,
        "bgr2yuv" => C::BgrToYuv,
        "rgb2yuv" => C::RgbToYuv,
        "yuv2bgr" => C::YuvToBgr,
        "yuv2rgb" => C::YuvToRgb,
        "bgr2lab" => C::BgrToLab,
        "lab2bgr" => C::LabToBgr,
        "rgba2bgra" => C::RgbaToBgra,
        "bgra2rgba" => C::BgraToRgba,
        "bgra2bgr" => C::BgraToBgr,
        other => panic!("code {other}"),
    }
}

#[test]
fn cvt_color_matches_opencv() {
    let mut report = Report::default();
    for c in cases("color") {
        let name = c["name"].as_str().unwrap_or_default();
        let code = color_code(p_str(&c, "code"));
        let src = match c.get("inline_input") {
            Some(v) => decode(v),
            None => input(c["input"].as_str().unwrap_or_default()).clone(),
        };
        let expected = &c["output"][0];
        let tol = tol_for(expected["dtype"].as_str().unwrap_or(""), c.get("noipp").is_some());
        match &src {
            Array::U8(m) => report.check(name, &cvt_color(m, code).expect("u8"), expected, tol),
            Array::U16(m) => report.check(name, &cvt_color(m, code).expect("u16"), expected, tol),
            Array::F32(m) => report.check(name, &cvt_color(m, code).expect("f32"), expected, 1e-4),
            Array::F64(_) => panic!("no f64 color cases"),
        };
    }
    report.finish("color");
}

#[test]
fn gaussian_blur_matches_opencv() {
    let mut report = Report::default();
    for c in cases("gaussian") {
        let name = c["name"].as_str().unwrap_or_default();
        let ksize = (p_usize(&c, "kw"), p_usize(&c, "kh"));
        let (sx, sy) = (p_f64(&c, "sigma_x"), p_f64(&c, "sigma_y"));
        let b = border(p_str(&c, "border"));
        let expected = &c["output"][0];
        match input(c["input"].as_str().unwrap_or_default()) {
            Array::U8(m) => report.check(name, &gaussian_blur(m, ksize, sx, sy, b).expect("u8"), expected, 0.0),
            Array::U16(m) => report.check(name, &gaussian_blur(m, ksize, sx, sy, b).expect("u16"), expected, 0.0),
            Array::F32(m) => report.check(name, &gaussian_blur(m, ksize, sx, sy, b).expect("f32"), expected, 1e-4),
            Array::F64(m) => report.check(name, &gaussian_blur(m, ksize, sx, sy, b).expect("f64"), expected, 1e-9),
        };
    }
    report.finish("gaussian");
}

#[test]
fn derivatives_match_opencv() {
    let mut report = Report::default();
    for c in cases("deriv") {
        let name = c["name"].as_str().unwrap_or_default();
        let expected = &c["output"][0];
        let op = p_str(&c, "op");
        let f64_out = c["params"].get("ddepth").and_then(Value::as_str) == Some("f64");
        each_array!(input(c["input"].as_str().unwrap_or_default()), m => {
            match op {
                "sobel" => {
                    let out: Mat<f32> = sobel(m, p_usize(&c, "dx"), p_usize(&c, "dy"), p_usize(&c, "ksize"), 1.0, 0.0, Border::Reflect101).expect("sobel");
                    report.check(name, &out, expected, 1e-3);
                }
                "laplacian" if f64_out => {
                    let out: Mat<f64> = laplacian(m, 1, 1.0, 0.0, Border::Reflect101).expect("laplacian");
                    report.check(name, &out, expected, 1e-9);
                }
                "laplacian" => {
                    let out: Mat<f32> = laplacian(m, 1, 1.0, 0.0, Border::Reflect101).expect("laplacian");
                    report.check(name, &out, expected, 1e-3);
                }
                "filter2d" => {
                    let k: Vec<f64> = c["params"]["kernel"].as_array().expect("kernel").iter().filter_map(Value::as_f64).collect();
                    let out: Mat<f32> = filter_2d(m, &k, 5, 5, 0.0, Border::Reflect101).expect("filter2d");
                    report.check(name, &out, expected, 1e-4);
                }
                "sep" => {
                    let kx: Vec<f64> = c["params"]["kx"].as_array().expect("kx").iter().filter_map(Value::as_f64).collect();
                    let ky: Vec<f64> = c["params"]["ky"].as_array().expect("ky").iter().filter_map(Value::as_f64).collect();
                    let out: Mat<f32> = sep_filter_2d(m, &kx, &ky, 0.0, Border::Reflect101).expect("sep");
                    report.check(name, &out, expected, 1e-4);
                }
                other => panic!("op {other}"),
            }
        });
    }
    report.finish("deriv");
}

#[test]
fn morphology_matches_opencv() {
    let mut report = Report::default();
    for c in cases("morph") {
        let name = c["name"].as_str().unwrap_or_default();
        if name == "structuring_elements" {
            for e in c["elements"].as_array().expect("elements") {
                let (w, h) = (e["w"].as_u64().unwrap_or(0) as usize, e["h"].as_u64().unwrap_or(0) as usize);
                let shape = match e["shape"].as_str() {
                    Some("ellipse") => MorphShape::Ellipse,
                    Some("cross") => MorphShape::Cross,
                    _ => MorphShape::Rect,
                };
                report.check(&format!("se_{w}x{h}"), &StructuringElement::new(shape, w, h).to_mat(), &e["mask"], 0.0);
            }
            continue;
        }
        let k = p_usize(&c, "k");
        let se = if p_str(&c, "shape") == "ellipse" { StructuringElement::ellipse(k) } else { StructuringElement::rect(k, k) };
        let iterations = c["params"].get("iterations").and_then(Value::as_u64).unwrap_or(1) as usize;
        let op = match p_str(&c, "op") {
            "dilate" => MorphOp::Dilate,
            "erode" => MorphOp::Erode,
            "open" => MorphOp::Open,
            "close" => MorphOp::Close,
            "tophat" => MorphOp::TopHat,
            other => panic!("op {other}"),
        };
        each_array!(input(c["input"].as_str().unwrap_or_default()), m => {
            report.check(name, &morphology_ex(m, op, &se, iterations).expect("morph"), &c["output"][0], 0.0);
        });
    }
    report.finish("morph");
}

/// Extremum values must agree within `tol`; a different location is accepted only when our value
/// at OpenCV's location ties our extremum within `tol` (OpenCV's float32 DFT breaks exact ties).
fn check_minmax(report: &mut Vec<String>, name: &str, got: &MinMaxLoc, expected: &Value, tol: f64, map: Option<&Mat<f32>>) {
    let e = expected.as_array().expect("minmax");
    let loc = |v: &Value| Point::new(v[0].as_i64().unwrap_or(-9) as i32, v[1].as_i64().unwrap_or(-9) as i32);
    let (emin, emax) = (e[0].as_f64().unwrap_or(f64::NAN), e[1].as_f64().unwrap_or(f64::NAN));
    let ties = |p: Point, v: f64| map.is_some_and(|m| (f64::from(m.at(p.x as usize, p.y as usize, 0)) - v).abs() <= tol);
    let min_ok = got.min_loc == loc(&e[2]) || ties(loc(&e[2]), got.min);
    let max_ok = got.max_loc == loc(&e[3]) || ties(loc(&e[3]), got.max);
    if (got.min - emin).abs() > tol || (got.max - emax).abs() > tol || !min_ok || !max_ok {
        report.push(format!("{name}: got {got:?}, expected {e:?}"));
    }
}

#[test]
fn match_template_matches_opencv() {
    // OpenCV correlates in float32 via DFT and drifts by up to ~4.5e-3 from exact NCC in
    // near-flat windows; the exact float64 NCC is the tight reference.
    let mut report = Report::default();
    let mut exact = Report::default();
    let mut minmax_failures = Vec::new();
    for c in cases("template") {
        let name = c["name"].as_str().unwrap_or_default();
        let tpl = decode(&c["template"]);
        let out = match (input(c["input"].as_str().unwrap_or_default()), &tpl) {
            (Array::U8(img), Array::U8(t)) => match_template_ccoeff_normed(img, t).expect("u8"),
            (Array::F32(img), Array::F32(t)) => match_template_ccoeff_normed(img, t).expect("f32"),
            _ => panic!("unexpected template dtypes"),
        };
        report.check(name, &out, &c["output"][0], 5e-3);
        if let Some(reference) = c.get("exact") {
            exact.check(name, &out, reference, 1e-5);
        }
        if let Some(mm) = c.get("minmax") {
            check_minmax(&mut minmax_failures, name, &min_max_loc(&out).expect("minmax"), mm, 5e-3, Some(&out));
        }
    }
    report.finish("template");
    exact.finish("template(exact float64 NCC)");
    assert!(minmax_failures.is_empty(), "{minmax_failures:#?}");
}

#[test]
fn statistics_match_opencv() {
    let mut failures = Vec::new();
    let mut report = Report::default();
    for c in cases("stats") {
        let name = c["name"].as_str().unwrap_or_default();
        let src = match c.get("inline_input") {
            Some(v) => decode(v),
            None => input(c["input"].as_str().unwrap_or_default()).clone(),
        };
        if let Some(mean) = c.get("mean") {
            let (m, s) = each_array!(&src, a => mean_std_dev(a).expect("meanstd"));
            let close = |a: &[f64], e: &Value| {
                a.iter().zip(e.as_array().expect("list")).all(|(x, y)| (x - y.as_f64().unwrap_or(f64::NAN)).abs() <= 1e-9 * x.abs().max(1.0))
            };
            if !close(&m, mean) || !close(&s, &c["std"]) {
                failures.push(format!("{name}: {m:?} {s:?}"));
            }
        } else if let Some(mm) = c.get("minmax") {
            let got = each_array!(&src, a => min_max_loc(a).expect("minmax"));
            check_minmax(&mut failures, name, &got, mm, 0.0, None);
        } else if let Some(hist) = c.get("hist") {
            let Array::U8(img) = &src else { panic!("hist input") };
            let small = resize(img, 40, 30, Interpolation::Area).expect("resize");
            let h = calc_hist(&small, &[4, 4, 4], &[(0.0, 256.0); 3]).expect("hist");
            let hm = Mat::from_vec(h.len(), 1, 1, h).expect("hist mat");
            let l2 = normalize_l2(&hm).expect("l2");
            let to_vec = |v: &Value| v.as_array().expect("list").iter().filter_map(Value::as_f64).collect::<Vec<_>>();
            let max_diff = |a: &[f32], e: &[f64]| a.iter().zip(e).map(|(a, e)| (f64::from(*a) - e).abs()).fold(0.0, f64::max);
            if max_diff(hm.data(), &to_vec(hist)) > 0.0 || max_diff(l2.data(), &to_vec(&c["hist_l2"])) > 1e-6 {
                failures.push(format!("{name}: histogram mismatch"));
            }
        } else {
            let out = each_array!(&src, a => normalize_min_max(&a.convert::<f32>(), 0.0, 255.0).expect("normalize"));
            report.check(name, &out, &c["output"][0], 1e-4);
        }
    }
    report.finish("stats");
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn inpaint_matches_opencv() {
    let mut report = Report::default();
    let Array::U8(mask) = input("mask") else { panic!("mask") };
    for c in cases("inpaint") {
        let name = c["name"].as_str().unwrap_or_default();
        let method = if p_str(&c, "method") == "telea" { InpaintMethod::Telea } else { InpaintMethod::NavierStokes };
        let radius = p_f64(&c, "radius");
        let expected = &c["output"][0];
        match decode(&c["image"]) {
            Array::U8(img) => report.check(name, &inpaint(&img, mask, radius, method).expect("u8"), expected, 0.0),
            Array::U16(img) => report.check(name, &inpaint(&img, mask, radius, method).expect("u16"), expected, 0.0),
            Array::F32(img) => report.check(name, &inpaint(&img, mask, radius, method).expect("f32"), expected, 1e-4),
            Array::F64(_) => panic!("no f64 inpaint"),
        };
    }
    report.finish("inpaint");
}

#[test]
fn warps_match_opencv() {
    let mut report = Report::default();
    let all = cases("warp");
    let maps = all.iter().find(|c| c["name"] == "maps").expect("maps");
    let (Array::F32(mx), Array::F32(my)) = (decode(&maps["map_x"]), decode(&maps["map_y"])) else { panic!("maps") };
    let rotation = all.iter().find(|c| c["name"] == "rotation").expect("rotation");
    let rot = get_rotation_matrix_2d(Point2f::new(50.0, 50.0), 17.0, 1.0);
    for (r, e) in rot.iter().flatten().zip(rotation["m"].as_array().expect("m").iter().flat_map(|row| row.as_array().expect("row"))) {
        assert!((r - e.as_f64().unwrap_or(f64::NAN)).abs() < 1e-12, "rotation matrix {rot:?}");
    }
    for c in all.iter().filter(|c| c.get("params").is_some()) {
        let name = c["name"].as_str().unwrap_or_default();
        let method = interp(p_str(c, "interp"));
        let b = match p_str(c, "border") {
            "constant1" => Border::constant(1.0),
            other => border(other),
        };
        let expected = &c["output"][0];
        let m: Option<Affine> = c["params"].get("m").map(|m| {
            let v: Vec<f64> = m.as_array().expect("m").iter().flat_map(|r| r.as_array().expect("row").clone()).filter_map(|v| v.as_f64()).collect();
            [[v[0], v[1], v[2]], [v[3], v[4], v[5]]]
        });
        let run = |src: &Array| -> Array {
            macro_rules! go {
                ($m:expr, $variant:ident) => {
                    Array::$variant(match &m {
                        Some(a) => warp_affine($m, a, p_usize(c, "width"), p_usize(c, "height"), method, b).expect("affine"),
                        None => remap($m, &mx, &my, method, b).expect("remap"),
                    })
                };
            }
            match src {
                Array::U8(s) => go!(s, U8),
                Array::U16(s) => go!(s, U16),
                Array::F32(s) => go!(s, F32),
                Array::F64(_) => panic!("no f64 warps"),
            }
        };
        match run(input(c["input"].as_str().unwrap_or_default())) {
            Array::U8(o) => report.check(name, &o, expected, 0.0),
            Array::U16(o) => report.check(name, &o, expected, 0.0),
            Array::F32(o) => report.check(name, &o, expected, 1e-5),
            Array::F64(o) => report.check(name, &o, expected, 1e-9),
        };
    }
    report.finish("warp");
}

#[test]
fn misc_ops_match_opencv() {
    let mut report = Report::default();
    let mut failures = Vec::new();
    for c in cases("misc") {
        let name = c["name"].as_str().unwrap_or_default();
        let expected = &c["output"][0];
        let src = input(c["input"].as_str().unwrap_or_default());
        match p_str(&c, "op") {
            "cc" => {
                let Array::U8(img) = decode(&c["image"]) else { panic!("cc image") };
                let conn = if p_usize(&c, "connectivity") == 8 { Connectivity::Eight } else { Connectivity::Four };
                let comps = connected_components_with_stats(&img, conn).expect("cc");
                report.check(name, &comps.labels, expected, 0.0);
                let (n, labels) = connected_components(&img, conn).expect("cc");
                assert_eq!(labels, comps.labels);
                if n != c["count"].as_u64().unwrap_or(0) as usize {
                    failures.push(format!("{name}: count {n}"));
                }
                for (s, e) in comps.stats.iter().zip(c["stats"].as_array().expect("stats")) {
                    let e: Vec<i64> = e.as_array().expect("row").iter().filter_map(Value::as_i64).collect();
                    if [s.left, s.top, s.width, s.height, s.area].iter().map(|&v| i64::from(v)).ne(e.iter().copied()) {
                        failures.push(format!("{name}: stats {s:?} vs {e:?}"));
                    }
                }
                for (s, e) in comps.stats.iter().zip(c["centroids"].as_array().expect("centroids")) {
                    let (ex, ey) = (e[0].as_f64().unwrap_or(f64::NAN), e[1].as_f64().unwrap_or(f64::NAN));
                    if (s.centroid.x - ex).abs() > 1e-9 || (s.centroid.y - ey).abs() > 1e-9 {
                        failures.push(format!("{name}: centroid {:?} vs ({ex}, {ey})", s.centroid));
                    }
                }
            }
            "dist" => {
                // Below 2^14 pixels OpenCV hands off to IPP (ulp-level differences; sqrt(FLT_MAX)
                // when no zero pixel exists); its own trueDistTrans is the exact reference.
                let Array::U8(img) = decode(&c["image"]) else { panic!("dist image") };
                let out = distance_transform_l2(&img).expect("dist");
                match c.get("noipp") {
                    Some(reference) => {
                        report.check(&format!("{name}(no-ipp)"), &out, &reference[0], 0.0);
                        if name != "dist_all_ones" {
                            report.check(name, &out, expected, 1e-5);
                        }
                    }
                    None => {
                        report.check(name, &out, expected, 0.0);
                    }
                }
            }
            "canny" => {
                let Array::U8(img) = src else { panic!("canny input") };
                report.check(name, &canny(img, p_f64(&c, "low"), p_f64(&c, "high")).expect("canny"), expected, 0.0);
            }
            "add_weighted" => {
                let (Array::F32(a), Array::F32(b)) = (src, decode(&c["second"])) else { panic!("aw") };
                let out = add_weighted(a, p_f64(&c, "alpha"), &b, p_f64(&c, "beta"), p_f64(&c, "gamma")).expect("aw");
                report.check(name, &out, expected, 0.0);
            }
            "csa" => {
                let Array::F32(a) = src else { panic!("csa") };
                report.check(name, &convert_scale_abs(a, p_f64(&c, "alpha"), 0.0), expected, 0.0);
            }
            "magnitude" => {
                let Array::F32(g) = src else { panic!("magnitude") };
                let gx: Mat<f32> = sobel(g, 1, 0, 3, 1.0, 0.0, Border::Reflect101).expect("gx");
                let gy: Mat<f32> = sobel(g, 0, 1, 3, 1.0, 0.0, Border::Reflect101).expect("gy");
                let out = magnitude(&gx, &gy).expect("magnitude");
                report.check(name, &out, expected, 1e-4);
                report.check("magnitude(no-ipp)", &out, &c["noipp"][0], 0.0);
            }
            "border" => {
                let Array::F32(a) = src else { panic!("border") };
                let roi = a.roi(Rect::new(20, 10, 40, 30)).expect("roi");
                report.check(name, &copy_make_border(roi, 7, 3, 9, 2, border(p_str(&c, "border"))).expect("border"), expected, 0.0);
            }
            "transpose" => {
                let Array::U8(a) = src else { panic!("transpose") };
                report.check(name, &transpose(a), expected, 0.0);
            }
            "extract" => {
                let Array::U8(a) = src else { panic!("extract") };
                report.check(name, &extract_channel(a, p_usize(&c, "channel")).expect("extract"), expected, 0.0);
                let planes = split(a);
                let views: Vec<MatView<'_, u8>> = planes.iter().map(Mat::view).collect();
                assert_eq!(&merge(&views).expect("merge"), a);
            }
            other => panic!("op {other}"),
        }
    }
    report.finish("misc");
    assert!(failures.is_empty(), "{failures:#?}");
}

fn points(v: &Value) -> Vec<Point2f> {
    v.as_array()
        .expect("points")
        .iter()
        .map(|p| Point2f::new(p[0].as_f64().unwrap_or(f64::NAN) as f32, p[1].as_f64().unwrap_or(f64::NAN) as f32))
        .collect()
}

#[test]
fn flow_and_features_match_opencv() {
    let all = cases("flow");
    let get = |n: &str| all.iter().find(|c| c["name"] == n).unwrap_or_else(|| panic!("case {n}"));
    let frames = get("frames");
    let (Array::U8(a), Array::U8(b)) = (decode(&frames["a"]), decode(&frames["b"])) else { panic!("frames") };
    let mut report = Report::default();

    let pd = get("pyrdown");
    report.check("pyr_down", &pyr_down(&a).expect("pyr"), &pd["output"][0], 0.0);
    let odd = a.roi(Rect::new(0, 0, 151, 99)).expect("roi");
    report.check("pyr_down_odd", &pyr_down(odd).expect("pyr"), &pd["odd"][0], 0.0);

    report.check("min_eigen", &corner_min_eigen_val(&a, 3, 3).expect("eig"), &get("min_eigen")["output"][0], 1e-7);

    let gftt = get("gftt");
    let found = good_features_to_track(&a, 200, 0.01, 4.0, None).expect("gftt");
    let expected = points(&gftt["points"]);
    let same = found.iter().zip(&expected).filter(|(f, e)| f == e).count();
    println!("gftt: {same}/{} identical corners ({} found)", expected.len(), found.len());
    assert!(same * 100 >= expected.len() * 98, "gftt corners diverge: {same}/{}", expected.len());

    let gm = get("gftt_mask");
    let Array::U8(mask) = decode(&gm["mask"]) else { panic!("mask") };
    let found_m = good_features_to_track(&a, 32, 0.05, 3.0, Some(mask.view())).expect("gftt");
    assert_eq!(found_m, points(&gm["points"]), "masked gftt");

    for level in [3usize, 2] {
        let c = get(&format!("lk_{level}"));
        let params = LkParams { win_width: 15, win_height: 15, max_level: level, ..LkParams::default() };
        let tracks = calc_optical_flow_pyr_lk(&a, &b, &expected, params).expect("lk");
        let next = points(&c["next"]);
        let status: Vec<bool> = c["status"].as_array().expect("status").iter().map(|v| v.as_u64() == Some(1)).collect();
        let (mut worst, mut status_mismatch) = (0.0f32, 0);
        for ((t, n), s) in tracks.iter().zip(&next).zip(&status) {
            if t.found != *s {
                status_mismatch += 1;
            } else if *s {
                worst = worst.max((t.point.x - n.x).abs().max((t.point.y - n.y).abs()));
            }
        }
        println!("lk level {level}: max point diff {worst:.3e}, status mismatches {status_mismatch}");
        assert!(worst < 2e-2 && status_mismatch <= 2, "pyr lk diverges: {worst} / {status_mismatch}");
    }

    let fb = calc_optical_flow_farneback(&a, &b, FarnebackParams::default()).expect("farneback");
    report.check("farneback", &fb, &get("farneback")["output"][0], 5e-2);
    let small = get("farneback_small");
    let (Array::U8(sa), Array::U8(sb)) = (decode(&small["a"]), decode(&small["b"])) else { panic!("small") };
    let fbs = calc_optical_flow_farneback(&sa, &sb, FarnebackParams::default()).expect("farneback");
    report.check("farneback_small", &fbs, &small["output"][0], 5e-2);

    for name in ["phase", "phase_odd"] {
        let c = get(name);
        let (fa, fbm) = (a.convert::<f32>(), b.convert::<f32>());
        let (shift, response) = if name == "phase" {
            phase_correlate(&fa, &fbm).expect("phase")
        } else {
            let rolled = Mat::from_fn(a.width(), a.height(), 1, |x, y, _| {
                fa.at((x + a.width() + 5) % a.width(), (y + a.height() - 3) % a.height(), 0)
            });
            phase_correlate(fa.roi(Rect::new(0, 0, 131, 97)).expect("roi"), rolled.roi(Rect::new(0, 0, 131, 97)).expect("roi")).expect("phase")
        };
        let e = c["shift"].as_array().expect("shift");
        let (ex, ey) = (e[0].as_f64().unwrap_or(f64::NAN), e[1].as_f64().unwrap_or(f64::NAN));
        let er = c["response"].as_f64().unwrap_or(f64::NAN);
        println!("{name}: shift ({:.6}, {:.6}) vs ({ex:.6}, {ey:.6}), response {response:.6} vs {er:.6}", shift.x, shift.y);
        assert!((shift.x - ex).abs() < 1e-3 && (shift.y - ey).abs() < 1e-3 && (response - er).abs() < 1e-3);
    }
    report.finish("flow");
}

fn int_points(v: &Value) -> Vec<Point> {
    v.as_array()
        .expect("points")
        .iter()
        .map(|p| Point::new(p[0].as_i64().unwrap_or(0) as i32, p[1].as_i64().unwrap_or(0) as i32))
        .collect()
}

#[test]
fn drawing_matches_opencv() {
    let mut report = Report::default();
    for c in cases("draw") {
        let name = c["name"].as_str().unwrap_or_default();
        match name {
            "box_points" => {
                let r: Vec<f32> = c["rect"].as_array().expect("rect").iter().filter_map(Value::as_f64).map(|v| v as f32).collect();
                let rect = RotatedRect { center: Point2f::new(r[0], r[1]), width: r[2], height: r[3], angle: r[4] };
                assert_eq!(box_points(rect).to_vec(), points(&c["points"]), "box points");
            }
            "ellipse2poly" => {
                assert_eq!(ellipse2poly(Point::new(50, 40), (30, 17), 15, 0, 360, 5), int_points(&c["points"]));
            }
            "bounding_rect" => {
                let Array::U8(m) = decode(&c["mask"]) else { panic!("mask") };
                let r = bounding_rect(&m);
                let e: Vec<usize> = c["rect"].as_array().expect("rect").iter().filter_map(Value::as_u64).map(|v| v as usize).collect();
                assert_eq!([r.x, r.y, r.width, r.height].to_vec(), e);
            }
            _ if name.starts_with("fill_convex") => {
                let mut img = Mat::<u8>::zeros(120, 90, 1);
                let color = if name == "fill_convex" { 1.0 } else { 200.0 };
                fill_convex_poly(&mut img, &int_points(&c["pts"]), color, LineType::Eight);
                report.check(name, &img, &c["output"][0], 0.0);
            }
            _ => {
                let mut img = Mat::<u8>::zeros(120, 90, 1);
                let ct: Vec<i32> = c["center"].as_array().expect("center").iter().filter_map(Value::as_i64).map(|v| v as i32).collect();
                let ax: Vec<i32> = c["axes"].as_array().expect("axes").iter().filter_map(Value::as_i64).map(|v| v as i32).collect();
                let angle = c["angle"].as_f64().unwrap_or(0.0);
                fill_ellipse(&mut img, Point::new(ct[0], ct[1]), (ax[0], ax[1]), angle, 255.0, LineType::Eight).expect("ellipse");
                report.check(name, &img, &c["output"][0], 0.0);
            }
        }
    }
    report.finish("draw");
}

#[test]
fn python_helpers_match_reference() {
    use raiw_cv::{image_alignment, image_math, image_quality, tiling};
    let mut report = Report::default();
    let all = cases("helpers");
    let get = |n: &str| all.iter().find(|c| c["name"] == n).unwrap_or_else(|| panic!("case {n}"));
    report.check("dct8", &image_math::dct_matrix(8), &get("dct8")["output"][0], 1e-15);
    report.check("dct5", &image_math::dct_matrix(5), &get("dct5")["output"][0], 1e-15);

    let ssim = get("ssim");
    let (Array::U8(a), Array::U8(b)) = (input("fixture_c1"), decode(&ssim["b"])) else { panic!("ssim") };
    let value = image_quality::structural_similarity_gray(a, &b).expect("ssim");
    let expected = ssim["value"].as_f64().unwrap_or(f64::NAN);
    assert!((value - expected).abs() < 1e-12, "ssim {value} vs {expected}");

    for plan in get("plan_tiles")["plans"].as_array().expect("plans") {
        let u = |k: &str| plan[k].as_i64().unwrap_or(0);
        let tiles = tiling::plan_tiles(u("w") as usize, u("h") as usize, u("tile") as usize, u("overlap") as isize).expect("plan");
        let got: Vec<[usize; 4]> = tiles.iter().map(|t| [t.x, t.y, t.width, t.height]).collect();
        let exp: Vec<[usize; 4]> = plan["tiles"]
            .as_array()
            .expect("tiles")
            .iter()
            .map(|t| std::array::from_fn(|i| t[i].as_u64().unwrap_or(0) as usize))
            .collect();
        assert_eq!(got, exp, "plan_tiles {plan}");
    }
    let f = get("feather");
    report.check("feather", &tiling::feather_weights(37, 23, 9), &f["output"][0], 0.0);

    let lw = get("local_error_weights");
    let rgb = |k: &str| match decode(&lw[k]) {
        Array::U8(m) => m,
        _ => panic!("{k}"),
    };
    let (reference, baseline, candidate) = (rgb("reference"), rgb("baseline"), rgb("candidate"));
    let weights = image_alignment::local_error_weights(&reference, &baseline, &candidate, 24.0).expect("weights");
    report.check("local_error_weights", &weights, &lw["output"][0], 1e-5);

    let at = get("attenuate");
    let Array::F32(flow) = decode(&at["flow"]) else { panic!("flow") };
    let Array::F32(w_ref) = decode(&lw["output"][0]) else { panic!("weights") };
    let (out, weighted) = image_alignment::attenuate_regenerated(&baseline, &flow, &w_ref).expect("attenuate");
    report.check("attenuate_out", &out, &at["output"][0], 0.0);
    report.check("attenuate_flow", &weighted, &at["output"][1], 1e-6);

    for mode in ["linear", "lanczos"] {
        let c = get(&format!("align_{mode}"));
        let Array::U8(donor) = decode(&c["donor"]) else { panic!("donor") };
        let Array::F32(small) = decode(&c["small_flow"]) else { panic!("small") };
        let boxes: Vec<(usize, usize, usize, usize)> = c["boxes"]
            .as_array()
            .expect("boxes")
            .iter()
            .map(|b| (b[0].as_u64().unwrap_or(0) as usize, b[1].as_u64().unwrap_or(0) as usize, b[2].as_u64().unwrap_or(0) as usize, b[3].as_u64().unwrap_or(0) as usize))
            .collect();
        let interp = if mode == "linear" { Interpolation::Linear } else { Interpolation::Lanczos4 };
        let (out, final_flow) = image_alignment::bound_flow(&donor, &small, &boxes, interp).expect("bound");
        report.check(&format!("align_{mode}_out"), &out, &c["output"][0], 1.0);
        report.check(&format!("align_{mode}_flow"), &final_flow, &c["output"][1], 1e-4);
    }
    report.finish("helpers");
}

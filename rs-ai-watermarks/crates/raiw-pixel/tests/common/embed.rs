//! Test-only port of invisible-watermark's `EmbedMaxDct.encode`, with PyWavelets' Haar arithmetic.
//!
//! Upstream reconstructs with `idwt2((ca1, (v1, h1, d1)))`, swapping the horizontal and vertical
//! detail bands; that is reproduced because the goldens hash its exact output.

use raiw_cv::{ColorConversion, Mat, cvt_color};

const H: f64 = std::f64::consts::FRAC_1_SQRT_2;
const SCALE: f64 = 36.0;

/// A dense row-major float64 plane.
#[derive(Clone)]
struct Plane {
    width: usize,
    height: usize,
    data: Vec<f64>,
}

impl Plane {
    fn at(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    fn from_fn(width: usize, height: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        let data = (0..height).flat_map(|y| (0..width).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect();
        Self { width, height, data }
    }
}

// pywt `downsampling_convolution`: `sum = 0; sum += f[0] * x[2k + 1]; sum += f[1] * x[2k]`.
fn lo(even: f64, odd: f64) -> f64 {
    odd * H + even * H
}

fn hi(even: f64, odd: f64) -> f64 {
    odd * -H + even * H
}

// pywt `upsampling_convolution_valid_sf`: approximation first, detail accumulated after.
fn rec_even(a: f64, d: f64) -> f64 {
    H * a + H * d
}

fn rec_odd(a: f64, d: f64) -> f64 {
    H * a + -H * d
}

fn dwt_rows(p: &Plane) -> (Plane, Plane) {
    let (w, h) = (p.width, p.height / 2);
    let a = Plane::from_fn(w, h, |x, y| lo(p.at(x, 2 * y), p.at(x, 2 * y + 1)));
    let d = Plane::from_fn(w, h, |x, y| hi(p.at(x, 2 * y), p.at(x, 2 * y + 1)));
    (a, d)
}

fn dwt_cols(p: &Plane) -> (Plane, Plane) {
    let (w, h) = (p.width / 2, p.height);
    let a = Plane::from_fn(w, h, |x, y| lo(p.at(2 * x, y), p.at(2 * x + 1, y)));
    let d = Plane::from_fn(w, h, |x, y| hi(p.at(2 * x, y), p.at(2 * x + 1, y)));
    (a, d)
}

fn idwt_cols(a: &Plane, d: &Plane) -> Plane {
    Plane::from_fn(a.width * 2, a.height, |x, y| {
        let (av, dv) = (a.at(x / 2, y), d.at(x / 2, y));
        if x % 2 == 0 { rec_even(av, dv) } else { rec_odd(av, dv) }
    })
}

fn idwt_rows(a: &Plane, d: &Plane) -> Plane {
    Plane::from_fn(a.width, a.height * 2, |x, y| {
        let (av, dv) = (a.at(x, y / 2), d.at(x, y / 2));
        if y % 2 == 0 { rec_even(av, dv) } else { rec_odd(av, dv) }
    })
}

/// numpy `floor_divide` for float64 (`npy_divmod`).
fn floor_divide(a: f64, b: f64) -> f64 {
    let modulo = a % b;
    let mut div = (a - modulo) / b;
    if modulo != 0.0 && ((b < 0.0) != (modulo < 0.0)) {
        div -= 1.0;
    }
    if div == 0.0 {
        return (0.0_f64).copysign(a / b);
    }
    let floored = div.floor();
    if div - floored > 0.5 { floored + 1.0 } else { floored }
}

fn encode_frame(ca: &mut Plane, bits: &[u8]) {
    let mut num = 0;
    for by in 0..ca.height / 4 {
        for bx in 0..ca.width / 4 {
            let index = |pos: usize| (by * 4 + pos / 4) * ca.width + bx * 4 + pos % 4;
            let mut best = 1;
            for pos in 2..16 {
                if ca.data[index(pos)].abs() > ca.data[index(best)].abs() {
                    best = pos;
                }
            }
            let bit = f64::from(bits[num % bits.len()]);
            let val = ca.data[index(best)];
            ca.data[index(best)] = if val >= 0.0 {
                (floor_divide(val, SCALE) + 0.25 + 0.5 * bit) * SCALE
            } else {
                -1.0 * (floor_divide(val.abs(), SCALE) + 0.25 + 0.5 * bit) * SCALE
            };
            num += 1;
        }
    }
}

/// `WatermarkEncoder.encode(bgr, "dwtDct")` for a message given as 0/1 bits.
pub fn embed(bgr: &Mat<u8>, bits: &[u8]) -> Mat<u8> {
    let mut yuv = cvt_color(bgr, ColorConversion::BgrToYuv).expect("yuv");
    let (w4, h4) = (bgr.width() / 4 * 4, bgr.height() / 4 * 4);
    let plane = Plane::from_fn(w4, h4, |x, y| f64::from(yuv.at(x, y, 1)));
    let (a, d) = dwt_rows(&plane);
    let (mut aa, ad) = dwt_cols(&a);
    let (da, dd) = dwt_cols(&d);
    encode_frame(&mut aa, bits);
    // Swapped bands: `idwt2((ca1, (v1, h1, d1)))` puts `ad` where `da` belongs and vice versa.
    let rows_a = idwt_cols(&aa, &da);
    let rows_d = idwt_cols(&ad, &dd);
    let out = idwt_rows(&rows_a, &rows_d);
    for y in 0..h4 {
        for x in 0..w4 {
            yuv.set(x, y, 1, out.at(x, y) as i32 as u8);
        }
    }
    cvt_color(&yuv, ColorConversion::YuvToBgr).expect("bgr")
}

/// Message bits as invisible-watermark builds them from a golden `message` object.
pub fn message_bits(message: &serde_json::Value) -> Vec<u8> {
    match message["kind"].as_str() {
        Some("bits") => {
            let value = message["value"].as_u64().expect("value");
            (0..48).rev().map(|i| (value >> i & 1) as u8).collect()
        }
        Some("bytes") => {
            let hex = message["hex"].as_str().expect("hex");
            (0..hex.len() / 2)
                .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex byte"))
                .flat_map(|byte| (0..8).rev().map(move |i| byte >> i & 1))
                .collect()
        }
        other => panic!("unknown message kind {other:?}"),
    }
}

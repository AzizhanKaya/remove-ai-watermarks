//! Float CIE L*a*b* conversions: the trilinear `RGB2Lab_f` LUT path and `Lab2RGBfloat`.

use std::sync::OnceLock;

use rayon::prelude::*;

use crate::element::{Element, cv_round_f32};
use crate::mat::{Mat, MatView};

/// Element types with a Lab conversion (only `f32`, as in the Python package).
pub trait LabElement: Element {
    #[doc(hidden)]
    fn rgb_to_lab(_src: &MatView<'_, Self>, _bgr: bool) -> Option<Mat<Self>> {
        None
    }
    #[doc(hidden)]
    fn lab_to_rgb(_src: &MatView<'_, Self>, _bgr: bool) -> Option<Mat<Self>> {
        None
    }
}

impl LabElement for u8 {}
impl LabElement for u16 {}

impl LabElement for f32 {
    fn rgb_to_lab(src: &MatView<'_, f32>, bgr: bool) -> Option<Mat<f32>> {
        Some(rgb_to_lab(src, bgr))
    }
    fn lab_to_rgb(src: &MatView<'_, f32>, bgr: bool) -> Option<Mat<f32>> {
        Some(lab_to_rgb(src, bgr))
    }
}

const LAB_LUT_DIM: usize = 33;
const LAB_BASE: f32 = 16384.0;
const TRILINEAR_BASE: i32 = 16;
const GAMMA_TAB_SIZE: usize = 1024;

const SRGB2XYZ_D65: [u64; 9] = [
    0x3fda_65a1_4488_c60d,
    0x3fd6_e297_396d_0918,
    0x3fc7_1819_d239_1d58,
    0x3fcb_38cd_a6e7_5ff6,
    0x3fe6_e297_396d_0918,
    0x3fb2_79aa_e6c8_f755,
    0x3f93_cc4a_c6cd_af4b,
    0x3fbe_836e_b4e9_8138,
    0x3fee_6842_7418_d691,
];
const XYZ2SRGB_D65: [u64; 9] = [
    0x4009_ec80_4102_ff8f,
    0xbff8_982a_9930_be0e,
    0xbfdf_e7ff_583a_53b9,
    0xbfef_0425_28ae_74f3,
    0x3ffe_040f_2389_7204,
    0x3fa5_46d3_f9e7_b80b,
    0x3fac_7de5_082c_f52c,
    0xbfca_1e14_bdfd_2631,
    0x3ff0_eabe_f06b_3786,
];
const D65: [u64; 3] = [
    0x3fee_6a22_b389_2ee8,
    0x3ff0_0000_0000_0000,
    0x3ff1_6b89_5076_3a19,
];

fn raw(v: u64) -> f64 {
    f64::from_bits(v)
}

fn apply_gamma(x: f32) -> f32 {
    let xd = f64::from(x);
    if xd <= 809.0 / 20000.0 {
        (xd / (323.0 / 25.0)) as f32
    } else {
        ((xd + 11.0 / 200.0) / (1.0 + 11.0 / 200.0)).powf(12.0 / 5.0) as f32
    }
}

fn apply_inv_gamma(x: f32) -> f32 {
    let xd = f64::from(x);
    if xd <= 7827.0 / 2_500_000.0 {
        (xd * (323.0 / 25.0)) as f32
    } else {
        (xd.powf(1.0 / (12.0 / 5.0)) * (1.0 + 11.0 / 200.0) - 11.0 / 200.0) as f32
    }
}

/// OpenCV's `f32_cbrt` (softfloat): a rational approximation whose mantissa is truncated.
fn soft_cbrt(x: f32) -> f32 {
    const A: [u64; 9] = [
        0x4046_a09e_6653_ba70,
        0x4068_08f4_6c61_16e0,
        0x405d_ca97_439c_ae14,
        0x402a_dd70_d282_7500,
        0x3fc4_f15f_83f5_5d2d,
        0x402d_9e20_660e_db21,
        0x4062_ff15_c028_5815,
        0x4065_10d0_6a81_12ce,
        0x4040_fecb_c9e2_c375,
    ];
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    let bits = x.to_bits();
    if x == 0.0 {
        return 0.0;
    }
    let a = A.map(raw);
    let ex0 = ((bits >> 23) & 0xff) as i32 - 127;
    let mut shx = ex0 % 3;
    if shx >= 0 {
        shx -= 3;
    }
    let ex = (ex0 - shx) / 3 - 1;
    let fr = f64::from_bits((((shx + 1023) as u64) << 52) | (u64::from(bits & 0x7f_ffff) << 29));
    let num = (((a[0] * fr + a[1]) * fr + a[2]) * fr + a[3]) * fr + a[4];
    let den = (((a[5] * fr + a[6]) * fr + a[7]) * fr + a[8]) * fr + 1.0;
    let fr = num / den;
    let frac = ((fr.to_bits() & ((1 << 52) - 1)) >> 29) as u32;
    f32::from_bits((bits & 0x8000_0000) + (((ex + 127) as u32) << 23) + frac)
}

/// `splineBuild`: natural cubic spline coefficients, four per segment.
fn spline_build(f: &[f32]) -> Vec<f32> {
    let n = f.len() - 1;
    let mut tab = vec![0.0f32; n * 4];
    for i in 1..n {
        let t = (f[i + 1] - f[i] * 2.0 + f[i - 1]) * 3.0;
        let l = 1.0 / (4.0 - tab[(i - 1) * 4]);
        tab[i * 4] = l;
        tab[i * 4 + 1] = (t - tab[(i - 1) * 4 + 1]) * l;
    }
    let mut cn = 0.0f32;
    for j in 0..n {
        let i = n - j - 1;
        let c = tab[i * 4 + 1] - tab[i * 4] * cn;
        let b = f[i + 1] - f[i] - (cn + c * 2.0) / 3.0;
        let d = (cn - c) / 3.0;
        tab[i * 4] = f[i];
        tab[i * 4 + 1] = b;
        tab[i * 4 + 2] = c;
        tab[i * 4 + 3] = d;
        cn = c;
    }
    tab
}

#[inline]
fn spline_interpolate(x: f32, tab: &[f32], n: usize) -> f32 {
    let ix = (x as i32).clamp(0, n as i32 - 1) as usize;
    let x = x - ix as f32;
    let t = &tab[ix * 4..ix * 4 + 4];
    ((t[3] * x + t[2]) * x + t[1]) * x + t[0]
}

struct LabTables {
    lut: Vec<i16>,
    weights: Vec<i32>,
    inv_gamma: Vec<f32>,
}

fn tables() -> &'static LabTables {
    static TABLES: OnceLock<LabTables> = OnceLock::new();
    TABLES.get_or_init(build_tables)
}

fn build_tables() -> LabTables {
    let lthresh = 216.0f32 / 24389.0;
    let lscale = 841.0f32 / 108.0;
    let lbias = 16.0f32 / 116.0;
    let scale_white = [1.0 / raw(D65[0]), 1.0, 1.0 / raw(D65[2])];
    let mut s = [0.0f32; 9];
    for i in 0..3 {
        s[i * 3] = (raw(SRGB2XYZ_D65[i * 3 + 2]) * scale_white[i]) as f32;
        s[i * 3 + 1] = (raw(SRGB2XYZ_D65[i * 3 + 1]) * scale_white[i]) as f32;
        s[i * 3 + 2] = (raw(SRGB2XYZ_D65[i * 3]) * scale_white[i]) as f32;
    }
    let f9033 = 24389.0f32 / 27.0;
    let lld = (LAB_LUT_DIM - 1) as f32;
    let dim = LAB_LUT_DIM;
    let mut prev = vec![0i16; dim * dim * dim * 3];
    for p in 0..dim {
        for q in 0..dim {
            for r in 0..dim {
                let idx = p * 3 + q * dim * 3 + r * dim * dim * 3;
                let (rr, gg, bb) = (
                    apply_gamma(p as f32 / lld),
                    apply_gamma(q as f32 / lld),
                    apply_gamma(r as f32 / lld),
                );
                let x = rr * s[0] + gg * s[1] + bb * s[2];
                let y = rr * s[3] + gg * s[4] + bb * s[5];
                let z = rr * s[6] + gg * s[7] + bb * s[8];
                let f = |v: f32| {
                    if v > lthresh {
                        soft_cbrt(v)
                    } else {
                        v.mul_add(lscale, lbias)
                    }
                };
                let (fx, fy, fz) = (f(x), f(y), f(z));
                let l = if y > lthresh { 116.0 * fy - 16.0 } else { f9033 * y };
                let a = 500.0 * (fx - fy);
                let b = 200.0 * (fy - fz);
                prev[idx] = cv_round_f32(LAB_BASE * l / 100.0) as i16;
                prev[idx + 1] = cv_round_f32(LAB_BASE * (a + 128.0) / 256.0) as i16;
                prev[idx + 2] = cv_round_f32(LAB_BASE * (b + 128.0) / 256.0) as i16;
            }
        }
    }
    let mut lut = vec![0i16; dim * dim * dim * 3 * 8];
    for p in 0..dim {
        for q in 0..dim {
            for r in 0..dim {
                for c in 0..8 {
                    let (dp, dq, dr) = (c >> 2, (c >> 1) & 1, c & 1);
                    let old = (p + dp).min(dim - 1) * 3
                        + (q + dq).min(dim - 1) * dim * 3
                        + (r + dr).min(dim - 1) * dim * dim * 3;
                    let new = p * 24 + q * dim * 24 + r * dim * dim * 24 + c;
                    lut[new] = prev[old];
                    lut[new + 8] = prev[old + 1];
                    lut[new + 16] = prev[old + 2];
                }
            }
        }
    }
    let tb = TRILINEAR_BASE as usize;
    let mut weights = vec![0i32; tb * tb * tb * 8];
    for p in 0..TRILINEAR_BASE {
        for q in 0..TRILINEAR_BASE {
            for r in 0..TRILINEAR_BASE {
                let (pp, qq, rr) = (TRILINEAR_BASE - p, TRILINEAR_BASE - q, TRILINEAR_BASE - r);
                let w = &mut weights
                    [(8 * p + 8 * TRILINEAR_BASE * q + 8 * TRILINEAR_BASE * TRILINEAR_BASE * r) as usize..];
                w[..8].copy_from_slice(&[
                    pp * qq * rr,
                    pp * qq * r,
                    pp * q * rr,
                    pp * q * r,
                    p * qq * rr,
                    p * qq * r,
                    p * q * rr,
                    p * q * r,
                ]);
            }
        }
    }
    let scale = 1.0f32 / GAMMA_TAB_SIZE as f32;
    let ig: Vec<f32> = (0..=GAMMA_TAB_SIZE)
        .map(|i| apply_inv_gamma(scale * i as f32))
        .collect();
    LabTables {
        lut,
        weights,
        inv_gamma: spline_build(&ig),
    }
}

fn trilinear(cx: i32, cy: i32, cz: i32, t: &LabTables) -> [i32; 3] {
    let (tx, ty, tz) = ((cx >> 9) as usize, (cy >> 9) as usize, (cz >> 9) as usize);
    let base = 24 * tx + 24 * LAB_LUT_DIM * ty + 24 * LAB_LUT_DIM * LAB_LUT_DIM * tz;
    let (x, y, z) = ((cx >> 5) & 15, (cy >> 5) & 15, (cz >> 5) & 15);
    let w = &t.weights[(8 * x + 8 * 16 * y + 8 * 256 * z) as usize..][..8];
    let mut out = [0i32; 3];
    for (ch, o) in out.iter_mut().enumerate() {
        let corners = &t.lut[base + ch * 8..base + ch * 8 + 8];
        let sum: i32 = corners.iter().zip(w).map(|(&c, &w)| i32::from(c) * w).sum();
        *o = (sum + (1 << 11)) >> 12;
    }
    out
}

fn rgb_to_lab(src: &MatView<'_, f32>, bgr: bool) -> Mat<f32> {
    let t = tables();
    let scn = src.channels();
    let bidx = if bgr { 0 } else { 2 };
    let mut dst = Mat::zeros(src.width(), src.height(), 3);
    dst.par_rows_mut().enumerate().for_each(|(y, row)| {
        for (s, d) in src.row(y).chunks_exact(scn).zip(row.chunks_exact_mut(3)) {
            let q = |v: f32| cv_round_f32(v.clamp(0.0, 1.0) * LAB_BASE);
            let [il, ia, ib] = trilinear(q(s[bidx]), q(s[1]), q(s[bidx ^ 2]), t);
            d[0] = il as f32 * 1.0 / LAB_BASE * 100.0;
            d[1] = ia as f32 * 1.0 / LAB_BASE * 256.0 - 128.0;
            d[2] = ib as f32 * 1.0 / LAB_BASE * 256.0 - 128.0;
        }
    });
    dst
}

fn lab_coeffs(bgr: bool) -> [f32; 9] {
    let bidx = if bgr { 0 } else { 2 };
    let mut c = [0.0f32; 9];
    for i in 0..3 {
        let wp = raw(D65[i]);
        let col = |j: usize| raw(XYZ2SRGB_D65[i + j * 3]);
        c[i + (bidx ^ 2) * 3] = (col(0) * wp) as f32;
        c[i + 3] = (col(1) * wp) as f32;
        c[i + bidx * 3] = (col(2) * wp) as f32;
    }
    c
}

fn lab_to_rgb(src: &MatView<'_, f32>, bgr: bool) -> Mat<f32> {
    let t = tables();
    let c = lab_coeffs(bgr);
    let gscale = GAMMA_TAB_SIZE as f32;
    let mut dst = Mat::zeros(src.width(), src.height(), 3);
    let width = src.width();
    let simd_end = width / 8 * 8;
    dst.par_rows_mut().enumerate().for_each(|(y, row)| {
        for (x, (s, d)) in src
            .row(y)
            .chunks_exact(3)
            .zip(row.chunks_exact_mut(3))
            .enumerate()
        {
            let [r, g, b] = if x < simd_end {
                lab_pixel_simd(s, &c)
            } else {
                lab_pixel_scalar(s, &c)
            };
            for (o, v) in d.iter_mut().zip([r, g, b]) {
                *o = spline_interpolate(v * gscale, &t.inv_gamma, GAMMA_TAB_SIZE);
            }
        }
    });
    dst
}

fn lab_pixel_scalar(s: &[f32], c: &[f32; 9]) -> [f32; 3] {
    let (li, ai, bi) = (s[0], s[1], s[2]);
    let (y, fy) = if li <= 8.0 {
        let y = li / 903.3;
        (y, 7.787 * y + 16.0 / 116.0)
    } else {
        let fy = (li + 16.0) / 116.0;
        (fy * fy * fy, fy)
    };
    let f = |v: f32| {
        if v <= 6.0 / 29.0 {
            (v - 16.0 / 116.0) / 7.787
        } else {
            v * v * v
        }
    };
    let x = f(ai / 500.0 + fy);
    let z = f(fy - bi / 200.0);
    let clip = |v: f32| {
        if v < 0.0 {
            0.0
        } else if v <= 1.0 {
            v
        } else {
            1.0
        }
    };
    [
        clip(c[0] * x + c[1] * y + c[2] * z),
        clip(c[3] * x + c[4] * y + c[5] * z),
        clip(c[6] * x + c[7] * y + c[8] * z),
    ]
}

fn lab_pixel_simd(s: &[f32], c: &[f32; 9]) -> [f32; 3] {
    let (li, ai, bi) = (s[0], s[1], s[2]);
    let (y, fy) = if li <= 8.0 {
        let y = li * (1.0 / 903.3);
        (y, 7.787 * y + 16.0 / 116.0)
    } else {
        let fy = (li + 16.0) * (1.0 / 116.0);
        (fy * fy * fy, fy)
    };
    let f = |v: f32| {
        if v <= 6.0 / 29.0 {
            (v - 16.0 / 116.0) * (1.0 / 7.787)
        } else {
            v * v * v
        }
    };
    let x = f(ai * (1.0 / 500.0) + fy);
    let z = f(bi * (-1.0 / 200.0) + fy);
    let clip = |v: f32| 0.0f32.max(v.min(1.0));
    [
        clip(c[0] * x + (c[1] * y + c[2] * z)),
        clip(c[3] * x + (c[4] * y + c[5] * z)),
        clip(c[6] * x + (c[7] * y + c[8] * z)),
    ]
}

//! Shared helpers for the raiw-pixel golden tests.
#![allow(dead_code)]

pub mod embed;

use std::io::Read;
use std::path::Path;

use base64::Engine;
use raiw_core::hash::sha256_bytes;
use raiw_core::testing::fixture;
use raiw_cv::Mat;
use serde_json::Value;

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 outputs for states `seed + (i + 1) * GOLDEN_GAMMA` (`pixel_common.splitmix_u64`).
pub fn splitmix_u64(seed: u64, count: usize) -> impl Iterator<Item = u64> {
    (1..=count as u64).map(move |i| {
        let mut z = seed.wrapping_add(i.wrapping_mul(GOLDEN_GAMMA));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

fn field(spec: &Value, key: &str) -> u64 {
    spec[key].as_u64().unwrap_or_else(|| panic!("spec field {key}: {spec}"))
}

/// `pixel_common.base_image`: an HxWx3 uint8 image from a JSON spec.
pub fn base_image(spec: &Value) -> Mat<u8> {
    let (width, height, seed) = (field(spec, "width") as usize, field(spec, "height") as usize, field(spec, "seed"));
    let kind = spec["kind"].as_str().expect("kind");
    if kind == "noise-range" {
        let (lo, hi) = (field(spec, "lo"), field(spec, "hi"));
        let wide: Vec<u64> = splitmix_u64(seed, width * height * 3).collect();
        return Mat::from_fn(width, height, 3, |x, y, c| (lo + wide[(y * width + x) * 3 + c] % (hi - lo)) as u8);
    }
    let noise: Vec<i64> = splitmix_u64(seed, width * height * 3).map(|z| (z >> 56) as i64).collect();
    Mat::from_fn(width, height, 3, |x, y, c| {
        let n = noise[(y * width + x) * 3 + c];
        match kind {
            "noise" => n as u8,
            "texture" => {
                let (x, y, c) = (x as i64, y as i64, c as i64);
                let pattern = (x * 7 + y * 3 + c * 50) % 101 - 50;
                let wave = ((x / 9 + y / 13 + c) % 7) * 6 - 18;
                (128 + pattern + wave + n % 21 - 10).clamp(0, 255) as u8
            }
            "flat" => spec["value"].as_u64().unwrap_or(128) as u8,
            other => panic!("unknown base kind {other}"),
        }
    })
}

pub fn b64(text: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD.decode(text).expect("base64")
}

pub fn inflate(text: &str) -> Vec<u8> {
    let raw = b64(text);
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(&raw[..]).read_to_end(&mut out).expect("zlib");
    out
}

/// `packed_bits` in reverse: the first `count` MSB-first bits.
pub fn unpack_bits(text: &str, count: usize) -> Vec<bool> {
    inflate(text).iter().flat_map(|byte| (0..8).rev().map(move |i| byte >> i & 1 == 1)).take(count).collect()
}

pub fn bit_string(bits: &[bool]) -> String {
    bits.iter().map(|&bit| if bit { '1' } else { '0' }).collect()
}

pub fn mat_sha256(mat: &Mat<u8>) -> String {
    sha256_bytes(mat.data())
}

/// RGB(A)/gray image bytes as an OpenCV-style BGR `Mat` (`image` crate decode).
pub fn decode_bgr(bytes: &[u8]) -> Option<Mat<u8>> {
    let rgb = image::load_from_memory(bytes).ok()?.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let mut data = rgb.into_raw();
    data.chunks_exact_mut(3).for_each(|px| px.swap(0, 2));
    Mat::from_vec(width, height, 3, data).ok()
}

pub fn fixture_bgr(relative: impl AsRef<Path>) -> Option<Mat<u8>> {
    decode_bgr(&std::fs::read(fixture(relative)).ok()?)
}

/// RGB `Mat` from the `image` crate.
pub fn fixture_rgb(relative: impl AsRef<Path>) -> Option<Mat<u8>> {
    let rgb = image::open(fixture(relative)).ok()?.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    Mat::from_vec(width, height, 3, rgb.into_raw()).ok()
}

//! Golden decoding and comparison helpers shared by the differential tests.
#![allow(dead_code)]

use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

use base64::Engine;
use raiw_core::testing::golden;
use raiw_cv::{Element, Mat};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// A decoded numpy array.
#[derive(Clone, Debug)]
pub enum Array {
    U8(Mat<u8>),
    U16(Mat<u16>),
    F32(Mat<f32>),
    F64(Mat<f64>),
}

pub fn shape_of(v: &Value) -> (usize, usize, usize) {
    let s: Vec<usize> = v["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|d| d.as_u64().unwrap_or(0) as usize)
        .collect();
    match s.len() {
        1 => (s[0], 1, 1),
        2 => (s[1], s[0], 1),
        _ => (s[1], s[0], s[2]),
    }
}

fn inflate(b64: &str) -> Vec<u8> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("base64");
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(&raw[..])
        .read_to_end(&mut out)
        .expect("zlib");
    out
}

fn typed<T: Element>(bytes: &[u8]) -> Vec<T> {
    bytes
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

pub fn decode(v: &Value) -> Array {
    let (w, h, c) = shape_of(v);
    let bytes = inflate(v["zlib"].as_str().expect("array payload"));
    match v["dtype"].as_str().expect("dtype") {
        "uint8" => Array::U8(Mat::from_vec(w, h, c, typed(&bytes)).expect("u8")),
        "uint16" => Array::U16(Mat::from_vec(w, h, c, typed(&bytes)).expect("u16")),
        "float32" => Array::F32(Mat::from_vec(w, h, c, typed(&bytes)).expect("f32")),
        "float64" => Array::F64(Mat::from_vec(w, h, c, typed(&bytes)).expect("f64")),
        other => panic!("unsupported dtype {other}"),
    }
}

/// Inputs shared by every case.
pub fn inputs() -> &'static HashMap<String, Array> {
    static INPUTS: OnceLock<HashMap<String, Array>> = OnceLock::new();
    INPUTS.get_or_init(|| {
        let doc = golden("cv", "inputs");
        doc.as_object()
            .expect("inputs")
            .iter()
            .map(|(k, v)| (k.clone(), decode(v)))
            .collect()
    })
}

pub fn input(name: &str) -> &'static Array {
    inputs()
        .get(name)
        .unwrap_or_else(|| panic!("unknown input {name}"))
}

pub fn cases(suite: &str) -> Vec<Value> {
    golden("cv", suite)["cases"].as_array().expect("cases").clone()
}

pub fn sha256<T: Element>(m: &Mat<T>) -> String {
    hex::encode(Sha256::digest(m.as_bytes()))
}

pub fn values(v: &Value) -> Option<(Vec<f64>, usize)> {
    let to_f64 = |bytes: Vec<u8>| -> Vec<f64> {
        match v["dtype"].as_str().unwrap_or("") {
            "uint8" => bytes.iter().map(|&b| f64::from(b)).collect(),
            "uint16" => typed::<u16>(&bytes).into_iter().map(f64::from).collect(),
            "int16" => typed::<i16>(&bytes).into_iter().map(f64::from).collect(),
            "int32" => typed::<i32>(&bytes).into_iter().map(f64::from).collect(),
            "float32" => typed::<f32>(&bytes).into_iter().map(f64::from).collect(),
            "float64" => typed::<f64>(&bytes),
            other => panic!("unsupported dtype {other}"),
        }
    };
    if let Some(full) = v["zlib"].as_str() {
        return Some((to_f64(inflate(full)), 1));
    }
    let sample = v["sample"].as_str()?;
    Some((
        to_f64(inflate(sample)),
        v["stride"].as_u64().unwrap_or(1) as usize,
    ))
}

/// Statistics of a comparison against a golden array.
#[derive(Clone, Copy, Debug, Default)]
pub struct Diff {
    pub exact: bool,
    pub max_abs: f64,
    pub mismatches: usize,
    pub compared: usize,
}

pub fn compare<T: Element>(actual: &Mat<T>, expected: &Value) -> Diff {
    let (w, h, c) = shape_of(expected);
    assert_eq!(
        (actual.width(), actual.height(), actual.channels()),
        (w, h, c),
        "shape mismatch"
    );
    let exact = sha256(actual) == expected["sha256"].as_str().unwrap_or("");
    let mut diff = Diff {
        exact,
        ..Diff::default()
    };
    if exact {
        diff.compared = actual.data().len();
        return diff;
    }
    if let Some((vals, stride)) = values(expected) {
        for (i, e) in vals.iter().enumerate() {
            let a = actual.data()[i * stride].to_f64();
            let d = if a.is_nan() && e.is_nan() {
                0.0
            } else {
                (a - e).abs()
            };
            if d > 0.0 {
                diff.mismatches += 1;
            }
            diff.max_abs = diff.max_abs.max(if d.is_nan() { f64::INFINITY } else { d });
            diff.compared += 1;
        }
    } else {
        diff.max_abs = f64::INFINITY;
    }
    diff
}

/// Accumulates per-case results and prints a table, failing on any case beyond its tolerance.
#[derive(Default)]
pub struct Report {
    rows: Vec<(String, Diff, f64)>,
}

impl Report {
    pub fn check<T: Element>(
        &mut self,
        name: &str,
        actual: &Mat<T>,
        expected: &Value,
        tolerance: f64,
    ) -> Diff {
        let d = compare(actual, expected);
        self.rows.push((name.to_string(), d, tolerance));
        d
    }

    pub fn finish(self, suite: &str) {
        let mut failed = Vec::new();
        let exact = self.rows.iter().filter(|r| r.1.exact).count();
        let worst = self.rows.iter().map(|r| r.1.max_abs).fold(0.0, f64::max);
        for (name, d, tol) in &self.rows {
            if !d.exact {
                println!(
                    "{suite}/{name}: max_abs={:.3e} mismatches={}/{}",
                    d.max_abs, d.mismatches, d.compared
                );
            }
            if d.max_abs > *tol {
                failed.push(format!("{name} (max_abs {:.3e} > {tol:.1e})", d.max_abs));
            }
        }
        println!(
            "{suite}: {exact}/{} bit-exact, worst max_abs {worst:.3e}",
            self.rows.len()
        );
        assert!(
            failed.is_empty(),
            "{suite}: {} cases out of tolerance: {failed:#?}",
            failed.len()
        );
    }
}

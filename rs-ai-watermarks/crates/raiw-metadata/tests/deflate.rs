//! Deflate streams against `tools/goldens/metadata_deflate.py`: Python's `zlib.compress`,
//! Pillow's PNG `IDAT` chunks, and zlib-ng's medium strategy, byte for byte.

use std::path::PathBuf;

use raiw_core::sha256_bytes;
use raiw_core::testing::golden;
use raiw_metadata::deflate::{
    PngFilter, Strategy, Zlib, zip_encode, zip_encode_with, zlib_compress, zlib_compress_level,
};
use serde_json::{Value, json};

const ALPHABET: &[u8] = b"the quick brown fox jumps over a lazy dog AIGC {}\":,0123456789\n";

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 56) as u8
    }
}

fn gen_bytes(kind: &str, n: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    match kind {
        "zeros" => return vec![0; n],
        "noise" => return (0..n).map(|_| rng.next()).collect(),
        _ => {}
    }
    let mut out: Vec<u8> = Vec::with_capacity(n + 64);
    while out.len() < n {
        if kind == "runs" {
            let value = rng.next();
            let count = 1 + usize::from(rng.next()) % 64;
            out.extend(std::iter::repeat_n(value, count));
        } else if rng.next() < 64 && out.len() > 16 {
            let start = (usize::from(rng.next()) * out.len()) >> 8;
            let length = 3 + usize::from(rng.next()) % 40;
            let end = (start + length).min(out.len());
            let copy = out[start..end].to_vec();
            out.extend(copy);
        } else {
            out.push(ALPHABET[usize::from(rng.next()) % ALPHABET.len()]);
        }
    }
    out.truncate(n);
    out
}

fn bands(mode: &str) -> usize {
    match mode {
        "LA" => 2,
        "RGB" => 3,
        "RGBA" => 4,
        _ => 1,
    }
}

/// Raw rows in Pillow's rawmode, as the generator builds them.
fn pixel_rows(mode: &str, width: usize, height: usize, style: &str, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = Lcg(seed);
    if mode == "1" {
        return (0..height)
            .map(|_| {
                let mut row: Vec<u8> = (0..width.div_ceil(8)).map(|_| rng.next()).collect();
                if !width.is_multiple_of(8) {
                    let mask = (0xFFu32 << (8 - width % 8)) as u8;
                    if let Some(last) = row.last_mut() {
                        *last &= mask;
                    }
                }
                row
            })
            .collect();
    }
    let nb = bands(mode);
    (0..height)
        .map(|y| {
            let mut row = Vec::with_capacity(width * nb);
            for x in 0..width {
                for band in 0..nb {
                    let value = match style {
                        "noise" => rng.next(),
                        "flat" => (((x / 16 + y / 16) % 4) * 60 + band * 10) as u8,
                        _ => (((x * 7 + y * 3) / 4 + band * 40 + usize::from(rng.next() & 7)) & 0xFF) as u8,
                    };
                    row.push(value);
                }
            }
            row
        })
        .collect()
}

fn digest(chunks: &[Vec<u8>]) -> Value {
    let joined = chunks.concat();
    json!({
        "sha256": sha256_bytes(&joined),
        "len": joined.len(),
        "chunks": chunks.iter().map(Vec::len).collect::<Vec<_>>(),
    })
}

fn field<'a>(case: &'a Value, key: &str) -> &'a Value {
    case.get(key).unwrap_or_else(|| panic!("golden case lacks {key}"))
}

fn usize_of(case: &Value, key: &str) -> usize {
    field(case, key).as_u64().expect("integer") as usize
}

#[test]
fn pillow_idat_streams_match_on_both_zlib_builds() {
    let doc = golden("metadata", "deflate");
    let images = field(&doc, "images").as_array().expect("images");
    assert!(images.len() >= 10);
    for case in images {
        let mode = field(case, "mode").as_str().expect("mode");
        let width = usize_of(case, "width");
        let height = usize_of(case, "height");
        let style = field(case, "style").as_str().expect("style");
        let seed = field(case, "seed").as_u64().expect("seed");
        let bufsize = usize_of(case, "bufsize");
        let rows = pixel_rows(mode, width, height, style, seed);
        let palette = mode == "P";
        let bits = if mode == "1" { 1 } else { 8 * bands(mode) };
        let mut filter = PngFilter::new(rows[0].len(), bits, palette);
        let filtered: Vec<Vec<u8>> = rows.iter().map(|row| filter.filter(row).to_vec()).collect();
        let strategy = if palette {
            Strategy::Default
        } else {
            Strategy::Filtered
        };
        let label = format!("{mode} {width}x{height} {style}");
        let classic = zip_encode(&filtered, strategy, bufsize).expect("classic stream");
        assert_eq!(digest(&classic), *field(case, "classic"), "classic {label}");
        if let Some(expected) = case.get("ng") {
            let ng = zip_encode_with(&filtered, strategy, bufsize, Zlib::Ng).expect("ng stream");
            assert_eq!(digest(&ng), *expected, "zlib-ng {label}");
        }
    }
}

#[test]
fn zlib_compress_matches_python_at_every_size_and_level() {
    let doc = golden("metadata", "deflate");
    for case in field(&doc, "streams").as_array().expect("streams") {
        let kind = field(case, "kind").as_str().expect("kind");
        let n = usize_of(case, "n");
        let seed = field(case, "seed").as_u64().expect("seed");
        let level = field(case, "level").as_u64().expect("level") as u8;
        let data = gen_bytes(kind, n, seed);
        let out = if level == 6 {
            zlib_compress(&data)
        } else {
            zlib_compress_level(&data, level).expect("supported level")
        };
        assert_eq!(
            sha256_bytes(&out),
            field(case, "sha256").as_str().expect("sha"),
            "{kind} {n} level {level}"
        );
        assert_eq!(out.len(), usize_of(case, "len"), "{kind} {n} level {level}");
    }
}

#[test]
fn unsupported_parameters_are_errors() {
    assert!(zlib_compress_level(b"x", 3).is_err());
    assert!(zlib_compress_level(b"x", 10).is_err());
    assert_eq!(zlib_compress(b""), b"x\x9c\x03\x00\x00\x00\x00\x01");
}

/// Pillow's own fixture re-saves, when the scratch extraction is present on this host.
#[test]
fn fixture_resaves_match_when_extracted() {
    let Some(dir) = std::env::var_os("RAIW_DEFLATE_FIXTURES").map(PathBuf::from) else {
        return;
    };
    for (name, width, bpp) in [
        ("ideogram-c2pa.png", 64, 3),
        ("adobe-trustmark-p.png", 512, 4),
        ("flux-1.png", 1024, 3),
    ] {
        let (Ok(filtered), Ok(idat)) = (
            std::fs::read(dir.join(format!("{name}.filtered"))),
            std::fs::read(dir.join(format!("{name}.idat"))),
        ) else {
            continue;
        };
        let rows: Vec<&[u8]> = filtered.chunks(width * bpp + 1).collect();
        let chunks = zip_encode(rows, Strategy::Filtered, (width * 4).max(65536)).expect("stream");
        assert_eq!(chunks.concat(), idat, "{name}");
    }
    if let (Ok(icc), Ok(expected)) = (
        std::fs::read(dir.join("icc.raw")),
        std::fs::read(dir.join("icc.z")),
    ) {
        assert_eq!(zlib_compress(&icc), expected);
    }
}

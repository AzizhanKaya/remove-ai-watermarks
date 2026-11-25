//! Known open DWT-DCT watermarks from Stable Diffusion, SDXL and FLUX.2 (`invisible_watermark.py`).
//!
//! The marks are fragile: they do not survive JPEG re-encoding or resizing, so a match is only
//! possible on pristine originals and absence is never proof.

use raiw_cv::MatView;

use crate::dwt_dct::decode_dwt_dct_lengths;

/// An open watermark scheme identified from decoded bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpenWatermark {
    StableDiffusionXl,
    Flux2,
    StableDiffusion1,
}

impl OpenWatermark {
    /// The scheme name `identify` reports in its `invisible_watermark` signal.
    pub const fn name(self) -> &'static str {
        match self {
            Self::StableDiffusionXl => "Stable Diffusion XL",
            Self::Flux2 => "FLUX.2 (Black Forest Labs)",
            Self::StableDiffusion1 => "Stable Diffusion 1.x / 2.x",
        }
    }
}

impl std::fmt::Display for OpenWatermark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Known 48-bit `bits` watermarks (dwtDct, no key), checked in this order.
pub const BITS_48: [(OpenWatermark, u64); 2] = [
    (OpenWatermark::StableDiffusionXl, 0b1011_0011_1110_1100_1001_0000_0111_1011_1011_0001_1001_1110),
    (OpenWatermark::Flux2, 0b0010_1010_1111_1110_1000_0111_1001_1100_1111_0101_0010_1110),
];
/// The invisible-watermark default string watermark (SD 1.x / 2.x).
pub const SD1_STRING: &[u8] = b"StableDiffusionV1";
/// Matching bits out of 48 for a positive (at most four bit errors).
pub const MATCH_48: i64 = 44;
/// Fraction of the 136 string bits that must match.
pub const MATCH_SD1_FRAC: f64 = 0.92;

const SD1_BITS: usize = 8 * SD1_STRING.len();

/// Number of matching bits between two `width`-bit integers.
pub fn bits_match(value: u64, reference: u64, width: u32) -> i64 {
    i64::from(width) - i64::from((value ^ reference).count_ones())
}

/// Fraction of matching bits between two equal-length byte strings (0 otherwise).
pub fn bytes_match_frac(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let diff: u32 = a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum();
    1.0 - f64::from(diff) / (8 * b.len()) as f64
}

fn bits_to_int(bits: &[bool]) -> u64 {
    bits.iter().fold(0, |value, &bit| (value << 1) | u64::from(bit))
}

fn bits_to_bytes(bits: &[bool], nbytes: usize) -> Vec<u8> {
    bits.chunks(8)
        .take(nbytes)
        .map(|chunk| chunk.iter().enumerate().fold(0_u8, |byte, (i, &bit)| byte | (u8::from(bit) << (7 - i))))
        .collect()
}

/// The scheme matching already decoded 48- and 136-bit messages, if any.
pub fn match_open_watermark(bits48: &[bool], bits136: &[bool]) -> Option<OpenWatermark> {
    let value = bits_to_int(bits48);
    if let Some(&(scheme, _)) = BITS_48.iter().find(|(_, reference)| bits_match(value, *reference, 48) >= MATCH_48) {
        return Some(scheme);
    }
    let raw = bits_to_bytes(bits136, SD1_STRING.len());
    (bytes_match_frac(&raw, SD1_STRING) >= MATCH_SD1_FRAC).then_some(OpenWatermark::StableDiffusion1)
}

/// Decode a BGR image and return the open watermark scheme it carries, if any.
///
/// Decoder refusals (images under 256x256 area, empty planes) report no scheme, like Python.
pub fn detect_invisible_watermark(bgr: MatView<'_, u8>) -> Option<OpenWatermark> {
    let decoded = decode_dwt_dct_lengths(bgr, &[48, SD1_BITS]).ok()?;
    match_open_watermark(decoded.get(48)?, decoded.get(SD1_BITS)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_match_python() {
        assert_eq!(bits_match(0b1010, 0b1010, 4), 4);
        assert_eq!(bits_match(0b1010, 0b1011, 4), 3);
        assert_eq!(bytes_match_frac(SD1_STRING, SD1_STRING), 1.0);
        assert_eq!(bytes_match_frac(b"abc", b"abcd"), 0.0);
        assert_eq!(bytes_match_frac(b"", b""), 0.0);
    }

    #[test]
    fn bit_packing_is_msb_first_and_truncates() {
        let bits = [true, false, false, false, false, false, false, true, true];
        assert_eq!(bits_to_bytes(&bits, 1), vec![0x81]);
        assert_eq!(bits_to_bytes(&bits, 4), vec![0x81, 0x80]);
        assert_eq!(bits_to_int(&bits[..4]), 0b1000);
    }

    #[test]
    fn exact_patterns_match_their_scheme() {
        for (scheme, reference) in BITS_48 {
            let bits: Vec<bool> = (0..48).rev().map(|i| reference >> i & 1 == 1).collect();
            assert_eq!(match_open_watermark(&bits, &[false; SD1_BITS]), Some(scheme));
        }
        let sd1: Vec<bool> =
            SD1_STRING.iter().flat_map(|byte| (0..8).rev().map(move |i| byte >> i & 1 == 1)).collect();
        assert_eq!(match_open_watermark(&[false; 48], &sd1), Some(OpenWatermark::StableDiffusion1));
        assert_eq!(match_open_watermark(&[false; 48], &[false; SD1_BITS]), None);
    }
}

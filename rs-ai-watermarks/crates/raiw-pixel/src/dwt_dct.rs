//! DWT-DCT decoder compatible with invisible-watermark's `dwtDct` path (`dwt_dct.py`).
//!
//! The bit test is `peak % 36 > 18.0` against Haar approximation values that sit exactly on
//! multiples of 0.5 for uint8 input, so the transform reproduces PyWavelets' arithmetic to the
//! last bit: on x86_64 its C convolution rounds every product and every sum separately (no fused
//! multiply-add; measured against exact rational arithmetic), and so does this one. The U plane
//! is produced from BGR on the fly with OpenCV's fixed-point `BGR2YUV`, one block-row at a time.

use rayon::prelude::*;
use raiw_cv::MatView;

use crate::error::DwtDctError;

/// PyWavelets' Haar filter tap, `0.7071067811865476`.
pub(crate) const HAAR: f64 = std::f64::consts::FRAC_1_SQRT_2;
const BLOCK: usize = 4;
/// Pixel rows (and columns) behind one 4x4 block of the approximation band.
const SPAN: usize = 2 * BLOCK;
/// Scale of the U channel; Y carries scale 0 and V is never read (`range(2)` upstream).
const SCALE: f64 = 36.0;
const MIN_AREA: usize = 256 * 256;

// OpenCV `RGB2YCrCb_i` fixed point for the YUV ordering (`yuv_shift = 14`).
const YUV_SHIFT: i32 = 14;
const R2Y: i32 = 4899;
const G2Y: i32 = 9617;
const B2Y: i32 = 1868;
const B2UI: i32 = 8061;
const UV_DELTA: i32 = 128 << YUV_SHIFT;
const ROUND: i32 = 1 << (YUV_SHIFT - 1);

/// Bits decoded for each requested watermark length, in request order without duplicates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedBits {
    entries: Vec<(usize, Vec<bool>)>,
}

impl DecodedBits {
    /// The bits decoded for `length`, when it was requested.
    pub fn get(&self, length: usize) -> Option<&[bool]> {
        self.entries.iter().find(|(len, _)| *len == length).map(|(_, bits)| bits.as_slice())
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &[bool])> {
        self.entries.iter().map(|(len, bits)| (*len, bits.as_slice()))
    }
}

/// Extract `wm_len` watermark bits from a BGR (or BGRA) image.
pub fn decode_dwt_dct(bgr: MatView<'_, u8>, wm_len: usize) -> Result<Vec<bool>, DwtDctError> {
    let mut decoded = decode_dwt_dct_lengths(bgr, &[wm_len])?;
    Ok(decoded.entries.pop().map(|(_, bits)| bits).unwrap_or_default())
}

/// Extract several watermark lengths with one transform and block scan.
pub fn decode_dwt_dct_lengths(bgr: MatView<'_, u8>, lengths: &[usize]) -> Result<DecodedBits, DwtDctError> {
    let (width, height) = bgr.size();
    if width * height * bgr.channels() == 0 || width.min(height) * width.max(height) < MIN_AREA {
        return Err(DwtDctError::TooSmall);
    }
    if lengths.is_empty() || lengths.contains(&0) {
        return Err(DwtDctError::InvalidLengths);
    }
    let blocks = block_bits(bgr)?;
    let mut unique: Vec<usize> = Vec::with_capacity(lengths.len());
    for &len in lengths {
        if !unique.contains(&len) {
            unique.push(len);
        }
    }
    let entries = unique.into_iter().map(|len| (len, vote(&blocks, len))).collect();
    Ok(DecodedBits { entries })
}

/// One bit per 4x4 block of the U plane's Haar approximation, in row-major block order.
///
/// This is `_DecodeMaxDct._plane_bits(trimmed, 1, 36)`; it skips the caller's area check.
pub fn block_bits(bgr: MatView<'_, u8>) -> Result<Vec<bool>, DwtDctError> {
    let channels = bgr.channels();
    if channels != 3 && channels != 4 {
        return Err(DwtDctError::Channels(channels));
    }
    let (trim_w, trim_h) = (bgr.width() / 4 * 4, bgr.height() / 4 * 4);
    if trim_w == 0 || trim_h == 0 {
        return Err(DwtDctError::EmptyPlane { width: trim_w, height: trim_h });
    }
    let (rows, cols) = (trim_h / SPAN, trim_w / SPAN);
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }
    let products = haar_products();
    let mut bits = vec![false; rows * cols];
    bits.par_chunks_mut(cols).enumerate().for_each_init(
        || BlockRow::new(cols),
        |scratch, (row, out)| scratch.decode(&bgr, &products, row, out),
    );
    Ok(bits)
}

/// `x * HAAR` for every uint8 `x`: the first pass's products, rounded exactly as pywt rounds them.
fn haar_products() -> [f64; 256] {
    std::array::from_fn(|value| value as f64 * HAAR)
}

#[inline]
fn u_value(pixel: &[u8]) -> u8 {
    let (b, g, r) = (i32::from(pixel[0]), i32::from(pixel[1]), i32::from(pixel[2]));
    let y = (b * B2Y + g * G2Y + r * R2Y + ROUND) >> YUV_SHIFT;
    (((b - y) * B2UI + UV_DELTA + ROUND) >> YUV_SHIFT).clamp(0, 255) as u8
}

/// Scratch for one block-row: the vertical pass over its eight pixel rows.
struct BlockRow {
    vertical: Vec<f64>,
}

impl BlockRow {
    fn new(cols: usize) -> Self {
        Self { vertical: vec![0.0; BLOCK * cols * SPAN] }
    }

    fn decode(&mut self, bgr: &MatView<'_, u8>, products: &[f64; 256], row: usize, out: &mut [bool]) {
        let channels = bgr.channels();
        let width = out.len() * SPAN;
        // Pass one, along columns: pywt sums `x[2k + 1] * h + x[2k] * h`, each term rounded.
        for (pair, line) in self.vertical.chunks_exact_mut(width).enumerate() {
            let top = bgr.row(row * SPAN + 2 * pair);
            let bottom = bgr.row(row * SPAN + 2 * pair + 1);
            for ((value, upper), lower) in
                line.iter_mut().zip(top.chunks_exact(channels)).zip(bottom.chunks_exact(channels))
            {
                *value = products[usize::from(u_value(lower))] + products[usize::from(u_value(upper))];
            }
        }
        // Pass two, along rows, then the 4x4 peak with the DC term ignored.
        for (col, bit) in out.iter_mut().enumerate() {
            let mut peak = 0.0_f64;
            for (i, line) in self.vertical.chunks_exact(width).enumerate() {
                let pairs = &line[col * SPAN..(col + 1) * SPAN];
                for (j, pair) in pairs.chunks_exact(2).enumerate() {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    let approx = pair[1] * HAAR + pair[0] * HAAR;
                    peak = peak.max(approx.abs());
                }
            }
            *bit = peak % SCALE > 0.5 * SCALE;
        }
    }
}

/// Majority vote per bit position: `sums * 255 > counts * 127` over blocks `k % len`.
fn vote(blocks: &[bool], len: usize) -> Vec<bool> {
    let mut ones = vec![0_u64; len];
    let mut counts = vec![0_u64; len];
    for (index, &bit) in blocks.iter().enumerate() {
        counts[index % len] += 1;
        ones[index % len] += u64::from(bit);
    }
    ones.iter().zip(&counts).map(|(&one, &count)| one * 255 > count * 127).collect()
}

#[cfg(test)]
mod tests {
    use raiw_cv::{ColorConversion, Mat, cvt_color};

    use super::*;

    #[test]
    fn haar_tap_is_pywt_constant() {
        assert_eq!(HAAR.to_bits(), 0.707_106_781_186_547_6_f64.to_bits());
    }

    #[test]
    fn fused_u_matches_cvt_color() {
        let mut state = 0x1234_5678_u32;
        let img = Mat::<u8>::from_fn(97, 61, 3, |_, _, _| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        });
        let yuv = cvt_color(&img, ColorConversion::BgrToYuv).expect("yuv");
        for y in 0..img.height() {
            for x in 0..img.width() {
                assert_eq!(u_value(img.pixel(x, y)), yuv.at(x, y, 1), "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn vote_is_exact_majority() {
        assert_eq!(vote(&[true, false, true, false], 2), vec![true, false]);
        assert_eq!(vote(&[true, false], 4), vec![true, false, false, false]);
        // A tie votes one: 0.5 * 255 > 127.
        assert_eq!(vote(&[true, true, false, false], 1), vec![true]);
        assert_eq!(vote(&[true, false, false], 1), vec![false]);
    }

    #[test]
    fn errors_follow_python_order() {
        let tiny = Mat::<u8>::zeros(255, 256, 3);
        assert!(matches!(decode_dwt_dct_lengths(tiny.view(), &[48]), Err(DwtDctError::TooSmall)));
        let ok = Mat::<u8>::zeros(256, 256, 3);
        assert!(matches!(decode_dwt_dct_lengths(ok.view(), &[]), Err(DwtDctError::InvalidLengths)));
        assert!(matches!(decode_dwt_dct_lengths(ok.view(), &[48, 0]), Err(DwtDctError::InvalidLengths)));
        let gray = Mat::<u8>::zeros(256, 256, 1);
        assert!(matches!(decode_dwt_dct_lengths(gray.view(), &[48]), Err(DwtDctError::Channels(1))));
        let row = Mat::<u8>::zeros(65_536, 1, 3);
        assert!(matches!(decode_dwt_dct_lengths(row.view(), &[48]), Err(DwtDctError::EmptyPlane { .. })));
    }

    #[test]
    fn lengths_are_deduplicated_in_order() {
        let img = Mat::<u8>::filled(256, 256, 3, 90);
        let decoded = decode_dwt_dct_lengths(img.view(), &[136, 48, 136]).expect("decode");
        assert_eq!(decoded.iter().map(|(len, _)| len).collect::<Vec<_>>(), vec![136, 48]);
        assert_eq!(decoded.get(48).map(<[bool]>::len), Some(48));
    }
}

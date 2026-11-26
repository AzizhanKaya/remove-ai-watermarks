//! Pillow's default JPEG path through libjpeg-turbo 3.x, bit-exact: `Image.save(..., "JPEG", quality=q)`
//! (baseline YCbCr 4:2:0, Annex K tables, `JDCT_ISLOW`) and `Image.open(...).convert("RGB")`
//! (`JDCT_ISLOW`, fancy upsampling). Entropy coding is lossless, so the round trip runs on the
//! quantized coefficients; [`encode`] adds the JFIF byte stream for callers that keep the file.

mod dct;
mod huffman;

use rayon::prelude::*;
use raiw_cv::{Mat, MatView};

use dct::Divisor;

const LUMA_TABLE: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51,
    87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];
const CHROMA_TABLE: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99,
];

/// `jpeg_set_quality(q, force_baseline=TRUE)` for one Annex K table.
fn scaled_table(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let quality = i64::from(quality.clamp(1, 100));
    let scale = if quality < 50 { 5000 / quality } else { 200 - quality * 2 };
    base.map(|value| ((i64::from(value) * scale + 50) / 100).clamp(1, 255) as u16)
}

const SCALEBITS: u32 = 16;
const ONE_HALF: i64 = 1 << (SCALEBITS - 1);

const fn fix(x: f64) -> i64 {
    (x * 65536.0 + 0.5) as i64
}

/// `rgb_ycc_convert` with libjpeg's tables folded into closed form.
#[inline]
fn rgb_to_ycc(r: u8, g: u8, b: u8) -> [u8; 3] {
    let (r, g, b) = (i64::from(r), i64::from(g), i64::from(b));
    let offset = (128 << SCALEBITS) + ONE_HALF - 1;
    let y = (fix(0.299) * r + fix(0.587) * g + fix(0.114) * b + ONE_HALF) >> SCALEBITS;
    let cb = (-fix(0.168_74) * r - fix(0.331_26) * g + fix(0.5) * b + offset) >> SCALEBITS;
    let cr = (fix(0.5) * r - fix(0.418_69) * g - fix(0.081_31) * b + offset) >> SCALEBITS;
    [y as u8, cb as u8, cr as u8]
}

/// `ycc_rgb_convert`.
#[inline]
fn ycc_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let (y, cb, cr) = (i64::from(y), i64::from(cb) - 128, i64::from(cr) - 128);
    let r = y + ((fix(1.402) * cr + ONE_HALF) >> SCALEBITS);
    let g = y + ((-fix(0.344_14) * cb + ONE_HALF + -fix(0.714_14) * cr) >> SCALEBITS);
    let b = y + ((fix(1.772) * cb + ONE_HALF) >> SCALEBITS);
    [r, g, b].map(|v| v.clamp(0, 255) as u8)
}

/// One component's padded sample plane and its quantized blocks.
#[derive(Clone, Debug)]
struct Component {
    blocks_wide: usize,
    blocks_high: usize,
    /// Real (downsampled) size inside the padded plane.
    width: usize,
    height: usize,
    table: usize,
    coefficients: Vec<[i32; 64]>,
}

/// A baseline 4:2:0 JPEG as quantized coefficients.
#[derive(Clone, Debug)]
pub(crate) struct Encoded {
    width: usize,
    height: usize,
    tables: [[u16; 64]; 2],
    components: [Component; 3],
}

fn quantize_plane(
    plane: &[u8],
    stride: usize,
    blocks_wide: usize,
    blocks_high: usize,
    divisors: &[Divisor; 64],
) -> Vec<[i32; 64]> {
    (0..blocks_wide * blocks_high)
        .into_par_iter()
        .map(|index| {
            let (by, bx) = (index / blocks_wide, index % blocks_wide);
            let mut block: [i64; 64] =
                std::array::from_fn(|k| i64::from(plane[(by * 8 + k / 8) * stride + bx * 8 + k % 8]) - 128);
            dct::forward(&mut block);
            std::array::from_fn(|k| divisors[k].quantize(block[k]))
        })
        .collect()
}

/// Color-convert, downsample and quantize an RGB image like libjpeg-turbo's compressor.
pub(crate) fn encode_coefficients(rgb: MatView<'_, u8>, quality: u8) -> Encoded {
    let (width, height) = rgb.size();
    let tables = [scaled_table(&LUMA_TABLE, quality), scaled_table(&CHROMA_TABLE, quality)];
    let divisors = tables.map(|table| table.map(|q| Divisor::new(u32::from(q) << 3)));
    let mut planes = [vec![0_u8; width * height], vec![0_u8; width * height], vec![0_u8; width * height]];
    for y in 0..height {
        for (x, px) in rgb.row(y).chunks_exact(3).enumerate() {
            let ycc = rgb_to_ycc(px[0], px[1], px[2]);
            for (plane, value) in planes.iter_mut().zip(ycc) {
                plane[y * width + x] = value;
            }
        }
    }
    let at = |plane: &[u8], x: usize, y: usize| plane[y.min(height - 1) * width + x.min(width - 1)];

    // Luma: edge replication to whole blocks.
    let (luma_w, luma_h) = (width.div_ceil(8), height.div_ceil(8));
    let luma: Vec<u8> = (0..luma_h * 8 * luma_w * 8).map(|i| at(&planes[0], i % (luma_w * 8), i / (luma_w * 8))).collect();
    let luma = Component {
        blocks_wide: luma_w,
        blocks_high: luma_h,
        width,
        height,
        table: 0,
        coefficients: quantize_plane(&luma, luma_w * 8, luma_w, luma_h, &divisors[0]),
    };

    // Chroma: `h2v2_downsample` with the alternating 1, 2 bias, then bottom replication.
    let (chroma_w, chroma_h) = (width.div_ceil(16), height.div_ceil(16));
    let (real_w, real_h) = (width.div_ceil(2), height.div_ceil(2));
    let chroma = |plane: &[u8]| {
        let stride = chroma_w * 8;
        let mut out = vec![0_u8; stride * chroma_h * 8];
        for r in 0..chroma_h * 8 {
            let src = r.min(real_h - 1);
            for c in 0..stride {
                let bias = if c % 2 == 0 { 1 } else { 2 };
                let sum = u32::from(at(plane, 2 * c, 2 * src))
                    + u32::from(at(plane, 2 * c + 1, 2 * src))
                    + u32::from(at(plane, 2 * c, 2 * src + 1))
                    + u32::from(at(plane, 2 * c + 1, 2 * src + 1));
                out[r * stride + c] = ((sum + bias) >> 2) as u8;
            }
        }
        Component {
            blocks_wide: chroma_w,
            blocks_high: chroma_h,
            width: real_w,
            height: real_h,
            table: 1,
            coefficients: quantize_plane(&out, stride, chroma_w, chroma_h, &divisors[1]),
        }
    };
    let (cb, cr) = (chroma(&planes[1]), chroma(&planes[2]));
    Encoded { width, height, tables, components: [luma, cb, cr] }
}

/// Inverse-transform one component into its padded plane.
fn reconstruct(component: &Component, table: &[u16; 64]) -> Vec<u8> {
    let stride = component.blocks_wide * 8;
    let mut plane = vec![0_u8; stride * component.blocks_high * 8];
    plane.par_chunks_mut(stride * 8).enumerate().for_each(|(by, band)| {
        let mut out = [0_u8; 64];
        for bx in 0..component.blocks_wide {
            dct::inverse(&component.coefficients[by * component.blocks_wide + bx], table, &mut out);
            for k in 0..8 {
                band[k * stride + bx * 8..k * stride + bx * 8 + 8].copy_from_slice(&out[k * 8..k * 8 + 8]);
            }
        }
    });
    plane
}

/// One upsampled chroma row (`h2v2_fancy_upsample`, or replication below three samples).
fn upsample_row(plane: &[u8], stride: usize, component: &Component, y: usize, out: &mut [u8]) {
    let (w, h) = (component.width, component.height);
    let near = y / 2;
    let row = |r: usize| &plane[r * stride..r * stride + w];
    if w <= 2 {
        for (x, value) in out.iter_mut().enumerate() {
            *value = row(near)[(x / 2).min(w - 1)];
        }
        return;
    }
    let far = if y % 2 == 0 { near.saturating_sub(1) } else { (near + 1).min(h - 1) };
    let (a, b) = (row(near), row(far));
    let sum = |c: usize| i32::from(a[c]) * 3 + i32::from(b[c]);
    let mut put = |x: usize, v: i32| {
        if x < out.len() {
            out[x] = (v >> 4) as u8;
        }
    };
    put(0, sum(0) * 4 + 8);
    put(1, sum(0) * 3 + sum(1) + 7);
    for c in 1..w - 1 {
        put(2 * c, sum(c) * 3 + sum(c - 1) + 8);
        put(2 * c + 1, sum(c) * 3 + sum(c + 1) + 7);
    }
    put(2 * (w - 1), sum(w - 1) * 3 + sum(w - 2) + 8);
    put(2 * w - 1, sum(w - 1) * 4 + 7);
}

/// Decompress like libjpeg-turbo's defaults and return RGB.
pub(crate) fn decode(encoded: &Encoded) -> Mat<u8> {
    let [luma, cb, cr] = &encoded.components;
    let planes = [luma, cb, cr].map(|c| reconstruct(c, &encoded.tables[c.table]));
    let (width, height) = (encoded.width, encoded.height);
    let luma_stride = luma.blocks_wide * 8;
    let chroma_stride = cb.blocks_wide * 8;
    let mut data = vec![0_u8; width * height * 3];
    data.par_chunks_mut(width * 3).enumerate().for_each(|(y, out)| {
        let (mut u, mut v) = (vec![0_u8; width], vec![0_u8; width]);
        upsample_row(&planes[1], chroma_stride, cb, y, &mut u);
        upsample_row(&planes[2], chroma_stride, cr, y, &mut v);
        for (x, px) in out.chunks_exact_mut(3).enumerate() {
            px.copy_from_slice(&ycc_to_rgb(planes[0][y * luma_stride + x], u[x], v[x]));
        }
    });
    Mat::from_vec(width, height, 3, data).unwrap_or_else(|_| Mat::zeros(width, height, 3))
}

/// `Image.save(buf, "JPEG", quality=q)` then `Image.open(buf).convert("RGB")` for an RGB image.
pub fn round_trip(rgb: MatView<'_, u8>, quality: u8) -> Mat<u8> {
    if rgb.is_empty() {
        return rgb.to_mat();
    }
    decode(&encode_coefficients(rgb, quality))
}

/// The JFIF file Pillow writes for `Image.save(buf, "JPEG", quality=q)` of an RGB image.
pub fn encode(rgb: MatView<'_, u8>, quality: u8) -> Vec<u8> {
    huffman::write_jfif(&encode_coefficients(rgb, quality))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_scaling_matches_libjpeg() {
        assert_eq!(scaled_table(&LUMA_TABLE, 95)[0], 2);
        assert_eq!(scaled_table(&LUMA_TABLE, 90)[0], 3);
        assert_eq!(scaled_table(&LUMA_TABLE, 50), LUMA_TABLE);
        assert_eq!(scaled_table(&CHROMA_TABLE, 10)[63], 255);
    }

    #[test]
    fn gray_pixels_stay_gray() {
        assert_eq!(rgb_to_ycc(128, 128, 128), [128, 128, 128]);
        assert_eq!(ycc_to_rgb(128, 128, 128), [128, 128, 128]);
        assert_eq!(rgb_to_ycc(255, 255, 255)[0], 255);
    }
}

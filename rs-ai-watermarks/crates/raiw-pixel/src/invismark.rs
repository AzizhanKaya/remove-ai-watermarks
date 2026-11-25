//! Local Microsoft Paint InvisMark payload (`microsoft_invismark.py`).
//!
//! A clean-room reader of the observed `Watermarker.dll` format: the top singular value of each
//! 4x4 luminance-LL block's AC DCT coefficients carries one bit of a 144-bit message tiled on a
//! 12x12 grid. Luminance, the LL plane and the block DCT reproduce numpy/OpenBLAS arithmetic
//! (fused multiply-add chains); the singular value comes from one-sided Jacobi instead of LAPACK.

use rayon::prelude::*;
use raiw_cv::{Element, Mat, MatView};

use crate::error::InvisMarkError;
use crate::math::dct4;
use crate::uuid::Uuid;

const BLOCK: usize = 4;
const MAP: usize = 12;
const PAYLOAD_BITS: usize = MAP * MAP;
const PAYLOAD_BYTES: usize = PAYLOAD_BITS / 8;
const QUANTUM: f64 = 24.0;
const CENTERS: [f64; 2] = [6.0, 18.0];
const MAX_CENTER_ERROR: f64 = 2.0;
const MIN_SUPPORT: u32 = 3;
const MAGIC: u8 = 0x4C;
/// The soft-binding algorithm a C2PA manifest uses for this mark.
pub const INVISMARK_ALGORITHM: &str = "com.microsoft.invismark.1";

/// A validated local Paint payload recovered from repeated pixel carriers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalInvisMarkDetection {
    pub watermark_id: Uuid,
    pub payload: [u8; PAYLOAD_BYTES],
    pub minimum_support: u32,
}

/// A disrupted payload: the rewritten RGB pixels and how many carriers changed.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalInvisMarkRemoval {
    pub detection: LocalInvisMarkDetection,
    pub changed_blocks: usize,
    pub rgb: Mat<u8>,
}

/// Per-block measurements over the luminance LL plane, in row-major block order.
#[derive(Clone, Debug)]
pub struct InvisMarkAnalysis {
    ll_width: usize,
    ll_height: usize,
    block_rows: usize,
    block_cols: usize,
    ll: Vec<f64>,
    sigma: Vec<f64>,
    bits: Vec<u8>,
    eligible: Vec<bool>,
}

impl InvisMarkAnalysis {
    /// `(rows, columns)` of the LL plane.
    pub const fn ll_shape(&self) -> (usize, usize) {
        (self.ll_height, self.ll_width)
    }

    pub fn ll(&self) -> &[f64] {
        &self.ll
    }

    /// `(rows, columns)` of the block grid.
    pub const fn blocks(&self) -> (usize, usize) {
        (self.block_rows, self.block_cols)
    }

    /// Top singular value of each block's 3x5 AC matrix.
    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    pub fn bits(&self) -> &[u8] {
        &self.bits
    }

    pub fn eligible(&self) -> &[bool] {
        &self.eligible
    }

    /// Payload bit index carried by block `index`.
    pub const fn bit_index(&self, index: usize) -> usize {
        (index / self.block_cols % MAP) * MAP + index % self.block_cols % MAP
    }

    /// Eligible votes per payload bit: `support[bit][value]`.
    pub fn support(&self) -> Vec<[u32; 2]> {
        let mut support = vec![[0_u32; 2]; PAYLOAD_BITS];
        for (index, (&bit, _)) in self.bits.iter().zip(&self.eligible).enumerate().filter(|(_, (_, e))| **e) {
            support[self.bit_index(index)][usize::from(bit)] += 1;
        }
        support
    }
}

/// Run the block analysis on an RGB image (any element type numpy would widen to float64).
pub fn analyze<T: Element + Into<f64>>(rgb: MatView<'_, T>) -> Result<InvisMarkAnalysis, InvisMarkError> {
    if rgb.channels() != 3 {
        return Err(InvisMarkError::Channels(rgb.channels()));
    }
    let (ll_width, ll_height) = (rgb.width() / 2, rgb.height() / 2);
    let (block_rows, block_cols) = (ll_height / BLOCK, ll_width / BLOCK);
    if block_rows == 0 || block_cols == 0 {
        return Err(InvisMarkError::NoBlocks { width: rgb.width(), height: rgb.height() });
    }
    let ll = ll_plane(&rgb, ll_width, ll_height);
    let transform = dct4();
    let measured: Vec<(f64, u8, bool)> = (0..block_rows * block_cols)
        .into_par_iter()
        .map(|index| {
            let coefficients = block_dct(&ll, ll_width, index / block_cols, index % block_cols, &transform);
            let sigma = top_singular(&ac_matrix(&coefficients)).0;
            let (bit, error) = classify(sigma);
            (sigma, bit, error <= MAX_CENTER_ERROR)
        })
        .collect();
    let (sigma, rest): (Vec<f64>, Vec<(u8, bool)>) = measured.into_iter().map(|(s, b, e)| (s, (b, e))).unzip();
    let (bits, eligible) = rest.into_iter().unzip();
    Ok(InvisMarkAnalysis { ll_width, ll_height, block_rows, block_cols, ll, sigma, bits, eligible })
}

/// Return a validated local Paint payload, or `None` for an uncertain image.
pub fn detect_local_invismark<T: Element + Into<f64>>(
    rgb: MatView<'_, T>,
) -> Result<Option<LocalInvisMarkDetection>, InvisMarkError> {
    Ok(validate(&analyze(rgb)?))
}

/// The detection the analysis supports, or `None` when any bit is weak, tied or the frame is invalid.
pub fn validate(analysis: &InvisMarkAnalysis) -> Option<LocalInvisMarkDetection> {
    let support = analysis.support();
    let winning: Vec<u32> = support.iter().map(|s| s[0].max(s[1])).collect();
    if winning.iter().any(|&w| w < MIN_SUPPORT) || support.iter().any(|s| s[0] == s[1]) {
        return None;
    }
    let mut payload = [0_u8; PAYLOAD_BYTES];
    for (i, s) in support.iter().enumerate() {
        if s[1] > s[0] {
            payload[i / 8] |= 0x80 >> (i % 8);
        }
    }
    let checksum = payload[1..17].iter().fold(0_u8, |acc, &b| acc.wrapping_add(b));
    if payload[0] != MAGIC || payload[17] != checksum {
        return None;
    }
    let mut le = [0_u8; 16];
    le.copy_from_slice(&payload[1..17]);
    let watermark_id = Uuid::from_bytes_le(le);
    if watermark_id.is_nil() {
        return None;
    }
    Some(LocalInvisMarkDetection { watermark_id, payload, minimum_support: winning.into_iter().min().unwrap_or(0) })
}

/// Invert confirmed local Paint carriers.
///
/// Returns `None` unless the payload validates, an optional expected UUID (parsed like Python's
/// `uuid.UUID(str(...))`) agrees, and the rewritten pixels no longer carry a valid payload. The
/// caller writes the pixels and should re-check a lossy output with [`detect_local_invismark`].
pub fn disrupt_local_invismark(
    rgb: MatView<'_, u8>,
    expected_watermark_id: Option<&str>,
) -> Result<Option<LocalInvisMarkRemoval>, InvisMarkError> {
    let analysis = analyze(rgb)?;
    let Some(detection) = validate(&analysis) else {
        return Ok(None);
    };
    if let Some(expected) = expected_watermark_id
        && Uuid::parse_python(expected) != Some(detection.watermark_id)
    {
        return Ok(None);
    }
    let payload_bit = |i: usize| detection.payload[i / 8] >> (7 - i % 8) & 1;
    let selected: Vec<usize> = (0..analysis.bits.len())
        .filter(|&i| analysis.eligible[i] && analysis.bits[i] == payload_bit(analysis.bit_index(i)))
        .collect();
    let transform = dct4();
    let rewritten: Vec<(usize, [f64; 16])> = selected
        .par_iter()
        .map(|&index| {
            let (row, col) = (index / analysis.block_cols, index % analysis.block_cols);
            let block = rewrite_block(&analysis.ll, analysis.ll_width, row, col, 1 - analysis.bits[index], &transform);
            (index, block)
        })
        .collect();
    let mut delta = vec![0.0_f64; analysis.ll.len()];
    for (index, block) in &rewritten {
        let (row, col) = (index / analysis.block_cols, index % analysis.block_cols);
        for (k, value) in block.iter().enumerate() {
            let at = (row * BLOCK + k / BLOCK) * analysis.ll_width + col * BLOCK + k % BLOCK;
            delta[at] = (value - analysis.ll[at]) * 0.5;
        }
    }
    let cleaned = Mat::from_fn(rgb.width(), rgb.height(), 3, |x, y, c| {
        let (lx, ly) = (x / 2, y / 2);
        let d = if lx < analysis.ll_width && ly < analysis.ll_height { delta[ly * analysis.ll_width + lx] } else { 0.0 };
        (f64::from(rgb.at(x, y, c)) + d).round_ties_even().clamp(0.0, 255.0) as u8
    });
    if detect_local_invismark(cleaned.view())?.is_some() {
        return Ok(None);
    }
    Ok(Some(LocalInvisMarkRemoval { detection, changed_blocks: selected.len(), rgb: cleaned }))
}

/// The one signed UUID of a `com.microsoft.invismark.1` C2PA soft binding, if exactly one is valid.
pub fn invismark_id_from_c2pa_info(algorithm: Option<&str>, value: Option<&str>) -> Option<Uuid> {
    let listed = |text: Option<&str>| -> Vec<String> {
        text.unwrap_or("").split(',').map(|v| v.trim().to_owned()).filter(|v| !v.is_empty()).collect()
    };
    if !listed(algorithm).iter().any(|a| a == INVISMARK_ALGORITHM) {
        return None;
    }
    let mut unique: Vec<Uuid> = Vec::new();
    for parsed in listed(value).iter().filter_map(|v| Uuid::parse_python(v)) {
        if !unique.contains(&parsed) {
            unique.push(parsed);
        }
    }
    (unique.len() == 1).then(|| unique[0])
}

// Luminance `rgb @ (0.299, 0.587, 0.114)` as OpenBLAS dgemv evaluates it, then the LL sum.
fn ll_plane<T: Element + Into<f64>>(rgb: &MatView<'_, T>, width: usize, height: usize) -> Vec<f64> {
    let luminance = |px: &[T]| {
        let (r, g, b) = (px[0].into(), px[1].into(), px[2].into());
        b.mul_add(0.114, r.mul_add(0.299, g * 0.587))
    };
    let mut ll = vec![0.0; width * height];
    ll.par_chunks_mut(width.max(1)).enumerate().for_each(|(y, out)| {
        let (top, bottom) = (rgb.row(2 * y), rgb.row(2 * y + 1));
        for (x, value) in out.iter_mut().enumerate() {
            let at = |row: &[T], dx: usize| luminance(&row[(2 * x + dx) * 3..(2 * x + dx) * 3 + 3]);
            *value = (((at(top, 0) + at(top, 1)) + at(bottom, 0)) + at(bottom, 1)) * 0.5;
        }
    });
    ll
}

/// One dot product of a 4x4 OpenBLAS dgemm: `fma(a3, b3, fma(a2, b2, fma(a1, b1, a0 * b0)))`.
#[inline]
fn gemm_dot(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[3].mul_add(b[3], a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0])))
}

fn matmul4(a: &[f64; 16], b: &[f64; 16], b_transposed: bool) -> [f64; 16] {
    std::array::from_fn(|k| {
        let (i, j) = (k / 4, k % 4);
        let row = [a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]];
        let col = if b_transposed {
            [b[j * 4], b[j * 4 + 1], b[j * 4 + 2], b[j * 4 + 3]]
        } else {
            [b[j], b[4 + j], b[8 + j], b[12 + j]]
        };
        gemm_dot(row, col)
    })
}

fn transpose4(m: &[f64; 16]) -> [f64; 16] {
    std::array::from_fn(|k| m[(k % 4) * 4 + k / 4])
}

/// `transform @ block @ transform.T` for block `(row, col)`.
fn block_dct(ll: &[f64], width: usize, row: usize, col: usize, transform: &[f64; 16]) -> [f64; 16] {
    let block: [f64; 16] = std::array::from_fn(|k| ll[(row * BLOCK + k / 4) * width + col * BLOCK + k % 4]);
    matmul4(&matmul4(transform, &block, false), transform, true)
}

fn ac_matrix(coefficients: &[f64; 16]) -> [[f64; 5]; 3] {
    std::array::from_fn(|r| std::array::from_fn(|c| coefficients[1 + r * 5 + c]))
}

/// Nearest-center bit and its circular distance (`argmin` keeps the first center on ties).
fn classify(sigma: f64) -> (u8, f64) {
    let residue = sigma % QUANTUM;
    let errors = CENTERS.map(|center| {
        let distance = (residue - center).abs();
        distance.min(QUANTUM - distance)
    });
    if errors[1] < errors[0] { (1, errors[1]) } else { (0, errors[0]) }
}

/// Top singular triplet `(sigma, u, v)` of a 3x5 matrix by one-sided Jacobi on its rows.
fn top_singular(m: &[[f64; 5]; 3]) -> (f64, [f64; 3], [f64; 5]) {
    let mut rows = *m;
    let mut basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let dot = |a: &[f64; 5], b: &[f64; 5]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    for _ in 0..60 {
        let mut rotated = false;
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let (alpha, beta, gamma) = (dot(&rows[p], &rows[p]), dot(&rows[q], &rows[q]), dot(&rows[p], &rows[q]));
            if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                continue;
            }
            rotated = true;
            let zeta = (beta - alpha) / (2.0 * gamma);
            let t = zeta.signum() / (zeta.abs() + zeta.mul_add(zeta, 1.0).sqrt());
            let c = 1.0 / t.mul_add(t, 1.0).sqrt();
            let s = c * t;
            for k in 0..5 {
                let (x, y) = (rows[p][k], rows[q][k]);
                rows[p][k] = c * x - s * y;
                rows[q][k] = s * x + c * y;
            }
            for basis_row in &mut basis {
                let (x, y) = (basis_row[p], basis_row[q]);
                basis_row[p] = c * x - s * y;
                basis_row[q] = s * x + c * y;
            }
        }
        if !rotated {
            break;
        }
    }
    let norms = rows.map(|r| dot(&r, &r).sqrt());
    let top = (0..3).fold(0, |best, i| if norms[i] > norms[best] { i } else { best });
    let sigma = norms[top];
    let v = if sigma > 0.0 { rows[top].map(|x| x / sigma) } else { [0.0; 5] };
    (sigma, [basis[0][top], basis[1][top], basis[2][top]], v)
}

/// The block with its top singular value moved to the center of `target_bit`, back in LL space.
fn rewrite_block(ll: &[f64], width: usize, row: usize, col: usize, target_bit: u8, transform: &[f64; 16]) -> [f64; 16] {
    let mut coefficients = block_dct(ll, width, row, col, transform);
    let (sigma, u, v) = top_singular(&ac_matrix(&coefficients));
    let target = ((sigma / QUANTUM).floor() + 0.25 + 0.5 * f64::from(target_bit)) * QUANTUM;
    let shift = target - sigma;
    for r in 0..3 {
        for c in 0..5 {
            coefficients[1 + r * 5 + c] += shift * u[r] * v[c];
        }
    }
    matmul4(&matmul4(&transpose4(transform), &coefficients, false), transform, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_recovers_a_rank_one_matrix() {
        let u = [0.6, 0.8, 0.0];
        let v = [0.0, 0.6, 0.0, 0.8, 0.0];
        let m: [[f64; 5]; 3] = std::array::from_fn(|r| std::array::from_fn(|c| 30.0 * u[r] * v[c]));
        let (sigma, uu, vv) = top_singular(&m);
        assert!((sigma - 30.0).abs() < 1e-12);
        let sign = uu[0].signum() * vv[1].signum();
        assert!(sign > 0.0 && (uu[1] / uu[0] - 4.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn classify_ties_take_the_first_center() {
        assert_eq!(classify(12.0), (0, 6.0));
        assert_eq!(classify(18.5), (1, 0.5));
        assert_eq!(classify(24.0 * 3.0 + 6.0).0, 0);
    }

    #[test]
    fn c2pa_requires_exact_algorithm_and_one_value() {
        let id = "83424621-03cb-40e3-9808-a9fae837156d";
        assert!(invismark_id_from_c2pa_info(Some(INVISMARK_ALGORITHM), Some(id)).is_some());
        assert!(invismark_id_from_c2pa_info(Some("io.iscc.v0"), Some(id)).is_none());
        assert!(invismark_id_from_c2pa_info(None, Some(id)).is_none());
    }
}

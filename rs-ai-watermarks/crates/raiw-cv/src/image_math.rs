//! Port of `_internal/image_math.py`.

use crate::mat::Mat;

/// `dct_matrix(np, size)`: the orthonormal `size x size` DCT-II basis (row = frequency).
pub fn dct_matrix(size: usize) -> Mat<f64> {
    let n = size as f64;
    let first = 1.0 / 2f64.sqrt();
    let norm = (2.0 / n).sqrt();
    Mat::from_fn(size, size, 1, |sample, freq, _| {
        let mut v = (std::f64::consts::PI * (2 * sample + 1) as f64 * freq as f64 / (2 * size) as f64).cos();
        if freq == 0 {
            v *= first;
        }
        v * norm
    })
}

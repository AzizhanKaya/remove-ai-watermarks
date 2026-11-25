//! Small shared transforms (`_internal/image_math.py`).

/// Orthonormal `size` x `size` DCT-II basis, row-major, evaluated in numpy's order.
pub fn dct_basis(size: usize) -> Vec<f64> {
    let n = size as f64;
    let first_row = 1.0 / 2.0_f64.sqrt();
    let norm = (2.0 / n).sqrt();
    (0..size * size)
        .map(|k| {
            let (frequency, sample) = ((k / size) as f64, (k % size) as f64);
            let mut value = (std::f64::consts::PI * (2.0 * sample + 1.0) * frequency / (2.0 * n)).cos();
            if k < size {
                value *= first_row;
            }
            value * norm
        })
        .collect()
}

/// The 4x4 basis as a fixed array.
pub fn dct4() -> [f64; 16] {
    let basis = dct_basis(4);
    std::array::from_fn(|k| basis[k])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_is_orthonormal() {
        for size in [4, 8, 32] {
            let b = dct_basis(size);
            for i in 0..size {
                for j in 0..size {
                    let dot: f64 = (0..size).map(|k| b[i * size + k] * b[j * size + k]).sum();
                    assert!((dot - f64::from(u8::from(i == j))).abs() < 1e-12);
                }
            }
        }
    }
}

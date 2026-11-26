//! NumPy reduction orders that decide the last bits of SciPy-derived coefficients and gains.

const PAIRWISE_BLOCK: usize = 128;
const BUFFER_SIZE: usize = 8192;

/// `np.sum` of a contiguous float64 array: pairwise sums of 8192-element buffers.
pub fn sum(values: &[f64]) -> f64 {
    values
        .chunks(BUFFER_SIZE)
        .fold(0.0, |acc, chunk| acc + pairwise(chunk))
}

/// `np.mean` of a contiguous float64 array.
pub fn mean(values: &[f64]) -> f64 {
    sum(values) / values.len() as f64
}

/// `float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))` without materializing the squares.
pub fn rms_f32(values: &[f32]) -> f64 {
    let mut total = 0.0;
    let mut buffer = vec![0.0f64; BUFFER_SIZE.min(values.len())];
    for chunk in values.chunks(BUFFER_SIZE) {
        for (slot, &v) in buffer.iter_mut().zip(chunk) {
            let v = f64::from(v);
            *slot = v * v;
        }
        total += pairwise(&buffer[..chunk.len()]);
    }
    (total / values.len() as f64).sqrt()
}

/// Row mean of a C-contiguous `(n, channels)` float32 matrix (`x.mean(axis=1)`).
pub fn row_mean_f32(row: &[f32]) -> f32 {
    pairwise(row) / row.len() as f32
}

/// NumPy's float32 `np.cos` SIMD kernel (FMA targets): Cody-Waite reduction plus minimax polynomials.
pub fn cos_f32(x: f32) -> f32 {
    let c = |bits: u32| f32::from_bits(bits);
    if x.is_nan() {
        return f32::NAN;
    }
    if x.abs() > 13_176_794.0 {
        return x.cos();
    }
    let quadrant = (x * c(0x3F22_F983) + c(0x4B40_0000)) - c(0x4B40_0000);
    let mut r = quadrant.mul_add(c(0xBFC9_0FD8), x);
    r = quadrant.mul_add(c(0xB4A8_885A), r);
    r = quadrant.mul_add(c(0xA7C2_34C4), r);
    let r2 = r * r;
    let cos = c(0x37CC_730B)
        .mul_add(r2, c(0xBAB6_036E))
        .mul_add(r2, c(0x3D2A_AA9E))
        .mul_add(r2, -0.5)
        .mul_add(r2, 1.0);
    let sin = (c(0x363E_9DDE)
        .mul_add(r2, c(0xB950_35DD))
        .mul_add(r2, c(0x3C08_88CD))
        .mul_add(r2, c(0xBE2A_AAAB))
        * r2)
        .mul_add(r, r);
    let iquadrant = quadrant as i32 + 1;
    let value = if iquadrant & 1 == 0 { sin } else { cos };
    if iquadrant & 2 == 2 { 0.0 - value } else { value }
}

fn pairwise<T>(a: &[T]) -> T
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::AddAssign,
{
    let n = a.len();
    if n < 8 {
        let mut res = T::default();
        for &v in a {
            res += v;
        }
        res
    } else if n <= PAIRWISE_BLOCK {
        let mut r = [T::default(); 8];
        r.copy_from_slice(&a[..8]);
        let whole = n - n % 8;
        let mut i = 8;
        while i < whole {
            for (j, slot) in r.iter_mut().enumerate() {
                *slot += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[whole..] {
            res += v;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise(&a[..half]) + pairwise(&a[half..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_sums_are_sequential() {
        assert_eq!(sum(&[1e16, 1.0, 1.0]), 1e16);
    }
}

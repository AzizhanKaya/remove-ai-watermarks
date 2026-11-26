//! Kernel construction: `getGaussianKernel` (bit-exact variant), its 8-bit fixed-point form, Sobel kernels.

use crate::element::cv_round;
use crate::error::{Result, invalid};

/// OpenCV's automatic Gaussian aperture: `cvRound(sigma * (u8 ? 3 : 4) * 2 + 1) | 1`.
pub fn gaussian_ksize(sigma: f64, is_u8: bool) -> usize {
    (cv_round(sigma * if is_u8 { 3.0 } else { 4.0 } * 2.0 + 1.0) | 1).max(1) as usize
}

/// `getGaussianKernelBitExact`: normalized 1D Gaussian taps (double precision).
pub fn gaussian_kernel(n: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        let fixed: &[f64] = match n {
            1 => &[1.0],
            3 => &[0.25, 0.5, 0.25],
            5 => &[0.0625, 0.25, 0.375, 0.25, 0.0625],
            7 => &[
                0.031_25, 0.109_375, 0.218_75, 0.281_25, 0.218_75, 0.109_375, 0.031_25,
            ],
            9 => &[
                4.0 / 256.0,
                13.0 / 256.0,
                30.0 / 256.0,
                51.0 / 256.0,
                60.0 / 256.0,
                51.0 / 256.0,
                30.0 / 256.0,
                13.0 / 256.0,
                4.0 / 256.0,
            ],
            _ => &[],
        };
        if !fixed.is_empty() {
            return fixed.to_vec();
        }
    }
    let sigma_x = if sigma > 0.0 {
        sigma
    } else {
        (n as f64).mul_add(0.15, 0.35)
    };
    let scale2x = -0.125 / (sigma_x * sigma_x);
    let half = (n - 1) / 2;
    let mut values = Vec::with_capacity(half);
    let mut sum = 0.0;
    let mut x = 1 - n as i64;
    for _ in 0..half {
        let t = ((x * x) as f64 * scale2x).exp();
        values.push(t);
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    sum += 1.0;
    if n.is_multiple_of(2) {
        sum += 1.0;
    }
    let mul = 1.0 / sum;
    let mut out = vec![0.0; n];
    for (i, v) in values.iter().enumerate() {
        out[i] = v * mul;
        out[n - 1 - i] = v * mul;
    }
    out[half] = mul;
    if n.is_multiple_of(2) {
        out[half + 1] = mul;
    }
    out
}

/// `getGaussianKernelFixedPoint_ED` with `bits` fractional bits (8 for `ufixedpoint16`, 16 for `ufixedpoint32`).
pub(crate) fn gaussian_kernel_fixed(n: usize, sigma: f64, bits: u32) -> Vec<u64> {
    let k = gaussian_kernel(n, sigma);
    let one = 1i64 << bits;
    let mult = one as f64;
    let half = n / 2;
    let mut out = vec![0u64; n];
    let mut err = 0.0;
    let mut sum = 0i64;
    for i in 0..half {
        let adj = k[i] * mult + err;
        let v = i64::from(cv_round(adj));
        err = adj - v as f64;
        out[i] = v as u64;
        out[n - 1 - i] = v as u64;
        sum += v;
    }
    out[half] = (one - 2 * sum) as u64;
    out
}

/// `getDerivKernels` for Sobel apertures (`ksize` 1, 3, 5, ..., 31): `(kx, ky)`.
pub fn sobel_kernels(dx: usize, dy: usize, ksize: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if ksize.is_multiple_of(2) || ksize > 31 || dx + dy == 0 {
        return Err(invalid(
            "sobel",
            "kernel size must be odd, at most 31, with a nonzero order",
        ));
    }
    let one = |order: usize| -> Result<Vec<f64>> {
        let ksize = if ksize == 1 && order > 0 { 3 } else { ksize };
        if ksize <= order {
            return Err(invalid("sobel", "derivative order exceeds the aperture"));
        }
        let ker: Vec<i64> = match (ksize, order) {
            (1, _) => vec![1],
            (3, 0) => vec![1, 2, 1],
            (3, 1) => vec![-1, 0, 1],
            (3, _) => vec![1, -2, 1],
            _ => {
                let mut k = vec![0i64; ksize + 1];
                k[0] = 1;
                for _ in 0..ksize - order - 1 {
                    let mut old = k[0];
                    for j in 1..=ksize {
                        let new = k[j] + k[j - 1];
                        k[j - 1] = old;
                        old = new;
                    }
                }
                for _ in 0..order {
                    let mut old = -k[0];
                    for j in 1..=ksize {
                        let new = k[j - 1] - k[j];
                        k[j - 1] = old;
                        old = new;
                    }
                }
                k.truncate(ksize);
                k
            }
        };
        Ok(ker.into_iter().map(|v| v as f64).collect())
    };
    Ok((one(dx)?, one(dy)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_kernels() {
        assert_eq!(gaussian_kernel_fixed(3, 0.0, 8), [64, 128, 64]);
        assert_eq!(
            sobel_kernels(1, 0, 3).unwrap(),
            (vec![-1.0, 0.0, 1.0], vec![1.0, 2.0, 1.0])
        );
        assert_eq!(sobel_kernels(0, 1, 5).unwrap().1, vec![-1.0, -2.0, 0.0, 2.0, 1.0]);
        assert_eq!(gaussian_ksize(2.0, true), 13);
        assert_eq!(gaussian_ksize(2.0, false), 17);
    }
}

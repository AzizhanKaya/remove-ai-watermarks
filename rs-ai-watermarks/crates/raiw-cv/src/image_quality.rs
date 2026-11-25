//! Port of `_internal/image_quality.py`.

use crate::border::Border;
use crate::element::Element;
use crate::error::{CvError, Result};
use crate::filter::gaussian_blur;
use crate::mat::{Mat, MatView};

/// `structural_similarity_gray(a, b)`: single-scale Gaussian SSIM (11x11, sigma 1.5) in float64.
pub fn structural_similarity_gray<'a, 'b, T: Element>(a: impl Into<MatView<'a, T>>, b: impl Into<MatView<'b, T>>) -> Result<f64> {
    let (a, b) = (a.into(), b.into());
    if a.size() != b.size() || a.channels() != b.channels() {
        return Err(CvError::ShapeMismatch { op: "structural_similarity_gray" });
    }
    let a = a.to_mat().convert::<f64>();
    let b = b.to_mat().convert::<f64>();
    let (c1, c2) = ((0.01f64 * 255.0).powi(2), (0.03f64 * 255.0).powi(2));
    let blur = |m: &Mat<f64>| gaussian_blur(m, (11, 11), 1.5, 0.0, Border::Reflect101);
    let product = |x: &Mat<f64>, y: &Mat<f64>| -> Mat<f64> {
        let data = x.data().iter().zip(y.data()).map(|(p, q)| p * q).collect();
        Mat::from_vec(x.width(), x.height(), x.channels(), data).unwrap_or_else(|_| Mat::zeros(0, 0, 1))
    };
    let mu_a = blur(&a)?;
    let mu_b = blur(&b)?;
    let saa = blur(&product(&a, &a))?;
    let sbb = blur(&product(&b, &b))?;
    let sab = blur(&product(&a, &b))?;
    let n = mu_a.data().len();
    let mut sum = 0.0;
    for i in 0..n {
        let (ma, mb) = (mu_a.data()[i], mu_b.data()[i]);
        let (ma2, mb2, mab) = (ma * ma, mb * mb, ma * mb);
        let (sa, sb, sx) = (saa.data()[i] - ma2, sbb.data()[i] - mb2, sab.data()[i] - mab);
        sum += ((2.0 * mab + c1) * (2.0 * sx + c2)) / ((ma2 + mb2 + c1) * (sa + sb + c2));
    }
    Ok(sum / n as f64)
}

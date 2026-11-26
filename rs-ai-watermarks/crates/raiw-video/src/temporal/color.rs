//! Ultra-low-frequency CIE Lab color anchoring (`anchor_low_freq_color`).

use raiw_cv::{Border, ColorConversion, Interpolation, Mat, cvt_color, gaussian_blur, resize};

use super::pixel::VideoPixel;
use crate::error::Result;

/// `_ultra_low_freq_blur`: a Gaussian through a quarter-scale pyramid.
fn ultra_low_freq_blur(image: &Mat<f32>, sigma: f64) -> Result<Mat<f32>> {
    let (width, height) = image.size();
    let (small_w, small_h) = ((width / 4).max(8), (height / 4).max(8));
    let small = resize(image, small_w, small_h, Interpolation::Area)?;
    let blurred = gaussian_blur(&small, (0, 0), (sigma / 4.0).max(1.0), 0.0, Border::Reflect101)?;
    Ok(resize(&blurred, width, height, Interpolation::Linear)?)
}

/// `anchor_low_freq_color`: transfer only frequencies below `sigma` of Lab illumination and
/// chroma from `source` onto `candidate`.
pub fn anchor_low_freq_color<T: VideoPixel>(
    source: &Mat<T>,
    candidate: &Mat<T>,
    sigma: Option<f64>,
) -> Result<Mat<T>> {
    let (width, height) = candidate.size();
    let sigma = sigma.map_or_else(|| (width.max(height) as f64 / 32.0).max(32.0), |s| s.max(8.0));
    let aligned = if source.size() == (width, height) {
        source.clone()
    } else {
        resize(source, width, height, Interpolation::Area)?
    };
    let scale = T::PEAK as f32;
    let unit = |m: &Mat<T>| m.map(|v| (v.to_f32() / scale).clamp(0.0, 1.0));
    let (src, cand) = (unit(&aligned), unit(candidate));
    let low_src = cvt_color(&ultra_low_freq_blur(&src, sigma)?, ColorConversion::BgrToLab)?;
    let low_cand = cvt_color(&ultra_low_freq_blur(&cand, sigma)?, ColorConversion::BgrToLab)?;
    let mut lab = cvt_color(&cand, ColorConversion::BgrToLab)?;
    for ((value, s), c) in lab
        .data_mut()
        .chunks_exact_mut(3)
        .zip(low_src.data().chunks_exact(3))
        .zip(low_cand.data().chunks_exact(3))
    {
        value[0] = (value[0] + (s[0] - c[0])).clamp(0.0, 100.0);
        value[1] = (value[1] + (s[1] - c[1])).clamp(-127.0, 127.0);
        value[2] = (value[2] + (s[2] - c[2])).clamp(-127.0, 127.0);
    }
    let bgr = cvt_color(&lab, ColorConversion::LabToBgr)?;
    Ok(
        bgr.map(|v| {
            T::from_f64(f64::from((v.clamp(0.0, 1.0) * scale).round_ties_even()).clamp(0.0, T::PEAK))
        }),
    )
}

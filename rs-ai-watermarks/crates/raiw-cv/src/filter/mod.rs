//! Linear filters: `GaussianBlur`, `Sobel`, `Laplacian`, `filter2D`, `sepFilter2D`.

mod engine;
mod kernels;

use crate::border::Border;
use crate::element::{Depth, Element};
use crate::error::{Result, invalid};
use crate::mat::{Mat, MatView};

use engine::{FixedGauss, FloatSep, run_filter2d, run_separable};
pub use kernels::{gaussian_kernel, gaussian_ksize, sobel_kernels};

fn wide<S: Element, D: Element>() -> bool {
    S::DEPTH == Depth::F64 || D::DEPTH == Depth::F64
}

fn separable<S: Element, D: Element>(
    src: &MatView<'_, S>,
    kx: &[f64],
    ky: &[f64],
    delta: f64,
    border: Border,
) -> Mat<D> {
    if wide::<S, D>() {
        run_separable(src, &FloatSep::<S, D, f64>::new(kx, ky, delta), border)
    } else {
        run_separable(src, &FloatSep::<S, D, f32>::new(kx, ky, delta), border)
    }
}

/// `cv::GaussianBlur(src, (kw, kh), sigma_x, sigma_y, border)`; a zero size derives from sigma.
pub fn gaussian_blur<'a, T: Blur>(
    src: impl Into<MatView<'a, T>>,
    ksize: (usize, usize),
    sigma_x: f64,
    sigma_y: f64,
    border: Border,
) -> Result<Mat<T>> {
    let src = src.into();
    if src.is_empty() {
        return Err(invalid("gaussian_blur", "empty input"));
    }
    let (mut kw, mut kh) = ksize;
    let sigma_y = if sigma_y <= 0.0 { sigma_x } else { sigma_y };
    let is_u8 = T::DEPTH == Depth::U8;
    if kw == 0 && sigma_x > 0.0 {
        kw = gaussian_ksize(sigma_x, is_u8);
    }
    if kh == 0 && sigma_y > 0.0 {
        kh = gaussian_ksize(sigma_y, is_u8);
    }
    if kw.is_multiple_of(2) || kh.is_multiple_of(2) {
        return Err(invalid("gaussian_blur", "kernel size must be positive and odd"));
    }
    if !matches!(border, Border::Constant(_)) {
        if src.height() == 1 {
            kh = 1;
        }
        if src.width() == 1 {
            kw = 1;
        }
    }
    if kw == 1 && kh == 1 {
        return Ok(src.to_mat());
    }
    Ok(T::blur(
        &src,
        (kw, kh),
        (sigma_x.max(0.0), sigma_y.max(0.0)),
        border,
    ))
}

/// Element types `gaussian_blur` accepts: 8/16-bit run OpenCV's bit-exact fixed-point path.
pub trait Blur: Element {
    #[doc(hidden)]
    fn blur(src: &MatView<'_, Self>, ksize: (usize, usize), sigma: (f64, f64), border: Border) -> Mat<Self>;
}

impl Blur for u8 {
    fn blur(src: &MatView<'_, u8>, ksize: (usize, usize), sigma: (f64, f64), border: Border) -> Mat<u8> {
        let ops = FixedGauss::<u8>::new(ksize, sigma);
        run_separable(src, &ops, border)
    }
}

impl Blur for u16 {
    fn blur(src: &MatView<'_, u16>, ksize: (usize, usize), sigma: (f64, f64), border: Border) -> Mat<u16> {
        let ops = FixedGauss::<u16>::new(ksize, sigma);
        run_separable(src, &ops, border)
    }
}

macro_rules! float_blur {
    ($t:ty) => {
        impl Blur for $t {
            fn blur(
                src: &MatView<'_, $t>,
                ksize: (usize, usize),
                sigma: (f64, f64),
                border: Border,
            ) -> Mat<$t> {
                separable::<$t, $t>(
                    src,
                    &gaussian_kernel(ksize.0, sigma.0),
                    &gaussian_kernel(ksize.1, sigma.1),
                    0.0,
                    border,
                )
            }
        }
    };
}

float_blur!(f32);
float_blur!(f64);

/// `cv::Sobel(src, ddepth=D, dx, dy, ksize, scale, delta, border)`.
#[allow(clippy::too_many_arguments)]
pub fn sobel<'a, S: Element, D: Element>(
    src: impl Into<MatView<'a, S>>,
    dx: usize,
    dy: usize,
    ksize: usize,
    scale: f64,
    delta: f64,
    border: Border,
) -> Result<Mat<D>> {
    let src = src.into();
    if src.is_empty() {
        return Err(invalid("sobel", "empty input"));
    }
    let (mut kx, mut ky) = sobel_kernels(dx, dy, ksize)?;
    if scale != 1.0 {
        let k = if dx == 0 { &mut kx } else { &mut ky };
        for v in k.iter_mut() {
            *v *= scale;
        }
    }
    Ok(separable::<S, D>(&src, &kx, &ky, delta, border))
}

/// `cv::Laplacian(src, ddepth=D, ksize, scale, delta, border)` for apertures 1 and 3.
pub fn laplacian<'a, S: Element, D: Element>(
    src: impl Into<MatView<'a, S>>,
    ksize: usize,
    scale: f64,
    delta: f64,
    border: Border,
) -> Result<Mat<D>> {
    let src = src.into();
    if src.is_empty() {
        return Err(invalid("laplacian", "empty input"));
    }
    let base: [f64; 9] = match ksize {
        1 => [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
        3 => [2.0, 0.0, 2.0, 0.0, -8.0, 0.0, 2.0, 0.0, 2.0],
        _ => return Err(invalid("laplacian", "only apertures 1 and 3 are supported")),
    };
    let kernel = base.map(|v| f64::from((v * scale) as f32));
    Ok(filter_2d_impl::<S, D>(&src, &kernel, 3, 3, (1, 1), delta, border))
}

fn filter_2d_impl<S: Element, D: Element>(
    src: &MatView<'_, S>,
    kernel: &[f64],
    kw: usize,
    kh: usize,
    anchor: (usize, usize),
    delta: f64,
    border: Border,
) -> Mat<D> {
    if wide::<S, D>() {
        run_filter2d::<S, D, f64>(src, kernel, kw, kh, anchor, delta, border)
    } else {
        run_filter2d::<S, D, f32>(src, kernel, kw, kh, anchor, delta, border)
    }
}

/// `cv::filter2D(src, ddepth=D, kernel (row-major kw x kh), anchor=center, delta, border)`.
pub fn filter_2d<'a, S: Element, D: Element>(
    src: impl Into<MatView<'a, S>>,
    kernel: &[f64],
    kw: usize,
    kh: usize,
    delta: f64,
    border: Border,
) -> Result<Mat<D>> {
    let src = src.into();
    if src.is_empty() || kernel.len() != kw * kh || kw == 0 || kh == 0 {
        return Err(invalid("filter_2d", "empty input or kernel size mismatch"));
    }
    Ok(filter_2d_impl::<S, D>(
        &src,
        kernel,
        kw,
        kh,
        (kw / 2, kh / 2),
        delta,
        border,
    ))
}

/// `cv::sepFilter2D(src, ddepth=D, kx, ky, anchor=center, delta, border)`.
pub fn sep_filter_2d<'a, S: Element, D: Element>(
    src: impl Into<MatView<'a, S>>,
    kx: &[f64],
    ky: &[f64],
    delta: f64,
    border: Border,
) -> Result<Mat<D>> {
    let src = src.into();
    if src.is_empty() || kx.is_empty() || ky.is_empty() {
        return Err(invalid("sep_filter_2d", "empty input or kernel"));
    }
    Ok(separable::<S, D>(&src, kx, ky, delta, border))
}

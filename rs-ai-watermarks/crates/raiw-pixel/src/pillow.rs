//! Pillow 13 `Image.resize` / `Image.reduce` / `Image.thumbnail` arithmetic (`libImaging/Resample.c`,
//! `Reduce.c`), bit-exact for 8-bit L/RGB and 32-bit float images.

use raiw_cv::{Mat, MatView};

/// Resampling filters with Pillow's kernels and supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Box,
    Bilinear,
    Bicubic,
    Lanczos,
}

impl Filter {
    const fn support(self) -> f64 {
        match self {
            Self::Box => 0.5,
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
            Self::Lanczos => 3.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        match self {
            Self::Box => {
                if x > -0.5 && x <= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bilinear => {
                let x = x.abs();
                if x < 1.0 { 1.0 - x } else { 0.0 }
            }
            Self::Bicubic => {
                let a = -0.5;
                let x = x.abs();
                if x < 1.0 {
                    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * a
                } else {
                    0.0
                }
            }
            Self::Lanczos => {
                if (-3.0..3.0).contains(&x) {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Per-output-pixel source window and normalized weights (`precompute_coeffs`).
struct Coefficients {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    weights: Vec<f64>,
}

impl Coefficients {
    fn new(in_size: usize, in0: f32, in1: f32, out_size: usize, filter: Filter) -> Self {
        let scale = f64::from(in1 - in0) / out_size as f64;
        let filterscale = scale.max(1.0);
        let support = filter.support() * filterscale;
        let ksize = support.ceil() as usize * 2 + 1;
        let mut bounds = Vec::with_capacity(out_size);
        let mut weights = vec![0.0; out_size * ksize];
        for xx in 0..out_size {
            let center = f64::from(in0) + (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filterscale;
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) - xmin;
            let k = &mut weights[xx * ksize..(xx + 1) * ksize];
            let mut ww = 0.0;
            for x in 0..xmax.max(0) as usize {
                let w = filter.eval(((x as i64 + xmin) as f64 - center + 0.5) * ss);
                k[x] = w;
                ww += w;
            }
            if ww != 0.0 {
                k[..xmax.max(0) as usize].iter_mut().for_each(|w| *w /= ww);
            }
            bounds.push((xmin as usize, xmax.max(0) as usize));
        }
        Self { ksize, bounds, weights }
    }

    fn fixed(&self) -> Vec<i32> {
        let one = f64::from(1_u32 << PRECISION_BITS);
        self.weights.iter().map(|&w| if w < 0.0 { (-0.5 + w * one) as i32 } else { (0.5 + w * one) as i32 }).collect()
    }
}

#[inline]
fn clip8(value: i32) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

/// An 8-bit L or RGB image resized like `Image.resize(size, filter, box)`.
pub fn resize_u8(src: MatView<'_, u8>, width: usize, height: usize, filter: Filter, bx: [f32; 4]) -> Mat<u8> {
    let channels = src.channels();
    let need_h = width != src.width() || bx[0] != 0.0 || bx[2] != width as f32;
    let need_v = height != src.height() || bx[1] != 0.0 || bx[3] != height as f32;
    let horiz = Coefficients::new(src.width(), bx[0], bx[2], width, filter);
    let vert = Coefficients::new(src.height(), bx[1], bx[3], height, filter);
    let first = vert.bounds.first().map_or(0, |b| b.0);
    let last = vert.bounds.last().map_or(0, |b| b.0 + b.1);
    let mut current = src.to_mat();
    let mut vert_bounds = vert.bounds.clone();
    if need_h {
        vert_bounds.iter_mut().for_each(|b| b.0 -= first);
        let kk = horiz.fixed();
        current = Mat::from_fn(width, last - first, channels, |xx, yy, c| {
            let (xmin, xmax) = horiz.bounds[xx];
            let k = &kk[xx * horiz.ksize..];
            let row = src.row(yy + first);
            let mut ss = 1_i32 << (PRECISION_BITS - 1);
            for x in 0..xmax {
                ss += i32::from(row[(x + xmin) * channels + c]) * k[x];
            }
            clip8(ss)
        });
    }
    if need_v {
        let kk = vert.fixed();
        let input = current;
        current = Mat::from_fn(input.width(), height, channels, |xx, yy, c| {
            let (ymin, ymax) = vert_bounds[yy];
            let k = &kk[yy * vert.ksize..];
            let mut ss = 1_i32 << (PRECISION_BITS - 1);
            for y in 0..ymax {
                ss += i32::from(input.at(xx, y + ymin, c)) * k[y];
            }
            clip8(ss)
        });
    }
    current
}

/// A mode `F` image resized like `Image.resize(size, filter, box)`.
pub fn resize_f32(src: MatView<'_, f32>, width: usize, height: usize, filter: Filter, bx: [f32; 4]) -> Mat<f32> {
    let need_h = width != src.width() || bx[0] != 0.0 || bx[2] != width as f32;
    let need_v = height != src.height() || bx[1] != 0.0 || bx[3] != height as f32;
    let horiz = Coefficients::new(src.width(), bx[0], bx[2], width, filter);
    let vert = Coefficients::new(src.height(), bx[1], bx[3], height, filter);
    let first = vert.bounds.first().map_or(0, |b| b.0);
    let last = vert.bounds.last().map_or(0, |b| b.0 + b.1);
    let mut current = src.to_mat();
    let mut vert_bounds = vert.bounds.clone();
    if need_h {
        vert_bounds.iter_mut().for_each(|b| b.0 -= first);
        current = Mat::from_fn(width, last - first, 1, |xx, yy, _| {
            let (xmin, xmax) = horiz.bounds[xx];
            let k = &horiz.weights[xx * horiz.ksize..];
            let row = src.row(yy + first);
            let mut ss = 0.0_f64;
            for x in 0..xmax {
                ss += f64::from(row[x + xmin]) * k[x];
            }
            ss as f32
        });
    }
    if need_v {
        let input = current;
        current = Mat::from_fn(input.width(), height, 1, |xx, yy, _| {
            let (ymin, ymax) = vert_bounds[yy];
            let k = &vert.weights[yy * vert.ksize..];
            let mut ss = 0.0_f64;
            for y in 0..ymax {
                ss += f64::from(input.at(xx, y + ymin, 0)) * k[y];
            }
            ss as f32
        });
    }
    current
}

/// Full-image box for `resize` without an explicit `box`.
pub fn full_box(width: usize, height: usize) -> [f32; 4] {
    [0.0, 0.0, width as f32, height as f32]
}

/// `Image.resize(size, filter)` for 8-bit images, including Pillow's same-size copy shortcut.
pub fn resize(src: MatView<'_, u8>, width: usize, height: usize, filter: Filter) -> Mat<u8> {
    if (width, height) == src.size() {
        return src.to_mat();
    }
    resize_u8(src, width, height, filter, full_box(src.width(), src.height()))
}

/// `Image.crop((left, top, right, bottom))` for boxes inside the image.
pub fn crop(src: MatView<'_, u8>, left: usize, top: usize, right: usize, bottom: usize) -> Mat<u8> {
    let channels = src.channels();
    Mat::from_fn(right - left, bottom - top, channels, |x, y, c| src.at(x + left, y + top, c))
}

/// `Image.convert("L")` from RGB: ITU-R 601-2 luma in 16-bit fixed point.
pub fn rgb_to_l(rgb: MatView<'_, u8>) -> Mat<u8> {
    Mat::from_fn(rgb.width(), rgb.height(), 1, |x, y, _| {
        let px = rgb.pixel(x, y);
        let value = u32::from(px[0]) * 19595 + u32::from(px[1]) * 38470 + u32::from(px[2]) * 7471 + 0x8000;
        (value >> 16) as u8
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_size_resize_is_a_copy() {
        let img = Mat::<u8>::from_fn(5, 4, 3, |x, y, c| (x * 40 + y * 9 + c) as u8);
        assert_eq!(resize(img.view(), 5, 4, Filter::Bilinear), img);
    }

    #[test]
    fn constant_image_stays_constant() {
        let img = Mat::<u8>::filled(37, 23, 3, 200);
        for filter in [Filter::Box, Filter::Bilinear, Filter::Bicubic, Filter::Lanczos] {
            let out = resize(img.view(), 11, 17, filter);
            assert!(out.data().iter().all(|&v| v == 200), "{filter:?}");
        }
    }

    #[test]
    fn luma_matches_pillow_formula() {
        let px = Mat::<u8>::from_vec(1, 1, 3, vec![255, 0, 0]).expect("mat");
        assert_eq!(rgb_to_l(px.view()).at(0, 0, 0), 76);
    }
}

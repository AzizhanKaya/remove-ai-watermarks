//! `cv::Canny(src, low, high)` with the 3x3 Sobel aperture and the L1 gradient norm.

use crate::border::Border;
use crate::element::cv_floor;
use crate::error::{CvError, Result};
use crate::filter::sobel;
use crate::mat::{Mat, MatView};

const TG22: i32 = 13_573;

/// Edge map (0 or 255) of an 8-bit single-channel image.
pub fn canny<'a>(src: impl Into<MatView<'a, u8>>, low: f64, high: f64) -> Result<Mat<u8>> {
    let src = src.into();
    if src.channels() != 1 {
        return Err(CvError::Channels { op: "canny", channels: src.channels() });
    }
    let (low, high) = if low > high { (high, low) } else { (low, high) };
    let (low, high) = (cv_floor(low), cv_floor(high));
    let dx: Mat<i16> = sobel(&src, 1, 0, 3, 1.0, 0.0, Border::Replicate)?;
    let dy: Mat<i16> = sobel(&src, 0, 1, 3, 1.0, 0.0, Border::Replicate)?;
    let (w, h) = (src.width(), src.height());
    let mag = |y: isize, x: isize| -> i32 {
        if y < 0 || y >= h as isize || x < 0 || x >= w as isize {
            return 0;
        }
        let (x, y) = (x as usize, y as usize);
        i32::from(dx.at(x, y, 0)).abs() + i32::from(dy.at(x, y, 0)).abs()
    };
    let stride = w + 2;
    let mut map = vec![1u8; stride * (h + 2)];
    let mut stack = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (yi, xi) = (y as isize, x as isize);
            let m = mag(yi, xi);
            let idx = (y + 1) * stride + x + 1;
            if m <= low {
                continue;
            }
            let xs = dx.at(x, y, 0);
            let ys = dy.at(x, y, 0);
            let ax = i32::from(xs).abs();
            let ay = i32::from(ys).abs() << 15;
            let tg22x = ax * TG22;
            let keep = if ay < tg22x {
                m > mag(yi, xi - 1) && m >= mag(yi, xi + 1)
            } else if ay > tg22x + (ax << 16) {
                m > mag(yi - 1, xi) && m >= mag(yi + 1, xi)
            } else {
                let s: isize = if (xs ^ ys) < 0 { -1 } else { 1 };
                m > mag(yi - 1, xi - s) && m > mag(yi + 1, xi + s)
            };
            if keep {
                if m > high {
                    map[idx] = 2;
                    stack.push(idx);
                } else {
                    map[idx] = 0;
                }
            }
        }
    }
    let neighbours = [stride + 1, stride, stride - 1, 1];
    while let Some(i) = stack.pop() {
        for &d in &neighbours {
            for j in [i - d, i + d] {
                if map[j] == 0 {
                    map[j] = 2;
                    stack.push(j);
                }
            }
        }
    }
    let mut out = Mat::zeros(w, h, 1);
    for y in 0..h {
        let row = out.row_mut(y);
        for (x, o) in row.iter_mut().enumerate() {
            if map[(y + 1) * stride + x + 1] == 2 {
                *o = 255;
            }
        }
    }
    Ok(out)
}

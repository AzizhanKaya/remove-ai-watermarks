//! Border extrapolation modes (`cv::BorderTypes`) and `copyMakeBorder`.

use crate::element::Element;
use crate::error::{Result, invalid};
use crate::mat::{Mat, MatView};
use crate::types::Scalar;

/// How pixels outside the image are synthesized.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Border {
    /// `iiiiii|abcdefgh|iiiiiii` with the given per-channel value.
    Constant(Scalar),
    /// `aaaaaa|abcdefgh|hhhhhhh`
    Replicate,
    /// `fedcba|abcdefgh|hgfedcb`
    Reflect,
    /// `cdefgh|abcdefgh|abcdefg`
    Wrap,
    /// `gfedcb|abcdefgh|gfedcba` (`BORDER_DEFAULT`)
    #[default]
    Reflect101,
}

impl Border {
    /// `BORDER_CONSTANT` with zeros.
    pub const ZERO: Self = Self::Constant(Scalar::all(0.0));

    pub fn constant(value: impl Into<Scalar>) -> Self {
        Self::Constant(value.into())
    }

    /// `cv::borderInterpolate`: the source index for `p`, or `None` for a constant border.
    #[inline]
    pub fn interpolate(self, p: isize, len: usize) -> Option<usize> {
        let n = len as isize;
        if (0..n).contains(&p) {
            return Some(p as usize);
        }
        match self {
            Self::Constant(_) => None,
            Self::Replicate => Some(if p < 0 { 0 } else { len - 1 }),
            Self::Reflect | Self::Reflect101 => {
                if len == 1 {
                    return Some(0);
                }
                let delta = isize::from(self == Self::Reflect101);
                let mut p = p;
                while !(0..n).contains(&p) {
                    p = if p < 0 {
                        -p - 1 + delta
                    } else {
                        n - 1 - (p - n) - delta
                    };
                }
                Some(p as usize)
            }
            Self::Wrap => Some(p.rem_euclid(n) as usize),
        }
    }

    pub(crate) fn constant_value(self, channel: usize) -> f64 {
        match self {
            Self::Constant(s) => s.get(channel),
            _ => 0.0,
        }
    }
}

/// Precomputed source indices for a border of `before`/`after` pixels around `len` samples.
pub(crate) fn border_table(len: usize, before: usize, after: usize, border: Border) -> Vec<Option<usize>> {
    (0..before + len + after)
        .map(|i| border.interpolate(i as isize - before as isize, len))
        .collect()
}

/// `cv::copyMakeBorder`.
pub fn copy_make_border<'a, T: Element>(
    src: impl Into<MatView<'a, T>>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    border: Border,
) -> Result<Mat<T>> {
    let src = src.into();
    if src.is_empty() {
        return Err(invalid("copy_make_border", "empty input"));
    }
    let (w, h, cn) = (src.width(), src.height(), src.channels());
    let fill: Vec<T> = (0..cn).map(|c| T::from_f64(border.constant_value(c))).collect();
    let xs = border_table(w, left, right, border);
    let ys = border_table(h, top, bottom, border);
    let mut out = Mat::zeros(w + left + right, h + top + bottom, cn);
    for (y, sy) in ys.iter().enumerate() {
        let row = out.row_mut(y);
        let Some(sy) = *sy else {
            row.chunks_exact_mut(cn).for_each(|px| px.copy_from_slice(&fill));
            continue;
        };
        let line = src.row(sy);
        row[left * cn..(left + w) * cn].copy_from_slice(line);
        for x in (0..left).chain(left + w..left + w + right) {
            let px = &mut row[x * cn..(x + 1) * cn];
            match xs[x] {
                Some(sx) => px.copy_from_slice(&line[sx * cn..(sx + 1) * cn]),
                None => px.copy_from_slice(&fill),
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_matches_opencv_examples() {
        let idx = |b: Border| (-3..11).map(|p| b.interpolate(p, 8).unwrap()).collect::<Vec<_>>();
        assert_eq!(
            idx(Border::Reflect101),
            [3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4]
        );
        assert_eq!(idx(Border::Reflect), [2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5]);
        assert_eq!(idx(Border::Replicate), [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7]);
        assert_eq!(idx(Border::Wrap), [5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]);
        assert_eq!(Border::ZERO.interpolate(-1, 8), None);
    }
}

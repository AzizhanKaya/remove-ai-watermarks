//! OpenCV operations raiw-cv does not provide yet, in its signature style.

use raiw_cv::{Mat, MatView};

use crate::geometry::Region;

/// `cv::boundingRect` of the nonzero pixels of a single-channel mask; `(0, 0, 0, 0)` when empty.
pub fn bounding_rect<'a>(mask: impl Into<MatView<'a, u8>>) -> Region {
    let mask = mask.into();
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..mask.height() {
        let row = mask.row(y);
        let Some(first) = row.iter().position(|&v| v != 0) else { continue };
        let last = row.iter().rposition(|&v| v != 0).unwrap_or(first);
        x0 = x0.min(first);
        x1 = x1.max(last);
        y0 = y0.min(y);
        y1 = y;
    }
    if x0 == usize::MAX {
        return Region::ZERO;
    }
    Region::new(x0 as i32, y0 as i32, (x1 - x0 + 1) as i32, (y1 - y0 + 1) as i32)
}

/// Whether any element of a mask is nonzero (`mask.any()`).
pub fn any_nonzero(mask: &Mat<u8>) -> bool {
    mask.data().iter().any(|&v| v != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounding_rect_of_sparse_mask() {
        let mut m = Mat::<u8>::zeros(10, 8, 1);
        m.set(2, 3, 0, 255);
        m.set(6, 5, 0, 1);
        assert_eq!(bounding_rect(&m), Region::new(2, 3, 5, 3));
        assert_eq!(bounding_rect(&Mat::<u8>::zeros(4, 4, 1)), Region::ZERO);
    }
}

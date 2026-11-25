//! Boxes in absolute and ROI-local pixel coordinates.

use serde::{Deserialize, Serialize};

/// An `(x, y, w, h)` box, serialized as Python's 4-tuple.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "[i32; 4]", into = "[i32; 4]")]
pub struct Region {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Region {
    pub const ZERO: Self = Self::new(0, 0, 0, 0);

    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }

    /// Whether two positive-area regions overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.w.min(self.h).min(other.w).min(other.h) <= 0 {
            return false;
        }
        self.x < other.x + other.w
            && other.x < self.x + self.w
            && self.y < other.y + other.h
            && other.y < self.y + self.h
    }
}

impl From<[i32; 4]> for Region {
    fn from([x, y, w, h]: [i32; 4]) -> Self {
        Self { x, y, w, h }
    }
}

impl From<Region> for [i32; 4] {
    fn from(r: Region) -> Self {
        [r.x, r.y, r.w, r.h]
    }
}

/// An inclusive `(x0, y0, x1, y1)` template match box in a locate box's coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "[i32; 4]", into = "[i32; 4]")]
pub struct MatchBox {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl MatchBox {
    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// The box of a `w x h` template whose top-left corner is `(x, y)`.
    pub const fn at(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self::new(x, y, x + w - 1, y + h - 1)
    }

    pub const fn width(&self) -> i32 {
        self.x1 - self.x0 + 1
    }

    pub const fn height(&self) -> i32 {
        self.y1 - self.y0 + 1
    }
}

impl From<[i32; 4]> for MatchBox {
    fn from([x0, y0, x1, y1]: [i32; 4]) -> Self {
        Self { x0, y0, x1, y1 }
    }
}

impl From<MatchBox> for [i32; 4] {
    fn from(b: MatchBox) -> Self {
        [b.x0, b.y0, b.x1, b.y1]
    }
}

/// A half-open `[x0, x1) x [y0, y1)` rectangle in absolute coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Span {
    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// As an `(x, y, w, h)` region (possibly empty).
    pub const fn region(&self) -> Region {
        Region::new(self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_requires_positive_area() {
        let a = Region::new(2, 2, 4, 4);
        assert!(a.overlaps(&Region::new(3, 3, 2, 2)));
        assert!(!a.overlaps(&Region::new(8, 8, 2, 2)));
        assert!(!a.overlaps(&Region::new(6, 2, 2, 2)));
        assert!(!a.overlaps(&Region::new(3, 3, 0, 2)));
    }

    #[test]
    fn serializes_as_python_tuples() {
        assert_eq!(serde_json::to_string(&Region::new(1, 2, 3, 4)).ok(), Some("[1,2,3,4]".to_owned()));
        assert_eq!(MatchBox::at(5, 6, 10, 4), MatchBox::new(5, 6, 14, 9));
    }
}

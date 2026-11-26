//! Small geometric value types mirroring OpenCV's `Rect`, `Point`, `Size` and `Scalar`.

/// An axis-aligned rectangle with a top-left corner and a size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    pub const fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self { x, y, width, height }
    }

    /// The rectangle spanning `[x0, x1) x [y0, y1)`, like a numpy slice `[y0:y1, x0:x1]`.
    pub fn from_corners(x0: usize, y0: usize, x1: usize, y1: usize) -> Self {
        Self {
            x: x0,
            y: y0,
            width: x1.saturating_sub(x0),
            height: y1.saturating_sub(y0),
        }
    }

    pub const fn right(&self) -> usize {
        self.x + self.width
    }

    pub const fn bottom(&self) -> usize {
        self.y + self.height
    }

    pub const fn area(&self) -> usize {
        self.width * self.height
    }

    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// An integer pixel coordinate (`x` is the column).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// A sub-pixel coordinate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point2f {
    pub x: f32,
    pub y: f32,
}

impl Point2f {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// A double-precision coordinate or displacement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

impl Point2d {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Up to four per-channel values. `Scalar::from(v)` is `(v, 0, 0, 0)` exactly like OpenCV.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scalar(pub [f64; 4]);

impl Scalar {
    pub const fn all(v: f64) -> Self {
        Self([v; 4])
    }

    pub const fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self([a, b, c, d])
    }

    pub const fn get(&self, channel: usize) -> f64 {
        if channel < 4 { self.0[channel] } else { 0.0 }
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Self([v, 0.0, 0.0, 0.0])
    }
}

impl From<[f64; 4]> for Scalar {
    fn from(v: [f64; 4]) -> Self {
        Self(v)
    }
}

/// Resampling kernels (`cv::InterpolationFlags`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Interpolation {
    Nearest,
    #[default]
    Linear,
    Cubic,
    Area,
    Lanczos4,
}

/// A rotated rectangle: center, size and angle in degrees.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RotatedRect {
    pub center: Point2f,
    pub width: f32,
    pub height: f32,
    pub angle: f32,
}

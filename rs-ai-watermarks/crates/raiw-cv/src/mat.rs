//! The core image type: row-major, interleaved channels, plus borrowed ROI views.

use std::fmt;

use rayon::prelude::*;

use crate::element::Element;
use crate::error::{CvError, Result};
use crate::types::Rect;

/// An owned image of `height` rows, each `width * channels` interleaved elements.
#[derive(Clone, PartialEq)]
pub struct Mat<T> {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<T>,
}

impl<T> fmt::Debug for Mat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mat<{}>({}x{}x{})",
            std::any::type_name::<T>(),
            self.width,
            self.height,
            self.channels
        )
    }
}

impl<T: Element> Mat<T> {
    /// A zero-filled image.
    pub fn zeros(width: usize, height: usize, channels: usize) -> Self {
        Self::filled(width, height, channels, T::ZERO)
    }

    pub fn filled(width: usize, height: usize, channels: usize, value: T) -> Self {
        Self {
            width,
            height,
            channels,
            data: vec![value; width * height * channels],
        }
    }

    /// Wrap an existing buffer without copying.
    pub fn from_vec(width: usize, height: usize, channels: usize, data: Vec<T>) -> Result<Self> {
        if data.len() != width * height * channels || channels == 0 {
            return Err(CvError::BufferSize {
                len: data.len(),
                width,
                height,
                channels,
            });
        }
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// Build an image from a per-element function `f(x, y, channel)`.
    pub fn from_fn(
        width: usize,
        height: usize,
        channels: usize,
        mut f: impl FnMut(usize, usize, usize) -> T,
    ) -> Self {
        let mut data = Vec::with_capacity(width * height * channels);
        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    data.push(f(x, y, c));
                }
            }
        }
        Self {
            width,
            height,
            channels,
            data,
        }
    }

    /// Copy a borrowed view into a contiguous owned image.
    pub fn from_view(view: &MatView<'_, T>) -> Self {
        let mut data = Vec::with_capacity(view.width * view.height * view.channels);
        for y in 0..view.height {
            data.extend_from_slice(view.row(y));
        }
        Self {
            width: view.width,
            height: view.height,
            channels: view.channels,
            data,
        }
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// `(width, height)`.
    pub const fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub const fn row_len(&self) -> usize {
        self.width * self.channels
    }

    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    pub fn row(&self, y: usize) -> &[T] {
        let len = self.row_len();
        &self.data[y * len..(y + 1) * len]
    }

    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        let len = self.row_len();
        &mut self.data[y * len..(y + 1) * len]
    }

    /// The channels of pixel `(x, y)`.
    pub fn pixel(&self, x: usize, y: usize) -> &[T] {
        let start = (y * self.width + x) * self.channels;
        &self.data[start..start + self.channels]
    }

    pub fn pixel_mut(&mut self, x: usize, y: usize) -> &mut [T] {
        let start = (y * self.width + x) * self.channels;
        &mut self.data[start..start + self.channels]
    }

    pub fn at(&self, x: usize, y: usize, channel: usize) -> T {
        self.data[(y * self.width + x) * self.channels + channel]
    }

    pub fn set(&mut self, x: usize, y: usize, channel: usize, value: T) {
        self.data[(y * self.width + x) * self.channels + channel] = value;
    }

    pub fn view(&self) -> MatView<'_, T> {
        MatView {
            data: &self.data,
            width: self.width,
            height: self.height,
            channels: self.channels,
            stride: self.row_len(),
        }
    }

    pub fn view_mut(&mut self) -> MatViewMut<'_, T> {
        let stride = self.row_len();
        MatViewMut {
            data: &mut self.data,
            width: self.width,
            height: self.height,
            channels: self.channels,
            stride,
        }
    }

    /// A zero-copy region of interest.
    pub fn roi(&self, rect: Rect) -> Result<MatView<'_, T>> {
        self.view().roi(rect)
    }

    /// A mutable region of interest, for `dst[y0:y1, x0:x1] = ...` style writes.
    pub fn roi_mut(&mut self, rect: Rect) -> Result<MatViewMut<'_, T>> {
        check_roi(rect, self.width, self.height)?;
        let stride = self.row_len();
        let start = rect.y * stride + rect.x * self.channels;
        let end = if rect.is_empty() {
            start
        } else {
            start + (rect.height - 1) * stride + rect.width * self.channels
        };
        Ok(MatViewMut {
            data: &mut self.data[start..end],
            width: rect.width,
            height: rect.height,
            channels: self.channels,
            stride,
        })
    }

    /// Contiguous copy of a region.
    pub fn crop(&self, rect: Rect) -> Result<Self> {
        Ok(Self::from_view(&self.roi(rect)?))
    }

    /// Element-wise map into a new image of the same shape.
    pub fn map<U: Element>(&self, f: impl Fn(T) -> U + Sync) -> Mat<U> {
        let data = self.data.par_iter().with_min_len(4096).map(|&v| f(v)).collect();
        Mat {
            width: self.width,
            height: self.height,
            channels: self.channels,
            data,
        }
    }

    /// `Mat::convertTo` without scaling: `saturate_cast` to the target type.
    pub fn convert<U: Element>(&self) -> Mat<U> {
        self.map(|v| U::from_f64(v.to_f64()))
    }

    /// `Mat::convertTo(dst, type, alpha, beta)`: `saturate_cast<U>(v * alpha + beta)`.
    pub fn convert_scaled<U: Element>(&self, alpha: f64, beta: f64) -> Mat<U> {
        self.map(|v| U::from_f64(v.to_f64() * alpha + beta))
    }

    /// Reinterpret the same buffer with a new geometry (`np.reshape` on the channel axis).
    pub fn reshape(self, width: usize, height: usize, channels: usize) -> Result<Self> {
        Self::from_vec(width, height, channels, self.data)
    }

    /// Rows as parallel mutable slices.
    pub fn par_rows_mut(&mut self) -> rayon::slice::ChunksMut<'_, T> {
        let len = self.row_len().max(1);
        self.data.par_chunks_mut(len)
    }

    pub fn rows(&self) -> std::slice::Chunks<'_, T> {
        self.data.chunks(self.row_len().max(1))
    }
}

pub(crate) fn check_roi(rect: Rect, width: usize, height: usize) -> Result<()> {
    if rect.right() > width || rect.bottom() > height {
        return Err(CvError::RoiOutOfBounds {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            image_width: width,
            image_height: height,
        });
    }
    Ok(())
}

/// A borrowed, possibly strided, read-only image.
#[derive(Clone, Copy)]
pub struct MatView<'a, T> {
    data: &'a [T],
    width: usize,
    height: usize,
    channels: usize,
    stride: usize,
}

impl<T> fmt::Debug for MatView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatView<{}>({}x{}x{})",
            std::any::type_name::<T>(),
            self.width,
            self.height,
            self.channels
        )
    }
}

impl<'a, T: Element> MatView<'a, T> {
    /// View a contiguous buffer.
    pub fn new(data: &'a [T], width: usize, height: usize, channels: usize) -> Result<Self> {
        Self::with_stride(data, width, height, channels, width * channels)
    }

    /// View a buffer whose rows start every `stride` elements.
    pub fn with_stride(
        data: &'a [T],
        width: usize,
        height: usize,
        channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let needed = if height == 0 {
            0
        } else {
            (height - 1) * stride + width * channels
        };
        if channels == 0 || data.len() < needed || stride < width * channels {
            return Err(CvError::BufferSize {
                len: data.len(),
                width,
                height,
                channels,
            });
        }
        Ok(Self {
            data,
            width,
            height,
            channels,
            stride,
        })
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub const fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub const fn row_len(&self) -> usize {
        self.width * self.channels
    }

    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// The backing slice (starting at the first pixel) and the row stride in elements.
    pub(crate) const fn raw(&self) -> (&'a [T], usize) {
        (self.data, self.stride)
    }

    /// Rows follow each other without padding (OpenCV's `isContinuous`).
    pub const fn is_contiguous(&self) -> bool {
        self.stride == self.width * self.channels || self.height <= 1
    }

    pub fn row(&self, y: usize) -> &'a [T] {
        let start = y * self.stride;
        &self.data[start..start + self.width * self.channels]
    }

    pub fn pixel(&self, x: usize, y: usize) -> &'a [T] {
        let start = y * self.stride + x * self.channels;
        &self.data[start..start + self.channels]
    }

    pub fn at(&self, x: usize, y: usize, channel: usize) -> T {
        self.data[y * self.stride + x * self.channels + channel]
    }

    pub fn roi(&self, rect: Rect) -> Result<Self> {
        check_roi(rect, self.width, self.height)?;
        let start = rect.y * self.stride + rect.x * self.channels;
        let data = if rect.is_empty() {
            &self.data[start..start]
        } else {
            &self.data[start..]
        };
        Ok(Self {
            data,
            width: rect.width,
            height: rect.height,
            channels: self.channels,
            stride: self.stride,
        })
    }

    pub fn to_mat(&self) -> Mat<T> {
        Mat::from_view(self)
    }

    /// Iterate over the elements of every row in order.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.height).flat_map(move |y| self.row(y).iter().copied())
    }
}

impl<'a, T: Element> From<&'a Mat<T>> for MatView<'a, T> {
    fn from(mat: &'a Mat<T>) -> Self {
        mat.view()
    }
}

impl<'a, T: Element> From<&MatView<'a, T>> for MatView<'a, T> {
    fn from(view: &MatView<'a, T>) -> Self {
        *view
    }
}

/// A borrowed, possibly strided, writable image region.
pub struct MatViewMut<'a, T> {
    data: &'a mut [T],
    width: usize,
    height: usize,
    channels: usize,
    stride: usize,
}

impl<T> fmt::Debug for MatViewMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatViewMut<{}>({}x{}x{})",
            std::any::type_name::<T>(),
            self.width,
            self.height,
            self.channels
        )
    }
}

impl<T: Element> MatViewMut<'_, T> {
    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub const fn channels(&self) -> usize {
        self.channels
    }

    pub fn row(&self, y: usize) -> &[T] {
        let start = y * self.stride;
        &self.data[start..start + self.width * self.channels]
    }

    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        let start = y * self.stride;
        &mut self.data[start..start + self.width * self.channels]
    }

    pub fn as_view(&self) -> MatView<'_, T> {
        MatView {
            data: self.data,
            width: self.width,
            height: self.height,
            channels: self.channels,
            stride: self.stride,
        }
    }

    /// Copy a same-shaped image into this region.
    pub fn copy_from<'b>(&mut self, src: impl Into<MatView<'b, T>>) -> Result<()> {
        let src = src.into();
        if src.size() != (self.width, self.height) || src.channels() != self.channels {
            return Err(CvError::ShapeMismatch { op: "copy_from" });
        }
        for y in 0..self.height {
            self.row_mut(y).copy_from_slice(src.row(y));
        }
        Ok(())
    }

    pub fn fill(&mut self, value: T) {
        for y in 0..self.height {
            self.row_mut(y).fill(value);
        }
    }
}

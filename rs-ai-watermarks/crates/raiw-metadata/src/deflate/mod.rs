//! Byte-exact deflate encoders of the zlib builds the Python reference can link.
//!
//! [`classic`] is madler zlib 1.4, the system `libz.so.1` that both Python's `zlib` module
//! and Pillow's `_imaging` resolve on the reference host (PNG `IDAT`, `iCCP`, `zlib.compress`).
//! [`ng`] is zlib-ng 2.3.3 `deflate_medium`, for builds that vendor zlib-ng. [`pillow`]
//! replays Pillow's `ZipEncode.c` row filters and `deflate` call pattern.

mod bits;
pub mod classic;
pub mod ng;
pub mod pillow;
mod tables;
mod trees;

pub use classic::{zlib_compress, zlib_compress_level};
pub use pillow::{PngFilter, Zlib, zip_encode, zip_encode_with};
pub use trees::Strategy;

/// The caller's input slice, consumed front to back (`next_in`/`avail_in`).
#[derive(Debug)]
pub struct Input<'a> {
    data: &'a [u8],
}

impl<'a> Input<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    pub fn remaining(&self) -> usize {
        self.data.len()
    }

    fn take(&mut self, max: usize) -> &'a [u8] {
        let (taken, rest) = self.data.split_at(self.data.len().min(max));
        self.data = rest;
        taken
    }
}

/// Delivered output plus the space left in the caller's current buffer (`avail_out`).
#[derive(Debug, Default)]
pub struct Output {
    pub data: Vec<u8>,
    pub avail: usize,
}

/// `deflate()` flush modes the reference drivers use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flush {
    None,
    Finish,
}

/// Successful `deflate()` outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    StreamEnd,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeflateError {
    #[error("unsupported compression level {0}")]
    Level(u8),
    #[error("unsupported memory level {0}")]
    MemLevel(u8),
    #[error("no progress possible (Z_BUF_ERROR)")]
    Buffer,
}

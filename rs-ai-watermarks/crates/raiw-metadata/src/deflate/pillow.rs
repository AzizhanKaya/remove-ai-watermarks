//! Pillow 13's `ZipEncode.c`: adaptive PNG row filters and its exact `deflate` call pattern.

use super::{DeflateError, Flush, Input, Output, Status, Strategy, classic, ng};

/// Pillow's per-row filter choice (`ZIP_PNG`), or the unfiltered palette mode (`ZIP_PNG_PALETTE`).
#[derive(Debug)]
pub struct PngFilter {
    bpp: usize,
    palette: bool,
    previous: Vec<u8>,
    none: Vec<u8>,
    up: Vec<u8>,
    prior: Vec<u8>,
    paeth: Vec<u8>,
}

#[inline]
fn cost(v: u8) -> i64 {
    if v < 128 { i64::from(v) } else { 256 - i64::from(v) }
}

impl PngFilter {
    /// `row_bytes` per scanline at `bits_per_pixel`; `palette` disables filtering.
    pub fn new(row_bytes: usize, bits_per_pixel: usize, palette: bool) -> Self {
        let row = |kind: u8| {
            let mut buffer = vec![0u8; row_bytes + 1];
            buffer[0] = kind;
            buffer
        };
        Self {
            bpp: bits_per_pixel.div_ceil(8),
            palette,
            previous: vec![0; row_bytes + 1],
            none: row(0),
            up: row(2),
            prior: row(1),
            paeth: row(4),
        }
    }

    /// The bytes Pillow feeds to `deflate` for `row`: the filter type then the filtered row.
    pub fn filter(&mut self, row: &[u8]) -> &[u8] {
        let bytes = self.none.len() - 1;
        self.none[1..].copy_from_slice(&row[..bytes]);
        let chosen = if self.palette { 0 } else { self.choose(bytes) };
        std::mem::swap(&mut self.previous, &mut self.none);
        self.none[0] = 0;
        match chosen {
            0 => &self.previous,
            2 => &self.up,
            1 => &self.prior,
            _ => &self.paeth,
        }
    }

    fn choose(&mut self, bytes: usize) -> u8 {
        let bpp = self.bpp;
        let (buffer, previous) = (&self.none, &self.previous);
        let mut chosen = 0;
        let mut sum: i64 = buffer[1..=bytes].iter().map(|&v| cost(v)).sum();
        if sum > 0 {
            let mut s = 0;
            for i in 1..=bytes {
                let v = buffer[i].wrapping_sub(previous[i]);
                self.up[i] = v;
                s += cost(v);
            }
            if s < sum {
                chosen = 2;
                sum = s;
            }
        }
        if sum > 0 {
            let mut s = 0;
            for i in 1..=bytes {
                let v = if i <= bpp {
                    buffer[i]
                } else {
                    buffer[i].wrapping_sub(buffer[i - bpp])
                };
                self.prior[i] = v;
                s += cost(v);
            }
            if s < sum {
                chosen = 1;
                sum = s;
            }
        }
        if sum > 0 {
            let mut s = 0;
            for i in 1..=bytes {
                let v = if i <= bpp {
                    buffer[i].wrapping_sub(previous[i])
                } else {
                    let a = i32::from(buffer[i - bpp]);
                    let b = i32::from(previous[i]);
                    let c = i32::from(previous[i - bpp]);
                    let pa = (b - c).abs();
                    let pb = (a - c).abs();
                    let pc = (a + b - 2 * c).abs();
                    let predictor = if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    };
                    buffer[i].wrapping_sub(predictor as u8)
                };
                self.paeth[i] = v;
                s += cost(v);
            }
            if s < sum {
                chosen = 4;
            }
        }
        chosen
    }
}

/// The zlib build behind Pillow's `deflate` calls.
///
/// manylinux wheels resolve `libz.so.1` from the system (madler zlib 1.4 on the reference
/// host); wheels that vendor zlib-ng link it statically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zlib {
    Classic,
    Ng,
}

#[derive(Debug)]
enum Engine {
    Classic(Box<classic::Deflater>),
    Ng(Box<ng::Deflater>),
}

impl Engine {
    fn deflate(
        &mut self,
        input: &mut Input<'_>,
        out: &mut Output,
        flush: Flush,
    ) -> Result<Status, DeflateError> {
        match self {
            Self::Classic(deflater) => deflater.deflate(input, out, flush),
            Self::Ng(deflater) => deflater.deflate(input, out, flush),
        }
    }
}

/// Compress `rows` (already filtered) the way Pillow's PNG saver does, on system zlib.
///
/// Level 6, 32 KiB window, memLevel 9; each returned buffer is one `encoder.encode(bufsize)`
/// result, i.e. one `IDAT` chunk payload (the last may be empty).
pub fn zip_encode<I, R>(rows: I, strategy: Strategy, bufsize: usize) -> Result<Vec<Vec<u8>>, DeflateError>
where
    I: IntoIterator<Item = R>,
    R: AsRef<[u8]>,
{
    zip_encode_with(rows, strategy, bufsize, Zlib::Classic)
}

/// [`zip_encode`] on an explicit zlib build.
pub fn zip_encode_with<I, R>(
    rows: I,
    strategy: Strategy,
    bufsize: usize,
    zlib: Zlib,
) -> Result<Vec<Vec<u8>>, DeflateError>
where
    I: IntoIterator<Item = R>,
    R: AsRef<[u8]>,
{
    let mut deflater = match zlib {
        Zlib::Classic => Engine::Classic(Box::new(classic::Deflater::new(6, 9, strategy)?)),
        Zlib::Ng => Engine::Ng(Box::new(ng::Deflater::new(6, 9, strategy)?)),
    };
    let mut rows = rows.into_iter();
    let mut held: Option<(R, usize)> = None;
    let mut finishing = false;
    let mut chunks = Vec::new();
    loop {
        let mut out = Output {
            data: Vec::with_capacity(bufsize),
            avail: bufsize,
        };
        if let Some((row, consumed)) = &mut held {
            let data = &row.as_ref()[*consumed..];
            if !data.is_empty() {
                let mut input = Input::new(data);
                deflater.deflate(&mut input, &mut out, Flush::None)?;
                *consumed = row.as_ref().len() - input.remaining();
            }
        }
        let mut done = false;
        if !finishing {
            while out.avail > 0 {
                let Some(row) = rows.next() else {
                    finishing = true;
                    break;
                };
                let mut input = Input::new(row.as_ref());
                deflater.deflate(&mut input, &mut out, Flush::None)?;
                let consumed = row.as_ref().len() - input.remaining();
                held = Some((row, consumed));
            }
        }
        if finishing && out.avail > 0 {
            while out.avail > 0 {
                let status = deflater.deflate(&mut Input::new(&[]), &mut out, Flush::Finish)?;
                if status == Status::StreamEnd {
                    done = true;
                    break;
                }
            }
        }
        chunks.push(out.data);
        if done {
            return Ok(chunks);
        }
    }
}

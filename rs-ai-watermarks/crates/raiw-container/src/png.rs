//! PNG chunk walkers: bounded chunk iteration, late metadata, and lossless chunk rewriting.

use std::io;
use std::path::Path;

use raiw_core::constants::{PNG_METADATA_CHUNKS, PNG_SIGNATURE};

use crate::source::{FileSource, ReadAt};

/// One chunk header: the chunk starts at `offset` and its payload at `offset + 8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PngChunk {
    pub kind: [u8; 4],
    pub offset: u64,
    pub length: u64,
}

impl PngChunk {
    pub const fn payload_offset(&self) -> u64 {
        self.offset + 8
    }

    /// Offset just past the CRC.
    pub const fn end(&self) -> u64 {
        self.offset + 8 + self.length + 4
    }
}

/// Chunks whose payload and CRC fit the file, up to and including `IEND`.
///
/// The walk ends early at a short header or an overlong length; [`PngChunks::complete`]
/// then reports `false`.
#[derive(Debug)]
pub struct PngChunks<'a, R: ?Sized> {
    source: &'a R,
    pos: u64,
    finished: bool,
    complete: bool,
}

impl<R: ReadAt + ?Sized> PngChunks<'_, R> {
    /// Whether the walk reached `IEND`.
    pub const fn complete(&self) -> bool {
        self.complete
    }
}

impl<R: ReadAt + ?Sized> Iterator for PngChunks<'_, R> {
    type Item = io::Result<PngChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        self.finished = true;
        let header = match self.source.read_array::<8>(self.pos) {
            Ok(Some(header)) => header,
            Ok(None) => return None,
            Err(err) => return Some(Err(err)),
        };
        let length = u64::from(u32::from_be_bytes([header[0], header[1], header[2], header[3]]));
        let kind = [header[4], header[5], header[6], header[7]];
        let chunk = PngChunk {
            kind,
            offset: self.pos,
            length,
        };
        if chunk.end() > self.source.size() {
            return None;
        }
        if &kind == b"IEND" {
            self.complete = true;
        } else {
            self.finished = false;
            self.pos = chunk.end();
        }
        Some(Ok(chunk))
    }
}

/// Walk a PNG's chunks; yields nothing when the signature does not match.
pub fn chunks<R: ReadAt + ?Sized>(source: &R) -> io::Result<PngChunks<'_, R>> {
    let signed = source.read_vec(0, 8)? == PNG_SIGNATURE;
    Ok(PngChunks {
        source,
        pos: 8,
        finished: !signed,
        complete: false,
    })
}

/// Bit depth declared in IHDR, or 0 when the bytes are not a PNG.
pub fn bit_depth(data: &[u8]) -> u8 {
    if !data.starts_with(PNG_SIGNATURE) || data.len() < 26 || &data[12..16] != b"IHDR" {
        return 0;
    }
    data[24]
}

/// Payloads of metadata chunks starting at or after `window`, seeking past `IDAT`.
///
/// Each length is clamped to the bytes that remain, so an inflated length cannot abort the
/// walk; `Err` is Python's `strict=True` raise.
pub fn late_metadata<R: ReadAt + ?Sized>(source: &R, window: u64) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    if source.read_vec(0, 8)? != PNG_SIGNATURE {
        return Ok(out);
    }
    let file_size = source.size();
    let mut pos = 8u64;
    while let Some(header) = source.read_array::<8>(pos)? {
        let length = u64::from(u32::from_be_bytes([header[0], header[1], header[2], header[3]]));
        let kind = &header[4..8];
        if kind == b"IEND" {
            break;
        }
        let data_start = pos + 8;
        let safe_length = length.min(file_size.saturating_sub(data_start));
        if PNG_METADATA_CHUNKS.contains(&kind) && data_start >= window {
            out.extend(source.read_vec(data_start, safe_length)?);
        }
        pos = data_start + safe_length + 4;
    }
    Ok(out)
}

/// `metadata._png_late_metadata` over a file.
pub fn late_metadata_file(path: &Path, window: u64) -> io::Result<Vec<u8>> {
    late_metadata(&FileSource::open(path)?, window)
}

/// Serialize one chunk with its CRC.
pub fn serialize_chunk(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 12);
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    let mut crc = crc32fast::Hasher::new();
    crc.update(kind);
    crc.update(payload);
    out.extend_from_slice(&crc.finalize().to_be_bytes());
    out
}

/// What [`rewrite_chunks`] does with one chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkAction {
    Keep,
    Drop,
    /// Re-emit the chunk with this payload and a fresh CRC.
    Replace(Vec<u8>),
}

/// Rewrite a PNG chunk by chunk, copying kept chunks (and every pixel byte) verbatim.
///
/// `decide` sees each chunk type and payload. Bytes after `IEND` are dropped; `None` for a
/// malformed length or a file that ends before `IEND`.
pub fn rewrite_chunks(
    data: &[u8],
    mut decide: impl FnMut(&[u8; 4], &[u8]) -> ChunkAction,
) -> Option<Vec<u8>> {
    if !data.starts_with(PNG_SIGNATURE) {
        return None;
    }
    let mut out = PNG_SIGNATURE.to_vec();
    let n = data.len();
    let mut pos = 8;
    while pos + 8 <= n {
        let length = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let kind = [data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]];
        let end = pos + 8 + length + 4;
        if length > n || end > n {
            return None;
        }
        match decide(&kind, &data[pos + 8..pos + 8 + length]) {
            ChunkAction::Keep => out.extend_from_slice(&data[pos..end]),
            ChunkAction::Drop => {}
            ChunkAction::Replace(payload) => out.extend(serialize_chunk(&kind, &payload)),
        }
        if &kind == b"IEND" {
            return Some(out);
        }
        pos = end;
    }
    None
}

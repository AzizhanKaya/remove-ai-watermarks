//! Random-access byte sources: an in-memory buffer or an open file read with `pread`.

use std::fs::File;
use std::io;
use std::path::Path;

/// Positioned reads over a fixed-size byte source, mirroring Python's `seek` + `read`.
///
/// A read past the end is short rather than an error, exactly like `stream.read(n)`.
pub trait ReadAt {
    /// Total size in bytes, fixed when the source was opened.
    fn size(&self) -> u64;

    /// Fill as much of `buf` as the source holds from `offset`; returns the byte count.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize>;

    /// Read up to `len` bytes from `offset`, allocating only what the source still holds.
    fn read_vec(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        let available = self.size().saturating_sub(offset).min(len);
        let mut buf = vec![0u8; usize::try_from(available).map_err(io::Error::other)?];
        let read = self.read_at(offset, &mut buf)?;
        buf.truncate(read);
        Ok(buf)
    }

    /// Read exactly `N` bytes, or `None` when the source ends first.
    fn read_array<const N: usize>(&self, offset: u64) -> io::Result<Option<[u8; N]>> {
        let mut buf = [0u8; N];
        Ok((self.read_at(offset, &mut buf)? == N).then_some(buf))
    }
}

impl ReadAt for [u8] {
    fn size(&self) -> u64 {
        self.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let Ok(start) = usize::try_from(offset) else {
            return Ok(0);
        };
        let Some(tail) = self.get(start..) else {
            return Ok(0);
        };
        let count = tail.len().min(buf.len());
        buf[..count].copy_from_slice(&tail[..count]);
        Ok(count)
    }
}

impl ReadAt for Vec<u8> {
    fn size(&self) -> u64 {
        self.as_slice().size()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        self.as_slice().read_at(offset, buf)
    }
}

impl<T: ReadAt + ?Sized> ReadAt for &T {
    fn size(&self) -> u64 {
        (**self).size()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        (**self).read_at(offset, buf)
    }
}

/// An open file whose size is captured once, read with positioned I/O.
#[derive(Debug)]
pub struct FileSource {
    file: File,
    size: u64,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::new(File::open(path)?)
    }

    pub fn new(file: File) -> io::Result<Self> {
        let size = file.metadata()?.len();
        Ok(Self { file, size })
    }

    pub fn file(&self) -> &File {
        &self.file
    }
}

impl ReadAt for FileSource {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let mut filled = 0;
        while filled < buf.len() {
            match positioned_read(&self.file, &mut buf[filled..], offset + filled as u64) {
                Ok(0) => break,
                Ok(read) => filled += read,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(err),
            }
        }
        Ok(filled)
    }
}

#[cfg(unix)]
fn positioned_read(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

#[cfg(windows)]
fn positioned_read(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_reads_are_short_at_the_end() {
        let data = b"abcdef".as_slice();
        assert_eq!(data.read_vec(4, 10).unwrap(), b"ef");
        assert_eq!(data.read_vec(9, 10).unwrap(), b"");
        assert_eq!(data.read_array::<2>(5).unwrap(), None);
        assert_eq!(data.read_array::<2>(4).unwrap(), Some(*b"ef"));
    }
}

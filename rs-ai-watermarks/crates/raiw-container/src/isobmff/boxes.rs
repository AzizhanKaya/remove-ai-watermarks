//! `[size:4][type:4]` box headers with 64-bit `largesize` and run-to-end (`size == 0`) forms.

use std::io;

use crate::source::ReadAt;

pub type FourCc = [u8; 4];

/// One valid box: `[start, end)` with its payload beginning at `payload`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoxEntry {
    pub start: u64,
    pub end: u64,
    pub kind: FourCc,
    pub payload: u64,
}

/// A half-open byte range `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    pub start: u64,
    pub end: u64,
}

impl Span {
    pub const fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    pub const fn len(self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    pub const fn is_empty(self) -> bool {
        self.end <= self.start
    }

    /// Whether `[start, end)` lies inside this span.
    pub const fn encloses(self, start: u64, end: u64) -> bool {
        self.start <= start && end <= self.end
    }
}

/// Read the header of the box at `pos`, which must end by `limit`.
pub fn read_box_header<R: ReadAt + ?Sized>(source: &R, pos: u64, limit: u64) -> io::Result<Option<BoxEntry>> {
    if pos.checked_add(8).is_none_or(|header_end| header_end > limit) {
        return Ok(None);
    }
    let Some(header) = source.read_array::<8>(pos)? else {
        return Ok(None);
    };
    let size32 = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let kind = [header[4], header[5], header[6], header[7]];
    let (size, payload) = match size32 {
        1 => {
            let Some(extended) = source.read_array::<8>(pos + 8)? else {
                return Ok(None);
            };
            (u64::from_be_bytes(extended), pos + 16)
        }
        0 => (limit - pos, pos + 8),
        size => (u64::from(size), pos + 8),
    };
    let Some(end) = pos.checked_add(size) else {
        return Ok(None);
    };
    if size < payload - pos || end > limit {
        return Ok(None);
    }
    Ok(Some(BoxEntry {
        start: pos,
        end,
        kind,
        payload,
    }))
}

/// Valid consecutive boxes in `[start, end)`; stops at the first malformed header.
#[derive(Debug)]
pub struct Boxes<'a, R: ?Sized> {
    source: &'a R,
    pos: u64,
    end: u64,
    done: bool,
}

impl<R: ReadAt + ?Sized> Iterator for Boxes<'_, R> {
    type Item = io::Result<BoxEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pos.saturating_add(8) > self.end {
            return None;
        }
        match read_box_header(self.source, self.pos, self.end) {
            Ok(Some(entry)) => {
                self.pos = entry.end;
                Some(Ok(entry))
            }
            Ok(None) => {
                self.done = true;
                None
            }
            Err(err) => {
                self.done = true;
                Some(Err(err))
            }
        }
    }
}

/// Iterate the boxes of one bounded container region (`isobmff.iter_file_boxes`).
pub fn iter_boxes<R: ReadAt + ?Sized>(source: &R, start: u64, end: u64) -> Boxes<'_, R> {
    Boxes {
        source,
        pos: start,
        end,
        done: false,
    }
}

/// Collect the boxes of one region.
pub fn boxes<R: ReadAt + ?Sized>(source: &R, start: u64, end: u64) -> io::Result<Vec<BoxEntry>> {
    iter_boxes(source, start, end).collect()
}

/// Top-level boxes of an in-memory buffer.
pub fn top_level_boxes(data: &[u8]) -> Vec<BoxEntry> {
    iter_boxes(data, 0, data.len() as u64)
        .map_while(Result::ok)
        .collect()
}

/// The last child of each type, as Python's `{type: (payload, end)}` dict keeps it.
pub fn last_of_type(children: &[BoxEntry], kind: &[u8]) -> Option<BoxEntry> {
    children.iter().rev().find(|child| child.kind == kind).copied()
}

/// Cheap sniff: an ISOBMFF file starts with an `ftyp` box.
pub fn is_isobmff(data: &[u8]) -> bool {
    data.len() >= 8 && &data[4..8] == b"ftyp"
}

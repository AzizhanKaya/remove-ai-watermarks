//! A piexif-exact EXIF reader and writer (`piexif.load`, `piexif.dump`, `piexif.insert`).
//!
//! Values keep piexif's Python shapes, so a load/scrub/dump round trip produces the same
//! bytes as the Python reference, including its quirks.

mod dump;
mod load;
pub mod tags;

use indexmap::IndexMap;

pub use dump::{DumpError, dump, insert_jpeg, split_jpeg_segments};
pub use load::{LoadError, load, load_jpeg_file};

/// The five IFDs of a piexif dict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ifd {
    Zeroth,
    Exif,
    Gps,
    Interop,
    First,
}

impl Ifd {
    pub const ALL: [Self; 5] = [Self::Zeroth, Self::Exif, Self::Gps, Self::Interop, Self::First];

    /// The piexif dict key (`"0th"`, `"Exif"`, ...).
    pub const fn key(self) -> &'static str {
        match self {
            Self::Zeroth => "0th",
            Self::Exif => "Exif",
            Self::Gps => "GPS",
            Self::Interop => "Interop",
            Self::First => "1st",
        }
    }
}

/// Tag numbers piexif names in `ImageIFD` / `ExifIFD`.
pub mod tag {
    pub const IMAGE_DESCRIPTION: u16 = 270;
    pub const MAKE: u16 = 271;
    pub const ORIENTATION: u16 = 274;
    pub const X_RESOLUTION: u16 = 282;
    pub const RESOLUTION_UNIT: u16 = 296;
    pub const SOFTWARE: u16 = 305;
    pub const ARTIST: u16 = 315;
    pub const JPEG_INTERCHANGE_FORMAT: u16 = 513;
    pub const JPEG_INTERCHANGE_FORMAT_LENGTH: u16 = 514;
    pub const EXIF_TAG: u16 = 34665;
    pub const GPS_TAG: u16 = 34853;
    pub const INTEROPERABILITY_TAG: u16 = 40965;
    pub const USER_COMMENT: u16 = 37510;
    pub const MAKER_NOTE: u16 = 37500;
}

/// One decoded value, in the shape piexif hands back.
#[derive(Debug, Clone, PartialEq)]
pub enum ExifValue {
    /// ASCII (minus its last byte) and UNDEFINED values: Python `bytes`.
    Bytes(Vec<u8>),
    /// A one-element integer tuple, unwrapped.
    Int(i64),
    /// An integer tuple of any other length.
    Ints(Vec<i64>),
    /// One `(numerator, denominator)` pair.
    Rational(i64, i64),
    /// A tuple of pairs.
    Rationals(Vec<(i64, i64)>),
    Float(f64),
    Floats(Vec<f64>),
}

impl ExifValue {
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            _ => None,
        }
    }

    pub const fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            _ => None,
        }
    }
}

pub type IfdMap = IndexMap<u16, ExifValue>;

/// `piexif.load` output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExifDict {
    pub zeroth: IfdMap,
    pub exif: IfdMap,
    pub gps: IfdMap,
    pub interop: IfdMap,
    pub first: IfdMap,
    pub thumbnail: Option<Vec<u8>>,
}

impl ExifDict {
    pub const fn ifd(&self, ifd: Ifd) -> &IfdMap {
        match ifd {
            Ifd::Zeroth => &self.zeroth,
            Ifd::Exif => &self.exif,
            Ifd::Gps => &self.gps,
            Ifd::Interop => &self.interop,
            Ifd::First => &self.first,
        }
    }

    pub const fn ifd_mut(&mut self, ifd: Ifd) -> &mut IfdMap {
        match ifd {
            Ifd::Zeroth => &mut self.zeroth,
            Ifd::Exif => &mut self.exif,
            Ifd::Gps => &mut self.gps,
            Ifd::Interop => &mut self.interop,
            Ifd::First => &mut self.first,
        }
    }

    /// Bytes value of one tag (`isinstance(value, bytes)`).
    pub fn bytes(&self, ifd: Ifd, tag: u16) -> Option<&[u8]> {
        self.ifd(ifd).get(&tag).and_then(ExifValue::as_bytes)
    }

    /// `_exif_text`: latin-1 decoded and stripped, empty when absent or not bytes.
    pub fn text(&self, ifd: Ifd, tag: u16) -> String {
        self.bytes(ifd, tag)
            .map(|raw| crate::py::strip(&crate::py::latin1(raw)).to_owned())
            .unwrap_or_default()
    }
}

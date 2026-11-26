//! What Pillow's `Image.open(path)` exposes: format, mode, size and the `info` mapping.
//!
//! Each plugin's header handlers are reproduced so `info` keys, order and value types
//! match Pillow 13. `load()` is modelled too, because a PNG reads the chunks after its
//! pixel data only then. Results are memoized per file content.

pub mod gif;
pub mod ifd;
pub mod iptc;
pub mod jpeg;
pub mod other;
pub mod png;
pub mod value;
pub mod webp;

use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use raiw_container::FileSource;

use crate::cache::{ContentKey, LruCache};
use crate::exif::{ExifDict, ExifValue, Ifd};
use crate::py::str_repr;
pub use value::{Info, InfoValue};

/// `ImageFile.SAFEBLOCK`.
pub const SAFEBLOCK: usize = 1024 * 1024;
const MAX_IMAGE_PIXELS: u64 = 89_478_485;

/// Why `Image.open` or `load()` failed, with the exception Python raises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenError {
    /// `UnidentifiedImageError`: no plugin accepted the file.
    Unidentified,
    /// An `OSError` raised out of a plugin.
    Io(String),
    /// A `ValueError` raised out of a plugin.
    Value(String),
    /// A `SyntaxError` raised out of `load()`.
    Syntax(String),
    /// `DecompressionBombError`.
    Bomb(String),
}

impl OpenError {
    /// `str(exc)` for a file at `path`.
    pub fn message(&self, path: &Path) -> String {
        match self {
            Self::Unidentified => format!("cannot identify image file {}", str_repr(&path.to_string_lossy())),
            Self::Io(msg) | Self::Value(msg) | Self::Syntax(msg) | Self::Bomb(msg) => msg.clone(),
        }
    }
}

/// Per-format state kept for `load()` and `getexif()`.
#[derive(Debug, Clone)]
pub enum Plugin {
    Png(png::PngOpen),
    Jpeg(jpeg::JpegOpen),
    Webp,
    Gif(usize),
    Bmp,
    Tiff(ifd::RawIfd, bool),
}

/// An opened image.
#[derive(Debug, Clone)]
pub struct Image {
    pub format: &'static str,
    pub mode: &'static str,
    pub size: (u32, u32),
    pub n_frames: usize,
    /// `info` right after `open`.
    pub info: Info,
    pub plugin: Plugin,
}

/// `load()`: the error it raised (if any) and `info` afterwards.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub info: Info,
    pub error: Option<OpenError>,
}

/// One file's memoized Pillow view.
#[derive(Debug)]
pub struct Probe {
    pub path: std::path::PathBuf,
    pub opened: Result<Image, OpenError>,
    data: Arc<Vec<u8>>,
    loaded: OnceLock<Loaded>,
}

fn decompression_bomb(size: (u32, u32)) -> Result<(), OpenError> {
    let pixels = u64::from(size.0.max(1)) * u64::from(size.1.max(1));
    if pixels > 2 * MAX_IMAGE_PIXELS {
        return Err(OpenError::Bomb(format!(
            "Image size ({pixels} pixels) exceeds limit of {} pixels, could be decompression bomb DOS attack.",
            2 * MAX_IMAGE_PIXELS
        )));
    }
    Ok(())
}

fn open_data(data: &[u8]) -> Result<Image, OpenError> {
    let image = if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        let state = png::open(data)?;
        Image {
            format: "PNG",
            mode: state.mode,
            size: (state.ihdr.width, state.ihdr.height),
            n_frames: state.n_frames,
            info: state.info.clone(),
            plugin: Plugin::Png(state),
        }
    } else if data.starts_with(b"\xff\xd8\xff") {
        let state = jpeg::open(data)?;
        Image {
            format: state.format,
            mode: state.mode,
            size: state.size,
            n_frames: state.n_frames,
            info: state.info.clone(),
            plugin: Plugin::Jpeg(state),
        }
    } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
        let state = webp::open(data)?;
        Image {
            format: "WEBP",
            mode: state.mode,
            size: state.size,
            n_frames: state.n_frames,
            info: state.info,
            plugin: Plugin::Webp,
        }
    } else if matches!(data.get(..6), Some(b"GIF87a" | b"GIF89a")) {
        let state = gif::open(data)?;
        Image {
            format: "GIF",
            mode: state.mode,
            size: state.size,
            n_frames: gif::count_frames(data, state.frame_offset),
            info: state.info,
            plugin: Plugin::Gif(state.frame_offset),
        }
    } else if data.starts_with(b"BM") {
        let state = other::open_bmp(data)?;
        Image {
            format: state.format,
            mode: state.mode,
            size: state.size,
            n_frames: 1,
            info: state.info,
            plugin: Plugin::Bmp,
        }
    } else if data.starts_with(b"II*\x00") || data.starts_with(b"MM\x00*") {
        let state = other::open_tiff(data)?;
        let (dir, little) = state.tiff_ifd.unwrap_or_default();
        Image {
            format: state.format,
            mode: state.mode,
            size: state.size,
            n_frames: 1,
            info: state.info,
            plugin: Plugin::Tiff(dir, little),
        }
    } else {
        return Err(OpenError::Unidentified);
    };
    decompression_bomb(image.size)?;
    Ok(image)
}

static CACHE: Mutex<LruCache<Arc<Probe>>> = Mutex::new(LruCache::new(4));

/// `Image.open(path)`, memoized on the file's content identity.
pub fn probe(path: &Path) -> Arc<Probe> {
    let key = ContentKey::of(path);
    if let Some(key) = &key
        && let Ok(cache) = CACHE.lock()
        && let Some(hit) = cache.get(key)
    {
        return hit;
    }
    let data = Arc::new(std::fs::read(path).unwrap_or_default());
    let opened = if data.is_empty() {
        Err(OpenError::Unidentified)
    } else {
        open_data(&data)
    };
    let probe = Arc::new(Probe {
        path: path.to_path_buf(),
        opened,
        data,
        loaded: OnceLock::new(),
    });
    if let Some(key) = key
        && let Ok(mut cache) = CACHE.lock()
    {
        cache.put(key, Arc::clone(&probe));
    }
    probe
}

impl Probe {
    pub fn image(&self) -> Option<&Image> {
        self.opened.as_ref().ok()
    }

    /// The whole file.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// `img.load()` and the `info` it leaves behind.
    pub fn loaded(&self) -> Option<&Loaded> {
        let image = self.image()?;
        Some(self.loaded.get_or_init(|| load(image, &self.data)))
    }

    /// Whether `exif_bytes_from_image` would load the image (a PNG without an `exif` key).
    fn exif_loads(&self) -> bool {
        self.image()
            .is_some_and(|image| image.format == "PNG" && !image.info.contains_key("exif"))
    }

    /// `info` after `exif_bytes_from_image` ran (it loads a PNG without `eXIf`).
    pub fn info_after_exif(&self) -> Option<&Info> {
        if self.exif_loads() {
            return self.loaded().map(|loaded| &loaded.info);
        }
        self.image().map(|image| &image.info)
    }

    /// `exif_bytes_from_image(img)`: the raw blob, or Pillow's decoded `getexif()` serialized.
    pub fn exif_bytes(&self) -> Option<Vec<u8>> {
        let image = self.image()?;
        if let Some(InfoValue::Bytes(raw)) = image.info.get("exif")
            && !raw.is_empty()
        {
            return Some(raw.clone());
        }
        let info = if self.exif_loads() {
            let loaded = self.loaded()?;
            if loaded.error.is_some() {
                return None;
            }
            &loaded.info
        } else {
            &image.info
        };
        let exif = self.getexif(image, info)?;
        if exif.zeroth.is_empty() {
            return None;
        }
        crate::exif::dump(&exif).ok()
    }

    /// `img.getexif()` reduced to the piexif view of its `tobytes()` serialization.
    fn getexif(&self, image: &Image, info: &Info) -> Option<ExifDict> {
        getexif(image, info, &self.data)
    }
}

fn getexif(image: &Image, info: &Info, data: &[u8]) -> Option<ExifDict> {
    let mut exif = match info.get("exif") {
        Some(InfoValue::Bytes(raw)) => pillow_exif(raw)?,
        Some(_) => return None,
        None => {
            if let Some(profile) = info.get("Raw profile type exif").and_then(InfoValue::as_str) {
                let hex: String = profile.split('\n').skip(3).collect();
                let raw = hex_decode(&hex)?;
                pillow_exif(&raw)?
            } else if let Plugin::Tiff(..) = image.plugin {
                pillow_exif(data)?
            } else {
                ExifDict::default()
            }
        }
    };
    if !exif.zeroth.contains_key(&crate::exif::tag::ORIENTATION) {
        let xmp = info
            .get("XML:com.adobe.xmp")
            .and_then(InfoValue::as_str)
            .filter(|s| !s.is_empty())
            .map(|s| s.as_bytes().to_vec())
            .or_else(|| {
                info.get("xmp")
                    .and_then(InfoValue::as_bytes)
                    .filter(|b| !b.is_empty())
                    .map(<[u8]>::to_vec)
            });
        if let Some(xmp) = xmp
            && let Some(orientation) = xmp_orientation(&xmp)
        {
            exif.zeroth
                .insert(crate::exif::tag::ORIENTATION, ExifValue::Int(orientation));
        }
    }
    Some(exif)
}

fn xmp_orientation(xmp: &[u8]) -> Option<i64> {
    let finder = memchr::memmem::Finder::new(b"tiff:Orientation");
    for at in finder.find_iter(xmp) {
        let rest = &xmp[at + 16..];
        let digit = if rest.starts_with(b"=\"") {
            rest.get(2)
        } else if rest.starts_with(b">") {
            rest.get(1)
        } else {
            None
        };
        if let Some(d) = digit.filter(|d| d.is_ascii_digit()) {
            return Some(i64::from(d - b'0'));
        }
    }
    None
}

/// `bytes.fromhex`: whitespace between pairs is allowed.
fn hex_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut chars = text.chars().filter(|c| !c.is_ascii_whitespace());
    while let Some(high) = chars.next() {
        let low = chars.next()?;
        out.push(u8::try_from(high.to_digit(16)? * 16 + low.to_digit(16)?).ok()?);
    }
    Some(out)
}

/// The tags Pillow's `Exif.load(raw)` + `tobytes()` carry, read the piexif way.
fn pillow_exif(raw: &[u8]) -> Option<ExifDict> {
    let mut tiff = raw;
    while let Some(rest) = tiff.strip_prefix(b"Exif\x00\x00") {
        tiff = rest;
    }
    if tiff.is_empty() {
        return Some(ExifDict::default());
    }
    ifd::header_endian(tiff)?;
    let mut dict = crate::exif::load(tiff).ok()?;
    dict.first.clear();
    dict.thumbnail = None;
    for ifd in Ifd::ALL {
        for value in dict.ifd_mut(ifd).values_mut() {
            if let ExifValue::Bytes(bytes) = value {
                for b in bytes.iter_mut() {
                    if *b >= 0x80 {
                        *b = b'?';
                    }
                }
            }
        }
    }
    Some(dict)
}

fn load(image: &Image, data: &[u8]) -> Loaded {
    match &image.plugin {
        Plugin::Png(state) => {
            let decoded = png::decode(data, state, false);
            if !decoded.ended {
                return Loaded {
                    info: image.info.clone(),
                    error: decoded.error.map(OpenError::Io),
                };
            }
            let (info, late_error) = png::load_end(data, state);
            Loaded {
                info,
                error: late_error.or_else(|| decoded.error.map(OpenError::Io)),
            }
        }
        Plugin::Jpeg(state) => {
            let error = match state.scan_start {
                None => Some(OpenError::Io("cannot load this image".into())),
                Some(start) if memchr::memmem::find(&data[start..], b"\xff\xd9").is_none() => Some(
                    OpenError::Io("image file is truncated (0 bytes not processed)".into()),
                ),
                Some(_) => None,
            };
            Loaded {
                info: image.info.clone(),
                error,
            }
        }
        _ => Loaded {
            info: image.info.clone(),
            error: crate::readability::generic_decode_error(data),
        },
    }
}

/// A `FileSource` for plugins that read lazily (kept for large-file callers).
pub fn source(path: &Path) -> std::io::Result<FileSource> {
    FileSource::open(path)
}

//! `unreadable_reason`: the fail-closed probe callers ask before reporting "no metadata".

use std::io;
use std::path::Path;
use std::sync::Mutex;

use raiw_container::FileSource;
use raiw_container::ebml::EBML_MAGIC;
use raiw_container::readability::{
    JXL_CONTAINER_MAGIC, ebml_tiling_reason, isobmff_tiling_reason, passes_on_magic,
};
use raiw_container::riff::declared_size_reason;

use crate::cache::{LruCache, memoize};
use crate::pil::{self, OpenError, Plugin};
use crate::py::str_repr;

const PILLOW_RASTERS: &[&str] = &["png", "jpeg", "webp", "gif", "bmp", "tiff"];

/// `forensic_metadata.sniff_format`.
pub fn sniff_format(head: &[u8]) -> String {
    if head.starts_with(b"\x89PNG") {
        "png".into()
    } else if head.starts_with(b"\xff\xd8") {
        "jpeg".into()
    } else if head.starts_with(b"RIFF") && head.get(8..12) == Some(b"WEBP") {
        "webp".into()
    } else if matches!(head.get(..6), Some(b"GIF87a" | b"GIF89a")) {
        "gif".into()
    } else if head.starts_with(b"BM") {
        "bmp".into()
    } else if head.starts_with(b"II*\x00") || head.starts_with(b"MM\x00*") {
        "tiff".into()
    } else if head.get(4..8) == Some(b"ftyp") {
        format!(
            "isobmff:{}",
            crate::py::latin1(head.get(8..12).unwrap_or_default())
        )
    } else {
        format!("unknown:{}", hex::encode(&head[..head.len().min(16)]))
    }
}

/// Python's `str(OSError)`: `[Errno N] message: 'path'`.
pub fn os_error_text(err: &io::Error, path: &Path) -> String {
    let text = err.to_string();
    let message = text.find(" (os error ").map_or(text.as_str(), |at| &text[..at]);
    match err.raw_os_error() {
        Some(code) => format!("[Errno {code}] {message}: {}", str_repr(&path.to_string_lossy())),
        None => message.to_owned(),
    }
}

/// A Pillow-style decode error for formats decoded through the `image` crate.
pub fn generic_decode_error(data: &[u8]) -> Option<OpenError> {
    image::load_from_memory(data)
        .err()
        .map(|_| OpenError::Io("image file is truncated".into()))
}

/// Whether OpenCV (`image_io.imread`) would decode the file.
fn pixel_stack_decodes(probe: &pil::Probe) -> bool {
    let data = probe.bytes();
    match probe.image().map(|image| &image.plugin) {
        Some(Plugin::Png(state)) => pil::png::decode(data, state, false).error.is_none(),
        Some(Plugin::Jpeg(state)) => state.scan_start.is_some(),
        Some(Plugin::Webp | Plugin::Gif(_)) => false,
        Some(_) => image::load_from_memory(data).is_ok(),
        None => libpng_decodes(data) || jpeg_header_decodes(data),
    }
}

/// libpng tolerates bad ancillary CRCs that make Pillow refuse a PNG.
fn libpng_decodes(data: &[u8]) -> bool {
    if !data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return false;
    }
    let mut pos = 8usize;
    let mut idat = Vec::new();
    let mut ihdr = None;
    while pos + 12 <= data.len() {
        let length = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let kind = &data[pos + 4..pos + 8];
        let Some(body) = data.get(pos + 8..pos + 8 + length) else {
            return false;
        };
        let Some(crc) = data.get(pos + 8 + length..pos + 12 + length) else {
            return false;
        };
        let critical = kind[0].is_ascii_uppercase();
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(kind);
        hasher.update(body);
        if critical && hasher.finalize().to_be_bytes() != crc {
            return false;
        }
        match kind {
            b"IHDR" if body.len() >= 13 => ihdr = Some(body.to_vec()),
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        pos += 12 + length;
    }
    let Some(ihdr) = ihdr else { return false };
    let mut synthetic = b"\x89PNG\r\n\x1a\n".to_vec();
    synthetic.extend(raiw_container::png::serialize_chunk(b"IHDR", &ihdr));
    synthetic.extend(raiw_container::png::serialize_chunk(b"IDAT", &idat));
    synthetic.extend(raiw_container::png::serialize_chunk(b"IEND", b""));
    match pil::png::open(&synthetic[..]) {
        Ok(state) => pil::png::decode(&synthetic[..], &state, false).error.is_none(),
        Err(_) => false,
    }
}

fn jpeg_header_decodes(data: &[u8]) -> bool {
    pil::jpeg::open(data).is_ok_and(|state| state.scan_start.is_some())
}

fn reason_uncached(path: &Path, size: u64) -> io::Result<Option<String>> {
    if size == 0 {
        return Ok(Some("the file is empty".into()));
    }
    let head = crate::scan::read_head(path, 32)?;
    let kind = sniff_format(&head);
    if kind.starts_with("isobmff:") || head.starts_with(JXL_CONTAINER_MAGIC) {
        return isobmff_tiling_reason(&FileSource::open(path)?);
    }
    if head.starts_with(EBML_MAGIC) {
        return ebml_tiling_reason(&FileSource::open(path)?);
    }
    if PILLOW_RASTERS.contains(&kind.as_str()) {
        let probe = pil::probe(path);
        let error = match &probe.opened {
            Err(OpenError::Bomb(_)) => return Ok(None),
            Err(err) => Some(err.clone()),
            Ok(_) => probe.loaded().and_then(|loaded| loaded.error.clone()),
        };
        return Ok(match error {
            None | Some(OpenError::Bomb(_)) => None,
            Some(_) if pixel_stack_decodes(&probe) => None,
            Some(err) => Some(format!("the image cannot be decoded: {}", err.message(path))),
        });
    }
    if head.starts_with(b"RIFF") && head.len() >= 8 {
        return Ok(declared_size_reason(&head, size));
    }
    if passes_on_magic(&head) {
        return Ok(None);
    }
    Ok(Some(
        "the content is not a recognized image, video or audio container".into(),
    ))
}

static CACHE: Mutex<LruCache<Option<String>>> = Mutex::new(LruCache::new(8));

/// `unreadable_reason(path)`: why `path` is not a readable container, or `None`.
pub fn unreadable_reason(path: &Path) -> Option<String> {
    let size = match std::fs::metadata(path) {
        Ok(meta) => meta.len(),
        Err(err) => return Some(format!("cannot be read: {}", os_error_text(&err, path))),
    };
    memoize(&CACHE, path, || match reason_uncached(path, size) {
        Ok(reason) => reason,
        Err(err) => Some(format!("cannot be read: {}", os_error_text(&err, path))),
    })
}

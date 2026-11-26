//! `PngImagePlugin`: chunk handlers, `info`, the decode check and the pixel decode.

use std::io;

use flate2::{Decompress, FlushDecompress, Status};
use raiw_container::ReadAt;

use super::value::{Info, InfoValue, float_pair, int_pair};
use super::{OpenError, SAFEBLOCK};
use crate::py::{latin1, utf8_strict};

const MAX_TEXT_CHUNK: usize = SAFEBLOCK;
const MAX_TEXT_MEMORY: usize = 64 * MAX_TEXT_CHUNK;

/// `IHDR` fields.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ihdr {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: u8,
    pub interlace: bool,
}

impl Ihdr {
    pub const fn channels(&self) -> usize {
        match self.color_type {
            2 => 3,
            4 => 2,
            6 => 4,
            _ => 1,
        }
    }

    pub const fn bits_per_pixel(&self) -> usize {
        self.channels() * self.bit_depth as usize
    }

    pub const fn row_bytes(&self, width: usize) -> usize {
        (width * self.bits_per_pixel()).div_ceil(8)
    }
}

/// `_MODES`: the Pillow mode of a bit depth and colour type.
pub fn mode_of(bit_depth: u8, color_type: u8) -> Option<&'static str> {
    Some(match (bit_depth, color_type) {
        (1, 0) => "1",
        (2 | 4 | 8, 0) => "L",
        (16, 0) => "I;16",
        (8 | 16, 2) => "RGB",
        (1 | 2 | 4 | 8, 3) => "P",
        (8, 4) => "LA",
        (16 | 8, 6) | (16, 4) => "RGBA",
        _ => return None,
    })
}

/// State after `_open`.
#[derive(Debug, Clone)]
pub struct PngOpen {
    pub info: Info,
    pub mode: &'static str,
    pub ihdr: Ihdr,
    pub palette: Option<Vec<u8>>,
    pub n_frames: usize,
    /// Offset of the first `IDAT` header.
    pub idat: u64,
    text_memory: usize,
    seq_num: Option<u32>,
    animated: bool,
}

enum ChunkError {
    /// `SyntaxError` and friends: the plugin rejects the file.
    Syntax(String),
    /// `ValueError` or `OSError` raised out of the handler.
    Raise(OpenError),
}

fn i32be(s: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([s[at], s[at + 1], s[at + 2], s[at + 3]])
}

fn i16be(s: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([s[at], s[at + 1]])
}

/// `_safe_zlib_decompress`: `Ok(None)` is `zlib.error`, `Err` the too-large `ValueError`.
fn safe_zlib_decompress(data: &[u8]) -> Result<Option<Vec<u8>>, ()> {
    let mut inflater = Decompress::new(true);
    let mut out = Vec::new();
    let mut buf = vec![0u8; (data.len() * 4).clamp(4096, 256 * 1024)];
    loop {
        let consumed = inflater.total_in() as usize;
        if out.len() >= MAX_TEXT_CHUNK {
            return if consumed < data.len() {
                Err(())
            } else {
                Ok(Some(out))
            };
        }
        let want = (MAX_TEXT_CHUNK - out.len()).min(buf.len());
        let before_out = inflater.total_out();
        let status = inflater.decompress(&data[consumed..], &mut buf[..want], FlushDecompress::None);
        let produced = (inflater.total_out() - before_out) as usize;
        out.extend_from_slice(&buf[..produced]);
        match status {
            Err(_) => return Ok(None),
            Ok(Status::StreamEnd) => return Ok(Some(out)),
            Ok(_) if produced == 0 && inflater.total_in() as usize == consumed => return Ok(Some(out)),
            Ok(_) => {}
        }
    }
}

impl PngOpen {
    fn text_memory(&mut self, chars: usize) -> Result<(), ChunkError> {
        self.text_memory += chars;
        if self.text_memory > MAX_TEXT_MEMORY {
            return Err(ChunkError::Raise(OpenError::Value(format!(
                "Too much memory used in text chunks: {}>MAX_TEXT_MEMORY",
                self.text_memory
            ))));
        }
        Ok(())
    }

    /// One chunk handler (`chunk_<cid>`); `Ok(false)` for an unknown chunk.
    fn handle(&mut self, cid: &[u8], s: &[u8]) -> Result<bool, ChunkError> {
        let truncated = |what: &str| ChunkError::Raise(OpenError::Value(format!("Truncated {what} chunk")));
        match cid {
            b"iCCP" => {
                // `s.find(b"\0")` is -1 without a NUL, so the method byte is then `s[0]`.
                let method_at = memchr::memchr(0, s).map_or(0, |i| i + 1);
                let comp_method = *s
                    .get(method_at)
                    .ok_or_else(|| ChunkError::Syntax("index out of range".into()))?;
                if comp_method != 0 {
                    return Err(ChunkError::Syntax(format!(
                        "Unknown compression method {comp_method} in iCCP chunk"
                    )));
                }
                let profile = match safe_zlib_decompress(s.get(method_at + 1..).unwrap_or_default()) {
                    Err(()) => return Err(too_large()),
                    Ok(profile) => profile,
                };
                self.info.insert(
                    "icc_profile".into(),
                    profile.map_or(InfoValue::None, InfoValue::Bytes),
                );
            }
            b"IHDR" => {
                if s.len() < 13 {
                    return Err(truncated("IHDR"));
                }
                self.ihdr = Ihdr {
                    width: i32be(s, 0),
                    height: i32be(s, 4),
                    bit_depth: s[8],
                    color_type: s[9],
                    interlace: s[12] != 0,
                };
                if let Some(mode) = mode_of(s[8], s[9]) {
                    self.mode = mode;
                }
                if s[12] != 0 {
                    self.info.insert("interlace".into(), InfoValue::Int(1));
                }
                if s[11] != 0 {
                    return Err(ChunkError::Syntax("unknown filter category".into()));
                }
            }
            b"PLTE" => {
                if self.mode == "P" {
                    self.palette = Some(s.to_vec());
                }
            }
            b"tRNS" => self.trns(s)?,
            b"gAMA" => {
                if s.len() < 4 {
                    return Err(truncated("gAMA"));
                }
                self.info.insert(
                    "gamma".into(),
                    InfoValue::Float(f64::from(i32be(s, 0)) / 100_000.0),
                );
            }
            b"cHRM" => {
                if s.len() < 32 {
                    return Err(truncated("cHRM"));
                }
                let values = (0..8)
                    .map(|k| InfoValue::Float(f64::from(i32be(s, 4 * k)) / 100_000.0))
                    .collect();
                self.info.insert("chromaticity".into(), InfoValue::Tuple(values));
            }
            b"sRGB" => {
                let Some(&intent) = s.first() else {
                    return Err(truncated("sRGB"));
                };
                self.info.insert("srgb".into(), InfoValue::Int(i64::from(intent)));
            }
            b"pHYs" => {
                if s.len() < 9 {
                    return Err(truncated("pHYs"));
                }
                let (px, py) = (i32be(s, 0), i32be(s, 4));
                match s[8] {
                    1 => {
                        let dpi = float_pair(f64::from(px) * 0.0254, f64::from(py) * 0.0254);
                        self.info.insert("dpi".into(), dpi);
                    }
                    0 => {
                        self.info
                            .insert("aspect".into(), int_pair(i64::from(px), i64::from(py)));
                    }
                    _ => {}
                }
            }
            b"tEXt" => {
                let (k, v) = split_nul(s);
                if !k.is_empty() {
                    let key = latin1(k);
                    let text = latin1(v);
                    let value = if k == b"exif" {
                        InfoValue::Bytes(v.to_vec())
                    } else {
                        InfoValue::Str(text.clone())
                    };
                    self.info.insert(key, value);
                    self.text_memory(text.chars().count())?;
                }
            }
            b"zTXt" => {
                let (k, v) = split_nul(s);
                let comp_method = v.first().copied().unwrap_or(0);
                if comp_method != 0 {
                    return Err(ChunkError::Syntax(format!(
                        "Unknown compression method {comp_method} in zTXt chunk"
                    )));
                }
                let body = v.get(1..).unwrap_or_default();
                let decoded = match safe_zlib_decompress(body) {
                    Err(()) => return Err(too_large()),
                    Ok(decoded) => decoded.unwrap_or_default(),
                };
                if !k.is_empty() {
                    let text = latin1(&decoded);
                    let chars = text.chars().count();
                    self.info.insert(latin1(k), InfoValue::Str(text));
                    self.text_memory(chars)?;
                }
            }
            b"iTXt" => self.itxt(s)?,
            b"eXIf" => {
                let mut exif = b"Exif\x00\x00".to_vec();
                exif.extend_from_slice(s);
                self.info.insert("exif".into(), InfoValue::Bytes(exif));
            }
            b"acTL" => {
                if s.len() < 8 {
                    return Err(ChunkError::Raise(OpenError::Value(
                        "APNG contains truncated acTL chunk".into(),
                    )));
                }
                if self.n_frames > 1 || self.animated {
                    self.animated = false;
                    return Ok(true);
                }
                let n = i32be(s, 0);
                if n == 0 || n > 0x8000_0000 {
                    return Ok(true);
                }
                self.n_frames = n as usize;
                self.animated = true;
                self.info
                    .insert("loop".into(), InfoValue::Int(i64::from(i32be(s, 4))));
            }
            b"fcTL" => {
                if s.len() < 26 {
                    return Err(ChunkError::Raise(OpenError::Value(
                        "APNG contains truncated fcTL chunk".into(),
                    )));
                }
                let seq = i32be(s, 0);
                if (self.seq_num.is_none() && seq != 0) || self.seq_num.is_some_and(|n| n + 1 != seq) {
                    return Err(ChunkError::Syntax("APNG contains frame sequence errors".into()));
                }
                self.seq_num = Some(seq);
                let (w, h, px, py) = (i32be(s, 4), i32be(s, 8), i32be(s, 12), i32be(s, 16));
                if u64::from(px) + u64::from(w) > u64::from(self.ihdr.width)
                    || u64::from(py) + u64::from(h) > u64::from(self.ihdr.height)
                {
                    return Err(ChunkError::Syntax("APNG contains invalid frames".into()));
                }
                let bbox = [px, py, px + w, py + h]
                    .map(|v| InfoValue::Int(i64::from(v)))
                    .to_vec();
                self.info.insert("bbox".into(), InfoValue::Tuple(bbox));
                let num = f64::from(i16be(s, 20));
                let den = match i16be(s, 22) {
                    0 => 100.0,
                    d => f64::from(d),
                };
                self.info
                    .insert("duration".into(), InfoValue::Float(num / den * 1000.0));
                self.info
                    .insert("disposal".into(), InfoValue::Int(i64::from(s[24])));
                self.info.insert("blend".into(), InfoValue::Int(i64::from(s[25])));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn trns(&mut self, s: &[u8]) -> Result<(), ChunkError> {
        match self.mode {
            "P" => {
                let zeros = memchr::memchr_iter(0, s).count();
                let simple = zeros == 1 && s.iter().all(|&b| b == 0 || b == 0xff);
                if simple {
                    if let Some(i) = memchr::memchr(0, s) {
                        self.info.insert("transparency".into(), InfoValue::Int(i as i64));
                    }
                } else {
                    self.info
                        .insert("transparency".into(), InfoValue::Bytes(s.to_vec()));
                }
            }
            "1" | "L" | "I;16" | "RGB" => {
                let need = if self.mode == "RGB" { 6 } else { 2 };
                if s.len() < need {
                    return Err(ChunkError::Raise(OpenError::Value("Truncated tRNS chunk".into())));
                }
                let value = match self.mode {
                    "1" => InfoValue::Int(if i16be(s, 0) != 0 { 255 } else { 0 }),
                    "RGB" => InfoValue::Tuple(
                        [0, 2, 4]
                            .map(|at| InfoValue::Int(i64::from(i16be(s, at))))
                            .to_vec(),
                    ),
                    _ => InfoValue::Int(i64::from(i16be(s, 0))),
                };
                self.info.insert("transparency".into(), value);
            }
            _ => {}
        }
        Ok(())
    }

    fn itxt(&mut self, s: &[u8]) -> Result<(), ChunkError> {
        let Some(nul) = memchr::memchr(0, s) else {
            return Ok(());
        };
        let (k, r) = (&s[..nul], &s[nul + 1..]);
        if r.len() < 2 {
            return Ok(());
        }
        let (cf, cm, r) = (r[0], r[1], &r[2..]);
        let mut parts = r.splitn(3, |&b| b == 0);
        let (Some(lang), Some(tk), Some(v)) = (parts.next(), parts.next(), parts.next()) else {
            return Ok(());
        };
        let value = if cf != 0 {
            if cm != 0 {
                return Ok(());
            }
            match safe_zlib_decompress(v) {
                Err(()) => return Err(too_large()),
                Ok(None) => return Ok(()),
                Ok(Some(decoded)) => decoded,
            }
        } else {
            v.to_vec()
        };
        if k == b"XML:com.adobe.xmp" {
            self.info.insert("xmp".into(), InfoValue::Bytes(value.clone()));
        }
        let (Some(lang), Some(tkey), Some(text)) = (utf8_strict(lang), utf8_strict(tk), utf8_strict(&value))
        else {
            return Ok(());
        };
        let chars = text.chars().count();
        self.info.insert(
            latin1(k),
            InfoValue::IText {
                text: text.to_owned(),
                lang: lang.to_owned(),
                tkey: tkey.to_owned(),
            },
        );
        self.text_memory(chars)
    }
}

fn too_large() -> ChunkError {
    ChunkError::Raise(OpenError::Value(
        "Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK".into(),
    ))
}

fn split_nul(s: &[u8]) -> (&[u8], &[u8]) {
    memchr::memchr(0, s).map_or((s, &[][..]), |i| (&s[..i], &s[i + 1..]))
}

fn is_cid(cid: &[u8]) -> bool {
    cid.len() == 4 && cid.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'_')
}

/// `_safe_read`: exactly `n` bytes or `OSError("Truncated File Read")`.
fn safe_read<R: ReadAt + ?Sized>(src: &R, pos: u64, n: u64) -> Result<Vec<u8>, OpenError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let data = src.read_vec(pos, n).map_err(|e| OpenError::Io(e.to_string()))?;
    if (data.len() as u64) < n {
        return Err(OpenError::Io("Truncated File Read".into()));
    }
    Ok(data)
}

fn read_exact_or_short<R: ReadAt + ?Sized>(src: &R, pos: u64, n: u64) -> io::Result<Vec<u8>> {
    src.read_vec(pos, n)
}

/// `PngImageFile._open`.
pub fn open<R: ReadAt + ?Sized>(src: &R) -> Result<PngOpen, OpenError> {
    let unidentified = |_: String| OpenError::Unidentified;
    let mut state = PngOpen {
        info: Info::new(),
        mode: "",
        ihdr: Ihdr::default(),
        palette: None,
        n_frames: 1,
        idat: 0,
        text_memory: 0,
        seq_num: None,
        animated: false,
    };
    let mut pos = 8u64;
    loop {
        let header = read_exact_or_short(src, pos, 8).map_err(|e| OpenError::Io(e.to_string()))?;
        if header.len() < 4 {
            return Err(OpenError::Unidentified);
        }
        let length = u64::from(i32be(&header, 0));
        let cid = &header[4..];
        if !is_cid(cid) {
            return Err(unidentified(format!("broken PNG file (chunk {cid:?})")));
        }
        let data_pos = pos + 8;
        if matches!(cid, b"IDAT" | b"IEND" | b"fdAT") {
            if cid == b"IDAT" && state.animated {
                state.info.insert("default_image".into(), InfoValue::Bool(true));
            }
            state.idat = pos;
            break;
        }
        let s = safe_read(src, data_pos, length)?;
        match state.handle(cid, &s) {
            Ok(_) => {}
            Err(ChunkError::Syntax(msg)) => return Err(unidentified(msg)),
            Err(ChunkError::Raise(err)) => return Err(err),
        }
        let crc = read_exact_or_short(src, data_pos + length, 4).map_err(|e| OpenError::Io(e.to_string()))?;
        if crc.len() < 4 {
            return Err(OpenError::Unidentified);
        }
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(cid);
        hasher.update(&s);
        if hasher.finalize() != i32be(&crc, 0) {
            return Err(OpenError::Unidentified);
        }
        pos = data_pos + length + 4;
    }
    if state.mode.is_empty() || state.ihdr.width == 0 || state.ihdr.height == 0 {
        return Err(OpenError::Unidentified);
    }
    if state.animated {
        state.n_frames += usize::from(state.info.contains_key("default_image"));
    }
    Ok(state)
}

/// Outcome of the IDAT decode: where the stream ended and how.
#[derive(Debug, Clone)]
pub struct Decoded {
    /// Pillow's error message, `None` for a complete decode.
    pub error: Option<String>,
    /// Whether `load_end` ran (it runs after a broken stream, not after a truncation).
    pub ended: bool,
    /// The raw filtered scanlines when requested.
    pub raw: Option<Vec<u8>>,
}

fn expected_raw_len(ihdr: &Ihdr) -> usize {
    let (w, h) = (ihdr.width as usize, ihdr.height as usize);
    if !ihdr.interlace {
        return h * (1 + ihdr.row_bytes(w));
    }
    adam7_passes(w, h)
        .iter()
        .filter(|(pw, ph)| *pw > 0 && *ph > 0)
        .map(|(pw, ph)| ph * (1 + ihdr.row_bytes(*pw)))
        .sum()
}

/// Adam7 passes as `(x0, y0, dx, dy)`.
const ADAM7: [(usize, usize, usize, usize); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

/// Pass sizes of the Adam7 interlace.
pub fn adam7_passes(w: usize, h: usize) -> [(usize, usize); 7] {
    ADAM7.map(|(x0, y0, dx, dy)| ((w + dx - 1 - x0) / dx, (h + dy - 1 - y0) / dy))
}

/// Decode the IDAT stream the way `ImageFile.load` drives `ZipDecode`.
pub fn decode<R: ReadAt + ?Sized>(src: &R, open: &PngOpen, keep: bool) -> Decoded {
    let expected = expected_raw_len(&open.ihdr);
    let row_len = 1 + open.ihdr.row_bytes(open.ihdr.width as usize);
    let mut inflater = Decompress::new(true);
    let mut raw = if keep {
        Vec::with_capacity(expected)
    } else {
        Vec::new()
    };
    let mut scratch = vec![0u8; 256 * 1024];
    let mut produced = 0usize;
    let mut pos = open.idat;
    let truncated = |raw: Vec<u8>| Decoded {
        error: Some("image file is truncated".into()),
        ended: false,
        raw: keep.then_some(raw),
    };
    loop {
        let Ok(header) = src.read_vec(pos, 8) else {
            return truncated(raw);
        };
        if header.len() < 8 {
            return truncated(raw);
        }
        let length = u64::from(i32be(&header, 0));
        if &header[4..8] != b"IDAT" {
            return Decoded {
                error: Some("image file is truncated (0 bytes not processed)".into()),
                ended: false,
                raw: keep.then_some(raw),
            };
        }
        let Ok(payload) = src.read_vec(pos + 8, length) else {
            return truncated(raw);
        };
        let mut offset = 0usize;
        while offset < payload.len() {
            let before_in = inflater.total_in();
            let before_out = inflater.total_out();
            let status = inflater.decompress(&payload[offset..], &mut scratch, FlushDecompress::None);
            let consumed = (inflater.total_in() - before_in) as usize;
            let out = (inflater.total_out() - before_out) as usize;
            if !open.ihdr.interlace {
                let start = produced;
                for (i, &byte) in scratch[..out].iter().enumerate() {
                    if (start + i).is_multiple_of(row_len) && byte > 4 && start + i < expected {
                        return Decoded {
                            error: Some("unrecognized data stream contents when reading image file".into()),
                            ended: true,
                            raw: keep.then_some(raw),
                        };
                    }
                }
            }
            if keep {
                let take = out.min(expected.saturating_sub(produced));
                raw.extend_from_slice(&scratch[..take]);
            }
            produced += out;
            offset += consumed;
            if produced >= expected {
                return Decoded {
                    error: None,
                    ended: true,
                    raw: keep.then_some(raw),
                };
            }
            match status {
                Err(_) => {
                    return Decoded {
                        error: Some("broken data stream when reading image file".into()),
                        ended: true,
                        raw: keep.then_some(raw),
                    };
                }
                Ok(Status::StreamEnd) => {
                    return Decoded {
                        error: Some("image file is truncated (0 bytes not processed)".into()),
                        ended: false,
                        raw: keep.then_some(raw),
                    };
                }
                Ok(_) if consumed == 0 && out == 0 => break,
                Ok(_) => {}
            }
        }
        if (payload.len() as u64) < length {
            return truncated(raw);
        }
        pos += 8 + length + 4;
    }
}

/// `load_end`: the handlers of every chunk after the image data, into a copy of `info`.
///
/// A handler that raises leaves the chunks before it applied, as in Python.
pub fn load_end<R: ReadAt + ?Sized>(src: &R, open: &PngOpen) -> (Info, Option<OpenError>) {
    let mut state = open.clone();
    let error = load_end_into(src, &mut state).err();
    (state.info, error)
}

fn load_end_into<R: ReadAt + ?Sized>(src: &R, state: &mut PngOpen) -> Result<(), OpenError> {
    let open = state.clone();
    let mut pos = open.idat;
    let mut in_image = true;
    while let Ok(header) = src.read_vec(pos, 8) {
        if header.len() < 8 || !is_cid(&header[4..8]) {
            break;
        }
        let length = u64::from(i32be(&header, 0));
        let cid = header[4..8].to_vec();
        let data_pos = pos + 8;
        pos = data_pos + length + 4;
        if in_image && cid == b"IDAT" {
            continue;
        }
        in_image = false;
        if cid == b"IEND" {
            break;
        }
        if cid == b"IDAT" || cid == b"fdAT" {
            let skip = if cid == b"fdAT" {
                length.saturating_sub(4)
            } else {
                length
            };
            safe_read(src, data_pos, skip)?;
            continue;
        }
        let s = safe_read(src, data_pos, length)?;
        match state.handle(&cid, &s) {
            Ok(_) => {}
            Err(ChunkError::Syntax(msg)) => return Err(OpenError::Syntax(msg)),
            Err(ChunkError::Raise(err)) => return Err(err),
        }
    }
    Ok(())
}

/// Unfilter (and de-interlace) raw scanlines into packed rows of `row_bytes` each.
pub fn unfilter(ihdr: &Ihdr, raw: &[u8]) -> Option<Vec<u8>> {
    let (w, h) = (ihdr.width as usize, ihdr.height as usize);
    let bpp = ihdr.bits_per_pixel().div_ceil(8);
    if !ihdr.interlace {
        return unfilter_pass(raw, ihdr.row_bytes(w), h, bpp);
    }
    let bits = ihdr.bits_per_pixel();
    let row_bytes = ihdr.row_bytes(w);
    let mut out = vec![0u8; row_bytes * h];
    let mut offset = 0;
    for ((pw, ph), (x0, y0, dx, dy)) in adam7_passes(w, h).into_iter().zip(ADAM7) {
        if pw == 0 || ph == 0 {
            continue;
        }
        let pass_row = ihdr.row_bytes(pw);
        let len = ph * (1 + pass_row);
        let rows = unfilter_pass(raw.get(offset..offset + len)?, pass_row, ph, bpp)?;
        offset += len;
        for py in 0..ph {
            for px in 0..pw {
                let (x, y) = (x0 + px * dx, y0 + py * dy);
                copy_pixel(&rows[py * pass_row..], px, &mut out[y * row_bytes..], x, bits);
            }
        }
    }
    Some(out)
}

fn copy_pixel(src: &[u8], sx: usize, dst: &mut [u8], dx: usize, bits: usize) {
    if bits >= 8 {
        let n = bits / 8;
        dst[dx * n..dx * n + n].copy_from_slice(&src[sx * n..sx * n + n]);
    } else {
        let per = 8 / bits;
        let mask = ((1u16 << bits) - 1) as u8;
        let sshift = 8 - bits * (sx % per + 1);
        let value = (src[sx / per] >> sshift) & mask;
        let dshift = 8 - bits * (dx % per + 1);
        dst[dx / per] = (dst[dx / per] & !(mask << dshift)) | (value << dshift);
    }
}

fn unfilter_pass(raw: &[u8], row_bytes: usize, rows: usize, bpp: usize) -> Option<Vec<u8>> {
    let mut out = vec![0u8; row_bytes * rows];
    let mut prev = vec![0u8; row_bytes];
    for y in 0..rows {
        let line = raw.get(y * (row_bytes + 1)..(y + 1) * (row_bytes + 1))?;
        let (kind, data) = (line[0], &line[1..]);
        let cur = &mut out[y * row_bytes..(y + 1) * row_bytes];
        for i in 0..row_bytes {
            let a = if i >= bpp { cur[i - bpp] } else { 0 };
            let b = prev[i];
            let c = if i >= bpp { prev[i - bpp] } else { 0 };
            let predictor = match kind {
                0 => 0,
                1 => a,
                2 => b,
                3 => u16::midpoint(u16::from(a), u16::from(b)) as u8,
                4 => {
                    let p = i16::from(a) + i16::from(b) - i16::from(c);
                    let (pa, pb, pc) = (
                        (p - i16::from(a)).abs(),
                        (p - i16::from(b)).abs(),
                        (p - i16::from(c)).abs(),
                    );
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    }
                }
                _ => return None,
            };
            cur[i] = data[i].wrapping_add(predictor);
        }
        prev.copy_from_slice(cur);
    }
    Some(out)
}

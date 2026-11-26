//! `JpegImagePlugin` / `MpoImagePlugin`: marker handlers and `info`.

use super::OpenError;
use super::ifd;
use super::value::{Info, InfoValue, float_pair, int_pair};

/// State after `_open` (and the MPO adoption).
#[derive(Debug, Clone)]
pub struct JpegOpen {
    pub info: Info,
    pub mode: &'static str,
    pub size: (u32, u32),
    pub format: &'static str,
    pub n_frames: usize,
    /// `(segment name, payload)` of every APPn and COM segment.
    pub applist: Vec<(String, Vec<u8>)>,
    /// Offset just after the SOS header, where the coded scan starts; `None` without SOS.
    pub scan_start: Option<usize>,
}

fn i16be(s: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*s.get(at)?, *s.get(at + 1)?]))
}

fn i32be(s: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes([
        *s.get(at)?,
        *s.get(at + 1)?,
        *s.get(at + 2)?,
        *s.get(at + 3)?,
    ]))
}

struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
    st: JpegOpen,
    icclist: Vec<Vec<u8>>,
}

impl Parser<'_> {
    fn read(&mut self, n: usize) -> &[u8] {
        let start = self.pos.min(self.data.len());
        let end = start.saturating_add(n).min(self.data.len());
        self.pos = end;
        &self.data[start..end]
    }

    /// The `n = i16(fp.read(2)) - 2; s = _safe_read(fp, n)` prologue of every handler.
    fn segment(&mut self) -> Result<Vec<u8>, OpenError> {
        let len = self.read(2).to_vec();
        let n = i64::from(i16be(&len, 0).ok_or(OpenError::Unidentified)?) - 2;
        if n <= 0 {
            return Ok(Vec::new());
        }
        let s = self.read(n as usize).to_vec();
        if (s.len() as i64) < n {
            return Err(OpenError::Io("Truncated File Read".into()));
        }
        Ok(s)
    }

    fn app(&mut self, marker: u16) -> Result<(), OpenError> {
        let s = self.segment()?;
        let name = format!("APP{}", marker & 15);
        self.st.applist.push((name, s.clone()));
        let info = &mut self.st.info;
        if marker == 0xFFE0 && s.starts_with(b"JFIF") {
            let version = i16be(&s, 5).ok_or(OpenError::Unidentified)?;
            info.insert("jfif".into(), InfoValue::Int(i64::from(version)));
            info.insert(
                "jfif_version".into(),
                int_pair(i64::from(version / 256), i64::from(version % 256)),
            );
            if let (Some(&unit), Some(x), Some(y)) = (s.get(7), i16be(&s, 8), i16be(&s, 10)) {
                if unit == 1 {
                    info.insert("dpi".into(), int_pair(i64::from(x), i64::from(y)));
                } else if unit == 2 {
                    info.insert("dpi".into(), float_pair(f64::from(x) * 2.54, f64::from(y) * 2.54));
                }
                info.insert("jfif_unit".into(), InfoValue::Int(i64::from(unit)));
                info.insert("jfif_density".into(), int_pair(i64::from(x), i64::from(y)));
            }
        } else if marker == 0xFFE1 && s.starts_with(b"Exif\x00\x00") {
            if let Some(InfoValue::Bytes(exif)) = info.get_mut("exif") {
                exif.extend_from_slice(&s[6..]);
            } else {
                info.insert("exif".into(), InfoValue::Bytes(s));
            }
        } else if marker == 0xFFE1 && s.starts_with(b"http://ns.adobe.com/xap/1.0/\x00") {
            let nul = memchr::memchr(0, &s).unwrap_or(s.len() - 1);
            info.insert("xmp".into(), InfoValue::Bytes(s[nul + 1..].to_vec()));
        } else if marker == 0xFFE2 && s.starts_with(b"FPXR\x00") {
            info.insert("flashpix".into(), InfoValue::Bytes(s));
        } else if marker == 0xFFE2 && s.starts_with(b"ICC_PROFILE\x00") {
            self.icclist.push(s);
        } else if marker == 0xFFED && s.starts_with(b"Photoshop 3.0\x00") {
            photoshop(info, &s)?;
        } else if marker == 0xFFEE && s.starts_with(b"Adobe") {
            let version = i16be(&s, 5).ok_or(OpenError::Unidentified)?;
            info.insert("adobe".into(), InfoValue::Int(i64::from(version)));
            if let Some(&transform) = s.get(11) {
                info.insert("adobe_transform".into(), InfoValue::Int(i64::from(transform)));
            }
        } else if marker == 0xFFE2 && s.starts_with(b"MPF\x00") {
            info.insert("mp".into(), InfoValue::Bytes(s[4..].to_vec()));
            let mpoffset = self.pos as i64 - s.len() as i64 + 4;
            info.insert("mpoffset".into(), InfoValue::Int(mpoffset));
        }
        Ok(())
    }

    fn sof(&mut self, marker: u16) -> Result<(), OpenError> {
        let s = self.segment()?;
        let (Some(h), Some(w)) = (i16be(&s, 1), i16be(&s, 3)) else {
            return Err(OpenError::Unidentified);
        };
        self.st.size = (u32::from(w), u32::from(h));
        if s[0] != 8 {
            return Err(OpenError::Unidentified);
        }
        self.st.mode = match s.get(5) {
            Some(1) => "L",
            Some(3) => "RGB",
            Some(4) => "CMYK",
            _ => return Err(OpenError::Unidentified),
        };
        if matches!(marker, 0xFFC2 | 0xFFC6 | 0xFFCA | 0xFFCE) {
            self.st.info.insert("progressive".into(), InfoValue::Int(1));
            self.st.info.insert("progression".into(), InfoValue::Int(1));
        }
        if !self.icclist.is_empty() {
            let mut list = std::mem::take(&mut self.icclist);
            list.sort();
            let count = *list[0].get(13).ok_or(OpenError::Unidentified)?;
            let value = if usize::from(count) == list.len() {
                InfoValue::Bytes(list.iter().flat_map(|p| p[14.min(p.len())..].to_vec()).collect())
            } else {
                InfoValue::None
            };
            self.st.info.insert("icc_profile".into(), value);
        }
        Ok(())
    }

    fn dqt(&mut self) -> Result<(), OpenError> {
        let mut s = &self.segment()?[..];
        while let Some(&v) = s.first() {
            let precision = if v / 16 == 0 { 1 } else { 2 };
            let qt_length = 1 + precision * 64;
            if s.len() < qt_length {
                return Err(OpenError::Unidentified);
            }
            s = &s[qt_length..];
        }
        Ok(())
    }
}

fn photoshop(info: &mut Info, s: &[u8]) -> Result<(), OpenError> {
    let slot = info
        .entry("photoshop".to_owned())
        .or_insert_with(|| InfoValue::Dict(Vec::new()));
    let mut entries = match std::mem::replace(slot, InfoValue::None) {
        InfoValue::Dict(entries) => entries,
        _ => Vec::new(),
    };
    let mut offset = 14;
    let result = (|| {
        while s.get(offset..offset + 4) == Some(b"8BIM") {
            offset += 4;
            let Some(code) = i16be(s, offset) else {
                return Ok(());
            };
            offset += 2;
            let name_len = usize::from(*s.get(offset).ok_or(OpenError::Unidentified)?);
            offset += 1 + name_len;
            offset += offset & 1;
            let Some(size) = i32be(s, offset) else {
                return Ok(());
            };
            offset += 4;
            let data = s
                .get(offset..(offset + size as usize).min(s.len()))
                .unwrap_or_default();
            let value = if code == 0x03ED {
                let (Some(xr), Some(ux), Some(yr), Some(uy)) =
                    (i32be(data, 0), i16be(data, 4), i32be(data, 8), i16be(data, 12))
                else {
                    return Ok(());
                };
                InfoValue::Dict(vec![
                    (
                        InfoValue::Str("XResolution".into()),
                        InfoValue::Float(f64::from(xr) / 65536.0),
                    ),
                    (
                        InfoValue::Str("DisplayedUnitsX".into()),
                        InfoValue::Int(i64::from(ux)),
                    ),
                    (
                        InfoValue::Str("YResolution".into()),
                        InfoValue::Float(f64::from(yr) / 65536.0),
                    ),
                    (
                        InfoValue::Str("DisplayedUnitsY".into()),
                        InfoValue::Int(i64::from(uy)),
                    ),
                ])
            } else {
                InfoValue::Bytes(data.to_vec())
            };
            let key = InfoValue::Int(i64::from(code));
            if let Some(slot) = entries.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = value;
            } else {
                entries.push((key, value));
            }
            offset += size as usize;
            offset += offset & 1;
        }
        Ok(())
    })();
    info.insert("photoshop".into(), InfoValue::Dict(entries));
    result
}

/// Pillow's `JpegImageFile._open` over the file bytes.
pub fn open(data: &[u8]) -> Result<JpegOpen, OpenError> {
    if !data.starts_with(b"\xff\xd8\xff") {
        return Err(OpenError::Unidentified);
    }
    let mut p = Parser {
        data,
        pos: 3,
        st: JpegOpen {
            info: Info::new(),
            mode: "",
            size: (0, 0),
            format: "JPEG",
            n_frames: 1,
            applist: Vec::new(),
            scan_start: None,
        },
        icclist: Vec::new(),
    };
    let mut s: Vec<u8> = vec![0xFF];
    while !s.is_empty() {
        if s[0] != 0xFF {
            s = p.read(1).to_vec();
            continue;
        }
        s.extend_from_slice(p.read(1));
        if s.len() < 2 {
            break;
        }
        let marker = u16::from_be_bytes([s[0], s[1]]);
        match marker {
            0xFFE0..=0xFFEF => p.app(marker)?,
            0xFFFE => {
                let body = p.segment()?;
                p.st.info.insert("comment".into(), InfoValue::Bytes(body.clone()));
                p.st.applist.push(("COM".into(), body));
            }
            0xFFC0..=0xFFC3 | 0xFFC5..=0xFFC7 | 0xFFC9..=0xFFCB | 0xFFCD..=0xFFCF | 0xFFDE => {
                p.sof(marker)?;
            }
            0xFFDB => p.dqt()?,
            0xFFC4 | 0xFFCC | 0xFFDA | 0xFFDC | 0xFFDD | 0xFFDF => {
                p.segment()?;
                if marker == 0xFFDA {
                    p.st.scan_start = Some(p.pos);
                    break;
                }
            }
            0xFFC8 | 0xFFD0..=0xFFD9 | 0xFFF0..=0xFFFD => {}
            0x0000 | 0xFFFF => {
                s = vec![0xFF];
                continue;
            }
            0xFF00 => {
                s = p.read(1).to_vec();
                continue;
            }
            _ => return Err(OpenError::Unidentified),
        }
        s = p.read(1).to_vec();
    }
    if p.st.mode.is_empty() || p.st.size.0 == 0 || p.st.size.1 == 0 {
        return Err(OpenError::Unidentified);
    }
    let mut st = p.st;
    read_dpi_from_exif(&mut st.info);
    adopt_mpo(&mut st)?;
    Ok(st)
}

/// `_read_dpi_from_exif`.
fn read_dpi_from_exif(info: &mut Info) {
    if info.contains_key("dpi") {
        return;
    }
    let Some(InfoValue::Bytes(exif)) = info.get("exif") else {
        return;
    };
    let mut tiff = &exif[..];
    while let Some(rest) = tiff.strip_prefix(b"Exif\x00\x00") {
        tiff = rest;
    }
    let dpi = (|| {
        let (dir, little) = ifd::first_ifd(tiff)?;
        let unit = ifd::single_int(dir.get(&0x0128)?, little)?;
        let x = dir.get(&0x011A)?;
        let dpi = match ifd::single_rational(x, little) {
            Some((_, 0)) => return None,
            Some((n, d)) => n as f64 / d as f64,
            None => ifd::single_int(x, little)? as f64,
        };
        Some(if unit == 3 { dpi * 2.54 } else { dpi })
    })();
    let value = dpi.map_or_else(|| int_pair(72, 72), |dpi| float_pair(dpi, dpi));
    info.insert("dpi".into(), value);
}

/// `jpeg_factory`: become an MPO when the MP index lists more than one image.
fn adopt_mpo(st: &mut JpegOpen) -> Result<(), OpenError> {
    let Some(InfoValue::Bytes(mp)) = st.info.get("mp") else {
        return Ok(());
    };
    let Some(little) = ifd::header_endian(mp) else {
        return Ok(());
    };
    let Some(next) = ifd::u32_at(mp, 4, little) else {
        return Ok(());
    };
    let dir = ifd::load(mp, next as usize, little);
    let Some(quant) = dir.get(&0xB001).and_then(|entry| ifd::single_int(entry, little)) else {
        return Ok(());
    };
    let Some((_, raw)) = dir.get(&0xB002) else {
        return Ok(());
    };
    for entry in 0..quant as usize {
        let Some(attribute) = ifd::u32_at(raw, entry * 16, little) else {
            return Err(OpenError::Unidentified);
        };
        if raw.len() < entry * 16 + 16 {
            return Err(OpenError::Unidentified);
        }
        if (attribute >> 24) & 7 != 0 {
            return Ok(());
        }
    }
    let hdr = st.applist.iter().any(|(name, content)| {
        name == "APP1" && memchr::memmem::find(content, b" hdrgm:Version=\"").is_some()
    });
    if quant > 1 && !hdr {
        st.format = "MPO";
        st.n_frames = quant as usize;
        st.info.shift_remove("mpoffset");
    }
    Ok(())
}

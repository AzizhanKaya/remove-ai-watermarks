//! `GifImagePlugin`: screen descriptor, first-frame extensions and `info`.

use super::OpenError;
use super::value::{Info, InfoValue};

#[derive(Debug, Clone)]
pub struct GifOpen {
    pub info: Info,
    pub mode: &'static str,
    pub size: (u32, u32),
    /// Offset of the first image's LZW data (after the minimum-code-size byte).
    pub frame_offset: usize,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn read(&mut self, n: usize) -> &[u8] {
        let start = self.pos.min(self.data.len());
        let end = start.saturating_add(n).min(self.data.len());
        self.pos = end;
        &self.data[start..end]
    }

    /// `GifImageFile.data`: one sub-block, `None` at a terminator or EOF.
    fn sub_block(&mut self) -> Option<Vec<u8>> {
        let len = *self.read(1).first()?;
        if len == 0 {
            return None;
        }
        Some(self.read(usize::from(len)).to_vec())
    }
}

fn palette_needed(p: &[u8]) -> Result<bool, OpenError> {
    for (i, rgb) in p.chunks(3).enumerate() {
        if rgb.len() < 3 {
            return Err(OpenError::Unidentified);
        }
        if !(i == usize::from(rgb[0]) && rgb[0] == rgb[1] && rgb[1] == rgb[2]) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn le16(d: &[u8], at: usize) -> Result<u16, OpenError> {
    d.get(at..at + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .ok_or(OpenError::Unidentified)
}

/// `GifImageFile._open` through `_seek(0)`.
pub fn open(data: &[u8]) -> Result<GifOpen, OpenError> {
    let mut c = Cursor { data, pos: 0 };
    let s = c.read(13).to_vec();
    if !matches!(s.get(..6), Some(b"GIF87a" | b"GIF89a")) {
        return Err(OpenError::Unidentified);
    }
    let mut info = Info::new();
    info.insert("version".into(), InfoValue::Bytes(s[..6].to_vec()));
    let size = (u32::from(le16(&s, 6)?), u32::from(le16(&s, 8)?));
    let flags = *s.get(10).ok_or(OpenError::Unidentified)?;
    let mut global = false;
    if flags & 128 != 0 {
        info.insert(
            "background".into(),
            InfoValue::Int(i64::from(*s.get(11).ok_or(OpenError::Unidentified)?)),
        );
        let bits = (flags & 7) + 1;
        let p = c.read(3 << bits).to_vec();
        global = palette_needed(&p)?;
    }
    let mut comment: Vec<u8> = Vec::new();
    let mut has_comment = false;
    let mut duration = None;
    let mut extension = None;
    let mut transparency = None;
    let mut local = None;
    let mut found = false;
    let mut s = c.read(1).to_vec();
    if s.is_empty() || s == b";" {
        return Err(OpenError::Unidentified);
    }
    loop {
        if s.is_empty() {
            s = c.read(1).to_vec();
        }
        if s.is_empty() || s == b";" {
            break;
        }
        if s == b"!" {
            let label = c.read(1).first().copied();
            let block = c.sub_block();
            match (label, &block) {
                (Some(249), Some(block)) => {
                    let flags = *block.first().ok_or(OpenError::Unidentified)?;
                    if flags & 1 != 0 {
                        transparency = Some(*block.get(3).ok_or(OpenError::Unidentified)?);
                    }
                    duration = Some(i64::from(le16(block, 1)?) * 10);
                }
                (Some(254), _) => {
                    let mut block = block;
                    if block.is_some() && has_comment {
                        comment.push(b'\n');
                    }
                    while let Some(part) = block {
                        has_comment = true;
                        comment.extend_from_slice(&part);
                        block = c.sub_block();
                    }
                    s.clear();
                    continue;
                }
                (Some(255), Some(block)) => {
                    extension = Some((block.clone(), c.pos));
                    if block.starts_with(b"NETSCAPE2.0") {
                        let next = c.sub_block();
                        if let Some(next) = next
                            && next.len() >= 3
                            && next[0] == 1
                        {
                            info.insert("loop".into(), InfoValue::Int(i64::from(le16(&next, 1)?)));
                        }
                    }
                }
                (None, _) => return Err(OpenError::Unidentified),
                _ => {}
            }
            while c.sub_block().is_some() {}
        } else if s == b"," {
            let d = c.read(9).to_vec();
            le16(&d, 6)?;
            let flags = *d.get(8).ok_or(OpenError::Unidentified)?;
            if flags & 128 != 0 {
                let bits = (flags & 7) + 1;
                let p = c.read(3 << bits).to_vec();
                local = Some(palette_needed(&p)?);
            }
            if c.read(1).is_empty() {
                return Err(OpenError::Unidentified);
            }
            found = true;
            break;
        }
        s.clear();
    }
    if !found {
        return Err(OpenError::Unidentified);
    }
    let paletted = local.unwrap_or(global);
    if let Some(t) = transparency {
        info.insert("transparency".into(), InfoValue::Int(i64::from(t)));
    }
    if has_comment {
        info.insert("comment".into(), InfoValue::Bytes(comment));
    }
    if let Some(d) = duration {
        info.insert("duration".into(), InfoValue::Int(d));
    }
    if let Some((block, tell)) = extension {
        info.insert(
            "extension".into(),
            InfoValue::Tuple(vec![InfoValue::Bytes(block), InfoValue::Int(tell as i64)]),
        );
    }
    Ok(GifOpen {
        info,
        mode: if paletted { "P" } else { "L" },
        size,
        frame_offset: c.pos,
    })
}

/// `n_frames`: image descriptors reachable by walking the block chain.
pub fn count_frames(data: &[u8], start: usize) -> usize {
    let mut c = Cursor { data, pos: start };
    let mut frames = 1;
    while c.sub_block().is_some() {}
    loop {
        let s = c.read(1).to_vec();
        match s.first() {
            None | Some(b';') => return frames,
            Some(b'!') => {
                c.read(1);
                while c.sub_block().is_some() {}
            }
            Some(b',') => {
                let d = c.read(9).to_vec();
                let Some(&flags) = d.get(8) else { return frames };
                if flags & 128 != 0 {
                    c.read(3 << ((flags & 7) + 1));
                }
                if c.read(1).is_empty() {
                    return frames;
                }
                frames += 1;
                while c.sub_block().is_some() {}
            }
            _ => {}
        }
    }
}

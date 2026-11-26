//! `IptcImagePlugin.getiptcinfo` for JPEG Photoshop resources and TIFF IPTC tags.

use indexmap::IndexMap;

use super::{Image, InfoValue, Plugin};

/// One dataset value: bytes, `None` for an empty field, or a repeated list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IptcValue {
    None,
    Bytes(Vec<u8>),
    List(Vec<Option<Vec<u8>>>),
}

pub type IptcInfo = IndexMap<(u8, u8), IptcValue>;

/// Errors `getiptcinfo` lets escape (it swallows only `IndexError` / `KeyError`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IptcError {
    #[error("invalid IPTC/NAA file")]
    Syntax,
    #[error("illegal field length in IPTC/NAA file")]
    FieldLength,
    #[error("unpack requires a buffer of 2 bytes")]
    Struct,
}

fn parse(data: &[u8]) -> Result<IptcInfo, IptcError> {
    let mut info = IptcInfo::new();
    let mut pos = 0usize;
    let read = |pos: &mut usize, n: usize| {
        let start = (*pos).min(data.len());
        let end = start.saturating_add(n).min(data.len());
        *pos = end;
        &data[start..end]
    };
    loop {
        let s = read(&mut pos, 5).to_vec();
        if s.iter().all(|&b| b == 0) {
            break;
        }
        let (Some(&a), Some(&b)) = (s.get(1), s.get(2)) else {
            break;
        };
        let tag = (a, b);
        if s[0] != 0x1C || ![1, 2, 3, 4, 5, 6, 7, 8, 9, 240].contains(&tag.0) {
            return Err(IptcError::Syntax);
        }
        let Some(&size_byte) = s.get(3) else {
            break;
        };
        let size = if size_byte > 132 {
            return Err(IptcError::FieldLength);
        } else if size_byte == 128 {
            0
        } else if size_byte > 128 {
            let raw = read(&mut pos, usize::from(size_byte - 128)).to_vec();
            raw.iter().fold(0usize, |acc, &x| {
                acc.saturating_mul(256).saturating_add(usize::from(x))
            })
        } else {
            let Some(&low) = s.get(4) else {
                return Err(IptcError::Struct);
            };
            usize::from(u16::from_be_bytes([size_byte, low]))
        };
        if tag == (8, 10) {
            break;
        }
        let value = (size > 0).then(|| read(&mut pos, size).to_vec());
        match info.get_mut(&tag) {
            Some(IptcValue::List(items)) => items.push(value),
            Some(existing) => {
                let first = match std::mem::replace(existing, IptcValue::None) {
                    IptcValue::Bytes(b) => Some(b),
                    _ => None,
                };
                *existing = IptcValue::List(vec![first, value]);
            }
            None => {
                info.insert(tag, value.map_or(IptcValue::None, IptcValue::Bytes));
            }
        }
    }
    Ok(info)
}

/// `getiptcinfo(img)`: `Ok(None)` without an IPTC block.
pub fn getiptcinfo(image: &Image) -> Result<Option<IptcInfo>, IptcError> {
    let data = match &image.plugin {
        Plugin::Jpeg(_) => match image.info.get("photoshop") {
            Some(InfoValue::Dict(entries)) if !entries.is_empty() => entries
                .iter()
                .find(|(k, _)| *k == InfoValue::Int(0x0404))
                .and_then(|(_, v)| v.as_bytes().map(<[u8]>::to_vec)),
            _ => None,
        },
        Plugin::Tiff(dir, _) => dir.get(&33723).map(|(_, raw)| raw.clone()),
        _ => None,
    };
    data.map(|raw| parse(&raw)).transpose()
}

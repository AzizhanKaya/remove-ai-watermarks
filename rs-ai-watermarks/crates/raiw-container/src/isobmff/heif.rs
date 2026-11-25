//! HEIF/AVIF container transforms (`irot`/`imir` in `meta.iprp.ipco`).

use std::io;
use std::path::Path;

use super::boxes::{FourCc, iter_boxes, read_box_header};
use crate::source::{FileSource, ReadAt};

/// What a HEIF `meta` box's first child can be, used to tell the FullBox form from the bare one.
const HEIF_META_CHILD_TYPES: [&FourCc; 7] = [b"hdlr", b"pitm", b"iinf", b"iprp", b"iloc", b"iref", b"idat"];

/// Whether the first `meta` box declares a non-zero `irot` angle or any `imir`.
///
/// `ipma` association is not re-derived: a stray `irot` reads as transformed. Malformed or
/// missing structure, and any read error, read as "no transform".
pub fn heif_transform_applied(path: &Path) -> bool {
    FileSource::open(path)
        .and_then(|source| source_transform_applied(&source))
        .unwrap_or(false)
}

pub fn source_transform_applied<R: ReadAt + ?Sized>(source: &R) -> io::Result<bool> {
    let sniff = source.read_vec(0, 8)?;
    if sniff.len() < 8 || &sniff[4..8] != b"ftyp" {
        return Ok(false);
    }
    for meta in iter_boxes(source, 0, source.size()) {
        let meta = meta?;
        if &meta.kind != b"meta" {
            continue;
        }
        let mut children_start = None;
        for offset in [4, 0] {
            if let Some(first) = read_box_header(source, meta.payload + offset, meta.end)?
                && HEIF_META_CHILD_TYPES.contains(&&first.kind)
            {
                children_start = Some(meta.payload + offset);
                break;
            }
        }
        let Some(start) = children_start else {
            return Ok(false);
        };
        for iprp in iter_boxes(source, start, meta.end) {
            let iprp = iprp?;
            if &iprp.kind != b"iprp" {
                continue;
            }
            for ipco in iter_boxes(source, iprp.payload, iprp.end) {
                let ipco = ipco?;
                if &ipco.kind != b"ipco" {
                    continue;
                }
                for property in iter_boxes(source, ipco.payload, ipco.end) {
                    let property = property?;
                    if &property.kind == b"irot" {
                        // The angle byte is read even past an empty box, as Python's `f.read(1)` does.
                        if source
                            .read_array::<1>(property.payload)?
                            .is_some_and(|[angle]| angle & 0x03 != 0)
                        {
                            return Ok(true);
                        }
                    } else if &property.kind == b"imir" {
                        return Ok(true);
                    }
                }
            }
        }
        return Ok(false);
    }
    Ok(false)
}

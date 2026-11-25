//! Bounded `moov` metadata-list items (`udta.meta` ISO form and the bare QuickTime `meta`).
//!
//! Values are read only up to [`MAX_TC260_VALUE_BYTES`], so a corrupt size never allocates
//! more than the normative label could need; media payloads are never touched.

use std::collections::HashMap;
use std::io;

use super::boxes::{BoxEntry, FourCc, Span, boxes, iter_boxes, read_box_header};
use crate::markers::MAX_TC260_VALUE_BYTES;
use crate::source::ReadAt;

/// First-child types that identify which serialized form a `meta` box uses.
const META_CHILD_TYPES: [&FourCc; 3] = [b"hdlr", b"keys", b"ilst"];
const MAX_KEY_NAME_BYTES: u64 = 128;

/// One `ilst` data value with the spans a stripper may blank in place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataItem {
    pub item_type: FourCc,
    pub item_type_span: Span,
    /// The `keys` name mapped to this item (ISO form); `None` in a keyless QuickTime list.
    pub key_name: Option<Vec<u8>>,
    pub key_span: Option<Span>,
    pub value_start: u64,
    pub value_end: u64,
    pub value: Vec<u8>,
}

impl MetadataItem {
    pub const fn value_span(&self) -> Span {
        Span::new(self.value_start, self.value_end)
    }
}

type KeyIndex = HashMap<u64, (Vec<u8>, Span)>;

/// Map 1-based `keys` indices to their names; any malformed entry discards the whole table.
fn key_indices<R: ReadAt + ?Sized>(source: &R, payload: u64, end: u64) -> io::Result<KeyIndex> {
    let mut found = KeyIndex::new();
    if payload + 8 > end {
        return Ok(found);
    }
    let Some(prefix) = source.read_array::<8>(payload)? else {
        return Ok(found);
    };
    let entry_count = u32::from_be_bytes([prefix[4], prefix[5], prefix[6], prefix[7]]);
    let mut pos = payload + 8;
    for index in 1..=u64::from(entry_count) {
        if pos + 8 > end {
            return Ok(KeyIndex::new());
        }
        let Some(header) = source.read_array::<8>(pos)? else {
            return Ok(KeyIndex::new());
        };
        let entry_size = u64::from(u32::from_be_bytes([header[0], header[1], header[2], header[3]]));
        let entry_end = pos + entry_size;
        if entry_size < 8 || entry_end > end {
            return Ok(KeyIndex::new());
        }
        let name_start = pos + 8;
        let name_size = entry_end - name_start;
        if (1..=MAX_KEY_NAME_BYTES).contains(&name_size) {
            let name = source.read_vec(name_start, name_size)?;
            if name.len() as u64 == name_size {
                found.insert(index, (name, Span::new(name_start, entry_end)));
            }
        }
        pos = entry_end;
    }
    Ok(found)
}

/// Children of one `meta` box, probing the bare (offset 0) then the FullBox (offset 4) form.
fn meta_children<R: ReadAt + ?Sized>(source: &R, payload: u64, end: u64) -> io::Result<Vec<BoxEntry>> {
    for offset in [0, 4] {
        let start = payload + offset;
        if let Some(first) = read_box_header(source, start, end)?
            && META_CHILD_TYPES.contains(&&first.kind)
        {
            return boxes(source, start, end);
        }
    }
    Ok(Vec::new())
}

/// `(payload, end)` of every `meta` box that may hold a label: `moov.udta.meta` and `moov.meta`.
fn label_meta_boxes<R: ReadAt + ?Sized>(source: &R, moov: &BoxEntry) -> io::Result<Vec<Span>> {
    let mut found = Vec::new();
    for child in iter_boxes(source, moov.payload, moov.end) {
        let child = child?;
        if &child.kind == b"udta" {
            for grandchild in iter_boxes(source, child.payload, child.end) {
                let grandchild = grandchild?;
                if &grandchild.kind == b"meta" {
                    found.push(Span::new(grandchild.payload, grandchild.end));
                }
            }
        } else if &child.kind == b"meta" {
            found.push(Span::new(child.payload, child.end));
        }
    }
    Ok(found)
}

/// Bounded keyed or keyless metadata-list entries of one `moov` box.
pub fn moov_items<R: ReadAt + ?Sized>(source: &R, moov: &BoxEntry) -> io::Result<Vec<MetadataItem>> {
    let mut items = Vec::new();
    for meta in label_meta_boxes(source, moov)? {
        let mut keys = KeyIndex::new();
        let mut ilsts = Vec::new();
        let mut keyed = false;
        for child in meta_children(source, meta.start, meta.end)? {
            if &child.kind == b"keys" {
                keyed = true;
                keys.extend(key_indices(source, child.payload, child.end)?);
            } else if &child.kind == b"ilst" {
                ilsts.push(child);
            }
        }
        for ilst in ilsts {
            for item in iter_boxes(source, ilst.payload, ilst.end) {
                let item = item?;
                let key = keys.get(&u64::from(u32::from_be_bytes(item.kind)));
                if keyed && key.is_none() {
                    // A keyed list maps items through `keys`; an unmapped index is not a label.
                    continue;
                }
                read_item_values(source, &item, key, &mut items)?;
            }
        }
    }
    Ok(items)
}

fn read_item_values<R: ReadAt + ?Sized>(
    source: &R,
    item: &BoxEntry,
    key: Option<&(Vec<u8>, Span)>,
    items: &mut Vec<MetadataItem>,
) -> io::Result<()> {
    for data in iter_boxes(source, item.payload, item.end) {
        let data = data?;
        let value_start = data.payload + 8;
        if &data.kind != b"data" || value_start > data.end || data.end - value_start > MAX_TC260_VALUE_BYTES {
            continue;
        }
        let size = data.end - value_start;
        let value = source.read_vec(value_start, size)?;
        if value.len() as u64 == size {
            items.push(MetadataItem {
                item_type: item.kind,
                item_type_span: Span::new(item.start + 4, item.start + 8),
                key_name: key.map(|(name, _)| name.clone()),
                key_span: key.map(|(_, span)| *span),
                value_start,
                value_end: data.end,
                value,
            });
        }
    }
    Ok(())
}

/// Metadata items across every top-level `moov` box.
pub fn metadata_items<R: ReadAt + ?Sized>(source: &R) -> io::Result<Vec<MetadataItem>> {
    let mut items = Vec::new();
    for top in iter_boxes(source, 0, source.size()) {
        let top = top?;
        if &top.kind == b"moov" {
            items.extend(moov_items(source, &top)?);
        }
    }
    Ok(items)
}

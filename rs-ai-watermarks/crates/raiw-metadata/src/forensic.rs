//! `collect_forensic_metadata`: the broad, evidence-only inspection record.

use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use raiw_container::FileSource;
use raiw_container::isobmff::{C2PA_BOX_TYPES, STREAM_SCAN_BYTES, boxes};
use raiw_core::constants::{PNG_METADATA_CHUNKS, RIFF_CODED_IMAGE_CHUNKS, RIFF_METADATA_CHUNKS};
use serde_json::{Map, Value, json};

use crate::exif::{self, ExifDict, ExifValue, Ifd, tags};
use crate::pil::iptc::{IptcValue, getiptcinfo};
use crate::pil::{self, InfoValue};
use crate::py::{latin1, round2, utf8_replace};
use crate::readability::sniff_format;
use crate::strip::png_lossless::png_text_decode;

pub const FORENSIC_METADATA_SCHEMA_VERSION: i64 = 1;
pub const FORENSIC_METADATA_RECORD_TYPE: &str = "forensic_metadata";
const B64_CAP: usize = 1 << 20;
const MAX_FULL_READ: u64 = 256 << 20;
const HEAD_READ: usize = 4 << 20;
const RAW_SCAN_HEAD: usize = crate::record::HEAD_WINDOW;
const RAW_SCAN_TAIL: usize = crate::scan::QUICK_SCAN_BYTES;

/// `_b64(data, cap)`: base64 of the first `cap` bytes, marked when capped.
pub fn b64(data: &[u8], cap: usize) -> String {
    let encoded = B64.encode(&data[..data.len().min(cap)]);
    if data.len() > cap {
        format!("{encoded}...TRUNCATED({} bytes total)", data.len())
    } else {
        encoded
    }
}

fn decode_bytes(v: &[u8]) -> Value {
    if v.len() <= 64
        && let Ok(text) = std::str::from_utf8(v)
    {
        return json!(text);
    }
    json!(format!("hex:{}", hex::encode(v)))
}

fn float_value(f: f64) -> Value {
    serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number)
}

/// `_decode_exif_value` for a piexif value.
fn exif_value(value: &ExifValue) -> Value {
    match value {
        ExifValue::Bytes(b) => decode_bytes(b),
        ExifValue::Int(i) => json!(i),
        ExifValue::Ints(v) => json!(v),
        ExifValue::Rational(n, d) => json!([n, d]),
        ExifValue::Rationals(v) => Value::Array(v.iter().map(|(n, d)| json!([n, d])).collect()),
        ExifValue::Float(f) => float_value(*f),
        ExifValue::Floats(v) => Value::Array(v.iter().map(|f| float_value(*f)).collect()),
    }
}

fn exif_json(dict: &ExifDict) -> (Map<String, Value>, Option<Vec<u8>>) {
    let mut out = Map::new();
    for ifd in [Ifd::Zeroth, Ifd::Exif, Ifd::Gps, Ifd::Interop, Ifd::First] {
        let mut decoded = Map::new();
        for (number, value) in dict.ifd(ifd) {
            let name = tags::lookup(ifd, *number)
                .map_or_else(|| format!("tag_{number}"), |(_, n, _)| (*n).to_owned());
            let rendered = match value {
                ExifValue::Bytes(b) if name == "MakerNote" => json!(format!("hex:{}", hex::encode(b))),
                other => exif_value(other),
            };
            decoded.insert(name, rendered);
        }
        out.insert(ifd.key().into(), Value::Object(decoded));
    }
    out.insert(
        "thumbnail".into(),
        dict.thumbnail
            .as_ref()
            .map_or(Value::Null, |t| json!(format!("{} bytes", t.len()))),
    );
    (out, dict.thumbnail.clone())
}

/// `read_full_exif`: every IFD by tag name, plus the embedded thumbnail.
fn read_full_exif(path: &Path, exif_blob: Option<&[u8]>, data: Option<&[u8]>) -> (Value, Option<Vec<u8>>) {
    let loaded = match data {
        Some(data) => exif::load(data),
        None => exif::load_jpeg_file(path),
    };
    let dict = if let Ok(dict) = loaded {
        dict
    } else {
        let Some(blob) = exif_blob.filter(|b| !b.is_empty()) else {
            return (json!({}), None);
        };
        match exif::load(blob) {
            Ok(dict) => dict,
            Err(err) => return (json!({"error": err.to_string()}), None),
        }
    };
    let (map, thumbnail) = exif_json(&dict);
    (Value::Object(map), thumbnail)
}

fn iptc_json(value: &IptcValue) -> Value {
    match value {
        IptcValue::None => Value::Null,
        IptcValue::Bytes(b) => decode_bytes(b),
        IptcValue::List(items) => Value::Array(
            items
                .iter()
                .map(|item| item.as_deref().map_or(Value::Null, decode_bytes))
                .collect(),
        ),
    }
}

/// `read_pil_info`: container basics, `info` passthrough, IPTC and the EXIF blob.
/// `(pil, iptc, exif blob)` of [`read_pil_info`].
type PilParts = (Map<String, Value>, Map<String, Value>, Option<Vec<u8>>);

fn read_pil_info(path: &Path) -> PilParts {
    let mut out = Map::new();
    let mut iptc = Map::new();
    let probe = pil::probe(path);
    let image = match &probe.opened {
        Ok(image) => image,
        Err(err) => {
            out.insert("error".into(), json!(err.message(path)));
            return (out, iptc, None);
        }
    };
    out.insert("format".into(), json!(image.format));
    out.insert("mode".into(), json!(image.mode));
    out.insert("width".into(), json!(image.size.0));
    out.insert("height".into(), json!(image.size.1));
    out.insert("n_frames".into(), json!(image.n_frames));
    let mut info = image.info.clone();
    if image.format == "GIF"
        && image.n_frames > 1
        && let Some(comment) = info.shift_remove("comment")
    {
        info.insert("comment".into(), comment);
    }
    if let Some(InfoValue::Tuple(dpi)) = info.get("dpi")
        && !dpi.is_empty()
    {
        let values: Vec<Value> = dpi
            .iter()
            .filter_map(|d| match d {
                InfoValue::Int(i) => Some(*i as f64),
                InfoValue::Float(f) => Some(*f),
                _ => None,
            })
            .map(|d| float_value(round2(d)))
            .collect();
        out.insert("dpi".into(), Value::Array(values));
    }
    if let Some(InfoValue::Bytes(icc)) = info.get("icc_profile")
        && !icc.is_empty()
    {
        let header = if icc.len() >= 24 {
            hex::encode(&icc[12..24])
        } else {
            String::new()
        };
        out.insert(
            "icc_profile".into(),
            json!({"length": icc.len(), "header_hex": header, "base64": b64(icc, B64_CAP)}),
        );
    }
    let exif_blob = info.get("exif").and_then(InfoValue::as_bytes).map(<[u8]>::to_vec);
    if let Ok(Some(found)) = getiptcinfo(image) {
        for ((record, dataset), value) in &found {
            iptc.insert(format!("{record}:{dataset}"), iptc_json(value));
        }
    }
    for (key, value) in &info {
        if matches!(key.as_str(), "icc_profile" | "exif" | "dpi") {
            continue;
        }
        let rendered = match value {
            InfoValue::Bytes(bytes) => match std::str::from_utf8(bytes) {
                Ok(text) => text.to_owned(),
                Err(_) => format!("base64:{}", b64(bytes, B64_CAP)),
            },
            other => other.py_str(),
        };
        out.insert(format!("info:{key}"), json!(rendered));
    }
    (out, iptc, exif_blob)
}

fn read_c2pa_store(path: &Path) -> Value {
    let Some(raw) = crate::c2pa_store::manifest_store_json(path) else {
        return json!({});
    };
    match serde_json::from_str::<Value>(&raw) {
        Ok(Value::Object(store)) => Value::Object(store),
        Ok(_) => json!({"error": "C2PA manifest store is not an object"}),
        Err(err) => json!({"error": err.to_string()}),
    }
}

fn be32(d: &[u8], at: usize) -> usize {
    u32::from_be_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]) as usize
}

fn le32(d: &[u8], at: usize) -> usize {
    u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]) as usize
}

fn be16(d: &[u8], at: usize) -> Option<usize> {
    Some(usize::from(u16::from_be_bytes([*d.get(at)?, *d.get(at + 1)?])))
}

/// `read_png_chunks`: every chunk in order, plus the bytes after IEND.
pub fn read_png_chunks(data: &[u8]) -> (Vec<Value>, Vec<u8>) {
    let mut chunks = Vec::new();
    let mut post_iend = Vec::new();
    let mut pos = 8;
    while pos + 12 <= data.len() {
        let length = be32(data, pos);
        let kind = &data[pos + 4..pos + 8];
        let ctype = latin1(kind);
        let body = &data[(pos + 8).min(data.len())..(pos + 8).saturating_add(length).min(data.len())];
        let mut entry = Map::new();
        entry.insert("type".into(), json!(ctype));
        entry.insert("length".into(), json!(length));
        match kind {
            b"tEXt" | b"zTXt" | b"iTXt" => {
                let text = png_text_decode(kind, body);
                let xmp = text.starts_with("XML:com.adobe.xmp");
                entry.insert("text".into(), json!(text));
                if xmp {
                    entry.insert("kind".into(), json!("xmp"));
                }
            }
            b"tIME" if length == 7 => {
                if body.len() < 7 {
                    chunks.push(json!({"error": "unpack requires a buffer of 7 bytes"}));
                    return (chunks, post_iend);
                }
                let year = u16::from_be_bytes([body[0], body[1]]);
                let time = format!(
                    "{year:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                    body[2], body[3], body[4], body[5], body[6]
                );
                entry.insert("time".into(), json!(time));
            }
            b"gAMA" if length == 4 && body.len() == 4 => {
                entry.insert("gamma".into(), float_value(be32(body, 0) as f64 / 100_000.0));
            }
            b"sRGB" if length == 1 && body.len() == 1 => {
                entry.insert("rendering_intent".into(), json!(body[0]));
            }
            b"iCCP" => {
                if let Some(nul) = memchr::memchr(0, body).filter(|&n| n > 0) {
                    entry.insert("profile_name".into(), json!(latin1(&body[..nul])));
                }
                entry.insert("base64".into(), json!(b64(body, B64_CAP)));
            }
            b"iDOT" => {
                entry.insert("apple_screenshot_marker".into(), json!(true));
            }
            b"IHDR" | b"IDAT" => {}
            _ if length > 0 => {
                entry.insert("base64".into(), json!(b64(body, B64_CAP)));
            }
            _ => {}
        }
        chunks.push(Value::Object(entry));
        pos += 12 + length;
        if kind == b"IEND" {
            post_iend = data[pos.min(data.len())..].to_vec();
            break;
        }
    }
    (chunks, post_iend)
}

fn set_jpeg_trailer(result: &mut Map<String, Value>, data: &[u8], eoi: usize) {
    let trailer = &data[(eoi + 2).min(data.len())..];
    result.insert("post_eoi_bytes".into(), json!(trailer.len()));
    if !trailer.is_empty() {
        result.insert("post_eoi_base64".into(), json!(b64(trailer, B64_CAP)));
    }
}

/// `read_jpeg_segments`: every APP segment in order, plus the post-EOI trailer size.
pub fn read_jpeg_segments(data: &[u8]) -> Value {
    let mut result = Map::new();
    let mut segments = Vec::new();
    result.insert("segments".into(), json!([]));
    result.insert("post_eoi_bytes".into(), json!(0));
    let mut pos = 2;
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            break;
        }
        let marker = data[pos + 1];
        if marker == 0xD9 {
            set_jpeg_trailer(&mut result, data, pos);
            break;
        }
        if marker == 0xDA {
            if let Some(eoi) = memchr::memmem::rfind(data, b"\xff\xd9") {
                set_jpeg_trailer(&mut result, data, eoi);
            }
            break;
        }
        let length = be16(data, pos + 2).unwrap_or(0);
        if !(0xE0..=0xEF).contains(&marker) {
            pos += 2 + length;
            continue;
        }
        let body = &data[(pos + 4).min(data.len())
            ..(pos + 2 + length)
                .clamp(pos + 4, data.len().max(pos + 4))
                .min(data.len())];
        let name = format!("APP{}", marker - 0xE0);
        let mut entry = Map::new();
        entry.insert("marker".into(), json!(name));
        entry.insert("length".into(), json!(length));
        if body.starts_with(b"http://ns.adobe.com/xap/1.0/\x00") {
            entry.insert("kind".into(), json!("xmp"));
            entry.insert("text".into(), json!(utf8_replace(&body[29..])));
        } else {
            let kind = if name == "APP2" && body.starts_with(b"MPF\x00") {
                Some("mpf")
            } else if name == "APP2" && body.starts_with(b"ICC_PROFILE") {
                Some("icc")
            } else if name == "APP2" && body.starts_with(b"FPXR") {
                Some("flashpix")
            } else if name == "APP11" {
                Some("c2pa_or_jumbf")
            } else if body.starts_with(b"Exif\x00\x00") {
                Some("exif")
            } else if body.starts_with(b"Photoshop 3.0\x00") {
                Some("iptc_iim")
            } else {
                None
            };
            if let Some(kind) = kind {
                entry.insert("kind".into(), json!(kind));
            }
            entry.insert("base64".into(), json!(b64(body, B64_CAP)));
        }
        segments.push(Value::Object(entry));
        pos += 2 + length;
    }
    result.insert("segments".into(), Value::Array(segments));
    Value::Object(result)
}

/// `_jpeg_forensics_bytes`: quantization, Huffman, SOF and scan-script structure.
pub fn jpeg_forensics(data: &[u8]) -> Map<String, Value> {
    let mut out = Map::new();
    if !data.starts_with(b"\xff\xd8") {
        return out;
    }
    let mut pos = 2;
    let mut scans = Vec::new();
    let mut dqt = Map::new();
    let mut dht = Vec::new();
    let mut comments = Vec::new();
    while pos + 4 <= data.len() {
        if data[pos] != 0xFF {
            break;
        }
        let marker = data[pos + 1];
        if matches!(marker, 0xD8 | 0x01 | 0xD0..=0xD7) {
            pos += 2;
            continue;
        }
        if marker == 0xD9 {
            break;
        }
        let length = be16(data, pos + 2).unwrap_or(0);
        let body = &data[(pos + 4).min(data.len())..(pos + 2 + length).max(pos + 4).min(data.len())];
        match marker {
            0xDB => {
                let mut off = 0;
                while off < body.len() {
                    let tid = body[off] & 0x0F;
                    let precision = body[off] >> 4;
                    let n = if precision != 0 { 128 } else { 64 };
                    let raw = &body[(off + 1).min(body.len())..(off + 1 + n).min(body.len())];
                    let values: Vec<usize> = if precision != 0 {
                        raw.chunks_exact(2)
                            .map(|p| usize::from(u16::from_be_bytes([p[0], p[1]])))
                            .collect()
                    } else {
                        raw.iter().map(|&v| usize::from(v)).collect()
                    };
                    dqt.insert(tid.to_string(), json!(values[..values.len().min(64)]));
                    off += 1 + n;
                }
            }
            0xC4 => dht.push(json!(hex::encode(body))),
            0xDD if body.len() >= 2 => {
                out.insert("restart_interval".into(), json!(be16(body, 0)));
            }
            0xE0 if body.starts_with(b"JFIF\x00") && body.len() >= 12 => {
                out.insert(
                    "jfif".into(),
                    json!({
                        "version": format!("{}.{}", body[5], body[6]),
                        "density_units": body[7],
                        "x_density": be16(body, 8),
                        "y_density": be16(body, 10),
                    }),
                );
            }
            0xEE if body.starts_with(b"Adobe") && body.len() >= 12 => {
                out.insert("adobe_transform".into(), json!(body[11]));
            }
            0xC0..=0xC2 if body.len() >= 6 => {
                out.insert("progressive".into(), json!(marker == 0xC2));
                out.insert("precision_bits".into(), json!(body[0]));
                out.insert("sof_height".into(), json!(be16(body, 1)));
                out.insert("sof_width".into(), json!(be16(body, 3)));
                let comps: Vec<(u8, u8, u8)> = (0..usize::from(body[5]))
                    .filter_map(|i| body.get(6 + i * 3..9 + i * 3))
                    .map(|c| (c[1] >> 4, c[1] & 0x0F, c[2]))
                    .collect();
                if comps.len() >= 3 {
                    let (h, v, _) = comps[0];
                    let label = match u32::from(h) * u32::from(v) {
                        1 => "4:4:4".to_owned(),
                        2 => "4:2:2".to_owned(),
                        _ => format!("{h}x{v}"),
                    };
                    out.insert("subsampling".into(), json!(label));
                }
            }
            0xFE => {
                let text: String = utf8_replace(body).chars().take(2000).collect();
                comments.push(json!(text));
            }
            0xDA => {
                if body.len() >= 3 {
                    let ns = usize::from(body[0]);
                    let tail = body.get(1 + ns * 2..).unwrap_or_default();
                    if tail.len() >= 3 {
                        scans.push(
                            json!({"ss": tail[0], "se": tail[1], "ah": tail[2] >> 4, "al": tail[2] & 0x0F}),
                        );
                    }
                }
                let end = memchr::memmem::find(&data[pos..], b"\xff\xd9").map(|i| pos + i);
                let mut next = memchr::memchr(0xFF, &data[(pos + 2).min(data.len())..]).map(|i| pos + 2 + i);
                while let Some(n) = next
                    && n + 1 < data.len()
                    && data[n + 1] == 0x00
                {
                    next = memchr::memchr(0xFF, &data[n + 2..]).map(|i| n + 2 + i);
                }
                match next {
                    Some(n) if end.is_none_or(|e| n < e) => {
                        pos = n;
                        continue;
                    }
                    _ => break,
                }
            }
            _ => {}
        }
        pos += 2 + length;
    }
    if !dqt.is_empty() {
        out.insert("quant_tables".into(), Value::Object(dqt));
    }
    if !dht.is_empty() {
        out.insert("huffman_tables_hex".into(), Value::Array(dht));
    }
    if !comments.is_empty() {
        out.insert("comments".into(), Value::Array(comments));
    }
    if !scans.is_empty() {
        out.insert("scan_count".into(), json!(scans.len()));
        out.insert("scan_script".into(), Value::Array(scans));
    }
    out
}

/// `read_webp_chunks`: the RIFF chunk inventory.
pub fn read_webp_chunks(data: &[u8]) -> Vec<Value> {
    let mut chunks = Vec::new();
    let declared_end = if data.len() >= 12 {
        8 + le32(data, 4)
    } else {
        data.len()
    };
    let end = data.len().min(declared_end);
    let mut pos = 12;
    while pos + 8 <= end {
        let kind = &data[pos..pos + 4];
        let length = le32(data, pos + 4);
        let body = &data[pos + 8..(pos + 8).saturating_add(length).min(end).max(pos + 8)];
        let mut entry = Map::new();
        entry.insert("type".into(), json!(latin1(kind)));
        entry.insert("length".into(), json!(length));
        if kind == b"XMP " {
            entry.insert("kind".into(), json!("xmp"));
            entry.insert("text".into(), json!(utf8_replace(body)));
        } else if !RIFF_CODED_IMAGE_CHUNKS.contains(&kind) && length > 0 {
            entry.insert("base64".into(), json!(b64(body, B64_CAP)));
        }
        chunks.push(Value::Object(entry));
        pos += 8 + length + (length & 1);
    }
    chunks
}

fn isobmff_inventory(data: &[u8]) -> Map<String, Value> {
    let mut out = Map::new();
    let list = |start: u64, end: u64| boxes(data, start, end).unwrap_or_default();
    let top = list(0, data.len() as u64);
    out.insert(
        "boxes".into(),
        json!(top.iter().map(|b| latin1(&b.kind)).collect::<Vec<_>>()),
    );
    let mut provenance = Vec::new();
    for entry in &top {
        let (s, e) = (entry.payload as usize, entry.end as usize);
        if C2PA_BOX_TYPES.contains(&&entry.kind) {
            provenance.push(json!({"type": latin1(&entry.kind), "length": e - s, "base64": b64(&data[s..e], STREAM_SCAN_BYTES as usize)}));
        }
        if &entry.kind == b"moov" {
            for child in list(entry.payload, entry.end) {
                let cs = child.payload as usize;
                if &child.kind == b"mvhd" && child.end as usize - cs >= 24 && data[cs] == 0 {
                    let creation = be32(data, cs + 4) as i64;
                    if creation != 0 {
                        out.insert("mvhd_creation_time".into(), json!(creation - 2_082_844_800));
                    }
                }
            }
        } else if &entry.kind == b"meta" {
            for child in list(entry.payload + 4, entry.end) {
                let (cs, ce) = (child.payload, child.end);
                match &child.kind {
                    b"iinf" => {
                        let count = be16(data, cs as usize + 4).unwrap_or(0);
                        out.insert("meta_item_count".into(), json!(count));
                        let mut types: Vec<String> = Vec::new();
                        for item in list(cs + 6, ce) {
                            let (is, ie) = (item.payload as usize, item.end as usize);
                            if &item.kind == b"infe" && ie - is >= 8 {
                                let off = is + 4 + if data[is] == 3 { 4 } else { 2 } + 2;
                                if off + 4 <= ie {
                                    types.push(latin1(&data[off..off + 4]));
                                }
                            }
                        }
                        if !types.is_empty() {
                            types.sort();
                            types.dedup();
                            out.insert("meta_item_types".into(), json!(types));
                        }
                    }
                    b"iprp" => {
                        out.insert("has_iprp".into(), json!(true));
                        for prop in list(cs, ce) {
                            if &prop.kind == b"ipco" {
                                let props = list(prop.payload, prop.end);
                                out.insert(
                                    "ipco_properties".into(),
                                    json!(props.iter().map(|p| latin1(&p.kind)).collect::<Vec<_>>()),
                                );
                                for aux in props.iter().filter(|p| &p.kind == b"auxC") {
                                    let raw = &data
                                        [(aux.payload as usize + 4).min(aux.end as usize)..aux.end as usize];
                                    let first = raw.split(|&b| b == 0).next().unwrap_or_default();
                                    out.insert("auxc_types".into(), json!(latin1(first)));
                                }
                            }
                        }
                    }
                    b"iref" => {
                        out.insert("has_iref".into(), json!(true));
                    }
                    _ => {}
                }
            }
        }
    }
    if !provenance.is_empty() {
        out.insert("provenance_boxes".into(), Value::Array(provenance));
    }
    let mut qt = Map::new();
    for (atom, key) in [
        (&b"\xa9mak"[..], "make"),
        (b"\xa9mod", "model"),
        (b"\xa9swr", "software"),
    ] {
        if let Some(idx) = memchr::memmem::find(data, atom) {
            let window = &data[(idx + 4).min(data.len())..(idx + 200).min(data.len())];
            let mut run = 0;
            let mut found = None;
            for (i, &b) in window.iter().enumerate() {
                if (0x20..=0x7e).contains(&b) {
                    run += 1;
                    if run == 80 {
                        found = Some((i + 1 - run, i + 1));
                        break;
                    }
                } else {
                    if run >= 4 {
                        found = Some((i - run, i));
                        break;
                    }
                    run = 0;
                }
            }
            if found.is_none() && run >= 4 {
                found = Some((window.len() - run, window.len()));
            }
            if let Some((a, b)) = found {
                qt.insert(key.into(), json!(latin1(&window[a..b])));
            }
        }
    }
    if !qt.is_empty() {
        out.insert("quicktime".into(), Value::Object(qt));
    }
    out
}

fn isobmff_provenance_path(path: &Path) -> Map<String, Value> {
    let mut out = Map::new();
    let mut names = Vec::new();
    let mut provenance = Vec::new();
    let mut collected = 0usize;
    let cap = STREAM_SCAN_BYTES as usize;
    if let Ok(source) = FileSource::open(path) {
        use raiw_container::ReadAt;
        for entry in boxes(&source, 0, source.size()).unwrap_or_default() {
            names.push(json!(latin1(&entry.kind)));
            let length = (entry.end - entry.payload) as usize;
            if C2PA_BOX_TYPES.contains(&&entry.kind) && collected < cap {
                let to_read = length.min(cap - collected);
                let payload = source.read_vec(entry.payload, to_read as u64).unwrap_or_default();
                let mut item = Map::new();
                item.insert("type".into(), json!(latin1(&entry.kind)));
                item.insert("length".into(), json!(length));
                item.insert("base64".into(), json!(b64(&payload, cap)));
                if to_read < length {
                    item.insert("truncated".into(), json!(true));
                }
                provenance.push(Value::Object(item));
                collected += payload.len();
            }
        }
    }
    out.insert("boxes".into(), Value::Array(names));
    if !provenance.is_empty() {
        out.insert("provenance_boxes".into(), Value::Array(provenance));
    }
    out
}

fn late_chunks(path: &Path, png_form: bool) -> Vec<Value> {
    use raiw_container::ReadAt;
    let mut chunks = Vec::new();
    let Ok(source) = FileSource::open(path) else {
        return chunks;
    };
    let size = source.size();
    let mut pos = if png_form { 8u64 } else { 12 };
    let end = if png_form {
        size
    } else {
        let header = source.read_vec(0, 12).unwrap_or_default();
        if header.len() < 12 || !header.starts_with(b"RIFF") || &header[8..12] != b"WEBP" {
            return chunks;
        }
        size.min(8 + le32(&header, 4) as u64)
    };
    while pos + if png_form { 12 } else { 8 } <= end {
        let Ok(header) = source.read_vec(pos, 8) else {
            break;
        };
        if header.len() < 8 {
            break;
        }
        let (kind, length) = if png_form {
            (header[4..8].to_vec(), be32(&header, 0) as u64)
        } else {
            (header[..4].to_vec(), le32(&header, 4) as u64)
        };
        let start = pos + 8;
        let safe = length.min(end.saturating_sub(start));
        let wanted = if png_form {
            PNG_METADATA_CHUNKS
        } else {
            RIFF_METADATA_CHUNKS
        };
        if wanted.contains(&kind.as_slice()) && start >= RAW_SCAN_HEAD as u64 {
            let body = source
                .read_vec(start, safe.min(B64_CAP as u64))
                .unwrap_or_default();
            let mut entry = Map::new();
            entry.insert("type".into(), json!(latin1(&kind)));
            entry.insert("length".into(), json!(length));
            entry.insert("base64".into(), json!(b64(&body, B64_CAP)));
            if (body.len() as u64) < safe {
                entry.insert("truncated".into(), json!(true));
            }
            chunks.push(Value::Object(entry));
        }
        pos = if png_form {
            start + safe + 4
        } else {
            start + safe + (safe & 1)
        };
        if png_form && kind == b"IEND" {
            break;
        }
    }
    chunks
}

/// `apple_live_photo_id`: the Live Photo content identifier.
pub fn apple_live_photo_id(head: &[u8]) -> Option<String> {
    let markers: [&[u8]; 3] = [b"content.identifier", b"com.apple.quicktime", b"Apple iOS"];
    if !markers.iter().any(|m| memchr::memmem::find(head, m).is_some()) {
        return None;
    }
    let shape = [8usize, 4, 4, 4, 12];
    (0..head.len()).find_map(|start| {
        let mut pos = start;
        for (i, len) in shape.iter().enumerate() {
            let part = head.get(pos..pos + len)?;
            if !part.iter().all(u8::is_ascii_hexdigit) {
                return None;
            }
            pos += len;
            if i < 4 {
                if head.get(pos) != Some(&b'-') {
                    return None;
                }
                pos += 1;
            }
        }
        Some(latin1(&head[start..pos]))
    })
}

fn mtime_seconds(meta: &std::fs::Metadata) -> f64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0.0, |d| d.as_secs() as f64 + f64::from(d.subsec_nanos()) * 1e-9)
}

/// `collect_forensic_metadata(path)`.
pub fn collect_forensic_metadata(path: &Path) -> std::io::Result<Value> {
    let meta = std::fs::metadata(path)?;
    let oversized = meta.len() > MAX_FULL_READ;
    let (data, head) = if oversized {
        (None, crate::scan::read_head(path, HEAD_READ)?)
    } else {
        let data = std::fs::read(path)?;
        (Some(data.clone()), data)
    };
    let mut record = Map::new();
    record.insert("schema_version".into(), json!(FORENSIC_METADATA_SCHEMA_VERSION));
    record.insert("record_type".into(), json!(FORENSIC_METADATA_RECORD_TYPE));
    record.insert("file".into(), json!(path.to_string_lossy()));
    record.insert("name".into(), json!(crate::strip::file_name(path)));
    record.insert("extension".into(), json!(raiw_core::fs::suffix_lower(path)));
    record.insert("size_bytes".into(), json!(meta.len()));
    record.insert("mtime".into(), float_value(mtime_seconds(&meta)));
    record.insert("birthtime".into(), Value::Null);
    let sha = match &data {
        Some(d) => raiw_core::sha256_bytes(d),
        None => raiw_core::sha256_file(path)?,
    };
    record.insert("sha256".into(), json!(sha));
    let format = sniff_format(&head);
    record.insert("content_format".into(), json!(format));
    if oversized {
        record.insert("oversized".into(), json!({"head_scanned_bytes": head.len()}));
        let mut windows = Map::new();
        windows.insert(
            "head_base64".into(),
            json!(b64(&head[..head.len().min(RAW_SCAN_HEAD)], B64_CAP)),
        );
        if meta.len() > RAW_SCAN_TAIL as u64 {
            windows.insert(
                "tail_base64".into(),
                json!(b64(&crate::scan::read_file_tail(path, RAW_SCAN_TAIL)?, B64_CAP)),
            );
        }
        record.insert("raw_metadata_windows".into(), Value::Object(windows));
    }
    if let Some(id) = apple_live_photo_id(&head[..head.len().min(2 << 20)]) {
        record.insert("live_photo_content_id".into(), json!(id));
    }
    let (pil, iptc, exif_blob) = read_pil_info(path);
    record.insert("pil".into(), Value::Object(pil));
    record.insert("iptc".into(), Value::Object(iptc));
    let (exif, thumbnail) = read_full_exif(path, exif_blob.as_deref(), data.as_deref());
    record.insert("exif".into(), exif);
    record.insert("c2pa_store".into(), read_c2pa_store(path));
    if let Some(data) = &data {
        if format == "png" {
            let (chunks, post) = read_png_chunks(data);
            record.insert("png_chunks".into(), Value::Array(chunks));
            if !post.is_empty() {
                record.insert("png_post_iend_bytes".into(), json!(post.len()));
                record.insert("png_post_iend_base64".into(), json!(b64(&post, B64_CAP)));
            }
        } else if format == "jpeg" {
            record.insert("jpeg".into(), read_jpeg_segments(data));
            record.insert("jpeg_forensics".into(), Value::Object(jpeg_forensics(data)));
        } else if format == "webp" {
            record.insert("webp_chunks".into(), Value::Array(read_webp_chunks(data)));
        } else if format.starts_with("isobmff") {
            record.insert("isobmff".into(), Value::Object(isobmff_inventory(data)));
        }
    } else if format == "png" {
        let late = late_chunks(path, true);
        if !late.is_empty() {
            record.insert("png_late_metadata_chunks".into(), Value::Array(late));
        }
    } else if format == "webp" {
        let late = late_chunks(path, false);
        if !late.is_empty() {
            record.insert("webp_late_metadata_chunks".into(), Value::Array(late));
        }
    } else if format.starts_with("isobmff") {
        record.insert("isobmff".into(), Value::Object(isobmff_provenance_path(path)));
    }
    if let Some(thumb) = thumbnail.filter(|t| !t.is_empty()) {
        record.insert("has_exif_thumbnail".into(), json!(true));
        let mut forensics = jpeg_forensics(&thumb);
        forensics.insert("base64".into(), json!(b64(&thumb, B64_CAP)));
        record.insert("exif_thumbnail_forensics".into(), Value::Object(forensics));
    }
    Ok(Value::Object(record))
}

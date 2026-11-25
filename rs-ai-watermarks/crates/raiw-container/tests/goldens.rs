//! Differential goldens from `tools/goldens/container.py`: every walker over the tracked
//! fixtures and synthetic edge cases, with metadata hooks replayed from recorded verdicts.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use base64::Engine;
use raiw_container::isobmff::{self, MediaStripOptions, STREAM_SCAN_BYTES};
use raiw_container::png::ChunkAction;
use raiw_container::{FileSource, MetadataHooks, ReadAt, ebml, flv, gif, jpeg, png, readability, riff, tiff};
use raiw_core::sha256_bytes;
use raiw_core::testing::{fixture, golden};
use serde_json::{Map, Value, json};

/// Replays the Python verdicts recorded per payload SHA-256; an unrecorded call is a divergence.
struct OracleHooks {
    verdicts: HashMap<String, HashMap<String, Value>>,
}

impl OracleHooks {
    fn load() -> Self {
        let raw = golden("container", "hooks");
        let verdicts = raw
            .as_object()
            .expect("hooks object")
            .iter()
            .map(|(hook, table)| {
                let table = table
                    .as_object()
                    .expect("table")
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                (hook.clone(), table)
            })
            .collect();
        Self { verdicts }
    }

    fn lookup(&self, hook: &str, data: &[u8]) -> &Value {
        let key = sha256_bytes(data);
        self.verdicts[hook].get(&key).unwrap_or_else(|| {
            panic!(
                "{hook} called with an unrecorded payload {key} ({} bytes)",
                data.len()
            )
        })
    }

    fn flag(&self, hook: &str, data: &[u8]) -> bool {
        self.lookup(hook, data).as_bool().expect("bool verdict")
    }
}

impl MetadataHooks for OracleHooks {
    fn is_tc260_label(&self, value: &[u8]) -> bool {
        self.flag("is_tc260_label", value)
    }

    fn names_ai_generator(&self, payload: &[u8]) -> bool {
        self.flag("names_ai_generator", payload)
    }

    fn is_app_export(&self, value: &[u8]) -> bool {
        self.flag("is_app_export", value)
    }

    fn is_audio_creation(&self, value: &[u8]) -> bool {
        self.flag("is_audio_creation", value)
    }

    fn carries_ai_metadata(&self, raw: &[u8]) -> bool {
        self.flag("carries_ai_metadata", raw)
    }

    fn ai_exif_values(&self, exif: &[u8]) -> Vec<Vec<u8>> {
        let values = self
            .lookup("ai_exif_values", exif)
            .as_array()
            .expect("value list");
        values
            .iter()
            .map(|v| hex::decode(v.as_str().expect("hex")).expect("hex value"))
            .collect()
    }
}

fn shas(values: &[Vec<u8>]) -> Value {
    json!(values.iter().map(|v| sha256_bytes(v)).collect::<Vec<_>>())
}

fn patches(found: &[raiw_container::av1::ObuPatch]) -> Value {
    json!(
        found
            .iter()
            .map(|p| json!([p.header_offset, p.header_byte, p.payload_start, p.payload_end]))
            .collect::<Vec<_>>()
    )
}

fn sei_json(scan: &isobmff::VideoSeiScan) -> Value {
    json!({
        "has_parsed_tracks": scan.has_parsed_tracks,
        "kling_tag_offsets": scan.kling_tag_offsets,
        "unregistered_uuids": scan.unregistered_uuids,
        "ai_sei_spans": scan.ai_sei_spans.iter().map(|s| json!([s.start, s.end])).collect::<Vec<_>>(),
        "av1_obu_patches": patches(&scan.av1_obu_patches),
    })
}

fn pair((bytes, count): &(Vec<u8>, usize)) -> Value {
    json!([sha256_bytes(bytes), count])
}

fn isobmff_report(path: &Path, data: &[u8], work: &Path, hooks: &OracleHooks) -> Value {
    let source = FileSource::open(path).unwrap();
    let boxes = isobmff::boxes(&source, 0, source.size()).unwrap();
    let region = isobmff::scan_c2pa_region(path, isobmff::DEFAULT_REGION_BYTES).unwrap_or_default();
    let tags: Map<String, Value> = isobmff::ai_metadata_tags(path)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, Value::String(v)))
        .collect();
    let mut report = json!({
        "boxes": boxes.iter().map(|b| json!([b.start, b.end, hex::encode(b.kind), b.payload])).collect::<Vec<_>>(),
        "tiling": readability::tiling_reason("ISOBMFF box", boxes.iter().map(|b| b.end), data.len() as u64),
        "region": [region.len(), sha256_bytes(&region)],
        "xai_in_region": isobmff::xai_signature_in_region(&region),
        "xai_payloads": shas(&isobmff::xai_signature_payloads(path).unwrap()),
        "tc260_payloads": shas(&isobmff::tc260_aigc_payloads(path, hooks).unwrap()),
        "ai_metadata_tags": tags,
        "audio_creation": isobmff::has_audio_creation_comment(path, hooks).unwrap(),
        "heif_transform": isobmff::heif_transform_applied(path),
        "sei": sei_json(&isobmff::scan_video_sei(path, false).unwrap_or_default()),
        "sei_first": sei_json(&isobmff::scan_video_sei(path, true).unwrap_or_default()),
        "strip_c2pa_boxes": pair(&isobmff::strip_c2pa_boxes(data, hooks)),
        "blank_xmp": pair(&isobmff::blank_ai_xmp_packets(data, hooks)),
        "blank_exif": pair(&isobmff::blank_ai_exif_tokens(data, hooks)),
        "blank_tc260": pair(&isobmff::blank_tc260_aigc_tags(data, hooks)),
    });
    let suffix = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    for audio in [false, true] {
        let out = work.join(format!("media-{audio}{suffix}"));
        let options = MediaStripOptions {
            strip_audio_comments: audio,
            ..MediaStripOptions::default()
        };
        let counts = isobmff::strip_media_file(path, &out, options, hooks).unwrap();
        report[format!("strip_media_{audio}")] = json!([
            sha256_bytes(&std::fs::read(&out).unwrap()),
            [counts.provenance_boxes, counts.native_tags]
        ]);
    }
    let out = work.join(format!("image{suffix}"));
    let counts = isobmff::strip_image_file(path, &out, STREAM_SCAN_BYTES, hooks).unwrap();
    report["strip_image"] = json!([
        sha256_bytes(&std::fs::read(&out).unwrap()),
        [counts.provenance_boxes, counts.tc260, counts.xmp, counts.exif]
    ]);
    report
}

fn ebml_report(path: &Path, data: &[u8], work: &Path, hooks: &OracleHooks) -> Value {
    let source = FileSource::open(path).unwrap();
    let ends = ebml::top_level_ends(&source).unwrap();
    let (total, ordinals) = ebml::attachment_ordinals(path);
    let copy = work.join(format!("scrub.{}", path.extension().unwrap().to_string_lossy()));
    std::fs::copy(path, &copy).unwrap();
    let scrubbed = ebml::scrub_av1_ai_obus(&copy).unwrap();
    json!({
        "ends": ends,
        "tiling": readability::tiling_reason("EBML element", ends.iter().copied(), data.len() as u64),
        "tc260_payloads": shas(&ebml::tc260_aigc_payloads(path, hooks)),
        "c2pa_attachments": shas(&ebml::c2pa_attachment_payloads(path)),
        "attachment_ordinals": [total, ordinals],
        "av1_obus": patches(&ebml::scan_av1_ai_obus(path, false)),
        "av1_obus_first": patches(&ebml::scan_av1_ai_obus(path, true)),
        "scrub": [scrubbed, sha256_bytes(&std::fs::read(&copy).unwrap())],
    })
}

fn riff_report(path: &Path, data: &[u8], hooks: &OracleHooks) -> Value {
    let declared =
        (data.len() >= 8).then(|| 8 + u64::from(u32::from_le_bytes([data[4], data[5], data[6], data[7]])));
    let late: Map<String, Value> = [0u64, 64, 1024 * 1024]
        .into_iter()
        .map(|w| {
            let bytes = riff::late_metadata_file(path, w, riff::DEFAULT_LATE_BYTES).unwrap_or_default();
            (w.to_string(), Value::String(sha256_bytes(&bytes)))
        })
        .collect();
    json!({"declared": declared, "avi_tc260_payloads": shas(&riff::tc260_aigc_payloads(path, hooks)), "late": late})
}

fn png_rule(keep_standard: bool) -> impl FnMut(&[u8; 4], &[u8]) -> ChunkAction {
    move |kind, payload| match kind {
        b"eXIf" if keep_standard => ChunkAction::Replace(payload.to_vec()),
        b"caBX" | b"eXIf" => ChunkAction::Drop,
        b"iCCP" if !keep_standard => ChunkAction::Drop,
        b"tEXt" | b"zTXt" | b"iTXt" => {
            let keyword = payload.split(|&b| b == 0).next().unwrap_or_default();
            if [b"parameters".as_slice(), b"prompt", b"Software"].contains(&keyword) {
                ChunkAction::Drop
            } else {
                ChunkAction::Keep
            }
        }
        _ => ChunkAction::Keep,
    }
}

fn optional_sha(bytes: Option<Vec<u8>>) -> Value {
    bytes.map_or(Value::Null, |b| Value::String(sha256_bytes(&b)))
}

fn png_report(path: &Path, data: &[u8]) -> Value {
    let late: Map<String, Value> = [0u64, 64, 1024 * 1024]
        .into_iter()
        .map(|w| {
            (
                w.to_string(),
                Value::String(sha256_bytes(
                    &png::late_metadata_file(path, w).unwrap_or_default(),
                )),
            )
        })
        .collect();
    json!({
        "bit_depth": png::bit_depth(data),
        "late": late,
        "strip_keep": optional_sha(png::rewrite_chunks(data, png_rule(true))),
        "strip_all": optional_sha(png::rewrite_chunks(data, png_rule(false))),
    })
}

fn jpeg_report(data: &[u8]) -> Value {
    let rule = |marker: u8, payload: &[u8]| {
        matches!(marker, 0xEB | 0xED) || (marker == 0xE1 && payload.starts_with(b"http://ns.adobe.com/xap/"))
    };
    json!({
        "app_payloads": jpeg::app_payloads(data).iter().map(|(m, p)| json!([m, sha256_bytes(p)])).collect::<Vec<_>>(),
        "samsung_trailer": sha256_bytes(jpeg::strip_samsung_trailer(data)),
        "strip": optional_sha(jpeg::strip_segments(data, rule)),
    })
}

fn gif_report(path: &Path, data: &[u8]) -> Value {
    let late: Map<String, Value> = [0u64, 64]
        .into_iter()
        .map(|w| {
            let bytes = gif::late_metadata_file(path, w, gif::DEFAULT_LATE_BYTES).unwrap_or_default();
            (w.to_string(), Value::String(sha256_bytes(&bytes)))
        })
        .collect();
    json!({
        "late": late,
        "strip_true": optional_sha(gif::strip_metadata(data, true)),
        "strip_false": optional_sha(gif::strip_metadata(data, false)),
    })
}

fn tiff_report(path: &Path, work: &Path, hooks: &OracleHooks) -> Value {
    let mut report = json!({});
    for keep in [true, false] {
        let out = work.join(format!("tiff-{keep}.tif"));
        let ok = tiff::strip_tiff_metadata_lossless(path, &out, keep, hooks).unwrap();
        report[format!("strip_{keep}")] = if ok {
            json!(sha256_bytes(&std::fs::read(&out).unwrap()))
        } else {
            Value::Null
        };
    }
    report
}

fn describe(path: &Path, hooks: &OracleHooks) -> Value {
    let data = std::fs::read(path).unwrap();
    let work = tempfile::tempdir().unwrap();
    let mut report = json!({"size": data.len(), "sha256": sha256_bytes(&data)});
    if isobmff::is_isobmff(&data[..data.len().min(8)]) || data.starts_with(readability::JXL_CONTAINER_MAGIC) {
        report["isobmff"] = isobmff_report(path, &data, work.path(), hooks);
    }
    if data.starts_with(ebml::EBML_MAGIC) {
        report["ebml"] = ebml_report(path, &data, work.path(), hooks);
    }
    if data.starts_with(b"RIFF") {
        report["riff"] = riff_report(path, &data, hooks);
    }
    if data.starts_with(b"FLV") {
        report["flv"] = json!({"tc260_payloads": shas(&flv::tc260_aigc_payloads(path, hooks))});
    }
    if data.starts_with(raiw_core::constants::PNG_SIGNATURE) {
        report["png"] = png_report(path, &data);
    }
    if data.starts_with(b"\xff\xd8") {
        report["jpeg"] = jpeg_report(&data);
    }
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        report["gif"] = gif_report(path, &data);
    }
    if tiff::is_classic_tiff(&data) || data.starts_with(b"II+\x00") || data.starts_with(b"MM\x00+") {
        report["tiff"] = tiff_report(path, work.path(), hooks);
    }
    report
}

fn assert_report(name: &str, actual: &Value, expected: &Value) {
    if actual == expected {
        return;
    }
    for (key, value) in expected.as_object().unwrap() {
        if let Some(section) = value.as_object() {
            for (field, want) in section {
                let got = &actual[key][field];
                assert_eq!(got, want, "{name}: {key}.{field}");
            }
        }
        assert_eq!(&actual[key], value, "{name}: {key}");
    }
    assert_eq!(actual, expected, "{name}");
}

#[test]
fn fixtures_match_python() {
    let hooks = OracleHooks::load();
    let goldens = golden("container", "fixtures");
    let entries = goldens.as_object().unwrap();
    assert!(entries.len() > 90, "fixture goldens look incomplete");
    for (relative, expected) in entries {
        assert_report(relative, &describe(&fixture(relative), &hooks), expected);
    }
}

#[test]
fn synthetic_inputs_match_python() {
    let hooks = OracleHooks::load();
    let goldens = golden("container", "synthetic");
    let dir = tempfile::tempdir().unwrap();
    for (name, expected) in goldens.as_object().unwrap() {
        let compressed = base64::engine::general_purpose::STANDARD
            .decode(expected["input_zlib_b64"].as_str().unwrap())
            .unwrap();
        let mut data = Vec::new();
        flate2::read::ZlibDecoder::new(compressed.as_slice())
            .read_to_end(&mut data)
            .unwrap();
        let path = dir
            .path()
            .join(format!("{name}{}", expected["suffix"].as_str().unwrap()));
        std::fs::write(&path, &data).unwrap();
        let mut actual = describe(&path, &hooks);
        actual["suffix"] = expected["suffix"].clone();
        actual["input_zlib_b64"] = expected["input_zlib_b64"].clone();
        assert_report(name, &actual, expected);
    }
}

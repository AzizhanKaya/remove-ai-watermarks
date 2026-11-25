//! Differential goldens from `tools/goldens/c2pa.py`: derived `c2pa_info` over every fixture,
//! the raw fallback scan, synthetic PNG chunks and manifest stores, and the registries.

use std::path::Path;

use base64::Engine;
use raiw_c2pa::extractor::{self, MetadataValue};
use raiw_c2pa::{C2paInfo, attribution, fallback, png, registry, soft_binding, store};
use raiw_container::{FileSource, ReadAt, isobmff, riff};
use raiw_core::testing::{fixture, golden};
use raiw_core::{JsonStyle, sha256_bytes, to_py_json};
use serde_json::{Map, Value, json};

fn info_json(info: Option<&C2paInfo>) -> String {
    info.map_or_else(|| "{}".to_owned(), |info| to_py_json(info, JsonStyle::COMPACT))
}

fn helpers(info: Option<&C2paInfo>) -> Value {
    let Some(info) = info else { return Value::Null };
    let entries = info.string_entries();
    let markers = entries
        .iter()
        .filter(|(k, _)| !matches!(*k, "has_c2pa" | "type"))
        .map(|(k, v)| (*k, v.as_str()));
    json!({
        "credential_level": info.credential_level().as_str(),
        "invalid_credential": info.has_invalid_credential(),
        "invismark": info.has_invismark(),
        "removal_hint": info.has_removal_hint(),
        "claim_generator_platform": attribution::claim_generator_platform(info.claim_generator.as_deref()),
        "vendor_platform": attribution::c2pa_vendor_platform(info.issuer.as_deref().unwrap_or_default()),
        "markers_claim_ai": raiw_c2pa::markers_claim_ai(markers),
    })
}

/// The raw bytes the Python generator hands the fallback scanner.
fn raw_manifest_bytes(path: &Path, data: &[u8]) -> Vec<u8> {
    if data.starts_with(b"\x89PNG") {
        return first_cabx_payload(path).unwrap_or_default();
    }
    if data.starts_with(b"\xff\xd8") {
        let mut out = Vec::new();
        let mut pos = 2;
        while pos + 4 <= data.len() && data[pos] == 0xFF && !matches!(data[pos + 1], 0xDA | 0xD9) {
            let length = usize::from(u16::from_be_bytes([data[pos + 2], data[pos + 3]]));
            if data[pos + 1] == 0xEB {
                out.extend_from_slice(&data[pos + 4..(pos + 2 + length).min(data.len())]);
            }
            pos += 2 + length;
        }
        return out;
    }
    if isobmff::is_isobmff(&data[..data.len().min(8)]) {
        return isobmff::scan_c2pa_region(path, isobmff::DEFAULT_REGION_BYTES).unwrap_or_default();
    }
    if data.starts_with(b"RIFF") {
        return riff::late_metadata_file(path, 0, riff::DEFAULT_LATE_BYTES).unwrap_or_default();
    }
    Vec::new()
}

/// `_find_c2pa_chunk(path).payload` regardless of the file suffix.
fn first_cabx_payload(path: &Path) -> Option<Vec<u8>> {
    let source = FileSource::open(path).ok()?;
    for chunk in raiw_container::png::chunks(&source).ok()? {
        let chunk = chunk.ok()?;
        if &chunk.kind == b"caBX" {
            let body = source.read_vec(chunk.payload_offset(), chunk.length).ok()?;
            if fallback::looks_like_c2pa(&body) {
                return Some(body);
            }
        }
    }
    None
}

fn reader_category(path: &Path) -> &'static str {
    match raiw_c2pa::reader::read_manifest_store_json(path) {
        Ok(None) => "none",
        Ok(Some(_)) => "store",
        Err(_) => "error",
    }
}

fn extractor_report(path: &Path, report: &mut Map<String, Value>) {
    match extractor::extract_ai_metadata(path) {
        Ok(metadata) => {
            let entries: Map<String, Value> = metadata
                .iter()
                .map(|(key, value)| {
                    let rendered = match value {
                        MetadataValue::Text(text) => text.clone(),
                        MetadataValue::C2pa(info) => info_json(Some(info.as_ref())),
                        MetadataValue::Chunk(bytes) => sha256_bytes(bytes),
                    };
                    (key.clone(), Value::String(rendered))
                })
                .collect();
            report.insert("extract_ai_metadata".into(), Value::Object(entries));
            report.insert(
                "ai_summary".into(),
                json!(extractor::ai_metadata_summary(path).unwrap()),
            );
            report.insert(
                "has_ai_metadata".into(),
                json!(extractor::has_ai_metadata(path).unwrap()),
            );
        }
        Err(_) => {
            report.insert("extract_ai_metadata".into(), json!("error"));
        }
    }
}

fn describe(path: &Path) -> Map<String, Value> {
    let data = std::fs::read(path).unwrap();
    let info = raiw_c2pa::extract_c2pa_info_uncached(path);
    let png_info = png::extract_c2pa_info_png(path);
    let raw = raw_manifest_bytes(path, &data);
    let fallback_info = (!raw.is_empty()).then(|| raiw_c2pa::parse_c2pa_chunk(&raw));
    let entries = soft_binding::registry_entries_in(&raw);
    let mut report = Map::new();
    report.insert("sha256".into(), json!(sha256_bytes(&data)));
    report.insert("reader".into(), json!(reader_category(path)));
    report.insert("info_json".into(), json!(info_json(info.as_ref())));
    report.insert("png_info_json".into(), json!(info_json(png_info.as_ref())));
    report.insert("raw".into(), json!([raw.len(), sha256_bytes(&raw)]));
    report.insert("fallback_json".into(), json!(info_json(fallback_info.as_ref())));
    report.insert(
        "soft_binding_vendors_in".into(),
        json!(soft_binding::vendors_in(&raw)),
    );
    report.insert(
        "soft_binding_entries".into(),
        json!(entries.iter().map(|e| e.algorithm).collect::<Vec<_>>()),
    );
    report.insert(
        "synthid_vendors_in".into(),
        json!(attribution::synthid_evidence_vendors_in(&raw, None, None)),
    );
    report.insert("has_c2pa_metadata".into(), json!(png::has_c2pa_metadata(path)));
    report.insert(
        "c2pa_chunk".into(),
        json!(png::extract_c2pa_chunk(path).map(|c| sha256_bytes(&c))),
    );
    report.insert("well_formed_png".into(), json!(png::is_well_formed_png(path)));
    report.insert("helpers".into(), helpers(info.as_ref()));
    extractor_report(path, &mut report);
    report
}

fn compare(name: &str, actual: &Map<String, Value>, expected: &Value) {
    for (key, value) in actual {
        let mut want = expected[key].clone();
        if key == "reader" {
            let category = want.as_str().unwrap().split(':').next().unwrap().to_owned();
            want = json!(category);
        }
        if key == "extract_ai_metadata" && want.as_str().is_some_and(|s| s.starts_with("error")) {
            want = json!("error");
        }
        assert_eq!(value, &want, "{name}: {key}");
    }
}

#[test]
fn fixtures_match_python() {
    let goldens = golden("c2pa", "fixtures");
    let entries = goldens.as_object().unwrap();
    assert!(entries.len() > 90, "fixture goldens look incomplete");
    let mut stores = 0;
    for (relative, expected) in entries {
        let report = describe(&fixture(relative));
        stores += usize::from(report["reader"] == "store");
        compare(relative, &report, expected);
    }
    assert_eq!(stores, 33, "every C2PA fixture is read by the validating reader");
}

#[test]
fn synthetic_pngs_match_python() {
    let goldens = golden("c2pa", "synthetic");
    let injected = golden("c2pa", "inject");
    let dir = tempfile::tempdir().unwrap();
    let chunk = raiw_container::png::serialize_chunk(b"caBX", &base_jumbf());
    for (name, expected) in goldens.as_object().unwrap() {
        let data = base64::engine::general_purpose::STANDARD
            .decode(expected["input_b64"].as_str().unwrap())
            .unwrap();
        let path = dir.path().join(format!("{name}.png"));
        std::fs::write(&path, &data).unwrap();
        compare(name, &describe(&path), expected);

        let source = dir.path().join(format!("inject-src-{name}.png"));
        std::fs::write(&source, &data).unwrap();
        let output = dir.path().join(format!("inject-dst-{name}.png"));
        let got = match png::inject_c2pa_chunk(&source, &output, &chunk) {
            Ok(()) => sha256_bytes(&std::fs::read(&output).unwrap()),
            Err(error) => format!("error:{error}"),
        };
        assert_eq!(json!(got), injected[name], "inject {name}");
    }
}

/// The JUMBF payload of the synthetic `cabx` case.
fn base_jumbf() -> Vec<u8> {
    let goldens = golden("c2pa", "synthetic");
    let data = base64::engine::general_purpose::STANDARD
        .decode(goldens["cabx"]["input_b64"].as_str().unwrap())
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cabx.png");
    std::fs::write(&path, data).unwrap();
    let chunk = png::extract_c2pa_chunk(&path).unwrap();
    chunk[8..chunk.len() - 4].to_vec()
}

#[test]
fn manifest_stores_match_python() {
    let goldens = golden("c2pa", "stores");
    for (name, expected) in goldens.as_object().unwrap() {
        let info = store::info_from_manifest_json(expected["store"].as_str().unwrap());
        assert_eq!(
            info_json(info.as_ref()),
            expected["info_json"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(helpers(info.as_ref()), expected["helpers"], "{name} helpers");
    }
}

fn hex_of(value: &Value) -> Vec<u8> {
    hex_decode(value.as_str().unwrap())
}

fn hex_decode(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

#[test]
fn units_match_python() {
    let units = golden("c2pa", "units");
    for case in units["cbor_text_after"].as_array().unwrap() {
        let got = fallback::cbor_text_after(&hex_of(&case[0]), &hex_of(&case[1]));
        assert_eq!(json!(got), case[2], "cbor {}", case[0]);
    }
    for case in units["raw_app11"].as_array().unwrap() {
        let got = fallback::raw_app11_jumbf_bytes(case[0].as_str().unwrap()).map(|b| hex_encode(&b));
        assert_eq!(json!(got), case[1], "app11 {:?}", case[0]);
    }
    for case in units["raw_app11_info"].as_array().unwrap() {
        let got = fallback::info_from_raw_app11_profile(case[0].as_str().unwrap());
        assert_eq!(
            info_json(got.as_ref()),
            case[1].as_str().unwrap(),
            "app11 info {:?}",
            case[0]
        );
    }
    for case in units["soft_binding_entries_in"].as_array().unwrap() {
        let got: Vec<&str> = soft_binding::registry_entries_in(&hex_of(&case[0]))
            .iter()
            .map(|e| e.algorithm)
            .collect();
        assert_eq!(json!(got), case[1], "entries {}", case[0]);
    }
    for case in units["soft_binding_vendors_in"].as_array().unwrap() {
        assert_eq!(
            json!(soft_binding::vendors_in(&hex_of(&case[0]))),
            case[1],
            "vendors {}",
            case[0]
        );
    }
    for case in units["issuer_words"].as_array().unwrap() {
        assert_eq!(
            json!(attribution::issuer_word_matches(case[0].as_str().unwrap())),
            case[1]
        );
    }
    for case in units["tool_words"].as_array().unwrap() {
        assert_eq!(
            json!(attribution::tool_word_matches(case[0].as_str().unwrap())),
            case[1]
        );
    }
    for case in units["claim_generator_platform"].as_array().unwrap() {
        assert_eq!(
            json!(attribution::claim_generator_platform(case[0].as_str())),
            case[1],
            "{:?}",
            case[0]
        );
    }
    for case in units["vendor_platform"].as_array().unwrap() {
        assert_eq!(
            json!(attribution::c2pa_vendor_platform(case[0].as_str().unwrap())),
            case[1],
            "{:?}",
            case[0]
        );
    }
    for case in units["synthid"].as_array().unwrap() {
        let got =
            attribution::synthid_evidence_vendors_in(&hex_of(&case[0]), case[1].as_bool(), case[2].as_bool());
        assert_eq!(json!(got), case[3], "synthid {}", case[0]);
    }
    for case in units["markers_claim_ai"].as_array().unwrap() {
        let markers = case[0].as_object().unwrap();
        let pairs = markers.iter().map(|(k, v)| (k.as_str(), v.as_str().unwrap()));
        assert_eq!(
            json!(raiw_c2pa::markers_claim_ai(pairs)),
            case[1],
            "markers {:?}",
            case[0]
        );
    }
    for case in units["is_ai_field"].as_array().unwrap() {
        assert_eq!(
            json!(extractor::is_ai_field(case[0].as_str().unwrap())),
            case[1],
            "{:?}",
            case[0]
        );
    }
    assert_eq!(
        info_json(Some(&raiw_c2pa::parse_c2pa_chunk(&base_jumbf()))),
        units["parse_c2pa_chunk"].as_str().unwrap()
    );
}

#[test]
fn registries_match_python() {
    let expected = golden("c2pa", "registries");
    let as_str = |bytes: &[u8]| String::from_utf8(bytes.to_vec()).unwrap();
    let vendors: Vec<Value> = registry::C2PA_AI_VENDORS
        .iter()
        .map(|v| {
            json!({
                "issuer": as_str(v.issuer), "org": v.org, "platform": v.platform, "needle": v.needle,
                "synthid": v.synthid, "asserts_ai": v.asserts_ai,
                "synthid_requires_watermark_action": v.synthid_requires_watermark_action,
                "signer_platform": v.signer_platform,
                "raw_signer_platforms": v.raw_signer_platforms.iter().map(|(t, p)| json!([as_str(t), p])).collect::<Vec<_>>(),
            })
        })
        .collect();
    assert_eq!(json!(vendors), expected["vendors"]);
    let issuers: Vec<Value> = registry::c2pa_issuers()
        .map(|(t, o)| json!([as_str(t), o]))
        .collect();
    assert_eq!(json!(issuers), expected["issuers"]);
    let signers: Vec<String> = registry::SYNTHID_EDIT_SIGNERS.iter().map(|s| as_str(s)).collect();
    assert_eq!(json!(signers), expected["synthid_edit_signers"]);
    let mut orgs = registry::C2PA_IDENTITY_AI_ORGS.clone();
    orgs.sort_unstable();
    assert_eq!(json!(orgs), expected["identity_ai_orgs"]);
    let by_org: Map<String, Value> = registry::C2PA_AI_VENDORS
        .iter()
        .filter_map(|v| registry::signer_platform_by_org(v.org).map(|p| (v.org.to_owned(), json!(p))))
        .collect();
    assert_eq!(Value::Object(by_org), expected["signer_platform_by_org"]);
    let platforms: Vec<Value> = registry::C2PA_SIGNER_PLATFORMS
        .iter()
        .map(|(t, p)| json!([as_str(t), p]))
        .collect();
    assert_eq!(json!(platforms), expected["signer_platforms"]);
    assert_eq!(
        json!(registry::C2PA_CLAIM_GENERATOR_PLATFORMS),
        expected["claim_generator_platforms"]
    );
    let tools: Vec<Value> = registry::C2PA_AI_TOOLS
        .iter()
        .map(|(t, l)| json!([as_str(t), l]))
        .collect();
    assert_eq!(json!(tools), expected["ai_tools"]);
    let actions: Vec<Value> = registry::C2PA_ACTIONS
        .iter()
        .map(|(t, l)| json!([as_str(t), l]))
        .collect();
    assert_eq!(json!(actions), expected["actions"]);
    let mut tokens = registry::AI_GENERATOR_TOKENS.to_vec();
    tokens.sort_unstable();
    assert_eq!(json!(tokens), expected["ai_generator_tokens"]);
    assert_eq!(json!(soft_binding::LABEL_OVERRIDES), expected["label_overrides"]);
    let row = |e: &soft_binding::SoftBindingAlgorithm| {
        json!([
            e.identifier,
            e.algorithm,
            e.kind.as_str(),
            e.decoded_media_types,
            e.encoded_media_types,
            e.display_label,
            e.date_entered,
            e.resolution_apis,
            e.deprecated
        ])
    };
    assert_eq!(
        json!(soft_binding::VENDOR_PREFIXES.iter().map(row).collect::<Vec<_>>()),
        expected["vendor_prefixes"]
    );
    assert_eq!(
        json!(soft_binding::REGISTRY.iter().map(row).collect::<Vec<_>>()),
        expected["soft_binding_rows"]
    );
    let labels: Vec<Value> = soft_binding::REGISTRY
        .iter()
        .map(|e| json!([e.algorithm, soft_binding::registered_label(e.algorithm)]))
        .collect();
    assert_eq!(json!(labels), expected["soft_binding_labels"]);
    assert_eq!(
        expected["soft_binding_source"][1],
        json!(soft_binding::SOURCE_REVISION)
    );
    let mut markers = raiw_c2pa::C2PA_STRUCTURAL_MARKERS.to_vec();
    markers.sort_unstable();
    assert_eq!(json!(markers), expected["structural_markers"]);
}

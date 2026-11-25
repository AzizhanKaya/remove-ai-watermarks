//! Behavioral assertions ported from the Python test suite (`tests/test_metadata*.py`,
//! `test_security_clamp.py`, `test_video.py`, `test_fail_closed.py`, `test_av_*`).

use raiw_container::isobmff::{self, MediaStripOptions};
use raiw_container::markers::{C2PA_UUID, KLING_SEI_TAG};
use raiw_container::{MetadataHooks, av1, ebml, flv, readability, riff, tiff};

/// Minimal stand-ins for the metadata-layer rules, enough for structural assertions.
struct TestHooks;

impl MetadataHooks for TestHooks {
    fn is_tc260_label(&self, value: &[u8]) -> bool {
        serde_json::from_slice::<serde_json::Value>(value)
            .ok()
            .and_then(|v| {
                v.as_object()
                    .map(|o| o.contains_key("Label") || o.contains_key("ContentProducer"))
            })
            .unwrap_or(false)
    }

    fn names_ai_generator(&self, payload: &[u8]) -> bool {
        raiw_container::markers::contains(payload, b"Midjourney")
    }

    fn is_app_export(&self, _value: &[u8]) -> bool {
        false
    }

    fn is_audio_creation(&self, _value: &[u8]) -> bool {
        false
    }

    fn carries_ai_metadata(&self, raw: &[u8]) -> bool {
        [
            "trainedAlgorithmicMedia",
            "ComfyUI",
            "ContentProducer",
            "Adobe Firefly",
        ]
        .iter()
        .any(|token| raiw_container::markers::contains(raw, token.as_bytes()))
    }

    fn ai_exif_values(&self, _exif: &[u8]) -> Vec<Vec<u8>> {
        Vec::new()
    }
}

const FTYP: &[u8] = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1";

fn bx(kind: &[u8], payload: &[u8]) -> Vec<u8> {
    [&((8 + payload.len()) as u32).to_be_bytes()[..], kind, payload].concat()
}

fn blanked(boxed: &[u8]) -> Vec<u8> {
    [&boxed[..4], b"free", &vec![0u8; boxed.len() - 8]].concat()
}

#[test]
fn jpegxl_jumb_box_becomes_free() {
    let data = [FTYP, b"\x00\x00\x00\x0djumbhello"].concat();
    let (out, count) = isobmff::strip_c2pa_boxes(&data, &TestHooks);
    assert_eq!(count, 1);
    assert_eq!(out, [FTYP, b"\x00\x00\x00\x0dfree\x00\x00\x00\x00\x00"].concat());
}

#[test]
fn strip_keeps_offsets_and_fails_safe() {
    let uuid = [&[0, 0, 0, 0x20][..], b"uuid", &C2PA_UUID, b"payload!"].concat();
    let mdat = b"\x00\x00\x00\x10mdatpixeldat";
    let data = [FTYP, &uuid, mdat].concat();
    let (out, count) = isobmff::strip_c2pa_boxes(&data, &TestHooks);
    assert_eq!((count, out.len()), (1, data.len()));
    assert_eq!(out, [FTYP, &blanked(&uuid), mdat].concat());

    let large = [&b"\x00\x00\x00\x01free"[..], &24u64.to_be_bytes(), &[0u8; 8]].concat();
    assert_eq!(isobmff::strip_c2pa_boxes(&large, &TestHooks).1, 0);
    for malformed in [
        [FTYP, b"\x00\x00\x00\x04XXXX"].concat(),
        [FTYP, b"\x00\x00\x00\x01uuid"].concat(),
        [
            FTYP,
            &bx(b"uuid", &[&C2PA_UUID[..], b"manifest-bytes"].concat()),
            b"\x7f\xff\xff\xfffree",
            &[0; 16],
        ]
        .concat(),
    ] {
        assert_eq!(
            isobmff::strip_c2pa_boxes(&malformed, &TestHooks),
            (malformed.clone(), 0)
        );
    }
    let png = [&b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"[..], &[0; 100]].concat();
    assert_eq!(isobmff::strip_c2pa_boxes(&png, &TestHooks), (png.clone(), 0));
}

#[test]
fn xmp_uuid_boxes_are_matched_by_content() {
    let mp4_ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom";
    let mdat = b"\x00\x00\x00\x10mdatvideodat";
    let ai = [
        &(0u8..16).collect::<Vec<_>>()[..],
        b"<x:xmpmeta><TC260:AIGC>{\"Label\":\"1\"}</TC260:AIGC></x:xmpmeta>",
    ]
    .concat();
    let boxed = bx(b"uuid", &ai);
    let (out, count) = isobmff::strip_c2pa_boxes(&[&mp4_ftyp[..], &boxed, mdat].concat(), &TestHooks);
    assert_eq!(
        (out, count),
        ([&mp4_ftyp[..], &blanked(&boxed), mdat].concat(), 1)
    );
    let plain = bx(
        b"uuid",
        &[
            &[0u8; 16][..],
            b"<x:xmpmeta><dc:rights>(c) me</dc:rights></x:xmpmeta>",
        ]
        .concat(),
    );
    let data = [&mp4_ftyp[..], &plain, mdat].concat();
    assert_eq!(isobmff::strip_c2pa_boxes(&data, &TestHooks).1, 0);
}

#[test]
fn top_level_box_encodings() {
    let large = [
        &1u32.to_be_bytes()[..],
        b"uuid",
        &24u64.to_be_bytes(),
        b"payload!",
    ]
    .concat();
    let boxes = isobmff::top_level_boxes(&large);
    assert_eq!(
        (boxes.len(), boxes[0].end, &boxes[0].kind, boxes[0].payload),
        (1, 24, b"uuid", 16)
    );
    let to_eof = [&0u32.to_be_bytes()[..], b"mdat", b"tail-to-eof"].concat();
    assert_eq!(isobmff::top_level_boxes(&to_eof)[0].end, to_eof.len() as u64);
    assert!(isobmff::top_level_boxes(b"\x00\x00\x00\x04ftypmore").is_empty());
    assert!(isobmff::top_level_boxes(b"\x00\x00\x03\xe7uuidx").is_empty());
    assert!(isobmff::is_isobmff(FTYP));
    assert!(!isobmff::is_isobmff(b"\x89PNG\r\n\x1a\n\x00\x00"));
    assert!(!isobmff::is_isobmff(b"abc"));
}

#[test]
fn xmp_packets_are_blanked_only_when_ai() {
    let packet =
        b"<?xpacket begin=\"x\"?><x:xmpmeta>trainedAlgorithmicMedia</x:xmpmeta><?xpacket end=\"w\"?>";
    let data = [&b"boxhdr"[..], packet, b"tail"].concat();
    let (out, count) = isobmff::blank_ai_xmp_packets(&data, &TestHooks);
    assert_eq!((count, out.len()), (1, data.len()));
    assert!(out.starts_with(b"boxhdr") && out.ends_with(b"tail"));
    assert!(!raiw_container::markers::contains(
        &out,
        b"trainedAlgorithmicMedia"
    ));
    let open = b"<?xpacket begin=\"x\"?><x:xmpmeta>trainedAlgorithmicMedia</x:xmpmeta>";
    assert_eq!(
        isobmff::blank_ai_xmp_packets(open, &TestHooks),
        (open.to_vec(), 0)
    );
}

#[test]
fn late_c2pa_box_is_found_past_mdat_and_capped() {
    let dir = tempfile::tempdir().unwrap();
    let mp4_ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom";
    let late = [
        &mp4_ftyp[..],
        &bx(b"mdat", &vec![0; 1_500_000]),
        &bx(
            b"uuid",
            &[&C2PA_UUID[..], b"OpenAI trainedAlgorithmicMedia"].concat(),
        ),
    ]
    .concat();
    let path = dir.path().join("late.mp4");
    std::fs::write(&path, late).unwrap();
    let region = isobmff::scan_c2pa_region(&path, isobmff::DEFAULT_REGION_BYTES).unwrap();
    assert!(raiw_container::markers::contains(&region, &C2PA_UUID));
    let capped = dir.path().join("capped.mp4");
    std::fs::write(&capped, [&mp4_ftyp[..], &bx(b"uuid", &[b'A'; 5000])].concat()).unwrap();
    assert!(isobmff::scan_c2pa_region(&capped, 1000).unwrap().len() <= 1000);
    let png = dir.path().join("x.mp4");
    std::fs::write(&png, b"\x89PNG\r\n\x1a\n not an isobmff file").unwrap();
    assert!(isobmff::scan_c2pa_region(&png, 1000).unwrap().is_empty());
}

#[test]
fn malformed_streaming_walk_copies_the_input() {
    let dir = tempfile::tempdir().unwrap();
    let data = [FTYP, b"\x00\x00\x03\xe7uuidshort"].concat();
    let source = dir.path().join("in.mp4");
    let output = dir.path().join("out.mp4");
    std::fs::write(&source, &data).unwrap();
    let counts =
        isobmff::strip_media_file(&source, &output, MediaStripOptions::default(), &TestHooks).unwrap();
    assert_eq!(
        (counts.provenance_boxes, counts.native_tags, counts.walk_failed),
        (0, 0, true)
    );
    assert_eq!(std::fs::read(&output).unwrap(), data);
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        2,
        "no staged leftovers"
    );
}

fn fullbox(kind: &[u8], payload: &[u8]) -> Vec<u8> {
    bx(kind, &[&[0u8; 4][..], payload].concat())
}

#[test]
fn late_video_sei_is_found_and_blanked_without_touching_slices() {
    let mp4_ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom";
    let idr = [&b"\x65"[..], &vec![0x22; 1200 * 1024], KLING_SEI_TAG, &[0x22; 64]].concat();
    let v0 = [&(idr.len() as u32).to_be_bytes()[..], &idr].concat();
    let a0 = [&b"\xff\xf1\x50\x80"[..], KLING_SEI_TAG, &[0xaa; 32]].concat();
    let custom = hex::decode("112233445566778899aabbccddeeff00").unwrap();
    let sei = [
        &b"\x06\x05\x14"[..],
        &custom,
        b"vend",
        b"\x05\x19",
        KLING_SEI_TAG,
        b"\x00\x80",
    ]
    .concat();
    let v1 = [&(sei.len() as u32).to_be_bytes()[..], &sei].concat();
    let v0_off = (mp4_ftyp.len() + 8) as u32;
    let a0_off = v0_off + v0.len() as u32;
    let v1_off = a0_off + a0.len() as u32;
    let avc1 = bx(
        b"avc1",
        &[&[0u8; 78][..], &bx(b"avcC", b"\x01\x64\x00\x1f\xff")].concat(),
    );
    let words = |values: &[u32]| values.iter().flat_map(|v| v.to_be_bytes()).collect::<Vec<u8>>();
    let stbl = |entry: Vec<u8>, sizes: &[u32], offsets: &[u32]| {
        bx(
            b"stbl",
            &[
                fullbox(b"stsd", &[&words(&[1])[..], &entry].concat()),
                fullbox(b"stsc", &words(&[1, 1, 1, 1])),
                fullbox(
                    b"stsz",
                    &[&words(&[0, sizes.len() as u32])[..], &words(sizes)].concat(),
                ),
                fullbox(
                    b"stco",
                    &[&words(&[offsets.len() as u32])[..], &words(offsets)].concat(),
                ),
            ]
            .concat(),
        )
    };
    let trak = |handler: &[u8], stbl: Vec<u8>| {
        bx(
            b"trak",
            &bx(
                b"mdia",
                &[
                    fullbox(b"hdlr", &[&[0u8; 4][..], handler, &[0; 12]].concat()),
                    bx(b"minf", &stbl),
                ]
                .concat(),
            ),
        )
    };
    let v_trak = trak(
        b"vide",
        stbl(avc1, &[v0.len() as u32, v1.len() as u32], &[v0_off, v1_off]),
    );
    let a_trak = trak(
        b"soun",
        stbl(bx(b"mp4a", &[0; 28]), &[a0.len() as u32], &[a0_off]),
    );
    let raw = [
        &mp4_ftyp[..],
        &bx(b"mdat", &[v0, a0.clone(), v1].concat()),
        &bx(b"moov", &[v_trak, a_trak].concat()),
    ]
    .concat();
    let tag_offset = u64::from(v1_off) + 4 + memchr::memmem::find(&sei, KLING_SEI_TAG).unwrap() as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("late.mp4");
    std::fs::write(&path, &raw).unwrap();
    let scan = isobmff::scan_video_sei(&path, false).unwrap();
    assert!(scan.has_parsed_tracks);
    assert_eq!(scan.kling_tag_offsets, [tag_offset]);
    assert!(
        scan.unregistered_uuids
            .contains(&"112233445566778899aabbccddeeff00".to_owned())
    );

    let output = dir.path().join("clean.mp4");
    let counts = isobmff::strip_media_file(&path, &output, MediaStripOptions::default(), &TestHooks).unwrap();
    assert_eq!(counts.native_tags, 1);
    let clean = std::fs::read(&output).unwrap();
    assert_eq!(clean.len(), raw.len());
    let at = tag_offset as usize;
    assert_eq!(&clean[at..at + 24], &[b' '; 24]);
    assert_eq!(
        memchr::memmem::find_iter(&clean, KLING_SEI_TAG).count(),
        2,
        "slice and audio bytes are untouched"
    );
}

#[test]
fn av1_metadata_types() {
    let obu = |payload: &[u8]| {
        [
            &[(5 << 3) | 0x02][..],
            &av1::encode_leb128(payload.len() as u64),
            payload,
        ]
        .concat()
    };
    let cll = obu(&[
        &av1::encode_leb128(1)[..],
        &1000u16.to_be_bytes(),
        &400u16.to_be_bytes(),
    ]
    .concat());
    let mdcv = obu(&[&av1::encode_leb128(2)[..], &[1; 24]].concat());
    let t35 = [
        &av1::encode_leb128(4)[..],
        b"\xb5",
        &C2PA_UUID,
        b"{\"Label\":\"1\"}",
    ]
    .concat();
    assert!(!av1::is_ai_obu_metadata_payload(&cll[2..]));
    assert!(!av1::is_ai_obu_metadata_payload(&mdcv[2..]));
    assert!(av1::is_ai_obu_metadata_payload(&t35));
    let sample = [cll.clone(), obu(&t35)].concat();
    let patches = av1::scan_sample_spans(&sample, 0);
    assert_eq!(patches.len(), 1);
    assert_eq!(patches[0].header_offset, cll.len() as u64);
}

fn ebml(id: &[u8], body: &[u8]) -> Vec<u8> {
    let size = if body.len() < 0x7F {
        vec![0x80 | body.len() as u8]
    } else {
        (0x4000 | body.len() as u16).to_be_bytes().to_vec()
    };
    [id, &size, body].concat()
}

#[test]
fn mkv_av1_scrub_refreshes_crcs() {
    let obu = |payload: &[u8]| {
        [
            &[(5 << 3) | 0x02][..],
            &av1::encode_leb128(payload.len() as u64),
            payload,
        ]
        .concat()
    };
    let cll = obu(&[&av1::encode_leb128(1)[..], &[0, 1, 0, 2]].concat());
    let ai = obu(&[
        &av1::encode_leb128(4)[..],
        b"\xb5",
        &C2PA_UUID,
        b"{\"Label\":\"1\"}",
    ]
    .concat());
    let entry = ebml(
        b"\xae",
        &[
            ebml(b"\xd7", b"\x01"),
            ebml(b"\x83", b"\x01"),
            ebml(b"\x86", b"V_AV1"),
            ebml(b"\x63\xa2", &[&b"\x81\x04\x0c\x00"[..], &cll, &ai].concat()),
        ]
        .concat(),
    );
    let tracks = ebml(
        b"\x16\x54\xae\x6b",
        &[ebml(b"\xbf", &crc32fast::hash(&entry).to_le_bytes()), entry].concat(),
    );
    let inner = [
        ebml(b"\xe7", b"\x00"),
        ebml(b"\xa3", &[&b"\x81\x00\x00\x80"[..], &cll, &ai].concat()),
    ]
    .concat();
    let cluster = ebml(
        b"\x1f\x43\xb6\x75",
        &[ebml(b"\xbf", &crc32fast::hash(&inner).to_le_bytes()), inner].concat(),
    );
    let mkv = [
        ebml(b"\x1a\x45\xdf\xa3", &ebml(b"\x42\x82", b"matroska")),
        ebml(b"\x18\x53\x80\x67", &[tracks, cluster].concat()),
    ]
    .concat();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.mkv");
    std::fs::write(&path, &mkv).unwrap();
    assert_eq!(ebml::scan_av1_ai_obus(&path, true).len(), 1);
    assert_eq!(ebml::scrub_av1_ai_obus(&path).unwrap(), 2);
    let out = std::fs::read(&path).unwrap();
    assert!(memchr::memmem::find(&out, &cll).is_some());
    assert!(memchr::memmem::find(&out, &C2PA_UUID).is_none());
    let tracks_at = memchr::memmem::find(&out, b"\x16\x54\xae\x6b").unwrap();
    let crc_at = tracks_at + memchr::memmem::find(&out[tracks_at..], b"\xbf\x84").unwrap();
    let cluster_at = memchr::memmem::find(&out, b"\x1f\x43\xb6\x75").unwrap();
    let stored = u32::from_le_bytes(out[crc_at + 2..crc_at + 6].try_into().unwrap());
    assert_eq!(stored, crc32fast::hash(&out[crc_at + 6..cluster_at]));
}

#[test]
fn truncated_ebml_segment_is_unreadable() {
    let short = [&b"\x1a\x45\xdf\xa3\x80\x18\x53\x80\x67\x88"[..], &[0; 4]].concat();
    assert!(
        readability::ebml_tiling_reason(short.as_slice())
            .unwrap()
            .is_some()
    );
    let full = [short, vec![0; 4]].concat();
    assert_eq!(readability::ebml_tiling_reason(full.as_slice()).unwrap(), None);
}

fn amf_str(value: &[u8], long: bool) -> Vec<u8> {
    if long {
        [&[0x0c][..], &(value.len() as u32).to_be_bytes(), value].concat()
    } else {
        [&[0x02][..], &(value.len() as u16).to_be_bytes(), value].concat()
    }
}

fn amf_property(name: &[u8], value: &[u8]) -> Vec<u8> {
    [&(name.len() as u16).to_be_bytes()[..], name, value].concat()
}

const TC260: &[u8] = b"{\"Label\":\"1\",\"ContentProducer\":\"00119144030008867405X210002\"}";

fn flv_file(script: &[u8]) -> Vec<u8> {
    let tag = [&[0x12][..], &(script.len() as u32).to_be_bytes()[1..], &[0; 7]].concat();
    [
        &b"FLV\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00"[..],
        &tag,
        script,
        &((11 + script.len()) as u32).to_be_bytes(),
    ]
    .concat()
}

#[test]
fn flv_walker_steps_over_every_amf0_type() {
    let end = b"\x00\x00\x09";
    let values: Vec<Vec<u8>> = vec![
        [&b"\x00"[..], &[0; 8]].concat(),
        b"\x01\x01".to_vec(),
        amf_str(b"a string", false),
        b"\x05".to_vec(),
        b"\x06".to_vec(),
        b"\x07\x00\x01".to_vec(),
        [&b"\x0b"[..], &[0; 10]].concat(),
        amf_str(b"a long string", true),
        [&b"\x0a\x00\x00\x00\x02\x00"[..], &[0; 8], b"\x01\x00"].concat(),
        [&b"\x03"[..], &amf_property(b"inner", b"\x01\x00"), end].concat(),
        [
            &b"\x08\x00\x00\x00\x01"[..],
            &amf_property(b"inner", b"\x05"),
            end,
        ]
        .concat(),
    ];
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.flv");
    for value in values {
        let script = [
            amf_str(b"onMetaData", false),
            vec![3],
            amf_property(b"x", &value),
            amf_property(b"AIGC", &amf_str(TC260, false)),
            end.to_vec(),
        ]
        .concat();
        std::fs::write(&path, flv_file(&script)).unwrap();
        assert_eq!(
            flv::tc260_aigc_payloads(&path, &TestHooks),
            [TC260.to_vec()],
            "{value:02x?}"
        );
    }
    let unknown = [
        amf_str(b"onMetaData", false),
        vec![3],
        amf_property(b"mystery", b"\x7f"),
        amf_property(b"AIGC", &amf_str(TC260, false)),
    ]
    .concat();
    std::fs::write(&path, flv_file(&unknown)).unwrap();
    assert!(flv::tc260_aigc_payloads(&path, &TestHooks).is_empty());
    let long = [
        amf_str(b"onMetaData", false),
        vec![3],
        amf_property(b"AIGC", &amf_str(TC260, true)),
        end.to_vec(),
    ]
    .concat();
    std::fs::write(&path, flv_file(&long)).unwrap();
    assert_eq!(flv::tc260_aigc_payloads(&path, &TestHooks).len(), 1);
    std::fs::write(&path, [&b"NOTFLV\x00\x00\x09"[..], &[0; 32]].concat()).unwrap();
    assert!(flv::tc260_aigc_payloads(&path, &TestHooks).is_empty());
    assert!(flv::tc260_aigc_payloads(&dir.path().join("missing.flv"), &TestHooks).is_empty());
}

fn riff_chunk(id: &[u8], payload: &[u8]) -> Vec<u8> {
    let pad: &[u8] = if payload.len() % 2 == 1 { b"\x00" } else { b"" };
    [id, &(payload.len() as u32).to_le_bytes(), payload, pad].concat()
}

#[test]
fn riff_reader_reads_avi_only() {
    let dir = tempfile::tempdir().unwrap();
    let body = [
        &b"AVI "[..],
        &riff_chunk(b"LIST", &[&b"INFO"[..], &riff_chunk(b"AIGC", TC260)].concat()),
    ]
    .concat();
    let avi = [&b"RIFF"[..], &(body.len() as u32).to_le_bytes(), &body].concat();
    let path = dir.path().join("clip.bin");
    std::fs::write(&path, &avi).unwrap();
    assert_eq!(riff::tc260_aigc_payloads(&path, &TestHooks), [TC260.to_vec()]);
    let webp_body = [&b"WEBP"[..], &riff_chunk(b"VP8L", &[0; 16])].concat();
    std::fs::write(
        &path,
        [&b"RIFF"[..], &(webp_body.len() as u32).to_le_bytes(), &webp_body].concat(),
    )
    .unwrap();
    assert!(riff::tc260_aigc_payloads(&path, &TestHooks).is_empty());
}

#[test]
fn png_late_scan_clamps_an_inflated_length() {
    let idat = vec![0u8; 1024 * 1024 + 16];
    let blob = [
        &b"\x89PNG\r\n\x1a\n"[..],
        &(idat.len() as u32).to_be_bytes(),
        b"IDAT",
        &idat,
        &[0; 4],
        &0x7FFF_FFFFu32.to_be_bytes(),
        b"tEXt",
        b"AIGC short",
    ]
    .concat();
    let late = raiw_container::png::late_metadata(blob.as_slice(), 0).unwrap();
    assert_eq!(late, b"AIGC short");
}

#[test]
fn big_endian_tiff_keeps_pixels_and_drops_the_aigc_comment() {
    let user_comment =
        b"ASCII\x00\x00\x00{\"AIGC\":{\"Label\":\"1\",\"ContentProducer\":\"001191110108335469089C10100\"}}";
    let date = b"2026:11:21 12:00:00\x00";
    let uc_off = 144u32;
    let dt_off = uc_off + user_comment.len() as u32;
    let ifd0: [(u16, u16, u32, u32); 8] = [
        (256, 4, 1, 1),
        (257, 4, 1, 1),
        (258, 3, 1, 8 << 16),
        (262, 3, 1, 1 << 16),
        (273, 4, 1, 110),
        (277, 3, 1, 1 << 16),
        (279, 4, 1, 1),
        (34665, 4, 1, 114),
    ];
    let exif: [(u16, u16, u32, u32); 2] = [
        (36867, 2, date.len() as u32, dt_off),
        (37510, 7, user_comment.len() as u32, uc_off),
    ];
    let mut be = b"MM\x00*\x00\x00\x00\x08".to_vec();
    be.extend((ifd0.len() as u16).to_be_bytes());
    for (tag, kind, count, value) in ifd0 {
        be.extend(
            [
                &tag.to_be_bytes()[..],
                &kind.to_be_bytes(),
                &count.to_be_bytes(),
                &value.to_be_bytes(),
            ]
            .concat(),
        );
    }
    be.extend(0u32.to_be_bytes());
    be.extend(b"\x7f\x00\x00\x00");
    be.extend((exif.len() as u16).to_be_bytes());
    for (tag, kind, count, value) in exif {
        be.extend(
            [
                &tag.to_be_bytes()[..],
                &kind.to_be_bytes(),
                &count.to_be_bytes(),
                &value.to_be_bytes(),
            ]
            .concat(),
        );
    }
    be.extend(0u32.to_be_bytes());
    be.extend_from_slice(user_comment);
    be.extend_from_slice(date);
    assert!(tiff::strip_tiff_bytes(&mut be, true, &TestHooks));
    assert_eq!(&be[..4], b"MM\x00*");
    assert_eq!(be[110], 0x7F);
    assert!(memchr::memmem::find(&be, b"ContentProducer").is_none());
    assert!(memchr::memmem::find(&be, b"2026:11:21 12:00:00").is_some());
    let mut bigtiff = [&b"II+\x00"[..], &[0; 16]].concat();
    assert!(!tiff::strip_tiff_bytes(&mut bigtiff, true, &TestHooks));
}

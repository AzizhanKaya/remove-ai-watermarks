"""Golden generator for the `container` suite (raiw-container).

Run from the repository root: `.venv/bin/python rs-ai-watermarks/tools/goldens/container.py`.

Every container walker runs over the tracked image/video fixtures plus small
synthetic inputs (embedded as base64). Calls that reach up into metadata-level
predicates are recorded by payload SHA-256 into `hooks.json`, so the Rust tests
replay the exact Python verdicts through an oracle `MetadataHooks`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import piexif  # noqa: E402

from remove_ai_watermarks import metadata  # noqa: E402
from remove_ai_watermarks._internal import av1_obu, ebml, flv, isobmff, riff, tiff_strip  # noqa: E402

OUT = REPO / "rs-ai-watermarks" / "goldens" / "container"
FIXTURE_ROOT = REPO / "data" / "fixtures"
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".m4a", ".heic", ".avif", ".mkv", ".webm"}


def sha(data: bytes) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


# Hook recording

HOOKS: dict[str, dict[str, object]] = {
    "is_tc260_label": {},
    "names_ai_generator": {},
    "is_app_export": {},
    "is_audio_creation": {},
    "carries_ai_metadata": {},
    "ai_exif_values": {},
}


def record(hook: str, data: bytes, verdict: object) -> None:
    key = sha(data)
    previous = HOOKS[hook].get(key)
    assert previous is None or previous == verdict, (hook, key)
    HOOKS[hook][key] = verdict


def install_hooks() -> None:
    real_tc260 = metadata.parse_tc260_aigc_json

    def tc260(value: bytes):
        parsed = real_tc260(value)
        record("is_tc260_label", value, parsed is not None)
        return parsed

    for module in (isobmff, ebml, riff, flv):
        module.parse_tc260_aigc_json = tc260

    real_generator = metadata.generator_from_metadata

    def generator(candidates, scan=b""):
        assert tuple(candidates) == ()
        found = real_generator(candidates, scan)
        record("names_ai_generator", scan, found is not None)
        return found

    isobmff.generator_from_metadata = generator

    real_provenance = metadata.app_provenance_from_metadata
    real_app_generator = metadata.app_generator_from_metadata

    def provenance(value):
        found = real_provenance(value)
        record("is_app_export", value, bool(found or real_app_generator(value)))
        return found

    isobmff.app_provenance_from_metadata = provenance
    isobmff.app_generator_from_metadata = real_app_generator

    real_creation = isobmff.creation_tool

    def creation(value):
        found = real_creation(value)
        record("is_audio_creation", value, found is not None)
        return found

    isobmff.creation_tool = creation

    real_carries = metadata.payload_carries_ai_metadata

    def carries(raw: bytes) -> bool:
        verdict = real_carries(raw)
        record("carries_ai_metadata", raw, bool(verdict))
        return verdict

    metadata.payload_carries_ai_metadata = carries

    state: dict[str, object] = {"active": False, "slice": None}
    real_load = piexif.load

    def load(data, *args, **kwargs):
        if state["active"] and isinstance(data, bytes):
            state["slice"] = data
        return real_load(data, *args, **kwargs)

    piexif.load = load
    real_targets = metadata._ai_exif_targets

    def targets(loaded):
        found = real_targets(loaded)
        if state["active"]:
            record("ai_exif_values", state["slice"], [value.hex() for _ifd, _tag, value, _name in found])
        return found

    metadata._ai_exif_targets = targets
    real_find = isobmff._find_ai_exif_spans

    def find(*args, **kwargs):
        state["active"] = True
        try:
            return real_find(*args, **kwargs)
        finally:
            state["active"] = False

    isobmff._find_ai_exif_spans = find


# Deterministic stand-ins for the metadata-level strip decisions


def png_rule_drop(chunk_type: bytes, payload: bytes, keep_standard: bool) -> bool:
    if chunk_type == b"caBX":
        return True
    if chunk_type == b"iCCP":
        return not keep_standard
    if chunk_type in (b"tEXt", b"zTXt", b"iTXt"):
        return payload.partition(b"\x00")[0] in {b"parameters", b"prompt", b"Software"}
    return False


def jpeg_rule_drop(marker: int, payload: bytes) -> bool:
    return marker in (0xEB, 0xED) or (marker == 0xE1 and payload.startswith(b"http://ns.adobe.com/xap/"))


def strip_png(path: Path, work: Path, keep_standard: bool) -> object:
    saved = (metadata._png_chunk_should_be_dropped, metadata._scrubbed_png_exif, metadata.xai_signature)
    metadata._png_chunk_should_be_dropped = lambda t, p, k, _d, *, xai_pair_present=False: png_rule_drop(t, p, k)
    metadata._scrubbed_png_exif = lambda payload: payload
    metadata.xai_signature = lambda _p: False
    try:
        out = work / f"png-{keep_standard}.png"
        ok = metadata._strip_png_metadata_lossless(path, out, keep_standard)
        return sha(out.read_bytes()) if ok else None
    finally:
        metadata._png_chunk_should_be_dropped, metadata._scrubbed_png_exif, metadata.xai_signature = saved


def strip_jpeg(path: Path, work: Path) -> object:
    saved = (metadata._jpeg_app_carries_ai, piexif.load)
    metadata._jpeg_app_carries_ai = jpeg_rule_drop

    def refuse(*_args, **_kwargs):
        raise ValueError("EXIF scrub belongs to the metadata crate")

    piexif.load = refuse
    try:
        out = work / "strip.jpg"
        ok = metadata._strip_jpeg_metadata_lossless(path, out)
        return sha(out.read_bytes()) if ok else None
    finally:
        metadata._jpeg_app_carries_ai, piexif.load = saved


# Per-input report


def isobmff_report(path: Path, data: bytes, work: Path) -> dict[str, object]:
    size = len(data)
    with path.open("rb") as stream:
        boxes = [[s, e, t.hex(), p] for s, e, t, p in isobmff.iter_file_boxes(stream, 0, size)]
        ends = [box[1] for box in boxes]
    region = isobmff.scan_c2pa_region(path)
    report: dict[str, object] = {
        "boxes": boxes,
        "tiling": metadata._tiling_reason("ISOBMFF box", ends, size),
        "region": [len(region), sha(region)],
        "xai_in_region": isobmff.isobmff_xai_signature_in_region(region),
        "xai_payloads": [sha(v) for v in isobmff.isobmff_xai_signature_payloads(path)],
        "tc260_payloads": [sha(v) for v in isobmff.tc260_aigc_payloads(path)],
        "ai_metadata_tags": isobmff.ai_metadata_tags(path),
        "audio_creation": isobmff.has_audio_creation_comment(path),
        "heif_transform": isobmff.heif_transform_applied(path),
    }
    for key, first in (("sei", False), ("sei_first", True)):
        scan = isobmff.scan_isobmff_video_sei(path, stop_on_first_kling=first)
        report[key] = {
            "has_parsed_tracks": scan.has_parsed_tracks,
            "kling_tag_offsets": list(scan.kling_tag_offsets),
            "unregistered_uuids": list(scan.unregistered_uuids),
            "ai_sei_spans": [list(s) for s in scan.ai_sei_spans],
            "av1_obu_patches": [list(p) for p in scan.av1_obu_patches],
        }
    for key, fn in (
        ("strip_c2pa_boxes", isobmff.strip_c2pa_boxes),
        ("blank_xmp", isobmff.blank_ai_xmp_packets),
        ("blank_exif", isobmff.blank_ai_exif_tokens),
        ("blank_tc260", isobmff.blank_tc260_aigc_tags),
    ):
        out, count = fn(data)
        report[key] = [sha(out), count]
    for audio in (False, True):
        out = work / f"media-{audio}{path.suffix}"
        counts = isobmff.strip_isobmff_media_file(path, out, strip_audio_comments=audio)
        report[f"strip_media_{str(audio).lower()}"] = [sha(out.read_bytes()), list(counts)]
    out = work / f"image{path.suffix}"
    counts = isobmff.strip_isobmff_image_file(path, out)
    report["strip_image"] = [sha(out.read_bytes()), list(counts)]
    return report


def ebml_report(path: Path, data: bytes, work: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        ends = list(ebml.iter_top_level_ends(stream, len(data)))
    total, ordinals = ebml.attachment_ordinals(path)
    copy = work / f"scrub{path.suffix}"
    shutil.copyfile(path, copy)
    scrubbed = ebml.scrub_ebml_av1_ai_obus(copy)
    return {
        "ends": ends,
        "tiling": metadata._tiling_reason("EBML element", ends, len(data)),
        "tc260_payloads": [sha(v) for v in ebml.tc260_aigc_payloads(path)],
        "c2pa_attachments": [sha(v) for v in ebml.c2pa_attachment_payloads(path)],
        "attachment_ordinals": [total, list(ordinals)],
        "av1_obus": [list(p) for p in ebml.scan_ebml_av1_ai_obus(path)],
        "av1_obus_first": [list(p) for p in ebml.scan_ebml_av1_ai_obus(path, stop_on_first=True)],
        "scrub": [scrubbed, sha(copy.read_bytes())],
    }


def riff_report(path: Path, data: bytes) -> dict[str, object]:
    declared = 8 + struct.unpack_from("<I", data, 4)[0] if len(data) >= 8 else None
    late = {str(w): sha(metadata._riff_late_metadata(path, w)) for w in (0, 64, 1024 * 1024)}
    return {
        "declared": declared,
        "avi_tc260_payloads": [sha(v) for v in riff.tc260_aigc_payloads(path)],
        "late": late,
    }


def png_report(path: Path, data: bytes, work: Path) -> dict[str, object]:
    return {
        "bit_depth": metadata._png_bit_depth(data),
        "late": {str(w): sha(metadata._png_late_metadata(path, w)) for w in (0, 64, 1024 * 1024)},
        "strip_keep": strip_png(path, work, True),
        "strip_all": strip_png(path, work, False),
    }


def jpeg_report(path: Path, data: bytes, work: Path) -> dict[str, object]:
    return {
        "app_payloads": [[marker, sha(payload)] for marker, payload in metadata._jpeg_app_payloads(data)],
        "samsung_trailer": sha(metadata._strip_samsung_trailer(data)),
        "strip": strip_jpeg(path, work),
    }


def gif_report(path: Path, work: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "late": {str(w): sha(metadata._gif_late_metadata(path, w)) for w in (0, 64)},
    }
    for keep in (True, False):
        out = work / f"gif-{keep}.gif"
        ok = metadata._strip_gif_metadata_lossless(path, out, keep)
        report[f"strip_{str(keep).lower()}"] = sha(out.read_bytes()) if ok else None
    return report


def tiff_report(path: Path, work: Path) -> dict[str, object]:
    report: dict[str, object] = {}
    for keep in (True, False):
        out = work / f"tiff-{keep}.tif"
        ok = tiff_strip.strip_tiff_metadata_lossless(path, out, keep_standard=keep)
        report[f"strip_{str(keep).lower()}"] = sha(out.read_bytes()) if ok else None
    return report


def flv_report(path: Path) -> dict[str, object]:
    return {"tc260_payloads": [sha(v) for v in flv.tc260_aigc_payloads(path)]}


def describe(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    report: dict[str, object] = {"size": len(data), "sha256": sha(data)}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if isobmff.is_isobmff(data[:8]) or data.startswith(metadata._JXL_CONTAINER_MAGIC):
            report["isobmff"] = isobmff_report(path, data, work)
        if data[:4] == ebml.EBML_MAGIC:
            report["ebml"] = ebml_report(path, data, work)
        if data[:4] == b"RIFF":
            report["riff"] = riff_report(path, data)
        if data[:3] == b"FLV":
            report["flv"] = flv_report(path)
        if data[:8] == metadata.PNG_SIGNATURE:
            report["png"] = png_report(path, data, work)
        if data[:2] == b"\xff\xd8":
            report["jpeg"] = jpeg_report(path, data, work)
        if data[:6] in (b"GIF87a", b"GIF89a"):
            report["gif"] = gif_report(path, work)
        if tiff_strip.is_classic_tiff(data[:4]) or data[:4] in (b"II+\x00", b"MM\x00+"):
            report["tiff"] = tiff_report(path, work)
    return report


# Synthetic builders

C2PA_UUID = metadata.C2PA_UUID
KLING = metadata.KLING_SEI_TAG
TC260 = b'{"Label":"1","ContentProducer":"001191440300708461136T","ProduceID":"abc"}'
XAI = b"Signature: " + base64.b64encode(bytes(range(60)))
SUNO = b"made with suno; created=2025-01-02T03:04:05Z; id=123e4567-e89b-12d3-a456-426614174000"
DREAMINA = b'{"product":"dreamina","exportType":"generation"}'
XMP_AI = (
    b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?><x:xmpmeta><rdf:Description '
    b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
    b'</x:xmpmeta><?xpacket end="w"?>'
)
XMP_PLAIN = b'<?xpacket begin="" id="x"?><x:xmpmeta><dc:rights>me</dc:rights></x:xmpmeta><?xpacket end="r"?>'


def box(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def full(kind: bytes, payload: bytes = b"", version: int = 0) -> bytes:
    return box(kind, bytes([version, 0, 0, 0]) + payload)


def ftyp(brand: bytes = b"isom") -> bytes:
    return box(b"ftyp", brand + b"\x00\x00\x02\x00" + brand + b"mp41")


def hdlr(handler: bytes) -> bytes:
    return full(b"hdlr", b"\x00" * 4 + handler + b"\x00" * 12 + b"\x00")


def data_item(value: bytes) -> bytes:
    return box(b"data", b"\x00\x00\x00\x01\x00\x00\x00\x00" + value)


def keyed_meta(entries: list[tuple[bytes, bytes]], *, iso: bool = True) -> bytes:
    keys = b"".join(struct.pack(">I", 8 + len(name)) + b"mdta" + name for name, _ in entries)
    keys_box = full(b"keys", struct.pack(">I", len(entries)) + keys)
    items = b"".join(box(struct.pack(">I", i + 1), data_item(value)) for i, (_, value) in enumerate(entries))
    children = hdlr(b"mdta") + keys_box + box(b"ilst", items)
    return full(b"meta", children) if iso else box(b"meta", children)


def quicktime_meta(items: list[tuple[bytes, bytes]]) -> bytes:
    ilst = b"".join(box(kind, data_item(value)) for kind, value in items)
    return box(b"meta", hdlr(b"mdir") + box(b"ilst", ilst))


def sample_entry(kind: bytes, children: bytes) -> bytes:
    return box(kind, b"\x00" * 78 + children)


def stbl(entry: bytes, sizes: list[int], offsets: list[int], *, co64: bool = False, fixed: int = 0) -> bytes:
    stsd = full(b"stsd", struct.pack(">I", 1) + entry)
    stsc = full(b"stsc", struct.pack(">I", 1) + struct.pack(">III", 1, len(sizes) // max(len(offsets), 1), 1))
    if fixed:
        stsz = full(b"stsz", struct.pack(">II", fixed, len(sizes)))
    else:
        stsz = full(b"stsz", struct.pack(">II", 0, len(sizes)) + b"".join(struct.pack(">I", s) for s in sizes))
    if co64:
        co = full(b"co64", struct.pack(">I", len(offsets)) + b"".join(struct.pack(">Q", o) for o in offsets))
    else:
        co = full(b"stco", struct.pack(">I", len(offsets)) + b"".join(struct.pack(">I", o) for o in offsets))
    return box(b"stbl", stsd + stsc + stsz + co)


def trak(handler: bytes, stbl_box: bytes) -> bytes:
    return box(b"trak", box(b"mdia", hdlr(handler) + box(b"minf", stbl_box)))


def nal(body: bytes, size_len: int = 4) -> bytes:
    return len(body).to_bytes(size_len, "big") + body


def sei_message(payload_type: int, payload: bytes) -> bytes:
    out = bytearray()
    for value in (payload_type, len(payload)):
        while value >= 255:
            out.append(255)
            value -= 255
        out.append(value)
    return bytes(out) + payload


def video_file(codec: str, samples: list[bytes], *, moof: bool = False, extra_moov: bytes = b"") -> bytes:
    head = ftyp()
    mdat_payload = b"".join(samples)
    offsets, cursor = [], len(head) + 8
    for sample in samples:
        offsets.append(cursor)
        cursor += len(sample)
    if codec == "avc":
        entry = sample_entry(b"avc1", box(b"avcC", b"\x01\x64\x00\x1f\xff\xe1"))
    elif codec == "avc2":
        entry = sample_entry(b"avc1", box(b"avcC", b"\x01\x64\x00\x1f\xfd\xe1"))
    elif codec == "hevc":
        entry = sample_entry(b"hvc1", box(b"hvcC", b"\x01" + b"\x00" * 20 + b"\x0f" + b"\x00" * 3))
    elif codec == "av1":
        config_obu = b"\x2a" + bytes([len(av1_metadata(6, b"urn:c2pa:cfg"))]) + av1_metadata(6, b"urn:c2pa:cfg")
        entry = sample_entry(b"av01", box(b"av1C", b"\x81\x00\x0c\x00" + config_obu))
    else:
        entry = sample_entry(b"vp09", box(b"vpcC", b"\x01\x00\x00\x00"))
    moov = box(
        b"moov",
        trak(b"vide", stbl(entry, [len(s) for s in samples], offsets))
        + trak(b"soun", stbl(sample_entry(b"mp4a", b""), [4], [offsets[0]]))
        + extra_moov,
    )
    return head + box(b"mdat", mdat_payload) + moov + (box(b"moof", box(b"mfhd", b"\x00" * 8)) if moof else b"")


def av1_metadata(meta_type: int, body: bytes) -> bytes:
    return av1_obu.encode_leb128(meta_type) + body


def av1_obu_unit(obu_type: int, payload: bytes, *, ext: bool = False) -> bytes:
    header = (obu_type << 3) | 0x02 | (0x04 if ext else 0)
    return bytes([header]) + (b"\x00" if ext else b"") + av1_obu.encode_leb128(len(payload)) + payload


def isobmff_cases() -> dict[str, bytes]:
    kling_sei = b"\x06" + sei_message(5, KLING + b"payload") + b"\x80"
    c2pa_sei = b"\x06" + sei_message(5, C2PA_UUID + b"jumb c2pa manifest") + b"\x80"
    escaped = b"\x06" + sei_message(5, KLING + b"\x00\x00\x03\x01" + bytes(16) + b"\x00\x00\x03\x00") + b"\x80"
    x264 = b"\x06" + sei_message(5, bytes.fromhex("dc45e9bde6d948b7962cd820d923eeef") + b"x264 core") + b"\x80"
    other_uuid = b"\x06" + sei_message(5, bytes(range(16)) + b'"contentproducer" "label"') + b"\x80"
    slice_nal = b"\x65\x88\x84\x00\x10"
    avc_samples = [
        nal(kling_sei) + nal(slice_nal),
        nal(c2pa_sei) + nal(x264) + nal(other_uuid),
        nal(escaped) + nal(slice_nal),
        nal(slice_nal) + b"\x00\x00\xff\xff",
    ]
    hevc_sei = b"\x4e\x01" + sei_message(5, KLING + b"hevc") + b"\x80"
    av1_samples = [
        av1_obu_unit(2, b"") + av1_obu_unit(5, av1_metadata(4, b"\xb5\x00\x3bkling-ai")) + av1_obu_unit(6, b"\x10" * 8),
        av1_obu_unit(5, av1_metadata(1, b"\x00" * 4)) + av1_obu_unit(5, av1_metadata(6, b'{"ContentProducer":1,"Label":2}'), ext=True),
        av1_obu_unit(5, av1_metadata(4, b"\xff\x00\x00tc260:aigc")),
    ]
    udta_meta = keyed_meta([(b"AIGC", TC260), (b"comment", XAI), (b"prompt", b"a cat"), (b"LvMetaInfo", DREAMINA)])
    cases = {
        "c2pa-uuid": ftyp() + box(b"uuid", C2PA_UUID + b"manifest") + box(b"free", b"x") + box(b"mdat", b"\x00" * 32),
        "xmp-uuid": ftyp() + box(b"uuid", bytes(16) + XMP_AI) + box(b"uuid", bytes(16) + XMP_PLAIN) + box(b"mdat", b"d"),
        "generator-uuid": ftyp() + box(b"uuid", bytes(16) + b"<xmp:CreatorTool>Midjourney</xmp:CreatorTool>"),
        "jumb-emsg": ftyp() + box(b"jumb", b"jumd c2pa") + full(b"emsg", b"urn:c2pa:manifest") + full(b"emsg", b"plain"),
        "largesize": ftyp() + struct.pack(">I", 1) + b"uuid" + struct.pack(">Q", 16 + 16 + 4) + C2PA_UUID + b"abcd",
        "size-zero": ftyp() + box(b"free", b"zz") + struct.pack(">I", 0) + b"mdat" + b"\x00" * 40,
        "truncated": ftyp() + box(b"uuid", C2PA_UUID + b"m") + struct.pack(">I", 4096) + b"mdat" + b"\x00" * 10,
        "tiny-box": ftyp() + struct.pack(">I", 4) + b"free" + b"\x00" * 8,
        "iso-udta-meta": ftyp(b"mp42")
        + box(b"moov", box(b"udta", udta_meta))
        + box(b"mdat", KLING + b"\x00" * 8),
        "quicktime-meta": ftyp(b"qt  ")
        + box(b"moov", quicktime_meta([(b"\xa9cmt", XAI), (b"\xa9too", TC260), (b"\xa9cmt", SUNO)]))
        + box(b"mdat", b"\x00" * 4),
        "keyed-topaz-description": ftyp()
        + box(
            b"moov",
            box(
                b"udta",
                keyed_meta(
                    [
                        (b"videoai", b"Topaz Video AI 5"),
                        (b"com.apple.quicktime.description", b" AI-enhanced using Topaz Labs \n"),
                        (b"com.apple.quicktime.comment", b"Signature: shortblob"),
                        (b"\xff\xfe", b"bad-utf8-name"),
                    ]
                ),
            ),
        ),
        "unmapped-key": ftyp() + box(b"moov", box(b"udta", full(b"meta", hdlr(b"mdta") + full(b"keys", struct.pack(">I", 0)) + box(b"ilst", box(struct.pack(">I", 7), data_item(TC260)))))),
        "audio-creation": ftyp(b"M4A ") + box(b"moov", box(b"udta", quicktime_meta([(b"\xa9cmt", SUNO)]))) + box(b"mdat", b"aac"),
        "avc-sei": video_file("avc", avc_samples),
        "avc-sei-2byte": video_file("avc2", [nal(kling_sei, 2) + nal(slice_nal, 2)]),
        "hevc-sei": video_file("hevc", [nal(hevc_sei) + nal(b"\x26\x01\xaf")]),
        "av1-obu": video_file("av1", av1_samples),
        "vp9-track": video_file("vp9", [b"\x00" * 16]),
        "moof-kling": video_file("avc", [nal(kling_sei)], moof=True),
        "heif-irot": ftyp(b"heic")
        + full(b"meta", hdlr(b"pict") + box(b"iprp", box(b"ipco", box(b"irot", b"\x01")))),
        "heif-irot-zero": ftyp(b"heic") + full(b"meta", hdlr(b"pict") + box(b"iprp", box(b"ipco", box(b"irot", b"\x04")))),
        "heif-imir-bare": ftyp(b"avif") + box(b"meta", hdlr(b"pict") + box(b"iprp", box(b"ipco", box(b"imir", b"\x00")))),
        "heif-unknown-meta": ftyp(b"heic") + box(b"meta", b"\x00" * 12),
        "heif-xmp-exif": ftyp(b"heic")
        + full(b"meta", hdlr(b"pict") + box(b"iinf", b"\x00" * 6))
        + box(
            b"mdat",
            b"\x00" * 7
            + XMP_AI
            + b"\x00" * 3
            + XMP_PLAIN
            + b"Exif\x00\x00"
            + piexif.dump({"0th": {piexif.ImageIFD.Software: b"Midjourney v6", piexif.ImageIFD.Make: b"Canon"}})[6:]
            + b"\x00" * 5,
        ),
        "jxl-container": metadata._JXL_CONTAINER_MAGIC + box(b"ftyp", b"jxl \x00\x00\x00\x00jxl ") + box(b"jumb", b"jumd") + box(b"jxlc", b"\xff\x0a"),
    }
    big_tail = b"\x00" * (1024 * 1024 + 4096)
    cases["late-moov"] = ftyp() + box(b"mdat", big_tail) + box(b"moov", box(b"udta", keyed_meta([(b"AIGC", TC260)])))
    cases["split-xmp"] = ftyp(b"heic") + box(b"mdat", b"\x00" * (1024 * 1024 - 40) + XMP_AI + b"\x00" * 64)
    return cases


def ebml_element(element_id: int, payload: bytes, *, unknown: bool = False) -> bytes:
    raw_id = element_id.to_bytes((element_id.bit_length() + 7) // 8, "big")
    if unknown:
        return raw_id + b"\x01\xff\xff\xff\xff\xff\xff\xff" + payload
    size = len(payload)
    for length in range(1, 9):
        if size < (1 << (7 * length)) - 1:
            break
    marker = 1 << (7 * length)
    return raw_id + (marker | size).to_bytes(length, "big") + payload


def ebml_cases() -> dict[str, bytes]:
    header = ebml_element(0x1A45DFA3, ebml_element(0x4282, b"webm"))
    simple = ebml_element(0x67C8, ebml_element(0x45A3, b"AIGC") + ebml_element(0x4487, TC260))
    other = ebml_element(0x67C8, ebml_element(0x45A3, b"TITLE") + ebml_element(0x4487, TC260))
    tags = ebml_element(0x1254C367, ebml_element(0x7373, simple + other))
    attachment = ebml_element(
        0x61A7, ebml_element(0x4660, b"application/c2pa\x00") + ebml_element(0x465C, b"jumbjumd c2pa store")
    )
    plain_attachment = ebml_element(0x61A7, ebml_element(0x4660, b"image/png") + ebml_element(0x465C, b"png"))
    attachments = ebml_element(0x1941A469, plain_attachment + attachment)
    ai_obu = av1_obu_unit(5, av1_metadata(6, b"urn:c2pa:frame"))
    hdr_obu = av1_obu_unit(5, av1_metadata(2, b"\x00" * 6))
    cfg = b"\x81\x00\x0c\x00" + av1_obu_unit(5, av1_metadata(6, b"kling-ai cfg"))
    track = ebml_element(
        0xAE,
        ebml_element(0xD7, b"\x01") + ebml_element(0x86, b"V_AV1") + ebml_element(0x63A2, cfg),
    )
    track2 = ebml_element(0xAE, ebml_element(0xD7, b"\x02") + ebml_element(0x86, b"A_OPUS"))
    tracks_body = track + track2
    tracks = ebml_element(0x1654AE6B, ebml_element(0xBF, zlib.crc32(tracks_body).to_bytes(4, "little")) + tracks_body)
    frame_a = av1_obu_unit(2, b"") + ai_obu
    frame_b = hdr_obu + av1_obu_unit(6, b"\x11" * 4)
    simple_block = ebml_element(0xA3, b"\x81\x00\x00\x80" + frame_a)
    xiph = ebml_element(0xA3, b"\x81\x00\x01\x82" + bytes([1, len(frame_a)]) + frame_b + frame_a)
    fixed = ebml_element(0xA3, b"\x81\x00\x02\x84" + b"\x01" + frame_a + frame_a)
    ebml_lace = ebml_element(
        0xA3,
        b"\x81\x00\x03\x86" + b"\x02" + bytes([0x80 | len(frame_a)]) + b"\xbf" + frame_a + frame_a + frame_b,
    )
    other_track = ebml_element(0xA3, b"\x82\x00\x00\x80" + ai_obu)
    block_group = ebml_element(0xA0, ebml_element(0xA1, b"\x81\x00\x00\x00" + ai_obu))
    cluster_body = ebml_element(0xE7, b"\x00") + simple_block + xiph + fixed + ebml_lace + other_track + block_group
    cluster = ebml_element(0x1F43B675, ebml_element(0xBF, zlib.crc32(cluster_body).to_bytes(4, "little")) + cluster_body)
    segment = ebml_element(0x18538067, tags + attachments + tracks + cluster)
    unknown_segment = ebml_element(0x18538067, tags + tracks + cluster, unknown=True)
    return {
        "mkv-full": header + segment,
        "mkv-unknown-size": header + unknown_segment,
        "mkv-truncated": (header + segment)[:-7],
        "mkv-bad-id": header + b"\x00\x00\x00\x00" + segment,
        "mkv-encoded-track": header
        + ebml_element(
            0x18538067,
            ebml_element(0x1654AE6B, ebml_element(0xAE, ebml_element(0xD7, b"\x01") + ebml_element(0x86, b"V_AV1") + ebml_element(0x6D80, b"")))
            + ebml_element(0x1F43B675, simple_block),
        ),
    }


def riff_chunk(kind: bytes, payload: bytes) -> bytes:
    return kind + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) & 1 else b"")


def riff_cases() -> dict[str, bytes]:
    info = b"INFO" + riff_chunk(b"INAM", b"odd") + riff_chunk(b"AIGC", TC260 + b"\x00 ") + riff_chunk(b"AIGC", b"{}")
    body = b"AVI " + riff_chunk(b"LIST", b"hdrl" + riff_chunk(b"avih", b"\x00" * 56)) + riff_chunk(b"LIST", info)
    avi = b"RIFF" + struct.pack("<I", len(body)) + body
    webp_body = b"WEBP" + riff_chunk(b"VP8L", b"\x2f" + b"\x00" * 99) + riff_chunk(b"XMP ", XMP_AI) + riff_chunk(b"EXIF", b"Exif\x00\x00II")
    webp = b"RIFF" + struct.pack("<I", len(webp_body)) + webp_body
    huge = b"RIFF" + struct.pack("<I", 4 + 8 + 50) + b"WEBP" + b"XMP " + struct.pack("<I", 0x7FFFFFFF) + XMP_AI[:50]
    return {
        "avi-tc260": avi,
        "avi-truncated": avi[:-20],
        "avi-declared-short": b"RIFF" + struct.pack("<I", 4) + body,
        "webp-late": webp,
        "webp-clamped": huge,
        "riff-oversized-header": b"RIFF" + struct.pack("<I", 1 << 20) + b"WAVE",
    }


def amf_string(value: bytes) -> bytes:
    return struct.pack(">H", len(value)) + value


def flv_cases() -> dict[str, bytes]:
    def tag(kind: int, payload: bytes) -> bytes:
        return bytes([kind]) + len(payload).to_bytes(3, "big") + b"\x00" * 7 + payload + struct.pack(">I", 11 + len(payload))

    header = b"FLV\x01\x05\x00\x00\x00\x09" + b"\x00\x00\x00\x00"
    nested = (
        amf_string(b"duration") + b"\x00" + struct.pack(">d", 1.5)
        + amf_string(b"stereo") + b"\x01\x01"
        + amf_string(b"title") + b"\x02" + amf_string(b"hello")
        + amf_string(b"date") + b"\x0b" + b"\x00" * 10
        + amf_string(b"list") + b"\x0a" + struct.pack(">I", 2) + b"\x05" + b"\x06"
        + amf_string(b"obj") + b"\x03" + amf_string(b"k") + b"\x00" + b"\x00" * 8 + b"\x00\x00\x09"
        + amf_string(b"ref") + b"\x07\x00\x01"
        + amf_string(b"long") + b"\x0c" + struct.pack(">I", 3) + b"abc"
    )
    ecma = b"\x02" + amf_string(b"onMetaData") + b"\x08" + struct.pack(">I", 9) + nested + amf_string(b"AIGC") + b"\x02" + amf_string(TC260) + b"\x00\x00\x09"
    obj = b"\x02" + amf_string(b"onMetaData") + b"\x03" + amf_string(b"AIGC") + b"\x0c" + struct.pack(">I", len(TC260)) + TC260 + b"\x00\x00\x09"
    bad = b"\x02" + amf_string(b"onMetaData") + b"\x03" + amf_string(b"x") + b"\x11" + amf_string(b"AIGC") + b"\x02" + amf_string(TC260)
    video = tag(9, b"\x17\x00" + b"\x00" * 20)
    return {
        "flv-ecma": header + video + tag(18, ecma),
        "flv-object-long": header + tag(18, obj),
        "flv-unknown-type": header + tag(18, bad),
        "flv-not-onmetadata": header + tag(18, b"\x02" + amf_string(b"onCuePoint") + b"\x03" + amf_string(b"AIGC") + b"\x02" + amf_string(TC260) + b"\x00\x00\x09"),
        "flv-truncated": (header + tag(18, ecma))[:-6],
    }


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def png_cases() -> dict[str, bytes]:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 16, 2, 0, 0, 0))
    idat = png_chunk(b"IDAT", zlib.compress(b"\x00" * 26))
    iend = png_chunk(b"IEND", b"")
    text = png_chunk(b"tEXt", b"parameters\x00steps: 20") + png_chunk(b"tEXt", b"Title\x00ok") + png_chunk(b"iCCP", b"icc\x00\x00x")
    late = png_chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + XMP_AI) + png_chunk(b"eXIf", b"MM\x00*\x00\x00\x00\x08")
    return {
        "png-text": sig + ihdr + text + png_chunk(b"caBX", b"jumbc2pa") + idat + late + iend,
        "png-trailing": sig + ihdr + idat + iend + b"garbage",
        "png-no-iend": sig + ihdr + idat,
        "png-inflated-length": sig + ihdr + idat + struct.pack(">I", 0xFFFFFF00) + b"tEXt" + b"prompt\x00x",
        "png-8bit": sig + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)) + idat + iend,
    }


def jpeg_segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def jpeg_cases() -> dict[str, bytes]:
    soi, eoi = b"\xff\xd8", b"\xff\xd9"
    app0 = jpeg_segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    xmp = jpeg_segment(0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + XMP_AI)
    exif = jpeg_segment(0xE1, b"Exif\x00\x00MM\x00*\x00\x00\x00\x08\x00\x00")
    c2pa = jpeg_segment(0xEB, b"JP\x00\x00\x00\x01jumbc2pa")
    iptc = jpeg_segment(0xED, b"Photoshop 3.0\x008BIM\x04\x04\x00\x00")
    sos = jpeg_segment(0xDA, b"\x01\x01\x00\x00\x3f\x00") + b"\x12\x34\xff\x00\x56"
    base = soi + app0 + b"\xff\x01" + exif + xmp + c2pa + iptc + jpeg_segment(0xFE, b"comment") + sos
    return {
        "jpeg-apps": base + eoi,
        "jpeg-samsung": base + eoi + b"\x00\x00SEFH PhotoEditor_Re_Edit_Data genAIType",
        "jpeg-samsung-in-scan": soi + sos[:-1] + b"PhotoEditor_Re_Edit_Data" + eoi + b"tail",
        "jpeg-rst": soi + b"\xff\xd0" + app0 + sos + eoi,
        "jpeg-bad-length": soi + app0 + b"\xff\xe1\x00\x01" + xmp,
        "jpeg-truncated": soi + app0 + b"\xff\xe1\x00",
        "jpeg-no-marker": soi + app0 + b"\x00\x11" + eoi,
    }


def gif_sub_blocks(data: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(data), 255):
        part = data[i : i + 255]
        out += bytes([len(part)]) + part
    return bytes(out) + b"\x00"


def gif_cases() -> dict[str, bytes]:
    screen = b"GIF89a" + struct.pack("<HH", 2, 2) + b"\x80\x00\x00" + b"\x00\x00\x00\xff\xff\xff"
    netscape = b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"
    icc = b"\x21\xff\x0bICCRGBG1012" + gif_sub_blocks(b"icc-profile")
    xmp = b"\x21\xff\x0bXMP DataXMP" + gif_sub_blocks(XMP_AI * 3)
    comment = b"\x21\xfe" + gif_sub_blocks(b"Made with Midjourney " * 20)
    gce = b"\x21\xf9\x04\x00\x0a\x00\x00\x00"
    frame = b"\x2c" + struct.pack("<HHHH", 0, 0, 2, 2) + b"\x81" + b"\x00" * 12 + b"\x02" + gif_sub_blocks(b"\x44\x01")
    plain = screen + netscape + icc + comment + gce + frame + xmp + gce + frame + b"\x3b"
    return {
        "gif-anim": plain,
        "gif87-no-gct": b"GIF87a" + struct.pack("<HH", 1, 1) + b"\x00\x00\x00" + frame + b"\x3b",
        "gif-truncated": plain[:-30],
        "gif-bad-introducer": screen + b"\x99" + b"\x3b",
        "gif-short-ext": screen + b"\x21\xfe\x05ab",
    }


def tiff_ifd(endian: str, entries: list[tuple[int, int, int, bytes]], next_ifd: int, base: int, extra: list[bytes]) -> tuple[bytes, bytes]:
    """Return (ifd_bytes, out_of_line_blob) with values placed after the IFD at ``base``."""
    count = len(entries)
    blob = bytearray()
    table = bytearray(struct.pack(f"{endian}H", count))
    data_start = base + 2 + 12 * count + 4
    for tag, kind, n, value in sorted(entries):
        if len(value) <= 4:
            field = value.ljust(4, b"\x00")
        else:
            field = struct.pack(f"{endian}I", data_start + len(blob))
            blob += value + (b"\x00" if len(value) & 1 else b"")
        table += struct.pack(f"{endian}HHI", tag, kind, n) + field
    table += struct.pack(f"{endian}I", next_ifd)
    return bytes(table), bytes(blob)


def tiff_file(endian: str) -> bytes:
    magic = b"II*\x00" if endian == "<" else b"MM\x00*"
    pixel = b"\x10\x20\x30\x40"
    pixel_off = 8
    ifd_off = 8 + len(pixel)
    desc = b"Generated by Midjourney\x00"
    soft = b"GIMP 2.10\x00"
    xmp = XMP_PLAIN
    icc = b"\x00" * 24
    entries = [
        (256, 3, 1, struct.pack(f"{endian}H", 2)),
        (257, 3, 1, struct.pack(f"{endian}H", 2)),
        (273, 4, 1, struct.pack(f"{endian}I", pixel_off)),
        (279, 4, 1, struct.pack(f"{endian}I", len(pixel))),
        (270, 2, len(desc), desc),
        (305, 2, len(soft), soft),
        (700, 1, len(xmp), xmp),
        (34675, 7, len(icc), icc),
        (40092, 1, 8, "AI\x00".encode("utf-16-le")[:6] + b"\x00\x00"),
    ]
    first, blob = tiff_ifd(endian, entries + [(34665, 4, 1, b"\x00\x00\x00\x00")], 0, ifd_off, [])
    exif_off = ifd_off + len(first) + len(blob)
    exif_comment = b"UNICODE\x00" + "stable diffusion".encode("utf-16-be")
    exif_ifd, exif_blob = tiff_ifd(endian, [(37510, 7, len(exif_comment), exif_comment), (36867, 2, 20, b"2024:01:01 00:00:00\x00")], 0, exif_off, [])
    entries_final = entries + [(34665, 4, 1, struct.pack(f"{endian}I", exif_off))]
    first, blob = tiff_ifd(endian, entries_final, 0, ifd_off, [])
    head = magic + struct.pack(f"{endian}I", ifd_off)
    return head[:4] + struct.pack(f"{endian}I", ifd_off) + pixel + first + blob + exif_ifd + exif_blob


def tiff_cases() -> dict[str, bytes]:
    little = tiff_file("<")
    big = tiff_file(">")
    oob = bytearray(little)
    entry_270 = little.index(struct.pack("<HH", 270, 2))
    oob[entry_270 + 8 : entry_270 + 12] = struct.pack("<I", 0xFFFFFF00)
    return {
        "tiff-le": little,
        "tiff-be": big,
        "tiff-truncated": little[:40],
        "tiff-oob-tag": bytes(oob),
        "bigtiff": b"II+\x00\x08\x00\x00\x00" + b"\x00" * 16,
        "tiff-bad-first-ifd": b"II*\x00\x04\x00\x00\x00" + b"\x00" * 8,
    }


def synthetic_cases() -> dict[str, bytes]:
    cases: dict[str, bytes] = {}
    for group in (isobmff_cases, ebml_cases, riff_cases, flv_cases, png_cases, jpeg_cases, gif_cases, tiff_cases):
        cases.update(group())
    return cases


# Direct unit goldens


def units() -> dict[str, object]:
    leb_inputs = [b"\x00", b"\x7f", b"\x80\x01", b"\xff\xff\xff\xff\x0f", b"\xff\xff\xff\xff\x10", b"\x80" * 8, b"\x80\x80", b""]
    payloads = [
        av1_metadata(1, b"urn:c2pa:"),
        av1_metadata(4, b"\xb5urn:c2pa:x"),
        av1_metadata(4, b"\xff\x00\x00kling-ai"),
        av1_metadata(4, b"\xff"),
        av1_metadata(4, b""),
        av1_metadata(6, b"jumb C2PA"),
        av1_metadata(6, b"jumb only"),
        av1_metadata(6, C2PA_UUID),
        av1_metadata(6, b'"ServiceProvider" "AIGC"'),
        av1_metadata(6, b"TrainedAlgorithmicMedia"),
        av1_metadata(6, b""),
        b"\x80",
    ]
    stream = (
        av1_obu_unit(2, b"")
        + av1_obu_unit(5, av1_metadata(6, b"urn:c2pa:a"))
        + av1_obu_unit(5, av1_metadata(6, b"urn:c2pa:b"), ext=True)
        + bytes([0x30]) + av1_metadata(6, b"kling-ai tail")
    )
    gif_blocks = [b"\x03abc\x02de\x00", b"\x03ab", b"\x00", b"", b"\x05abc"]
    gif_exts = [
        b"\x21\xff\x0bXMP DataXMP\x03abc\x00",
        b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00",
        b"\x21\xff\x0bICCRGBG1012\x01x\x00",
        b"\x21\xff\x0bXMP DataXMP\x09ab",
        b"\x21\xfe\x02hi\x00",
        b"\x21\xfe\x09hi",
        b"\x21\xf9\x04\x00\x00\x00\x00\x00",
        b"\x21",
        b"\x2c\xfe\x00",
    ]
    jpeg_buffers = [
        b"\xff\xd8\xff\xe1\x00\x05abc\xff\xed\x00\x04xy\xff\xda",
        b"\xff\xe1\x00\x05abc\xff\xd0\xff\xed\x00\x04xy",
        b"\xff\xe1\x00\x09abc",
        b"\x00\xff\xe1",
        b"\xff\xd8\xff\x01\xff\xe0\x00\x02\xff\xe1\x00\x03z",
    ]
    region_inputs = [
        ftyp() + b"\xa9cmt\x00" + XAI + b"\x00",
        ftyp() + b"\xa9cmt\x00Signature: " + b"A" * 63 + b"\x00",
        ftyp() + b"\xa9cmt\x00Signature: " + b"A" * 66 + b"\x00",
        ftyp() + b"\xa9cmt\x00Signature: " + b"Q" * 62 + b"=A\x00",
        ftyp() + b"\xa9cmt\x00Signature:\t" + b"QR" * 32 + b"== \n\x00",
        ftyp() + b"\xa9cmt\x00" + XAI,
        b"no ftyp here" + XAI,
    ]
    return {
        "read_leb128": [[x.hex(), av1_obu.read_leb128(x, 0)] for x in leb_inputs],
        "encode_leb128": [[v, av1_obu.encode_leb128(v).hex()] for v in (0, 1, 127, 128, 300, 2**32 - 1, 2**40)],
        "is_ai_obu_metadata_payload": [[p.hex(), av1_obu.is_ai_obu_metadata_payload(p)] for p in payloads],
        "scan_av1_obu_ai_spans": [stream.hex(), [list(s) for s in av1_obu.scan_av1_obu_ai_spans(stream, 100)]],
        "scan_av1_obu_stream_spans": [
            [list(s) for s in av1_obu.scan_av1_obu_stream_spans(lambda o, n: stream[o : o + n], 0, len(stream))],
            [list(s) for s in av1_obu.scan_av1_obu_stream_spans(lambda o, n: stream[o : o + n], 0, len(stream), stop_on_first=True)],
        ],
        "gif_sub_block_payload": [[b.hex(), (lambda r: None if r is None else r.hex())(metadata._gif_sub_block_payload(b, 0))] for b in gif_blocks],
        "gif_extension_metadata": [[e.hex(), metadata._gif_extension_metadata(e).hex()] for e in gif_exts],
        "jpeg_app_payloads": [[b.hex(), [[m, p.hex()] for m, p in metadata._jpeg_app_payloads(b)]] for b in jpeg_buffers],
        "xai_in_region": [[r.hex(), isobmff.isobmff_xai_signature_in_region(r)] for r in region_inputs],
    }


SUFFIXES = (
    ("xmp", ".heic"),
    ("heif", ".heic"),
    ("split", ".heic"),
    ("quicktime", ".mov"),
    ("keyed", ".mov"),
    ("audio", ".m4a"),
    ("jxl", ".jxl"),
    ("mkv", ".mkv"),
    ("avi", ".avi"),
    ("webp", ".webp"),
    ("riff", ".wav"),
    ("flv", ".flv"),
    ("png", ".png"),
    ("jpeg", ".jpg"),
    ("gif", ".gif"),
    ("tiff", ".tif"),
    ("bigtiff", ".tif"),
    ("", ".mp4"),
)


def main() -> None:
    install_hooks()
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures: dict[str, object] = {}
    for path in sorted(FIXTURE_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
            rel = path.relative_to(FIXTURE_ROOT).as_posix()
            if rel.startswith(("provenance/", "visible/")):
                fixtures[rel] = describe(path)
    synthetic: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in synthetic_cases().items():
            suffix = next(s for prefix, s in SUFFIXES if name.startswith(prefix))
            path = Path(tmp) / f"{name}{suffix}"
            path.write_bytes(data)
            entry = describe(path)
            entry["suffix"] = suffix
            entry["input_zlib_b64"] = base64.b64encode(zlib.compress(data, 9)).decode()
            synthetic[name] = entry
    write("fixtures", fixtures)
    write("synthetic", synthetic)
    write("units", units())
    write("hooks", HOOKS)


def write(name: str, value: object) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    print(f"wrote {name}.json")


if __name__ == "__main__":
    main()

"""Golden generator for the `c2pa` suite (raiw-c2pa).

Run from the repository root: `.venv/bin/python rs-ai-watermarks/tools/goldens/c2pa.py`.

Records the derived `c2pa_info` (never the raw manifest store, whose URNs regenerate) for
every tracked image/video fixture, the raw fallback scan, the registries, and synthetic
stores and PNG chunks for the fail-closed and attribution branches.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import sys
import tempfile
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# This file is named c2pa.py: keep its directory off the path so `import c2pa` finds c2pa-python.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
sys.path.insert(0, str(REPO / "src"))

from remove_ai_watermarks._internal import c2pa, constants, extractor, isobmff  # noqa: E402
from remove_ai_watermarks._internal import _generated_c2pa_soft_bindings as generated  # noqa: E402
from remove_ai_watermarks import metadata  # noqa: E402

OUT = REPO / "rs-ai-watermarks" / "goldens" / "c2pa"
FIXTURE_ROOT = REPO / "data" / "fixtures"
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".m4a", ".heic", ".avif"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dumps(value: object) -> str:
    return json.dumps(value)


def reader_outcome(path: Path) -> str:
    try:
        store = c2pa._manifest_json_uncached(str(path), strict=True)
    except Exception as error:  # noqa: BLE001
        return f"error:{type(error).__name__}"
    return "none" if store is None else "store"


def raw_manifest_bytes(path: Path, data: bytes) -> bytes:
    """The raw bytes a fallback scan would see: caBX payload, APP11 payloads, or ISOBMFF region."""
    if data.startswith(b"\x89PNG"):
        chunk = c2pa._find_c2pa_chunk(path)
        return chunk.payload if chunk else b""
    if data.startswith(b"\xff\xd8"):
        out = bytearray()
        pos = 2
        while pos + 4 <= len(data) and data[pos] == 0xFF and data[pos + 1] not in (0xDA, 0xD9):
            length = int.from_bytes(data[pos + 2 : pos + 4], "big")
            if data[pos + 1] == 0xEB:
                out += data[pos + 4 : pos + 2 + length]
            pos += 2 + length
        return bytes(out)
    if isobmff.is_isobmff(data[:8]):
        return isobmff.scan_c2pa_region(path)
    if data.startswith(b"RIFF"):
        return metadata._riff_late_metadata(path, 0)
    return b""


def helpers(info: dict) -> dict:
    return {
        "credential_level": c2pa.c2pa_credential_level(info),
        "invalid_credential": c2pa.c2pa_info_has_invalid_credential(info),
        "invismark": c2pa.c2pa_info_has_invismark(info),
        "removal_hint": c2pa.c2pa_info_has_removal_hint(info),
        "claim_generator_platform": c2pa.claim_generator_platform(info.get("claim_generator")),
        "vendor_platform": c2pa.c2pa_vendor_platform(info.get("issuer", "")),
        "markers_claim_ai": c2pa.markers_claim_ai({k: str(v) for k, v in info.items() if k not in {"has_c2pa", "type"}}),
    }


def describe(path: Path) -> dict:
    data = path.read_bytes()
    c2pa._extract_c2pa_info_cached.cache_clear()
    info = c2pa.extract_c2pa_info(path)
    png_info = c2pa._extract_c2pa_info_png(path)
    raw = raw_manifest_bytes(path, data)
    fallback = {}
    if raw:
        c2pa.parse_c2pa_chunk(raw, fallback)
    chunk = c2pa.extract_c2pa_chunk(path)
    report = {
        "sha256": sha(data),
        "reader": reader_outcome(path),
        "info_json": dumps(info),
        "png_info_json": dumps(png_info),
        "raw": [len(raw), sha(raw)],
        "fallback_json": dumps(fallback),
        "soft_binding_vendors_in": c2pa.soft_binding_vendors_in(raw),
        "soft_binding_entries": [e.algorithm for e in c2pa.soft_binding_registry_entries_in(raw)],
        "synthid_vendors_in": c2pa.synthid_evidence_vendors_in(raw),
        "has_c2pa_metadata": c2pa.has_c2pa_metadata(path),
        "c2pa_chunk": None if chunk is None else sha(chunk),
        "well_formed_png": c2pa._is_well_formed_png(path),
        "helpers": helpers(info) if info else None,
    }
    try:
        ai = extractor.extract_ai_metadata(path)
        report["extract_ai_metadata"] = {
            k: (sha(v) if isinstance(v, bytes) else dumps(v) if isinstance(v, dict) else str(v)) for k, v in ai.items()
        }
        report["ai_summary"] = extractor.get_ai_metadata_summary(path)
        report["has_ai_metadata"] = extractor.has_ai_metadata(path)
    except Exception as error:  # noqa: BLE001
        report["extract_ai_metadata"] = f"error:{type(error).__name__}"
    return report


# Synthetic PNGs


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


SIG = b"\x89PNG\r\n\x1a\n"
IHDR = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
IDAT = png_chunk(b"IDAT", zlib.compress(b"\x00\x00"))
IEND = png_chunk(b"IEND", b"")
JUMBF = (
    b"\x00\x00\x01\x00jumb\x00\x00\x00\x1ejumdc2pa\x00\x11\x00\x10\x80\x00\x00\xaa\x008\x9bq\x00"
    b"c2pa.claim\xa3dnamex\x1aAdobe Firefly 3.0 (test)\x0bspecVersionc2.1"
    b" issuer Google LLC Adobe OpenAI c2pa.created c2pa.edited c2pa.watermarked.unbound"
    b" trainedAlgorithmicMedia com.adobe.trustmark.Q io.iscc.v0 com.suno.fingerprint.v2"
    b" GPT-4o DALL-E Firefly 20240517170000Z 20240518170000Z 20240519170000Z 20240520170000Z"
)
APP11_HEX = "\n app11 " + str(len(JUMBF)) + "\n" + "\n".join(JUMBF.hex()[i : i + 72] for i in range(0, len(JUMBF.hex()), 72)) + "\n"


def png_cases() -> dict[str, bytes]:
    app11 = b"Raw profile type app11\x00" + APP11_HEX.encode()
    ztxt = b"Raw profile type app11\x00\x00" + zlib.compress(APP11_HEX.encode())
    itxt = b"Raw profile type app11\x00\x01\x00\x00\x00" + zlib.compress(APP11_HEX.encode())
    itxt_plain = b"Raw profile type app11\x00\x00\x00en\x00\x00" + APP11_HEX.encode()
    return {
        "cabx": SIG + IHDR + png_chunk(b"caBX", JUMBF) + IDAT + IEND,
        "cabx-not-c2pa": SIG + IHDR + png_chunk(b"caBX", b"plain bytes") + IDAT + IEND,
        "cabx-after-idat": SIG + IHDR + IDAT + png_chunk(b"caBX", JUMBF) + IEND,
        "cabx-truncated": (SIG + IHDR + png_chunk(b"caBX", JUMBF) + IDAT + IEND)[:120],
        "app11-text": SIG + IHDR + png_chunk(b"tEXt", app11) + IDAT + IEND,
        "app11-ztxt": SIG + IHDR + png_chunk(b"zTXt", ztxt) + IDAT + IEND,
        "app11-itxt": SIG + IHDR + png_chunk(b"iTXt", itxt) + IDAT + IEND,
        "app11-itxt-plain": SIG + IHDR + png_chunk(b"iTXt", itxt_plain) + IDAT + IEND,
        "app11-then-cabx": SIG + IHDR + png_chunk(b"tEXt", app11) + png_chunk(b"caBX", JUMBF[:40] + b"Microsoft") + IDAT + IEND,
        "app11-bad-hex": SIG + IHDR + png_chunk(b"tEXt", b"Raw profile type app11\x00\napp11 12\nzz00") + IDAT + IEND,
        "no-iend": SIG + IHDR + IDAT,
        "ai-text": SIG + IHDR + png_chunk(b"tEXt", b"parameters\x00steps: 20") + png_chunk(b"tEXt", b"Title\x00ok")
        + png_chunk(b"iTXt", b"prompt\x00\x00\x00\x00\x00a cat") + png_chunk(b"zTXt", b"workflow\x00\x00" + zlib.compress(b"{}"))
        + png_chunk(b"caBX", JUMBF) + IDAT + png_chunk(b"tEXt", b"negative_prompt\x00late") + IEND,
        "not-png": b"GIF89a" + b"\x00" * 20,
    }


def inject_cases(work: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    chunk = png_chunk(b"caBX", JUMBF)
    for name, data in png_cases().items():
        src = work / f"inject-src-{name}.png"
        src.write_bytes(data)
        dst = work / f"inject-dst-{name}.png"
        try:
            c2pa.inject_c2pa_chunk(src, dst, chunk)
            out[name] = sha(dst.read_bytes())
        except ValueError as error:
            out[name] = f"error:{error}"
    return out


# Synthetic manifest stores


def store(manifests: dict, active: str = "urn:a", **extra: object) -> str:
    return json.dumps({"active_manifest": active, "manifests": manifests, **extra})


def results(success=(), informational=(), failure=()) -> dict:
    def codes(values):
        return [{"code": code} for code in values]

    return {"activeManifest": {"success": codes(success), "informational": codes(informational), "failure": codes(failure)}}


OK_CODES = ["claimSignature.insideValidity", "claimSignature.validated", "assertion.dataHash.match"]


def store_cases() -> dict[str, str]:
    actions = {
        "label": "c2pa.actions.v2",
        "data": {
            "actions": [
                {"action": "c2pa.created", "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                 "softwareAgent": {"name": "GPT-4o via ChatGPT"}},
                {"action": "c2pa.watermarked.bound", "description": "SynthID applied"},
                {"action": "c2pa.color_adjustments", "parameters": {"com.adobe.digitalSourceType": "compositeWithTrainedAlgorithmicMedia"}},
                {"action": 7},
                "bad",
            ]
        },
    }
    soft = {"label": "c2pa.soft-binding", "data": {"alg": "com.adobe.trustmark.Q", "blocks": [{"value": "abc"}, {"value": "x" * 300}]}}
    unknown_soft = {"label": "c2pa.soft-binding.v2", "data": {"alg": "org.unknown.mark", "blocks": []}}
    disclosure = {"label": "c2pa.ai-disclosure", "data": {"modelName": "Gen-4", "contentProfile": {"humanOversightLevel": "human_validated"}}}
    disclosure2 = {"label": "c2pa.ai-disclosure", "data": {"modelType": "diffusion", "contentProfile": {"humanOversightLevel": "prompt_guided"}}}
    ingredient_assertion = {"label": "c2pa.ingredient.v3", "data": {"digitalSourceType": "algorithmicMedia"}}
    google = {
        "claim_generator_info": [{"name": "Google Gemini"}],
        "signature_info": {"issuer": "Google LLC", "common_name": "Google C2PA", "time": "2025-01-01T00:00:00Z"},
        "assertions": [actions, soft],
        "ingredients": [{"active_manifest": "urn:b", "validation_results": results(OK_CODES)},
                        {"active_manifest": "urn:bad", "validation_results": results(failure=["assertion.dataHash.mismatch"])},
                        {"digitalSourceType": "trainedAlgorithmicMedia"}],
    }
    child = {"claim_generator": "Adobe Firefly", "signature_info": {"issuer": "Adobe Inc."}, "assertions": [disclosure]}
    bad_child = {"claim_generator": "Midjourney", "signature_info": {"issuer": "Bria Artificial"}}
    youtube = {
        "claim_generator": "YouTube Video Processing Services",
        "signature_info": {"issuer": "Google LLC", "common_name": "YouTube"},
        "assertions": [{"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.opened"}, {"action": "c2pa.transcoded",
                       "digitalSourceType": "trainedAlgorithmicMedia"}]}}],
    }
    return {
        "full-chain": store({"urn:a": google, "urn:b": child, "urn:bad": bad_child}, validation_results=results(OK_CODES), validation_state="Valid"),
        "disclosure-only": store({"urn:a": {"claim_generator": "Runway", "assertions": [disclosure, disclosure2, ingredient_assertion]}},
                                 validation_results=results(OK_CODES, ["timeStamp.untrusted"])),
        "disclosure-enhanced": store({"urn:a": {"assertions": [disclosure]}}),
        "unknown-soft-binding": store({"urn:a": {"signature_info": {"issuer": "Google LLC"}, "assertions": [unknown_soft, actions]}}),
        "youtube": store({"urn:a": youtube}, validation_results=results(OK_CODES)),
        "invalid-binding": store({"urn:a": {"claim_generator": "x"}}, validation_results=results(
            ["claimSignature.validated"], [], ["assertion.bmffHash.mismatch", "signingCredential.untrusted", "signingCredential.expired"])),
        "invalid-signature": store({"urn:a": {}}, validation_results=results(failure=["claimSignature.mismatch", "signingCredential.ocsp.revoked"])),
        "trusted": store({"urn:a": {"signature_info": {"issuer": "Ideogram, Inc", "time": 12}}}, validation_results=results(
            ["signingCredential.trusted", "claimSignature.validated", "assertion.boxesHash.match"])),
        "flat-status": store({"urn:a": {"claim_generator": "Canva​"}}, validation_status=[{"code": "claimSignature.missing"}, {"code": ""}, "x"]),
        "procedural": store({"urn:a": {"assertions": [{"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.created",
                            "digitalSourceType": "algorithmicMedia"}]}}]}}),
        "cycle": store({"urn:a": {"ingredients": [{"active_manifest": "urn:a"}, {"active_manifest": "urn:c"}]},
                        "urn:c": {"ingredients": [{"active_manifest": "urn:a"}], "claim_generator": "Dreamina"}}),
        "no-active": json.dumps({"manifests": {"urn:a": {}}}),
        "active-missing": store({}, active="urn:zzz"),
        "error": json.dumps({"error": "boom", "manifests": {}}),
        "empty": "{}",
        "not-object": "[1, 2]",
        "invalid-json": "{not json",
        "unicode": store({"urn:a": {"claim_generator": "Grok Imagine — 北京", "signature_info": {"issuer": "xAI Grok Imagine", "time": [1, "a"]}}}),
    }


# Direct units


def units() -> dict[str, object]:
    cbor = [
        (b"xxname\x65hello", b"name"),
        (b"name\x78\x05abcde", b"name"),
        (b"name\x79\x00\x03abc", b"name"),
        (b"name\x79\x00\x09abc", b"name"),
        (b"name\x61\xff", b"name"),
        (b"name\x40", b"name"),
        (b"name", b"name"),
        (b"nope", b"name"),
        (b"name\x78", b"name"),
    ]
    app11 = [
        APP11_HEX,
        "app11 4 6a756d62",
        "  app11\t12 \n" + ("00" * 4 + "6a756d62" + "00" * 4),
        "app11 12 0000",
        "app11 x 00",
        "app1112 00",
        "app11 4 0g",
        "",
        "app11 4\n" + (b"\x00\x00\x00\x10jumbc2pa".hex()),
        "app11 4\n" + (b"0123456789".hex()),
    ]
    buffers = [
        b"com.adobe.trustmark.Q",
        b"xcom.adobe.trustmark.Q",
        b"com.adobe.trustmark.Qx",
        b"com.adobe.trustmark.Q.v2",
        b"(com.adobe.trustmark.Q)",
        b"com.suno.fingerprint",
        b"com.suno.fingerprint.v1",
        b"com.suno.fingerprint.v1.",
        b"com.suno.fingerprint..v1",
        b"com.suno.fingerprintx",
        b"-com.suno.fingerprint",
        b"io.iscc.v0 com.digimarc.validate.1 com.microsoft.invismark.1",
        b"",
    ]
    words = ["Nova Canvas by Canva", "Nova Canvas", "Made with DALL-E 3", "DALL·E", "GPT-4o mini", "Firefly2", "CHATGPT", "Google LLC"]
    platforms = ["Suno/1.0", "STRASSE suno", "ﬁrefly 3", "Adobe_Firefly", "dreamina", "dolar", "capcut c2pa-rs", "", None, "Grok Imagine"]
    vendor_texts = ["Google LLC", "google", "Volcano Engine Inc", "fal.ai", "xAI", "OpenAI, Truepic", "Runway", "nobody", "Bria AI"]
    synthid = [
        (b"Google LLC", None, None),
        (b"Google LLC YouTube", None, None),
        (b"Google LLC YouTube SynthID", None, None),
        (b"OpenAI", None, None),
        (b"OpenAI c2pa.watermarked", None, None),
        (b"OpenAI Google", True, False),
        (b"Google YouTube", False, True),
    ]
    markers = [
        {"issuer": "Google LLC", "actions": "opened"},
        {"issuer": "Ideogram"},
        {"source_type": "algorithmicMedia"},
        {"source_type": "compositeSynthetic"},
        {"source_type": "c2pa.ai-disclosure (generated)"},
        {"claim_generator": "Suno/2"},
        {"claim_generator": "YouTube"},
        {"ai_tool": "Sora"},
        {"ai_generator": "Midjourney"},
        {},
    ]
    keys = ["parameters", "PROMPT", "Model hash", "negative_prompt", "sd:mode", "Title", "Software", "comfy_workflow", "ß", "seed", "Seed"]
    return {
        "cbor_text_after": [[p.hex(), k.hex(), c2pa.cbor_text_after(p, k)] for p, k in cbor],
        "raw_app11": [[t, (lambda r: None if r is None else r.hex())(c2pa.raw_app11_jumbf_bytes(t))] for t in app11],
        "raw_app11_info": [[t, dumps(c2pa.c2pa_info_from_raw_app11_profile(t))] for t in app11],
        "soft_binding_entries_in": [[b.hex(), [e.algorithm for e in c2pa.soft_binding_registry_entries_in(b)]] for b in buffers],
        "soft_binding_vendors_in": [[b.hex(), c2pa.soft_binding_vendors_in(b)] for b in buffers],
        "issuer_words": [[w, c2pa.registry_word_matches(w, constants.C2PA_ISSUERS)] for w in words],
        "tool_words": [[w, c2pa.registry_word_matches(w, constants.C2PA_AI_TOOLS)] for w in words],
        "claim_generator_platform": [[p, c2pa.claim_generator_platform(p)] for p in platforms],
        "vendor_platform": [[t, c2pa.c2pa_vendor_platform(t)] for t in vendor_texts],
        "synthid": [[b.hex(), w, s, c2pa.synthid_evidence_vendors_in(b, has_watermark_action=w, records_synthid_action=s)] for b, w, s in synthid],
        "markers_claim_ai": [[m, c2pa.markers_claim_ai(m)] for m in markers],
        "is_ai_field": [[k, extractor._is_ai_field(k)] for k in keys],
        "parse_c2pa_chunk": dumps(_parsed(JUMBF)),
    }


def _parsed(payload: bytes) -> dict:
    info: dict = {}
    c2pa.parse_c2pa_chunk(payload, info)
    return info


def registries() -> dict[str, object]:
    vendor_fields = (
        "issuer", "org", "platform", "needle", "synthid", "asserts_ai", "synthid_requires_watermark_action",
        "signer_platform", "raw_signer_platforms",
    )

    def plain(value):
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, tuple):
            return [plain(v) for v in value]
        return value

    return {
        "vendors": [{f: plain(getattr(v, f)) for f in vendor_fields} for v in constants.C2PA_AI_VENDORS],
        "issuers": [[k.decode(), v] for k, v in constants.C2PA_ISSUERS.items()],
        "synthid_edit_signers": [s.decode() for s in constants.SYNTHID_EDIT_SIGNERS],
        "identity_ai_orgs": sorted(constants.C2PA_IDENTITY_AI_ORGS),
        "signer_platform_by_org": constants.C2PA_SIGNER_PLATFORM_BY_ORG,
        "signer_platforms": [[k.decode(), v] for k, v in constants.C2PA_SIGNER_PLATFORMS],
        "claim_generator_platforms": [list(p) for p in constants.C2PA_CLAIM_GENERATOR_PLATFORMS],
        "ai_tools": [[k.decode(), v] for k, v in constants.C2PA_AI_TOOLS.items()],
        "actions": [[k.decode(), v] for k, v in constants.C2PA_ACTIONS.items()],
        "ai_generator_tokens": sorted(constants.AI_GENERATOR_TOKENS),
        "label_overrides": [[k.decode(), v] for k, v in constants._C2PA_SOFT_BINDING_LABEL_OVERRIDES.items()],
        "vendor_prefixes": [list(plain(tuple(getattr(e, f) for f in e.__slots__))) for e in constants.C2PA_VENDOR_SOFT_BINDING_PREFIXES],
        "soft_binding_rows": [list(plain(row)) for row in generated.C2PA_SOFT_BINDING_ROWS],
        "soft_binding_labels": [[k.decode(), v] for k, v in constants.C2PA_SOFT_BINDINGS.items()],
        "soft_binding_source": [generated.C2PA_SOFT_BINDING_SOURCE_URL, generated.C2PA_SOFT_BINDING_SOURCE_REVISION],
        "structural_markers": sorted(c2pa.C2PA_STRUCTURAL_MARKERS),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = {}
    for path in sorted(FIXTURE_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
            rel = path.relative_to(FIXTURE_ROOT).as_posix()
            if rel.startswith(("provenance/", "visible/")):
                fixtures[rel] = describe(path)
    synthetic = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, data in png_cases().items():
            path = work / f"{name}.png"
            path.write_bytes(data)
            entry = describe(path)
            entry["input_b64"] = base64.b64encode(data).decode()
            synthetic[name] = entry
        injected = inject_cases(work)
    stores = {}
    for name, text in store_cases().items():
        info = c2pa.c2pa_info_from_manifest_store(text)
        stores[name] = {"store": text, "info_json": dumps(info), "helpers": helpers(info) if info else None}
    write("fixtures", fixtures)
    write("synthetic", synthetic)
    write("inject", injected)
    write("stores", stores)
    write("units", units())
    write("registries", registries())


def write(name: str, value: object) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {name}.json")


if __name__ == "__main__":
    main()

"""Record DWT-DCT decoder and open-watermark verdict goldens for raiw-pixel.

Run from the repository root with the oracle environment:
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/pixel_dwt.py`
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from imwatermark import WatermarkDecoder, WatermarkEncoder
from pixel_common import (
    REPO,
    b64,
    base_image,
    bit_string,
    canonical,
    packed_bits,
    provenance_images,
    sha256,
    write,
)

from remove_ai_watermarks import image_io
from remove_ai_watermarks.dwt_dct import _DecodeMaxDct, decode_dwt_dct_lengths
from remove_ai_watermarks.invisible_watermark import _BITS_48, _SD1_STRING, detect_invisible_watermark

SUITE = "pixel_dwt"
LENGTHS = (48, 8 * len(_SD1_STRING))

CANONICAL_HASHES = """
import hashlib, json, sys
import cv2, numpy as np
sys.path.insert(0, "src")
from remove_ai_watermarks import image_io
out = {}
for item in json.load(sys.stdin):
    img = image_io.imread(item["path"]) if item["kind"] == "file" else np.load(item["path"])
    if img is None:
        out[item["path"]] = None
        continue
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) if img.ndim == 3 and img.size else None
    out[item["path"]] = {
        "bgr": hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest(),
        "yuv": None if yuv is None else hashlib.sha256(yuv.tobytes()).hexdigest(),
        "bgr_from_yuv": None if yuv is None else hashlib.sha256(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR).tobytes()).hexdigest(),
    }
print(json.dumps(out))
"""


def local_hashes(img: np.ndarray) -> dict[str, Any]:
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) if img.ndim == 3 and img.size else None
    return {
        "bgr": sha256(img),
        "yuv": None if yuv is None else sha256(yuv),
        "bgr_from_yuv": None if yuv is None else sha256(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)),
    }


def decode_record(img: np.ndarray, path: Path | None = None) -> dict[str, Any]:
    """Everything the Rust decoder must reproduce for one BGR array."""
    record: dict[str, Any] = {"shape": list(img.shape), "sha256": sha256(img)}
    try:
        decoded = decode_dwt_dct_lengths(img, LENGTHS)
    except Exception as exc:
        record["error"] = type(exc).__name__
        record["message"] = str(exc)
    else:
        rows, cols = img.shape[0] // 4 * 4, img.shape[1] // 4 * 4
        trimmed = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:rows, :cols]
        blocks = _DecodeMaxDct(LENGTHS)._plane_bits(trimmed, 1, 36)
        record["blocks"] = int(blocks.size)
        record["block_ones"] = int(blocks.sum())
        record["block_bits"] = packed_bits(blocks)
        record["bits"] = {str(length): bit_string(decoded[length]) for length in LENGTHS}
        if img.size and min(img.shape[:2]) * max(img.shape[:2]) >= 256 * 256 and blocks.size:
            upstream = {length: WatermarkDecoder("bits", length).decode(img, "dwtDct") for length in LENGTHS}
            for length in LENGTHS:
                assert np.array_equal(np.asarray(upstream[length], dtype=bool), decoded[length]), "upstream parity"
    record["scheme"] = detect_invisible_watermark(path or Path("in-memory.png"), image=img)
    return record


def cross_check(items: list[dict[str, Any]], local: dict[str, Any]) -> None:
    remote = canonical(CANONICAL_HASHES, items)
    for item in items:
        assert remote[item["path"]] == local[item["path"]], f"OpenCV 4.10 disagrees on {item['path']}"


def fixture_cases() -> list[dict[str, Any]]:
    cases, items, local = [], [], {}
    for path in provenance_images():
        img = image_io.imread(path)
        rel = path.relative_to(REPO / "data" / "fixtures").as_posix()
        if img is None:
            cases.append({"name": path.name, "fixture": rel, "unreadable": True})
            continue
        items.append({"kind": "file", "path": str(path)})
        local[str(path)] = local_hashes(img)
        cases.append({"name": path.name, "fixture": rel, **decode_record(img, path)})
    cross_check(items, local)
    return cases


def message_bits(message: dict[str, Any]) -> tuple[str, Any]:
    if message["kind"] == "bits":
        return "bits", [int(b) for b in format(int(message["value"]), "048b")]
    return "bytes", bytes.fromhex(message["hex"])


def embed(base: np.ndarray, message: dict[str, Any]) -> np.ndarray:
    kind, payload = message_bits(message)
    encoder = WatermarkEncoder()
    encoder.set_watermark(kind, payload)
    return encoder.encode(base, "dwtDct")


SDXL = {"kind": "bits", "value": _BITS_48["Stable Diffusion XL"]}
FLUX = {"kind": "bits", "value": _BITS_48["FLUX.2 (Black Forest Labs)"]}
SD1 = {"kind": "bytes", "hex": _SD1_STRING.hex()}
OTHER48 = {"kind": "bits", "value": 0xA5A5_0F0F_3C3C}

EMBEDS: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = [
    ("sdxl-noise-512", {"kind": "noise", "width": 512, "height": 512, "seed": 1}, SDXL),
    ("flux-noise-512", {"kind": "noise", "width": 512, "height": 512, "seed": 2}, FLUX),
    ("sd1-noise-512", {"kind": "noise", "width": 512, "height": 512, "seed": 3}, SD1),
    ("sdxl-noise-256", {"kind": "noise", "width": 256, "height": 256, "seed": 4}, SDXL),
    ("flux-noise-odd-263x257", {"kind": "noise", "width": 263, "height": 257, "seed": 5}, FLUX),
    ("sd1-noise-odd-301x229", {"kind": "noise", "width": 301, "height": 229, "seed": 6}, SD1),
    ("sdxl-noise-wide-1031x67", {"kind": "noise", "width": 1031, "height": 67, "seed": 7}, SDXL),
    ("flux-noise-tall-70x947", {"kind": "noise", "width": 70, "height": 947, "seed": 8}, FLUX),
    ("sdxl-texture-256", {"kind": "texture", "width": 256, "height": 256, "seed": 12}, SDXL),
    ("sd1-texture-odd-301x229", {"kind": "texture", "width": 301, "height": 229, "seed": 13}, SD1),
    ("other48-texture-384x320", {"kind": "texture", "width": 384, "height": 320, "seed": 9}, OTHER48),
    ("sdxl-flat-256", {"kind": "flat", "width": 256, "height": 256, "seed": 0, "value": 128}, SDXL),
    ("clean-noise-512", {"kind": "noise", "width": 512, "height": 512, "seed": 10}, None),
    ("clean-texture-640x480", {"kind": "texture", "width": 640, "height": 480, "seed": 11}, None),
]

JPEG_NEGATIVES = [
    ("sdxl-noise-256-q95", "sdxl-noise-256", 95),
    ("sd1-texture-odd-301x229-q90", "sd1-texture-odd-301x229", 90),
    ("sdxl-texture-256-q75", "sdxl-texture-256", 75),
]

DEGENERATE = [
    ("too-small-255x256", {"kind": "texture", "width": 255, "height": 256, "seed": 20}),
    ("too-small-257x255", {"kind": "texture", "width": 257, "height": 255, "seed": 21}),
    ("minimum-256x256", {"kind": "noise", "width": 256, "height": 256, "seed": 22}),
    ("row-1x65536", {"kind": "texture", "width": 65536, "height": 1, "seed": 23}),
    ("rows-2x32768", {"kind": "texture", "width": 32768, "height": 2, "seed": 24}),
    ("rows-4x16384", {"kind": "texture", "width": 16384, "height": 4, "seed": 25}),
    ("rows-8x8192", {"kind": "noise", "width": 8192, "height": 8, "seed": 26}),
    ("column-65536x1", {"kind": "texture", "width": 1, "height": 65536, "seed": 27}),
    ("columns-8192x9", {"kind": "noise", "width": 9, "height": 8192, "seed": 28}),
    ("odd-width-12x5463", {"kind": "texture", "width": 5463, "height": 12, "seed": 29}),
]


def synthetic_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    embeds, jpegs, degenerate = [], [], []
    encoded: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as scratch:
        items, local = [], {}

        def track(name: str, img: np.ndarray) -> None:
            path = str(Path(scratch) / f"{name}.npy")
            np.save(path, img)
            items.append({"kind": "array", "path": path})
            local[path] = local_hashes(img)

        for name, spec, message in EMBEDS:
            base = base_image(spec)
            img = base if message is None else embed(base, message)
            encoded[name] = img
            track(name, base)
            track(f"{name}-encoded", img)
            embeds.append({"name": name, "base": spec, "message": message, **decode_record(img)})
        for name, source, quality in JPEG_NEGATIVES:
            ok, data = cv2.imencode(".jpg", encoded[source], [cv2.IMWRITE_JPEG_QUALITY, quality])
            assert ok
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            track(name, img)
            jpegs.append({"name": name, "source": source, "quality": quality, "jpeg": b64(data.tobytes()), **decode_record(img)})
        for name, spec in DEGENERATE:
            img = base_image(spec)
            degenerate.append({"name": name, "base": spec, **decode_record(img)})
        cross_check(items, local)
    return embeds, jpegs, degenerate


def pattern_cases() -> dict[str, Any]:
    from remove_ai_watermarks import invisible_watermark as iw

    rng = np.random.default_rng(48)
    match48 = []
    for ref in _BITS_48.values():
        for flips in (0, 3, 4, 5, 24):
            value = ref
            for bit in rng.choice(48, size=flips, replace=False):
                value ^= 1 << int(bit)
            match48.append({"value": value, "ref": ref, "matches": iw._bits_match(value, ref)})
    sd1 = []
    for flips in (0, 5, 10, 11, 12, 70):
        raw = bytearray(_SD1_STRING)
        for bit in rng.choice(8 * len(raw), size=flips, replace=False):
            raw[int(bit) // 8] ^= 0x80 >> (int(bit) % 8)
        sd1.append({"hex": bytes(raw).hex(), "fraction": iw._bytes_match_frac(bytes(raw), _SD1_STRING)})
    return {
        "patterns48": {name: value for name, value in _BITS_48.items()},
        "sd1_hex": _SD1_STRING.hex(),
        "match48": match48,
        "sd1": sd1,
        "thresholds": {"match48": iw._MATCH_48, "sd1_fraction": iw._MATCH_SD1_FRAC},
    }


def main() -> None:
    write(SUITE, "fixtures", {"cases": fixture_cases()})
    embeds, jpegs, degenerate = synthetic_cases()
    write(SUITE, "embeds", {"lengths": list(LENGTHS), "cases": embeds})
    write(SUITE, "jpeg", {"cases": jpegs})
    write(SUITE, "degenerate", {"cases": degenerate})
    write(SUITE, "patterns", pattern_cases())


if __name__ == "__main__":
    main()

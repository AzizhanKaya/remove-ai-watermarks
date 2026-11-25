"""Record Microsoft local InvisMark goldens for raiw-pixel.

Run from the repository root with the oracle environment:
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/pixel_invismark.py`

LAPACK's SVD and OpenBLAS's per-block GEMM are not reproducible across machines, so singular values
are recorded as samples and aggregates; block bits and eligibility are exact except for blocks whose
residue lies within `EDGE` of a decision boundary, which are listed so the comparison can skip them.
"""

from __future__ import annotations

import tempfile
import uuid
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pixel_common import REPO, b64, base_image, packed_bits, provenance_images, sha256, write

from remove_ai_watermarks import microsoft_invismark as mi

SUITE = "pixel_invismark"
SAMPLE = 64
EDGE = 1e-6
BOUNDARIES = np.array([4.0, 8.0, 16.0, 20.0])


def analysis_record(rgb: np.ndarray) -> dict[str, Any]:
    try:
        analysis = mi._analyze(rgb)
    except Exception as exc:
        return {"error": type(exc).__name__}
    ll = analysis.ll
    block_rows, block_columns = analysis.bits.shape
    blocks = (
        ll[: block_rows * 4, : block_columns * 4].reshape(block_rows, 4, block_columns, 4).transpose(0, 2, 1, 3)
    )
    transform = mi._dct_matrix()
    coefficients = transform @ blocks @ transform.T
    matrices = coefficients.reshape(block_rows, block_columns, -1)[..., 1:].reshape(block_rows, block_columns, 3, 5)
    sigma = np.linalg.svd(matrices, compute_uv=False)[..., 0].ravel()
    residue = sigma % 24.0
    near = np.flatnonzero(np.min(np.abs(residue[:, None] - BOUNDARIES[None, :]), axis=1) < EDGE)
    support = np.zeros((144, 2), dtype=np.int64)
    np.add.at(support, (analysis.bit_indices[analysis.eligible], analysis.bits[analysis.eligible]), 1)
    recovered = mi._validated_detection(rgb)
    return {
        "ll_shape": list(ll.shape),
        "ll_sum": float(ll.sum()),
        "blocks": [block_rows, block_columns],
        "eligible": packed_bits(analysis.eligible.ravel()),
        "eligible_bits": packed_bits(analysis.bits[analysis.eligible]),
        "eligible_count": int(analysis.eligible.sum()),
        "near_boundary": near.tolist(),
        "sigma_sample": sigma[:SAMPLE].tolist(),
        "sigma_sum": float(sigma.sum()),
        "support": support.tolist(),
        "detection": None if recovered is None else detection_record(recovered[0]),
    }


def detection_record(detection: mi.LocalInvisMarkDetection) -> dict[str, Any]:
    return {
        "watermark_id": str(detection.watermark_id),
        "payload": detection.payload.hex(),
        "minimum_support": detection.minimum_support,
    }


def fixture_cases() -> list[dict[str, Any]]:
    cases = []
    for path in provenance_images():
        rgb, alpha = mi._read_rgb(path)
        cases.append(
            {
                "name": path.name,
                "fixture": path.relative_to(REPO / "data" / "fixtures").as_posix(),
                "sha256": sha256(rgb),
                "shape": list(rgb.shape),
                "has_alpha": alpha is not None,
                **analysis_record(np.ascontiguousarray(rgb)),
            }
        )
    return cases


def _dct4() -> np.ndarray:
    matrix = np.empty((4, 4), dtype=np.float64)
    for frequency in range(4):
        scale = np.sqrt(1 / 4) if frequency == 0 else np.sqrt(2 / 4)
        for sample in range(4):
            matrix[frequency, sample] = scale * np.cos(np.pi * (2 * sample + 1) * frequency / 8)
    return matrix


def synthetic_writer(image: np.ndarray, watermark_id: uuid.UUID, *, valid_checksum: bool = True) -> np.ndarray:
    """The independent producer of tests/test_microsoft_invismark_local.py, verbatim in behavior."""
    guid = watermark_id.bytes_le
    checksum = sum(guid) & 0xFF
    if not valid_checksum:
        checksum ^= 1
    payload = bytes((0x4C,)) + guid + bytes((checksum,))
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    rgb = image.astype(np.float64)
    luminance = rgb @ np.array((0.299, 0.587, 0.114), dtype=np.float64)
    height, width = luminance.shape[0] // 2 * 2, luminance.shape[1] // 2 * 2
    plane = luminance[:height, :width]
    ll = (plane[0::2, 0::2] + plane[0::2, 1::2] + plane[1::2, 0::2] + plane[1::2, 1::2]) * 0.5
    original_ll = ll.copy()
    dct = _dct4()
    for block_y in range(ll.shape[0] // 4):
        for block_x in range(ll.shape[1] // 4):
            top, left = block_y * 4, block_x * 4
            block = ll[top : top + 4, left : left + 4]
            coefficients = dct @ block @ dct.T
            ac = coefficients.reshape(-1)[1:].reshape(3, 5)
            u, singular, vh = np.linalg.svd(ac, full_matrices=False)
            bit = int(bits[(block_y % 12) * 12 + (block_x % 12)])
            singular[0] = (np.floor(singular[0] / 24) + 0.25 + 0.5 * bit) * 24
            rewritten = coefficients.reshape(-1).copy()
            rewritten[1:] = (u @ np.diag(singular) @ vh).reshape(-1)
            ll[top : top + 4, left : left + 4] = dct.T @ rewritten.reshape(4, 4) @ dct
    delta = (ll - original_ll) * 0.5
    pixel_delta = np.zeros(luminance.shape)
    pixel_delta[:height, :width] = np.repeat(np.repeat(delta, 2, axis=0), 2, axis=1)
    return np.clip(np.rint(rgb + pixel_delta[..., None]), 0, 255).astype(np.uint8)


def delta_payload(after: np.ndarray, before: np.ndarray) -> str:
    delta = after.astype(np.int16) - before.astype(np.int16)
    assert np.abs(delta).max() < 128
    return b64(zlib.compress(delta.astype(np.int8).tobytes(), 9))


ID_A = uuid.UUID("83424621-03cb-40e3-9808-a9fae837156d")
ID_B = uuid.UUID("0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0")
CARRIERS = [
    ("carrier-192", {"kind": "noise-range", "lo": 24, "hi": 232, "width": 192, "height": 192, "seed": 31}, ID_A, True),
    ("carrier-bad-checksum-192", {"kind": "noise-range", "lo": 24, "hi": 232, "width": 192, "height": 192, "seed": 31}, ID_A, False),
    ("carrier-odd-261x205", {"kind": "noise-range", "lo": 24, "hi": 232, "width": 261, "height": 205, "seed": 32}, ID_B, True),
    ("carrier-texture-300x200", {"kind": "texture", "width": 300, "height": 200, "seed": 33}, ID_B, True),
]
CLEAN = [
    ("clean-192", {"kind": "noise-range", "lo": 24, "hi": 232, "width": 192, "height": 192, "seed": 31}),
    ("clean-texture-97x61", {"kind": "texture", "width": 97, "height": 61, "seed": 34}),
    ("tiny-7x5", {"kind": "noise", "width": 7, "height": 5, "seed": 35}),
]


def synthetic_cases() -> list[dict[str, Any]]:
    cases = []
    with tempfile.TemporaryDirectory() as scratch:
        for name, spec, watermark_id, valid in CARRIERS:
            base = base_image(spec)
            carrier = synthetic_writer(base, watermark_id, valid_checksum=valid)
            source = Path(scratch) / f"{name}.png"
            Image.fromarray(carrier, "RGB").save(source)
            record: dict[str, Any] = {
                "name": name,
                "base": spec,
                "watermark_id": str(watermark_id),
                "carrier_delta": delta_payload(carrier, base),
                "carrier_sha256": sha256(carrier),
                **analysis_record(carrier),
            }
            if record["detection"] is not None:
                wrong = mi.disrupt_local_invismark(source, Path(scratch) / "wrong.png", expected_watermark_id=uuid.uuid4())
                assert wrong is None
                output = Path(scratch) / f"{name}-clean.png"
                removal = mi.disrupt_local_invismark(source, output, expected_watermark_id=watermark_id)
                assert removal is not None
                cleaned = np.asarray(Image.open(output).convert("RGB"))
                record["disruption"] = {
                    "changed_blocks": removal.changed_blocks,
                    "delta": delta_payload(cleaned, carrier),
                    "sha256": sha256(cleaned),
                    "after": None if mi.detect_local_invismark(output) is None else "detected",
                }
            cases.append(record)
        for name, spec in CLEAN:
            img = base_image(spec)
            cases.append({"name": name, "base": spec, **analysis_record(img)})
    return cases


C2PA_INFOS = [
    ("com.microsoft.invismark.1", str(ID_A)),
    ("io.iscc.v0", str(ID_A)),
    ("com.microsoft.invismark.1", f"{ID_A}, {ID_B}"),
    ("com.microsoft.invismark.1", f"{ID_A}, {ID_A}"),
    ("com.adobe.trustmark.P, com.microsoft.invismark.1", f"not-a-uuid, {ID_B}"),
    ("com.microsoft.invismark.1 ,", "{" + str(ID_B).upper() + "}"),
    ("com.microsoft.invismark.1", "urn:uuid:" + str(ID_A)),
    ("com.microsoft.invismark.1", ID_A.hex),
    ("com.microsoft.invismark.1", "+" + ID_A.hex[1:]),
    ("com.microsoft.invismark.1", "0x" + ID_A.hex[2:]),
    ("com.microsoft.invismark.1", ID_A.hex[:8] + "_" + ID_A.hex[9:]),
    ("com.microsoft.invismark.1", ID_A.hex[:8] + "__" + ID_A.hex[10:]),
    ("com.microsoft.invismark.1", "-" + ID_A.hex[1:]),
    ("com.microsoft.invismark.1", "zz" + ID_A.hex[2:]),
    ("com.microsoft.invismark.1", ""),
    ("com.microsoft.invismark.1", None),
    (None, str(ID_A)),
    ("com.microsoft.invismark.10", str(ID_A)),
    ("com.microsoft.invismark.1", "00000000-0000-0000-0000-000000000000"),
]


def c2pa_cases() -> list[dict[str, Any]]:
    out = []
    for algorithm, value in C2PA_INFOS:
        result = mi.invismark_id_from_c2pa_info({"soft_binding_algorithm": algorithm, "soft_binding_value": value})
        out.append({"algorithm": algorithm, "value": value, "watermark_id": None if result is None else str(result)})
    return out


def main() -> None:
    write(SUITE, "fixtures", {"edge": EDGE, "sample": SAMPLE, "cases": fixture_cases()})
    write(SUITE, "synthetic", {"edge": EDGE, "sample": SAMPLE, "cases": synthetic_cases()})
    write(SUITE, "c2pa", {"cases": c2pa_cases()})


if __name__ == "__main__":
    main()

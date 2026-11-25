"""Shared helpers for the `classify*` golden generators (raiw-classify).

Synthetic images come from a counter-based splitmix64 hash that the Rust tests
regenerate bit for bit (`hash_image`), so no image binaries are committed.
Float32 tensors are written with 9 significant digits (exact float32 round trip);
float64 values keep Python's shortest repr.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GOLDENS = REPO / "rs-ai-watermarks" / "goldens"
FIXTURES = REPO / "data" / "fixtures"
SUITE = GOLDENS / "classify"
BUNDLES = GOLDENS / "classify_bundles"

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def splitmix(index: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64 of `index + seed * golden`; mirrored by the Rust test helper."""
    with np.errstate(over="ignore"):
        z = index.astype(np.uint64) + np.uint64(seed) * _GOLDEN + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def hash_image(spec: dict[str, Any]) -> np.ndarray:
    """Build the HxWx3 uint8 image described by `spec` (`w`, `h`, `seed`, `lo`, `hi`, optional `paint`)."""
    w, h = int(spec["w"]), int(spec["h"])
    lo, hi = int(spec.get("lo", 0)), int(spec.get("hi", 256))
    idx = np.arange(h * w * 3, dtype=np.uint64)
    vals = splitmix(idx, int(spec["seed"])) % np.uint64(hi - lo) + np.uint64(lo)
    arr = vals.astype(np.uint8).reshape(h, w, 3)
    for paint in spec.get("paint", []):
        x0, x1 = paint.get("x0", 0), paint.get("x1", w)
        y0, y1 = paint.get("y0", 0), paint.get("y1", h)
        arr[y0:y1, x0:x1, paint["c"]] = paint["v"]
    if spec.get("checker16"):
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        arr[:, :, 0] = ((((yy // 16) + (xx // 16)) % 2) * 240).astype(np.uint8)
    return arr


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f32_list(arr: Any) -> list[float]:
    """Float32 values as 9-significant-digit floats (parse in Rust, then cast through f32)."""
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    return [float(f"{float(v):.9g}") for v in flat]


def f64_list(arr: Any) -> list[float]:
    return [float(v) for v in np.asarray(arr, dtype=np.float64).reshape(-1)]


def mask_bits(arr: Any) -> str:
    """A 0/1 float mask as a compact `"0110..."` string."""
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    if not np.all((flat == 0.0) | (flat == 1.0)):
        raise ValueError("mask is not binary")
    return "".join("1" if v else "0" for v in flat)


def tensor_digest(arr: np.ndarray) -> dict[str, Any]:
    """Exact float32 tensor identity (SHA-256 of little-endian bytes) plus a coarse sum for diagnosis."""
    f32 = np.ascontiguousarray(arr, dtype="<f4")
    return {
        "shape": list(f32.shape),
        "sha256": sha256_bytes(f32.tobytes()),
        "sum": float(f32.astype(np.float64).sum()),
    }


def clean_json(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and reject non-finite floats."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if not math.isfinite(f):
            return {"nonfinite": repr(f)}
        return f
    return value


def write_golden(name: str, payload: Any) -> Path:
    SUITE.mkdir(parents=True, exist_ok=True)
    path = SUITE / f"{name}.json"
    path.write_text(json.dumps(clean_json(payload), indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return path

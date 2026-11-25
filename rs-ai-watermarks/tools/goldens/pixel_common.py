"""Shared helpers for the `pixel_*` golden generators.

The generators need PyWavelets, invisible-watermark and TrustMark next to the package sources, so
they run under the oracle environment from the repository root, for example
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/pixel_dwt.py`.
That environment carries OpenCV 5.x; every cv2 result a golden depends on is cross-checked against
the canonical OpenCV 4.10 in `.venv` through a subprocess.

Synthetic bases are integer formulas over a SplitMix64 stream, rebuilt bit-identically by the Rust
tests (`crates/raiw-pixel/tests/common/mod.rs`).
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
WORKSPACE = REPO / "rs-ai-watermarks"
GOLDENS = WORKSPACE / "goldens"
CANONICAL_PYTHON = REPO / ".venv" / "bin" / "python"
sys.path.insert(0, str(REPO / "src"))

GOLDEN = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def splitmix_u64(seed: int, count: int) -> np.ndarray:
    """SplitMix64 outputs for states `seed + (i + 1) * GOLDEN`."""
    with np.errstate(over="ignore"):
        z = np.uint64(seed & MASK) + (np.arange(1, count + 1, dtype=np.uint64) * np.uint64(GOLDEN))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def base_image(spec: dict[str, Any]) -> np.ndarray:
    """A deterministic HxWx3 uint8 image (channel order is whatever the caller treats it as)."""
    height, width, seed = int(spec["height"]), int(spec["width"]), int(spec["seed"])
    noise = (splitmix_u64(seed, height * width * 3) >> np.uint64(56)).astype(np.int64).reshape(height, width, 3)
    kind = spec["kind"]
    if kind == "noise":
        return noise.astype(np.uint8)
    if kind == "noise-range":
        lo, hi = int(spec["lo"]), int(spec["hi"])
        wide = splitmix_u64(seed, height * width * 3).reshape(height, width, 3)
        return (lo + (wide % np.uint64(hi - lo)).astype(np.int64)).astype(np.uint8)
    if kind == "texture":
        y, x, c = np.indices((height, width, 3), dtype=np.int64)
        pattern = (x * 7 + y * 3 + c * 50) % 101 - 50
        wave = ((x // 9 + y // 13 + c) % 7) * 6 - 18
        return np.clip(128 + pattern + wave + noise % 21 - 10, 0, 255).astype(np.uint8)
    if kind == "flat":
        return np.full((height, width, 3), int(spec.get("value", 128)), dtype=np.uint8)
    raise ValueError(f"unknown base kind {kind!r}")


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def packed_bits(bits: np.ndarray) -> str:
    """Bits (any truthy dtype) packed MSB-first, zlib-compressed, base64."""
    return b64(zlib.compress(np.packbits(np.asarray(bits, dtype=bool)).tobytes(), 9))


def bit_string(bits: Any) -> str:
    return "".join("1" if bool(bit) else "0" for bit in bits)


def write(suite: str, name: str, payload: Any) -> None:
    out = GOLDENS / suite
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size} bytes)")


def canonical(script: str, payload: Any) -> Any:
    """Run `script` under the OpenCV 4.10 `.venv`; it reads JSON on stdin and prints JSON."""
    result = subprocess.run(
        [str(CANONICAL_PYTHON), "-I", "-c", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    return json.loads(result.stdout)


def provenance_images() -> list[Path]:
    root = REPO / "data" / "fixtures" / "provenance"
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

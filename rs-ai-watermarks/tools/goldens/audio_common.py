"""Shared helpers for the `audio` golden generators.

The generators need SciPy and Torch next to the package sources. Run them from the repository root
with the oracle environment (the dev `.venv` ships without SciPy), for example
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/audio.py`.

Synthetic inputs are described by small JSON specs and rebuilt bit-identically by the Rust tests
(`tests/common/mod.rs`): float64 formulas evaluated in a fixed order through libm, then cast to
float32, with a SplitMix64 stream for noise.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
WORKSPACE = REPO / "rs-ai-watermarks"
SUITE_DIR = WORKSPACE / "goldens" / "audio"
sys.path.insert(0, str(REPO / "src"))

try:
    import scipy  # noqa: F401
except ImportError:  # pragma: no cover - environment guard
    sys.exit("The audio goldens need SciPy; run them with an interpreter that has numpy and scipy.")

TWO_PI = 2.0 * math.pi
GOLDEN = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1


def splitmix_uniform(seed: int, count: int) -> np.ndarray:
    """Uniform float64 in [0, 1) from SplitMix64 outputs `seed + (i + 1) * GOLDEN`."""
    with np.errstate(over="ignore"):
        state = np.uint64(seed & MASK) + (np.arange(1, count + 1, dtype=np.uint64) * np.uint64(GOLDEN))
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def component(spec: dict[str, Any], frames: int, rate: int) -> np.ndarray:
    t = np.arange(frames) / rate
    kind = spec["kind"]
    amp = float(spec.get("amp", 1.0))
    if kind == "tone":
        return amp * np.sin(TWO_PI * float(spec["freq"]) * t + float(spec.get("phase", 0.0)))
    if kind == "noise":
        return amp * (2.0 * splitmix_uniform(int(spec["seed"]), frames) - 1.0)
    if kind == "chirp":
        f0, f1 = float(spec["f0"]), float(spec["f1"])
        k = (f1 - f0) / (frames / rate)
        return amp * np.sin(TWO_PI * (f0 * t + 0.5 * k * t * t))
    if kind == "burst":
        period, on = int(spec["period"]), int(spec["on"])
        envelope = ((np.arange(frames) % period) < on).astype(np.float64)
        return amp * np.sin(TWO_PI * float(spec["freq"]) * t) * envelope
    if kind == "dc":
        return np.full(frames, float(spec["value"]))
    if kind == "silence":
        return np.zeros(frames)
    raise ValueError(kind)


def synth(spec: dict[str, Any]) -> np.ndarray:
    """`(frames, channels)` float32 samples for a signal spec."""
    frames, rate = int(spec["frames"]), int(spec["rate"])
    columns = []
    for channel in spec["channels"]:
        value = np.zeros(frames)
        for part in channel:
            value = value + component(part, frames, rate)
        columns.append(value)
    return np.column_stack(columns).astype(np.float32)


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def digest(array: Any, *, samples: int = 64) -> dict[str, Any]:
    """Small fingerprint of a float array: exact hash plus tolerance-comparable statistics."""
    value = np.asarray(array)
    flat = value.reshape(-1)
    wide = flat.astype(np.float64)
    picks = sorted({int(i) for i in np.linspace(0, max(len(flat) - 1, 0), num=min(samples, len(flat)))})
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": sha(value),
        "sum": float(np.sum(wide)),
        "sum_sq": float(np.sum(wide * wide)),
        "max_abs": float(np.max(np.abs(wide))) if len(wide) else 0.0,
        "head": [float(v) for v in flat[:8]],
        "tail": [float(v) for v in flat[-8:]],
        "picks": picks,
        "values": [float(flat[i]) for i in picks],
    }


def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        sys.exit("ffmpeg is required")
    return path


def encode_raw(samples: np.ndarray, rate: int, path: Path, args: list[str], layout: str | None = None) -> None:
    """Pipe float32 frames into ffmpeg with fixed arguments (the Rust tests run the same command)."""
    command = [ffmpeg(), "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate), "-ac", str(samples.shape[1])]
    if layout:
        command += ["-channel_layout", layout]
    command += ["-i", "pipe:0", *args, "-fflags", "+bitexact", "-flags:a", "+bitexact", str(path)]
    subprocess.run(command, input=np.ascontiguousarray(samples, dtype="<f4").tobytes(), check=True)  # noqa: S603


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, payload: Any) -> None:
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    path = SUITE_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, allow_nan=True) + "\n")
    print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size} bytes)")


def scrub(text: str, *roots: Path) -> str:
    """Replace temporary roots with stable placeholders before recording paths."""
    for index, root in enumerate(roots):
        text = text.replace(str(root), f"<ROOT{index}>")
    return text

"""Shared helpers for the `gpu_*` golden generators (raiw-gpu).

Inputs are either a seeded synthetic signal that the Rust tests regenerate
bit-for-bit (`synthetic_signal`), short int16 PCM excerpts embedded as base64,
or tracked fixtures decoded in place by ffmpeg (the Rust side reruns the same
ffmpeg command and checks the decoded bytes by SHA-256).
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GOLDENS = REPO / "rs-ai-watermarks" / "goldens"
FIXTURES = REPO / "data" / "fixtures"


def synthetic_signal(samples: int, sample_rate: int, seed: int) -> np.ndarray:
    """Two tones plus LCG noise; mirrored by `raiw_gpu` test helpers."""
    state = seed & 0xFFFFFFFF
    out = np.empty(samples, dtype=np.float32)
    for n in range(samples):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        noise = state / 4294967296.0 * 2.0 - 1.0
        t = n / sample_rate
        value = 0.3 * math.sin(2 * math.pi * 440.0 * t) + 0.2 * math.sin(2 * math.pi * 1234.5 * t + 0.3) + 0.05 * noise
        out[n] = value
    return out


def pcm16_b64(samples: np.ndarray) -> tuple[str, np.ndarray]:
    """Quantize to int16 PCM; return the base64 payload and the exact float32 it decodes to."""
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii"), (pcm.astype(np.float32) / np.float32(32767.0))


def ffmpeg_args(fixture: str, sample_rate: int, start: float, duration: float) -> list[str]:
    """Arguments after the ffmpeg binary; the Rust tests replay them verbatim."""
    return [
        "-v",
        "error",
        "-ss",
        f"{start}",
        "-t",
        f"{duration}",
        "-i",
        str(FIXTURES / fixture),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]


def decode_fixture(fixture: str, sample_rate: int, start: float, duration: float) -> tuple[np.ndarray, str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to decode audio fixtures")
    raw = subprocess.run([ffmpeg, *ffmpeg_args(fixture, sample_rate, start, duration)], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<f4").copy(), hashlib.sha256(raw).hexdigest()


def fixture_input(fixture: str, sample_rate: int, start: float, duration: float) -> tuple[dict[str, Any], np.ndarray]:
    samples, digest = decode_fixture(fixture, sample_rate, start, duration)
    record = {
        "kind": "fixture",
        "fixture": fixture,
        "sample_rate": sample_rate,
        "start": start,
        "duration": duration,
        "decoded_sha256": digest,
        "samples": int(samples.shape[0]),
    }
    return record, samples


def strided(values: np.ndarray, count: int) -> dict[str, Any]:
    """A short evenly strided excerpt plus summary statistics of a long float array."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    step = max(1, flat.shape[0] // count)
    return {
        "len": int(flat.shape[0]),
        "step": step,
        "excerpt": [float(v) for v in flat[::step][:count]],
        "mean": float(flat.mean()),
        "abs_mean": float(np.abs(flat).mean()),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def write(suite: str, name: str, document: dict[str, Any]) -> None:
    out = GOLDENS / suite
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size} bytes)")

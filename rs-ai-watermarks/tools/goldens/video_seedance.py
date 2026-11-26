"""Goldens for the 2D VAE video recipes: SynthID regeneration (8-bit, native guide, 10-bit) and
the Seedance texture-v1 recipe, on short low-resolution clips cut from a tracked fixture.

Run from the repository root with the oracle environment (CPU torch, diffusers):
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/video_seedance.py`

Clips are rebuilt by the Rust tests from the recorded ffmpeg arguments and sha256-checked. Each
case records Python's metrics, wall time, and its delivered file (base64) so the Rust side can
compare decoded frames by PSNR and maximum difference.
"""

from __future__ import annotations

import base64
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_common import BASE, FFMPEG, FIXTURES, sha256, write_golden  # noqa: E402

import torch  # noqa: E402

from remove_ai_watermarks.video_invisible import load_video_vae_runtime, regenerate_video_candidate  # noqa: E402
from remove_ai_watermarks.video_seedance import regenerate_seedance_experimental  # noqa: E402

SUITE = "video_seedance"
FIXTURE = "provenance/higgsfield-veo-3-1-lite.mp4"
CLIPS: dict[str, list[str]] = {
    "veo_320": [
        "-ss", "1", "-t", "1", "-i", "{fixture}", "-vf", "scale=320:180",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-fflags", "+bitexact",
    ],
    "veo_10bit": [
        "-ss", "1", "-t", "0.5", "-i", "{fixture}", "-vf", "scale=160:90", "-an",
        "-c:v", "libx265", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p10le",
        "-x265-params", "log-level=error:pools=4", "-fflags", "+bitexact",
    ],
}
CASES: dict[str, tuple[str, dict[str, Any]]] = {
    "synthid_256": ("veo_320", {"long_side": 256}),
    "synthid_native_guide": ("veo_320", {"long_side": 128, "native_guide": True}),
    "synthid_unstabilized": ("veo_320", {"long_side": 128, "temporal_stabilize": False, "batch_size": 3}),
    "synthid_10bit": ("veo_10bit", {"long_side": 128}),
}


def build(directory: Path) -> dict[str, Path]:
    built = {}
    for name, args in CLIPS.items():
        out = directory / f"{name}.mp4"
        resolved = [str(FIXTURES / FIXTURE) if a == "{fixture}" else a for a in args]
        subprocess.run([FFMPEG, *BASE, *resolved, str(out)], check=True)
        built[name] = out
    return built


def main() -> None:
    torch.set_num_threads(4)
    runtime = load_video_vae_runtime(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        clips = build(directory)
        cases: dict[str, Any] = {}
        for name, (clip, kwargs) in CASES.items():
            out = directory / f"{name}.mp4"
            start = time.perf_counter()
            metrics = regenerate_video_candidate(clips[clip], out, device="cpu", runtime=runtime, **kwargs)
            elapsed = time.perf_counter() - start
            cases[name] = {"clip": clip, "options": kwargs, "metrics": asdict(metrics),
                           "python_seconds": elapsed, "output_b64": base64.b64encode(out.read_bytes()).decode()}
            print(name, asdict(metrics), f"{elapsed:.1f}s")
        for quality in ("delivery", "master"):
            name = f"seedance_{quality}"
            out = directory / f"{name}.mp4"
            start = time.perf_counter()
            metrics = regenerate_seedance_experimental(clips["veo_320"], out, device="cpu", quality=quality,
                                                       runtime=runtime)
            elapsed = time.perf_counter() - start
            cases[name] = {"clip": "veo_320", "options": {"quality": quality}, "metrics": asdict(metrics),
                           "python_seconds": elapsed}
            if quality == "delivery":
                cases[name]["output_b64"] = base64.b64encode(out.read_bytes()).decode()
            print(name, asdict(metrics), f"{elapsed:.1f}s")
        write_golden(SUITE, "recipes", {
            "fixture": FIXTURE,
            "base_args": BASE,
            "clips": CLIPS,
            "sha256": {name: sha256(path) for name, path in clips.items()},
            "cases": cases,
        })


if __name__ == "__main__":
    main()

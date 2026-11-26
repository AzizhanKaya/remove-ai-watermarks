"""Shared helpers for the `video` golden generators.

Run from the repository root with the development environment, for example
`OMP_NUM_THREADS=4 .venv/bin/python rs-ai-watermarks/tools/goldens/video.py`.

Synthetic clips are described by ffmpeg argument lists in `CLIPS`. The Rust tests rebuild them with
the same system ffmpeg and check the recorded sha256 before comparing, so both sides read identical
bytes. Generated media lives in a temporary directory, never in the repository.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
WORKSPACE = REPO / "rs-ai-watermarks"
FIXTURES = REPO / "data" / "fixtures"
# The sibling generator `c2pa.py` would shadow the `c2pa` package; drop the script directory.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
sys.path.insert(0, str(REPO / "src"))

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
BASE = ["-y", "-v", "error", "-threads", "4"]

# One synthetic clip per timing/stream edge case. `{clip:<name>}` refers to an earlier clip.
CLIPS: list[dict[str, Any]] = [
    {
        "name": "cfr_audio",
        "suffix": ".mp4",
        "args": [
            "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-g", "12",
            "-c:a", "aac", "-b:a", "64k", "-shortest",
        ],
    },
    {
        "name": "vfr",
        "suffix": ".mkv",
        "args": [
            "-f", "lavfi", "-i", "testsrc2=size=128x80:rate=30:duration=1.5",
            "-vf", "setpts='if(lt(N,20),N*1001/(30000*TB),(20*1001/30000+(N-20)/15)/TB)'",
            "-fps_mode", "vfr", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-fflags", "+bitexact",
        ],
    },
    {
        "name": "odd_size",
        "suffix": ".mp4",
        "args": [
            "-f", "lavfi", "-i", "testsrc2=size=161x97:rate=25:duration=1",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv444p",
        ],
    },
    {
        "name": "rotated",
        "suffix": ".mp4",
        "args": ["-display_rotation:v:0", "90", "-i", "{clip:cfr_audio}", "-c", "copy"],
    },
    {
        "name": "two_audio",
        "suffix": ".mp4",
        "args": [
            "-i", "{clip:cfr_audio}",
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=44100:duration=2",
            "-map", "0:v", "-map", "0:a", "-map", "1:a",
            "-c:v", "copy", "-c:a:0", "copy", "-c:a:1", "aac", "-b:a:1", "48k",
        ],
    },
    {
        "name": "offset_start",
        "suffix": ".mp4",
        "args": ["-i", "{clip:cfr_audio}", "-c", "copy", "-output_ts_offset", "1.5"],
    },
    {
        "name": "tagged_bt709",
        "suffix": ".mp4",
        "args": [
            "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=24:duration=1",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
        ],
    },
    {
        "name": "hevc_10bit",
        "suffix": ".mp4",
        "args": [
            "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=24:duration=1",
            "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
            "-x265-params", "log-level=error:pools=4",
        ],
    },
    {
        "name": "webm_vp9",
        "suffix": ".webm",
        "args": [
            "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=24:duration=1",
            "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-b:v", "200k",
            "-fflags", "+bitexact",
        ],
    },
    {
        "name": "delivered_same",
        "suffix": ".mp4",
        "args": ["-i", "{clip:cfr_audio}", "-c:v", "libx264", "-crf", "30", "-preset", "veryfast", "-c:a", "copy"],
    },
    {
        "name": "delivered_small",
        "suffix": ".mp4",
        "args": [
            "-i", "{clip:cfr_audio}", "-vf", "scale=80:48", "-r", "12", "-frames:v", "12",
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-an",
        ],
    },
]

# Clips whose bytes are a prefix of another clip (unreadable/truncated input).
TRUNCATED = {"truncated": ("cfr_audio", 0.5)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_clips(directory: Path) -> dict[str, Path]:
    """Build every synthetic clip into `directory` and return name -> path."""
    built: dict[str, Path] = {}
    for clip in CLIPS:
        out = directory / f"{clip['name']}{clip['suffix']}"
        args = [str(built[a[6:-1]]) if a.startswith("{clip:") else a for a in clip["args"]]
        subprocess.run([FFMPEG, *BASE, *args, str(out)], check=True)
        built[clip["name"]] = out
    for name, (base, fraction) in TRUNCATED.items():
        data = built[base].read_bytes()
        out = directory / f"{name}{built[base].suffix}"
        out.write_bytes(data[: int(len(data) * fraction)])
        built[name] = out
    return built


def clip_manifest(built: dict[str, Path]) -> dict[str, Any]:
    return {
        "base_args": BASE,
        "clips": CLIPS,
        "truncated": {name: list(spec) for name, spec in TRUNCATED.items()},
        "sha256": {name: sha256(path) for name, path in built.items()},
    }


def write_golden(suite: str, name: str, payload: Any) -> None:
    out = WORKSPACE / "goldens" / suite / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")

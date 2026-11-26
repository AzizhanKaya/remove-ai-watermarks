"""Goldens for visible-mark removal: the cleaned frames before encoding, and result counts.

`OMP_NUM_THREADS=4 .venv/bin/python rs-ai-watermarks/tools/goldens/video_removal.py` from the repository root.
Re-encoded outputs differ between Python's and Rust's encoder runs, so the golden records the
pre-encode frames from `iter_clean_video_frames` (per-frame digest and channel sums) plus the
`remove_video_visible` / `remove_video_region` result fields.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from video_common import FIXTURES, build_clips, write_golden

from remove_ai_watermarks import video
from remove_ai_watermarks import video_visible as vv
from remove_ai_watermarks.metadata import get_ai_metadata

cv2.setNumThreads(4)

GALLERY = ("sora", "veo", "kling", "dreamina", "hailuo")


def frames_summary(frames: list[Any]) -> list[dict[str, Any]]:
    return [
        {"sha256": hashlib.sha256(f.tobytes()).hexdigest(), "sum": [int(f[:, :, c].sum()) for c in range(3)], "removed": r}
        for f, r in frames
    ]


def visible_case(key: str) -> dict[str, Any]:
    path = FIXTURES / "visible" / key / "example.mp4"
    markers = get_ai_metadata(path)
    active = video._non_contradicted_marks(video.VIDEO_VISIBLE_MARKS, markers)
    scans, base = video._scan_or_probe_video_marks(path, active)
    selected = video._select_stable_visible_marks(scans, markers, video.VIDEO_VISIBLE_MARKS)
    marks, primary, scan, regions, detected, pad, style, slot_pads, slot_styles = video._merge_selected_visible_marks(
        selected, len(base.detections)
    )
    frames = list(
        vv.iter_clean_video_frames(
            path, scan, regions, backend="cv2", padding_fraction=pad, mask_style=style,
            slot_padding_fractions=slot_pads, slot_mask_styles=slot_styles,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        result = video.remove_video_visible(path, Path(tmp) / "out.mp4")
        fields = {k: v for k, v in asdict(result).items() if k not in {"source", "output"}}
    return {"primary": primary, "marks": list(marks), "detected": detected, "frames": frames_summary(frames), "result": fields}


def erase_case(built: dict[str, Path], name: str, **kwargs: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        result = video.remove_video_region(built[name], Path(tmp) / f"out{built[name].suffix}", **kwargs)
        fields = {k: v for k, v in asdict(result).items() if k not in {"source", "output"}}
    return {"clip": name, "kwargs": {k: v for k, v in kwargs.items()}, "result": fields}


def main() -> None:
    out: dict[str, Any] = {"visible": {key: visible_case(key) for key in GALLERY}}
    with tempfile.TemporaryDirectory() as tmp:
        built = build_clips(Path(tmp))
        out["erase"] = [
            erase_case(built, "cfr_audio", regions=[(10, 10, 40, 20)]),
            erase_case(built, "cfr_audio", regions=[(10, 10, 40, 20), (100, 60, 30, 20)], frame_range=(5, 30)),
            erase_case(built, "cfr_audio", regions=[(20, 20, 24, 16)], track=True),
            erase_case(built, "vfr", regions=[(8, 8, 30, 20)]),
        ]
    write_golden("video_removal", "removal", out)


if __name__ == "__main__":
    main()

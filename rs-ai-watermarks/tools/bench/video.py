"""Timing reference for the raiw-video commands (compare with
`cargo test --release -p raiw-video --test bench -- --ignored --nocapture`).

Run from the repository root: `OMP_NUM_THREADS=4 .venv/bin/python rs-ai-watermarks/tools/bench/video.py`.
Each case times one end-to-end library call per clip, best of `REPEAT`, with OpenCV capped at 4 threads.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "goldens"))

from video_common import FIXTURES  # noqa: E402

from remove_ai_watermarks import video  # noqa: E402

cv2.setNumThreads(4)
REPEAT = 2
CLIPS = {
    "metadata": FIXTURES / "provenance" / "xai-grok-imagine-video.mp4",
    "identify": FIXTURES / "visible" / "veo" / "example.mp4",
    "visible": FIXTURES / "visible" / "sora" / "example.mp4",
    "erase": FIXTURES / "visible" / "kling" / "example.mp4",
}


def best(func) -> float:
    times = []
    for _ in range(REPEAT):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        results["video_metadata_remove"] = best(lambda: video.remove_video_metadata(CLIPS["metadata"], out / "m.mp4"))
        results["video_identify_visible"] = best(lambda: video.identify_video(CLIPS["identify"]))
        results["video_visible_auto"] = best(lambda: video.remove_video_visible(CLIPS["visible"], out / "v.mp4"))
        results["video_erase_box"] = best(
            lambda: video.remove_video_region(CLIPS["erase"], out / "e.mp4", regions=[(700, 450, 220, 70)])
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""Time the Python metadata reference over a fixed fixture sample (one cold pass per file).

Run from the repository root: `.venv/bin/python rs-ai-watermarks/tools/bench/metadata.py`.
Compare with `cargo test -p raiw-metadata --release --test bench -- --ignored --nocapture`.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

from remove_ai_watermarks import metadata as md  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SAMPLE = [
    "provenance/chatgpt-1.png",
    "provenance/doubao-1.png",
    "provenance/mj-1.png",
    "provenance/higgsfield-kling-o1.png",
    "provenance/qwen-image.png",
    "provenance/flux-1.jpg",
    "provenance/grok-1.jpg",
    "provenance/google-photos-ios-edit.jpg",
    "provenance/recraft-v3.webp",
    "provenance/google-gemini-omni-app.mp4",
    "provenance/higgsfield-minimax-h3.mp4",
    "visible/veo/provider-original.mp4",
]


def timed(label: str, fn) -> None:
    start = time.perf_counter()
    for name in SAMPLE:
        fn(ROOT / "data" / "fixtures" / name)
    print(f"{label}: {(time.perf_counter() - start) * 1000 / len(SAMPLE):.1f} ms/file")


def main() -> None:
    timed("has_ai_metadata", md.has_ai_metadata)
    for cache in (md._scan_head_cached, md._aigc_label_cached, md._xai_signature_cached):
        cache.cache_clear()
    timed("get_ai_metadata", md.get_ai_metadata)
    with tempfile.TemporaryDirectory() as tmp:
        timed("remove_ai_metadata", lambda p: md.remove_ai_metadata(p, Path(tmp) / p.name))


if __name__ == "__main__":
    main()

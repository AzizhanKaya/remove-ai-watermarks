"""Time the Python container walkers on the tracked fixtures (compare with the Rust `bench_container` test).

Run from the repository root: `.venv/bin/python rs-ai-watermarks/tools/bench/container.py`.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from remove_ai_watermarks import metadata  # noqa: E402
from remove_ai_watermarks._internal import isobmff  # noqa: E402

FIXTURES = REPO / "data" / "fixtures"
ROUNDS = 5


def timed(label: str, fn) -> None:
    fn()
    start = time.perf_counter()
    for _ in range(ROUNDS):
        fn()
    elapsed = (time.perf_counter() - start) / ROUNDS * 1000
    print(f"{label:<28} {elapsed:9.2f} ms/round")


def main() -> None:
    media = sorted(p for p in FIXTURES.rglob("*") if p.suffix.lower() in {".mp4", ".mov"})
    pngs = sorted(p for p in FIXTURES.rglob("*.png"))
    jpegs = sorted(p for p in FIXTURES.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"})
    media_bytes = [p.read_bytes() for p in media]
    jpeg_bytes = [p.read_bytes() for p in jpegs]
    print(f"{len(media)} ISOBMFF, {len(pngs)} PNG, {len(jpegs)} JPEG fixtures; {ROUNDS} rounds")
    timed("scan_c2pa_region", lambda: [isobmff.scan_c2pa_region(p) for p in media])
    timed("scan_isobmff_video_sei", lambda: [isobmff.scan_isobmff_video_sei(p) for p in media])
    timed("strip_c2pa_boxes", lambda: [isobmff.strip_c2pa_boxes(d) for d in media_bytes])
    timed("blank_ai_xmp_packets", lambda: [isobmff.blank_ai_xmp_packets(d) for d in media_bytes])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        timed("strip_isobmff_media_file", lambda: [isobmff.strip_isobmff_media_file(p, out) for p in media])
        timed("strip_isobmff_image_file", lambda: [isobmff.strip_isobmff_image_file(p, out) for p in media])
    timed("png_late_metadata", lambda: [metadata._png_late_metadata(p, 0) for p in pngs])
    timed("jpeg_app_payloads", lambda: [metadata._jpeg_app_payloads(d) for d in jpeg_bytes])


if __name__ == "__main__":
    main()

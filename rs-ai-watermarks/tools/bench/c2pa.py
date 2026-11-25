"""Time the Python C2PA path on the tracked fixtures (compare with the Rust `bench_c2pa` test).

Run from the repository root: `.venv/bin/python rs-ai-watermarks/tools/bench/c2pa.py`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# This file is named c2pa.py: keep its directory off the path so `import c2pa` finds c2pa-python.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]
sys.path.insert(0, str(REPO / "src"))

from remove_ai_watermarks._internal import c2pa, isobmff  # noqa: E402

FIXTURES = REPO / "data" / "fixtures"
ROUNDS = 5


def timed(label: str, fn) -> None:
    fn()
    start = time.perf_counter()
    for _ in range(ROUNDS):
        fn()
    elapsed = (time.perf_counter() - start) / ROUNDS * 1000
    print(f"{label:<30} {elapsed:9.2f} ms/round")


def main() -> None:
    files = sorted(
        p for p in FIXTURES.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov"}
    )
    stores = [s for s in (c2pa._manifest_json_uncached(str(p)) for p in files) if s]
    raws = [isobmff.scan_c2pa_region(p) for p in files if p.suffix == ".mp4"]
    raws = [r for r in raws if r]
    print(f"{len(files)} fixtures, {len(stores)} manifest stores; {ROUNDS} rounds")
    timed("extract_c2pa_info (uncached)", lambda: [c2pa._extract_c2pa_info_impl(p) for p in files])
    timed("read_manifest_store_json", lambda: [c2pa._manifest_json_uncached(str(p)) for p in files])
    timed("c2pa_info_from_manifest_store", lambda: [c2pa.c2pa_info_from_manifest_store(s) for s in stores])
    timed("parse_c2pa_chunk (fallback)", lambda: [c2pa.parse_c2pa_chunk(r, {}) for r in raws])
    timed("soft_binding_vendors_in", lambda: [c2pa.soft_binding_vendors_in(r) for r in raws])
    timed("_extract_c2pa_info_png", lambda: [c2pa._extract_c2pa_info_png(p) for p in files])


if __name__ == "__main__":
    main()

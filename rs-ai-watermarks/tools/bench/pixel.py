"""Python timings for the raiw-pixel hot paths.

Run from the repository root with the oracle venv (PyWavelets is not in `.venv`):
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/bench/pixel.py`.
Compare with `cargo test -p raiw-pixel --release --test bench -- --ignored --nocapture`, which times
the same synthetic images (SplitMix64 noise, `tools/goldens/pixel_common.base_image`).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "goldens"))

from pixel_common import base_image  # noqa: E402

from remove_ai_watermarks import microsoft_invismark  # noqa: E402
from remove_ai_watermarks.dwt_dct import decode_dwt_dct_lengths  # noqa: E402

SIZES = (1024, 2048, 4096)
REPEATS = 5


def timed(run: object) -> float:
    run()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    results = {}
    for side in SIZES:
        bgr = base_image({"kind": "noise", "width": side, "height": side, "seed": side})
        results[f"dwt_dct_{side}"] = timed(lambda bgr=bgr: decode_dwt_dct_lengths(bgr, (48, 136)))
        rgb = bgr[..., ::-1]
        results[f"invismark_{side}"] = timed(lambda rgb=rgb: microsoft_invismark._validated_detection(rgb))
    print(json.dumps({name: round(seconds * 1000, 2) for name, seconds in results.items()}, indent=2))


if __name__ == "__main__":
    main()

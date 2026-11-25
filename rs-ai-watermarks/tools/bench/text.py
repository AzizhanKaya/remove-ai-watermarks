"""Time the Python text hot paths that raiw-text ports (suite: text).

Run from the repository root:
    .venv/bin/python rs-ai-watermarks/tools/bench/text.py

The Rust counterpart is `cargo test -p raiw-text --release --test bench_text -- --ignored --nocapture`;
both build the same inputs from data/evaluations/text-watermark/fidelity-cases.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from remove_ai_watermarks import text as T  # noqa: E402


def inputs() -> tuple[str, list[tuple[str, str]]]:
    cases = json.loads((ROOT / "data/evaluations/text-watermark/fidelity-cases.json").read_text(encoding="utf-8"))["cases"]
    pairs = [(case["source"], case["candidate"]) for case in cases]
    prose = "\n\n".join(source for source, _ in pairs)
    document = (prose + "\n\n```python\nvalue = 1\n```\n\nSee https://raiw.cc/docs and `code` here.\n\n") * 4
    marked = T.encode_unicode_tags(T.encode_vsrmark(document, "bench-payload"), "hidden") + "​‌"
    return marked, pairs


def bench(label: str, function, repeat: int) -> None:
    function()
    start = time.perf_counter()
    for _ in range(repeat):
        function()
    elapsed = (time.perf_counter() - start) / repeat
    print(f"{label:<34} {elapsed * 1000:10.3f} ms")


def main() -> None:
    document, pairs = inputs()
    print(f"document: {len(document)} chars, {len(pairs)} fact pairs")
    bench("identify_text", lambda: T.identify_text(document), 5)
    bench("remove_text_metadata", lambda: T.remove_text_metadata(document), 5)
    bench("split_protected_text_spans", lambda: T.split_protected_text_spans(document), 5)
    bench("verify_text_rewrite_facts (all)", lambda: [T.verify_text_rewrite_facts(s, c) for s, c in pairs], 3)
    bench("_mask_local_rewrite_facts (all)", lambda: [T._mask_local_rewrite_facts(s) for s, _ in pairs], 3)
    T._classify_text_language("warm up the identifier")
    bench("_classify_text_language (all)", lambda: [T._classify_text_language(s) for s, _ in pairs], 3)
    tokens = [((i * 1103515245 + 12345) >> 16) % 4990 + 10 for i in range(4000)]
    bench("score_synthid_threefry_text", lambda: T.score_synthid_threefry_text(tokens, 123456789), 3)
    bench("score_textseal2_text", lambda: T.score_textseal2_text(tokens, 424242), 3)


if __name__ == "__main__":
    main()

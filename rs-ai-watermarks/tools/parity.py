"""End-to-end parity and speed loop: run each CLI case on the Python and Rust binaries and diff them.

Usage (from the repository root):

    .venv/bin/python rs-ai-watermarks/tools/parity.py [--filter identify] [--repeat 3] [--release]

Each case runs both CLIs on the same input, compares exit codes, stdout (parsed as JSON
when `--json` is passed, ignoring keys that legitimately differ such as absolute output
paths), and the sha256 of any file written, then prints wall-clock timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO / "rs-ai-watermarks"
FIXTURES = REPO / "data" / "fixtures"
PYTHON = [str(REPO / ".venv" / "bin" / "python"), "-m", "remove_ai_watermarks.cli"]

IMAGES = sorted(p for p in (FIXTURES / "provenance").iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
VIDEOS = sorted(p for p in (FIXTURES / "provenance").iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv"})
VISIBLE = sorted((FIXTURES / "visible").glob("*/example.png"))


@dataclass(frozen=True)
class Case:
    """One CLI invocation template; `{src}` and `{out}` are substituted per input."""

    name: str
    args: tuple[str, ...]
    inputs: tuple[Path, ...]
    writes: bool = False
    ignore_keys: frozenset[str] = field(default_factory=frozenset)


CASES = (
    Case("identify-metadata", ("identify", "--json", "--no-visible", "{src}"), tuple(IMAGES)),
    Case("identify", ("identify", "--json", "{src}"), tuple(IMAGES + VISIBLE)),
    Case("metadata-check", ("metadata", "--check", "{src}"), tuple(IMAGES)),
    Case("metadata-remove", ("metadata", "--remove", "-o", "{out}", "{src}"), tuple(IMAGES), writes=True),
    Case("visible", ("visible", "-o", "{out}", "{src}"), tuple(VISIBLE), writes=True),
    Case("video-metadata", ("video", "metadata", "--check", "--json", "{src}"), tuple(VIDEOS)),
    Case("video-identify-metadata", ("video", "identify", "--json", "--no-visible", "{src}"), tuple(VIDEOS)),
)


@dataclass
class Run:
    code: int
    stdout: str
    stderr: str
    seconds: float
    output_sha: str | None


def run(command: list[str], output: Path | None, repeat: int) -> Run:
    best = float("inf")
    completed = None
    for _ in range(repeat):
        if output is not None and output.exists():
            output.unlink()
        start = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env={**os.environ, "NO_COLOR": "1"})
        best = min(best, time.perf_counter() - start)
    assert completed is not None
    digest = hashlib.sha256(output.read_bytes()).hexdigest() if output is not None and output.exists() else None
    return Run(completed.returncode, completed.stdout, completed.stderr, best, digest)


def normalized(text: str, ignore: frozenset[str], scratch: Path) -> object:
    """JSON with scratch-directory paths and ignored keys masked; plain text otherwise."""
    text = text.replace(str(scratch), "<tmp>")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()

    def mask(node: object) -> object:
        if isinstance(node, dict):
            return {k: ("<ignored>" if k in ignore else mask(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [mask(v) for v in node]
        return node

    return mask(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filter", default="", help="Only run cases whose name contains this text.")
    parser.add_argument("--repeat", type=int, default=1, help="Runs per side; the fastest is reported.")
    parser.add_argument("--release", action="store_true", help="Use target/release instead of target/debug.")
    args = parser.parse_args()

    profile = "release" if args.release else "debug"
    rust = [str(WORKSPACE / "target" / profile / "rs-ai-watermarks")]
    if not Path(rust[0]).exists():
        print(f"missing {rust[0]}; build it with `cargo build --{profile}`" if args.release else f"missing {rust[0]}")
        return 1

    totals = {"cases": 0, "parity": 0, "py": 0.0, "rs": 0.0}
    for case in CASES:
        if args.filter not in case.name:
            continue
        for source in case.inputs:
            with tempfile.TemporaryDirectory() as scratch_dir:
                scratch = Path(scratch_dir)
                results = []
                for side, base in (("py", PYTHON), ("rs", rust)):
                    out = scratch / f"{side}_{source.stem}_clean{source.suffix}" if case.writes else None
                    command = [*base, *(a.replace("{src}", str(source)).replace("{out}", str(out)) for a in case.args)]
                    results.append(run(command, out, args.repeat))
                py, rs = results
                same = (
                    py.code == rs.code
                    and normalized(py.stdout, case.ignore_keys, scratch) == normalized(rs.stdout, case.ignore_keys, scratch)
                    and py.output_sha == rs.output_sha
                )
                totals["cases"] += 1
                totals["parity"] += same
                totals["py"] += py.seconds
                totals["rs"] += rs.seconds
                mark = "ok  " if same else "DIFF"
                speed = py.seconds / rs.seconds if rs.seconds else float("inf")
                print(
                    f"{mark} {case.name:<24} {source.name:<44} exit {py.code}/{rs.code}  "
                    f"py {py.seconds * 1000:8.1f} ms  rs {rs.seconds * 1000:8.1f} ms  x{speed:6.1f}"
                )
                if not same and os.environ.get("PARITY_VERBOSE"):
                    print("  py:", py.stdout[:400].replace("\n", " "), py.stderr[-300:])
                    print("  rs:", rs.stdout[:400].replace("\n", " "), rs.stderr[-300:])
    if totals["cases"]:
        print(
            f"\n{totals['parity']}/{totals['cases']} identical; "
            f"python {totals['py']:.2f} s vs rust {totals['rs']:.2f} s (x{totals['py'] / max(totals['rs'], 1e-9):.1f})"
        )
    return 0 if totals["parity"] == totals["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())

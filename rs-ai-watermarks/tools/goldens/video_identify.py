"""Goldens for `video.py`: provenance rules, identification, scans, and stream-copy metadata removal.

`OMP_NUM_THREADS=4 .venv/bin/python rs-ai-watermarks/tools/goldens/video_identify.py` from the repository
root. The per-frame scan is summarized per (file, mark) to keep the golden small: the frames with a
candidate box, the exact boxes' digest, and the confidence sum and maximum.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from video_common import FIXTURES, sha256, write_golden

from remove_ai_watermarks import video
from remove_ai_watermarks import video_visible as vv
from remove_ai_watermarks.metadata import get_ai_metadata

cv2.setNumThreads(4)


def fixtures() -> list[Path]:
    paths = sorted((FIXTURES / "provenance").glob("*.mp4"))
    paths += sorted((FIXTURES / "visible").glob("*/example.mp4"))
    paths += sorted((FIXTURES / "visible").glob("*/provider-original.mp4"))
    return paths


def key(path: Path) -> str:
    return str(path.relative_to(FIXTURES))


def scan_summary(path: Path) -> dict[str, Any]:
    scans = vv.scan_video_marks(path, video.VIDEO_VISIBLE_MARKS, collect_timestamps=False)
    out: dict[str, Any] = {}
    for mark, scan in scans.items():
        boxed = [(d.frame_index, list(d.region)) for d in scan.detections if d.region is not None]
        out[mark] = {
            "frames": len(scan.detections),
            "boxed": len(boxed),
            "regions_sha256": hashlib.sha256(json.dumps(boxed).encode()).hexdigest(),
            "conf_sum": sum(d.confidence for d in scan.detections),
            "conf_max": max((d.confidence for d in scan.detections), default=0.0),
            "geometry": [scan.width, scan.height, scan.fps],
        }
    return out


def marker_cases(paths: list[Path]) -> list[dict[str, str]]:
    cases = [dict(get_ai_metadata(p)) for p in paths]
    cases += [
        {},
        {"claim_generator": "Sora", "issuer": "Google LLC", "source_type": "trainedAlgorithmicMedia"},
        {"claim_generator": "OpenAI Sora", "issuer": "OpenAI", "source_type": "trainedAlgorithmicMedia"},
        {"issuer": "Google LLC", "source_type": "trainedAlgorithmicMedia", "c2pa_signature": "invalid"},
        {"issuer": "ByteDance", "source_type": "trainedAlgorithmicMedia", "claim_generator": "Dreamina"},
        {"aigc_label": "x", "aigc_producer": "MiniMax"},
        {"aigc_label": "x", "aigc_producer": "kling"},
        {"aigc_label": "x", "aigc_producer": "unknown-vendor"},
        {"aigc_label": "x", "aigc_producer": "91110108MACC4D63XF"},
        {"app_aigc": "App AIGC disclosure (CapCut)"},
        {"app_provenance": "App export provenance (Dreamina)"},
        {"kling_sei": "present"},
        {"xai_signature": "sig"},
        {"topazcloud": "1"},
        {"description": "AI-enhanced using Topaz Labs"},
        {"workflow": "{}"},
        {"audio_generator": "ElevenLabs"},
        {"synthid_watermark": "SynthID", "issuer": "Google LLC", "source_type": "trainedAlgorithmicMedia"},
        {"soft_binding": "Adobe TrustMark", "soft_binding_algorithm": "com.adobe.trustmark.Q"},
        {"soft_binding": "Trufo PawPrint, Digimarc Validate"},
        {"issuer": "YouTube", "actions": "opened, transcoded"},
        {"issuer": "Microsoft", "source_type": "trainedAlgorithmicMedia", "c2pa_integrity": "valid",
         "c2pa_signature": "valid"},
        {"claim_generator": "TikTok Ad Creative Toolbox", "issuer": "TikTok"},
    ]
    return cases


def rules(markers: dict[str, str]) -> dict[str, Any]:
    marks = video.VIDEO_VISIBLE_MARKS
    return {
        "markers": markers,
        "claims_ai": video._video_markers_claim_ai(markers),
        "platform": video._platform_from_video_metadata(markers),
        "hints": list(video._video_invisible_watermark_hints(markers)),
        "bytedance": video._is_bytedance_video_provenance(markers),
        "cohort": video.video_invisible_target_cohort(markers),
        "cohort_veo": video.video_invisible_target_cohort(markers, "veo"),
        "contradicts": [m for m in marks if vv.contradicts_video_provenance(m, markers)],
        "confirms": [m for m in marks if vv.confirms_video_provenance(m, markers)],
        "recipes": {
            str(mark): video._recommend_video_recipe(
                markers=markers,
                selected_mark=mark,
                claims_ai=video._video_markers_claim_ai(markers),
                invisible_hints=video._video_invisible_watermark_hints(markers),
                audio_streams=(),
            )
            for mark in (None, "veo", "sora", "seedance")
        },
    }


def stabilize_cases() -> list[dict[str, Any]]:
    import random

    rng = random.Random(1234)
    patterns = {}
    for pattern in ("stable", "drift", "gappy", "weak"):
        detections = []
        for i in range(40):
            conf = {"stable": 0.7, "drift": 0.6, "gappy": 0.65, "weak": 0.33}[pattern] + rng.uniform(-0.1, 0.1)
            x = 100 + (i * 3 if pattern == "drift" else rng.randint(0, 1))
            gap = (pattern == "gappy" and i % 5 == 2) or (i > 30 and pattern != "stable")
            detections.append([i, round(conf, 6), None if gap else [x, 200, 60, 20]])
        patterns[pattern] = detections
    accepted = {}
    for mark in video.VIDEO_VISIBLE_MARKS:
        for pattern, detections in patterns.items():
            for provenance in (False, True):
                locs = [vv.FrameLocalization(i, c, tuple(r) if r else None) for i, c, r in detections]
                out = vv.stabilize_localizations(mark, locs, provenance=provenance)
                accepted[f"{mark}/{pattern}/{int(provenance)}"] = [r[0] if r else None for r in out]
    return {"patterns": patterns, "accepted_x": accepted}


def main() -> None:
    write_golden("video_identify", "stabilize", stabilize_cases())
    if sys.argv[1:] == ["--stabilize-only"]:
        return
    only = set(sys.argv[1:])
    paths = [p for p in fixtures() if not only or key(p) in only]
    write_golden("video_identify", "rules", [rules(m) for m in marker_cases(paths)])
    reports: dict[str, Any] = {}
    scans: dict[str, Any] = {}
    removals: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for path in paths:
            k = key(path)
            print(k, flush=True)
            reports[k] = {
                "visible": asdict(video.identify_video(path)),
                "metadata_only": asdict(video.identify_video(path, check_visible=False)),
            }
            scans[k] = scan_summary(path)
            if "provenance/" in k:
                out = Path(tmp) / f"{path.stem}_clean{path.suffix}"
                result = video.remove_video_metadata(path, out)
                removals[k] = {
                    "detected": result.detected,
                    "remaining": result.remaining,
                    "output_sha256": sha256(out),
                }
    write_golden("video_identify", "reports", reports)
    write_golden("video_identify", "scans", scans)
    write_golden("video_identify", "metadata_removal", removals)


if __name__ == "__main__":
    main()

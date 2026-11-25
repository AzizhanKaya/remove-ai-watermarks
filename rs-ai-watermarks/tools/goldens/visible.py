"""Record the Python visible-mark registry's detections, masks and removals for raiw-visible.

Run from the repository root: .venv/bin/python rs-ai-watermarks/tools/goldens/visible.py

Every result is recorded twice, with OpenCV's IPP on (the shipped wheels) and off (the
reference raiw-cv reproduces bit for bit). Images are never stored: fixtures are read in
place, synthetic inputs are integer formulas rebuilt by the Rust tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "rs-ai-watermarks" / "goldens" / "visible"
FIXTURES = ROOT / "data" / "fixtures"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

sys.path.insert(0, str(ROOT / "src"))

from remove_ai_watermarks import image_io  # noqa: E402
from remove_ai_watermarks import watermark_registry as reg  # noqa: E402
from remove_ai_watermarks._text_mark_engine import TextMarkDetection, TextMarkEngine  # noqa: E402
from remove_ai_watermarks.gemini_engine import get_watermark_config  # noqa: E402
from remove_ai_watermarks.pill_engine import PillDetection  # noqa: E402
from remove_ai_watermarks.qwen_engine import QwenSymbolDetection  # noqa: E402


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n")


# Synthetic inputs (mirrored by tests/common/synthetic.rs)


def synthetic(name: str) -> np.ndarray:
    if name == "flat-512":
        return np.full((512, 512, 3), 100, np.uint8)
    if name == "gradient-1024x768":
        h, w = 768, 1024
        yy, xx = np.mgrid[0:h, 0:w]
        return np.stack([(xx * 255) // (w - 1), (yy * 255) // (h - 1), ((xx + yy) * 255) // (w + h - 2)], axis=2).astype(
            np.uint8
        )
    if name == "noise-1200x900":
        h, w = 900, 1200
        yy, xx, cc = np.mgrid[0:h, 0:w, 0:3].astype(np.int64)
        return (((xx * 73856093) ^ (yy * 19349663) ^ (cc * 83492791)) % 251).astype(np.uint8)
    if name == "tiny-150x120":
        return np.full((120, 150, 3), 80, np.uint8)
    if name == "gray-800x600":
        h, w = 600, 800
        yy, xx = np.mgrid[0:h, 0:w]
        return ((xx * 3 + yy * 5) % 256).astype(np.uint8)
    if name == "bgra-900x700":
        h, w = 700, 900
        yy, xx, cc = np.mgrid[0:h, 0:w, 0:4].astype(np.int64)
        img = (((xx * 2654435761) ^ (yy * 40503) ^ (cc * 977)) % 241).astype(np.uint8)
        img[..., 3] = 255
        return img
    if name == "stripes-1536x1024":
        h, w = 1024, 1536
        yy, xx = np.mgrid[0:h, 0:w]
        v = np.where(((xx // 7) + (yy // 11)) % 2 == 0, 60, 190)
        return np.repeat(v[:, :, None], 3, axis=2).astype(np.uint8)
    raise KeyError(name)


SYNTHETIC = [
    "flat-512",
    "gradient-1024x768",
    "noise-1200x900",
    "tiny-150x120",
    "gray-800x600",
    "bgra-900x700",
    "stripes-1536x1024",
]


def fixture_images() -> list[Path]:
    paths = [p for p in sorted((FIXTURES / "visible").glob("*/*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    paths += [p for p in sorted((FIXTURES / "provenance").iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    return paths


def slug(path: Path) -> str:
    return "-".join(path.relative_to(FIXTURES).with_suffix("").parts) + path.suffix.lower().replace(".", "_")


# Detection records


def region(r) -> list[int]:
    return [int(v) for v in r]


def engine_fields(det) -> dict:
    if isinstance(det, TextMarkDetection):
        return {
            "kind": "symbol" if isinstance(det, QwenSymbolDetection) else "text",
            "match_box": None if det.match_box is None else [int(v) for v in det.match_box],
            "coverage": float(det.coverage),
            "provenance": bool(det.provenance),
            "template_asset": det.template_asset,
        }
    if isinstance(det, PillDetection):
        return {"kind": "pill", "verified_label": bool(det.verified_label)}
    return {"kind": "none"}


def mark_detection(m: reg.MarkDetection) -> dict:
    out = {"detected": bool(m.detected), "confidence": float(m.confidence), "region": region(m.region)}
    if m.engine_detection is not None:
        out["engine"] = engine_fields(m.engine_detection)
    return out


def gemini_raw(image: np.ndarray) -> dict:
    strict, relaxed = reg._engine("gemini").detect_watermark_both(image)

    def one(d) -> dict:
        return {
            "detected": bool(d.detected),
            "confidence": float(d.confidence),
            "region": region(d.region),
            "size": d.size.value,
            "spatial": float(d.spatial_score),
            "gradient": float(d.gradient_score),
            "variance": float(d.variance_score),
        }

    return {"strict": one(strict), "relaxed": one(relaxed)}


def mask_record(mask: np.ndarray | None) -> dict | None:
    if mask is None:
        return None
    x, y, w, h = cv2.boundingRect(mask)
    return {"sha256": sha(mask), "bbox": [x, y, w, h], "count": int(np.count_nonzero(mask))}


def detect_all(image: np.ndarray) -> dict:
    marks = []
    for m in reg.known_marks():
        strict, relaxed = m.detect_both(image)
        entry = {"key": m.key, "strict": mark_detection(strict), "relaxed": mark_detection(relaxed)}
        for level, det in (("strict", strict), ("relaxed", relaxed)):
            if det.detected:
                loc = m.localize(image, provenance=level == "relaxed", detection=det)
                entry[f"{level}_mask"] = mask_record(loc.mask)
        marks.append(entry)
    pill = reg._engine("jimeng_pill")
    return {
        "marks": marks,
        "gemini": gemini_raw(image),
        "pill_texture": float(pill.footprint_texture(image)),
    }


def diff_stats(before: np.ndarray, after: np.ndarray, box: list[int] | None) -> dict:
    diff = np.abs(after.astype(np.int64) - before.astype(np.int64))
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    inside = np.zeros(diff.shape, bool)
    if box is not None:
        x, y, w, h = box
        inside[y : y + h, x : x + w] = True
    out = {"outside_max": int(diff[~inside].max()) if (~inside).any() else 0}
    if box is not None:
        sel = diff[inside]
        out["inside_max"] = int(sel.max())
        out["inside_mean"] = float(sel.mean())
    return out


def remove_record(image: np.ndarray, provenance: frozenset[str]) -> dict:
    report = reg.remove_auto_marks_detailed(image, provenance=provenance, backend="cv2")
    marks = []
    union = np.zeros(image.shape[:2], bool)
    for mr in report.marks:
        x, y, w, h = mr.mask_bbox
        union[y : y + h, x : x + w] = True
        marks.append(
            {
                "key": mr.key,
                "label": mr.label,
                "location": mr.location,
                "region": region(mr.region),
                "mask_bbox": region(mr.mask_bbox),
                "backend": mr.backend,
                "confidence_before": float(mr.confidence_before),
                "confidence_after": None if mr.confidence_after is None else float(mr.confidence_after),
                "status": mr.status,
                "residual_region": None if mr.residual_region is None else region(mr.residual_region),
                "diff": diff_stats(image, report.image, list(mr.mask_bbox)),
            }
        )
    diff = np.abs(report.image.astype(np.int64) - image.astype(np.int64))
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    return {
        "provenance": sorted(provenance),
        "status": report.status,
        "labels": report.labels,
        "marks": marks,
        "sha256": sha(report.image),
        "outside_max": int(diff[~union].max()) if (~union).any() else 0,
    }


def provenance_for(path: Path) -> frozenset[str] | None:
    parts = path.relative_to(FIXTURES).parts
    if parts[0] != "visible":
        return None
    try:
        return frozenset({reg.get_mark(parts[1]).product})
    except KeyError:
        return None


def image_record(image: np.ndarray, provenance: frozenset[str] | None) -> dict:
    record = {}
    for mode, ipp in (("ipp", True), ("noipp", False)):
        cv2.ipp.setUseIPP(ipp)
        try:
            entry = {"detect": detect_all(image), "remove": remove_record(image, frozenset())}
            if provenance:
                entry["remove_provenance"] = remove_record(image, provenance)
        finally:
            cv2.ipp.setUseIPP(True)
        record[mode] = entry
    return record


# Static tables


def config_record(cfg) -> dict:
    out = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        out[f.name] = list(v) if isinstance(v, tuple) else v
    return out


def engine_tables() -> dict:
    tables = {}
    sizes = [(256, 256), (900, 1600), (1600, 900), (2048, 2048), (720, 1280), (300, 4000), (4000, 300), (1152, 1536)]
    for key in reg.mark_keys():
        engine = reg._engine(key)
        engines = []
        if isinstance(engine, TextMarkEngine):
            engines = [engine, *getattr(engine, "_variants", ())]
        elif hasattr(engine, "_label") and engine._label is not None:
            engines = [engine._label]
        rows = []
        for e in engines:
            locs = []
            for h, w in sizes:
                img = np.zeros((h, w, 3), np.uint8)
                loc = e.locate(img)
                locs.append({"h": h, "w": w, "base": int(e.scale_base(img)), "loc": [loc.x, loc.y, loc.w, loc.h]})
            rows.append({"config": config_record(e.config), "locate": locs})
        tables[key] = rows
    registry = []
    for m in reg.known_marks():
        registry.append(
            {
                "key": m.key,
                "label": m.label,
                "location": m.location,
                "in_auto": m.in_auto,
                "product": m.product,
                "manufacturer": m.manufacturer,
                "label_regime": m.label_regime,
                "platform": m.platform,
                "provenance_signals": list(m.provenance_signals),
                "provenance_platform_tokens": list(m.provenance_platform_tokens),
                "tc260_producer_codes": list(m.tc260_producer_codes),
                "can_corroborate": m.can_corroborate,
            }
        )
    gemini_cfg = []
    for w, h in [(1024, 1024), (1025, 1025), (2048, 512), (1536, 1152), (300, 300)]:
        c = get_watermark_config(w, h)
        gemini_cfg.append({"w": w, "h": h, "margin_right": c.margin_right, "logo_size": c.logo_size})
    return {"engines": tables, "registry": registry, "gemini_config": gemini_cfg}


def force_masks() -> dict:
    out = {}
    for name in ("gradient-1024x768", "flat-512"):
        image = synthetic(name)
        rows = {}
        for m in reg.known_marks():
            loc = m.localize(image, force=True)
            rows[m.key] = {"detected": bool(loc.detected), "mask": mask_record(loc.mask)}
        out[name] = rows
    return out


def main() -> None:
    only = set(sys.argv[1:])
    index = []
    for path in fixture_images():
        name = slug(path)
        if only and name not in only:
            continue
        bgr, alpha = image_io.read_bgr_and_alpha(path)
        assert bgr is not None, path
        record = {
            "source": str(path.relative_to(FIXTURES)),
            "decode": {"shape": list(bgr.shape), "sha256": sha(bgr), "alpha": None if alpha is None else sha(alpha)},
            **image_record(bgr, provenance_for(path)),
        }
        write(f"image-{name}", record)
        index.append({"name": name, "source": record["source"]})
        print(name, flush=True)
    for name in SYNTHETIC:
        if only and name not in only:
            continue
        image = synthetic(name)
        record = {"synthetic": name, "decode": {"shape": list(image.shape), "sha256": sha(image)}}
        record.update(image_record(image, None))
        write(f"image-{name}", record)
        index.append({"name": name, "synthetic": name})
        print(name, flush=True)
    if not only:
        write("index", index)
        write("tables", engine_tables())
        write("force", force_masks())


if __name__ == "__main__":
    main()

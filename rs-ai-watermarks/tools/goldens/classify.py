"""Golden generator for raiw-classify: real v10 traces, 512-d/956-d features and ViT preprocessing.

Run from the repository root with the oracle environment (needs the `source-classify` extra):

    ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/classify.py

The pinned `multi-provider-source-v10.npz` resolves through the Hugging Face cache.
`classify_bundles.py` records the synthetic-bundle behaviors of the Python tests.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify_common import (  # noqa: E402
    FIXTURES,
    f32_list,
    f64_list,
    hash_image,
    mask_bits,
    sha256_bytes,
    tensor_digest,
    write_golden,
)

from remove_ai_watermarks import source_classify as sc  # noqa: E402
from remove_ai_watermarks._internal import source_features_v4 as v4  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# Synthetic geometries covering tile-grid, lattice-crop, aspect and size branches of the extractors.
FEATURE_SPECS: list[dict[str, Any]] = [
    {"name": "noise_256", "w": 256, "h": 256, "seed": 1},
    {"name": "noise_400x300", "w": 400, "h": 300, "seed": 2, "lo": 20, "hi": 235},
    {"name": "noise_700x511", "w": 700, "h": 511, "seed": 3},
    {"name": "noise_512", "w": 512, "h": 512, "seed": 4, "lo": 40, "hi": 200},
    {"name": "noise_1500x257", "w": 1500, "h": 257, "seed": 5},
    {"name": "noise_2100x1000", "w": 2100, "h": 1000, "seed": 6, "lo": 0, "hi": 64},
    {"name": "noise_1000x2100", "w": 1000, "h": 2100, "seed": 7, "lo": 100, "hi": 140},
    {"name": "flat_gray_600", "w": 600, "h": 600, "seed": 8, "lo": 128, "hi": 129},
    {"name": "wide_r_strip", "w": 1408, "h": 768, "seed": 9, "lo": 30, "hi": 120,
     "paint": [{"c": 0, "v": 250}, {"c": 0, "v": 20, "x0": 480, "x1": 928}]},
    {"name": "checker16_1024", "w": 1024, "h": 1024, "seed": 10, "lo": 60, "hi": 70, "checker16": True},
]

V3_IMAGES = ("provenance/flux-1.png", "provenance/grok-1.jpg", "provenance/recraft-v3.webp")
V3_SPECS = ("noise_256", "noise_400x300", "noise_1500x257")


def fixture_images() -> list[Path]:
    paths = sorted(p for p in (FIXTURES / "provenance").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    paths += sorted(p for p in (FIXTURES / "visible").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths


def golden_name(path: Path) -> str:
    rel = path.relative_to(FIXTURES).with_suffix("")
    return "fixture_" + re.sub(r"[^a-z0-9]+", "_", str(rel).lower()).strip("_")


def _pil_rgb(img: Image.Image) -> np.ndarray:
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    return np.asarray(rgb, dtype=np.uint8)


def _v5_state(feats: np.ndarray, model: Any) -> dict[str, Any]:
    z_det_std, attr_logits, r_std, mean_sub, u_vec, accept_excess = sc._eval_v5_core_state(feats, model.base_v5)
    return {
        "z_det_std": f64_list(z_det_std),
        "attr_logits": f64_list(attr_logits),
        "r_std": f64_list(r_std),
        "mean_sub": f64_list(mean_sub),
        "u_vec": f64_list(u_vec),
        "accept_excess": float(accept_excess),
    }


def _routed_boxes(rgb_img: Image.Image, pmap: np.ndarray, vmask: np.ndarray, geom: tuple[int, ...]) -> list[list[int]]:
    spec = sc.ROUTED_SPEC_V10
    src = sc._resize_short_edge_bilinear(rgb_img, spec.img_size) if min(rgb_img.size) < spec.img_size else rgb_img
    sw, sh = src.size
    if (sw, sh) == rgb_img.size:
        boxes = sc._select_routed_boxes(
            pmap, vmask, geom, (sw, sh),
            crop_sz=spec.img_size, patch_size=spec.patch_size, grid=spec.grid, min_dist_sq=spec.min_dist_sq,
        )
        return [list(b) for b in boxes]
    return []


def trace_fixture(path: Path, model: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {"fixture": str(path.relative_to(FIXTURES))}
    with Image.open(path) as img:
        rec["size"] = list(img.size)
        rec["mode"] = img.mode
        rec["decode_sha256"] = sha256_bytes(_pil_rgb(img).tobytes())
        full = sc.classify_source(path).to_dict()
        rec["result"] = full
        if min(img.size) < sc.MIN_SIDE:
            return rec
        feats = sc.extract_512d_v4_features(img)
        rec["features"] = None if feats is None else f64_list(feats)
        if feats is None:
            return rec
        rgb_img = sc._orient_rgb_image(img)
        rec["oriented_size"] = list(rgb_img.size)
        spec = sc.ROUTED_SPEC_V10
        pix, vmask, geom = sc._preprocess_vit_padded(
            rgb_img, target=spec.img_size, patch_size=spec.patch_size, grid=spec.grid
        )
        rec["primary_pixels"] = tensor_digest(pix)
        rec["valid_mask"] = mask_bits(vmask)
        rec["geom"] = list(geom)
        session = model.onnx_holder.get_session()
        vec, pmap, gfx = session.run(
            ["vit_vec", "patch_clean_24x24", "patch_gfx_24x24"], {"pixel_values": pix, "valid_mask": vmask}
        )
        rec["vit_primary"] = f32_list(vec[0])
        rec["patch_clean"] = f32_list(pmap[0])
        rec["patch_gfx_sum"] = float(np.asarray(gfx, dtype=np.float64).sum())
        pmap64 = np.asarray(pmap[0], dtype=np.float64)
        rec["routed_boxes"] = _routed_boxes(rgb_img, pmap64, vmask[0], geom)
        pix2, mask2 = sc._preprocess_vit_routed_2chw(
            rgb_img, pmap64, vmask[0], geom,
            crop_sz=spec.img_size, patch_size=spec.patch_size, grid=spec.grid, min_dist_sq=spec.min_dist_sq,
        )
        rec["detail_pixels"] = tensor_digest(pix2)
        vec2 = session.run(["vit_vec"], {"pixel_values": pix2, "valid_mask": mask2})[0]
        rec["vit_detail"] = [f32_list(vec2[0]), f32_list(vec2[1])]

        x_1d = np.asarray(feats, dtype=np.float64)
        v_1d = np.asarray(vec[0], dtype=np.float64)
        v_det = np.asarray(vec2, dtype=np.float64)
        calls: list[int] = []

        def loader() -> np.ndarray:
            calls.append(1)
            return v_det

        replay = sc._classify_loaded_model(
            x_1d, model, vit_features=v_1d, vit_detail_loader=loader, image_size=rgb_img.size
        ).to_dict()
        if replay != full:
            raise AssertionError(f"replay mismatch for {path}: {replay} != {full}")
        rec["loader_called"] = bool(calls)

        state = _v5_state(x_1d, model)
        rec["core"] = state
        gfx_score = sc.eval_born_digital_graphics_score(x_1d)
        z = np.asarray(state["z_det_std"])
        r_std = np.asarray(state["r_std"])
        syn_in = sc._build_routed_syn_input_1d(
            x_1d, z, state["accept_excess"], r_std, v_1d, slices=spec.slices, gfx_score=gfx_score
        )
        syn_raw = sc._eval_syn_head_scalar(syn_in, model)
        syn_margin = syn_raw - model.syn_threshold
        eff = sc._apply_phys_veto(state["accept_excess"], syn_margin, model)
        need_gray, need_local = sc._should_run_v10_routed_tiles(
            x_1d, v_1d, eff, syn_margin, rgb_img.size, model.nt_cfg, gfx_score=gfx_score
        )
        mm_1v = sc._eval_mm_prov_head_log_odds(
            sc.build_v10_mm_prov_input_1d(np.asarray(state["attr_logits"]), z, state["accept_excess"], r_std, v_1d),
            model.mm_prov,
        )
        mm_3v = sc._eval_mm_prov_head_log_odds(
            sc.build_v10_mm_prov_input_1d(
                np.asarray(state["attr_logits"]), z, state["accept_excess"], r_std, v_1d, v_det
            ),
            model.mm_prov,
        )
        loc = sc._eval_local_head_scalar_v9(
            sc.build_v10_local_input_1d(x_1d, state["accept_excess"], v_1d, v_det), model.local_head
        )
        rec["trace"] = {
            "gfx_score": gfx_score,
            "syn_input": f64_list(syn_in),
            "syn_raw": syn_raw,
            "eff_phys_excess": eff,
            "need_gray": need_gray,
            "need_local": need_local,
            "mm_log_odds_1v": f64_list(mm_1v),
            "mm_log_odds_3v": f64_list(mm_3v),
            "local_raw_3v": loc,
        }
    return rec


def run_fixtures(only: str | None) -> None:
    model = sc._load_model()
    if not isinstance(model, sc._SourceModelV10):
        raise SystemExit("pinned model is not v10")
    names = []
    for path in fixture_images():
        name = golden_name(path)
        if only and only not in name:
            continue
        rec = trace_fixture(path, model)
        write_golden(name, rec)
        names.append(name)
        print(name, rec["result"]["label"], rec["result"]["reason"], rec.get("loader_called"), flush=True)
    if not only:
        write_golden("fixtures_index", {"names": names, "weights_sha256": sc.MODEL_SHA256})


def run_features() -> None:
    cases = []
    for spec in FEATURE_SPECS:
        arr = hash_image(spec)
        img = Image.fromarray(arr, mode="RGB")
        feats = v4.extract_v4_features(img)
        case: dict[str, Any] = {
            "spec": spec,
            "rgb_sha256": sha256_bytes(arr.tobytes()),
            "features": None if feats is None else f64_list(feats),
        }
        if spec["name"] in V3_SPECS:
            v3 = sc.extract_956d_features(img)
            case["features_956"] = None if v3 is None else f64_list(v3)
        cases.append(case)
        print("features", spec["name"], flush=True)
    v3_fixtures = []
    for rel in V3_IMAGES:
        with Image.open(FIXTURES / rel) as img:
            v3 = sc.extract_956d_features(img)
        v3_fixtures.append({"fixture": rel, "features_956": None if v3 is None else f64_list(v3)})
    write_golden("features", {"cases": cases, "v3_fixtures": v3_fixtures})


def run_preprocess() -> None:
    """ViT preprocessing digests (PIL bilinear/lanczos, crops, letterbox) for every schema family."""
    cases = []
    specs = [s for s in FEATURE_SPECS if s["name"] in ("noise_400x300", "noise_700x511", "noise_1500x257", "checker16_1024", "wide_r_strip")]
    specs += [
        {"name": "small_300x280", "w": 300, "h": 280, "seed": 11},
        {"name": "tall_390x410", "w": 390, "h": 410, "seed": 12},
        {"name": "ultra_wide_2000x300", "w": 2000, "h": 300, "seed": 13},
        {"name": "ultra_tall_300x1200", "w": 300, "h": 1200, "seed": 14},
        {"name": "exact_384", "w": 384, "h": 384, "seed": 15},
    ]
    for spec in specs:
        img = Image.fromarray(hash_image(spec), mode="RGB")
        case: dict[str, Any] = {"spec": spec}
        case["v7_center"] = tensor_digest(sc.preprocess_vit_384_chw(img))
        case["v7_detail"] = tensor_digest(sc.preprocess_vit_384_detail_2chw(img, short_edge=672))
        case["v8_native"] = tensor_digest(sc.preprocess_vit_384_native_2chw(img))
        for tag, fn in (("v9", sc.preprocess_vit_392_padded_v9), ("v10", sc.preprocess_vit_384_padded_v10)):
            pix, vmask, geom = fn(img)
            case[f"{tag}_padded"] = tensor_digest(pix)
            case[f"{tag}_valid_mask"] = mask_bits(vmask)
            case[f"{tag}_geom"] = list(geom)
            grid = int(round(vmask.shape[1] ** 0.5))
            pmap = np.asarray(
                (np.arange(grid * grid, dtype=np.float64) * 7919 % 1009).reshape(grid, grid) / 1009.0
            )
            routed_fn = sc.preprocess_vit_392_routed_2chw_v9 if tag == "v9" else sc.preprocess_vit_384_routed_2chw_v10
            pix2, _ = routed_fn(img, pmap, vmask[0], geom)
            case[f"{tag}_routed"] = tensor_digest(pix2)
            pix_fallback, _ = routed_fn(img)
            case[f"{tag}_routed_fallback"] = tensor_digest(pix_fallback)
        for size in ((256, 256), (97, 513), (1031, 77)):
            for name, resample in (("bilinear", Image.Resampling.BILINEAR), ("lanczos", Image.Resampling.LANCZOS)):
                out = np.asarray(img.resize(size, resample), dtype=np.uint8)
                case[f"resize_{name}_{size[0]}x{size[1]}"] = sha256_bytes(out.tobytes())
        cases.append(case)
        print("preprocess", spec["name"], flush=True)
    write_golden("preprocess", {"cases": cases})


def main(argv: list[str]) -> None:
    what = argv[1] if len(argv) > 1 else "all"
    only = argv[2] if len(argv) > 2 else None
    if what in ("all", "features"):
        run_features()
    if what in ("all", "preprocess"):
        run_preprocess()
    if what in ("all", "fixtures"):
        run_fixtures(only)


if __name__ == "__main__":
    main(sys.argv)

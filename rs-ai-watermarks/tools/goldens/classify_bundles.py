"""Golden generator for raiw-classify synthetic bundles (every schema, v3 through v10).

Runs `tests/test_source_classify.py` under an in-process recorder that captures each
synthetic `.npz` a test writes, every `_classify_loaded_model` call (features, ViT
vectors, lazy detail-loader use, result), every load-time rejection, and the public
feature builders called directly by a test. A second pass classifies synthetic
`hash_image` inputs end to end with a few of those bundles.

    ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/classify_bundles.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify_common import BUNDLES, REPO, f64_list, hash_image, write_golden  # noqa: E402

from remove_ai_watermarks import source_classify as sc  # noqa: E402

TEST_FILE = REPO / "tests" / "test_source_classify.py"

_real_savez = np.savez
_real_load = sc._load_model_from_path
_real_classify = sc._classify_loaded_model

PAYLOADS: dict[str, dict[str, Any]] = {}
MODEL_PATHS: dict[int, str] = {}
STORED: dict[str, str] = {}
RECORD: dict[str, list[Any]] = {"calls": [], "load_errors": [], "builders": []}
STATE = {"depth": 0, "test": ""}


def _payload_digest(payload: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(payload):
        arr = np.asanyarray(payload[key])
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(repr(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def write_npz_deterministic(path: Path, payload: dict[str, Any]) -> None:
    """Deflate-compressed `.npz` with fixed timestamps so regeneration is byte-stable."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, value in payload.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w") as handle:
                np.lib.format.write_array(handle, np.asanyarray(value), allow_pickle=False)


def store_bundle(path: str) -> str | None:
    key = str(Path(path))
    if key in STORED:
        return STORED[key]
    payload = PAYLOADS.get(key)
    if payload is None:
        return None
    name = f"b_{_payload_digest(payload)}.npz"
    BUNDLES.mkdir(parents=True, exist_ok=True)
    target = BUNDLES / name
    if not target.exists():
        write_npz_deterministic(target, payload)
    STORED[key] = name
    return name


def savez_hook(file: Any, *args: Any, **kwds: Any) -> None:
    _real_savez(file, *args, **kwds)
    if not args:
        PAYLOADS[str(Path(file))] = {k: np.array(v, copy=True) for k, v in kwds.items()}


def load_hook(resolved: Path) -> Any:
    try:
        model = _real_load(resolved)
    except Exception as exc:
        RECORD["load_errors"].append(
            {"test": STATE["test"], "bundle": store_bundle(str(resolved)), "error": type(exc).__name__, "message": str(exc)}
        )
        raise
    MODEL_PATHS[id(model)] = str(resolved)
    return model


load_hook.cache_clear = _real_load.cache_clear  # type: ignore[attr-defined]


def _opt(arr: Any) -> Any:
    return None if arr is None else np.asarray(arr, dtype=np.float64).tolist()


def classify_hook(features: Any, model: Any, **kwargs: Any) -> Any:
    loader = kwargs.get("vit_detail_loader")
    loader_rec: dict[str, Any] = {"called": 0}
    if loader is not None:

        def wrapped() -> Any:
            out = loader()
            loader_rec["called"] += 1
            loader_rec["value"] = np.asarray(out, dtype=np.float64)
            return out

        kwargs["vit_detail_loader"] = wrapped
    STATE["depth"] += 1
    try:
        res = _real_classify(features, model, **kwargs)
    finally:
        STATE["depth"] -= 1
    path = MODEL_PATHS.get(id(model))
    if path is not None:
        size = kwargs.get("image_size")
        RECORD["calls"].append(
            {
                "test": STATE["test"],
                "bundle": store_bundle(path),
                "features": f64_list(features),
                "vit_features": _opt(kwargs.get("vit_features")),
                "vit_detail_features": _opt(kwargs.get("vit_detail_features")),
                "has_loader": loader is not None,
                "loader_calls": loader_rec["called"],
                "loader_value": _opt(loader_rec.get("value")),
                "image_size": None if size is None else list(size),
                "result": res.to_dict(),
            }
        )
    return res


BUILDERS = (
    "build_v7_syn_input_1d",
    "build_v7_mm_prov_input_1d",
    "build_v8_syn_input_1d",
    "build_v8_mm_prov_input_1d",
    "build_v9_syn_input_1d",
    "build_v10_syn_input_1d",
    "build_v9_mm_prov_input_1d",
    "build_v10_mm_prov_input_1d",
    "build_v9_local_input_1d",
    "build_v10_local_input_1d",
    "select_v9_routed_boxes",
    "select_v10_routed_boxes",
    "augment_892d_to_956d",
)


def _wrap_builder(name: str) -> Any:
    real = getattr(sc, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        out = real(*args, **kwargs)
        if STATE["depth"] == 0:
            RECORD["builders"].append(
                {
                    "test": STATE["test"],
                    "fn": name,
                    "args": [a if isinstance(a, (tuple, list)) else _opt(a) for a in args],
                    "kwargs": {k: _opt(v) for k, v in kwargs.items()},
                    "out": [list(b) for b in out] if name.startswith("select_") else _opt(out),
                }
            )
        return out

    return wrapper


class Recorder:
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: Any) -> None:
        STATE["test"] = item.name


def install() -> None:
    np.savez = savez_hook  # type: ignore[assignment]
    sc._load_model_from_path = load_hook  # type: ignore[assignment]
    sc._classify_loaded_model = classify_hook  # type: ignore[assignment]
    for name in BUILDERS:
        setattr(sc, name, _wrap_builder(name))


def run_tests() -> None:
    install()
    code = pytest.main(["-q", "-p", "no:cacheprovider", "-x", str(TEST_FILE)], plugins=[Recorder()])
    if code != 0:
        raise SystemExit(f"pytest failed with {code}")


def _load_test_module() -> Any:
    spec = importlib.util.spec_from_file_location("raiw_classify_tests", TEST_FILE)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot import test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_e2e(tmp: Path) -> list[dict[str, Any]]:
    """End-to-end classify_source on hash images with bundles built by the test helpers."""
    t = _load_test_module()
    n_cls = len(sc.ALL_PROVIDER_CLASSES)
    g_idx = sc.ALL_PROVIDER_CLASSES.index("google")
    cases: list[tuple[str, Path, dict[str, Any]]] = []

    cases.append(("v3_google", t._write_synthetic_v3_npz(tmp / "e_v3.npz", target_class="google", target_global_bias=3.2, router_threshold=1.0), {"w": 256, "h": 256, "seed": 101}))
    cases.append(("v3_abstain", t._write_synthetic_v3_npz(tmp / "e_v3b.npz", target_class="google", target_global_bias=0.2, router_threshold=1.0), {"w": 300, "h": 260, "seed": 102}))
    cases.append(("v4_bytedance", t._write_synthetic_v4_npz(tmp / "e_v4.npz", target_class="bytedance", target_bias=3.2, router_threshold=1.0), {"w": 256, "h": 256, "seed": 103}))
    cases.append(("v5_lin", t._write_synthetic_v5_npz(tmp / "e_v5.npz", s2_hgb_weight=0.0), {"w": 256, "h": 256, "seed": 104}))

    vit_pos = np.zeros(sc.VIT_OUT_DIM_V7, dtype=np.float32)
    vit_pos[0] = 2.5
    cases.append(("v7_transcoded", t._write_synthetic_v7_npz(tmp / "e_v7.npz", vit_vec=vit_pos, det_score=1.0, det_threshold=3.0, syn_threshold=0.5, syn_conf_scale=1.2), {"w": 320, "h": 288, "seed": 105, "lo": 20, "hi": 235}))
    vit_base = np.zeros(sc.VIT_OUT_DIM_V7, dtype=np.float32)
    vit_base[0] = 0.35
    dyn = t._build_minimal_const_vit28_onnx_bytes(vit_base, r_chan_syn_slope=1.5)
    ms = {
        "vit_onnx_bytes": np.frombuffer(dyn, dtype=np.uint8).copy(),
        "ms_short_edge": np.asarray(672, dtype=np.int64),
        "ms_min_short_px": np.asarray(512, dtype=np.int64),
        "ms_min_long_px": np.asarray(640, dtype=np.int64),
        "ms_gray_lo": np.asarray(-2.5, dtype=np.float64),
        "ms_gfx_ceil": np.asarray(1.0, dtype=np.float64),
        "ms_detail_alpha": np.asarray(0.75, dtype=np.float64),
        "ms_thr_offset": np.asarray(0.0, dtype=np.float64),
    }
    wide_r = {"w": 1408, "h": 768, "seed": 106, "lo": 30, "hi": 120, "paint": [{"c": 0, "v": 250}, {"c": 0, "v": 20, "x0": 480, "x1": 928}]}
    cases.append(("v7_widescreen_ms", t._write_synthetic_v7_npz(tmp / "e_v7ms.npz", det_score=0.5, det_threshold=3.0, syn_threshold=0.5, overrides=ms), wide_r))

    d_mm = sc.MM_PROV_IN_DIM_V8
    d_vit = sc.VIT_OUT_DIM_V8
    vb8 = np.zeros(d_vit, dtype=np.float32)
    vb8[0] = 1.8
    vb8[1 + g_idx] = 0.2
    vb8[55] = 0.92
    dyn8 = t._build_minimal_const_vit_onnx_bytes(vb8, g_chan_col_slopes={1 + g_idx: 3.5})
    mm_skip = np.zeros((d_mm, n_cls + 2), dtype=np.float64)
    mm_skip[22 + d_vit + 1 + g_idx, g_idx] = 1.0
    mm_b2 = np.zeros(n_cls + 2, dtype=np.float64)
    mm_b2[n_cls] = -0.5
    mm_b2[n_cls + 1] = -3.0
    d_s2 = 2 + 1 + 102
    v8_over = {
        "vit_onnx_bytes": np.frombuffer(dyn8, dtype=np.uint8).copy(),
        "mm_prov_in_mean": np.zeros(d_mm), "mm_prov_in_scale": np.ones(d_mm),
        "mm_prov_w1": np.zeros((d_mm, 8)), "mm_prov_b1": np.zeros(8), "mm_prov_ln_g": np.ones(8), "mm_prov_ln_b": np.zeros(8),
        "mm_prov_w2": np.zeros((8, n_cls + 2)), "mm_prov_b2": mm_b2, "mm_prov_skip_w": mm_skip,
        "mm_prov_thresholds": np.full(n_cls, 1.5), "mm_prov_min_gap": np.asarray(0.8), "mm_prov_conf_scale": np.asarray(1.0),
        "mm_prov_phys_beta": np.asarray(0.0),
        "nt_min_short_px": np.asarray(384, dtype=np.int64), "nt_min_long_px": np.asarray(448, dtype=np.int64),
        "nt_gray_lo": np.asarray(-2.4), "nt_gray_hi": np.asarray(0.0), "nt_gfx_ceil": np.asarray(0.5),
        "nt_min_photo_frac": np.asarray(0.22), "nt_detail_alpha": np.asarray(0.6), "nt_thr_offset": np.asarray(0.0),
        "nt_attr_enable": np.asarray(1, dtype=np.int64), "nt_attr_min_short_px": np.asarray(448, dtype=np.int64),
        "nt_attr_uncertain_gap": np.asarray(2.5),
        "mm_s2_classes": np.asarray(["google"], dtype="<U32"),
        "mm_s2_google_subfamilies": np.asarray(["google:imagen_3", "google:gemini_3_pro_image"], dtype="<U48"),
        "mm_s2_google_in_mean": np.zeros(d_s2), "mm_s2_google_in_scale": np.ones(d_s2),
        "mm_s2_google_w1": np.zeros((d_s2, 8)), "mm_s2_google_b1": np.zeros(8), "mm_s2_google_w2": np.zeros((8, 2)),
        "mm_s2_google_b2": np.array([-1.0, 1.4]), "mm_s2_google_skip_w": np.zeros((d_s2, 2)),
        "mm_s2_google_blend_alpha": np.asarray(1.0), "mm_s2_google_min_prob_margin": np.asarray(0.10),
    }
    wide_g = {"w": 1408, "h": 768, "seed": 107, "lo": 40, "hi": 100, "paint": [{"c": 1, "v": 250}, {"c": 1, "v": 15, "x0": 480, "x1": 928}]}
    cases.append(("v8_attr_tiles", t._write_synthetic_v8_npz(tmp / "e_v8.npz", det_score=0.5, det_threshold=3.0, syn_threshold=0.5, overrides=v8_over), wide_g))

    for tag, writer, d_loc, skip_idx in (("v9", t._write_synthetic_v9_npz, sc.LOCAL_IN_DIM_V9, 21), ("v10", t._write_synthetic_v10_npz, sc.LOCAL_IN_DIM_V10, 36)):
        loc_skip = np.zeros(d_loc)
        loc_skip[skip_idx] = 1.0
        over = t._routed_local_and_nt_overrides(d_loc, loc_skip, loc_threshold=2.5, min_long_px=384, peak_trigger=1.0, contrast_trigger=0.7)
        bundle = writer(tmp / f"e_{tag}.npz", det_score=0.2, det_threshold=3.0, syn_threshold=1.5, overrides=over)
        cases.append((f"{tag}_routed_square", bundle, {"w": 416, "h": 416, "seed": 108, "lo": 20, "hi": 220}))
        cases.append((f"{tag}_routed_wide", bundle, {"w": 1200, "h": 500, "seed": 109, "lo": 0, "hi": 255}))
        cases.append((f"{tag}_small", bundle, {"w": 300, "h": 280, "seed": 110}))

    out = []
    for name, bundle, spec in cases:
        img = Image.fromarray(hash_image(spec), mode="RGB")
        res = sc.classify_source(img, weights_path=bundle).to_dict()
        out.append({"name": name, "bundle": store_bundle(str(bundle)), "image": spec, "result": res})
        print("e2e", name, res["label"], res["reason"], res["synthetic_detected"], flush=True)
    return out


def main() -> None:
    if BUNDLES.exists():
        shutil.rmtree(BUNDLES)
    run_tests()
    tmp = Path(__import__("tempfile").mkdtemp(prefix="raiw-classify-e2e-"))
    try:
        e2e = run_e2e(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    write_golden("bundles", {**RECORD, "e2e": e2e})
    print(
        "calls", len(RECORD["calls"]), "load_errors", len(RECORD["load_errors"]),
        "builders", len(RECORD["builders"]), "bundles", len(list(BUNDLES.glob("*.npz"))),
    )


if __name__ == "__main__":
    main()

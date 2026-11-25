"""Record OpenCV 4.10 reference outputs for the raiw-cv golden suite.

Run from the repository root: .venv/bin/python rs-ai-watermarks/tools/goldens/cv.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "rs-ai-watermarks" / "goldens" / "cv"


# Array encoding


FULL = os.environ.get("RAIW_GOLDEN_FULL") == "1"
SAMPLE = 12000


def arr(a: np.ndarray, full: bool = True) -> dict:
    """Encode an array: always its hash; payload in full, or a strided sample for big outputs."""
    a = np.ascontiguousarray(a)
    entry = {"dtype": str(a.dtype), "shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    flat = a.ravel()
    if full or FULL or flat.size <= SAMPLE:
        entry["zlib"] = base64.b64encode(zlib.compress(a.tobytes(), 9)).decode("ascii")
    elif a.dtype.kind == "f":
        stride = -(-flat.size // SAMPLE)
        entry["stride"] = stride
        entry["sample"] = base64.b64encode(zlib.compress(flat[::stride].tobytes(), 9)).decode("ascii")
    return entry


def out_arr(a: np.ndarray, needs_values: bool = False) -> dict:
    return arr(a, full=needs_values and a.size <= 4 * SAMPLE)


def write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def no_ipp(fn):
    cv2.ipp.setUseIPP(False)
    try:
        return fn()
    finally:
        cv2.ipp.setUseIPP(True)


def case(name: str, inp: str | list[str], params: dict, out, noipp=None) -> dict:
    """A golden case; integer outputs keep values only when IPP makes them inexact."""
    entry = {"name": name, "input": inp, "params": params}
    outs = out if isinstance(out, tuple) else (out,)
    entry["output"] = [out_arr(np.asarray(o), noipp is not None) if isinstance(o, np.ndarray) else o for o in outs]
    if noipp is not None:
        nouts = noipp if isinstance(noipp, tuple) else (noipp,)
        entry["noipp"] = [out_arr(np.asarray(o)) if isinstance(o, np.ndarray) else o for o in nouts]
    return entry


# Inputs


def build_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20241125)
    noise_c3 = rng.integers(0, 256, (61, 83, 3), dtype=np.uint8)
    noise_c1 = rng.integers(0, 256, (61, 83), dtype=np.uint8)
    noise_c4 = rng.integers(0, 256, (37, 45, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:96, 0:128].astype(np.float64)
    smooth = np.stack(
        [
            127.5 + 100 * np.sin(xx / 9.0) * np.cos(yy / 13.0),
            127.5 + 90 * np.cos((xx + yy) / 17.0),
            np.clip(xx * 2.0, 0, 255),
        ],
        axis=2,
    )
    smooth = np.clip(smooth + rng.normal(0, 6, smooth.shape), 0, 255).round().astype(np.uint8)
    fixture = cv2.imread(str(ROOT / "data/fixtures/visible/gemini/example.png"), cv2.IMREAD_COLOR)
    crop = np.ascontiguousarray(fixture[-150:, -200:])
    mask = np.zeros((90, 110), np.uint8)
    cv2.circle(mask, (30, 40), 14, 255, -1)
    cv2.rectangle(mask, (60, 20), (95, 34), 255, -1)
    mask[70:78, 10:100] = 255
    mask[(rng.random(mask.shape) > 0.995)] = 255
    small = rng.integers(0, 256, (9, 7, 3), dtype=np.uint8)
    return {
        "noise_c3": noise_c3,
        "noise_c1": noise_c1,
        "noise_c4": noise_c4,
        "smooth_c3": smooth,
        "smooth_c1": cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY),
        "fixture_c3": crop,
        "fixture_c1": cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
        "mask": mask,
        "small_c3": small,
        "noise_c3_f32": noise_c3.astype(np.float32) / 255.0,
        "noise_c1_f32": noise_c1.astype(np.float32) / 255.0,
        "smooth_c3_f32": smooth.astype(np.float32) / 255.0,
        "smooth_c1_f32": cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32),
        "smooth_c3_u16": smooth.astype(np.uint16) * 257 + rng.integers(0, 257, smooth.shape).astype(np.uint16),
        "fixture_c1_f64": cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64),
    }


INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos4": cv2.INTER_LANCZOS4,
}
BORDER = {
    "reflect101": cv2.BORDER_REFLECT_101,
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
    "wrap": cv2.BORDER_WRAP,
}


def resize_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    sources = ["noise_c3", "noise_c1", "smooth_c3", "noise_c4", "smooth_c3_f32", "noise_c1_f32", "smooth_c3_u16"]
    for src_name in sources:
        src = inputs[src_name]
        h, w = src.shape[:2]
        sizes = [(37, 29), (2 * w, 2 * h), (100, 77), (w // 2, h // 2), (w // 3, h // 3), (w + 1, h - 1), (16, 160)]
        for interp_name, interp in INTERP.items():
            for dw, dh in sizes:
                out = cv2.resize(src, (dw, dh), interpolation=interp)
                ref = no_ipp(lambda: cv2.resize(src, (dw, dh), interpolation=interp))
                name = f"{src_name}_{interp_name}_{dw}x{dh}"
                cases.append(
                    case(name, src_name, {"width": dw, "height": dh, "interp": interp_name}, out, ref if not np.array_equal(out, ref) else None)
                )
    for src_name, fx, interp_name in [("small_c3", 8.0, "cubic"), ("noise_c3", 0.5, "linear"), ("noise_c1", 1.7, "cubic"), ("smooth_c3", 0.37, "area")]:
        src = inputs[src_name]
        interp = INTERP[interp_name]
        out = cv2.resize(src, None, fx=fx, fy=fx, interpolation=interp)
        ref = no_ipp(lambda: cv2.resize(src, None, fx=fx, fy=fx, interpolation=interp))
        cases.append(
            case(f"{src_name}_by_{fx}_{interp_name}", src_name, {"fx": fx, "fy": fx, "interp": interp_name}, out, ref if not np.array_equal(out, ref) else None)
        )
    return cases


COLOR = {
    "bgr2rgb": cv2.COLOR_BGR2RGB,
    "bgr2gray": cv2.COLOR_BGR2GRAY,
    "rgb2gray": cv2.COLOR_RGB2GRAY,
    "gray2bgr": cv2.COLOR_GRAY2BGR,
    "bgr2hsv": cv2.COLOR_BGR2HSV,
    "bgr2ycrcb": cv2.COLOR_BGR2YCrCb,
    "ycrcb2bgr": cv2.COLOR_YCrCb2BGR,
    "bgr2yuv": cv2.COLOR_BGR2YUV,
    "rgb2yuv": cv2.COLOR_RGB2YUV,
    "yuv2bgr": cv2.COLOR_YUV2BGR,
    "yuv2rgb": cv2.COLOR_YUV2RGB,
    "bgr2lab": cv2.COLOR_BGR2Lab,
    "lab2bgr": cv2.COLOR_Lab2BGR,
    "rgba2bgra": cv2.COLOR_RGBA2BGRA,
    "bgra2rgba": cv2.COLOR_BGRA2RGBA,
    "bgra2bgr": cv2.COLOR_BGRA2BGR,
}


def color_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    plan = [
        ("noise_c3", ["bgr2rgb", "bgr2gray", "rgb2gray", "bgr2hsv", "bgr2ycrcb", "ycrcb2bgr", "bgr2yuv", "rgb2yuv", "yuv2bgr", "yuv2rgb"]),
        ("smooth_c3", ["bgr2gray", "bgr2hsv", "bgr2ycrcb", "ycrcb2bgr", "bgr2yuv", "yuv2bgr"]),
        ("noise_c1", ["gray2bgr"]),
        ("noise_c4", ["rgba2bgra", "bgra2rgba", "bgra2bgr", "bgr2gray"]),
        ("noise_c3_f32", ["bgr2gray", "bgr2hsv", "bgr2ycrcb", "bgr2lab", "bgr2yuv", "yuv2bgr", "yuv2rgb"]),
        ("smooth_c3_f32", ["bgr2gray", "bgr2lab", "ycrcb2bgr", "rgb2yuv"]),
        ("smooth_c3_u16", ["bgr2gray", "bgr2ycrcb", "ycrcb2bgr"]),
    ]
    cases = []
    for src_name, codes in plan:
        src = inputs[src_name]
        for code in codes:
            out = cv2.cvtColor(src, COLOR[code])
            ref = no_ipp(lambda: cv2.cvtColor(src, COLOR[code]))
            cases.append(case(f"{src_name}_{code}", src_name, {"code": code}, out, ref if not np.array_equal(out, ref) else None))
    lab = cv2.cvtColor(inputs["smooth_c3_f32"], cv2.COLOR_BGR2Lab)
    for name, src in [("lab_of_smooth", lab), ("lab_of_noise", cv2.cvtColor(inputs["noise_c3_f32"], cv2.COLOR_BGR2Lab))]:
        out = cv2.cvtColor(src, cv2.COLOR_Lab2BGR)
        cases.append({**case(name, "", {"code": "lab2bgr"}, out), "inline_input": arr(src)})
    return cases


def gaussian_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    plan = [
        ("noise_c3", (0, 0), 2.0, 0.0, "reflect101"),
        ("noise_c3", (3, 3), 0.7, 0.0, "reflect101"),
        ("noise_c1", (11, 11), 1.5, 0.0, "reflect101"),
        ("noise_c1", (5, 5), 0.0, 0.0, "reflect101"),
        ("noise_c1", (3, 3), 0.35, 0.0, "reflect101"),
        ("smooth_c3", (0, 0), 1.5, 0.0, "reflect"),
        ("smooth_c3", (0, 0), 3.0, 1.0, "constant"),
        ("noise_c1", (7, 9), 0.0, 0.0, "replicate"),
        ("noise_c4", (0, 0), 1.0, 0.0, "reflect101"),
        ("mask", (0, 0), 1.5, 0.0, "reflect101"),
        ("noise_c3_f32", (0, 0), 2.0, 0.0, "reflect101"),
        ("smooth_c3_f32", (0, 0), 12.0, 0.0, "reflect101"),
        ("smooth_c3_f32", (3, 3), 0.7, 0.0, "reflect101"),
        ("smooth_c1_f32", (0, 0), 5.0, 2.0, "reflect101"),
        ("fixture_c1_f64", (11, 11), 1.5, 0.0, "reflect101"),
        ("smooth_c3_u16", (0, 0), 2.0, 0.0, "reflect101"),
    ]
    cases = []
    for src_name, ksize, sx, sy, border in plan:
        out = cv2.GaussianBlur(inputs[src_name], ksize, sigmaX=sx, sigmaY=sy, borderType=BORDER[border])
        params = {"kw": ksize[0], "kh": ksize[1], "sigma_x": sx, "sigma_y": sy, "border": border}
        cases.append(case(f"{src_name}_{ksize[0]}x{ksize[1]}_{sx}_{sy}_{border}", src_name, params, out))
    return cases


def deriv_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    for src_name in ["noise_c1", "smooth_c1_f32", "fixture_c1", "smooth_c3"]:
        src = inputs[src_name]
        for dx, dy, k in [(1, 0, 3), (0, 1, 3), (1, 1, 3), (1, 0, 5), (2, 0, 3)]:
            out = cv2.Sobel(src, cv2.CV_32F, dx, dy, ksize=k)
            cases.append(case(f"sobel_{src_name}_{dx}{dy}_k{k}", src_name, {"op": "sobel", "dx": dx, "dy": dy, "ksize": k, "ddepth": "f32"}, out))
        for ddepth, name in [(cv2.CV_32F, "f32"), (cv2.CV_64F, "f64")]:
            if src.dtype != np.uint8 and ddepth == cv2.CV_64F:
                continue
            out = cv2.Laplacian(src, ddepth)
            cases.append(case(f"laplacian_{src_name}_{name}", src_name, {"op": "laplacian", "ksize": 1, "ddepth": name}, out))
    srm = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], np.float32) / 4
    kb = np.array([1, 4, 6, 4, 1], np.float32) / 16
    for src_name in ["smooth_c1_f32", "noise_c1_f32"]:
        src = inputs[src_name]
        out = cv2.filter2D(src, cv2.CV_32F, srm, borderType=cv2.BORDER_REFLECT_101)
        cases.append(case(f"filter2d_{src_name}", src_name, {"op": "filter2d", "kernel": srm.ravel().tolist(), "kw": 5, "kh": 5}, out))
        out = cv2.sepFilter2D(src, cv2.CV_32F, kb, kb, borderType=cv2.BORDER_REFLECT_101)
        cases.append(case(f"sep_{src_name}", src_name, {"op": "sep", "kx": kb.tolist(), "ky": kb.tolist()}, out))
    return cases


def morph_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))  # noqa: E731
    rect = lambda k: cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))  # noqa: E731
    elements = []
    for k in [1, 2, 3, 5, 7, 9, 13, 25, 31]:
        elements.append(("ellipse", k, k, ell(k)))
    elements.append(("ellipse", 11, 5, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 5))))
    elements.append(("cross", 5, 5, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))))
    cases.append({"name": "structuring_elements", "elements": [{"shape": s, "w": w, "h": h, "mask": arr(m)} for s, w, h, m in elements]})
    ops = {"dilate": None, "erode": None, "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE, "tophat": cv2.MORPH_TOPHAT}
    plan = [
        ("mask", "dilate", "ellipse", 9),
        ("mask", "dilate", "ellipse", 25),
        ("mask", "dilate", "rect", 25),
        ("mask", "erode", "ellipse", 5),
        ("mask", "open", "rect", 2),
        ("mask", "close", "rect", 5),
        ("noise_c1", "dilate", "ellipse", 7),
        ("noise_c1", "erode", "rect", 3),
        ("smooth_c1", "tophat", "rect", 7),
        ("fixture_c1", "tophat", "rect", 7),
        ("fixture_c1", "tophat", "ellipse", 11),
        ("smooth_c1_f32", "tophat", "rect", 7),
        ("noise_c3", "dilate", "ellipse", 5),
        ("noise_c1_f32", "erode", "ellipse", 9),
    ]
    for src_name, op, shape, k in plan:
        se = ell(k) if shape == "ellipse" else rect(k)
        src = inputs[src_name]
        if op == "dilate":
            out = cv2.dilate(src, se)
        elif op == "erode":
            out = cv2.erode(src, se)
        else:
            out = cv2.morphologyEx(src, ops[op], se)
        cases.append(case(f"{src_name}_{op}_{shape}{k}", src_name, {"op": op, "shape": shape, "k": k}, out))
    out = cv2.dilate(inputs["mask"], np.ones((5, 5), np.uint8), iterations=2)
    cases.append(case("mask_dilate_rect5_iter2", "mask", {"op": "dilate", "shape": "rect", "k": 5, "iterations": 2}, out))
    return cases


def exact_ncc(img: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """TM_CCOEFF_NORMED in float64 (OpenCV's float32 DFT correlation drifts in flat windows)."""
    img = img.astype(np.float64).reshape(img.shape[0], img.shape[1], -1)
    tpl = tpl.astype(np.float64).reshape(tpl.shape[0], tpl.shape[1], -1)
    h, w = tpl.shape[:2]
    win = np.lib.stride_tricks.sliding_window_view(img, (h, w), axis=(0, 1))
    tz = tpl - tpl.mean(axis=(0, 1))
    wz = win - win.mean(axis=(3, 4), keepdims=True)
    num = np.einsum("yxchw,hwc->yx", wz, tz)
    den = np.sqrt((wz * wz).sum(axis=(2, 3, 4)) * (tz * tz).sum())
    return np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 0.0).astype(np.float32)


def template_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    plan = [
        ("fixture_c1", (120, 80, 50, 40)),
        ("fixture_c1", (10, 10, 30, 30)),
        ("smooth_c1", (40, 30, 21, 17)),
        ("noise_c1", (20, 15, 9, 9)),
        ("mask", (20, 20, 25, 25)),
        ("smooth_c1_f32", (50, 20, 33, 24)),
        ("smooth_c3", (40, 30, 21, 17)),
    ]
    for src_name, (x, y, w, h) in plan:
        img = inputs[src_name]
        tpl = img[y : y + h, x : x + w].copy()
        if src_name == "mask":
            tpl = cv2.dilate(tpl, np.ones((3, 3), np.uint8))
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        mm = cv2.minMaxLoc(res) if res.ndim == 2 else None
        noipp = no_ipp(lambda: cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED))
        entry = case(f"{src_name}_{x}_{y}_{w}x{h}", src_name, {"x": x, "y": y, "w": w, "h": h}, res, noipp)
        entry["template"] = arr(tpl)
        entry["exact"] = arr(exact_ncc(img, tpl), full=False)
        entry["minmax"] = [mm[0], mm[1], list(mm[2]), list(mm[3])]
        cases.append(entry)
    flat = np.full((10, 10), 7, np.uint8)
    res = cv2.matchTemplate(inputs["noise_c1"], flat, cv2.TM_CCOEFF_NORMED)
    entry = case("flat_template", "noise_c1", {}, res)
    entry["template"] = arr(flat)
    cases.append(entry)
    big = inputs["fixture_c1"]
    tpl = big[40:120, 30:190].copy()
    res = cv2.matchTemplate(big, tpl, cv2.TM_CCOEFF_NORMED)
    entry = case("fixture_large_template", "fixture_c1", {}, res)
    entry["template"] = arr(tpl)
    entry["exact"] = arr(exact_ncc(big, tpl), full=False)
    mm = cv2.minMaxLoc(res)
    entry["minmax"] = [mm[0], mm[1], list(mm[2]), list(mm[3])]
    cases.append(entry)
    return cases


def stats_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    for name in ["noise_c1", "noise_c3", "smooth_c1_f32", "fixture_c1", "mask"]:
        mean, sdv = cv2.meanStdDev(inputs[name])
        cases.append({"name": f"meanstd_{name}", "input": name, "mean": mean.ravel().tolist(), "std": sdv.ravel().tolist()})
    for name in ["noise_c1", "smooth_c1_f32", "fixture_c1_f64"]:
        mn, mx, lo, hi = cv2.minMaxLoc(inputs[name])
        cases.append({"name": f"minmax_{name}", "input": name, "minmax": [mn, mx, list(lo), list(hi)]})
    ties = np.zeros((5, 6), np.float32)
    ties[1, 2] = ties[3, 4] = 1.0
    ties[2, 1] = ties[4, 0] = -1.0
    mn, mx, lo, hi = cv2.minMaxLoc(ties)
    cases.append({"name": "minmax_ties", "inline_input": arr(ties), "minmax": [mn, mx, list(lo), list(hi)]})
    for name in ["smooth_c1_f32", "fixture_c1"]:
        src = inputs[name]
        out = cv2.normalize(src.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        cases.append(case(f"normalize_{name}", name, {"op": "normalize_minmax"}, out))
    small = cv2.resize(inputs["smooth_c3"], (40, 30), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    hist_l2 = cv2.normalize(hist, hist.copy())
    cases.append({"name": "calc_hist", "input": "smooth_c3", "hist": hist.ravel().tolist(), "hist_l2": hist_l2.ravel().tolist()})
    return cases


def inpaint_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    mask = inputs["mask"]
    img_c3 = cv2.resize(inputs["fixture_c3"], (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    img_c1 = cv2.cvtColor(img_c3, cv2.COLOR_BGR2GRAY)
    smooth = cv2.resize(inputs["smooth_c3"], (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    plan = [
        ("img_c3", img_c3, 3.0, "telea"),
        ("img_c3", img_c3, 5.0, "telea"),
        ("img_c1", img_c1, 3.0, "telea"),
        ("img_c3", img_c3, 3.0, "ns"),
        ("img_c1", img_c1, 3.0, "ns"),
        ("smooth_c3", smooth, 3.0, "telea"),
        ("img_c1_f32", img_c1.astype(np.float32), 3.0, "telea"),
        ("img_c1_u16", img_c1.astype(np.uint16) * 200, 2.0, "ns"),
    ]
    for name, img, radius, method in plan:
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        out = cv2.inpaint(img, mask, radius, flag)
        entry = case(f"{name}_{method}_{radius}", "mask", {"radius": radius, "method": method}, out)
        entry["image"] = arr(img)
        cases.append(entry)
    return cases


def warp_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    src = inputs["smooth_c3"]
    h, w = src.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    fx = (xx + 3.3 * np.sin(yy / 7.0) - 0.4).astype(np.float32)
    fy = (yy + 2.1 * np.cos(xx / 5.0) + 0.27).astype(np.float32)
    maps = {"map_x": arr(fx), "map_y": arr(fy)}
    remaps = [
        ("smooth_c3", "linear", "reflect101"),
        ("smooth_c3", "lanczos4", "reflect101"),
        ("smooth_c3", "cubic", "reflect"),
        ("smooth_c3", "linear", "reflect"),
        ("smooth_c1", "nearest", "constant1"),
        ("smooth_c1", "linear", "constant1"),
        ("smooth_c3_f32", "linear", "reflect"),
        ("smooth_c3_f32", "lanczos4", "reflect101"),
        ("smooth_c3_u16", "linear", "reflect101"),
    ]
    for name, interp, border in remaps:
        bmode = cv2.BORDER_CONSTANT if border.startswith("constant") else BORDER[border]
        kwargs = {"borderValue": 1} if border == "constant1" else {}
        out = cv2.remap(inputs[name], fx, fy, INTERP[interp], borderMode=bmode, **kwargs)
        ref = no_ipp(lambda: cv2.remap(inputs[name], fx, fy, INTERP[interp], borderMode=bmode, **kwargs))
        entry = case(f"remap_{name}_{interp}_{border}", name, {"op": "remap", "interp": interp, "border": border}, out, ref if not np.array_equal(out, ref) else None)
        cases.append(entry)
    rot = cv2.getRotationMatrix2D((50, 50), 17, 1.0)
    shear = np.float32([[1.0, 0.21, 6.0], [0.0, 1.0, 0.0]])
    shift = np.float32([[1, 0, 2.37], [0, 1, -1.61]])
    affines = [
        ("smooth_c1", rot, (128, 96), "linear", "constant"),
        ("smooth_c1", shift, (128, 96), "linear", "reflect"),
        ("smooth_c3", shear, (140, 96), "linear", "constant"),
        ("smooth_c1", rot, (100, 90), "nearest", "constant"),
        ("smooth_c1_f32", shift, (128, 96), "linear", "reflect"),
    ]
    for name, m, size, interp, border in affines:
        out = cv2.warpAffine(inputs[name], m, size, flags=INTERP[interp], borderMode=BORDER[border])
        ref = no_ipp(lambda: cv2.warpAffine(inputs[name], m, size, flags=INTERP[interp], borderMode=BORDER[border]))
        params = {"op": "affine", "m": np.asarray(m, np.float64).tolist(), "width": size[0], "height": size[1], "interp": interp, "border": border}
        cases.append(case(f"affine_{name}_{interp}_{border}_{size[0]}", name, params, out, ref if not np.array_equal(out, ref) else None))
    return [{"name": "maps", **maps}, {"name": "rotation", "m": rot.tolist(), "center": [50, 50], "angle": 17, "scale": 1.0}, *cases]


def misc_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    cases = []
    mask = inputs["mask"]
    bridged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    for name, img in [("mask", mask), ("bridged", bridged)]:
        for conn in (8, 4):
            n, labels, stats, cent = cv2.connectedComponentsWithStats(img, connectivity=conn)
            entry = case(f"cc_{name}_{conn}", "mask", {"op": "cc", "connectivity": conn}, labels)
            entry["image"] = arr(img)
            entry["count"] = int(n)
            entry["stats"] = stats.tolist()
            entry["centroids"] = cent.tolist()
            cases.append(entry)
    allowed = np.ones((96, 128), np.uint8)
    allowed[20:40, 30:70] = 0
    allowed[60:62, 100:120] = 0
    large = np.ones((150, 200), np.uint8)
    large[30:70, 40:120] = 0
    large[100:103, 10:190] = 0
    large[(np.random.default_rng(5).random(large.shape) > 0.999)] = 0
    for name, img in [("boxes", allowed), ("mask_inv", 255 - mask), ("large", large), ("all_ones", np.ones((20, 30), np.uint8))]:
        out = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ref = no_ipp(lambda: cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        entry = case(f"dist_{name}", "mask", {"op": "dist"}, out, ref if not np.array_equal(out, ref) else None)
        entry["image"] = arr(img)
        cases.append(entry)
    for name, lo, hi in [("fixture_c1", 50, 150), ("smooth_c1", 20, 60), ("noise_c1", 100, 200), ("fixture_c1", 150, 50)]:
        out = cv2.Canny(inputs[name], lo, hi)
        cases.append(case(f"canny_{name}_{lo}_{hi}", name, {"op": "canny", "low": lo, "high": hi}, out))
    f = inputs["smooth_c3_f32"]
    blurred = cv2.GaussianBlur(f, (0, 0), 2.0)
    cases.append(case("add_weighted", "smooth_c3_f32", {"op": "add_weighted", "alpha": 1.6, "beta": -0.6, "gamma": 0.0}, cv2.addWeighted(f, 1.6, blurred, -0.6, 0.0)))
    cases[-1]["second"] = arr(blurred)
    cases.append(case("convert_scale_abs", "smooth_c3_f32", {"op": "csa", "alpha": 255.0 / 1.0}, cv2.convertScaleAbs(f, alpha=255.0)))
    g = inputs["smooth_c1_f32"]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    cases.append(case("magnitude", "smooth_c1_f32", {"op": "magnitude"}, cv2.magnitude(gx, gy), no_ipp(lambda: cv2.magnitude(gx, gy))))
    for name, border in [("reflect101", "reflect101"), ("reflect", "reflect"), ("replicate", "replicate"), ("constant", "constant"), ("wrap", "wrap")]:
        out = cv2.copyMakeBorder(inputs["smooth_c3_f32"][10:40, 20:60], 7, 3, 9, 2, BORDER[border])
        cases.append(case(f"border_{name}", "smooth_c3_f32", {"op": "border", "border": border}, out))
    cases.append(case("transpose", "noise_c3", {"op": "transpose"}, cv2.transpose(inputs["noise_c3"])))
    cases.append(case("extract", "noise_c3", {"op": "extract", "channel": 1}, cv2.extractChannel(inputs["noise_c3"], 1)))
    return cases


def motion_pair() -> tuple[np.ndarray, np.ndarray]:
    """A textured frame and a sub-pixel shifted, slightly warped successor."""
    fixture = cv2.imread(str(ROOT / "data/fixtures/visible/gemini/example.png"), cv2.IMREAD_GRAYSCALE)
    a = np.ascontiguousarray(fixture[400:560, 600:800])
    m = np.float32([[1.0, 0.01, 2.6], [-0.008, 1.0, -1.3]])
    b = cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return a, b


def flow_cases() -> list[dict]:
    a, b = motion_pair()
    cases = [{"name": "frames", "a": arr(a), "b": arr(b)}]
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    cases.append({"name": "farneback", "output": [arr(flow, full=False)]})
    small_a = cv2.resize(a, (80, 64), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(b, (80, 64), interpolation=cv2.INTER_AREA)
    cases.append({"name": "farneback_small", "a": arr(small_a), "b": arr(small_b), "output": [arr(cv2.calcOpticalFlowFarneback(small_a, small_b, None, 0.5, 3, 15, 3, 5, 1.2, 0))]})
    pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01, minDistance=4)
    cases.append({"name": "gftt", "params": {"max": 200, "quality": 0.01, "min_distance": 4}, "points": pts.reshape(-1, 2).tolist()})
    mask = np.zeros_like(a)
    mask[20:140, 30:170] = 255
    pts_m = cv2.goodFeaturesToTrack(a, maxCorners=32, qualityLevel=0.05, minDistance=3, mask=mask)
    cases.append({"name": "gftt_mask", "mask": arr(mask), "params": {"max": 32, "quality": 0.05, "min_distance": 3}, "points": pts_m.reshape(-1, 2).tolist()})
    eig = cv2.cornerMinEigenVal(a, 3, 3)
    cases.append({"name": "min_eigen", "output": [arr(eig, full=False)]})
    for level in (3, 2):
        p1, st, err = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(15, 15), maxLevel=level)
        cases.append({"name": f"lk_{level}", "level": level, "next": p1.reshape(-1, 2).tolist(), "status": st.ravel().tolist(), "err": err.ravel().tolist()})
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    (sx, sy), resp = cv2.phaseCorrelate(fa, fb)
    cases.append({"name": "phase", "shift": [sx, sy], "response": resp})
    (sx, sy), resp = cv2.phaseCorrelate(fa[:97, :131], np.roll(fa, (3, -5), axis=(0, 1))[:97, :131])
    cases.append({"name": "phase_odd", "shift": [sx, sy], "response": resp})
    cases.append({"name": "pyrdown", "output": [arr(cv2.pyrDown(a))], "odd": [arr(cv2.pyrDown(a[:99, :151]))]})
    return cases


def draw_cases() -> list[dict]:
    cases = []
    canvas = np.zeros((90, 120), np.uint8)
    rect = ((60.3, 44.8), (70.0, 22.0), 23.0)
    box = cv2.boxPoints(rect)
    cases.append({"name": "box_points", "rect": [60.3, 44.8, 70.0, 22.0, 23.0], "points": box.tolist()})
    pts = np.rint(box).astype(np.int32)
    poly = canvas.copy()
    cv2.fillConvexPoly(poly, pts, 1)
    cases.append({"name": "fill_convex", "pts": pts.tolist(), "output": [arr(poly)]})
    clipped = canvas.copy()
    tri = np.array([[-20, 10], [80, -15], [130, 95]], np.int32)
    cv2.fillConvexPoly(clipped, tri, 200)
    cases.append({"name": "fill_convex_clipped", "pts": tri.tolist(), "output": [arr(clipped)]})
    for center, axes, angle in [((60, 45), (40, 25), 0), ((30, 60), (12, 30), 0), ((70, 40), (55, 18), 30), ((5, 5), (2, 1), 0)]:
        e = canvas.copy()
        cv2.ellipse(e, center, axes, angle, 0, 360, 255, -1)
        cases.append({"name": f"ellipse_{center[0]}_{axes[0]}_{angle}", "center": list(center), "axes": list(axes), "angle": angle, "output": [arr(e)]})
    cases.append({"name": "ellipse2poly", "points": cv2.ellipse2Poly((50, 40), (30, 17), 15, 0, 360, 5).tolist()})
    mask = np.zeros((50, 70), np.uint8)
    mask[12:31, 7:44] = 1
    mask[40, 60] = 1
    cases.append({"name": "bounding_rect", "mask": arr(mask), "rect": list(cv2.boundingRect(mask))})
    return cases


def helper_cases(inputs: dict[str, np.ndarray]) -> list[dict]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from remove_ai_watermarks._internal import image_alignment, image_math, image_quality, tiling

    cases = [{"name": "dct8", "output": [arr(image_math.dct_matrix(np, 8))]}, {"name": "dct5", "output": [arr(image_math.dct_matrix(np, 5))]}]
    a = inputs["fixture_c1"]
    b = cv2.GaussianBlur(a, (0, 0), 1.2)
    cases.append({"name": "ssim", "a": "fixture_c1", "value": image_quality.structural_similarity_gray(a, b), "b": arr(b)})
    plans = []
    for w, h, t, o in [(2048, 1536, 1024, 128), (1000, 700, 1024, 64), (3000, 1100, 1024, 2000), (1025, 1024, 1024, 0)]:
        plans.append({"w": w, "h": h, "tile": t, "overlap": o, "tiles": [list(tt) for tt in tiling.plan_tiles(w, h, t, o)]})
    cases.append({"name": "plan_tiles", "plans": plans})
    cases.append({"name": "feather", "output": [arr(tiling.feather_weights(37, 23, 9))], "w": 37, "h": 23, "overlap": 9})
    rng = np.random.default_rng(11)
    reference = cv2.cvtColor(cv2.resize(inputs["fixture_c3"], (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    baseline = np.clip(reference.astype(np.int16) + rng.integers(-12, 13, reference.shape), 0, 255).astype(np.uint8)
    candidate = np.clip(baseline.astype(np.int16) + rng.integers(-6, 7, reference.shape), 0, 255).astype(np.uint8)
    weights = image_alignment.local_error_weights(reference, baseline, candidate, sigma=24.0)
    cases.append({"name": "local_error_weights", "reference": arr(reference), "baseline": arr(baseline), "candidate": arr(candidate), "sigma": 24.0, "output": [arr(weights)]})
    yy, xx = np.indices(reference.shape[:2], dtype=np.float32)
    flow = np.stack([2.5 * np.sin(yy / 17.0), 1.5 * np.cos(xx / 23.0)], axis=2).astype(np.float32)
    out, weighted = image_alignment.attenuate_regenerated(baseline, flow, weights.astype(np.float32))
    cases.append({"name": "attenuate", "flow": arr(flow), "output": [arr(out), arr(weighted, full=False)]})
    donor = cv2.warpAffine(reference, np.float32([[1, 0, 1.7], [0, 1, -0.9]]), (160, 120), borderMode=cv2.BORDER_REFLECT)
    for boxes, mode in [((), "linear"), (((20, 30, 70, 60),), "lanczos")]:
        out, final_flow = image_alignment.align_regenerated(reference, donor, boxes, resampling=mode)
        scale = min(1.0, 768 / max(reference.shape[:2]))
        size = (max(64, round(reference.shape[1] * scale)), max(64, round(reference.shape[0] * scale)))
        grays = [cv2.GaussianBlur(cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA), (0, 0), 2) for im in (reference, donor)]
        small = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(grays[0], grays[1], None)
        cases.append({"name": f"align_{mode}", "boxes": [list(b) for b in boxes], "mode": mode, "donor": arr(donor), "small_flow": arr(small), "output": [arr(out), arr(final_flow, full=False)]})
    return cases


def main() -> None:
    inputs = build_inputs()
    write("inputs", {name: arr(a) for name, a in inputs.items()})
    write("resize", {"cases": resize_cases(inputs)})
    write("color", {"cases": color_cases(inputs)})
    write("gaussian", {"cases": gaussian_cases(inputs)})
    write("deriv", {"cases": deriv_cases(inputs)})
    write("morph", {"cases": morph_cases(inputs)})
    write("template", {"cases": template_cases(inputs)})
    write("stats", {"cases": stats_cases(inputs)})
    write("inpaint", {"cases": inpaint_cases(inputs)})
    write("warp", {"cases": warp_cases(inputs)})
    write("misc", {"cases": misc_cases(inputs)})
    write("flow", {"cases": flow_cases()})
    write("draw", {"cases": draw_cases()})
    write("helpers", {"cases": helper_cases(inputs)})


if __name__ == "__main__":
    main()

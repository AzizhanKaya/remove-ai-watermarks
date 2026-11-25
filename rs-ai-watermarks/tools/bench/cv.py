"""Time the OpenCV calls that raiw-cv reproduces (single call, best of N).

Run from the repository root: .venv/bin/python rs-ai-watermarks/tools/bench/cv.py
Compare with: cargo test -p raiw-cv --release --test bench -- --ignored --nocapture
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def best(fn, repeats: int) -> float:
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"opencv {cv2.__version__}, threads {cv2.getNumThreads()}")
    for side in (1024, 4096):
        repeats = 7 if side == 1024 else 3
        img = cv2.GaussianBlur(rng.integers(0, 256, (side, side, 3), dtype=np.uint8), (0, 0), 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)
        mask = np.zeros((side, side), np.uint8)
        cv2.circle(mask, (side // 2, side // 2), side // 12, 255, -1)
        tmpl = gray[side // 3 : side // 3 + 64, side // 4 : side // 4 + 96].copy()
        ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        yy, xx = np.indices((side, side), dtype=np.float32)
        mx, my = xx + 0.37, yy - 0.81
        rot = cv2.getRotationMatrix2D((side / 2, side / 2), 7.0, 1.0)
        ops = {
            "resize_linear_half": lambda: cv2.resize(img, (side // 2 + 1, side // 2 + 1), interpolation=cv2.INTER_LINEAR),
            "resize_area_0.37": lambda: cv2.resize(img, (int(side * 0.37), int(side * 0.37)), interpolation=cv2.INTER_AREA),
            "resize_lanczos_up": lambda: cv2.resize(img, (side * 5 // 4, side * 5 // 4), interpolation=cv2.INTER_LANCZOS4),
            "cvt_bgr2gray": lambda: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            "cvt_bgr2ycrcb": lambda: cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb),
            "gaussian_u8_s2": lambda: cv2.GaussianBlur(img, (0, 0), 2.0),
            "gaussian_f32_s2": lambda: cv2.GaussianBlur(gray_f, (0, 0), 2.0),
            "sobel_f32": lambda: cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            "dilate_ellipse9": lambda: cv2.dilate(mask, ell),
            "tophat_rect7": lambda: cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((7, 7), np.uint8)),
            "match_template": lambda: cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED),
            "remap_linear": lambda: cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
            "warp_affine": lambda: cv2.warpAffine(gray, rot, (side, side)),
        }
        if side == 1024:
            ops["inpaint_telea"] = lambda: cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            ops["farneback"] = lambda: cv2.calcOpticalFlowFarneback(gray, np.roll(gray, 2, axis=1), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        for name, fn in ops.items():
            print(f"{side:>5} {name:<20} {best(fn, repeats):9.2f} ms")


if __name__ == "__main__":
    main()

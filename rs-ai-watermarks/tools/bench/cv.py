"""Time the OpenCV calls that raiw-cv reproduces at video frame sizes (best of N, 6 threads).

Run from the repository root: .venv/bin/python rs-ai-watermarks/tools/bench/cv.py
Compare with: RAYON_NUM_THREADS=6 cargo test -p raiw-cv --release --test bench -- --ignored --nocapture
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


def frame(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 3)


def main() -> None:
    cv2.setNumThreads(6)
    print(f"opencv {cv2.__version__}, threads {cv2.getNumThreads()}")
    for w, h in ((1920, 1080), (3840, 2160)):
        repeats = 5 if w == 1920 else 3
        img = frame(w, h, 7)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (h // 12, h // 12), 0, 0, 360, 255, -1)
        tmpl = gray[h // 3 : h // 3 + 64, w // 4 : w // 4 + 96].copy()
        roi = gray[h // 3 - 100 : h // 3 + 200, w // 4 - 150 : w // 4 + 300].copy()
        ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        yy, xx = np.indices((h, w), dtype=np.float32)
        mx, my = xx + 0.37, yy - 0.81
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), 7.0, 1.0)
        ops = {
            "resize_linear_720p": lambda: cv2.resize(img, (w * 2 // 3, h * 2 // 3), interpolation=cv2.INTER_LINEAR),
            "resize_area_half": lambda: cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
            "resize_area_0.37": lambda: cv2.resize(img, (w * 37 // 100, h * 37 // 100), interpolation=cv2.INTER_AREA),
            "resize_lanczos_2/3": lambda: cv2.resize(img, (w * 2 // 3, h * 2 // 3), interpolation=cv2.INTER_LANCZOS4),
            "cvt_bgr2gray": lambda: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            "cvt_bgr2ycrcb": lambda: cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb),
            "gaussian_u8_s2": lambda: cv2.GaussianBlur(img, (0, 0), 2.0),
            "sobel_f32": lambda: cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            "dilate_ellipse9": lambda: cv2.dilate(mask, ell),
            "tophat_rect7": lambda: cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((7, 7), np.uint8)),
            "match_template_roi": lambda: cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED),
            "remap_linear": lambda: cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
            "warp_affine": lambda: cv2.warpAffine(gray, rot, (w, h)),
        }
        if w == 1920:
            small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            shifted = np.roll(small, 2, axis=1)
            pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=4)
            nxt = np.roll(gray, (1, 2), axis=(0, 1))
            ops["match_template_full"] = lambda: cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            ops["inpaint_telea"] = lambda: cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            ops["farneback_540p"] = lambda: cv2.calcOpticalFlowFarneback(small, shifted, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            ops["pyr_lk_200"] = lambda: cv2.calcOpticalFlowPyrLK(gray, nxt, pts, None, winSize=(15, 15), maxLevel=3)
        for name, fn in ops.items():
            print(f"{w}x{h} {name:<20} {best(fn, repeats):9.2f} ms")


if __name__ == "__main__":
    main()

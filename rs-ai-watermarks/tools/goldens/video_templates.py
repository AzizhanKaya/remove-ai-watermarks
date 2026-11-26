"""Freeze the video detector templates as grayscale PNG assets.

`video_visible._template_sources()` renders several silhouettes with Pillow's bundled font at
import time; Rust has no Pillow text rasterizer, so the exact arrays ship as frozen assets in
`crates/raiw-video/assets/video_templates/`, the way the Python package freezes its Hershey renders.

`.venv/bin/python rs-ai-watermarks/tools/goldens/video_templates.py` from the repository root.
"""

from __future__ import annotations

import hashlib
import json

import cv2

from video_common import WORKSPACE

from remove_ai_watermarks.video_visible import _template_sources

OUT = WORKSPACE / "crates" / "raiw-video" / "assets" / "video_templates"

# Keys loaded from the Python package's own assets; Rust embeds those files in place.
PACKAGE_ASSETS = {
    "dola": "video_dola_hershey_duplex.png",
    "kling-1": "video_kling_ai_hershey_simplex.png",
    "kling-2": "video_kling_ai_hershey_duplex.png",
    "kling-4": "video_klingai_hershey_simplex.png",
    "kling-5": "video_klingai_hershey_duplex.png",
    "wan": "wan_alpha.png",
    "doubao": "doubao_alpha.png",
    "jimeng": "jimeng_alpha.png",
    "jimeng_pill": "jimeng_pill.png",
    "yuanbao": "yuanbao_alpha.png",
    "qwen": "qwen_alpha.png",
    "baidu": "baidu_alpha.png",
    "runninghub": "runninghub_alpha.png",
    "microsoft": "microsoft_alpha.png",
    "generic_ai_label": "generic_ai_label_alpha.png",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.png"):
        stale.unlink()
    manifest = {}
    for key, array in sorted(_template_sources().items()):
        entry = {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        if key in PACKAGE_ASSETS:
            entry["package_asset"] = PACKAGE_ASSETS[key]
        else:
            path = OUT / f"{key}.png"
            if not cv2.imwrite(str(path), array, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
                raise RuntimeError(f"could not write {path}")
        manifest[key] = entry
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} templates to {OUT}")


if __name__ == "__main__":
    main()

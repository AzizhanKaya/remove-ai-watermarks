"""Torch CPU timings for the video neural paths ported to raiw-gpu.

Run from the repository root with the diffusion/video-detect extras, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/bench/gpu_video.py`.
Compare with `cargo test -p raiw-gpu --test vae autoencoder_timing -- --ignored --nocapture`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from remove_ai_watermarks._internal import video_message_regeneration  # noqa: E402


def best(run: object, repeats: int = 2) -> float:
    run()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    return min(samples)


def main() -> None:
    torch.set_num_threads(4)
    runtime = video_message_regeneration.load_runtime("cpu")
    vae, factor = runtime.vae, runtime.scaling_factor
    x = torch.rand(1, 3, 512, 512) * 2 - 1
    with torch.inference_mode():
        z = vae.encode(x).latent_dist.mode() * factor

        def encode() -> None:
            vae.encode(x).latent_dist.mode()

        def decode() -> None:
            vae.decode(z / factor)

        results = {"torch": torch.__version__, "threads": torch.get_num_threads(),
                   "vae_512_encode_s": best(encode), "vae_512_decode_s": best(decode)}
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

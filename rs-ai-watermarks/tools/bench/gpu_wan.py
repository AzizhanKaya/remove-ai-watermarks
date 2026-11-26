"""Torch CPU float32 timings for the Wan 2.1 paths ported to raiw-gpu.

Run from the repository root with the diffusion extra, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/bench/gpu_wan.py`.
Compare with `cargo test -p raiw-gpu --test wan wan_480p_timing -- --ignored --nocapture`.

The Python Wan path itself only runs on CUDA bf16 or MPS; this times the same diffusers
modules on CPU float32 (the transformer streamed block by block, as in the goldens).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "goldens"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gpu_common import synthetic_signal  # noqa: E402
from gpu_wan_transformer import StreamedTransformer  # noqa: E402
from gpu_wan_vae import load_vae  # noqa: E402

HEIGHT, WIDTH = 480, 832


def timed(run: object) -> float:
    start = time.perf_counter()
    run()
    return time.perf_counter() - start


def main() -> None:
    torch.set_num_threads(4)
    results: dict[str, object] = {"torch": torch.__version__, "threads": torch.get_num_threads()}
    with torch.inference_mode():
        vae = load_vae()
        video = torch.rand(1, 3, 5, HEIGHT, WIDTH) * 2 - 1
        latents = vae.encode(video).latent_dist.mode()
        results["vae_5x480x832_encode_s"] = timed(lambda: vae.encode(video).latent_dist.mode())
        results["vae_5x480x832_decode_s"] = timed(lambda: vae.decode(latents))
        del vae
        oracle = StreamedTransformer()
        hidden = torch.from_numpy(synthetic_signal(16 * (HEIGHT // 8) * (WIDTH // 8), 977, 4)).reshape(
            1, 16, 1, HEIGHT // 8, WIDTH // 8
        )
        context = torch.from_numpy(synthetic_signal(512 * 4096, 977, 70)).reshape(1, 512, 4096)
        results["transformer_1560_tokens_s"] = timed(
            lambda: oracle.forward(hidden, torch.tensor([618]), context, oracle.layers)
        )
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

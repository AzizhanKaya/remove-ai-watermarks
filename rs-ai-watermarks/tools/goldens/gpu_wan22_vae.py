"""Golden generator for the Wan 2.2 VAE cases (raiw-gpu `WanVae`, `AutoencoderKLWan`, `is_residual`).

Run from the repository root, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_wan22_vae.py`.

The Python package has no Wan 2.2 path, so the oracle is the diffusers module itself: the pinned
`Wan-AI/Wan2.2-TI2V-5B-Diffusers` `vae/` subfolder in float32 on CPU without tiling (the
diffusers default), recording `encode(...).latent_dist.mode()` and `decode(...).sample` for
seeded synthetic clips laid out as frames `[T, 3, H, W]`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_common import strided, synthetic_signal, write  # noqa: E402

SUITE = "gpu_wan22"
REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"


def load_vae() -> object:
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", revision=REVISION, torch_dtype=torch.float32)
    return vae.eval()


def case(vae: object, frames: int, height: int, width: int, seed: int) -> dict[str, object]:
    pixels = synthetic_signal(frames * 3 * height * width, 977, seed).reshape(frames, 3, height, width)
    record = {"kind": "synthetic", "frames": frames, "height": height, "width": width, "rate": 977, "seed": seed}
    with torch.inference_mode():
        video = torch.from_numpy(np.ascontiguousarray(pixels)).permute(1, 0, 2, 3).unsqueeze(0)
        latents = vae.encode(video).latent_dist.mode()
        decoded = vae.decode(latents).sample
    latents = latents[0]
    decoded = decoded[0].permute(1, 0, 2, 3)
    return {
        "input": record,
        "latent_shape": list(latents.shape),
        "latents": strided(latents.numpy(), 4096),
        "decoded_shape": list(decoded.shape),
        "decoded": strided(decoded.numpy(), 1024),
    }


def main() -> None:
    torch.set_num_threads(4)
    vae = load_vae()
    cases = {
        "clip_5x64x96": case(vae, 5, 64, 96, 71),
        "clip_9x32x64": case(vae, 9, 32, 64, 72),
    }
    write(SUITE, "vae", {"repo": REPO, "revision": REVISION, "cases": cases})


if __name__ == "__main__":
    main()

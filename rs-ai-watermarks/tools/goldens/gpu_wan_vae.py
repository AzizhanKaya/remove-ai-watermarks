"""Golden generator for the `gpu_wan` VAE cases (raiw-gpu `WanVae`, `AutoencoderKLWan`).

Run from the repository root, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_wan_vae.py`.

The Python Wan path refuses CPU, so this calls the diffusers module directly: the pinned
`vae/` subfolder in float32 with tiling and slicing enabled exactly like
`video_wan.load_video_wan_runtime`, then records `encode(...).latent_dist.mode()` and
`decode(...).sample` for seeded synthetic clips laid out as frames `[T, 3, H, W]`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gpu_common import strided, synthetic_signal, write  # noqa: E402

from remove_ai_watermarks import video_wan  # noqa: E402

SUITE = "gpu_wan"


def load_vae() -> object:
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        video_wan.WAN21_MODEL_ID,
        subfolder="vae",
        revision=video_wan.WAN21_MODEL_REVISION,
        torch_dtype=torch.float32,
    )
    vae.enable_tiling()
    vae.enable_slicing()
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
        "clip_9x64": case(vae, 9, 64, 64, 61),
        "tiled_5x64x320": case(vae, 5, 64, 320, 62),
    }
    write(
        SUITE,
        "vae",
        {"repo": video_wan.WAN21_MODEL_ID, "revision": video_wan.WAN21_MODEL_REVISION, "cases": cases},
    )


if __name__ == "__main__":
    main()

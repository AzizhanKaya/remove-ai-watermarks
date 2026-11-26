"""Golden generator for the `gpu_vae` suite (raiw-gpu `SdVae`, `stabilityai/sd-vae-ft-mse`).

Run from the repository root with the diffusion extra, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_vae.py`.

Loads the VAE exactly as `video_message_regeneration.load_runtime("cpu")` (pinned float32
safetensors) and records `encode` latents (`mode() * scaling_factor`) and `decode` samples
for two seeded synthetic frames and one fixture frame decoded in place by ffmpeg (BGR).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gpu_common import FIXTURES, strided, synthetic_signal, write  # noqa: E402

from remove_ai_watermarks._internal import video_message_regeneration  # noqa: E402

SUITE = "gpu_vae"
FIXTURE = "provenance/google-gemini-omni-flash-api.mp4"
FRAME_W, FRAME_H = 96, 64


def fixture_frame() -> tuple[dict[str, object], np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to decode the fixture frame")
    args = ["-v", "error", "-ss", "1.0", "-i", str(FIXTURES / FIXTURE), "-frames:v", "1", "-vf",
            f"scale={FRAME_W}:{FRAME_H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run([ffmpeg, *args], check=True, capture_output=True).stdout
    record = {"kind": "fixture_frame", "fixture": FIXTURE, "before": args[:5], "after": args[6:], "width": FRAME_W,
              "height": FRAME_H, "decoded_sha256": hashlib.sha256(raw).hexdigest()}
    bgr = np.frombuffer(raw, dtype=np.uint8).reshape(1, FRAME_H, FRAME_W, 3)
    rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return record, rgb.numpy()


def case(vae: object, factor: float, pixels: np.ndarray) -> dict[str, object]:
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(pixels, dtype=np.float32))
        latents = torch.cat([vae.encode(frame[None]).latent_dist.mode() for frame in x]) * factor
        decoded = torch.cat([vae.decode(z[None] / factor).sample for z in latents])
    return {
        "latent_shape": list(latents.shape),
        "latents": [float(v) for v in latents.reshape(-1)],
        "decoded": strided(decoded.numpy(), 512),
    }


def main() -> None:
    torch.set_num_threads(4)
    runtime = video_message_regeneration.load_runtime("cpu")
    vae, factor = runtime.vae, runtime.scaling_factor
    frames, height, width, seed = 2, 64, 64, 51
    pixels = synthetic_signal(frames * 3 * height * width, 977, seed).reshape(frames, 3, height, width)
    record = {"kind": "synthetic", "frames": frames, "height": height, "width": width, "rate": 977, "seed": seed}
    cases = {"synthetic_2x64": {"input": record, **case(vae, factor, pixels)}}
    record, pixels = fixture_frame()
    cases["fixture_frame"] = {"input": record, **case(vae, factor, pixels)}
    # diffusers tiling (256 px tiles, 0.25 overlap) as the synthid profile enables above 768 px
    frames, height, width, seed = 1, 64, 328, 53
    pixels = synthetic_signal(frames * 3 * height * width, 977, seed).reshape(frames, 3, height, width)
    record = {"kind": "synthetic", "frames": frames, "height": height, "width": width, "rate": 977, "seed": seed}
    vae.enable_tiling()
    cases["synthetic_tiled"] = {"input": record, "tiled": True, **case(vae, factor, pixels)}
    vae.disable_tiling()
    write(SUITE, "autoencoder", {"repo": video_message_regeneration.MODEL_ID,
                                 "revision": video_message_regeneration.REVISION, "scaling_factor": factor,
                                 "cases": cases})


if __name__ == "__main__":
    main()

"""Golden generator for the Wan 2.2 TI2V 5B transformer case (raiw-gpu `WanTransformer`).

Run from the repository root, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_wan22_transformer.py`.

The float32 checkpoint (19 GB) streams block by block exactly like the Wan 2.1 oracle
(`gpu_wan_transformer.StreamedTransformer`), on the same synthetic latents (48 channels) and
context. The scalar timestep is what the video-to-video path feeds every token, which
`expand_timesteps` reproduces element for element.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_common import write  # noqa: E402
from gpu_wan22_vae import REPO, REVISION, SUITE  # noqa: E402
from gpu_wan_transformer import StreamedTransformer, transformer_cases  # noqa: E402


def main() -> None:
    torch.set_num_threads(4)
    oracle = StreamedTransformer(REPO, REVISION)
    write(SUITE, "transformer", {"repo": REPO, "revision": REVISION, **transformer_cases(oracle)})


if __name__ == "__main__":
    main()

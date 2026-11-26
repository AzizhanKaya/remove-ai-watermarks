"""Golden generator for the Wan 2.2 TI2V 5B pipeline cases (raiw-gpu `WanPipeline::regenerate_chunk`).

Run from the repository root, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_wan22_pipeline.py`.

The Wan 2.1 pipeline oracle (`gpu_wan_pipeline.py`) on the 5B checkpoint: the float32 VAE
untiled (the diffusers default), the pinned scheduler at its own `flow_shift=5.0`, the
streamed float32 transformer, and `video_wan._run_wan_window` on the same fixture frames.
`WanVideoToVideoPipeline` derives its spatial factor as `2 ** len(temperal_downsample)` (8),
which misses Wan 2.2's patchify, so the pipeline's factor is set to the VAE's 16.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_common import synthetic_signal, write  # noqa: E402
from gpu_wan22_vae import REPO, REVISION, SUITE, load_vae  # noqa: E402
from gpu_wan_pipeline import HEIGHT, WIDTH, StreamedModule, fixture_frames  # noqa: E402
from gpu_wan_transformer import StreamedTransformer  # noqa: E402

from remove_ai_watermarks import video_wan  # noqa: E402


def runtime() -> video_wan.VideoWanRuntime:
    from diffusers.pipelines.wan.pipeline_wan_video2video import WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_pretrained(REPO, subfolder="scheduler", revision=REVISION)
    transformer = StreamedModule(StreamedTransformer(REPO, REVISION))
    vae = load_vae()
    pipe = WanVideoToVideoPipeline(tokenizer=None, text_encoder=None, transformer=transformer, vae=vae,
                                   scheduler=scheduler)
    pipe.vae_scale_factor_spatial = vae.config.scale_factor_spatial
    pipe.set_progress_bar_config(disable=True)
    embeds = torch.from_numpy(synthetic_signal(512 * 4096, 977, 70)).reshape(1, 512, 4096)
    return video_wan.VideoWanRuntime(model=REPO, revision=REVISION, resolved_device="cpu", pipe=pipe,
                                     prompt_embeds=embeds)


def main() -> None:
    torch.set_num_threads(4)
    record, bgr = fixture_frames()
    wan = runtime()
    cases = {}
    for name, strength, seed in (("seedance", video_wan.WAN21_SEEDANCE_STRENGTH, 5),
                                 ("synthid", video_wan.WAN21_SYNTHID_STRENGTH, 6)):
        start = time.perf_counter()
        out = video_wan._run_wan_window(list(bgr), wan, strength=strength, seed=seed, orig_h=HEIGHT, orig_w=WIDTH)
        elapsed = time.perf_counter() - start
        cases[name] = {
            "strength": strength,
            "seed": seed,
            "steps": video_wan.WAN21_INFERENCE_STEPS,
            "flow_shift": float(wan.pipe.scheduler.config.flow_shift),
            "seconds_cpu_fp32": round(elapsed, 2),
            "output_bgr_b64": base64.b64encode(np.ascontiguousarray(out, dtype=np.uint8).tobytes()).decode("ascii"),
        }
        print(f"{name}: {elapsed:.1f}s")
    write(SUITE, "pipeline", {"repo": REPO, "revision": REVISION, "input": record,
                              "context": {"kind": "synthetic", "tokens": 512, "dim": 4096, "rate": 977, "seed": 70},
                              "cases": cases})


if __name__ == "__main__":
    main()

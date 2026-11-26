"""Golden generator for the `gpu_wan` pipeline cases (raiw-gpu `WanPipeline::regenerate_chunk`).

Run from the repository root, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_wan_pipeline.py`.

`video_wan.load_video_wan_runtime` refuses CPU and float32, so this builds the same
`WanVideoToVideoPipeline` by hand (pinned float32 VAE with tiling and slicing, the pinned
UniPC scheduler with `flow_shift=3.0`, and the streamed float32 transformer from
`gpu_wan_transformer.py`), wraps it in a `VideoWanRuntime` on "cpu", and records
`video_wan._run_wan_window` on fixture frames. The prompt embedding is the synthetic
`[1, 512, 4096]` buffer of the transformer golden, since UMT5-XXL is not run here.
"""

from __future__ import annotations

import base64
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_common import FIXTURES, synthetic_signal, write  # noqa: E402
from gpu_wan_transformer import REPO, REVISION, StreamedTransformer  # noqa: E402
from gpu_wan_vae import load_vae  # noqa: E402

from remove_ai_watermarks import video_wan  # noqa: E402

SUITE = "gpu_wan"
FIXTURE = "provenance/google-gemini-omni-flash-api.mp4"
FRAMES, WIDTH, HEIGHT = 6, 96, 64


class StreamedModule(torch.nn.Module):
    """The pipeline's `transformer` component, backed by the streamed oracle."""

    def __init__(self, oracle: StreamedTransformer) -> None:
        super().__init__()
        self.inner = oracle.model
        self.oracle = oracle

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def config(self) -> object:
        return self.inner.config

    def forward(self, hidden_states, timestep, encoder_hidden_states, attention_kwargs=None, return_dict=False):
        return (self.oracle.forward(hidden_states, timestep, encoder_hidden_states, self.oracle.layers),)


def fixture_frames() -> tuple[dict[str, object], np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to decode the fixture frames")
    args = ["-v", "error", "-ss", "1.0", "-i", str(FIXTURES / FIXTURE), "-frames:v", str(FRAMES), "-vf",
            f"scale={WIDTH}:{HEIGHT}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run([ffmpeg, *args], check=True, capture_output=True).stdout
    record = {"kind": "fixture_frames", "fixture": FIXTURE, "before": args[:5], "after": args[6:], "frames": FRAMES,
              "width": WIDTH, "height": HEIGHT, "decoded_sha256": hashlib.sha256(raw).hexdigest()}
    return record, np.frombuffer(raw, dtype=np.uint8).reshape(FRAMES, HEIGHT, WIDTH, 3)


def runtime() -> video_wan.VideoWanRuntime:
    from diffusers.pipelines.wan.pipeline_wan_video2video import WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    base = UniPCMultistepScheduler.from_pretrained(REPO, subfolder="scheduler", revision=REVISION)
    scheduler = UniPCMultistepScheduler.from_config(base.config, flow_shift=video_wan.WAN21_FLOW_SHIFT)
    transformer = StreamedModule(StreamedTransformer())
    pipe = WanVideoToVideoPipeline(tokenizer=None, text_encoder=None, transformer=transformer, vae=load_vae(),
                                   scheduler=scheduler)
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
            "seconds_cpu_fp32": round(elapsed, 2),
            "output_bgr_b64": base64.b64encode(np.ascontiguousarray(out, dtype=np.uint8).tobytes()).decode("ascii"),
        }
        print(f"{name}: {elapsed:.1f}s")
    write(SUITE, "pipeline", {"repo": REPO, "revision": REVISION, "input": record,
                              "context": {"kind": "synthetic", "tokens": 512, "dim": 4096, "rate": 977, "seed": 70},
                              "cases": cases})


if __name__ == "__main__":
    main()

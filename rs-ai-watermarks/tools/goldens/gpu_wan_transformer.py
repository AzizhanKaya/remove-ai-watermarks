"""Golden generator for the `gpu_wan` transformer, scheduler and prompt-cache cases.

Run from the repository root, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_wan_transformer.py`.

The float32 `WanTransformer3DModel` (5.7 GB) does not fit next to a desktop session, so the
oracle builds the pinned config with one block, loads the non-block weights, and replays
`WanTransformer3DModel.forward` loading each of the 30 pinned blocks into that slot in turn
(checked against the stock `forward` on a one-block model first). The scheduler is the pinned
`UniPCMultistepScheduler.from_config(..., flow_shift=3.0)` stepped on synthetic buffers.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gpu_common import strided, synthetic_signal, write  # noqa: E402

from remove_ai_watermarks import video_wan  # noqa: E402

SUITE = "gpu_wan"
REPO, REVISION = video_wan.WAN21_MODEL_ID, video_wan.WAN21_MODEL_REVISION


def shards() -> list[str]:
    index = hf_hub_download(REPO, "transformer/diffusion_pytorch_model.safetensors.index.json", revision=REVISION)
    names = sorted(set(json.loads(Path(index).read_text())["weight_map"].values()))
    return [hf_hub_download(REPO, f"transformer/{name}", revision=REVISION) for name in names]


def tensors(paths: list[str], keep) -> dict[str, torch.Tensor]:
    out = {}
    for path in paths:
        with safe_open(path, "pt") as handle:
            out.update({key: handle.get_tensor(key) for key in handle.keys() if keep(key)})
    return out


class StreamedTransformer:
    """`WanTransformer3DModel.forward` with the block weights loaded one at a time."""

    def __init__(self) -> None:
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

        config = WanTransformer3DModel.load_config(REPO, subfolder="transformer", revision=REVISION)
        self.layers = config["num_layers"]
        config["num_layers"] = 1
        self.model = WanTransformer3DModel.from_config(config).eval()
        self.paths = shards()
        missing, unexpected = self.model.load_state_dict(
            tensors(self.paths, lambda key: not key.startswith("blocks.")), strict=False
        )
        assert not unexpected and all(key.startswith("blocks.0.") for key in missing), (missing, unexpected)

    def load_block(self, index: int) -> None:
        prefix = f"blocks.{index}."
        state = {key[len(prefix):]: value for key, value in tensors(self.paths, lambda k: k.startswith(prefix)).items()}
        self.model.blocks[0].load_state_dict(state, strict=True)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, layers: int) -> torch.Tensor:
        model = self.model
        _, _, frames, height, width = hidden_states.shape
        p_t, p_h, p_w = model.config.patch_size
        rotary_emb = model.rope(hidden_states)
        hidden = model.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, timestep_proj, context, _ = model.condition_embedder(timestep, context, None, timestep_seq_len=None)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        for index in range(layers):
            self.load_block(index)
            hidden = model.blocks[0](hidden, context, timestep_proj, rotary_emb)
        shift, scale = (model.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden = (model.norm_out(hidden.float()) * (1 + scale) + shift).type_as(hidden)
        hidden = model.proj_out(hidden)
        hidden = hidden.reshape(1, frames // p_t, height // p_h, width // p_w, p_t, p_h, p_w, -1)
        hidden = hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def scheduler_cases() -> dict[str, object]:
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    base = UniPCMultistepScheduler.from_pretrained(REPO, subfolder="scheduler", revision=REVISION)
    n, cases = 1024, {}
    for name, strength in (("synthid", video_wan.WAN21_SYNTHID_STRENGTH), ("seedance", video_wan.WAN21_SEEDANCE_STRENGTH)):
        scheduler = UniPCMultistepScheduler.from_config(base.config, flow_shift=video_wan.WAN21_FLOW_SHIFT)
        scheduler.set_timesteps(video_wan.WAN21_INFERENCE_STEPS)
        steps = video_wan.WAN21_INFERENCE_STEPS
        init = min(int(steps * strength), steps)
        timesteps = scheduler.timesteps[(steps - init) * scheduler.order:]
        # `[batch, channels, ...]` like latents; the einsum in UniP/UniC needs both axes
        shape = (1, 16, n // 16)
        original = torch.from_numpy(synthetic_signal(n, 977, 80)).reshape(shape)
        noise = torch.randn(shape, generator=torch.Generator("cpu").manual_seed(81))
        sample = scheduler.add_noise(original, noise, timesteps[:1])
        trajectory = [strided(sample.numpy(), 4096)]
        for i, t in enumerate(timesteps):
            velocity = torch.from_numpy(synthetic_signal(n, 977, 90 + i)).reshape(shape) + 0.25 * sample
            sample = scheduler.step(velocity, t, sample, return_dict=False)[0]
            trajectory.append(strided(sample.numpy(), 4096))
        cases[name] = {
            "strength": strength,
            "timesteps": [int(t) for t in timesteps],
            "trajectory": trajectory,
        }
    full = UniPCMultistepScheduler.from_config(base.config, flow_shift=video_wan.WAN21_FLOW_SHIFT)
    full.set_timesteps(video_wan.WAN21_INFERENCE_STEPS)
    return {
        "sigmas": [float(s) for s in full.sigmas],
        "timesteps": [int(t) for t in full.timesteps],
        "samples": n,
        "cases": cases,
    }


def prompt_reader_case() -> dict[str, object]:
    tensor = torch.from_numpy(synthetic_signal(32, 977, 85)).reshape(1, 4, 8).bfloat16()
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return {
        "pt_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "shape": list(tensor.shape),
        "values": [float(v) for v in tensor.float().reshape(-1)],
        "default_cache_file": video_wan._wan_prompt_cache_path().name,
    }


def transformer_cases() -> dict[str, object]:
    oracle = StreamedTransformer()
    frames, height, width, seed = 3, 8, 12, 71
    latents = torch.from_numpy(synthetic_signal(16 * frames * height * width, 977, seed)).reshape(
        1, 16, frames, height, width
    )
    context = torch.from_numpy(synthetic_signal(512 * 4096, 977, 70)).reshape(1, 512, 4096)
    with torch.inference_mode():
        oracle.load_block(0)
        one_block = oracle.model(latents, torch.tensor([999]), context, return_dict=False)[0]
        replay = oracle.forward(latents, torch.tensor([999]), context, layers=1)
        assert torch.equal(one_block, replay), "streamed replay diverges from WanTransformer3DModel.forward"
        cases = {}
        for timestep in (999, 642):
            out = oracle.forward(latents, torch.tensor([timestep]), context, layers=oracle.layers)
            cases[f"t{timestep}"] = {"timestep": timestep, "output": strided(out[0].numpy(), 8192)}
    return {
        "latents": {"kind": "synthetic", "frames": frames, "height": height, "width": width, "rate": 977, "seed": seed},
        "context": {"kind": "synthetic", "tokens": 512, "dim": 4096, "rate": 977, "seed": 70},
        "cases": cases,
    }


def main() -> None:
    torch.set_num_threads(4)
    write(SUITE, "scheduler", {"repo": REPO, "revision": REVISION, **scheduler_cases()})
    write(SUITE, "prompt", prompt_reader_case())
    if "--skip-transformer" not in sys.argv:
        write(SUITE, "transformer", {"repo": REPO, "revision": REVISION, **transformer_cases()})


if __name__ == "__main__":
    main()

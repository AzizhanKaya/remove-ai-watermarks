"""Produce the Wan 2.1 default-prompt embedding cache that raiw-gpu's `PromptEmbedding` reads.

Run once from the repository root, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_wan_prompt_cache.py`.

This is the cold-miss branch of `video_wan.load_video_wan_runtime` on its own: the pinned
UMT5-XXL text encoder (about 23 GB to download, about 12 GB of RAM in bfloat16) encodes
`WAN21_DEFAULT_PROMPT` on CPU through `WanVideoToVideoPipeline.encode_prompt`, and
`_store_prompt_payload` writes it atomically to `video_wan._wan_prompt_cache_path()`. Any
Python `wan21-*` run on CUDA or MPS writes the same file (bfloat16 on CUDA, float16 on MPS).
The printed SHA-256 can be pinned with `PromptEmbedding::load_cached(Some(..))`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from remove_ai_watermarks import video_wan  # noqa: E402
from remove_ai_watermarks._internal.two_stage_pipeline import _store_prompt_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-encode even when the cache exists")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16",
                        help="text encoder dtype (CUDA runs use bfloat16, MPS runs float16)")
    args = parser.parse_args()
    path = video_wan._wan_prompt_cache_path()
    if path.is_file() and not args.force:
        print(f"exists: {path}")
    else:
        from diffusers.pipelines.wan.pipeline_wan_video2video import WanVideoToVideoPipeline

        torch.set_num_threads(4)
        pipe = WanVideoToVideoPipeline.from_pretrained(
            video_wan.WAN21_MODEL_ID,
            revision=video_wan.WAN21_MODEL_REVISION,
            transformer=None,
            vae=None,
            torch_dtype=getattr(torch, args.dtype),
        )
        with torch.inference_mode():
            embeds, _ = pipe.encode_prompt(
                prompt=video_wan.WAN21_DEFAULT_PROMPT,
                device=torch.device("cpu"),
                num_videos_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        _store_prompt_payload(path, embeds)
        print(f"wrote {path} {tuple(embeds.shape)} {embeds.dtype}")
    print(f"sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()

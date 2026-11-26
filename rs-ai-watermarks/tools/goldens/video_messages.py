"""Goldens for `video message identify|detect|remove` and the known-message accumulator.

Run from the repository root with the oracle environment (torch, av, diffusers):
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/video_messages.py`

`scores` replays seeded float32 logits through `MessageAccumulator` in uneven batches. `clip`
embeds a known 256-bit message into a short synthetic clip with the pinned VideoSeal JIT
embedder, encodes it with x264 (plus a sine audio track), and records identify, blind detect,
and codec/VAE removal on CPU torch. The encoded clip travels base64-encoded in the golden so
both sides read identical bytes; no media is written to the repository.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_common import FFMPEG, write_golden  # noqa: E402

import torch  # noqa: E402

from remove_ai_watermarks import video_messages  # noqa: E402
from remove_ai_watermarks._internal import video_message_models  # noqa: E402
from remove_ai_watermarks._internal.video_message_scores import MessageAccumulator, reference_bits  # noqa: E402

SUITE = "video_messages"
REFERENCE = "a5" * 8 + "3c" * 8 + "0f" * 8 + "e1" * 8
# The public embedder is weak at this size; a 3x residual survives x264 CRF 10 at 4:2:0.
AMPLIFY = 3.0
CLIP_ARGS = [
    "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=24:duration=1",
    "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=1",
]


def score_case(seed: int, bits: int, frames: int, batches: list[int], bias: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    expected = rng.integers(0, 2, bits)
    signs = np.where(expected == 1, 1.0, -1.0)
    logits = (rng.normal(0.0, 1.0, (frames, bits)) + bias * signs).astype(np.float32)
    accumulator = MessageAccumulator(tuple(int(b) for b in expected))
    start = 0
    for size in batches:
        accumulator.update(logits[start : start + size])
        start += size
    assert start == frames
    return {
        "expected": [int(b) for b in expected],
        "batches": batches,
        "frames": frames,
        "logits_f32le_b64": base64.b64encode(logits.astype("<f4").tobytes()).decode("ascii"),
        "reading": asdict(accumulator.finish()),
    }


def scores() -> None:
    cases = {
        "correlated_256": score_case(1, 256, 25, [5, 8, 3, 9], 0.6),
        "random_256": score_case(2, 256, 14, [8, 6], 0.0),
        "short_256": score_case(3, 256, 7, [4, 3], 0.3),
        "weak_1024": score_case(4, 1024, 13, [8, 5], 0.05),
    }
    hexes = {REFERENCE: [int(b) for b in reference_bits(REFERENCE)]}
    write_golden(SUITE, "scores", {"cases": cases, "reference_bits": hexes})


def decode_rgb(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])])


def embed_clip(directory: Path) -> Path:
    plain = directory / "plain.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-threads", "4", *CLIP_ARGS, "-c:v", "libx264", "-crf", "0",
         "-pix_fmt", "yuv444p", "-an", str(plain)],
        check=True,
    )
    frames = decode_rgb(plain)
    graph = video_message_models._asset("videoseal")
    model = torch.jit.load(str(graph), map_location="cpu")
    model.train(False)
    bits = torch.tensor([reference_bits(REFERENCE)], dtype=torch.float32)
    imgs = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    with torch.inference_mode():
        marked = model.embed(imgs, bits, is_video=True)
    marked = marked["imgs_w"] if isinstance(marked, dict) else marked
    marked = imgs + AMPLIFY * (marked - imgs)
    rgb = (marked.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    height, width = rgb.shape[1:3]
    out = directory / "marked.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-threads", "4", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s:v", f"{width}x{height}", "-r", "24", "-i", "pipe:0", *CLIP_ARGS[4:],
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "10",
         "-pix_fmt", "yuv420p", "-g", "12", "-c:a", "aac", "-b:a", "96k", "-shortest",
         "-fflags", "+bitexact", str(out)],
        input=rgb.tobytes(),
        check=True,
    )
    return out


def portable(value: Any, directory: Path) -> Any:
    """Replace temp-directory paths by bare names so both sides compare."""
    text = json.dumps(value, default=str).replace(str(directory) + "/", "")
    return json.loads(text)


def clip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        marked = embed_clip(directory)
        cases: dict[str, Any] = {}
        cases["identify"] = asdict(video_messages.identify_video_message(marked, reference_hex=REFERENCE))
        cases["identify_other"] = asdict(
            video_messages.identify_video_message(marked, reference_hex="0" * 64, scheme="pixelseal")
        )
        cases["detect"] = asdict(video_messages.detect_video_watermark_presence(marked))
        for name, kwargs in {
            "remove_codec": {"method": "codec", "force": True},
            "remove_vae": {"method": "vae", "noise_std": 0.15},
            "remove_auto": {"method": "auto"},
        }.items():
            out = directory / f"{name}.mp4"
            try:
                cases[name] = asdict(
                    video_messages.remove_video_message(
                        marked, out, reference_hex=REFERENCE, long_side=128, device="cpu", **kwargs
                    )
                )
            except video_messages.VideoMessageVerificationIncomplete as exc:
                cases[name] = {"error": str(exc), "before": asdict(exc.before), "after": asdict(exc.after)}
        write_golden(
            SUITE,
            "clip",
            {
                "reference": REFERENCE,
                "clip_b64": base64.b64encode(marked.read_bytes()).decode("ascii"),
                "cases": portable(cases, directory),
            },
        )


def main() -> None:
    torch.set_num_threads(4)
    scores()
    clip()


if __name__ == "__main__":
    main()

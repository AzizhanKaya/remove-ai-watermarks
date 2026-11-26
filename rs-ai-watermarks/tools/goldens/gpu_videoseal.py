"""Golden generator for the `gpu_videoseal` suite (raiw-gpu `VideoMessageDetector`).

Run from the repository root with the video-detect extra, e.g.
`OMP_NUM_THREADS=4 ~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_videoseal.py [schemes...]`.

For each scheme (default `videoseal pixelseal`) records `decode_presence_and_logits` on seeded
synthetic RGB frames (one size that needs the antialiased resize, one native 256x256) and two
fixture frames decoded in place by ffmpeg (`rgb24 / 255`).
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

from gpu_common import FIXTURES, synthetic_signal, write  # noqa: E402

from remove_ai_watermarks._internal import video_message_models  # noqa: E402

SUITE = "gpu_videoseal"
FIXTURE = "provenance/google-gemini-omni-flash-api.mp4"
FIXTURE_W, FIXTURE_H = 320, 180


def synthetic(frames: int, height: int, width: int, seed: int) -> tuple[dict[str, object], np.ndarray]:
    values = synthetic_signal(frames * height * width * 3, 977, seed).reshape(frames, height, width, 3)
    unit = np.clip(values + 0.5, 0.0, 1.0).astype(np.float32)
    record = {"kind": "synthetic", "frames": frames, "height": height, "width": width, "rate": 977, "seed": seed}
    return record, unit


def fixture_frames() -> tuple[dict[str, object], np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to decode fixture frames")
    before = ["-v", "error", "-ss", "1.0", "-i"]
    after = ["-frames:v", "2", "-vf", f"scale={FIXTURE_W}:{FIXTURE_H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run([ffmpeg, *before, str(FIXTURES / FIXTURE), *after], check=True, capture_output=True).stdout
    record = {"kind": "fixture_frames", "fixture": FIXTURE, "before": before, "after": after, "frames": 2,
              "height": FIXTURE_H, "width": FIXTURE_W, "decoded_sha256": hashlib.sha256(raw).hexdigest()}
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(2, FIXTURE_H, FIXTURE_W, 3).astype(np.float32) / 255.0
    return record, frames


def load_chunkyseal_slice(slice_path: Path, detector: torch.nn.Module) -> None:
    """`_load_chunkyseal_slice` that also accepts ZIP64 (24-byte) data descriptors.

    The pinned slice crosses the 4 GiB mark of the original archive at storage 313, where
    torch's writer switches to ZIP64 descriptors; the shipped reader only parses the 16-byte
    form and rejects the verified file. Parameters are copied in place to bound peak memory.
    """
    import struct
    import zlib

    with slice_path.open("rb") as stream, torch.no_grad():
        for index, (name, param) in enumerate(detector.state_dict().items()):
            entry = f"checkpoint/data/{video_message_models._CHUNKYSEAL_FIRST_STORAGE_KEY + index}"
            header = stream.read(30)
            flags = struct.unpack("<H", header[6:8])[0]
            name_len, extra_len = struct.unpack("<HH", header[26:30])
            if stream.read(name_len).decode() != entry:
                raise RuntimeError(f"storage order mismatch at {entry} ({name})")
            stream.read(extra_len)
            payload = stream.read(param.numel() * 4)
            if flags & 0x0008:
                descriptor = stream.read(16)
                if struct.unpack("<II", descriptor[8:16]) != (len(payload), len(payload)):
                    descriptor += stream.read(8)
                    if struct.unpack("<QQ", descriptor[8:24]) != (len(payload), len(payload)):
                        raise RuntimeError(f"bad ZIP64 descriptor for {entry}")
                if struct.unpack("<I", descriptor[4:8])[0] != zlib.crc32(payload) & 0xFFFFFFFF:
                    raise RuntimeError(f"CRC mismatch for {entry}")
            param.copy_(torch.from_numpy(np.frombuffer(payload, dtype="<f4").reshape(param.shape)))


def main() -> None:
    torch.set_num_threads(4)
    schemes = sys.argv[1:] or ["videoseal", "pixelseal"]
    video_message_models._load_chunkyseal_slice = load_chunkyseal_slice
    inputs = {
        "synthetic_resized": synthetic(2, 72, 120, 61),
        "synthetic_native": synthetic(1, 256, 256, 63),
        "fixture_frames": fixture_frames(),
    }
    for scheme in schemes:
        model = video_message_models.load_detector(scheme)
        cases = {}
        for name, (record, frames) in inputs.items():
            presence, message = video_message_models.decode_presence_and_logits(model, frames)
            cases[name] = {"input": record, "presence": [float(v) for v in presence],
                           "message": [[float(v) for v in row] for row in message]}
        write(SUITE, scheme, {"scheme": scheme, "source_commit": video_message_models.SOURCE_COMMIT, "cases": cases})
        video_message_models.clear_model_cache()


if __name__ == "__main__":
    main()

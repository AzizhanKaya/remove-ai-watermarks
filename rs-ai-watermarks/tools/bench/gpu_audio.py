"""Torch CPU timings for the neural audio paths ported to raiw-gpu.

Run from the repository root with every audio extra installed, e.g. the oracle venv:
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/bench/gpu_audio.py`.
Compare with `cargo test -p raiw-gpu --release --test timing -- --ignored --nocapture`,
which runs the same workloads (seconds of audio per model) on wgpu and the CPU backend.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from remove_ai_watermarks._internal import (  # noqa: E402
    audio_codec,
    audio_dac,
    audio_snac,
    audioseal,
    perth,
    silentcipher,
    wavmark,
)
from remove_ai_watermarks._internal._silentcipher_network import magnitude  # noqa: E402

THREADS = 4

WORKLOADS = {
    "audioseal": (16_000, 10.0),
    "dac": (44_100, 1.0),
    "snac": (44_100, 1.0),
    "encodec_6kbps": (24_000, 1.0),
    "wavmark": (16_000, 2.0),
    "perth": (16_000, 10.0),
    "silentcipher_phase": (44_100, 3.0),
}
REPEATS = 2


def timed(run: object) -> float:
    """Best of `REPEATS` runs after one warm-up, as the Rust timing test."""
    run()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    return min(samples)


def signal(rate: int, seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(t.shape[0])).astype(np.float32)


def main() -> None:
    audio_dac.is_available = lambda: True
    audio_snac.is_available = lambda: True
    torch.set_num_threads(THREADS)
    audio_codec.is_available = lambda: True
    results: dict[str, object] = {"torch": torch.__version__, "threads": torch.get_num_threads()}

    rate, seconds = WORKLOADS["audioseal"]
    detector = audioseal.load_pinned_detector()
    audio = signal(rate, seconds)
    results["audioseal"] = {"seconds_audio": seconds, "torch_cpu_s": timed(lambda: audioseal.read(detector, audio))}

    rate, seconds = WORKLOADS["dac"]
    dac = audio_dac.load_pinned_model()
    audio = signal(rate, seconds)
    audio = audio[: audio.shape[0] // 512 * 512]
    x = torch.from_numpy(audio).view(1, 1, -1)

    with torch.inference_mode():
        codes = dac.encode(x, n_quantizers=3).audio_codes

    def run_dac_encode() -> None:
        with torch.inference_mode():
            dac.encode(x, n_quantizers=3)

    def run_dac_decode() -> None:
        with torch.inference_mode():
            dac.decode(audio_codes=codes)

    results["dac_encode"] = {"seconds_audio": seconds, "torch_cpu_s": timed(run_dac_encode)}
    results["dac_decode"] = {"seconds_audio": seconds, "torch_cpu_s": timed(run_dac_decode)}
    del dac

    rate, seconds = WORKLOADS["snac"]
    snac = audio_snac.load_pinned_model()
    x = torch.from_numpy(signal(rate, seconds)).view(1, 1, -1)

    def run_snac() -> None:
        with torch.inference_mode():
            snac.reconstruct(x, n_quantizers=3)

    results["snac"] = {"seconds_audio": seconds, "torch_cpu_s": timed(run_snac)}
    del snac

    rate, seconds = WORKLOADS["encodec_6kbps"]
    encodec = audio_codec.load_pinned_model()
    x = torch.from_numpy(signal(rate, seconds)).view(1, 1, -1)

    def run_encodec() -> None:
        with torch.inference_mode():
            encodec(x, bandwidth=6.0)

    results["encodec_6kbps"] = {"seconds_audio": seconds, "torch_cpu_s": timed(run_encodec)}
    del encodec

    rate, seconds = WORKLOADS["wavmark"]
    model = wavmark.load_model()
    audio = signal(rate, seconds)
    results["wavmark"] = {"seconds_audio": seconds, "torch_cpu_s": timed(lambda: wavmark.read(model, audio))}

    rate, seconds = WORKLOADS["perth"]
    model = perth.load_model()
    audio = signal(rate, seconds)
    results["perth"] = {"seconds_audio": seconds, "torch_cpu_s": timed(lambda: perth.read(model, audio))}

    rate, seconds = WORKLOADS["silentcipher_phase"]
    decoder = silentcipher.load_model("cpu")
    audio = signal(rate, seconds)
    carrier = torch.from_numpy(audio * np.sqrt(0.002837200844477648 / np.mean(audio**2))).unsqueeze(0)

    def run_phase() -> None:
        with torch.inference_mode():
            decoder.decoder(magnitude(carrier, decoder.window)[:, None])

    results["silentcipher_phase"] = {"seconds_audio": seconds, "torch_cpu_s": timed(run_phase)}
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

"""Timing reference for the raiw-audio hot paths (compare with `cargo test --release -p raiw-audio
--test bench -- --ignored --nocapture`).

Run from the repository root with the oracle environment:
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/bench/audio.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "goldens"))

from audio_common import encode_raw, synth  # noqa: E402
from scipy import signal  # noqa: E402

from remove_ai_watermarks._internal import audio_io, audio_spectral  # noqa: E402
from remove_ai_watermarks._internal import audio_resampling as src  # noqa: E402
from remove_ai_watermarks._internal.audio_dac import blend_hf_crossover  # noqa: E402
from remove_ai_watermarks._internal.audio_quality import compute_perceptual_metrics, measure_fidelity  # noqa: E402


def spec(rate: int, seconds: int, seed: int) -> dict:
    return {
        "rate": rate,
        "frames": rate * seconds,
        "channels": [
            [{"kind": "noise", "seed": seed, "amp": 0.3}, {"kind": "tone", "freq": 1250.0, "amp": 0.1}],
            [{"kind": "noise", "seed": seed + 1, "amp": 0.3}, {"kind": "tone", "freq": 440.0, "amp": 0.1}],
        ],
    }


def best(func, repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    results = {}
    x48 = synth(spec(48000, 60, 1))
    x44 = synth(spec(44100, 60, 3))
    x16 = synth(spec(16000, 60, 5))
    results["resample_48k_to_16k_60s_stereo"] = best(lambda: signal.resample_poly(x48, 1, 3, axis=0, padtype="constant"))
    results["resample_44k1_to_16k_60s_stereo"] = best(lambda: signal.resample_poly(x44, 160, 441, axis=0, padtype="constant"))
    results["resample_16k_to_44k1_60s_stereo"] = best(lambda: signal.resample_poly(x16, 441, 160, axis=0, padtype="constant"))
    clock = src.RationalClock(len(x48), 48000)
    results["clock_forward_inverse_48k_60s_stereo"] = best(lambda: src.inverse(src.forward(x48, clock), clock))
    chunks = [x48[s : s + 48000 * 8] for s in range(0, len(x48), 48000 * 8)]
    results["spectral_filter_48k_60s_stereo"] = best(lambda: list(audio_spectral.filter_audio_chunks(chunks, 48000)))
    x30 = x44[: 44100 * 30]
    results["hf_crossover_44k1_30s_stereo"] = best(lambda: blend_hf_crossover(x30 * 0.9, x30, 44100, 14000.0))
    with tempfile.TemporaryDirectory() as tmp:
        source, output = Path(tmp) / "a.wav", Path(tmp) / "b.wav"
        encode_raw(x48, 48000, source, ["-c:a", "pcm_f32le"])
        encode_raw(x48 * 0.9, 48000, output, ["-c:a", "pcm_f32le"])
        info = audio_io.probe_audio(source)
        results["decode_48k_60s_stereo"] = best(lambda: np.concatenate(list(audio_io.read_audio(source, info))))
        results["measure_fidelity_48k_60s_stereo"] = best(lambda: measure_fidelity(source, info, output))
        results["perceptual_metrics_48k_60s_stereo"] = best(lambda: compute_perceptual_metrics(source, info, output), 1)
    print(json.dumps({k: round(v, 4) for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()

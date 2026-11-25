"""Golden generator for the `gpu_perth` suite (raiw-gpu Perth implicit decoder).

Run from the repository root with the Perth extra (`resemble-perth==1.0.1`, librosa), e.g.
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_perth.py`.

Cases: seeded synthetic audio, a Perth-embedded signal stored as int16 PCM, and a
tracked fixture's audio decoded in place by ffmpeg; each records the unclipped
decoder output, the clipped confidence, and the 32 kHz resampled input statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gpu_common import fixture_input, pcm16_b64, strided, synthetic_signal, write  # noqa: E402

from remove_ai_watermarks._internal import perth  # noqa: E402

SUITE = "gpu_perth"
SR = perth.SAMPLE_RATE


def case(model: object, samples: np.ndarray) -> dict[str, object]:
    import librosa

    native = librosa.resample(samples, orig_sr=SR, target_sr=model.hp.sample_rate, res_type="polyphase")
    with torch.inference_mode():
        magnitudes, _ = model.ap.signal_to_magphase(torch.from_numpy(native.copy()).float())
        raw = float(model.decoder(magnitudes[None])[0])
    reading = perth.read(model, samples)
    return {
        "resampled": strided(native, 64),
        "frames": int(magnitudes.shape[1]),
        "raw": raw,
        "confidence": reading.confidence,
        "detected": reading.detected,
    }


def main() -> None:
    model = perth.load_model()
    cases = {}
    samples = synthetic_signal(2 * SR, SR, 41)
    cases["synthetic_2s"] = {"input": {"kind": "synthetic", "samples": 2 * SR, "sample_rate": SR, "seed": 41}}
    cases["synthetic_2s"].update(case(model, samples))

    carrier = synthetic_signal(20_000, SR, 43)
    marked = np.asarray(perth.embed(model, carrier), dtype=np.float32)
    payload, quantized = pcm16_b64(marked)
    cases["marked"] = {"input": {"kind": "pcm16", "sample_rate": SR, "pcm16_b64": payload}}
    cases["marked"].update(case(model, quantized))

    record, samples = fixture_input("provenance/google-gemini-omni-flash-api.mp4", SR, 0.5, 2.0)
    cases["fixture_gemini_omni"] = {"input": record}
    cases["fixture_gemini_omni"].update(case(model, samples))
    write(SUITE, "decoder", {"package": f"resemble-perth=={perth.PACKAGE_VERSION}", "cases": cases})


if __name__ == "__main__":
    main()

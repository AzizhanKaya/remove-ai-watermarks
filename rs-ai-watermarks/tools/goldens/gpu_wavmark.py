"""Golden generator for the `gpu_wavmark` suite (raiw-gpu WavMark decoder).

Run from the repository root with the WavMark extra installed (`wavmark==0.0.3`), e.g.
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_wavmark.py`.

Cases: seeded synthetic audio, a WavMark-embedded signal stored as int16 PCM, and a
tracked fixture's audio decoded in place by ffmpeg. Each records raw `Model.decode`
scores for a few windows plus the project receiver's reading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gpu_common import fixture_input, pcm16_b64, synthetic_signal, write  # noqa: E402

from remove_ai_watermarks._internal import wavmark  # noqa: E402

SUITE = "gpu_wavmark"
SR = wavmark.SAMPLE_RATE
STARTS = (0, 800, 3200)


def case(model: object, samples: np.ndarray) -> dict[str, object]:
    starts = [s for s in STARTS if s + SR <= samples.shape[0]]
    batch = np.stack([samples[s : s + SR] for s in starts]).astype(np.float32)
    with torch.inference_mode():
        raw = model.decode(torch.from_numpy(batch)).numpy()
    reading = wavmark.read(model, samples)
    return {
        "raw": {str(s): [float(v) for v in row] for s, row in zip(starts, raw, strict=True)},
        "reading": {
            "payload": list(reading.payload) if reading.payload is not None else None,
            "start_samples": list(reading.start_samples),
        },
    }


def main() -> None:
    model = wavmark.load_model()
    cases = {}
    samples = synthetic_signal(2 * SR, SR, 31)
    cases["synthetic_2s"] = {"input": {"kind": "synthetic", "samples": 2 * SR, "sample_rate": SR, "seed": 31}}
    cases["synthetic_2s"].update(case(model, samples))

    carrier = synthetic_signal(19_200, SR, 33)
    marked = np.asarray(wavmark.embed(model, carrier), dtype=np.float32)
    payload, quantized = pcm16_b64(marked)
    cases["marked"] = {"input": {"kind": "pcm16", "sample_rate": SR, "message": list(wavmark.MESSAGE), "pcm16_b64": payload}}
    cases["marked"].update(case(model, quantized))

    record, samples = fixture_input("provenance/google-gemini-omni-flash-api.mp4", SR, 0.5, 2.0)
    cases["fixture_gemini_omni"] = {"input": record}
    cases["fixture_gemini_omni"].update(case(model, samples))
    write(SUITE, "decoder", {"repo": wavmark.MODEL_REPOSITORY, "revision": wavmark.MODEL_REVISION, "cases": cases})


if __name__ == "__main__":
    main()

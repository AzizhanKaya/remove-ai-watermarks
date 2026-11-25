"""Golden generator for the `gpu_silentcipher` suite (raiw-gpu SilentCipher decoder).

Run from the repository root with an environment that has the SilentCipher extra
(scipy for `stats.mode`), e.g. the oracle venv:
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/gpu_silentcipher.py`.

Records raw decoder scores for a few phase shifts of power-normalized input, the
`_decode_sequence` verdicts on them, and the full receiver reading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gpu_common import fixture_input, synthetic_signal, write  # noqa: E402

from remove_ai_watermarks._internal import silentcipher  # noqa: E402
from remove_ai_watermarks._internal._silentcipher_network import magnitude  # noqa: E402

SUITE = "gpu_silentcipher"
SR = silentcipher.SAMPLE_RATE
SHIFTS = (0, 10, 1000, 2040)


def normalized(samples: np.ndarray) -> np.ndarray:
    power = np.mean(samples**2)
    return samples * np.sqrt(0.002837200844477648 / power)


def scores(model: silentcipher.DecoderModel, values: np.ndarray, shift: int) -> np.ndarray:
    signal = torch.from_numpy(values.copy()).unsqueeze(0).to(dtype=torch.float32)
    with torch.inference_mode():
        carrier = magnitude(signal[:, shift:], model.window)
        return model.decoder(carrier[:, None])[0, 0].numpy()


def case(model: silentcipher.DecoderModel, samples: np.ndarray, full_read: bool) -> dict[str, object]:
    values = normalized(samples)
    shifts = {}
    for shift in SHIFTS:
        raw = scores(model, values, shift)
        payload, consistency = silentcipher._decode_sequence(raw)
        shifts[str(shift)] = {
            "frames": int(raw.shape[1]),
            "scores": [[float(v) for v in row] for row in raw],
            "payload": list(payload) if payload is not None else None,
            "consistency": consistency,
        }
    record: dict[str, object] = {"shifts": shifts}
    if full_read:
        reading = silentcipher.read(model, samples)
        record["reading"] = {
            "payload": list(reading.payload) if reading.payload is not None else None,
            "symbol_consistency": reading.symbol_consistency,
            "phase_offset": reading.phase_offset,
            "phases_checked": reading.phases_checked,
        }
    return record


def main() -> None:
    model = silentcipher.load_model("cpu")
    cases = {}
    samples = synthetic_signal(silentcipher.MIN_READ_SAMPLES, SR, 21)
    cases["synthetic_3s"] = {
        "input": {"kind": "synthetic", "samples": samples.shape[0], "sample_rate": SR, "seed": 21},
        **case(model, samples, full_read=True),
    }
    record, samples = fixture_input("provenance/youtube-reencode-gemini-omni-flash.mp4", SR, 1.0, 3.2)
    cases["fixture_gemini_youtube"] = {"input": record, **case(model, samples, full_read=False)}
    write(SUITE, "decoder", {"repo": silentcipher.MODEL_REPOSITORY, "revision": silentcipher.MODEL_REVISION, "cases": cases})


if __name__ == "__main__":
    main()

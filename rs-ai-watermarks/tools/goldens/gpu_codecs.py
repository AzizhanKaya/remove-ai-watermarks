"""Golden generator for the `gpu_codecs` suite (raiw-gpu DAC, SNAC and EnCodec codecs).

Run from the repository root: `.venv/bin/python rs-ai-watermarks/tools/goldens/gpu_codecs.py`.
The pinned checkpoints must already sit in the Hugging Face cache (the Python
loaders use `local_files_only=True`).

Each case feeds one mono stem on the 44.1 kHz model grid straight into the network
(`encode(n_quantizers=3)` + `decode(audio_codes)` for DAC, `reconstruct` for SNAC)
and records the codes plus a strided excerpt and statistics of the output. A
`randn` document pins torch's seeded CPU normal stream used by SNAC's noise blocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from gpu_common import fixture_input, strided, synthetic_signal, write  # noqa: E402

from remove_ai_watermarks._internal import audio_codec, audio_dac, audio_snac  # noqa: E402

SUITE = "gpu_codecs"
SR = 44_100
FIXTURE = "provenance/higgsfield-kling-3-0-turbo.mp4"
DAC_HOP = 512
ENCODEC_BANDWIDTH = 6.0


def dac_case(model: object, samples: np.ndarray) -> dict[str, object]:
    x = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).view(1, 1, -1)
    with torch.inference_mode():
        encoded = model.encode(x, n_quantizers=audio_dac.FIXED_QUANTIZERS)
        decoded = model.decode(audio_codes=encoded.audio_codes).audio_values.reshape(-1).numpy()
    return {"codes": encoded.audio_codes[0].tolist(), "output": strided(decoded, 256)}


def snac_codes(model: object, samples: np.ndarray, n_quantizers: int) -> list[list[int]]:
    length = samples.shape[0]
    pad = (audio_snac.HOP_LENGTH - length % audio_snac.HOP_LENGTH) % audio_snac.HOP_LENGTH
    x = torch.nn.functional.pad(torch.from_numpy(samples).view(1, 1, -1), (0, pad))
    codes = []
    with torch.inference_mode():
        residual = model.encoder(x)
        for quantizer in model.quantizer.quantizers[:n_quantizers]:
            code = quantizer.encode(residual)
            residual = residual - quantizer.decode_code(code)
            codes.append(code[0].tolist())
    return codes


def snac_case(model: object, samples: np.ndarray) -> dict[str, object]:
    x = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).view(1, 1, -1)
    with torch.inference_mode():
        decoded = model.reconstruct(x, n_quantizers=audio_snac.FIXED_QUANTIZERS).reshape(-1).numpy()
    return {"codes": snac_codes(model, samples, audio_snac.FIXED_QUANTIZERS), "output": strided(decoded, 256)}


def encodec_case(model: object, samples: np.ndarray) -> dict[str, object]:
    x = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).view(1, 1, -1)
    with torch.inference_mode():
        codes = model.encode(x, bandwidth=ENCODEC_BANDWIDTH).audio_codes[0, 0]
        decoded = model(x, bandwidth=ENCODEC_BANDWIDTH).audio_values.reshape(-1).numpy()
    return {"bandwidth": ENCODEC_BANDWIDTH, "codes": codes.tolist(), "output": strided(decoded, 256)}


def synthetic(samples: int, seed: int) -> tuple[dict[str, object], np.ndarray]:
    return synthetic_at(samples, SR, seed)


def synthetic_at(samples: int, rate: int, seed: int) -> tuple[dict[str, object], np.ndarray]:
    record = {"kind": "synthetic", "samples": samples, "sample_rate": rate, "seed": seed}
    return record, synthetic_signal(samples, rate, seed)


def fixture(multiple: int) -> tuple[dict[str, object], np.ndarray]:
    record, samples = fixture_input(FIXTURE, SR, 0.5, 0.5)
    keep = samples.shape[0] // multiple * multiple
    record["truncate_to"] = keep
    return record, samples[:keep]


def fixture_at(rate: int) -> tuple[dict[str, object], np.ndarray]:
    return fixture_input(FIXTURE, rate, 0.5, 0.5)


def main() -> None:
    # scipy only serves the crossover filters, not the networks recorded here
    audio_dac.is_available = lambda: True
    audio_snac.is_available = lambda: True
    audio_codec.is_available = lambda: True
    dac = audio_dac.load_pinned_model()
    cases = {}
    for name, (record, samples) in {
        "synthetic": synthetic(43 * DAC_HOP, 5),
        "fixture_kling": fixture(DAC_HOP),
    }.items():
        cases[name] = {"input": record, **dac_case(dac, samples)}
    write(SUITE, "dac", {"repo": audio_dac.MODEL_REPOSITORY, "revision": audio_dac.MODEL_REVISION, "cases": cases})

    snac = audio_snac.load_pinned_model()
    cases = {}
    for name, (record, samples) in {
        "synthetic": synthetic(2 * audio_snac.HOP_LENGTH, 9),
        "synthetic_unaligned": synthetic(5_000, 13),
        "fixture_kling": fixture(1),
    }.items():
        cases[name] = {"input": record, **snac_case(snac, samples)}
    write(SUITE, "snac", {"repo": audio_snac.MODEL_REPOSITORY, "revision": audio_snac.MODEL_REVISION, "cases": cases})

    encodec = audio_codec.load_pinned_model()
    cases = {}
    for name, (record, samples) in {
        "synthetic": synthetic_at(12_000, 24_000, 17),
        "fixture_kling": fixture_at(24_000),
    }.items():
        cases[name] = {"input": record, **encodec_case(encodec, samples)}
    write(SUITE, "encodec", {"repo": audio_codec.MODEL_REPOSITORY, "revision": audio_codec.MODEL_REVISION, "cases": cases})

    draws = {}
    for n in (16, 37, 64, 200):
        gen = torch.Generator(device="cpu").manual_seed(0)
        draws[str(n)] = [float(v) for v in torch.randn((1, 1, n), generator=gen, dtype=torch.float32).view(-1)]
    write(SUITE, "randn", {"seed": 0, "draws": draws})


if __name__ == "__main__":
    main()

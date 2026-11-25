"""Golden generator for the `audio` suite: receiver logic of the public decoders.

The real `audioseal.read`, `wavmark.read`, `wavmark.read_openvoice_default`, `perth.read` and
`silentcipher._decode_sequence` run on raw network outputs drawn from the SplitMix64 stream
(`audio_common.splitmix_uniform`), which the Rust tests regenerate bit for bit.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import torch
from audio_common import splitmix_uniform, write

from remove_ai_watermarks._internal import audioseal, perth, silentcipher, wavmark

PATTERN = importlib.import_module("wavmark").wm_add_util.fix_pattern[:16]


def uniform32(seed, count, lo=0.0, hi=1.0):
    return (lo + (hi - lo) * splitmix_uniform(seed, count)).astype(np.float32)


def reading_dict(reading):
    from dataclasses import asdict

    return asdict(reading)


def error_of(func):
    try:
        return {"ok": reading_dict(func())}
    except (ValueError, RuntimeError) as exc:
        return {"error": {"type": type(exc).__name__, "message": str(exc)}}


def audioseal_cases():
    cases = []
    for seed, count, scale in [(1, 1, 1.0), (2, 7, 1.0), (3, 100, 1.0), (4, 16000, 1.0), (5, 40000, 1.0), (6, 9000, 0.6)]:
        probs = uniform32(seed, count, 0.0, scale)
        message = uniform32(seed + 100, 16, -1.0, 2.0)

        def detector(wav, sample_rate, probs=probs, message=message):
            result = torch.zeros((1, 2, len(probs)), dtype=torch.float32)
            result[0, 1, :] = torch.from_numpy(probs)
            return result, torch.from_numpy(message).view(1, -1)

        cases.append(
            {
                "probs": {"seed": seed, "count": count, "lo": 0.0, "hi": scale},
                "message": {"seed": seed + 100, "count": 16, "lo": -1.0, "hi": 2.0},
                "result": error_of(lambda: audioseal.read(detector, np.zeros(10, np.float32))),
            }
        )
    return cases


class RawWavMark:
    """Rows from the window's first sample (inputs are `arange`): sync on multiples of `every`."""

    def __init__(self, every, flips=0):
        self.every, self.flips = every, flips

    def parameters(self):
        yield torch.zeros(1)

    def decode(self, batch):
        rows = []
        for window in batch.numpy():
            start = int(window[0])
            raw = uniform32(start + 7, 32)
            if self.every and start % self.every == 0:
                raw[:16] = np.asarray(PATTERN, dtype=np.float32)
            rows.append(raw)
        return torch.from_numpy(np.stack(rows))


def wavmark_cases():
    cases = []
    for length, every in [(16000, 0), (16000, 1), (20001, 1600), (40000, 2400), (64123, 4000)]:
        samples = np.arange(length, dtype=np.float32)
        cases.append({"length": length, "every": every, "result": error_of(lambda: wavmark.read(RawWavMark(every), samples))})
    cases.append({"length": 15999, "every": 1, "result": error_of(lambda: wavmark.read(RawWavMark(1), np.arange(15999, dtype=np.float32)))})
    return cases


class SignatureModel:
    def __init__(self, flips):
        self.flips = flips

    def parameters(self):
        yield torch.zeros(1)

    def decode(self, batch):
        reference = np.unpackbits(np.frombuffer(b"default ", dtype=np.uint8)).astype(np.float32)
        rows = []
        for index, window in enumerate(batch.numpy()):
            bits = reference[index * 32 : (index + 1) * 32].copy()
            noise = uniform32(int(window[0]) + 11, 32, 0.0, 0.4)
            raw = bits * 0.6 + noise
            for k in range(self.flips[index]):
                raw[(k * 7) % 32] = 1.0 - bits[(k * 7) % 32]
            rows.append(raw)
        return torch.from_numpy(np.stack(rows))


def openvoice_cases():
    cases = []
    for length, flips in [(16000, (0, 0)), (16000, (5, 0)), (48000, (0, 0)), (48000, (3, 9)), (48000, (12, 14)), (40000, (2, 0))]:
        samples = np.arange(length, dtype=np.float32)

        def run(samples=samples, flips=flips):
            whole, fragments = wavmark.read_openvoice_default(SignatureModel(flips), samples)
            return SimpleNamespace(whole=whole, fragments=fragments)

        try:
            result = run()
            record = {"whole": reading_dict(result.whole), "fragments": [reading_dict(f) for f in result.fragments]}
        except (ValueError, RuntimeError) as exc:
            record = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        cases.append({"length": length, "flips": list(flips), "result": record})
    return cases


def perth_cases():
    cases = []
    for raw in ([0.3], [1.7], [-0.2], [float("nan")], [0.1, 0.2], [0.5]):
        model = SimpleNamespace(
            hp=SimpleNamespace(sample_rate=16000),
            device="cpu",
            ap=SimpleNamespace(signal_to_magphase=lambda s: (s, None)),
            decoder=lambda m, raw=raw: torch.tensor([raw], dtype=torch.float32),
        )
        cases.append({"raw": [repr(v) for v in raw], "result": error_of(lambda: perth.read(model, np.zeros(16000, np.float32)))})
    cases.append({"raw": ["0.3"], "short": True, "result": error_of(lambda: perth.read(None, np.zeros(15999, np.float32)))})
    return cases


def scores(kind, seed, frames):
    noise = uniform32(seed, 5 * frames, 0.0, 0.5).reshape(5, frames)
    if kind == "valid":
        symbols = [0, *[1 + (i * 3) % 4 for i in range(20)]]
    elif kind == "two_zeros":
        symbols = [0, 0, *[1 + i % 4 for i in range(19)]]
    elif kind == "no_zero":
        symbols = [1 + i % 4 for i in range(21)]
    else:
        symbols = [int(v * 5) % 5 for v in splitmix_uniform(seed + 1, 21)]
    out = noise.copy()
    for t in range(frames):
        out[symbols[t % 21], t] += 1.0
    flips = int(seed % 5)
    for k in range(flips):
        t = (k * 13 + 5) % frames
        out[(symbols[t % 21] + 1) % 5, t] += 2.0
    return out


def silentcipher_cases():
    cases = []
    for kind, seed, frames in [
        ("valid", 10, 63),
        ("valid", 11, 70),
        ("valid", 13, 105),
        ("two_zeros", 14, 63),
        ("no_zero", 15, 84),
        ("random", 16, 63),
        ("random", 17, 200),
        ("valid", 18, 62),
    ]:
        raw = scores(kind, seed, frames)
        try:
            payload, consistency = silentcipher._decode_sequence(raw)
            result = {"payload": None if payload is None else list(payload), "consistency": consistency}
        except silentcipher.ModelCheckError as exc:
            result = {"error": str(exc)}
        cases.append({"kind": kind, "seed": seed, "frames": frames, "scores_sum": float(raw.astype(np.float64).sum()), "result": result})
    return cases


def main() -> None:
    write(
        "decoders",
        {
            "pattern": list(PATTERN),
            "audioseal": audioseal_cases(),
            "wavmark": wavmark_cases(),
            "openvoice": openvoice_cases(),
            "perth": perth_cases(),
            "silentcipher": silentcipher_cases(),
        },
    )


if __name__ == "__main__":
    main()

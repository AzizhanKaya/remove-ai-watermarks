"""Golden generator for the `audio` suite: SciPy-equivalent DSP kernels.

Records `resample_poly`, Butterworth SOS designs, `sosfiltfilt`, the spectral removal profile,
bounded-context chunking and the rational resynthesis clock on seeded synthetic signals.
"""

from __future__ import annotations

import numpy as np
from audio_common import digest, sha, synth, write
from scipy import signal

from remove_ai_watermarks._internal import audio_resampling as src
from remove_ai_watermarks._internal import audio_spectral
from remove_ai_watermarks._internal.audio_chunks import transform_chunks


def tones(rate, frames, *freqs, amp=0.2):
    return [{"kind": "tone", "freq": f, "amp": amp, "phase": 0.3 * i} for i, f in enumerate(freqs)]


def noise(seed, amp=0.3):
    return [{"kind": "noise", "seed": seed, "amp": amp}]


SIGNALS = {
    "tones48k": {"rate": 48000, "frames": 48000, "channels": [tones(48000, 0, 500, 1250, 7000)]},
    "noise44k": {"rate": 44100, "frames": 44100 + 37, "channels": [noise(11)]},
    "chirp16k": {
        "rate": 16000,
        "frames": 16000 * 2 + 5,
        "channels": [[{"kind": "chirp", "f0": 50.0, "f1": 7900.0, "amp": 0.5}]],
    },
    "speech24k": {
        "rate": 24000,
        "frames": 24000 * 2,
        "channels": [
            [
                {"kind": "burst", "freq": 220.0, "amp": 0.4, "period": 4800, "on": 2600},
                {"kind": "noise", "seed": 5, "amp": 0.02},
            ]
        ],
    },
    "clip22k": {
        "rate": 22050,
        "frames": 22050 + 3,
        "channels": [[{"kind": "tone", "freq": 440.0, "amp": 1.3}]],
    },
    "silence32k": {"rate": 32000, "frames": 3200, "channels": [[{"kind": "silence"}]]},
    "impulse_like96k": {"rate": 96000, "frames": 9601, "channels": [noise(99, 0.9)]},
    "tiny": {"rate": 48000, "frames": 13, "channels": [noise(3)]},
    "single": {"rate": 48000, "frames": 1, "channels": [noise(4)]},
}

RESAMPLE_CASES = [
    ("tones48k", 1, 3),
    ("tones48k", 147, 160),
    ("noise44k", 160, 441),
    ("noise44k", 80, 147),
    ("noise44k", 160, 147),
    ("chirp16k", 441, 160),
    ("chirp16k", 3, 2),
    ("speech24k", 2, 3),
    ("speech24k", 147, 80),
    ("clip22k", 320, 441),
    ("silence32k", 1, 2),
    ("impulse_like96k", 1, 6),
    ("impulse_like96k", 147, 320),
    ("tiny", 147, 160),
    ("tiny", 160, 147),
    ("single", 147, 160),
    ("single", 1, 3),
]

DESIGNS = [
    ("bandstop", 4, [1000.0, 1500.0], rate)
    for rate in (3001, 8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000)
] + [
    ("lowpass", 4, 120.0, rate)
    for rate in (8000, 16000, 44100, 48000)
] + [
    (btype, 8, 14000.0, rate)
    for btype in ("low", "high")
    for rate in (44100, 48000, 96000)
] + [("low", 8, 9000.0, 22050), ("high", 8, 8000.0, 24000)]


def design(btype, order, cutoff, rate):
    return np.asarray(signal.butter(order, cutoff, btype=btype, fs=rate, output="sos"), dtype=np.float64)


def resample_golden():
    cases = []
    for name, up, down in RESAMPLE_CASES:
        x = synth(SIGNALS[name])[:, 0].copy()
        y = signal.resample_poly(x, up, down, axis=0, padtype="constant")
        cases.append({"signal": name, "up": up, "down": down, "input_sha256": sha(x), "output": digest(y)})
    write("resample", {"signals": SIGNALS, "cases": cases})


def butter_golden():
    cases = []
    for btype, order, cutoff, rate in DESIGNS:
        sos = design(btype, order, cutoff, rate)
        cases.append(
            {
                "btype": btype,
                "order": order,
                "cutoff": cutoff,
                "rate": rate,
                "sos": [list(map(float, row)) for row in sos],
                "zi": [list(map(float, row)) for row in signal.sosfilt_zi(sos)],
            }
        )
    write("butter", {"cases": cases})


def sosfiltfilt_golden():
    cases = []
    for name, (btype, order, cutoff, rate), padlen, wide in [
        ("tones48k", ("bandstop", 4, [1000.0, 1500.0], 48000), None, False),
        ("tones48k", ("bandstop", 4, [1000.0, 1500.0], 48000), 27, False),
        ("noise44k", ("low", 8, 14000.0, 44100), None, False),
        ("noise44k", ("high", 8, 14000.0, 44100), None, False),
        ("speech24k", ("lowpass", 4, 120.0, 24000), 27, True),
        ("chirp16k", ("bandstop", 4, [1000.0, 1500.0], 16000), 5, False),
        ("tiny", ("bandstop", 4, [1000.0, 1500.0], 48000), 12, False),
    ]:
        x = synth(SIGNALS[name])[:, 0].copy()
        if wide:
            x = x.astype(np.float64)
        sos = design(btype, order, cutoff, rate)
        y = signal.sosfiltfilt(sos, x, axis=0, padlen=padlen)
        cases.append(
            {
                "signal": name,
                "design": [btype, order, cutoff, rate],
                "padlen": padlen,
                "float64_input": wide,
                "output": digest(y),
                "output_f32": digest(np.asarray(y, dtype=np.float32)),
            }
        )
    write("sosfiltfilt", {"signals": SIGNALS, "cases": cases})


def stereo(rate, seconds):
    frames = int(rate * seconds)
    return {
        "rate": rate,
        "frames": frames,
        "channels": [
            [{"kind": "tone", "freq": 500.0, "amp": 0.2}, {"kind": "tone", "freq": 1250.0, "amp": 0.1}],
            [{"kind": "tone", "freq": 3000.0, "amp": 0.2}, {"kind": "tone", "freq": 1250.0, "amp": 0.1}],
        ],
    }


def spectral_golden():
    cases = []
    specs = {
        "stereo48k": stereo(48000, 3.0003),
        "stereo16k": stereo(16000, 1.0019375),
        "stereo44k": stereo(44100, 2.5),
        "single": {"rate": 48000, "frames": 1, "channels": [[{"kind": "dc", "value": 0.1}], [{"kind": "dc", "value": -0.1}]]},
        "short_tail": stereo(48000, 0.2013),
        "surround": {
            "rate": 48000,
            "frames": 24011,
            "channels": [noise(20 + c, 0.2) + tones(48000, 0, 1250.0 + 10 * c, amp=0.1) for c in range(6)],
        },
    }
    for name, chunk in [
        ("stereo48k", None),
        ("stereo48k", 24000),
        ("stereo48k", 4800),
        ("stereo16k", None),
        ("stereo16k", 1600),
        ("stereo44k", 44100 * 8),
        ("stereo44k", 10000),
        ("single", None),
        ("short_tail", 4800),
        ("surround", 9000),
    ]:
        spec = specs[name]
        x = synth(spec)
        chunks = [x] if chunk is None else [x[s : s + chunk] for s in range(0, len(x), chunk)]
        out = list(audio_spectral.filter_audio_chunks(chunks, spec["rate"]))
        cases.append(
            {
                "signal": name,
                "chunk": chunk,
                "input_sha256": sha(x),
                "lengths": [len(c) for c in out],
                "output": digest(np.concatenate(out)),
            }
        )
    errors = {}
    for rate in (3000, 2999):
        try:
            list(audio_spectral.filter_audio_chunks([np.zeros((10, 1), np.float32)], rate))
            errors[str(rate)] = None
        except ValueError as exc:
            errors[str(rate)] = str(exc)
    try:
        list(transform_chunks([np.zeros((3, 1), np.float32), np.zeros((3, 1), np.float32)], lambda v: v, context_frames=4))
        errors["context"] = None
    except ValueError as exc:
        errors["context"] = str(exc)
    write("spectral", {"signals": specs, "cases": cases, "errors": errors})


def clock_golden():
    records = []
    for frames, rate, model_rate, hop, minimum in [
        (1, 48000, 44100, 512, 1024),
        (2, 48000, 44100, 512, 1024),
        (557, 48000, 44100, 512, 1024),
        (480240, 48000, 44100, 512, 1024),
        (16003, 16000, 44100, 512, 512),
        (48001, 48000, 44100, 512, 512),
        (44101, 44100, 44100, 512, 512),
        (220501, 22050, 44100, 3072, 3072),
        (96007, 96000, 44100, 3072, 3072),
        (8000, 8000, 44100, 512, 1024),
    ]:
        clock = src.RationalClock(frames, rate, model_rate=model_rate, hop=hop, minimum_model_frames=minimum)
        records.append({"args": [frames, rate, model_rate, hop, minimum], "record": clock.record(), "padded_frames": clock.padded_frames})
    roundtrips = []
    for name, spec in {
        "stereo48k": {"rate": 48000, "frames": 1115, "channels": [noise(12), noise(13)]},
        "mono16k": {"rate": 16000, "frames": 16003, "channels": [tones(16000, 0, 440.0)]},
        "pair2": {"rate": 48000, "frames": 2, "channels": [noise(1), noise(2)]},
        "single": {"rate": 48000, "frames": 1, "channels": [noise(3)]},
        "hf96k": {"rate": 96000, "frames": 9600, "channels": [tones(96000, 0, 23500.0, 997.0)]},
    }.items():
        x = synth(spec)
        clock = src.RationalClock(len(x), spec["rate"], hop=512 if name != "hf96k" else 3072, minimum_model_frames=1024)
        model = src.forward(x, clock)
        padded = src.pad_model(model, clock)
        back = src.inverse(src.unpad_model(padded, clock), clock)
        roundtrips.append(
            {
                "signal": name,
                "spec": spec,
                "hop": clock.hop,
                "input_sha256": sha(x),
                "model": digest(model),
                "padded": digest(padded),
                "inverse": digest(back),
            }
        )
    write("clock", {"records": records, "roundtrips": roundtrips})


def fade_golden():
    """`_stream_neural_track`'s raised-cosine fade, which goes through NumPy's float32 SIMD cosine."""
    fades = []
    for overlap in (1, 2, 7, 100, 1000, 12345, 32000, 44100, 88200, 96000):
        fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, overlap + 1, dtype=np.float32) / (overlap + 1))
        fades.append({"overlap": overlap, "fade": digest(fade)})
    from audio_common import splitmix_uniform

    x = (40.0 * splitmix_uniform(77, 100_000) - 20.0).astype(np.float32)
    write("fade", {"fades": fades, "cos_input": {"seed": 77, "count": 100_000}, "cos": digest(np.cos(x))})


def main() -> None:
    fade_golden()
    resample_golden()
    butter_golden()
    sosfiltfilt_golden()
    spectral_golden()
    clock_golden()


if __name__ == "__main__":
    main()

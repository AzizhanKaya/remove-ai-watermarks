"""Golden generator for the `audio` suite: inspection, verification and removal orchestration.

Neural forward passes are replaced by deterministic fake networks that the Rust tests implement
identically (`tests/fakes/mod.rs`); the receiver logic (`audioseal.read`, `wavmark.read`,
`read_openvoice_default`, `perth.read`, `silentcipher.read`) and all orchestration run unchanged.
"""

from __future__ import annotations

import importlib
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from audio_common import encode_raw, scrub, synth, write
from scipy import signal as sps

from remove_ai_watermarks import audio
from remove_ai_watermarks._internal import (
    _silentcipher_network,
    audio_codec,
    audio_dac,
    audio_snac,
    audioseal,
    perth,
    silentcipher,
    wavmark,
)
from remove_ai_watermarks._internal.c2pa import markers_claim_ai

PATTERN = importlib.import_module("wavmark").wm_add_util.fix_pattern[:16]


def band_amplitude(samples: np.ndarray, rate: float, freq: float = 1250.0) -> float:
    x = np.asarray(samples, dtype=np.float64)
    t = np.arange(len(x)) / rate
    return float(abs(2 * np.mean(x * np.exp(-2j * np.pi * freq * t))))


# Fake networks (mirrored by tests/fakes/mod.rs).


class FakeAudioSeal:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(self, wav, sample_rate):
        x = wav.reshape(-1).numpy()
        if self.mode == "energy":
            found = float(np.mean(x**2)) > 0.001
        elif self.mode == "positive":
            found = True
        elif self.mode == "negative":
            found = False
        else:
            found = self.mode == "resistant" or band_amplitude(x, 16000.0) > 0.01
        if self.mode == "crash" and not found:
            raise RuntimeError("AudioSeal post-verify crashed")
        prob = 0.9 if found else 0.1
        result = torch.zeros((1, 2, len(x)), dtype=torch.float32)
        result[0, 1, :] = prob
        message = torch.tensor([[float(b) for b in (0, 1) * 8]], dtype=torch.float32)
        return result, message


class FakeWavMark:
    def parameters(self):
        yield torch.zeros(1)

    def decode(self, batch):
        rows = []
        for window in batch.numpy():
            found = band_amplitude(window, 16000.0) > 0.01
            rows.append([float(b) for b in PATTERN] + [float(b) for b in (1, 0) * 8] if found else [0.25] * 32)
        return torch.tensor(rows, dtype=torch.float32)


class FakePerth:
    hp = SimpleNamespace(sample_rate=16000)
    device = "cpu"
    ap = SimpleNamespace(signal_to_magphase=lambda s: (s, None))

    def decoder(self, magnitudes):
        found = band_amplitude(magnitudes[0].numpy(), 16000.0) > 0.01
        return torch.tensor([[0.8 if found else 0.2]], dtype=torch.float32)


class FakeSilentCipherDecoder:
    def __call__(self, carrier):
        x = carrier.reshape(-1).numpy()
        found = band_amplitude(x, 44100.0) > 0.005
        symbols = [0, *[1 + (i % 4) for i in range(20)]] if found else [0] * 21
        scores = np.zeros((5, 63), dtype=np.float32)
        for t in range(63):
            scores[symbols[t % 21], t] = 1.0
        return torch.from_numpy(scores).view(1, 1, 5, 63)


def fake_librosa_resample(y, *, orig_sr, target_sr, **_):
    g = math.gcd(int(orig_sr), int(target_sr))
    return np.asarray(sps.resample_poly(np.asarray(y, np.float32), target_sr // g, orig_sr // g), dtype=np.float32)


def codec_fn(mode: str):
    if mode == "comb":
        def apply(x):
            shifted = torch.cat((torch.zeros(18, dtype=x.dtype), x[:-18])) if len(x) > 18 else torch.zeros_like(x)
            return 0.5 * (x + shifted)
    elif mode == "zero":
        def apply(x):
            return x * 0.0
    else:
        gain = float(mode)

        def apply(x):
            return x * gain
    return apply


class FakeDac:
    def __init__(self, mode):
        self.fn = codec_fn(mode)

    def encode(self, value, *, n_quantizers):
        assert n_quantizers == 3
        return SimpleNamespace(audio_codes=value.clone())

    def decode(self, *, audio_codes):
        return SimpleNamespace(audio_values=self.fn(audio_codes.reshape(-1)).view(1, 1, -1))


class FakeSnac:
    def __init__(self, mode):
        self.fn = codec_fn(mode)

    def reconstruct(self, value, *, n_quantizers):
        assert n_quantizers == 3
        return self.fn(value.reshape(-1)).view(1, 1, -1)


class FakeEncodec:
    def __init__(self, mode):
        self.fn = codec_fn(mode)

    def __call__(self, inputs, *, bandwidth):
        out = torch.stack([self.fn(inputs[c, 0]) for c in range(inputs.shape[0])]).unsqueeze(1)
        return SimpleNamespace(audio_values=out)


def install(monkey: dict, fakes: dict, markers: dict | None, source_paths: list[Path]) -> None:
    """Patch availability, model loaders and metadata for one scenario."""

    def setattr_(obj, name, value):
        monkey.setdefault((obj, name), getattr(obj, name))
        setattr(obj, name, value)

    for module, key in ((audioseal, "audioseal"), (wavmark, "wavmark"), (perth, "perth"), (silentcipher, "silentcipher")):
        setattr_(module, "is_available", (lambda on: (lambda: on))(key in fakes))
    if "audioseal" in fakes:
        setattr_(audioseal, "load_pinned_detector", (lambda m: (lambda: FakeAudioSeal(m)))(fakes["audioseal"]))
    if "wavmark" in fakes:
        setattr_(wavmark, "load_model", lambda: FakeWavMark())
    if "perth" in fakes:
        setattr_(perth, "load_model", lambda: FakePerth())
    if "silentcipher" in fakes:
        window = torch.hann_window(4096)
        setattr_(silentcipher, "load_model", lambda device=None: silentcipher.DecoderModel(FakeSilentCipherDecoder(), window, "cpu"))
        setattr_(_silentcipher_network, "magnitude", lambda signal, _window: signal)
        import librosa

        setattr_(librosa, "resample", fake_librosa_resample)
    for module, key, factory in (
        (audio_dac, "dac", lambda m: (lambda **kw: FakeDac(m))),
        (audio_snac, "snac", lambda m: (lambda **kw: FakeSnac(m))),
        (audio_codec, "encodec", lambda m: (lambda *a, **kw: FakeEncodec(m))),
    ):
        setattr_(module, "is_available", (lambda on: (lambda: on))(key in fakes))
        if key in fakes:
            setattr_(module, "load_pinned_model", factory(fakes[key]))
    resolved = {p.resolve() for p in source_paths}
    setattr_(audio.metadata, "get_ai_metadata", lambda p: dict(markers or {}) if Path(p).resolve() in resolved else {})


def restore(monkey: dict) -> None:
    for (obj, name), value in monkey.items():
        setattr(obj, name, value)
    monkey.clear()


def stereo(rate, frames, mark=0.1):
    return {
        "rate": rate,
        "frames": frames,
        "channels": [
            [{"kind": "tone", "freq": 500.0, "amp": 0.2}, {"kind": "tone", "freq": 1250.0, "amp": mark}],
            [{"kind": "tone", "freq": 3000.0, "amp": 0.2}, {"kind": "tone", "freq": 1250.0, "amp": mark}],
        ],
    }


def surround(rate, frames):
    return {
        "rate": rate,
        "frames": frames,
        "channels": [
            [{"kind": "tone", "freq": 300.0 + 150 * c, "amp": 0.15}, {"kind": "tone", "freq": 1250.0, "amp": 0.05}]
            for c in range(6)
        ],
    }


S16 = ["-c:a", "pcm_s16le"]
F32 = ["-c:a", "pcm_f32le"]
EL = {"claim_generator": "ElevenLabs", "issuer": "ElevenLabs", "source_type": "trainedAlgorithmicMedia (AI-generated)"}
GOOGLE = {"issuer": "Google LLC", "synthid_watermark": "present according to Google LLC provenance"}

SCENARIOS = [
    # identify
    ("identify_missing_decoder", stereo(16000, 16000), S16, None, {}, None, "identify", {}),
    ("identify_audioseal_marked", stereo(16000, 16000), S16, None, {"audioseal": "band"}, None, "identify", {}),
    ("identify_auto_short", stereo(16000, 12800), S16, None, {"audioseal": "band", "wavmark": "band"}, None, "identify", {"scheme": "auto"}),
    ("identify_all_48k", stereo(48000, 3 * 48000 + 31), S16, None, {"audioseal": "band", "wavmark": "band", "perth": "band", "silentcipher": "band"}, None, "identify", {"scheme": "all"}),
    ("identify_all_missing", stereo(16000, 16000), S16, None, {"audioseal": "band"}, None, "identify", {"scheme": "all"}),
    ("identify_silentcipher_16k", stereo(16000, 3 * 16000 + 31), S16, None, {"silentcipher": "band"}, None, "identify", {"scheme": "silentcipher"}),
    ("identify_silentcipher_short", stereo(16000, 32000), S16, None, {"silentcipher": "band"}, None, "identify", {"scheme": "silentcipher"}),
    ("identify_long_segments", stereo(16000, 31 * 16000), S16, None, {"audioseal": "band"}, None, "identify", {}),
    ("identify_crash_auto", stereo(16000, 16000, mark=0.0), S16, None, {"audioseal": "crash", "perth": "negative"}, None, "identify", {"scheme": "auto"}),
    ("identify_surround", surround(48000, 24000), F32, "5.1(side)", {"audioseal": "band"}, None, "identify", {"scheme": "auto"}),
    ("identify_no_invisible", stereo(16000, 16000), S16, None, {"audioseal": "band"}, EL, "identify", {"check_invisible": False}),
    ("recipe_elevenlabs", stereo(16000, 16000), S16, None, {"audioseal": "negative"}, EL, "identify", {}),
    ("recipe_google", stereo(16000, 16000), S16, None, {"audioseal": "negative"}, GOOGLE, "identify", {}),
    ("recipe_suno", stereo(16000, 16000), S16, None, {"audioseal": "negative"}, {"c2pa_manifest": "C2PA manifest store (5806 bytes)", "claim_generator": "Suno", "source_type": "trainedAlgorithmicMedia (AI-generated)"}, "identify", {}),
    ("recipe_transport", stereo(16000, 16000), S16, None, {"audioseal": "negative"}, {"c2pa_manifest": "C2PA manifest store (4942 bytes)", "claim_generator": "Anthropic Files", "issuer": "Anthropic Claude", "actions": "opened, com.anthropic.claude.provided"}, "identify", {}),
    ("recipe_detected", stereo(16000, 16000), S16, None, {"audioseal": "positive"}, None, "identify", {}),
    # removal
    ("remove_spectral_verified", stereo(16000, 16000), S16, None, {"audioseal": "band"}, None, "remove", {}),
    ("remove_resistant", stereo(16000, 16000), S16, None, {"audioseal": "resistant"}, None, "remove", {}),
    ("remove_force_resistant", stereo(16000, 16000), S16, None, {"audioseal": "resistant"}, None, "remove", {"force": True, "method": "spectral"}),
    ("remove_skip_negative", stereo(16000, 16000), S16, None, {"audioseal": "negative"}, None, "remove", {}),
    ("remove_auto_fallback", stereo(16000, 16000), S16, None, {"audioseal": "energy", "encodec": "0.1"}, None, "remove", {}),
    ("remove_forced_spectral_unverified", stereo(16000, 16031), S16, None, {}, None, "remove", {"force": True, "verify": False}),
    ("remove_encodec_mp3", stereo(16000, 16000), S16, None, {"audioseal": "band", "encodec": "0.05"}, None, "remove", {"method": "encodec", "output": "out.mp3"}),
    ("remove_encodec_bandwidth", stereo(16000, 16000), S16, None, {"encodec": "0.5"}, None, "remove", {"method": "encodec", "bandwidth_kbps": 12.0, "force": True, "verify": False}),
    ("remove_dac_comb", stereo(16000, 16001), S16, None, {"audioseal": "band", "dac": "comb"}, None, "remove", {"method": "dac"}),
    ("remove_dac_force_cpu", stereo(16000, 16001), S16, None, {"dac": "0.5"}, None, "remove", {"method": "dac", "vocoder_device": "cpu", "force": True, "verify": False}),
    ("remove_auto_m4a", stereo(16000, 16000), S16, None, {"audioseal": "band"}, None, "remove", {"scheme": "auto", "method": "auto", "output": "clean.m4a"}),
    ("remove_spectral_device_rejected", stereo(16000, 16000), S16, None, {}, None, "remove", {"method": "spectral", "vocoder_device": "cuda", "force": True}),
    ("remove_dac_flac", stereo(16000, 16000), S16, None, {"dac": "comb"}, None, "remove", {"method": "dac", "force": True, "verify": False, "output": "out.flac"}),
    ("remove_dac_44k_crossover", stereo(44100, 44100 + 7), F32, None, {"audioseal": "band", "dac": "comb"}, None, "remove", {"method": "dac", "allow_positive": True}),
    ("remove_dac_silent_quality", stereo(16000, 16000), S16, None, {"audioseal": "band", "dac": "zero"}, None, "remove", {"method": "dac"}),
    ("remove_dac_silent_forced", stereo(16000, 16000), S16, None, {"audioseal": "band", "dac": "zero"}, None, "remove", {"method": "dac", "force": True}),
    ("remove_dac_quality_override", stereo(16000, 16000), S16, None, {"audioseal": "negative", "dac": "zero"}, None, "remove", {"method": "dac", "scheme": "auto", "allow_unverified": True, "allow_quality_override": True}),
    ("remove_dac_still_positive", stereo(16000, 16000), S16, None, {"audioseal": "positive", "dac": "0.9"}, None, "remove", {"method": "dac", "scheme": "auto", "allow_unverified": True}),
    ("remove_dac_allow_positive", stereo(16000, 16000), S16, None, {"audioseal": "positive", "dac": "0.9"}, None, "remove", {"method": "dac", "scheme": "auto", "allow_positive": True}),
    ("remove_snac_comb", stereo(16000, 16000), S16, None, {"audioseal": "band", "snac": "comb"}, None, "remove", {"method": "snac"}),
    ("remove_dac_surround", surround(48000, 24000), F32, "5.1(side)", {"dac": "comb"}, None, "remove", {"method": "dac", "force": True, "verify": False}),
    ("remove_encodec_surround", surround(48000, 24000), F32, "5.1(side)", {"encodec": "0.5"}, None, "remove", {"method": "encodec", "force": True, "verify": False}),
    ("remove_dac_long_windows", {"rate": 8000, "frames": 8000 * 31 + 5, "channels": [[{"kind": "tone", "freq": 440.0, "amp": 0.3}, {"kind": "noise", "seed": 7, "amp": 0.02}]]}, S16, None, {"dac": "comb"}, None, "remove", {"method": "dac", "force": True, "verify": False}),
    ("remove_elevenlabs_auto", stereo(16000, 16000), S16, None, {"audioseal": "negative", "dac": "0.9"}, EL, "remove", {"scheme": "auto"}),
    ("remove_google_no_dac", stereo(16000, 16000), S16, None, {}, GOOGLE, "remove", {"scheme": "auto", "allow_unverified": True}),
    ("remove_all_missing_decoder", stereo(16000, 16000), S16, None, {"audioseal": "band"}, None, "remove", {"scheme": "all", "method": "spectral"}),
    ("remove_auto_short_wavmark", stereo(16000, 12800), S16, None, {"audioseal": "band", "wavmark": "band"}, None, "remove", {"scheme": "auto"}),
    ("remove_auto_crash", stereo(16000, 16000), S16, None, {"audioseal": "crash", "perth": "negative"}, None, "remove", {"scheme": "auto", "method": "spectral"}),
    ("remove_long_segments", stereo(16000, 31 * 16000), S16, None, {"audioseal": "band"}, None, "remove", {"method": "spectral"}),
    ("remove_all_schemes_verified", stereo(44100, 3 * 44100 + 17), S16, None, {"audioseal": "band", "wavmark": "band", "perth": "band", "silentcipher": "band"}, None, "remove", {"scheme": "all", "method": "spectral"}),
    ("remove_verify_disabled", stereo(16000, 16000), S16, None, {}, None, "remove", {"verify": False}),
    ("remove_same_path", stereo(16000, 16000), S16, None, {}, None, "remove", {"force": True, "verify": False, "output": "src.wav"}),
    ("remove_dac_six_untagged", {"rate": 16000, "frames": 8000, "channels": [[{"kind": "tone", "freq": 440.0, "amp": 0.15}]] * 6}, S16, "6.0", {"dac": "comb"}, GOOGLE, "remove", {"scheme": "auto", "allow_unverified": True}),
    ("identify_six_untagged", {"rate": 16000, "frames": 8000, "channels": [[{"kind": "tone", "freq": 440.0, "amp": 0.15}]] * 6}, S16, "6.0", {}, GOOGLE, "identify", {"scheme": "auto"}),
]


def run(kind: str, source: Path, output: Path, kwargs: dict):
    kwargs = dict(kwargs)
    kwargs.pop("output", None)
    if kind == "identify":
        return audio.identify_audio(source, **kwargs)
    return audio.remove_audio_invisible(source, output, **kwargs)


def main() -> None:
    records = []
    monkey: dict = {}
    for name, spec, args, layout, fakes, markers, kind, kwargs in SCENARIOS:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "src.wav"
            encode_raw(synth(spec), spec["rate"], source, args, layout)
            output = root / kwargs.get("output", "out.wav")
            install(monkey, fakes, markers, [source])
            record = {
                "name": name,
                "spec": spec,
                "args": args,
                "layout": layout,
                "fakes": fakes,
                "markers": markers,
                "claims_ai": markers_claim_ai(markers) if markers else False,
                "kind": kind,
                "kwargs": kwargs,
            }
            try:
                result = run(kind, source, output, kwargs)
                record["result"] = json.loads(scrub(json.dumps(asdict(result), default=str), root))
                if kind == "identify":
                    vendors = ("google", "elevenlabs", "suno", "udio", "openai", "cartesia", "meta", "resemble", "microsoft")
                    record["adaptive"] = {
                        scheme: {
                            str(vendor): list(audio.adaptive_auto_profiles(scheme, result, vendor=vendor))
                            for vendor in (None, *vendors)
                        }
                        for scheme in ("audioseal", "wavmark", "perth", "silentcipher", "all", "auto")
                    }
                    record["auto_scrub"] = audio.should_auto_scrub_audio(result, method="auto", explicit_vendor=None)
                record["output_exists"] = output.exists()
            except (ValueError, RuntimeError, ImportError, OSError) as exc:
                error = {"type": type(exc).__name__, "message": scrub(str(exc), root)}
                if isinstance(exc, audio.AudioVerificationIncomplete):
                    error["before"] = json.loads(scrub(json.dumps(asdict(exc.before), default=str), root))
                    error["after"] = json.loads(scrub(json.dumps(asdict(exc.after), default=str), root))
                record["error"] = error
                record["output_exists"] = output.exists()
            finally:
                restore(monkey)
            records.append(record)
            print(name, "error" if "error" in record else "ok", record.get("error", {}).get("message", "")[:100])
    write("pipeline", {"scenarios": records})


if __name__ == "__main__":
    main()

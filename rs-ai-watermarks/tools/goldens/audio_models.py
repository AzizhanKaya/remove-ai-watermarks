"""Golden generator for the `audio` suite: real pinned models on short tracked clips.

One model per process, to bound memory: run `identify` (AudioSeal), `remove` (AudioSeal before and
after a spectral pass) and `dac` (DAC only) as separate invocations, e.g.
`~/.cache/rs-ai-watermarks/oracle-venv/bin/python rs-ai-watermarks/tools/goldens/audio_models.py identify`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from audio_common import REPO, digest, scrub, write

from remove_ai_watermarks import audio
from remove_ai_watermarks._internal import audio_io

IDENTIFY_FIXTURE = "provenance/xai-grok-imagine-video.mp4"
DAC_FIXTURE = "provenance/higgsfield-genjutsu-motion-control.mp4"


def as_json(value, *roots):
    return json.loads(scrub(json.dumps(asdict(value), default=str), *roots))


def only_audioseal():
    for module in (audio.wavmark, audio.perth, audio.silentcipher):
        module.is_available = lambda: False


def identify() -> None:
    only_audioseal()
    path = REPO / "data" / "fixtures" / IDENTIFY_FIXTURE
    report = audio.identify_audio(path, scheme="audioseal")
    write("models_identify", {"fixture": IDENTIFY_FIXTURE, "report": as_json(report, REPO)})


def remove() -> None:
    only_audioseal()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        source = REPO / "data" / "fixtures" / IDENTIFY_FIXTURE
        output = root / "clean.wav"
        result = audio.remove_audio_invisible(source, output, scheme="audioseal", method="spectral", allow_unverified=True)
        samples = np.concatenate(list(audio_io.read_audio(output, audio_io.probe_audio(output))))
        write(
            "models_remove",
            {"fixture": IDENTIFY_FIXTURE, "result": as_json(result, root, REPO), "output": digest(samples)},
        )


def dac() -> None:
    for module in (audio.audioseal, audio.wavmark, audio.perth, audio.silentcipher):
        module.is_available = lambda: False
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        source = REPO / "data" / "fixtures" / DAC_FIXTURE
        output = root / "dac.wav"
        result = audio.remove_audio_invisible(
            source, output, method="dac", vocoder_device="cpu", force=True, verify=False
        )
        samples = np.concatenate(list(audio_io.read_audio(output, audio_io.probe_audio(output))))
        write("models_dac", {"fixture": DAC_FIXTURE, "result": as_json(result, root, REPO), "output": digest(samples)})


if __name__ == "__main__":
    {"identify": identify, "remove": remove, "dac": dac}[sys.argv[1]]()

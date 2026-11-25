"""Golden generator for the `audio` suite: ffmpeg IO, fidelity and perceptual metrics.

Synthetic sources are encoded with `audio_common.encode_raw` (the Rust tests rerun the same
command and check the file hash); tracked AAC fixtures in `data/fixtures` are probed and decoded
in place. Every generated file lives in a temporary directory.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from audio_common import REPO, digest, encode_raw, file_sha, scrub, synth, write

from remove_ai_watermarks._internal import audio_io
from remove_ai_watermarks._internal.audio_quality import (
    audio_quality_passed,
    compute_perceptual_metrics,
    measure_fidelity,
)

FIXTURES = [
    "provenance/dreamina-seedance-2-mini.mp4",
    "provenance/higgsfield-happyhorse.mp4",
    "provenance/higgsfield-minimax-h3.mp4",
    "provenance/xai-grok-imagine-video.mp4",
    "visible/sora/provider-original.mp4",
]


def stereo_tones(rate, frames, amp=(0.35, 0.30)):
    return {
        "rate": rate,
        "frames": frames,
        "channels": [
            [{"kind": "tone", "freq": 330.0, "amp": amp[0]}, {"kind": "tone", "freq": 1250.0, "amp": 0.05}],
            [{"kind": "tone", "freq": 440.0, "amp": amp[1], "phase": 0.4}, {"kind": "noise", "seed": 9, "amp": 0.01}],
        ],
    }


SOURCES = {
    "s16_stereo48k": (stereo_tones(48000, 2501 + 48000), ["-c:a", "pcm_s16le"], "src.wav", None),
    "f32_stereo24k": (stereo_tones(24000, 24000 * 2 + 137), ["-c:a", "pcm_f32le"], "src.wav", None),
    "s24_mono44k": (
        {"rate": 44100, "frames": 44100 + 11, "channels": [[{"kind": "chirp", "f0": 100.0, "f1": 9000.0, "amp": 0.6}]]},
        ["-c:a", "pcm_s24le"],
        "src.wav",
        None,
    ),
    "side51_16k": (
        {
            "rate": 16000,
            "frames": 16000 + 1600,
            "channels": [[{"kind": "tone", "freq": 200.0 * (c + 1), "amp": 0.1 + 0.05 * c}] for c in range(6)],
        },
        ["-c:a", "pcm_f32le"],
        "src.wav",
        "5.1(side)",
    ),
    "flac_stereo22k": (stereo_tones(22050, 22050 + 999), ["-c:a", "flac"], "src.flac", None),
    "aiff_stereo16k": (stereo_tones(16000, 16000), ["-c:a", "pcm_s16be"], "src.aiff", None),
    "clipped_mono48k": (
        {"rate": 48000, "frames": 24000, "channels": [[{"kind": "tone", "freq": 440.0, "amp": 1.25}]]},
        ["-c:a", "pcm_f32le"],
        "src.wav",
        None,
    ),
    "odd_rate19k": (
        {"rate": 19000, "frames": 19000, "channels": [[{"kind": "silence"}]]},
        ["-c:a", "pcm_f32le"],
        "src.wav",
        None,
    ),
}

WRITES = [
    ("s16_stereo48k", ".wav", False),
    ("s16_stereo48k", ".wav", True),
    ("s16_stereo48k", ".flac", False),
    ("s16_stereo48k", ".mka", False),
    ("f32_stereo24k", ".mp3", False),
    ("f32_stereo24k", ".m4a", False),
    ("f32_stereo24k", ".aac", False),
    ("f32_stereo24k", ".ogg", False),
    ("f32_stereo24k", ".opus", False),
    ("s24_mono44k", ".wav", False),
    ("s24_mono44k", ".flac", False),
    ("side51_16k", ".wav", False),
    ("side51_16k", ".flac", False),
    ("side51_16k", ".m4a", False),
    ("flac_stereo22k", ".wav", False),
    ("aiff_stereo16k", ".flac", False),
    ("clipped_mono48k", ".wav", False),
]


def info_json(info):
    return json.loads(json.dumps(asdict(info), default=str))


def error_text(func):
    try:
        func()
    except (ValueError, RuntimeError, OSError) as exc:
        return {"type": type(exc).__name__, "message": str(exc)}
    return None


def main() -> None:
    sources = {}
    writes = []
    errors = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = {}
        for name, (spec, args, filename, layout) in SOURCES.items():
            folder = root / name
            folder.mkdir()
            path = folder / filename
            encode_raw(synth(spec), spec["rate"], path, args, layout)
            info = audio_io.probe_audio(path)
            frames = np.concatenate(list(audio_io.read_audio(path, info, chunk_frames=4096)))
            chunks = [len(c) for c in audio_io.read_audio(path, info, chunk_frames=12345)]
            paths[name] = (path, info, frames)
            sources[name] = {
                "spec": spec,
                "args": args,
                "filename": filename,
                "layout": layout,
                "file_sha256": file_sha(path),
                "info": info_json(info),
                "decoded": digest(frames),
                "chunk_lengths_12345": chunks,
            }
        for name, suffix, bitexact in WRITES:
            path, info, frames = paths[name]
            output = path.parent / f"out{'-bitexact' if bitexact else ''}{suffix}"
            pieces = [frames[:20000], frames[20000:]] if len(frames) > 20000 else [frames]
            try:
                count = audio_io.write_audio(info, output, pieces, expected_frames=len(frames), bitexact=bitexact)
            except (RuntimeError, ValueError) as exc:
                writes.append({"source": name, "suffix": suffix, "bitexact": bitexact, "write_error": str(exc)})
                assert not output.exists()
                continue
            delivered = audio_io.probe_audio(output)
            lossy = suffix in {".mp3", ".m4a", ".ogg", ".oga", ".opus", ".aac"}
            record = {
                "source": name,
                "suffix": suffix,
                "bitexact": bitexact,
                "codec_args": audio_io._audio_codec_args(suffix, info, bitexact=bitexact),
                "count": count,
                "file_sha256": file_sha(output),
                "info": info_json(delivered),
                "decoded": digest(np.concatenate(list(audio_io.read_audio(output, delivered)))),
            }
            try:
                fidelity = measure_fidelity(path, info, output)
                record["fidelity"] = asdict(fidelity)
                record["quality_passed"] = audio_quality_passed(fidelity, lossy=lossy)
            except RuntimeError as exc:
                record["fidelity_error"] = str(exc)
            record["perceptual"] = asdict(compute_perceptual_metrics(path, info, output))
            writes.append(record)
        stereo_path, stereo_info, stereo = paths["s16_stereo48k"]
        mono_path, mono_info, _ = paths["odd_rate19k"]
        errors["mp3_rate"] = error_text(
            lambda: audio_io.write_audio(mono_info, root / "odd.mp3", [np.zeros((19000, 1), np.float32)])
        )
        side_info = paths["side51_16k"][1]
        errors["mp3_channels"] = error_text(lambda: audio_io._audio_codec_args(".mp3", side_info, bitexact=False))
        errors["opus_channels"] = error_text(lambda: audio_io._audio_codec_args(".opus", side_info, bitexact=False))
        errors["bad_suffix"] = error_text(lambda: audio_io._audio_codec_args(".xyz", stereo_info, bitexact=False))
        errors["bitexact_flac"] = error_text(
            lambda: audio_io.write_audio(stereo_info, root / "b.flac", [stereo], bitexact=True)
        )
        errors["flac_range"] = error_text(
            lambda: audio_io.write_audio(stereo_info, root / "r.flac", [np.full((10, 2), 1.5, np.float32)])
        )
        errors["nan"] = error_text(
            lambda: audio_io.write_audio(stereo_info, root / "n.wav", [np.full((10, 2), np.nan, np.float32)])
        )
        errors["channels"] = error_text(
            lambda: audio_io.write_audio(stereo_info, root / "c.wav", [np.zeros((10, 1), np.float32)])
        )
        errors["count"] = error_text(
            lambda: audio_io.write_audio(stereo_info, root / "k.wav", [stereo[:-1]], expected_frames=len(stereo))
        )
        junk = root / "not-audio.bin"
        junk.write_bytes(b"not a media container")
        errors["no_audio"] = error_text(lambda: audio_io.probe_audio(junk))
        errors["missing_stream"] = error_text(lambda: audio_io.probe_audio(stereo_path, stream=3))
        for value in errors.values():
            if value is not None:
                value["message"] = scrub(value["message"], root.resolve(), root)

        # Mid-track dropout and stereo drift (test_windowed_audio_quality_gate_catches_mid_track_dropout...).
        dropout_spec = stereo_tones(16000, 64000)
        drop = synth(dropout_spec)
        drop[16000:40000] = 0.0
        source16 = root / "drop-src.wav"
        dropped = root / "drop-out.wav"
        encode_raw(synth(dropout_spec), 16000, source16, ["-c:a", "pcm_s16le"])
        encode_raw(drop, 16000, dropped, ["-c:a", "pcm_s16le"])
        info16 = audio_io.probe_audio(source16)
        fidelity = measure_fidelity(source16, info16, dropped)
        swapped = root / "drop-swap.wav"
        encode_raw(synth(dropout_spec)[:, ::-1].copy(), 16000, swapped, ["-c:a", "pcm_s16le"])
        dropout = {
            "spec": dropout_spec,
            "dropout": [16000, 40000],
            "source_sha256": file_sha(source16),
            "output_sha256": file_sha(dropped),
            "fidelity": asdict(fidelity),
            "quality_passed": audio_quality_passed(fidelity, lossy=False),
            "swapped_fidelity": asdict(measure_fidelity(source16, info16, swapped)),
        }

    fixtures = []
    for relative in FIXTURES:
        path = REPO / "data" / "fixtures" / relative
        info = audio_io.probe_audio(path)
        frames = np.concatenate(list(audio_io.read_audio(path, info)))
        fixtures.append({"fixture": relative, "info": info_json(info), "decoded": digest(frames)})

    write("io", {"sources": sources, "writes": writes, "errors": errors, "dropout": dropout, "fixtures": fixtures})


if __name__ == "__main__":
    main()

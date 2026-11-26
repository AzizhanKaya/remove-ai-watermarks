"""Goldens for the video tool, probe, decode, encode and quality boundary.

`OMP_NUM_THREADS=4 .venv/bin/python rs-ai-watermarks/tools/goldens/video.py` from the repository root.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from video_common import FIXTURES, build_clips, clip_manifest, write_golden

from remove_ai_watermarks import video_encoding as ve
from remove_ai_watermarks.video_decode import open_video_capture, open_video_capture_16bit
from remove_ai_watermarks.video_quality import measure_video_fidelity, video_quality_passed

cv2.setNumThreads(4)


def fixture_videos() -> list[Path]:
    paths = sorted((FIXTURES / "provenance").glob("*.mp4"))
    paths += sorted((FIXTURES / "visible").glob("*/example.mp4"))
    paths += sorted((FIXTURES / "visible").glob("*/provider-original.mp4"))
    return paths


def rel(path: Path, built: dict[str, Path]) -> str:
    for name, clip in built.items():
        if clip == path:
            return f"clip:{name}"
    return str(path.relative_to(FIXTURES))


def ts_digest(values: tuple[float, ...]) -> str:
    return hashlib.sha256(json.dumps(list(values)).encode()).hexdigest()


def decode_summary(path: Path, *, per_frame: bool) -> dict[str, Any]:
    cap = open_video_capture(path)
    props = {
        "opened": bool(cap.isOpened()),
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }
    digest = hashlib.sha256()
    frames: list[str] = []
    msec: list[float] = []
    count = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        data = frame.tobytes()
        digest.update(data)
        if per_frame:
            frames.append(hashlib.sha256(data).hexdigest())
            msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        count += 1
    cap.release()
    out: dict[str, Any] = {**props, "frames": count, "sha256": digest.hexdigest()}
    if per_frame:
        out["frame_sha256"] = frames
        out["pos_msec"] = msec
    return out


def probe_summary(path: Path) -> dict[str, Any]:
    timestamps = ve.probe_video_timestamps(path)
    try:
        duration: Any = ve.probe_video_duration(path)
    except ValueError as exc:
        duration = f"error: {exc}"
    keyframes = ve.probe_video_keyframes(path)
    return {
        "profile": asdict(ve.probe_video_encode_profile(path)),
        "timestamps_count": len(timestamps),
        "timestamps_head": list(timestamps[:4]),
        "timestamps_sha256": ts_digest(timestamps),
        "duration": duration,
        "keyframes": None if keyframes is None else {**asdict(keyframes), "keyframes": keyframes.keyframes[:8]},
        "keyframe_count": None if keyframes is None else len(keyframes.keyframes),
        "decoded_frames": ve.count_decoded_video_frames(path),
    }


def commands() -> list[dict[str, Any]]:
    profiles = {
        "default": ve.VideoEncodeProfile(),
        "bt709": ve.VideoEncodeProfile(
            color_range="tv", color_space="bt709", color_transfer="bt709", color_primaries="bt709",
            time_base="1/12800", start_pts=0,
        ),
        "hdr10": ve.VideoEncodeProfile(
            pixel_format="yuv420p10le", color_range="tv", color_space="bt2020nc",
            color_transfer="smpte2084", color_primaries="bt2020", time_base="1/90000",
        ),
        "full_range": ve.VideoEncodeProfile(pixel_format="yuv444p", color_range="pc", time_base="1001/30000"),
    }
    cases = []
    for suffix in (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv"):
        for name, profile in profiles.items():
            for crf, timestamped, copyts in ((14, False, False), (0, True, True), (18, True, False)):
                pix = "bgr48le" if profile.pixel_format in ve.HIGH_BIT_DEPTH_PIXEL_FORMATS else "bgr24"
                cmd = ve.raw_video_command(
                    Path(f"out{suffix}"), width=160, height=96, fps=24.0 if crf else 29.97002997,
                    crf=crf, profile=profile, timestamped_input=timestamped,
                    copy_input_timestamps=copyts, input_pix_fmt=pix,
                )
                cmd[0] = "ffmpeg"
                cases.append({
                    "suffix": suffix, "profile": name, "crf": crf, "timestamped": timestamped,
                    "copyts": copyts, "input_pix_fmt": pix, "command": cmd,
                })
    cmd = ve.raw_video_command(
        Path("seg.mp4"), width=64, height=32, fps=30.0, crf=14, profile=profiles["default"], h264_profile="main"
    )
    cmd[0] = "ffmpeg"
    cases.append({"suffix": ".mp4", "profile": "default", "crf": 14, "timestamped": False, "copyts": False,
                  "input_pix_fmt": "bgr24", "h264_profile": "main", "command": cmd})
    return cases


def end_repairs() -> list[dict[str, Any]]:
    cases = []
    source = {"streams": [{"time_base": "1/12800", "duration_ts": 25600, "start_pts": 0}]}
    for packets in (
        [{"pts": 0, "dts": -1024}, {"pts": 2048, "dts": -512}, {"pts": 1024, "dts": 0}, {"pts": 24576, "dts": 24064}],
        [{"pts": 0, "dts": 0}, {"pts": 512, "dts": 512}, {"pts": 25088, "dts": 25088}],
        [{"pts": 0, "dts": 0}, {"pts": 30000, "dts": 30000}],
    ):
        encoded = {"streams": [{"time_base": "1/12800"}], "packets": packets}
        try:
            repair = ve.video_end_repair(source, encoded)
            result: Any = {**asdict(repair), "bitstream_filter": repair.bitstream_filter}
        except ValueError as exc:
            result = f"error: {exc}"
        cases.append({"source": source, "encoded": encoded, "result": result})
    return cases


def quality(built: dict[str, Path]) -> list[dict[str, Any]]:
    src = built["cfr_audio"]
    same = built["delivered_same"]
    small = built["delivered_small"]
    geometry = (160, 96, 24.0)
    out = []
    for name, delivered, kwargs in (
        ("same", same, {}),
        ("same_timestamps", same, {"preserve_timestamps": True}),
        ("small", small, {}),
    ):
        try:
            report: Any = measure_video_fidelity(src, delivered, source_geometry=geometry, duration=None, **kwargs)
            passed: Any = video_quality_passed(report)
        except ValueError as exc:
            report, passed = f"error: {exc}", None
        out.append({"name": name, "report": report, "passed": passed})
    report = measure_video_fidelity(src, small, source_geometry=geometry, duration=1.0)
    out.append({"name": "small_duration", "report": report, "passed": video_quality_passed(report)})
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        built = build_clips(work)
        write_golden("video", "clips", clip_manifest(built))

        decode = {}
        probe = {}
        for path in [*built.values(), *fixture_videos()]:
            key = rel(path, built)
            synthetic = key.startswith("clip:")
            decode[key] = decode_summary(path, per_frame=synthetic)
            probe[key] = probe_summary(path)
        hevc = built["hevc_10bit"]
        cap16 = open_video_capture_16bit(hevc, width=128, height=72)
        digest = hashlib.sha256()
        n16 = 0
        while True:
            ok, frame = cap16.read()
            if not ok:
                break
            digest.update(frame.tobytes())
            n16 += 1
        cap16.release()
        decode["clip:hevc_10bit:bgr48"] = {"frames": n16, "sha256": digest.hexdigest()}
        write_golden("video", "decode", decode)
        write_golden("video", "probe", probe)
        write_golden("video", "commands", commands())
        write_golden("video", "end_repair", end_repairs())
        write_golden("video", "quality", quality(built))


if __name__ == "__main__":
    main()

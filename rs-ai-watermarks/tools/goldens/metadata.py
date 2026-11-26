"""Golden metadata scans, strips and records from the Python reference.

Run from the repository root: `.venv/bin/python rs-ai-watermarks/tools/goldens/metadata.py [media|images|all]`.
Synthetic inputs are generated with ffmpeg / Pillow into a temp dir and embedded (base64) in the
golden, so the Rust test replays the same bytes; fixtures are referenced in place.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# This directory holds a `c2pa.py` generator that would shadow the c2pa-python package.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

from remove_ai_watermarks import metadata as md  # noqa: E402
from remove_ai_watermarks.metadata_record import collect_metadata_record  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "rs-ai-watermarks" / "goldens" / "metadata"
FIXTURES = ROOT / "data" / "fixtures"
MEDIA_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".flv", ".avi", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}
SUNO = "made with suno; created=2025-01-02T03:04:05Z; id=0b7f9d7e-8c1a-4e5b-9a7d-2f3c4b5a6d7e"
AIGC = '{"Label":"1","ContentProducer":"001191110108MA01KP2T8U00000","ProduceID":"p-1"}'
RANDOM_ID_EXTS = {".mkv", ".webm", ".mka", ".ogg", ".oga", ".opus"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def synth_media(work: Path) -> list[Path]:
    """Short ffmpeg-made containers with and without AI tags."""
    video = ["-f", "lavfi", "-i", "testsrc=size=32x32:rate=5", "-t", "0.4", "-pix_fmt", "yuv420p"]
    audio = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000", "-t", "0.2"]
    tags = ["-metadata", f"AIGC={AIGC}", "-metadata", "title=clip"]
    made: list[Path] = []

    def make(name: str, *args: str) -> None:
        path = work / name
        ffmpeg(*args, str(path))
        made.append(path)

    x264 = ["-c:v", "libx264", "-preset", "ultrafast", "-bitexact", "-fflags", "+bitexact"]
    make("plain.mp4", *video, *x264)
    make("aigc.mp4", *video, *x264, *tags, "-movflags", "use_metadata_tags")
    make("aigc.mov", *video, *x264, *tags, "-movflags", "use_metadata_tags")
    make("plain.mkv", *video, *x264)
    make("aigc.mkv", *video, *x264, *tags)
    make("aigc.webm", *video, "-c:v", "libvpx", "-b:v", "50k", *tags)
    make("aigc.flv", *video, "-c:v", "flv", *tags)
    make("aigc.avi", *video, "-c:v", "mpeg4", *tags)
    make("plain.wav", *audio, "-c:a", "pcm_s16le", "-bitexact")
    make("suno.wav", *audio, "-c:a", "pcm_s16le", "-bitexact", "-metadata", f"comment={SUNO}")
    make("aigc.mp3", *audio, "-c:a", "libmp3lame", *tags)
    make("suno.mp3", *audio, "-c:a", "libmp3lame", "-metadata", f"comment={SUNO}")
    make("aigc.flac", *audio, "-c:a", "flac", *tags)
    make("aigc.ogg", *audio, "-c:a", "libvorbis", *tags)
    make("suno.m4a", *audio, "-c:a", "aac", "-metadata", f"comment={SUNO}", "-movflags", "use_metadata_tags")
    plain = (work / "plain.mp4").read_bytes()
    for name, data in (
        ("truncated.mp4", plain[: len(plain) // 2]),
        ("random.mp4", bytes((i * 131 + 7) % 251 for i in range(3000))),
        ("empty.mp4", b""),
        ("truncated.mkv", (work / "plain.mkv").read_bytes()[:400]),
    ):
        path = work / name
        path.write_bytes(data)
        made.append(path)
    return made


def media_digest(path: Path) -> dict[str, Any]:
    """Packets and tags of a stream copy whose muxer writes random IDs (Matroska UID, Ogg serial)."""
    packets = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0", "-c", "copy", "-f", "framemd5", "-"],
        capture_output=True,
        check=True,
    ).stdout
    tags = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    ).stdout
    return {"framemd5_sha256": sha(packets), "tags": json.loads(tags), "size": path.stat().st_size}


def norm(value: Any, work: Path) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return json.loads(text.replace(str(work), "{dir}").replace(str(FIXTURES), "{fixtures}"))


def outcome(fn: Any) -> Any:
    try:
        return {"ok": fn()}
    except Exception as exc:  # noqa: BLE001 - the exception class is the recorded outcome
        message = str(exc)
        if isinstance(exc, RuntimeError):
            message = message.split(": ", 1)[0]  # ffmpeg's stderr carries heap addresses
        return {"error": type(exc).__name__, "message": message}


def record_summary(path: Path) -> dict[str, Any]:
    record = collect_metadata_record(path)
    store = record.pop("c2pa_store", None)
    for key in ("metadata_base64", "tail_base64"):
        raw = base64.b64decode(record.pop(key))
        record[key.replace("_base64", "_sha256")] = sha(raw)
        record[key.replace("_base64", "_len")] = len(raw)
    record["has_c2pa_store"] = store is not None
    return record


def case(path: Path, work: Path, label: str) -> dict[str, Any]:
    out_dir = work / "out"
    out_dir.mkdir(exist_ok=True)
    entry: dict[str, Any] = {"name": label}
    entry["has_ai_metadata"] = outcome(lambda: md.has_ai_metadata(path))
    entry["get_ai_metadata"] = outcome(lambda: md.get_ai_metadata(path))
    entry["aigc_label"] = outcome(lambda: md.aigc_label(path))
    entry["unreadable_reason"] = md.unreadable_reason(path)
    entry["record"] = outcome(lambda: record_summary(path))
    for keep in (True, False):
        target = out_dir / f"{'keep' if keep else 'all'}-{path.name}"

        def strip(target: Path = target, keep: bool = keep) -> dict[str, Any]:
            written = md.remove_ai_metadata(path, target, keep_standard=keep)
            if written.suffix.lower() in RANDOM_ID_EXTS:
                return media_digest(written)
            return {"sha256": sha(written.read_bytes()), "size": written.stat().st_size}

        entry[f"remove_keep_{keep}"] = outcome(strip)
        verified = out_dir / f"v{'keep' if keep else 'all'}-{path.name}"
        entry[f"verify_keep_{keep}"] = outcome(
            lambda verified=verified, keep=keep: dict(md.strip_and_verify(path, verified, keep_standard=keep)[1])
        )
    return norm(entry, work)


def media_cases() -> list[dict[str, Any]]:
    cases = []
    work = Path(tempfile.mkdtemp(prefix="raiw-md-"))
    try:
        for path in sorted(FIXTURES.rglob("*")):
            if path.suffix.lower() in MEDIA_EXTS:
                rel = str(path.relative_to(FIXTURES))
                entry = case(path, work, rel)
                entry["fixture"] = rel
                cases.append(entry)
        synth = work / "synth"
        synth.mkdir()
        for path in synth_media(synth):
            entry = case(path, work, path.name)
            entry["input_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            cases.append(entry)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return cases


SIG = "Signature: " + "QUJD" * 20
UUID = "0b7f9d7e-8c1a-4e5b-9a7d-2f3c4b5a6d7e"
TC260_XMP = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/">'
    "<TC260:AIGC>{&quot;Label&quot;:&quot;1&quot;,&quot;ContentProducer&quot;:&quot;p&quot;}</TC260:AIGC>"
    "</rdf:Description></rdf:RDF></x:xmpmeta>"
)
IPTC_XMP = (
    '<?xpacket begin=""?><x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
    'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia" '
    'Iptc4xmpExt:AISystemUsed="Firefly Image 3"/></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    import zlib

    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def insert_png_chunk(png: bytes, chunk: bytes, before: bytes) -> bytes:
    at = png.index(before) - 4
    return png[:at] + chunk + png[at:]


def synth_images(work: Path) -> list[Path]:
    import io

    import numpy as np
    import piexif
    from PIL import Image, PngImagePlugin

    rng = np.random.default_rng(7)
    rgb = Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
    made: list[Path] = []

    def save(name: str, data: bytes) -> None:
        (work / name).write_bytes(data)
        made.append(work / name)

    def png(image: Image.Image, info: PngImagePlugin.PngInfo | None = None, **kw: Any) -> bytes:
        buf = io.BytesIO()
        image.save(buf, "PNG", pnginfo=info, **kw)
        return buf.getvalue()

    def jpeg(image: Image.Image, **kw: Any) -> bytes:
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=90, **kw)
        return buf.getvalue()

    exif_ai = piexif.dump(
        {"0th": {piexif.ImageIFD.Software: b"Adobe Firefly", piexif.ImageIFD.Make: b"Canon", 274: 6}, "Exif": {}}
    )
    exif_xai = piexif.dump({"0th": {270: SIG.encode(), 315: UUID.encode(), 274: 6, 271: b"Canon"}})
    exif_app = piexif.dump(
        {"0th": {270: b'{"product":"dreamina","exportType":"generation"}'}, "Exif": {37510: b"ASCII\x00\x00\x00ok"}}
    )
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "masterpiece, Steps: 20")
    info.add_text("Author", "me")
    info.add_text("Title", "NovelAI Diffusion V3")
    info.add_text("Comment", "kept")
    save("text_params.png", png(rgb, info))
    info = PngImagePlugin.PngInfo()
    info.add_itxt("XML:com.adobe.xmp", TC260_XMP, zip=True)
    info.add_text("hf-job-id", " job-123 ")
    save("itxt_tc260.png", png(rgb, info))
    info = PngImagePlugin.PngInfo()
    info.add_text("XML:com.adobe.xmp", IPTC_XMP, zip=True)
    save("ztxt_iptc.png", png(rgb, info, dpi=(300, 300)))
    info = PngImagePlugin.PngInfo()
    info.add_text("AIGC", AIGC)
    info.add_text("Description", "plain caption")
    save("aigc_text.png", png(rgb, info))
    save("exif_xai.png", png(rgb, exif=exif_xai))
    raw = exif_xai[6:]
    profile = "\nexif\n" + f"{len(raw):8d}\n" + "\n".join(raw.hex()[i : i + 72] for i in range(0, len(raw) * 2, 72)) + "\n"
    info = PngImagePlugin.PngInfo()
    info.add_text("Raw profile type exif", profile, zip=True)
    save("raw_profile_exif.png", png(rgb, info))
    deep = np.asarray(rgb, dtype=np.uint16) * 257
    buf = io.BytesIO()
    Image.fromarray(deep[:, :, 0]).save(buf, "PNG")
    gray16 = buf.getvalue()
    gray16 = insert_png_chunk(gray16, png_chunk(b"tEXt", b"Software\x00Midjourney v6"), b"IDAT")
    gray16 = insert_png_chunk(gray16, png_chunk(b"tEXt", b"Author\x00someone"), b"IDAT")
    gray16 = insert_png_chunk(gray16, png_chunk(b"eXIf", exif_ai[6:]), b"IDAT")
    save("gray16.png", gray16)
    pal = rgb.quantize(colors=12)
    pal.info["transparency"] = 3
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", "a cat")
    save("palette.png", png(pal, info, transparency=3))
    save("gray.png", png(rgb.convert("L"), dpi=(72, 72)))
    late = png(rgb)
    late = insert_png_chunk(late, png_chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + IPTC_XMP.encode()), b"IEND")
    save("late_xmp.png", late)
    bad = png(rgb)
    chunk = png_chunk(b"eXIf", exif_ai[6:])
    bad = insert_png_chunk(bad, chunk[:-4] + b"\x00\x00\x00\x00", b"IDAT")
    save("bad_exif_crc.png", bad)
    save("truncated.png", png(rgb)[:300])
    cloud = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
        'dcterms:provenance="https://cai-manifests.adobe.com/manifests/abc"/></rdf:RDF></x:xmpmeta>'
    )
    info = PngImagePlugin.PngInfo()
    info.add_itxt("XML:com.adobe.xmp", cloud)
    save("cloud.png", png(rgb, info))
    save("exif_ai.jpg", jpeg(rgb, exif=exif_ai, dpi=(300, 300)))
    plain = jpeg(rgb)
    xmp_seg = b"http://ns.adobe.com/xap/1.0/\x00" + IPTC_XMP.encode()
    save("xmp_iptc.jpg", plain[:2] + b"\xff\xe1" + struct.pack(">H", len(xmp_seg) + 2) + xmp_seg + plain[2:])
    iptc = b"\x1c\x02\x78" + struct.pack(">H", len(SIG)) + SIG.encode() + b"\x1c\x02\x50" + struct.pack(">H", 36) + UUID.encode()
    res = b"8BIM\x04\x04\x00\x00" + struct.pack(">I", len(iptc)) + iptc + (b"\x00" if len(iptc) % 2 else b"")
    seg = b"Photoshop 3.0\x00" + res
    save("iptc_xai.jpg", plain[:2] + b"\xff\xed" + struct.pack(">H", len(seg) + 2) + seg + plain[2:])
    save("samsung.jpg", plain + b"PhotoEditor_Re_Edit_Data" + b'{"genAIType":4}')
    aigc_seg = b"AIGC" + AIGC.encode()
    com = b"\xff\xfe" + struct.pack(">H", 9) + b"comment"
    save("bare_aigc.jpg", plain[:2] + b"\xff\xeb" + struct.pack(">H", len(aigc_seg) + 2) + aigc_seg + com + plain[2:])
    save("app_json.jpg", jpeg(rgb, exif=exif_app))
    save("truncated.jpg", plain[: len(plain) // 2])
    buf = io.BytesIO()
    rgb.save(buf, "WEBP", lossless=True, exif=exif_ai)
    save("exif.webp", buf.getvalue())
    buf = io.BytesIO()
    rgb.convert("P").save(buf, "GIF", comment=b"made with midjourney")
    save("comment.gif", buf.getvalue())
    buf = io.BytesIO()
    rgb.save(buf, "TIFF", tiffinfo={305: "Stable Diffusion XL"})
    save("software.tif", buf.getvalue())
    buf = io.BytesIO()
    rgb.save(buf, "BMP")
    save("plain.bmp", buf.getvalue())
    save("misnamed_png.jpg", png(rgb))
    return made


def shrink(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: shrink(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shrink(v) for v in value]
    if isinstance(value, str) and len(value) > 160:
        return f"sha256:{sha(value.encode())}:{len(value)}"
    return value


def forensic_summary(path: Path) -> Any:
    from remove_ai_watermarks.forensic_metadata import collect_forensic_metadata

    record = collect_forensic_metadata(path)
    for key in ("mtime", "birthtime", "file"):
        record.pop(key, None)
    record["c2pa_store"] = bool(record.get("c2pa_store"))
    return shrink(record)


def detectors(path: Path) -> dict[str, Any]:
    return {
        "xai_signature": md.xai_signature(path),
        "iptc_ai_system": md.iptc_ai_system(path),
        "samsung_genai": md.samsung_genai(path),
        "huggingface_job": md.huggingface_job(path),
        "c2pa_cloud_manifest": md.c2pa_cloud_manifest(path),
        "synthid_source": md.synthid_source(path),
        "exif_generator": md.exif_generator(path),
    }


def image_cases() -> list[dict[str, Any]]:
    cases = []
    work = Path(tempfile.mkdtemp(prefix="raiw-md-"))
    try:
        sources = [p for p in sorted(FIXTURES.rglob("*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        synth = work / "synth"
        synth.mkdir()
        sources += synth_images(synth)
        for path in sources:
            synthetic = path.is_relative_to(synth)
            label = path.name if synthetic else str(path.relative_to(FIXTURES))
            entry = case(path, work, label)
            entry["detectors"] = outcome(lambda path=path: detectors(path))
            entry["forensic"] = outcome(lambda path=path: forensic_summary(path))
            entry = norm(entry, work)
            if synthetic:
                entry["input_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            else:
                entry["fixture"] = label
            cases.append(entry)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return cases


def main() -> None:
    suite = sys.argv[1] if len(sys.argv) > 1 else "media"
    OUT.mkdir(parents=True, exist_ok=True)
    if suite in ("media", "all"):
        cases = media_cases()
        (OUT / "media.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(cases)} media cases")
    if suite in ("images", "all"):
        cases = image_cases()
        (OUT / "images.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(cases)} image cases")


if __name__ == "__main__":
    main()

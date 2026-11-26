"""Golden deflate streams: Python's zlib, Pillow's PNG IDAT, and zlib-ng's medium strategy.

Run from the repository root: `.venv/bin/python rs-ai-watermarks/tools/goldens/metadata_deflate.py`.
Inputs come from a 64-bit LCG that the Rust test reimplements, so only hashes are stored.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import struct
import zlib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "rs-ai-watermarks" / "goldens" / "metadata" / "deflate.json"
MASK = (1 << 64) - 1
ALPHABET = b'the quick brown fox jumps over a lazy dog AIGC {}":,0123456789\n'
LIBZ_NG = Path("/usr/lib/libz-ng.so.2")


class Lcg:
    def __init__(self, seed: int) -> None:
        self.x = seed & MASK

    def next(self) -> int:
        self.x = (self.x * 6364136223846793005 + 1442695040888963407) & MASK
        return self.x >> 56


def gen(kind: str, n: int, seed: int) -> bytes:
    rng = Lcg(seed)
    out = bytearray()
    if kind == "zeros":
        return bytes(n)
    if kind == "noise":
        return bytes(rng.next() for _ in range(n))
    while len(out) < n:
        if kind == "runs":
            value = rng.next()
            out += bytes([value]) * (1 + rng.next() % 64)
        elif rng.next() < 64 and len(out) > 16:
            start = (rng.next() * len(out)) >> 8
            length = 3 + rng.next() % 40
            out += out[start : start + length]
        else:
            out.append(ALPHABET[rng.next() % len(ALPHABET)])
    return bytes(out[:n])


BANDS = {"1": 1, "L": 1, "LA": 2, "P": 1, "RGB": 3, "RGBA": 4}


def pixels(mode: str, width: int, height: int, style: str, seed: int) -> bytes:
    """Raw rows in Pillow's rawmode (1-bit rows packed MSB first, pad bits zero)."""
    rng = Lcg(seed)
    if mode == "1":
        row_bytes = (width + 7) // 8
        out = bytearray()
        for _ in range(height):
            row = bytearray(rng.next() for _ in range(row_bytes))
            if width % 8:
                row[-1] &= (0xFF << (8 - width % 8)) & 0xFF
            out += row
        return bytes(out)
    bands = BANDS[mode]
    out = bytearray()
    for y in range(height):
        for x in range(width):
            for band in range(bands):
                if style == "noise":
                    value = rng.next()
                elif style == "flat":
                    value = ((x // 16 + y // 16) % 4) * 60 + band * 10
                else:
                    value = ((x * 7 + y * 3) // 4 + band * 40 + (rng.next() & 7)) & 0xFF
                out.append(value)
    return bytes(out)


def idat_chunks(png: bytes) -> list[bytes]:
    chunks, pos = [], 8
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        kind = png[pos + 4 : pos + 8]
        if kind == b"IDAT":
            chunks.append(png[pos + 8 : pos + 8 + length])
        pos += 12 + length
    return chunks


def digest(chunks: list[bytes]) -> dict[str, object]:
    joined = b"".join(chunks)
    return {"sha256": hashlib.sha256(joined).hexdigest(), "len": len(joined), "chunks": [len(c) for c in chunks]}


class ZStream(ctypes.Structure):
    _fields_ = [
        ("next_in", ctypes.c_void_p),
        ("avail_in", ctypes.c_uint32),
        ("total_in", ctypes.c_size_t),
        ("next_out", ctypes.c_void_p),
        ("avail_out", ctypes.c_uint32),
        ("total_out", ctypes.c_size_t),
        ("msg", ctypes.c_char_p),
        ("state", ctypes.c_void_p),
        ("zalloc", ctypes.c_void_p),
        ("zfree", ctypes.c_void_p),
        ("opaque", ctypes.c_void_p),
        ("data_type", ctypes.c_int),
        ("adler", ctypes.c_uint32),
        ("reserved", ctypes.c_ulong),
    ]


def ng_zip_encode(lib: ctypes.CDLL, rows: list[bytes], strategy: int, bufsize: int) -> list[bytes]:
    """Pillow's ZipEncode.c call pattern on zlib-ng's native API."""
    strm = ZStream()
    if lib.zng_deflateInit2(ctypes.byref(strm), 6, 8, 15, 9, strategy) != 0:
        raise RuntimeError("zng_deflateInit2 failed")
    chunks: list[bytes] = []
    keep = None
    y, finishing = 0, False
    while True:
        out = ctypes.create_string_buffer(bufsize)
        strm.next_out = ctypes.addressof(out)
        strm.avail_out = bufsize
        if strm.next_in and strm.avail_in > 0:
            lib.zng_deflate(ctypes.byref(strm), 0)
        done = False
        if not finishing:
            while strm.avail_out > 0:
                if y >= len(rows):
                    finishing = True
                    break
                keep = ctypes.create_string_buffer(rows[y], len(rows[y]))
                strm.next_in = ctypes.addressof(keep)
                strm.avail_in = len(rows[y])
                y += 1
                lib.zng_deflate(ctypes.byref(strm), 0)
        if finishing and strm.avail_out > 0:
            while strm.avail_out > 0:
                if lib.zng_deflate(ctypes.byref(strm), 4) == 1:
                    done = True
                    break
        chunks.append(out.raw[: bufsize - strm.avail_out])
        if done:
            lib.zng_deflateEnd(ctypes.byref(strm))
            return chunks


IMAGES = [
    ("RGB", 64, 64, "smooth", 1),
    ("RGB", 600, 400, "noise", 2),
    ("RGBA", 300, 200, "smooth", 3),
    ("L", 257, 129, "smooth", 4),
    ("LA", 100, 50, "noise", 5),
    ("P", 200, 150, "flat", 6),
    ("1", 99, 77, "noise", 7),
    ("RGB", 20000, 3, "smooth", 8),
    ("RGB", 1, 1, "smooth", 9),
    ("RGB", 1, 300, "noise", 10),
    ("RGB", 512, 512, "flat", 11),
    ("RGB", 1100, 900, "smooth", 12),
    ("L", 3000, 40, "flat", 13),
]
STREAMS = [(kind, n) for kind in ("noise", "runs", "text", "zeros") for n in (0, 1, 2, 3, 4, 7, 100, 1000, 65535, 65536, 70000, 300000, 1200000)]
LEVELS = [(kind, 150000, level) for kind in ("text", "runs") for level in (4, 5, 7, 8, 9)]


def main() -> None:
    lib = ctypes.CDLL(str(LIBZ_NG)) if LIBZ_NG.exists() else None
    images = []
    for mode, width, height, style, seed in IMAGES:
        raw = pixels(mode, width, height, style, seed)
        image = Image.frombytes(mode, (width, height), raw)
        if mode == "P":
            image.putpalette(list(range(256)) * 3)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        chunks = idat_chunks(buffer.getvalue())
        bufsize = max(65536, width * 4)
        case: dict[str, object] = {
            "mode": mode,
            "width": width,
            "height": height,
            "style": style,
            "seed": seed,
            "bufsize": bufsize,
            "classic": digest(chunks),
        }
        if lib is not None:
            filtered = zlib.decompress(b"".join(chunks))
            row = len(filtered) // height
            rows = [filtered[i : i + row] for i in range(0, len(filtered), row)]
            strategy = 0 if mode == "P" else 1
            case["ng"] = digest(ng_zip_encode(lib, rows, strategy, bufsize))
        images.append(case)
    streams = [
        {"kind": kind, "n": n, "seed": n + 17, "level": 6, **digest([zlib.compress(gen(kind, n, n + 17))])}
        for kind, n in STREAMS
    ]
    streams += [
        {"kind": kind, "n": n, "seed": level, "level": level, **digest([zlib.compress(gen(kind, n, level), level)])}
        for kind, n, level in LEVELS
    ]
    for stream in streams:
        del stream["chunks"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    document = {"zlib": zlib.ZLIB_RUNTIME_VERSION, "images": images, "streams": streams}
    OUT.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(images)} images and {len(streams)} streams to {OUT}")


if __name__ == "__main__":
    main()

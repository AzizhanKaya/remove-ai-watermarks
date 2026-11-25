"""Golden generator for raiw-text (suite: text).

Run from the repository root:
    .venv/bin/python rs-ai-watermarks/tools/goldens/text.py

Inputs come from three places: every string argument the Python text tests pass to the
tracked functions (harvested by running those tests with recording wrappers), the tracked
fidelity cases in data/evaluations/text-watermark/, and synthetic edge cases below. Outputs
are always recomputed afterwards with the pristine, unpatched implementation.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import functools
import io
import json
import re
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "rs-ai-watermarks" / "goldens" / "text"
sys.path.insert(0, str(ROOT / "src"))

from remove_ai_watermarks import text as T  # noqa: E402

TEXT_FUNCTIONS = (
    "identify_text",
    "remove_text_metadata",
    "split_protected_text_spans",
    "decode_stego_payloads",
    "encode_c2patxt",
    "encode_c2pa_ascii_armor",
    "encode_vsrmark",
    "encode_unicode_tags",
    "encode_innamark",
    "verify_text_rewrite_facts",
    "_mask_local_rewrite_facts",
    "_guard_local_rewrite_candidate",
    "_preserves_dominant_script",
    "_preserves_rewrite_extent",
    "_preserves_rewrite_language",
    "_classify_text_language",
    "_normalize_numeral_token",
    "_normalize_english_integer",
    "_normalize_russian_integer",
    "score_gumbel_text",
    "score_textgrain_text",
    "score_textseal2_text",
    "score_kgw_text",
    "score_unigram_text",
    "score_synthid_threefry_text",
)
API_FUNCTIONS = ("_sentence_slots",)
TEST_GLOBS = ("test_text_*.py", "test_windows_text_watermark.py")
SKIPPED_TESTS = {
    "test_text_mark_engine.py",
    "test_text_mark_faint_mask.py",
    "test_text_mark_memory.py",
    "test_text_mark_variants.py",
    "test_text_draft.py",
    "test_text_restoration.py",
    "test_text_benchmark.py",
    "test_text_fidelity_benchmark.py",
    "test_text_paired_benchmark.py",
}

RECORDED: dict[str, dict[str, tuple[list, dict]]] = {}


def _plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    raise TypeError(type(value).__name__)


def _has_surrogate(value) -> bool:
    return isinstance(value, str) and any(0xD800 <= ord(ch) <= 0xDFFF for ch in value)


def _record(name: str, args: tuple, kwargs: dict) -> None:
    try:
        plain_args, plain_kwargs = _plain(list(args)), _plain(kwargs)
    except TypeError:
        return
    if any(_has_surrogate(item) for item in [*args, *kwargs.values()]):
        return
    key = json.dumps([plain_args, plain_kwargs], sort_keys=True, ensure_ascii=False)
    RECORDED.setdefault(name, {}).setdefault(key, (list(args), dict(kwargs)))


def _tracker(name: str, function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        _record(name, args, kwargs)
        return function(*args, **kwargs)

    return wrapper


# ── Minimal pytest stand-in: the shared environment has no pytest, and harvesting only
# needs the tests to call the tracked functions, not to judge their assertions. ──


class _Skip(Exception):
    pass


class _Raises:
    def __init__(self, expected, match=None):
        self.expected = expected
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, kind, value, _traceback):
        self.value = value
        return kind is not None and issubclass(kind, self.expected)


class _Approx:
    def __init__(self, expected, rel=None, abs=None):  # noqa: A002 - pytest signature
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        import math

        return math.isclose(other, self.expected, rel_tol=self.rel or 1e-6, abs_tol=self.abs or 1e-12)


class _MonkeyPatch:
    _missing = object()

    def __init__(self):
        self.undo_stack = []

    def setattr(self, target, name, value=_missing, raising=True):
        self.undo_stack.append(("attr", target, name, getattr(target, name, self._missing)))
        setattr(target, name, value)

    def delattr(self, target, name, raising=True):
        self.undo_stack.append(("attr", target, name, getattr(target, name, self._missing)))
        with contextlib.suppress(AttributeError):
            delattr(target, name)

    def setitem(self, mapping, key, value):
        self.undo_stack.append(("item", mapping, key, mapping.get(key, self._missing)))
        mapping[key] = value

    def delitem(self, mapping, key, raising=True):
        self.undo_stack.append(("item", mapping, key, mapping.get(key, self._missing)))
        mapping.pop(key, None)

    def setenv(self, name, value, prepend=None):
        import os

        self.setitem(os.environ, name, str(value))

    def delenv(self, name, raising=True):
        import os

        self.delitem(os.environ, name)

    def undo(self):
        for kind, target, name, old in reversed(self.undo_stack):
            if kind == "attr":
                if old is self._missing:
                    with contextlib.suppress(AttributeError):
                        delattr(target, name)
                else:
                    setattr(target, name, old)
            elif old is self._missing:
                target.pop(name, None)
            else:
                target[name] = old
        self.undo_stack.clear()


class _Caplog:
    def __init__(self):
        import logging

        self.records = []
        self.handler = logging.Handler()
        self.handler.emit = self.records.append
        logging.getLogger().addHandler(self.handler)

    def set_level(self, level, logger=None):
        import logging

        logging.getLogger(logger).setLevel(level)

    @property
    def text(self):
        return "\n".join(record.getMessage() for record in self.records)

    def close(self):
        import logging

        logging.getLogger().removeHandler(self.handler)


def _install_pytest_shim() -> None:
    import types

    shim = types.ModuleType("pytest")

    def parametrize(names, values, ids=None, **_kwargs):
        def decorate(function):
            function.__dict__.setdefault("_params", []).append((names, list(values)))
            return function

        return decorate

    def fixture(function=None, **_kwargs):
        def mark(inner):
            inner._fixture = True
            return inner

        return mark(function) if function is not None else mark

    def importorskip(name, *_args, **_kwargs):
        import importlib

        try:
            return importlib.import_module(name)
        except ImportError as error:
            raise _Skip(name) from error

    def fail(message="", pytrace=True):
        raise AssertionError(message)

    shim.mark = types.SimpleNamespace(parametrize=parametrize, skipif=lambda *a, **k: (lambda f: f), slow=lambda f: f)
    shim.fixture = fixture
    shim.raises = _Raises
    shim.approx = _Approx
    shim.fail = fail
    shim.skip = lambda *a, **k: (_ for _ in ()).throw(_Skip())
    shim.importorskip = importorskip
    shim.MonkeyPatch = _MonkeyPatch
    shim.LogCaptureFixture = _Caplog
    sys.modules["pytest"] = shim


def _expand(function):
    import itertools

    layers = []
    for names, values in reversed(function.__dict__.get("_params", [])):
        names = [name.strip() for name in names.split(",")] if isinstance(names, str) else list(names)
        layers.append([dict(zip(names, value if len(names) > 1 else (value,), strict=True)) for value in values])
    for combination in itertools.product(*layers) if layers else [()]:
        merged = {}
        for part in combination:
            merged.update(part)
        yield merged


def _run_test(module, function, params) -> None:
    import inspect
    import tempfile

    cleanups = []
    monkeypatch = _MonkeyPatch()

    def resolve(name):
        if name in params:
            return params[name]
        if name == "tmp_path":
            return Path(tempfile.mkdtemp(prefix="raiw-golden-"))
        if name == "monkeypatch":
            return monkeypatch
        if name == "caplog":
            caplog = _Caplog()
            cleanups.append(caplog.close)
            return caplog
        provider = getattr(module, name, None)
        if callable(provider) and getattr(provider, "_fixture", False):
            kwargs = {arg: resolve(arg) for arg in inspect.signature(provider).parameters}
            value = provider(**kwargs)
            if inspect.isgenerator(value):
                generator = value
                value = next(generator)
                cleanups.append(lambda: next(generator, None))
            return value
        raise _Skip(name)

    try:
        kwargs = {name: resolve(name) for name in inspect.signature(function).parameters}
        function(**kwargs)
    except BaseException:  # noqa: BLE001 - harvesting ignores verdicts
        pass
    finally:
        monkeypatch.undo()
        for cleanup in reversed(cleanups):
            with contextlib.suppress(BaseException):
                cleanup()


def _run_test_files(files: list[Path]) -> None:
    import importlib.util

    sys.path.insert(0, str(ROOT / "tests"))
    for path in files:
        spec = importlib.util.spec_from_file_location(f"golden_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except BaseException:  # noqa: BLE001 - an unimportable module only loses its inputs
            continue
        for name, value in vars(module).items():
            if name.startswith("test_") and callable(value):
                for params in _expand(value):
                    _run_test(module, value, params)


def harvest() -> None:
    _install_pytest_shim()
    originals = {name: getattr(T, name) for name in TEXT_FUNCTIONS}
    for name, function in originals.items():
        setattr(T, name, _tracker(name, function))
    from remove_ai_watermarks import text_api as A

    api_originals = {name: getattr(A, name) for name in API_FUNCTIONS}
    for name, function in api_originals.items():
        setattr(A, name, _tracker(name, function))
    files = sorted(
        {path for pattern in TEST_GLOBS for path in (ROOT / "tests").glob(pattern) if path.name not in SKIPPED_TESTS}
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _run_test_files(files)
    for name, function in originals.items():
        setattr(T, name, function)
    for name, function in api_originals.items():
        setattr(A, name, function)


def norm(value):
    """JSON-normalize sets, frozensets, tuple-keyed dicts and dataclasses deterministically."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: norm(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted((norm(item) for item in value), key=lambda item: json.dumps(item, ensure_ascii=False))
    if isinstance(value, (list, tuple)):
        return [norm(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: norm(item) for key, item in value.items()}
        pairs = [[norm(key), norm(item)] for key, item in value.items()]
        return sorted(pairs, key=lambda pair: json.dumps(pair[0], ensure_ascii=False))
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    return value


def write(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def recorded_strings(name: str, position: int = 0) -> list[str]:
    values = []
    for args, kwargs in RECORDED.get(name, {}).values():
        if len(args) > position and isinstance(args[position], str) and not kwargs:
            values.append(args[position])
    return values


def unique(items):
    return list(dict.fromkeys(items))


SYNTHETIC_JUMBF = (
    b"jumbc2pa\x00\x00\x00\x01"
    b"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    b"OpenAI ChatGPT GPT-6 Astra com.cognitive-proof.vsrmark.1"
    b"20261109120000Z"
)


def synthetic_documents() -> list[str]:
    england = "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    vs = T.encode_variation_selectors
    header = struct.pack("!8sBI", b"C2PATXT\x00", 1, len(SYNTHETIC_JUMBF))
    bad_body = struct.pack(">4sBH", b"VSR1", 1, 4) + b"test"
    bad_frame = bad_body + struct.pack(">I", (zlib.crc32(bad_body) ^ 0xFFFFFFFF) & 0xFFFFFFFF)
    carrier = "This carrier sentence has enough visible characters to hold a VSRMark claim."
    cover = " ".join(f"word{i}" for i in range(120))
    b64 = base64.b64encode(SYNTHETIC_JUMBF).decode("ascii")
    subtext_frame = b"\xab\xcd" + (4).to_bytes(4, "big") + b"\x01\x00" + b"\x01hi!"
    gt_bits = f"{0xCAFEBABE:032b}{0x12345678:032b}"
    docs = [
        "",
        " ",
        "\n",
        "\r\n\r\n",
        "﻿",
        "﻿BOM-prefixed document.\r\nSecond line.\r\n",
        "Plain ASCII prose without any marks.",
        "Mixed scripts: English, русский, ελληνικά, العربية, עברית, 日本語, 한국어, हिन्दी.",
        "Astral \U0001d518\U0001d52b\U0001d526\U0001d520\U0001d52c\U0001d521\U0001d522 and 𝟘𝟙𝟚 digits.",
        "CR only line\rendings\rhere\r",
        T.encode_c2patxt("Generated analysis paragraph with ❤️ and 👩‍💻 emoji.\n", SYNTHETIC_JUMBF, padding_bytes=8),
        T.encode_c2patxt("Text", b""),
        T.encode_c2patxt("Body\r\n", SYNTHETIC_JUMBF) + "\r\n",
        "Plain text body." + vs(header + SYNTHETIC_JUMBF),
        "Short " + vs(b"C2PATXT"),
        T.encode_c2pa_ascii_armor(SYNTHETIC_JUMBF, comment_style="html") + "# Heading\n\nBody paragraph.\n",
        T.encode_c2pa_ascii_armor("https://example.com/manifests/123.c2pa", comment_style="hash") + "Body\n",
        T.encode_c2pa_ascii_armor("data:application/c2pa;base64,!!!") + "Body\n",
        "Intro\n" + T.encode_c2pa_ascii_armor(SYNTHETIC_JUMBF) + "Outro\n",
        "```text\n-----BEGIN C2PA MANIFEST-----\nhttps://example.com/x\n-----END C2PA MANIFEST-----\n```\n",
        "<!doctype html>\n<html>\n<head>\n  <title>Test</title>\n"
        f'  <script type="application/c2pa">data:application/c2pa;base64,{b64}</script>\n'
        '  <link rel="c2pa-manifest" href="https://example.com/manifest.c2pa">\n'
        "</head>\n<body><p>Hello world.</p></body>\n</html>\n",
        '<html><head></head><body><link rel="c2pa-manifest" href="https://example.com/m.c2pa"></body></html>',
        f"<html><head>é<script type='application/c2pa'>{b64}</script></head></html>",
        T.encode_vsrmark(carrier, '{"claim":"claude-fable-5-2","id":"9f8a"}'),
        T.encode_unicode_tags("", "pud") + T.encode_vsrmark(carrier, "x"),
        carrier[:15] + vs(bad_frame) + carrier[15:],
        T.encode_vsrmark("Use 5 kg of flour and the ❤️ emoji carefully in this sentence.", "q"),
        T.encode_unicode_tags("Hello​‌ world ❤️ 👩‍💻!", "ASTRA-TEXTGRAIN"),
        f"Flag {england}, emoji 👨🏻‍❤️‍👨🏻, and Persian می‌شود.",
        f"{england}\U000e0078 ا‌ب ‮alert passw&#x200B;оrd &ZeroWidthSpace; `code &#x200B; passwоrd` пароль",
        "пaрoль cоус pａssword",
        "漢字葛\U000e0100と100 km\nمرحبا ⁧عالم⁩ جميل",
        T.encode_innamark(cover, "author:astra-6", compressed=False, crc32=True),
        T.encode_innamark(cover, "author:astra-6", compressed=False, crc32=False),
        T.encode_innamark(cover, "author:astra-6", compressed=True, crc32=True),
        "Normal ‌⁡⁡‍⁣⁡‍‍‌ sentence.",
        "First line\n" + "".join("⁡" if bit == "0" else "⁢" for bit in gt_bits) + "Second line.",
        "Carrier" + vs(subtext_frame[:6]) + " " + vs(subtext_frame[6:]) + " text.",
        "Emoji" + "".join("︎" if bit == "0" else "️" for bit in "".join(f"{b:08b}" for b in b"\x00AI")) + " carrier.",
        "Hidden" + vs("emoji-encoder payload".encode()) + " text.",
        "Zero" + "".join("​" if bit == "0" else "‌" for bit in "".join(f"{b:08b}" for b in b"hi")) + "width.",
        "Delimited " + "‍".join("".join("​" if bit == "0" else "‌" for bit in f"{ord(c):b}") for c in "AI") + " zw.",
        "See the result citeturn0search0 and continue.",
        "Read\tciteturn0search0 citeturn1view2\tthen proceed.",
        "Wait citeturn2view0…",
        "Gemini [cite_start]claim[cite: 1, 2] and [span_3](start_span) grok_render_citation_card_json={\"id\":1} done.",
        "ChatGPT citeturn0search0 marker.",
        "Hello​͏⁠​ world.",
        "​͏⁠​Hello​͏⁠​",
        "👩​͏⁠​‍💻",
        "Keep these two words together and 10 000 units « bonjour ».",
        "Thin spaces and　ideographic gaps.",
        "Glyph ❤\U000e0001\U000e0070\U000e0075\U000e0064\U000e007f️.",
        "Glyph 2⁠×⁠3 and 4⁠÷⁠2.",
        "Thai ก​ข word break and `code​space` literal.",
        "Soft hy­phen and co­‑op and x­​y.",
        "Hebrew אָ͏ַ and plain a͏b.",
        "Devanagari क्‍ and Malayalam ണ്‍. Saurashtra ꢒ꣄‍ꢓ.",
        "Mongolian ᠠ᠋ and Khmer ក឴ and Hangul ᄀᅠ.",
        "PUA  and noncharacter ﷐ and \U000e0080 reserved.",
        "Bidi ‫שלום‬ and ⁦text⁩ and ‏עברית.",
        "Entities &#8203; &#x200C; &zwj; &shy; &amp; &#65;.",
        "Fullwidth ＡＢＣ and Greek Αθήνα and Cherokee ᎠᎡ.",
        "# Title\n\n| a | b |\n\n[ref]: https://example.com\n\n1. item\n- other\n",
        "Math $x$ and $5 and $10, $$\\sum_i x_i$$ and \\(a+b\\) and \\[c\\].",
        "<script>alert(1)</script> and <pre>code</pre> and <prefix> tag.",
        "Path ./src/runner.py and ../a/b.txt and /usr/bin/env.sh here.",
        "URL https://raiw.cc/docs/spec) and http://x.y]z.",
        "~~~\ncode\n~~~\nAfter fence\n\n```\nunterminated",
        "Tabs\tand\x1cfile\x1dseparators\x1e\x1f.",
        "Line separator paragraph\x85next.",
    ]
    return docs


def synthetic_pairs() -> list[tuple[str, str]]:
    return [
        ("The cluster processed forty requests per second.", "Throughput reached 40 requests every second."),
        ("We shipped one hundred and twenty-five crates.", "We shipped 125 crates."),
        ("We shipped two thousand five hundred crates.", "We shipped 2,500 crates."),
        ("Он купил двести тридцать пять книг.", "Он приобрёл 235 книг."),
        ("Пятилетняя программа завершилась.", "Программа на пять лет завершилась."),
        ("The tank holds 5 L of water.", "The tank holds 5 liters of water."),
        ("It takes at least 3 hours.", "It takes no less than 3 hours."),
        ("Revenue rose to 5%.", "Revenue rose by 5%."),
        ("Take the pill twice a day.", "Take the pill two times daily."),
        ("We meet every other week.", "We meet biweekly."),
        ("The value is 1.5e3 units.", "The value is 1500 units."),
        ("Values: 1,250.50 and ١٢٣.", "Values: 1250.5 and 123."),
        ("Dr. Jane Q. Smith visited NASA.", "Smith visited the agency."),
        ("Maria Lopez, a nurse, arrived early.", "A nurse arrived early."),
        ("The system allows users to record notes.", "The system enables users to record notes."),
        ("Users must register.", "Users may register."),
        ("Every type of soil was tested.", "Various soils were tested."),
        ("From spring to autumn the park is open.", "In summer the park is open."),
        ("", ""),
        ("Same text.", "Same text."),
    ]


def build_cases() -> dict[str, list]:
    fidelity = json.loads((ROOT / "data/evaluations/text-watermark/fidelity-cases.json").read_text(encoding="utf-8"))
    fidelity_pairs = [(case["source"], case["candidate"]) for case in fidelity["cases"]]
    documents = unique([*recorded_strings("identify_text"), *recorded_strings("remove_text_metadata"), *synthetic_documents()])
    span_texts = unique([*recorded_strings("split_protected_text_spans"), *documents])
    pairs = unique(
        [
            *[
                (args[0], args[1])
                for args, kwargs in RECORDED.get("verify_text_rewrite_facts", {}).values()
                if len(args) == 2 and not kwargs
            ],
            *fidelity_pairs,
            *synthetic_pairs(),
        ]
    )
    prose = unique([text for pair in pairs for text in pair])
    return {"documents": documents, "span_texts": span_texts, "pairs": pairs, "prose": prose}


def identify_cases(documents: list[str]) -> list[dict]:
    rows = []
    for text in documents:
        report = T.identify_text(text)
        metadata = T.remove_text_metadata(text)
        rows.append(
            {
                "text": text,
                "report": norm(report),
                "json": json.dumps(dataclasses.asdict(report), default=str, indent=2),
                "metadata": norm(metadata),
                "stego": T.decode_stego_payloads(text),
            }
        )
    return rows


def span_cases(texts: list[str]) -> list[dict]:
    return [{"text": text, "spans": [[span.kind, span.text, span.locked] for span in T.split_protected_text_spans(text)]} for text in texts]


def encoder_cases() -> dict[str, list]:
    carrier = "This carrier sentence has enough visible characters to hold a VSRMark claim."
    rows: dict[str, list] = {"c2patxt": [], "armor": [], "vsrmark": [], "tags": [], "innamark": [], "errors": []}
    for text, manifest, padding in [
        ("Hello", SYNTHETIC_JUMBF, 0),
        ("é decomposed", b"x", 3),
        (T.encode_c2patxt("Old", b"old"), b"new", 0),
        ("", b"", 0),
    ]:
        rows["c2patxt"].append({"text": text, "manifest": manifest.hex(), "padding": padding, "out": T.encode_c2patxt(text, manifest, padding_bytes=padding)})
    for source, style in [(SYNTHETIC_JUMBF, "none"), (SYNTHETIC_JUMBF, "html"), (SYNTHETIC_JUMBF, "hash"), ("  https://e.x/m  ", "hash"), ("urn:c2pa:1", "none")]:
        kind = "bytes" if isinstance(source, bytes) else "uri"
        value = source.hex() if isinstance(source, bytes) else source
        rows["armor"].append({"kind": kind, "value": value, "style": style, "out": T.encode_c2pa_ascii_armor(source, comment_style=style)})
    for text, payload in [(carrier, '{"claim":"x"}'), (carrier + " ​️ extra", "é"), ("a" * 40, "")]:
        rows["vsrmark"].append({"text": text, "payload": payload, "out": T.encode_vsrmark(text, payload)})
    for text, hidden in [("Visible", "hidden"), ("", ""), ("x", "~ !")]:
        rows["tags"].append({"text": text, "hidden": hidden, "out": T.encode_unicode_tags(text, hidden)})
    cover = " ".join(f"w{i}" for i in range(200))
    for payload, crc in [("author:astra-6", True), ("plain", False), ("ünïcode", True)]:
        rows["innamark"].append({"text": cover, "payload": payload, "crc32": crc, "out": T.encode_innamark(cover, payload, compressed=False, crc32=crc)})
    for label, call in [
        ("vsrmark_short", lambda: T.encode_vsrmark("short", "payload")),
        ("tags_non_ascii", lambda: T.encode_unicode_tags("x", "é")),
        ("tags_control", lambda: T.encode_unicode_tags("x", "\n")),
        ("innamark_short", lambda: T.encode_innamark("a b", "payload")),
    ]:
        try:
            call()
        except ValueError as error:
            rows["errors"].append({"case": label, "message": str(error)})
    return rows


EXTRACTORS = (
    "_extract_frequencies",
    "_extract_numeral_occurrences",
    "_extract_measurements",
    "_extract_numeric_bounds",
    "_extract_mid_sentence_proper_nouns",
    "_selected_name_phrases",
    "_extract_rewrite_constraints",
    "_extract_explicit_record_distribution",
    "_extract_explicit_signal_directions",
    "_extract_explicit_membership_stances",
    "_explicit_type_coverage_count",
    "_extract_handover_status",
    "_extract_event_phases",
    "_extract_influence_qualifications",
    "_extract_coordinated_outcomes",
    "_extract_vague_durations",
    "_extract_weekend_timing",
    "_explicit_seasons",
    "_has_active_negation",
    "_extract_explicit_percent_change_roles",
    "_sentence_content_words",
)
PAIR_CHECKS = (
    "verify_text_rewrite_facts",
    "_preserves_rewrite_numerals",
    "_preserves_rewrite_negation",
    "_constraint_texts_for_capability_frames",
    "_changes_explicit_expected_action",
    "_changes_actor_apposition",
    "_changes_influence_qualification",
    "_changes_coordinated_outcome_qualification",
    "_preserves_explicit_seasons",
    "_preserves_explicit_percent_change_roles",
    "_preserves_dominant_script",
    "_preserves_rewrite_extent",
)


def fact_cases(prose: list[str], pairs: list[tuple[str, str]]) -> dict[str, list]:
    extracts = [{"text": text, **{name: norm(getattr(T, name)(text)) for name in EXTRACTORS}} for text in prose]
    checks = [{"source": s, "candidate": c, **{name: norm(getattr(T, name)(s, c)) for name in PAIR_CHECKS}} for s, c in pairs]
    masks = []
    for text in unique([*prose, *recorded_strings("_mask_local_rewrite_facts")]):
        masked, replacements = T._mask_local_rewrite_facts(text)
        masks.append({"text": text, "masked": masked, "replacements": replacements})
    normalizers = {
        name: [{"token": token, "out": getattr(T, name)(token)} for token in unique(recorded_strings(name))]
        for name in ("_normalize_numeral_token", "_normalize_english_integer", "_normalize_russian_integer")
    }
    normalizers["_normalize_numeral_token"] += [
        {"token": token, "out": T._normalize_numeral_token(token)}
        for token in ["1e999999999999999999", "1e1000000000000000000", "0.001e-999999999999999999", "1e-1999999999999999998", "1.2.3", "١٢٣", "−5", "+.5", "007", "0.00", "1e-5000", "12,345.6789e-2", "1,23"]
    ]
    return {"extracts": extracts, "checks": checks, "masks": masks, "normalizers": normalizers}


def language_cases(prose: list[str], pairs: list[tuple[str, str]]) -> dict[str, list]:
    texts = unique([*prose, *recorded_strings("_classify_text_language")])
    classify = []
    for text in texts:
        label, score = T._classify_text_language(text)
        classify.append({"text": text, "label": label, "score": score})
    preserves = [{"source": s, "candidate": c, "out": T._preserves_rewrite_language(s, c)} for s, c in pairs]
    guards = []
    for args, kwargs in RECORDED.get("_guard_local_rewrite_candidate", {}).values():
        if kwargs or len(args) != 5 or args[4] is not None or not isinstance(args[2], dict):
            continue
        guards.append({"passage": args[0], "masked": args[1], "facts": args[2], "candidate": args[3], "out": T._guard_local_rewrite_candidate(*args)})
    for passage in prose[:120]:
        masked, facts = T._mask_local_rewrite_facts(passage)
        candidate = masked.replace("the", "this", 1)
        guards.append({"passage": passage, "masked": masked, "facts": facts, "candidate": candidate, "out": T._guard_local_rewrite_candidate(passage, masked, facts, candidate, None)})
    return {"classify": classify, "preserves": preserves, "guards": guards}


def score_cases() -> dict[str, list]:
    base = [((i * 1103515245 + 12345) >> 16) % 4990 + 10 for i in range(240)]
    sequences = [base, base[:5], [], [1, 2, 3, 1, 2, 3, 1, 2, 3], list(range(-5, 40))]
    for args, _kwargs in RECORDED.get("score_gumbel_text", {}).values():
        if args and isinstance(args[0], list) and len(sequences) < 40:
            sequences.append(args[0])
    rows = []
    for tokens in sequences:
        for key in (99991, 424242, -7):
            row = {"tokens": tokens, "key": key}
            row["gumbel"] = norm(T.score_gumbel_text(tokens, key))
            row["gumbel_ctx1"] = norm(T.score_gumbel_text(tokens, key, context_len=1))
            row["textgrain"] = norm(T.score_textgrain_text(tokens, key))
            row["textgrain_small"] = norm(T.score_textgrain_text(tokens, key, num_blocks=3, num_columns=5))
            row["textseal2"] = norm(T.score_textseal2_text(tokens, key))
            row["textseal2_b"] = norm(T.score_textseal2_text(tokens, key, key + 1))
            row["kgw"] = norm(T.score_kgw_text(tokens, key))
            row["kgw_wide"] = norm(T.score_kgw_text(tokens, key, context_width=3, greenlist_ratio=0.5))
            row["unigram"] = norm(T.score_unigram_text(tokens, key))
            rows.append(row)
    threefry = []
    for ids, key, ngram, depth, history, eos in [
        ([7, 44, 3, 98, 22, 5, 5, 17, 43, 2, 19, 17], 654, 5, 30, 1024, ()),
        ([2, 3, 9, 2, 3, 9, 2, 3, 9, 2, 3, 1], 655, 3, 65, 2, (1,)),
        ([77, 3, 1, 644, 49, 23, 5, 9, 2, 62, 19, 4], (1 << 47) + 32768, 2, 129, 5, (2, 0)),
        ([6, 7, 8, 9, 10, 300, 1, 2, 3], 0, 3, 64, 2, (6,)),
        (base, 123456789, 5, 30, 1024, ()),
        (base * 4, (1 << 63) - 1, 4, 200, 64, ()),
    ]:
        result = T.score_synthid_threefry_text(ids, key, ngram_len=ngram, depth=depth, context_history_size=history, eos_token_ids=eos, alpha=0.5)
        threefry.append({"ids": ids, "key": key, "ngram_len": ngram, "depth": depth, "history": history, "eos": list(eos), "result": norm(result)})
    return {"blake2": rows, "threefry": threefry}


def api_cases(prose: list[str]) -> list[dict]:
    from remove_ai_watermarks import text_api as A

    texts = unique([*recorded_strings("_sentence_slots"), *prose, "Dr. Reed uses a ruler.\tThe block is 1.5 meters wide.", "Mr. A. B. Smith left. Then? Yes!  Done.\n\nNext"])
    return [{"text": text, "out": norm(A._sentence_slots(text))} for text in texts]


def pattern_sources() -> dict[str, str]:
    found = {name: value.pattern for name, value in vars(T).items() if isinstance(value, re.Pattern)}
    found.update({name: value for name, value in vars(T).items() if name.endswith("_PATTERN") and isinstance(value, str)})
    return found


def main() -> None:
    harvest()
    cases = build_cases()
    write("patterns", pattern_sources())
    write("identify", identify_cases(cases["documents"]))
    write("spans", span_cases(cases["span_texts"]))
    write("encoders", encoder_cases())
    write("facts", fact_cases(cases["prose"], cases["pairs"]))
    write("language", language_cases(cases["prose"], cases["pairs"]))
    write("scores", score_cases())
    write("sentence_slots", api_cases(cases["prose"]))
    counts = {name: len(calls) for name, calls in sorted(RECORDED.items())}
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()

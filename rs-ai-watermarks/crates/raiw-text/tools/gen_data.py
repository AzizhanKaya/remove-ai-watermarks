"""Regenerate raiw-text lookup tables from the Python reference environment.

Run from the repository root:
    .venv/bin/python rs-ai-watermarks/crates/raiw-text/tools/gen_data.py

Unicode properties come from the interpreter's ``unicodedata`` (CPython 3.12, Unicode 15.0),
so character classes, casefolding and regex shorthands match the Python reference exactly.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import unicodedata
from pathlib import Path

from remove_ai_watermarks._internal import c2pa
from remove_ai_watermarks._internal import constants as c
from remove_ai_watermarks._internal.emoji_flag_codes import EMOJI_FLAG_TAG_CODES
from remove_ai_watermarks._internal.standardized_variants import STANDARDIZED_VARIATION_BASES

DATA = Path(__file__).resolve().parents[1] / "data"

CATEGORIES = (
    "Cn Lu Ll Lt Lm Lo Mn Mc Me Nd Nl No Pc Pd Ps Pe Pi Pf Po Sm Sc Sk So Zs Zl Zp Cc Cf Cs Co"
).split()

# Flag bits stored per range.
RTL, LATIN, CYRILLIC, UPPER, LOWER, DIGIT, SPACE, NUMERIC = (1 << bit for bit in range(8))


def unicode_table() -> bytes:
    words: dict[str, int] = {"": 0}
    ranges: list[tuple[int, int, int, int, int]] = []
    previous = None
    for cp in range(sys.maxunicode + 1):
        ch = chr(cp)
        cat = CATEGORIES.index(unicodedata.category(ch))
        name = unicodedata.name(ch, "")
        flags = 0
        if unicodedata.bidirectional(ch) in ("R", "AL", "AN"):
            flags |= RTL
        if "LATIN" in name:
            flags |= LATIN
        if "CYRILLIC" in name:
            flags |= CYRILLIC
        if ch.isupper():
            flags |= UPPER
        if ch.islower():
            flags |= LOWER
        if ch.isdigit():
            flags |= DIGIT
        if ch.isspace():
            flags |= SPACE
        if ch.isnumeric():
            flags |= NUMERIC
        word = 0
        if ch.isalpha():
            first = unicodedata.name(ch, "UNKNOWN").split(" ", 1)[0]
            word = words.setdefault(first, len(words))
        record = (cat, flags, unicodedata.combining(ch), word)
        if record != previous:
            ranges.append((cp, *record))
            previous = record
    out = bytearray(b"RUP1")
    out += struct.pack("<II", len(ranges), len(words))
    for start, cat, flags, ccc, word in ranges:
        out += struct.pack("<IBBBxH", start, cat, flags, ccc, word)
    out += "\n".join(words).encode("utf-8")
    return bytes(out)


def casefold_table() -> str:
    lines = []
    for cp in range(sys.maxunicode + 1):
        ch = chr(cp)
        folded = ch.casefold()
        if folded != ch:
            lines.append(f"{cp:X} " + " ".join(f"{ord(x):X}" for x in folded))
    return "\n".join(lines) + "\n"


def variants_table() -> str:
    lines = []
    for selector in sorted(STANDARDIZED_VARIATION_BASES):
        bases = sorted(ord(base) for base in STANDARDIZED_VARIATION_BASES[selector])
        lines.append(f"{selector:X} " + " ".join(f"{base:X}" for base in bases))
    return "\n".join(lines) + "\n"


def flag_codes() -> str:
    return "\n".join(sorted(EMOJI_FLAG_TAG_CODES)) + "\n"


def c2pa_registry() -> dict[str, object]:
    def pairs(mapping: dict[bytes, str]) -> list[list[str]]:
        return [[token.decode("utf-8"), label] for token, label in mapping.items()]

    def entry(item: c.C2paSoftBindingAlgorithm) -> dict[str, object]:
        return {"algorithm": item.algorithm, "kind": item.kind, "display_label": item.display_label}

    return {
        "issuers": pairs(c.C2PA_ISSUERS),
        "ai_tools": pairs(c.C2PA_AI_TOOLS),
        "actions": pairs(c.C2PA_ACTIONS),
        "soft_bindings": pairs(c.C2PA_SOFT_BINDINGS),
        "soft_binding_registry": [entry(item) for item in c.C2PA_SOFT_BINDING_REGISTRY],
        "vendor_soft_binding_prefixes": [entry(item) for item in c.C2PA_VENDOR_SOFT_BINDING_PREFIXES],
        "vendors": [
            {
                "issuer": vendor.issuer.decode("utf-8"),
                "org": vendor.org,
                "platform": vendor.platform,
                "needle": vendor.needle,
                "synthid": vendor.synthid,
                "synthid_requires_watermark_action": vendor.synthid_requires_watermark_action,
            }
            for vendor in c.C2PA_AI_VENDORS
        ],
        "synthid_edit_signers": [signer.decode("utf-8") for signer in c.SYNTHID_EDIT_SIGNERS],
        "claim_generator_platforms": [list(pair) for pair in c.C2PA_CLAIM_GENERATOR_PLATFORMS],
        "ai_disclosure_levels": list(c2pa._AI_DISCLOSURE_LEVELS),
        "ai_disclosure_enhanced_levels": sorted(c2pa._AI_DISCLOSURE_ENHANCED_LEVELS),
    }


def main() -> None:
    assert unicodedata.unidata_version == "15.0.0", unicodedata.unidata_version
    assert re.compile(r"\w").pattern
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "unicode.bin").write_bytes(unicode_table())
    (DATA / "casefold.txt").write_text(casefold_table(), encoding="utf-8")
    (DATA / "variants.txt").write_text(variants_table(), encoding="utf-8")
    (DATA / "flag_codes.txt").write_text(flag_codes(), encoding="utf-8")
    registry = json.dumps(c2pa_registry(), ensure_ascii=False, indent=1) + "\n"
    (DATA / "c2pa_registry.json").write_text(registry, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Golden TC260 labels and signature verdicts from the Python reference.

Run from the repository root: `.venv/bin/python rs-ai-watermarks/tools/goldens/metadata_tc260.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

from remove_ai_watermarks._internal.tc260_signature import check_tc260_signature
from remove_ai_watermarks.metadata import aigc_label

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "rs-ai-watermarks" / "goldens" / "metadata" / "tc260_signatures.json"
SOURCES = sorted((ROOT / "data" / "fixtures" / "provenance").iterdir())


def verdict(label: dict[str, str]) -> dict[str, object] | None:
    result = check_tc260_signature(label)
    return None if result is None else {"label": result.label, "signer": result.signer, "describe": result.describe()}


def main() -> None:
    cases = []
    for path in SOURCES:
        label = aigc_label(path)
        if not label or "ReservedCode1" not in label:
            continue
        cases.append({"name": path.name, "label": label, "verdict": verdict(label)})
        produce_id = label.get("ProduceID", "")
        if produce_id:
            altered = {**label, "ProduceID": produce_id[:-1] + ("0" if produce_id[-1] != "0" else "1")}
            cases.append({"name": f"{path.name}#altered", "label": altered, "verdict": verdict(altered)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()

"""Conformance vectors for the reader's tokens and input.

The app's Dart code must reproduce every ``expected`` value in the committed file exactly.
Regenerate with ``akili reader-vectors`` after any change to :mod:`akili.reader.tokens` (which
must also bump ``READER_SCHEMA_VERSION``); a test fails if the file drifts from the code. All
messages are synthetic.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from akili.reader.tokens import (
    READER_SCHEMA_VERSION,
    encode,
    hash_key,
    reader_manifest,
    shape_of,
    token_id,
    tokenise,
)

VECTOR_FILE = f"reader_vectors.v{READER_SCHEMA_VERSION}.json"

# (id, note, message)
_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "account_debit",
        "account layout with masked account, date, time and reference",
        "Sample Sacco: Your a/c ***1234 debited KES 1,250.00 to EXAMPLE GROCERS on 29/10/26 "
        "14:05. Avl bal KES 8,400.00. Ref SK29XY7.",
    ),
    ("glued_currency", "currency glued to the amount splits", "Paid KES1,250.00 to DEMO CAFE."),
    ("currency_dot", "Ksh. before the amount", "Received Ksh.500 from JOHN DOE 0712***678."),
    ("mixed_reference", "letters and digits stay one token", "SJK7H9B2C1 Confirmed. TXN123456"),
    ("times", "24 hour, 12 hour and seconds", "at 14:05 or 2:05 PM or 14:05:33"),
    (
        "dates",
        "date shapes",
        "29/10/26 29-10-2026 2026-10-29 29 Oct 2026 29-OCT-26",
    ),
    ("punctuation", "single characters and spacing", "Bal: KES 1,000.|Ref#77 ; ok"),
    (
        "non_ascii",
        "non-ASCII is a single-character token",
        "Dear Member, Asante " + chr(0x2013) + " ok",
    ),
    ("empty", "no tokens", ""),
    ("long", "tokens past MAX_TOKENS are dropped", " ".join(["word"] * 120)),
)


def build_vectors() -> dict[str, Any]:
    cases = []
    for case_id, note, message in _CASES:
        tokens = tokenise(message)
        cases.append(
            {
                "id": case_id,
                "note": note,
                "message": message,
                "expected": {
                    "tokens": [
                        {
                            "text": t.text,
                            "start": t.start,
                            "end": t.end,
                            "shape": int(shape_of(t.text)),
                            "hash_key": hash_key(t.text),
                            "id": token_id(t.text),
                        }
                        for t in tokens[:12]
                    ],
                    "token_count": len(tokens),
                    "input": encode(message),
                },
            }
        )
    return {"manifest": reader_manifest(), "cases": cases}


def render(vectors: dict[str, Any]) -> str:
    return json.dumps(vectors, indent=1, ensure_ascii=True) + "\n"


def default_path() -> Path:
    return Path(str(resources.files("akili.resources").joinpath(VECTOR_FILE)))


def write_vectors(path: Path | None = None) -> Path:
    target = path or default_path()
    target.write_text(render(build_vectors()), encoding="utf-8")
    return target

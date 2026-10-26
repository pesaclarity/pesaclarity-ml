"""Conformance vectors for the feature pipeline.

The app's Dart implementation must reproduce every ``expected`` value in the committed vector
file byte for byte. Regenerate with ``akili vectors`` after any change to :mod:`akili.features`
(which must also bump ``FEATURE_SCHEMA_VERSION``); a test fails if the file drifts from the code.
All payees are synthetic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any

from akili.config import FeatureConfig
from akili.features import (
    AMOUNT_EDGES_MINOR,
    FEATURE_SCHEMA_VERSION,
    NUM_TOKEN,
    PHONE_TOKEN,
    encode,
    feature_manifest,
    fnv1a_32,
    local_time,
    tokenise,
)
from akili.types import Direction, Transaction

VECTOR_FILE = f"feature_vectors.v{FEATURE_SCHEMA_VERSION}.json"
INT64_MAX = 2**63 - 1

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NOON = "2026-03-18T12:00:00+03:00"
_OUT, _IN = Direction.OUT, Direction.IN

# (id, note, payee, amount_minor, direction, ISO 8601 timestamp with offset)
_CASES: tuple[tuple[str, str, str, int, Direction, str], ...] = (
    # Tokenisation
    (
        "till_basic",
        "till number becomes <num>",
        "DEMO FRESH MART TILL 000123",
        120_000,
        _OUT,
        _NOON,
    ),
    (
        "paybill_account",
        "paybill and account numbers",
        "SAMPLE WATER COMPANY PAYBILL 000400 ACC 0001",
        250_000,
        _OUT,
        _NOON,
    ),
    ("phone_masked", "masked phone", "TEST PERSON 07 0700***007", 200_000, _IN, _NOON),
    (
        "phone_spaced",
        "one space either side of the stars",
        "TEST PERSON 0700 *** 007",
        50_000,
        _IN,
        _NOON,
    ),
    (
        "phone_tabbed",
        "ASCII tab counts as the optional space",
        "TEST PERSON 0700\t***\t007",
        50_000,
        _IN,
        _NOON,
    ),
    (
        "phone_two_spaces",
        "two spaces break the mask",
        "TEST PERSON 0700  ***007",
        50_000,
        _OUT,
        _NOON,
    ),
    (
        "phone_five_digits",
        "5 leading digits is not a mask",
        "TEST PERSON 07000***007",
        50_000,
        _OUT,
        _NOON,
    ),
    ("phone_one_star", "one star is not a mask", "TEST PERSON 0700*007", 50_000, _OUT, _NOON),
    (
        "phone_glued_to_letters",
        "mask must not touch letters",
        "TESTX0700***007",
        50_000,
        _OUT,
        _NOON,
    ),
    (
        "phone_twice",
        "two masks in a row stay separate",
        "0700***001 0700***002",
        50_000,
        _IN,
        _NOON,
    ),
    (
        "phone_then_numbers",
        "<num> after <phone> is kept",
        "TEST PERSON 0700***007 000123 000456",
        50_000,
        _OUT,
        _NOON,
    ),
    (
        "phone_literal_text",
        "the text <phone> is not a phone token",
        "DEMO <PHONE> SHOP",
        50_000,
        _OUT,
        _NOON,
    ),
    ("empty", "empty payee: all padding", "", 10_000, _OUT, _NOON),
    ("whitespace_only", "only whitespace", " \t\r\n ", 10_000, _OUT, _NOON),
    ("punctuation_only", "only punctuation", "--- *** ... ///", 10_000, _OUT, _NOON),
    ("mixed_case", "ASCII case folding", "Demo Coffee HOUSE", 45_000, _OUT, _NOON),
    (
        "punctuation_inside",
        "apostrophe, hyphen, ampersand split",
        "MAMA'S KIBANDA-CAFE & GRILL",
        60_000,
        _OUT,
        _NOON,
    ),
    (
        "alnum_mix",
        "any digit makes the whole token <num>",
        "SAMPLE 4G DATA 2GB",
        9_900,
        _OUT,
        _NOON,
    ),
    (
        "numbers_collapse",
        "consecutive <num> collapse",
        "PAYBILL 000400 000401 ACC 1 2 3",
        300_000,
        _OUT,
        _NOON,
    ),
    ("underscore", "underscore separates", "DEMO_SHOP_TILL_000111", 30_000, _OUT, _NOON),
    ("accented", "non-ASCII letters separate", "D\u00c9MO CAF\u00c9", 35_000, _OUT, _NOON),
    ("combining_accent", "combining mark separates", "DEMO CAFE\u0301 BAR", 35_000, _OUT, _NOON),
    (
        "dotted_capital_i",
        "no Unicode case mapping",
        "\u0130STANBUL DEMO GRILL",
        35_000,
        _OUT,
        _NOON,
    ),
    ("kelvin_sign", "K (U+212A) is not k", "\u212aIOSK DEMO", 35_000, _OUT, _NOON),
    (
        "fullwidth",
        "no NFKC: fullwidth letters separate",
        "\uff24\uff25\uff2d\uff2f SHOP",
        35_000,
        _OUT,
        _NOON,
    ),
    (
        "arabic_indic_digits",
        "only ASCII digits are digits",
        "DEMO \u0660\u0667\u0660\u0660",
        35_000,
        _OUT,
        _NOON,
    ),
    ("superscript_digit", "superscript is a separator", "DEMO\u00b2 MART", 35_000, _OUT, _NOON),
    ("emoji", "astral code points separate", "DEMO \U0001f6d2SHOP", 35_000, _OUT, _NOON),
    ("cjk", "CJK separates", "\u793a\u4f8b DEMO", 35_000, _OUT, _NOON),
    ("zero_width_space", "U+200B separates", "DEMO\u200bSHOP", 35_000, _OUT, _NOON),
    ("no_break_space", "U+00A0 separates", "DEMO\u00a0SHOP", 35_000, _OUT, _NOON),
    ("swahili", "plain Swahili words", "FAKE DUKA LA MAMA MBOGA", 15_000, _OUT, _NOON),
    ("exactly_max_tokens", "8 tokens fill every slot", "A B C D E F G H", 15_000, _OUT, _NOON),
    ("over_max_tokens", "9th token dropped", "A B C D E F G H I", 15_000, _OUT, _NOON),
    (
        "long_name",
        "long name truncated to the first 8 tokens",
        " ".join(f"SAMPLE WORD{chr(65 + i % 26)}" for i in range(40)),
        15_000,
        _IN,
        _NOON,
    ),
    ("long_token", "one 600-character token", "DEMO " + "X" * 600, 15_000, _OUT, _NOON),
    # Amounts
    ("amount_zero", "zero", "DEMO BANK CHARGES", 0, _OUT, _NOON),
    ("amount_one_cent", "one cent", "DEMO BANK CHARGES", 1, _OUT, _NOON),
    ("amount_below_edge", "KES 49.99", "DEMO BANK CHARGES", 4_999, _OUT, _NOON),
    ("amount_on_edge", "KES 50.00 is in the next bucket", "DEMO BANK CHARGES", 5_000, _OUT, _NOON),
    ("amount_mid", "KES 9,999.99", "DEMO HARDWARE", 999_999, _OUT, _NOON),
    ("amount_top_edge", "KES 100,000.00", "DEMO EMPLOYER LTD SALARY", 10_000_000, _IN, _NOON),
    ("amount_int64_max", "largest int64", "DEMO EMPLOYER LTD SALARY", INT64_MAX, _IN, _NOON),
    ("amount_negative", "sign is ignored", "DEMO REVERSAL", -250_000, _IN, _NOON),
    # Time
    (
        "midnight_local",
        "00:00 local, 1st of month",
        "DEMO APARTMENTS RENT",
        2_500_000,
        _OUT,
        "2026-04-01T00:00:00+03:00",
    ),
    (
        "before_midnight_local",
        "last ms of the month",
        "DEMO APARTMENTS RENT",
        2_500_000,
        _OUT,
        "2026-03-31T23:59:59.999+03:00",
    ),
    (
        "midnight_from_utc",
        "21:00Z is local midnight",
        "DEMO APARTMENTS RENT",
        2_500_000,
        _OUT,
        "2026-03-31T21:00:00Z",
    ),
    (
        "utc_crosses_date",
        "22:30Z Sunday is 01:30 Monday local",
        "DEMO MATATU SACCO",
        15_000,
        _OUT,
        "2026-01-04T22:30:00Z",
    ),
    (
        "other_offset",
        "+05:30 input",
        "DEMO MATATU SACCO",
        15_000,
        _OUT,
        "2026-06-15T02:00:00+05:30",
    ),
    (
        "hour_bucket_end",
        "03:59:59.999 local",
        "DEMO PETROL STATION",
        300_000,
        _OUT,
        "2026-05-09T03:59:59.999+03:00",
    ),
    (
        "hour_bucket_start",
        "04:00 local",
        "DEMO PETROL STATION",
        300_000,
        _OUT,
        "2026-05-09T04:00:00+03:00",
    ),
    ("last_hour", "23:00 local", "SAMPLE LOUNGE BAR", 150_000, _OUT, "2026-05-10T23:00:00+03:00"),
    (
        "day_5",
        "5th is month start",
        "DEMO ACADEMY FEES",
        3_000_000,
        _OUT,
        "2026-02-05T23:59:59+03:00",
    ),
    (
        "day_6",
        "6th is mid-month",
        "DEMO ACADEMY FEES",
        3_000_000,
        _OUT,
        "2026-02-06T00:00:00+03:00",
    ),
    (
        "day_24",
        "24th is mid-month",
        "SAMPLE SACCO DEPOSIT",
        500_000,
        _OUT,
        "2026-02-24T23:59:59+03:00",
    ),
    (
        "day_25",
        "25th is month end",
        "SAMPLE COMPANY PAYROLL",
        8_500_000,
        _IN,
        "2026-02-25T00:00:00+03:00",
    ),
    ("leap_day", "29 February", "DEMO GROCERS", 120_000, _OUT, "2028-02-29T12:00:00+03:00"),
    (
        "pre_epoch",
        "before 1970; floor division",
        "DEMO GROCERS",
        120_000,
        _OUT,
        "1969-12-31T20:59:59.999Z",
    ),
    (
        "epoch_local_midnight",
        "1970-01-01 00:00 local",
        "DEMO GROCERS",
        120_000,
        _OUT,
        "1969-12-31T21:00:00Z",
    ),
    ("far_future", "year 2100", "DEMO GROCERS", 120_000, _IN, "2100-12-31T23:59:59+03:00"),
)


def epoch_ms(ts: datetime) -> int:
    """Milliseconds since the Unix epoch (UTC), floored."""
    return (ts - _EPOCH) // timedelta(milliseconds=1)


def build_vectors(cfg: FeatureConfig | None = None) -> dict[str, Any]:
    """The full vector document for ``cfg`` (default: the shipping feature config)."""
    cfg = cfg or FeatureConfig()
    cases = []
    for case_id, note, payee, amount, direction, iso in _CASES:
        ts = datetime.fromisoformat(iso)
        tx = Transaction(payee, amount, direction, ts)
        cases.append(
            {
                "id": case_id,
                "note": note,
                "input": {
                    "payee": payee,
                    "amount_minor": amount,
                    "direction": direction.value,
                    "timestamp_ms": epoch_ms(ts),
                    "timestamp": iso,
                },
                "expected": {
                    "local_time": local_time(ts).isoformat(timespec="milliseconds"),
                    "tokens": tokenise(payee),
                    "features": encode(tx, cfg),
                },
            }
        )
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("vector case ids must be unique")
    return {
        **feature_manifest(cfg),
        "amount_edges_minor": list(AMOUNT_EDGES_MINOR),
        "hash_checks": [
            {"text": t, "fnv1a_32": fnv1a_32(t)}
            for t in ("", "a", "foobar", NUM_TOKEN, PHONE_TOKEN, "demo")
        ],
        "cases": cases,
    }


def render(doc: dict[str, Any]) -> str:
    """Canonical text of a vector file: ASCII-only JSON, two-space indent, trailing newline."""
    return json.dumps(doc, indent=2, ensure_ascii=True) + "\n"


def default_path() -> Path:
    """The committed vector file inside the source tree."""
    return Path(str(resources.files("akili.resources").joinpath(VECTOR_FILE)))


def write_vectors(path: str | Path | None = None) -> Path:
    out = Path(path) if path is not None else default_path()
    out.write_text(render(build_vectors()), encoding="utf-8", newline="\n")
    return out

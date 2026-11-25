"""The statement-payee test set and per-category scores for a categoriser file.

Bank and M-PESA statements, and the app's SMS reader, give the categoriser the counterparty's
name alone (``EXAMPLE GROCERS``) or a statement's details column (``MERCHANT PAYMENT TO EXAMPLE
SUPERMARKET``, ``PAY BILL TO 000123 - EXAMPLE ENERGY``). This set holds such rows, every one
invented: the names use the markers ``EXAMPLE`` and ``MFANO``, which the synthetic generator
never writes, and words of their own, so the set measures payees the model has not seen.
Numbers start with 000 and phones are ``0700***0nn`` placeholders, as in the generator.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from akili.categories import default_categories
from akili.config import FeatureConfig
from akili.features import NAIROBI, encode
from akili.types import Direction, Transaction

MARKERS = ("EXAMPLE", "MFANO")

# Category, direction, amount in KES, local hour, day of month, and names (marker added).
_NAMES: tuple[tuple[str, Direction, int, int, int, tuple[str, ...]], ...] = (
    (
        "groceries",
        Direction.OUT,
        1_800,
        18,
        14,
        ("GROCERS", "SUPERMARKET", "MINI MART", "FRESH PRODUCE", "BUTCHERY"),
    ),
    (
        "fuel",
        Direction.OUT,
        3_500,
        8,
        12,
        ("FUEL STATION", "PETROL STATION", "SERVICE STATION", "ENERGY"),
    ),
    ("eating_out", Direction.OUT, 650, 13, 16, ("CAFE", "HOTEL", "GRILL", "EATERY")),
    ("snacks", Direction.OUT, 150, 16, 11, ("BAKERY", "SNACKS", "SWEETS")),
    (
        "transport",
        Direction.OUT,
        450,
        7,
        9,
        ("TAXI RIDES", "SHUTTLE", "MATATU SACCO", "BODA RIDERS", "CAB SERVICES", "BUS LINES"),
    ),
    ("transport", Direction.OUT, 4_500, 6, 19, ("RAILWAY EXPRESS", "AIRLINES", "COACH TICKETS")),
    (
        "utilities",
        Direction.OUT,
        2_500,
        19,
        7,
        ("WATER CO", "POWER TOKENS", "INTERNET LTD", "FIBRE"),
    ),
    ("school_fees", Direction.OUT, 15_000, 10, 6, ("SCHOOL FEES", "ACADEMY", "COLLEGE")),
    ("health", Direction.OUT, 420, 11, 18, ("PHARMACY", "CLINIC", "CHEMIST", "MEDICAL CENTRE")),
    ("rent", Direction.OUT, 22_000, 9, 2, ("APARTMENTS", "PROPERTIES")),
    ("shopping", Direction.OUT, 3_200, 15, 20, ("HARDWARE", "ELECTRONICS", "BOUTIQUE")),
    ("lifestyle", Direction.OUT, 1_200, 20, 21, ("LOUNGE", "CINEMA", "GYM")),
    ("giving", Direction.OUT, 1_000, 10, 15, ("CHURCH", "HARAMBEE")),
    ("savings", Direction.OUT, 5_000, 12, 26, ("SACCO", "INVESTMENT CLUB")),
    ("loan_repayment", Direction.OUT, 2_000, 12, 13, ("CREDIT", "LOANS")),
    ("salary", Direction.IN, 45_000, 10, 26, ("EMPLOYER LTD", "COMPANY PAYROLL")),
    ("salary", Direction.IN, 62_000, 9, 28, ("LTD SALARY", "LIMITED NET PAY", "PAYROLL")),
)
_WORDS = ("BARAKA", "NURU", "TUMAINI", "JUMBO", "UPENDO", "AMANI")


@dataclass(frozen=True)
class Row:
    payee: str
    category: str
    direction: Direction
    amount_minor: int
    timestamp: datetime


_FORMS = {
    "till": ("{name}", "MERCHANT PAYMENT TO {name}", "POS TXN {name}"),
    "paybill": ("{name}", "PAY BILL TO 000{n:03d} - {name}"),
    "in": ("{name}", "PAY BILL FROM 000{n:03d} - {name}"),
}
_PEOPLE = ("{name}", "CUSTOMER TRANSFER TO 0700***0{n:02d} - {name}")
_PEOPLE_IN = ("{name}", "CUSTOMER TRANSFER FROM 0700***0{n:02d} - {name}")
# Money in from people: category, KES, day of month, hour (the label is a guess the set does not
# score: any of the received categories counts).
_RECEIVED = (
    ("money_received", 1_500, 12, 14),
    ("support_received", 5_000, 3, 19),
    ("gifts_received", 800, 20, 16),
    ("money_received", 3_000, 17, 9),
)
_IN_PEOPLE = (
    ("MFANO", "KAMAU"),
    ("EXAMPLE", "ACHIENG"),
    ("MFANO", "NJERI"),
    ("EXAMPLE", "KIPRONO"),
)


_TILL = {
    "groceries",
    "fuel",
    "eating_out",
    "snacks",
    "transport",
    "health",
    "shopping",
    "lifestyle",
}


def rows() -> list[Row]:
    """The test set, the same on every call."""
    out: list[Row] = []
    n = 0
    for category, direction, kes, hour, day, names in _NAMES:
        for i, name in enumerate(names):
            for j, marker in enumerate(MARKERS):
                word = _WORDS[(i + j) % len(_WORDS)]
                full = f"{marker} {word} {name}" if j else f"{marker} {name}"
                kind = "in" if direction is Direction.IN else "paybill"
                for form in _FORMS["till" if category in _TILL else kind]:
                    n += 1
                    out.append(
                        Row(
                            form.format(name=full, n=n % 1000),
                            category,
                            direction,
                            kes * 100,
                            datetime(2026, 9, day, hour, n % 60, tzinfo=NAIROBI),
                        )
                    )
    # People, as statements name them: the label is the person's, so only check that money to
    # or from a person is not taken for a business category.
    for k, (first, last) in enumerate((("MFANO", "WANJA"), ("EXAMPLE", "OTIENO"))):
        for form in _PEOPLE:
            out.append(
                Row(
                    form.format(name=f"{first} {last}", n=k),
                    "family_support",
                    Direction.OUT,
                    300_000,
                    datetime(2026, 9, 3, 18, 0, tzinfo=NAIROBI),
                )
            )
    for k, ((first, last), (category, kes, day, hour)) in enumerate(
        zip(_IN_PEOPLE, _RECEIVED, strict=True)
    ):
        for form in _PEOPLE_IN:
            out.append(
                Row(
                    form.format(name=f"{first} {last}", n=k + 10),
                    category,
                    Direction.IN,
                    kes * 100,
                    datetime(2026, 9, day, hour, 0, tzinfo=NAIROBI),
                )
            )
    return out


SENT_TO_PEOPLE = {"family_support", "friends_social"}
RECEIVED_FROM_PEOPLE = {"money_received", "support_received", "gifts_received"}
PERSON_CATEGORIES = SENT_TO_PEOPLE | RECEIVED_FROM_PEOPLE


def _person_right(got: str, want: str) -> bool:
    """Money to a person is right in either sent category, money from one in any received one:
    which of them only the person knows."""
    for group in (SENT_TO_PEOPLE, RECEIVED_FROM_PEOPLE):
        if want in group:
            return got in group
    return False


def score(run: Callable[[np.ndarray], np.ndarray], features: FeatureConfig) -> dict[str, Any]:
    """Accuracy and per-category precision and recall of ``run`` (int32 rows to softmax rows in
    labels.json order) on :func:`rows`. Money to people counts as right in either sent category,
    money from people in any received one."""
    labels = list(default_categories().labels)
    data = rows()
    x = np.asarray(
        [
            encode(Transaction(r.payee, r.amount_minor, r.direction, r.timestamp), features)
            for r in data
        ],
        np.int32,
    )
    got = [labels[int(i)] for i in run(x).argmax(-1)]
    want = [r.category for r in data]
    right = [g == w or _person_right(g, w) for g, w in zip(got, want, strict=True)]
    per: dict[str, dict[str, float]] = {}
    for c in sorted({*want} - PERSON_CATEGORIES):
        tp = sum(g == c and w == c for g, w in zip(got, want, strict=True))
        predicted = sum(g == c for g in got)
        actual = sum(w == c for w in want)
        per[c] = {
            "precision": round(tp / predicted, 3) if predicted else 0.0,
            "recall": round(tp / actual, 3),
            "rows": actual,
        }
    people = {}
    for name, group in (
        ("sent_to_people", SENT_TO_PEOPLE),
        ("received_from_people", RECEIVED_FROM_PEOPLE),
    ):
        hits = [ok for w, ok in zip(want, right, strict=True) if w in group]
        people[name] = {"accuracy": round(sum(hits) / len(hits), 3), "rows": len(hits)}
    return {
        "rows": len(data),
        "accuracy": round(sum(right) / len(data), 3),
        "categories": per,
        "people": people,
        "wrong": [
            {"payee": r.payee, "want": w, "got": g}
            for r, w, g, ok in zip(data, want, got, right, strict=True)
            if not ok
        ],
    }


def write(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

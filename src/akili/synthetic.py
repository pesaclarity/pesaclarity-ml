"""Synthetic transaction generator for tests and smoke training.

Every payee is invented and marked DEMO, SAMPLE, FAKE or TEST; till and paybill numbers start
with 000 and phone numbers are masked 0700 000 0xx-style placeholders. Nothing here is, or is
derived from, real data. Never replace it with real messages.

With ``payee_style: varied`` the names are built from random syllables (``DEMO KALIMO GROCERS``,
``TEST WANDERU ZIKOTA``), so each user gets their own long tail of merchants and people, as on a
real statement. The words are generated, not taken from any list of real businesses or people.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

from akili.categories import CategorySet, default_categories
from akili.config import SyntheticDataConfig
from akili.features import NAIROBI
from akili.types import Direction, Transaction

IdKind = Literal["till", "paybill", "person", "none"]


@dataclass(frozen=True)
class _Profile:
    names: tuple[str, ...]
    direction: Direction
    median_kes: float
    sigma: float
    hours: tuple[int, int]
    id_kind: IdKind = "none"
    month_days: tuple[int, int] = (1, 28)
    weekend_bias: float = 0.0


_OUT, _IN = Direction.OUT, Direction.IN

PROFILES: dict[str, _Profile] = {
    "salary": _Profile(
        ("DEMO EMPLOYER LTD SALARY", "SAMPLE COMPANY PAYROLL", "FAKE NGO LTD SALARY"),
        _IN,
        85_000,
        0.4,
        (8, 17),
        month_days=(25, 28),
    ),
    "business_income": _Profile(
        ("DEMO CLIENT INVOICE", "SAMPLE AGENCY PAYMENT", "FAKE CONSULTING LTD"),
        _IN,
        30_000,
        0.8,
        (8, 19),
        "paybill",
    ),
    "money_received": _Profile(("TEST PERSON",), _IN, 2_000, 0.9, (7, 22), "person"),
    "groceries": _Profile(
        (
            "DEMO FRESH MART",
            "SAMPLE SUPERMARKET",
            "FAKE MAMA MBOGA",
            "DEMO GROCERS",
            "SAMPLE MINIMART",
        ),
        _OUT,
        1_200,
        0.8,
        (7, 21),
        "till",
    ),
    "eating_out": _Profile(
        (
            "DEMO KIBANDA CAFE",
            "SAMPLE NYAMA CHOMA GRILL",
            "FAKE CHAPATI HOTEL",
            "DEMO COFFEE HOUSE",
        ),
        _OUT,
        600,
        0.6,
        (12, 21),
        "till",
    ),
    "transport": _Profile(
        ("DEMO MATATU SACCO", "SAMPLE BODA RIDES", "FAKE TAXI APP", "DEMO BUS SERVICE"),
        _OUT,
        150,
        0.7,
        (6, 20),
        "till",
    ),
    "fuel": _Profile(
        ("DEMO PETROL STATION", "SAMPLE FUEL SERVICE STN", "FAKE GAS PUMP"),
        _OUT,
        3_000,
        0.5,
        (6, 20),
        "till",
    ),
    "airtime_data": _Profile(
        ("SAMPLE AIRTIME PURCHASE", "DEMO DATA BUNDLES", "FAKE TELCO BUNDLES"),
        _OUT,
        100,
        0.8,
        (6, 23),
    ),
    "rent": _Profile(
        ("DEMO APARTMENTS RENT", "SAMPLE LANDLORD ACC", "FAKE PROPERTY AGENTS"),
        _OUT,
        25_000,
        0.3,
        (8, 20),
        "paybill",
        month_days=(1, 5),
    ),
    "utilities": _Profile(
        ("DEMO POWER TOKENS", "SAMPLE WATER COMPANY", "FAKE FIBRE INTERNET"),
        _OUT,
        2_500,
        0.6,
        (7, 22),
        "paybill",
    ),
    "school_fees": _Profile(
        ("DEMO ACADEMY FEES", "SAMPLE HIGH SCHOOL", "FAKE UNIVERSITY FEES"),
        _OUT,
        30_000,
        0.6,
        (8, 17),
        "paybill",
        month_days=(1, 10),
    ),
    "health": _Profile(
        ("DEMO PHARMACY", "SAMPLE CLINIC", "FAKE HOSPITAL", "DEMO HEALTH COVER"),
        _OUT,
        1_500,
        0.9,
        (8, 21),
        "till",
    ),
    "family_support": _Profile(
        ("TEST PERSON",),
        _OUT,
        3_000,
        0.7,
        (7, 21),
        "person",
        month_days=(1, 8),
    ),
    "friends_social": _Profile(
        ("TEST PERSON",),
        _OUT,
        500,
        0.8,
        (17, 23),
        "person",
        weekend_bias=0.5,
    ),
    "lifestyle": _Profile(
        ("DEMO CINEMA", "SAMPLE LOUNGE BAR", "FAKE STREAMING SUB", "DEMO GYM"),
        _OUT,
        1_500,
        0.7,
        (16, 23),
        "till",
        weekend_bias=0.5,
    ),
    "betting": _Profile(
        ("DEMO BET LTD", "SAMPLE SPORTS PREDICT", "FAKE JACKPOT GAMES"),
        _OUT,
        200,
        1.0,
        (10, 23),
        "paybill",
        weekend_bias=0.4,
    ),
    "shopping": _Profile(
        ("DEMO FASHION HOUSE", "SAMPLE ELECTRONICS", "FAKE SHOE STORE", "DEMO HARDWARE"),
        _OUT,
        3_500,
        0.8,
        (9, 20),
        "till",
        weekend_bias=0.3,
    ),
    "giving": _Profile(
        ("DEMO CHURCH TITHE", "SAMPLE HARAMBEE FUND", "FAKE CHARITY TRUST"),
        _OUT,
        1_000,
        0.8,
        (7, 14),
        "paybill",
        weekend_bias=0.6,
    ),
    "fees_charges": _Profile(
        ("SAMPLE TRANSACTION COST", "DEMO BANK CHARGES", "FAKE OVERDRAFT FEE"),
        _OUT,
        30,
        1.0,
        (0, 23),
    ),
    "loan_repayment": _Profile(
        ("DEMO OVERDRAFT REPAYMENT", "SAMPLE MOBILE LOAN", "FAKE BANK LOAN ACC"),
        _OUT,
        2_000,
        0.9,
        (7, 22),
        "paybill",
    ),
    "savings": _Profile(
        ("DEMO MONEY MARKET FUND", "SAMPLE SACCO DEPOSIT", "FAKE SAVINGS LOCK"),
        _OUT,
        5_000,
        0.8,
        (8, 20),
        "paybill",
        month_days=(25, 28),
    ),
    "tax": _Profile(
        ("DEMO REVENUE AUTHORITY", "SAMPLE TAX PAYMENT"),
        _OUT,
        5_000,
        0.8,
        (8, 17),
        "paybill",
    ),
}

FAKE_MARKERS: tuple[str, ...] = ("DEMO", "SAMPLE", "FAKE", "TEST")
START_DATE = datetime(2026, 1, 1, tzinfo=NAIROBI)

_SYLLABLES: tuple[str, ...] = (
    "ka", "li", "mo", "te", "nza", "ra", "bu", "shi", "wa", "ndu", "zi", "ro", "pe", "lu",
    "mbi", "yo", "ki", "ta", "nu", "sa", "go", "fu", "ve", "dhi", "ko", "me", "ri", "tho",
)  # fmt: skip
_BUSINESS_WORDS: dict[str, tuple[str, ...]] = {
    "business_income": ("ENTERPRISES", "AGENCIES", "CONSULTANTS", "TRADERS"),
    "groceries": ("GROCERS", "MINIMART", "FRESH PRODUCE", "SUPERSTORE", "DUKA"),
    "eating_out": ("CAFE", "GRILL", "EATERY", "BAKERY", "HOTEL"),
    "transport": ("SACCO", "SHUTTLE", "RIDES", "COURIERS"),
    "fuel": ("SERVICE STATION", "PETROL", "ENERGY"),
    "rent": ("APARTMENTS", "HEIGHTS", "COURT", "PROPERTIES"),
    "utilities": ("WATER CO", "POWER TOKENS", "FIBRE", "GAS REFILL"),
    "school_fees": ("ACADEMY", "SCHOOL", "COLLEGE", "KINDERGARTEN"),
    "health": ("PHARMACY", "CLINIC", "MEDICAL CENTRE", "CHEMIST"),
    "lifestyle": ("LOUNGE", "GYM", "SALON", "CINEMA"),
    "betting": ("BET", "SPORTS PREDICT", "JACKPOT"),
    "shopping": ("FASHIONS", "ELECTRONICS", "HARDWARE", "BOUTIQUE"),
    "giving": ("CHURCH", "HARAMBEE", "WELFARE GROUP", "FOUNDATION"),
    "loan_repayment": ("CREDIT", "LOANS", "FINANCE"),
    "savings": ("SACCO", "CHAMA", "INVESTMENT CLUB"),
}


def _invented_word(rng: np.random.Generator) -> str:
    n = int(rng.integers(2, 4))
    return "".join(str(rng.choice(_SYLLABLES)) for _ in range(n)).upper()


class _VariedNames:
    """Per-user favourite payees, a few shared popular ones and occasional one-offs."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._favourites: dict[tuple[int, str], list[str]] = {}

    def _invent(self, category: str, profile: _Profile) -> str:
        rng = self._rng
        marker = str(rng.choice(FAKE_MARKERS[:3]))
        if profile.id_kind == "person":
            return f"TEST {_invented_word(rng)} {_invented_word(rng)}"
        words = _BUSINESS_WORDS.get(category)
        if words is None:
            return str(rng.choice(profile.names))
        return f"{marker} {_invented_word(rng)} {rng.choice(words)}"

    def pick(self, user: int, category: str, profile: _Profile) -> str:
        rng = self._rng
        favourites = self._favourites.setdefault((user, category), [])
        roll = rng.random()
        if roll < 0.15 and profile.id_kind != "person":
            return str(rng.choice(profile.names))
        if roll < 0.25 or not favourites:
            name = self._invent(category, profile)
            if len(favourites) < 3:
                favourites.append(name)
            return name
        return favourites[int(rng.integers(0, len(favourites)))]


def _payee(profile: _Profile, rng: np.random.Generator, name: str | None = None) -> str:
    if name is None:
        name = str(rng.choice(profile.names))
    n = int(rng.integers(1, 100))
    match profile.id_kind:
        case "till":
            return f"{name} TILL 000{n:03d}"
        case "paybill":
            return f"{name} PAYBILL 000{n:03d} ACC 00{n:02d}"
        case "person":
            return f"{name} {n:02d} 0700***0{n:02d}"
        case "none":
            return name


def _timestamp(profile: _Profile, rng: np.random.Generator) -> datetime:
    month = int(rng.integers(0, 3))
    lo, hi = profile.month_days
    day = int(rng.integers(lo, hi + 1))
    base = START_DATE.replace(month=START_DATE.month + month, day=day)
    if profile.weekend_bias and rng.random() < profile.weekend_bias:
        base += timedelta(days=(5 - base.weekday()) % 7 + int(rng.integers(0, 2)))
    hour = int(rng.integers(profile.hours[0], profile.hours[1] + 1))
    return base.replace(hour=hour, minute=int(rng.integers(0, 60)))


def generate(cfg: SyntheticDataConfig, categories: CategorySet | None = None) -> list[Transaction]:
    """Generate labelled synthetic transactions for ``cfg.n_users`` fake users.

    Each user has their own category mix (Dirichlet), so partitions by user are non-IID, as
    federated clients will be.
    """
    cats = categories or default_categories()
    labels = [c for c in cats.labels if c in PROFILES]
    rng = np.random.default_rng(cfg.seed)
    varied = (
        _VariedNames(np.random.default_rng([cfg.seed, 1])) if cfg.payee_style == "varied" else None
    )
    out: list[Transaction] = []
    for u in range(cfg.n_users):
        user_id = f"synthetic-user-{u:05d}"
        mix = rng.dirichlet(np.full(len(labels), 0.8))
        for _ in range(cfg.transactions_per_user):
            category = labels[int(rng.choice(len(labels), p=mix))]
            profile = PROFILES[category]
            amount_kes = int(np.round(profile.median_kes * rng.lognormal(0, profile.sigma)))
            label = category
            if rng.random() < cfg.label_noise:
                label = labels[int(rng.integers(0, len(labels)))]
            out.append(
                Transaction(
                    payee=_payee(
                        profile, rng, varied.pick(u, category, profile) if varied else None
                    ),
                    amount_minor=max(amount_kes, 1) * 100,
                    direction=profile.direction,
                    timestamp=_timestamp(profile, rng),
                    category=label,
                    user_id=user_id,
                )
            )
    return out


def commercial_opt_ins(cfg: SyntheticDataConfig, user_ids: Iterable[str]) -> set[str]:
    """Deterministic set of synthetic users who opted in to the commercial track."""
    rng = np.random.default_rng(cfg.seed + 1)
    return {u for u in sorted(set(user_ids)) if rng.random() < cfg.commercial_opt_in_rate}


def write_jsonl(txs: Iterable[Transaction], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for tx in txs:
            fh.write(
                json.dumps(
                    {
                        "payee": tx.payee,
                        "amount_minor": tx.amount_minor,
                        "direction": tx.direction.value,
                        "timestamp": tx.timestamp.isoformat(),
                        "category": tx.category,
                        "user_id": tx.user_id,
                    }
                )
                + "\n"
            )


def read_jsonl(path: str | Path) -> Iterator[Transaction]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            yield Transaction(
                payee=row["payee"],
                amount_minor=int(row["amount_minor"]),
                direction=Direction(row["direction"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                category=row.get("category"),
                user_id=row.get("user_id"),
            )

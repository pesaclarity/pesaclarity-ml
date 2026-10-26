"""Feature extraction shared by training and the app.

The app reproduces this exactly in Dart; ``docs/FEATURES.md`` is the language-neutral spec and
``resources/feature_vectors.v1.json`` the conformance vectors. Every step is ASCII-only integer
arithmetic so the result never depends on locale, Unicode tables or the host time zone.

Layout of the int32 vector (length ``max_tokens + 5``)::

    [token_0 .. token_{max_tokens-1}, amount_bucket, direction, hour_bucket, weekday, month_phase]

Token id 0 is padding; real tokens hash into ``1 .. hash_buckets - 1``.

Any change to the output of :func:`encode` must bump :data:`FEATURE_SCHEMA_VERSION`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import numpy.typing as npt

from akili.config import FeatureConfig
from akili.types import Direction, Transaction

FEATURE_SCHEMA_VERSION = 1
"""Version of the feature pipeline. Exported models record it; the app refuses a mismatch."""

AMOUNT_EDGES_MINOR: tuple[int, ...] = tuple(
    kes * 100 for kes in (50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000)
)
"""Bucket edges in minor units (KES cents). Bucket = number of edges <= |amount|."""

N_AMOUNT_BUCKETS = len(AMOUNT_EDGES_MINOR) + 1
N_DIRECTIONS = 2
N_HOUR_BUCKETS = 6
N_WEEKDAYS = 7
N_MONTH_PHASES = 3
N_CONTEXT_FEATURES = 5
CONTEXT_CARDINALITIES: tuple[int, ...] = (
    N_AMOUNT_BUCKETS,
    N_DIRECTIONS,
    N_HOUR_BUCKETS,
    N_WEEKDAYS,
    N_MONTH_PHASES,
)

CONTEXT_LAYOUT: tuple[str, ...] = (
    "amount_bucket",
    "direction",
    "hour_bucket",
    "weekday",
    "month_phase",
)

NAIROBI = timezone(timedelta(hours=3), "EAT")
"""Africa/Nairobi as a fixed UTC+3 offset; Kenya does not observe daylight saving time."""

NUM_TOKEN = "<num>"
PHONE_TOKEN = "<phone>"

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
# Character classes are spelled out so no engine applies Unicode semantics to \d, \s or \b.
_MASKED_PHONE = re.compile(
    r"(?<![0-9a-z])[0-9]{2,4}[\t\n\v\f\r ]?\*{2,}[\t\n\v\f\r ]?[0-9]{2,4}(?![0-9a-z])"
)
_WORD = re.compile(r"[0-9a-z]+")
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193


def fnv1a_32(text: str) -> int:
    """32-bit FNV-1a over the UTF-8 bytes of ``text``."""
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def _words(segment: str) -> list[str]:
    return [NUM_TOKEN if re.search(r"[0-9]", w) else w for w in _WORD.findall(segment)]


def tokenise(payee: str) -> list[str]:
    """Normalise payee text into tokens.

    ASCII letters are lower-cased; every other character except digits separates tokens. Masked
    phone numbers (``0700***007``) become ``<phone>``, any token holding a digit becomes
    ``<num>``, and runs of ``<num>`` collapse to one, so ids and numbers do not leak.
    """
    text = payee.translate(_ASCII_LOWER)
    raw: list[str] = []
    pos = 0
    for match in _MASKED_PHONE.finditer(text):
        raw.extend(_words(text[pos : match.start()]))
        raw.append(PHONE_TOKEN)
        pos = match.end()
    raw.extend(_words(text[pos:]))

    tokens: list[str] = []
    for token in raw:
        if token == NUM_TOKEN and tokens and tokens[-1] == NUM_TOKEN:
            continue
        tokens.append(token)
    return tokens


def token_ids(payee: str, cfg: FeatureConfig) -> list[int]:
    """Hashed, padded token ids of length ``cfg.max_tokens``."""
    ids = [fnv1a_32(t) % (cfg.hash_buckets - 1) + 1 for t in tokenise(payee)[: cfg.max_tokens]]
    return ids + [0] * (cfg.max_tokens - len(ids))


def amount_bucket(amount_minor: int) -> int:
    """Bucket of the absolute amount, in integer minor units."""
    amount = abs(amount_minor)
    return sum(1 for edge in AMOUNT_EDGES_MINOR if amount >= edge)


def month_phase(day: int) -> int:
    """0 = start (1st-5th, rent), 1 = middle, 2 = end (25th on, payday)."""
    if day <= 5:
        return 0
    if day >= 25:
        return 2
    return 1


def local_time(ts: datetime) -> datetime:
    """``ts`` in Nairobi time. Naive datetimes are refused: their meaning depends on the host."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("transaction timestamps must be timezone-aware")
    return ts.astimezone(NAIROBI)


def encode(tx: Transaction, cfg: FeatureConfig) -> list[int]:
    """Encode one transaction into the packed feature vector."""
    ts = local_time(tx.timestamp)
    return [
        *token_ids(tx.payee, cfg),
        amount_bucket(tx.amount_minor),
        1 if tx.direction is Direction.IN else 0,
        ts.hour // 4,
        ts.weekday(),
        month_phase(ts.day),
    ]


def feature_width(cfg: FeatureConfig) -> int:
    return cfg.max_tokens + N_CONTEXT_FEATURES


def feature_manifest(cfg: FeatureConfig) -> dict[str, Any]:
    """What the app needs to check that a model expects inputs it can produce."""
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_config": cfg.model_dump(),
        "input": {"name": "features", "dtype": "int32", "shape": [-1, feature_width(cfg)]},
        "layout": [f"token_{i}" for i in range(cfg.max_tokens)] + list(CONTEXT_LAYOUT),
    }


@dataclass(frozen=True)
class EncodedBatch:
    features: npt.NDArray[np.int32]
    labels: npt.NDArray[np.int32]


def encode_batch(
    txs: Sequence[Transaction], cfg: FeatureConfig, labels: Sequence[str]
) -> EncodedBatch:
    """Encode labelled transactions; ``labels`` is the categoriser's ordered label space."""
    index = {label: i for i, label in enumerate(labels)}
    x = np.zeros((len(txs), feature_width(cfg)), dtype=np.int32)
    y = np.zeros((len(txs),), dtype=np.int32)
    for row, tx in enumerate(txs):
        if tx.category is None or tx.category not in index:
            raise ValueError(f"transaction {row} has no target label: {tx.category!r}")
        x[row] = encode(tx, cfg)
        y[row] = index[tx.category]
    return EncodedBatch(features=x, labels=y)

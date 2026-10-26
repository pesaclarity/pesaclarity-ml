"""Feature extraction shared by training and the app.

The app must reproduce this exactly in Dart, so it is deliberately simple: FNV-1a hashing of
normalised payee tokens (no vocabulary file, so no payee list ships with the model) plus a few
bucketed numeric features, packed into one int32 vector per transaction.

Layout of the vector (length ``max_tokens + 5``)::

    [token_0 .. token_{max_tokens-1}, amount_bucket, direction, hour_bucket, weekday, month_phase]

Token id 0 is padding; real tokens hash into ``1 .. hash_buckets - 1``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from akili.config import FeatureConfig
from akili.types import Direction, Transaction

AMOUNT_EDGES_KES: tuple[float, ...] = (
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
)
"""Upper-exclusive bucket edges; amounts at or above the last edge share the top bucket."""

N_AMOUNT_BUCKETS = len(AMOUNT_EDGES_KES) + 1
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

NUM_TOKEN = "<num>"
PHONE_TOKEN = "<phone>"

_MASKED_PHONE = re.compile(r"\b\d{2,4}\s?\*{2,}\s?\d{2,4}\b")
_SPLIT = re.compile(r"[^0-9a-z]+")
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193


def fnv1a_32(text: str) -> int:
    """32-bit FNV-1a over the UTF-8 bytes of ``text``."""
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def tokenise(payee: str) -> list[str]:
    """Normalise payee text; digit runs become ``<num>`` so ids and numbers do not leak."""
    text = _MASKED_PHONE.sub(f" {PHONE_TOKEN} ", payee.lower())
    tokens: list[str] = []
    for raw in text.split():
        parts = [PHONE_TOKEN] if raw == PHONE_TOKEN else [p for p in _SPLIT.split(raw) if p]
        for part in parts:
            token = NUM_TOKEN if any(ch.isdigit() for ch in part) else part
            if token == NUM_TOKEN and tokens and tokens[-1] == NUM_TOKEN:
                continue
            tokens.append(token)
    return tokens


def token_ids(payee: str, cfg: FeatureConfig) -> list[int]:
    """Hashed, padded token ids of length ``cfg.max_tokens``."""
    ids = [fnv1a_32(t) % (cfg.hash_buckets - 1) + 1 for t in tokenise(payee)[: cfg.max_tokens]]
    return ids + [0] * (cfg.max_tokens - len(ids))


def amount_bucket(amount_kes: float) -> int:
    amount = abs(amount_kes)
    for i, edge in enumerate(AMOUNT_EDGES_KES):
        if amount < edge:
            return i
    return len(AMOUNT_EDGES_KES)


def month_phase(day: int) -> int:
    """0 = start (1st-5th, rent), 1 = middle, 2 = end (25th on, payday)."""
    if day <= 5:
        return 0
    if day >= 25:
        return 2
    return 1


def encode(tx: Transaction, cfg: FeatureConfig) -> list[int]:
    """Encode one transaction into the packed feature vector."""
    ts = tx.timestamp
    return [
        *token_ids(tx.payee, cfg),
        amount_bucket(tx.amount_kes),
        1 if tx.direction is Direction.IN else 0,
        ts.hour // 4,
        ts.weekday(),
        month_phase(ts.day),
    ]


def feature_width(cfg: FeatureConfig) -> int:
    return cfg.max_tokens + N_CONTEXT_FEATURES


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

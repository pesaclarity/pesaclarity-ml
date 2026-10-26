"""Shared record types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Direction(StrEnum):
    OUT = "out"
    IN = "in"


@dataclass(frozen=True, slots=True)
class Transaction:
    """One parsed transaction as the categoriser sees it. ``category`` is the label, if known."""

    payee: str
    amount_kes: float
    direction: Direction
    timestamp: datetime
    category: str | None = None
    user_id: str | None = None

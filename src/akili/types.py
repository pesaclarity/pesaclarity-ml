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
    """One parsed transaction as the categoriser sees it. ``category`` is the label, if known.

    ``amount_minor`` is in integer minor units (cents for KES), as the app's parser produces it.
    ``timestamp`` must be timezone-aware; features read it in Africa/Nairobi time.
    """

    payee: str
    amount_minor: int
    direction: Direction
    timestamp: datetime
    category: str | None = None
    user_id: str | None = None

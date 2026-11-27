"""The phone model's reply: one JSON line choosing a chart, then the answer.

The same rules as the app's ``slm_answer.dart`` (``SlmAnswer.parse`` and
``ungroundedNumbers``), so the evaluation counts what the app would show.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

CHARTS = ("bar", "line", "donut", "none")
DATA = (
    "top_categories",
    "top_payees",
    "in_out_weeks",
    "budget_vs_spent",
    "months_trend",
    "fees",
    "none",
)
PERIODS = ("since_payday", "this_month", "last_3_months")


@dataclass(frozen=True)
class Choice:
    chart: str
    data: str
    period: str

    def line(self) -> str:
        return json.dumps(
            {"chart": self.chart, "data": self.data, "period": self.period},
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class Parsed:
    choice: Choice | None
    text: str


def parse(reply: str) -> Parsed:
    """The app's ``SlmAnswer.parse``: the choice, or None when the first line is not the schema."""
    trimmed = reply.lstrip()
    first, _, rest = trimmed.partition("\n")
    first, rest = first.strip(), rest.strip()
    if not first.startswith("{"):
        return Parsed(None, reply.strip())
    try:
        m = json.loads(first)
    except ValueError:
        return Parsed(None, rest)
    if not isinstance(m, dict) or len(m) != 3:
        return Parsed(None, rest)
    c, d, p = m.get("chart"), m.get("data"), m.get("period")
    if c not in CHARTS or d not in DATA or p not in PERIODS:
        return Parsed(None, rest)
    return Parsed(Choice(c, d, p), rest)


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _plain(n: str) -> str:
    s = n.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def ungrounded_numbers(answer: str, facts: str) -> list[str]:
    """Numbers in ``answer`` that the facts do not have (the app's "Check this number")."""
    known = {_plain(m) for m in _NUMBER.findall(facts)}
    out = []
    for found in _NUMBER.findall(answer):
        n = found.rstrip(",")
        if _plain(n) not in known:
            out.append(n)
    return out


_ZERO = re.compile(r"(?<![\d.,])-?KES 0(?![\d.,])")
#: Sentences that say nothing about the person's money (the first version's stock lines).
_EMPTY = ("the chart shows it", "nice work")


def filler_sentences(answer: str) -> list[str]:
    """Sentences that quote an amount of zero ("You also paid KES 0 in fees") or say nothing."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+", answer.strip()):
        if s and (_ZERO.search(s) or any(e in s.lower() for e in _EMPTY)):
            out.append(s)
    return out

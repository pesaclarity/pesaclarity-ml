# ruff: noqa: E501, PLR0911, PLR0915
"""Training data for the phone language model: made-up months, fact sheets, questions, answers.

Nothing here is, or is derived from, real data. Each example is one fake person's month:

1. their records come from :func:`akili.synthetic.generate` (invented DEMO/SAMPLE/FAKE/TEST
   payees, amounts from its profiles), spread over the three whole months before a random
   "today" and the month so far, with made-up balances, bills and budgets;
2. the fact sheet is written by :mod:`akili.slm.facts`, exactly as the app writes it;
3. a question is drawn: a paraphrase of the app's presets or another money question, in English
   and some Swahili and Sheng;
4. the target reply is written by rules: the right chart (one JSON line), then two or three
   sentences that quote amounts from the fact sheet as written. No number is worked out here.

The split is by person: test people never appear in training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from akili.categories import CategoryType, default_categories
from akili.config import SyntheticDataConfig
from akili.slm.answer import Choice
from akili.slm.facts import Budget, Entry, FactsInput, Line, facts, sections_text, system_text
from akili.synthetic import generate
from akili.types import Direction

# ---- made-up months ----------------------------------------------------------------------------

_ID_TAIL = re.compile(r"\s+(?:(?:TILL|PAYBILL)\s+\d+.*|\d+\s+0700\*+\d+)$")
_ACCOUNTS = ("M-Pesa", "KCB", "Equity", "Airtel Money", "Cash")
_SAVINGS = ("M-Shwari", "SACCO", "Money market")
_BILLS = ("Rent", "Internet", "Electricity", "School fees", "Water", "Chama", "Insurance")


def _payee(raw: str) -> str:
    return _ID_TAIL.sub("", raw).strip()


def month_of(rng: np.random.Generator, pool: list, today: date) -> FactsInput:  # type: ignore[type-arg]
    """One fake person's records up to ``today``, from their synthetic transactions."""
    cats = default_categories()
    first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    for _ in range(2):
        first = (first - timedelta(days=1)).replace(day=1)
    span = (today - first).days + 1
    entries: list[Entry] = []
    salary_days: list[date] = []
    payday_day = int(rng.integers(20, 29))
    for tx in pool:
        c = cats.get(tx.category) if tx.category else None
        kind = c.type if c else CategoryType.EXPENSE
        if tx.category == "salary":
            # Paid on the same day each month.
            month_start = first
            for _ in range(int(rng.integers(0, 4))):
                month_start = (month_start + timedelta(days=32)).replace(day=1)
            day = month_start.replace(day=payday_day)
            if day > today or day in salary_days:
                continue
            salary_days.append(day)
        else:
            day = first + timedelta(days=int(rng.integers(0, span)))
        name = c.name if c else None
        if tx.direction == Direction.IN:
            income = tx.amount_minor if kind == CategoryType.INCOME else 0
            spending = -tx.amount_minor if kind == CategoryType.EXPENSE else 0
        else:
            income = 0
            spending = tx.amount_minor if kind == CategoryType.EXPENSE else 0
        if tx.category in ("own_transfer",):
            income = spending = 0
        if rng.random() < 0.06:
            name = None  # not sorted yet
        fee = 0
        if tx.direction == Direction.OUT and rng.random() < 0.5:
            fee = int(rng.choice([0, 700, 1300, 2300, 2900, 5300, 5700, 10000]))
        entries.append(Entry(day, income, spending, fee, name, _payee(tx.payee)))
    entries.sort(key=lambda e: e.day)
    payday = max((d for d in salary_days), default=None)
    if payday is None and rng.random() < 0.3:
        payday = today - timedelta(days=int(rng.integers(0, 30)))

    def amount(lo: int, hi: int) -> int:
        v = int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return v * 100 if rng.random() < 0.8 else v * 100 + int(rng.integers(1, 100))

    n_acc = int(rng.integers(0, 4))
    balances = [Line(n, amount(50, 150_000)) for n in rng.choice(_ACCOUNTS, n_acc, replace=False)]
    if n_acc and rng.random() < 0.5:
        balances.append(Line(str(rng.choice(_SAVINGS)), amount(500, 400_000), savings=True))
    bills = [
        Line(
            str(n),
            amount(300, 40_000),
            None if rng.random() < 0.15 else today + timedelta(days=int(rng.integers(-5, 30))),
        )
        for n in rng.choice(_BILLS, int(rng.integers(0, 4)), replace=False)
    ]
    spent: dict[str, int] = {}
    for e in entries:
        if e.day >= today.replace(day=1) and e.category and e.spending_minor > 0:
            spent[e.category] = spent.get(e.category, 0) + e.spending_minor
    budgets = []
    if spent and rng.random() < 0.6:
        names = list(spent)
        for n in rng.choice(names, min(len(names), int(rng.integers(1, 5))), replace=False):
            used = spent[str(n)]
            planned = max(100, round(used * rng.uniform(0.6, 1.6) / 50_000)) * 50_000
            budgets.append(Budget(str(n), planned, used))
    fees = int(rng.integers(0, 2000)) * 100 if rng.random() < 0.3 else None
    return FactsInput(today, entries, payday, balances, bills, budgets, fees)


# ---- questions ---------------------------------------------------------------------------------

#: Kinds of question and their paraphrases. The app's presets are the first of their kinds.
QUESTIONS: dict[str, tuple[str, ...]] = {
    "where": (
        "Where did my money go this month?",
        "Where is my money going?",
        "What did I spend the most on this month?",
        "Break down my spending this month",
        "What are my biggest expenses?",
        "where did all my money go",
        "Show me my spending by category",
        "Pesa yangu imeenda wapi mwezi huu?",
        "Nimetumia pesa kwa nini sana mwezi huu?",
        "Doh yangu imeenda wapi?",
        "Niambie matumizi yangu makubwa",
    ),
    "where_payday": (
        "Where has my money gone since payday?",
        "What have I spent on since I got paid?",
        "Since my salary came in, where did it go?",
        "Tangu mshahara uingie, pesa imeenda wapi?",
        "Mshahara imeenda wapi?",
    ),
    "share": (
        "What share of my spending goes to each category?",
        "Show my spending as a pie chart",
        "How is my spending split up?",
        "Matumizi yangu yamegawanyika vipi?",
    ),
    "payees": (
        "Who did I pay the most this month?",
        "Which shops and people got most of my money?",
        "Who am I sending money to the most?",
        "Top payees this month?",
        "Nimelipa nani pesa nyingi mwezi huu?",
        "Nani amekula doh yangu sana?",
    ),
    "spend_less": (
        "How can I spend less?",
        "How do I save more money?",
        "Where can I cut back?",
        "Give me tips to reduce my spending",
        "I want to spend less next month, any ideas?",
        "Nifanye nini nitumie pesa kidogo?",
        "Naweza punguza matumizi wapi?",
        "Nisaidie ku-save doh",
    ),
    "budget": (
        "Am I on track with my budget?",
        "How am I doing against my budget?",
        "Have I gone over any budget?",
        "Is my budget okay this month?",
        "Bajeti yangu iko aje?",
        "Nimezidi bajeti?",
    ),
    "trend": (
        "How has my spending changed over the last 3 months?",
        "Am I spending more than before?",
        "Compare my spending month by month",
        "Show my spending trend",
        "Matumizi yangu yamebadilika vipi miezi mitatu iliyopita?",
    ),
    "weeks": (
        "How much came in and went out each week?",
        "Show money in and out by week this month",
        "Week by week, what did I earn and spend?",
        "Pesa iliingia na kutoka kiasi gani kila wiki?",
    ),
    "weeks_payday": (
        "Money in versus money out since payday?",
        "How much have I spent compared to what came in since I got paid?",
        "Tangu payday, pesa iliingia ngapi na nikatumia ngapi?",
    ),
    "fees": (
        "How much did I pay in fees this month?",
        "Are transaction charges eating my money?",
        "How much am I losing to M-Pesa charges?",
        "Nimelipa charges ngapi mwezi huu?",
    ),
    "left": (
        "How much money do I have left?",
        "What's my balance?",
        "How much is in my accounts?",
        "Nina pesa ngapi sasa?",
        "Niko na doh ngapi?",
    ),
    "bills": (
        "What bills are coming up?",
        "What do I need to pay soon?",
        "Any bills due?",
        "Nina bili gani zinakuja?",
    ),
    "offtopic": (
        "What will the weather be tomorrow?",
        "Will the shilling go up next year?",
        "Which stock should I buy?",
        "Tell me a joke",
        "How much does my neighbour earn?",
        "Mvua itanyesha kesho?",
    ),
}

# How often each kind is asked: the presets most.
_WEIGHTS = {
    "where": 3, "where_payday": 1, "share": 1, "payees": 1.5, "spend_less": 2.5, "budget": 2.5,
    "trend": 1.5, "weeks": 1, "weeks_payday": 1, "fees": 1, "left": 1, "bills": 1, "offtopic": 1,
}  # fmt: skip

# ---- answers -----------------------------------------------------------------------------------


@dataclass
class _Sheet:
    """The fact sheet as the writer reads it: each section's lines, exactly as in the text."""

    lines: dict[str, list[str]]

    def get(self, title: str) -> list[str]:
        return self.lines.get(title, [])

    def flow(self, prefix: str) -> tuple[str, str, str, str] | None:
        for line in self.get("Money in and out"):
            if line.startswith(prefix):
                m = re.match(r"(.*): money in (.*), spent (.*), left (.*)$", line)
                if m:
                    return m.group(1), m.group(2), m.group(3), m.group(4)
        return None


def _split(line: str) -> tuple[str, str]:
    name, _, value = line.rpartition(": ")
    return name, value


def _pick(rng: np.random.Generator, options: tuple[str, ...] | list[str]) -> str:
    return str(options[int(rng.integers(0, len(options)))])


_NOT_SURE = (
    "I am not sure: the facts I have do not cover that.",
    "I'm not sure. Your records here don't answer that question.",
    "I am not sure about that; I can only see your own money facts.",
)


def _none(period: str = "this_month") -> Choice:
    return Choice("none", "none", period)


#: The chart each kind of question gets. It depends on the question alone: when the records lack
#: what it shows (no budgets, no spending yet), the app's chart is simply empty and not drawn, and
#: the words say so. ``since_payday`` falls back to this month in the app when there is no payday.
CHART_FOR_KIND: dict[str, Choice] = {
    "where": Choice("bar", "top_categories", "this_month"),
    "where_payday": Choice("bar", "top_categories", "since_payday"),
    "share": Choice("donut", "top_categories", "this_month"),
    "payees": Choice("bar", "top_payees", "this_month"),
    "spend_less": Choice("bar", "top_categories", "this_month"),
    "budget": Choice("bar", "budget_vs_spent", "this_month"),
    "trend": Choice("line", "months_trend", "last_3_months"),
    "weeks": Choice("bar", "in_out_weeks", "this_month"),
    "weeks_payday": Choice("bar", "in_out_weeks", "since_payday"),
    "fees": Choice("donut", "fees", "this_month"),
    "left": Choice("none", "none", "this_month"),
    "bills": Choice("none", "none", "this_month"),
    "offtopic": Choice("none", "none", "this_month"),
}


def write(kind: str, sheet: _Sheet, rng: np.random.Generator) -> tuple[Choice, str]:
    """The target reply for a question of ``kind`` on ``sheet``: its chart and 2-3 sentences."""
    _, text = _words(kind, sheet, rng)
    return CHART_FOR_KIND[kind], text


def _kes(v: str) -> float:
    """An amount as written (``KES 1,200.50``, ``-KES 30``) as a number, to compare amounts.

    Only the rules compare; the reply still quotes amounts exactly as written, never a result.
    """
    m = re.fullmatch(r"(-?)KES ([\d,]+(?:\.\d+)?)", v.strip())
    if not m:
        return 0.0
    x = float(m.group(2).replace(",", ""))
    return -x if m.group(1) else x


def _zero(v: str) -> bool:
    return _kes(v) == 0


@dataclass(frozen=True)
class _Month:
    name: str
    money_in: str
    spent: str


def _months(sheet: _Sheet) -> list[_Month]:
    out = []
    for t in sheet.get("Last 3 months"):
        m = re.match(r"(.*): money in (.*), spent (.*)$", t)
        if m:
            out.append(_Month(m.group(1), m.group(2), m.group(3)))
    return out


def _spent_in(m: _Month) -> str:
    return f"nothing in {m.name}" if _zero(m.spent) else f"{m.spent} in {m.name}"


def _against_last_month(p, total: str, last: _Month | None) -> str | None:  # type: ignore[no-untyped-def]
    """This month's spending so far set against last month's whole month, when both exist."""
    if last is None or _zero(total) or _zero(last.spent):
        return None
    if _kes(total) > _kes(last.spent):
        return p(
            f"That is already more than the {last.spent} you spent in all of {last.name}.",
            f"You have already passed {last.name}'s {last.spent}, and the month is not over.",
        )
    if _kes(total) == _kes(last.spent):
        return f"That matches the {last.spent} you spent in all of {last.name}."
    return p(
        f"That is still below the {last.spent} you spent in all of {last.name}.",
        f"So far that is less than {last.name}'s {last.spent}.",
    )


def _words(kind: str, sheet: _Sheet, rng: np.random.Generator) -> tuple[Choice, str]:
    """The reply's sentences. They answer the question first, then say what stands out and why
    it matters, comparing amounts (larger, smaller, more than half, more than last month) by
    rule. Every amount is quoted exactly as the facts write it; no number is worked out, and an
    amount of zero is never quoted (it says nothing)."""
    p = lambda *o: _pick(rng, o)  # noqa: E731
    cats = [_split(x) for x in sheet.get("Top spending this month")]
    payees = [_split(x) for x in sheet.get("Top payees this month")]
    month = sheet.flow("This month so far")
    payday = sheet.flow("Since payday")
    fees = sheet.get("Fees this month")
    budgets = sheet.get("Budgets this month")
    months = _months(sheet)
    last = months[-1] if months else None
    total = month[2] if month else None

    if kind in ("where", "where_payday", "share"):
        if not cats:
            return _none(), p(
                "I am not sure: no spending is recorded this month yet.",
                "There is no spending recorded this month yet, so I am not sure where it went.",
            )
        period = "since_payday" if kind == "where_payday" and payday else "this_month"
        chart = "donut" if kind == "share" else "bar"
        n1, v1 = cats[0]
        share = _kes(v1) / _kes(total) if total and _kes(total) > 0 else 0.0
        s: list[str] = []
        if total and _kes(v1) == _kes(total):
            s.append(p(f"All of your spending this month went to {n1}: {v1}.",
                       f"Everything you have spent this month, {v1}, went to {n1}."))  # fmt: skip
            share = 0.0
            cats = cats[:1]
        elif kind == "share" and total and share:
            s.append(p(f"{n1} takes the biggest share: {v1} of the {total} you have spent this month.",
                       f"The largest slice is {n1}, {v1} out of {total} this month."))  # fmt: skip
        elif total and share:
            s.append(p(f"Most of your money this month went to {n1}: {v1} of the {total} you have spent.",
                       f"{n1} is your biggest spend this month, {v1} out of {total} in all.",
                       f"Your money went mainly to {n1}, at {v1} of {total} spent this month."))  # fmt: skip
        else:
            s.append(f"{n1} is your biggest spend this month, at {v1}.")
        n2, v2 = cats[1] if len(cats) > 1 else ("", "")
        if not share:
            pass
        elif n1 == "Not sorted":
            s.append(
                "Those entries are not sorted yet, so sorting them would show what they really are."
            )
        elif share >= 0.75:
            s.append(p("That is almost all of it, so the rest of your spending is small.",
                       "Almost everything went there, so that one category decides your month."))  # fmt: skip
        elif share > 0.5:
            s.append(p("That is more than half of your spending, so it matters most.",
                       "It is more than half of the total, so the other categories are much smaller."))  # fmt: skip
        elif n2:
            if _kes(v1) >= 2 * _kes(v2):
                s.append(f"It is more than twice the next one, {n2} at {v2}.")
            else:
                s.append(p(f"{n2} is close behind at {v2}, so your spending is spread out.",
                           f"Next is {n2} at {v2}, not far behind."))  # fmt: skip
        if kind == "where_payday" and payday:
            if _zero(payday[2]) and _zero(payday[1]):
                s.append("Since payday nothing has come in or been spent yet.")
            elif _zero(payday[2]):
                s.append(f"Since payday nothing has been spent yet, and {payday[1]} came in.")
            elif _zero(payday[1]):
                s.append(f"Since payday you have spent {payday[2]}, and nothing new has come in.")
            elif _kes(payday[3]) < 0:
                s.append(
                    f"Since payday you have spent {payday[2]}, more than the {payday[1]} that came in."
                )
            else:
                s.append(
                    f"Since payday you have spent {payday[2]} of the {payday[1]} that came in."
                )
        else:
            c = _against_last_month(p, total, last) if total else None
            if c:
                s.append(c)
        return Choice(chart, "top_categories", period), " ".join(s[:3])

    if kind == "payees":
        if not payees:
            return _none(), "I am not sure: no payments to named payees are recorded this month."
        n1, v1 = payees[0]
        s = [p(f"You paid {n1} the most this month, {v1}.",
               f"{n1} got the most from you this month: {v1}.")]  # fmt: skip
        if len(payees) > 1:
            n2, v2 = payees[1]
            if _kes(v1) >= 2 * _kes(v2):
                s.append(f"That is more than twice the next, {n2} at {v2}.")
            elif _kes(v1) == _kes(v2):
                s.append(f"{n2} got the same, {v2}.")
            else:
                s.append(f"{n2} is close behind at {v2}.")
        if total and _kes(total) > 0 and _kes(v1) / _kes(total) > 0.5:
            s.append(f"That one payee is more than half of the {total} you have spent.")
        return Choice("bar", "top_payees", "this_month"), " ".join(s[:3])

    if kind == "spend_less":
        if not cats:
            return _none(), (
                "I am not sure yet: no spending is recorded this month. "
                "Once there is, I can show where to cut back."
            )
        n1, v1 = cats[0]
        if total and not _zero(total):
            s = [p(f"Start with {n1}: at {v1} of the {total} you have spent this month, it is the biggest.",
                   f"{n1} is the first place to look, {v1} of your {total} this month.")]  # fmt: skip
        else:
            s = [f"Start with {n1}: at {v1} it is your biggest spend this month."]
        over = [b for b in budgets if " over by " in b]
        if over:
            name, rest = over[0].split(": planned ", 1)
            by = rest.rsplit("over by ", 1)[1]
            s.append(f"You are also over your {name} budget by {by}, so hold back there first.")
        elif len(cats) > 1:
            n2, v2 = cats[1]
            s.append(p(f"{n2} at {v2} is next, so a small cut there helps too.",
                       f"After that, look at {n2}, which is {v2}."))  # fmt: skip
        c = _against_last_month(p, total, last) if total else None
        f = fees[0] if fees else "KES 0"
        if c and total and last and _kes(total) > _kes(last.spent):
            s.append(f"You have already spent more than in {last.name}, so cutting now matters.")
        elif not _zero(f) and total and _kes(f) >= 0.03 * _kes(total):
            s.append(f"Your fees of {f} are worth cutting too: fewer, larger transfers cost less.")
        return Choice("bar", "top_categories", "this_month"), " ".join(s[:3])

    if kind == "budget":
        if not budgets:
            return _none(), p(
                "I am not sure: you have not set any budgets for this month.",
                "There are no budgets set this month, so I am not sure. You can add one in Plan.",
            )
        s = []
        over = [b for b in budgets if " over by " in b]
        under = []
        for b in budgets:
            if " over by " in b:
                continue
            name, rest = b.split(": planned ", 1)
            planned = rest.partition(", spent ")[0]
            left = rest.rsplit("left ", 1)[1]
            under.append((name, planned, left))
        if over:
            name, rest = over[0].split(": planned ", 1)
            planned = rest.partition(", spent ")[0]
            by = rest.rsplit("over by ", 1)[1]
            s.append(p(f"Not quite: you planned {planned} for {name} and are over by {by}.",
                       f"No, {name} is over its budget: {by} more than the {planned} you planned."))  # fmt: skip
            if len(over) > 1:
                s.append(
                    f"{over[1].split(': planned ')[0]} is over too, so those two need attention."
                )
        else:
            s.append(p("Yes, you are within all your budgets so far this month.",
                       "You are on track: no budget is over so far this month."))  # fmt: skip
        if under:
            name, planned, left = min(under, key=lambda u: _kes(u[2]) / max(_kes(u[1]), 1))
            if _zero(left):
                s.append(f"{name} is used up: you have spent all of its {planned}.")
            elif _kes(left) < 0.25 * _kes(planned):
                s.append(f"{name} is close to its limit, with only {left} left of {planned}.")
            elif len(under) > 1:
                s.append(f"Even {name}, the tightest, still has {left} of {planned} left.")
            else:
                s.append(f"{name} still has {left} of its {planned} left.")
        return Choice("bar", "budget_vs_spent", "this_month"), " ".join(s[:3])

    if kind == "trend":
        if len(months) < 3:
            return _none("last_3_months"), _pick(rng, _NOT_SURE)
        a, b, c = (_kes(m.spent) for m in months)
        listed = f"{_spent_in(months[0])}, {_spent_in(months[1])} and {_spent_in(months[2])}"
        if a < b < c:
            s = [f"Your spending has gone up each month: {listed}."]
        elif a > b > c:
            s = [f"Your spending has gone down each month: {listed}."]
        else:
            top = max(months, key=lambda m: _kes(m.spent))
            s = [f"It has gone up and down: {listed}.", f"{top.name} was the highest."]
        if total and not _zero(total) and last and not _zero(last.spent):
            if _kes(total) > _kes(last.spent):
                s.append(f"This month you have already spent {total}, more than in {last.name}.")
            else:
                s.append(f"So far this month you have spent {total}, below {last.name}.")
        return Choice("line", "months_trend", "last_3_months"), " ".join(s[:3])

    if kind in ("weeks", "weeks_payday"):
        use = payday if kind == "weeks_payday" and payday else month
        period = "since_payday" if use is payday and payday else "this_month"
        if use is None:
            return _none(), _pick(rng, _NOT_SURE)
        label = "Since payday" if period == "since_payday" else "This month"
        money_in, out, left = use[1], use[2], use[3]
        if _zero(money_in) and _zero(out):
            return Choice("bar", "in_out_weeks", period), (
                f"{label}, nothing has come in or gone out yet, so there is nothing to compare."
            )
        if _zero(money_in):
            s = [f"{label}, nothing has come in yet and you have spent {out}.",
                 "So everything you spend is coming from money you already had."]  # fmt: skip
        elif _zero(out):
            s = [f"{label}, {money_in} came in and you have not spent anything yet."]
        else:
            s = [f"{label}, {money_in} came in and you spent {out}."]
            if _kes(left) < 0:
                s.append(f"You are short by {left[1:]}, so spending is ahead of money in.")
            elif _kes(out) < 0.5 * _kes(money_in):
                s.append(f"You have spent less than half of it, which leaves {left}.")
            else:
                s.append(f"That leaves {left}, so most of what came in is already spent.")
        return Choice("bar", "in_out_weeks", period), " ".join(s)

    if kind == "fees":
        f = fees[0] if fees else "KES 0"
        if _zero(f):
            return Choice("donut", "fees", "this_month"), p(
                "You paid no fees this month, so charges are not costing you anything.",
                "No fees this month: charges are not eating into your money.",
            )
        s = [p(f"You paid {f} in fees and charges this month.",
               f"Fees and charges came to {f} this month.")]  # fmt: skip
        if total and not _zero(total):
            if _kes(f) >= 0.05 * _kes(total):
                s.append(
                    f"That is a noticeable part of the {total} you have spent, so it is worth cutting."
                )
            elif _kes(f) < 0.01 * _kes(total):
                s.append(
                    f"That is small next to the {total} you have spent, so charges are not the main drain."
                )
            else:
                s.append(f"Next to the {total} you have spent it is modest, but it still adds up.")
        s.append(p("Fewer, larger transfers and paying tills directly cost less.",
                   "Paying by till or paybill, rather than many small transfers, usually costs less."))  # fmt: skip
        return Choice("donut", "fees", "this_month"), " ".join(s[:3])

    if kind == "left":
        bal = sheet.get("Balances now")
        bills = sheet.get("Upcoming bills")
        if not bal:
            if month and not _zero(month[1]):
                if _kes(month[3]) < 0:
                    return _none(), (
                        f"I don't see your account balances, but this month you have spent "
                        f"{month[2]}, more than the {month[1]} that came in."
                    )
                return _none(), (
                    f"I don't see your account balances, but this month {month[1]} came in "
                    f"and you spent {month[2]}, which leaves {month[3]}."
                )
            return _none(), _pick(rng, _NOT_SURE)
        day = _split(bal[0])[1]
        sav = [_split(b)[1] for b in bal if b.startswith("Savings accounts: ")]
        s = [f"You have {day} in your day-to-day accounts." if not _zero(day)
             else "Your day-to-day accounts are empty right now."]  # fmt: skip
        if sav and not _zero(sav[0]):
            s.append(f"Your savings hold another {sav[0]}.")
        if bills:
            bname, brest = bills[0].split(": ", 1)
            bamt = brest.split(", due ")[0]
            if _kes(day) >= _kes(bamt):
                s.append(f"That covers your next bill, {bname} at {bamt}.")
            else:
                s.append(f"That is less than your next bill, {bname} at {bamt}, so plan for it.")
        return _none(), " ".join(s[:3])

    if kind == "bills":
        bills = sheet.get("Upcoming bills")
        if not bills:
            return _none(), "I don't see any upcoming bills in your records."
        s = [f"Your next bill is {bills[0]}."]
        if len(bills) > 1:
            s.append(f"After that: {bills[1]}.")
        bal = sheet.get("Balances now")
        if bal:
            day = _split(bal[0])[1]
            amt = bills[0].split(": ", 1)[1].split(", due ")[0]
            if _kes(day) >= _kes(amt):
                s.append(f"Your day-to-day accounts hold {day}, enough for it.")
            elif _zero(day):
                s.append("Your day-to-day accounts are empty, so set money aside for it.")
            else:
                s.append(
                    f"Your day-to-day accounts hold {day}, less than that, so set money aside."
                )
        return _none(), " ".join(s[:3])

    return _none(), _pick(rng, _NOT_SURE)


# ---- examples ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    person: str
    kind: str
    question: str
    system: str
    facts: str
    target: str
    choice: Choice

    def to_json(self) -> dict:  # type: ignore[type-arg]
        return {
            "person": self.person,
            "kind": self.kind,
            "question": self.question,
            "system": self.system,
            "facts": self.facts,
            "target": self.target,
            "choice": self.choice.__dict__,
        }


def _vary(rng: np.random.Generator, q: str) -> str:
    """Small surface changes people make: case, punctuation."""
    r = rng.random()
    if r < 0.15:
        return q.lower()
    if r < 0.25:
        return q.rstrip("?") if q.endswith("?") else q + "?"
    return q


def examples(n_people: int, seed: int, questions_per_person: int = 1) -> list[Example]:
    """``n_people`` made-up months (one fact sheet each) with their questions and target replies."""
    rng = np.random.default_rng(seed)
    pool = generate(
        SyntheticDataConfig(
            n_users=n_people, transactions_per_user=70, payee_style="varied", seed=seed
        )
    )
    by_user: dict[str, list] = {}  # type: ignore[type-arg]
    for tx in pool:
        by_user.setdefault(tx.user_id or "", []).append(tx)
    kinds = list(QUESTIONS)
    w = np.array([_WEIGHTS[k] for k in kinds], dtype=float)
    w /= w.sum()
    out = []
    for user, txs in by_user.items():
        today = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 730)))
        # Some people have few records this month: keep a random share.
        keep = txs[: int(rng.integers(max(1, len(txs) // 4), len(txs) + 1))]
        inp = month_of(rng, keep, today)
        secs = facts(inp)
        text = sections_text(secs)
        sheet = _Sheet({s.title: list(s.lines) for s in secs})
        for _ in range(questions_per_person):
            kind = kinds[int(rng.choice(len(kinds), p=w))]
            q = _vary(rng, _pick(rng, QUESTIONS[kind]))
            choice, answer = write(kind, sheet, rng)
            out.append(
                Example(
                    f"{seed}:{user}",
                    kind,
                    q,
                    system_text(text),
                    text,
                    f"{choice.line()}\n{answer}",
                    choice,
                )
            )
    return out


def write_jsonl(rows: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def build(out: Path, n_train: int = 3000, n_test: int = 300, seed: int = 20261126) -> None:
    """``train.jsonl`` and ``test.jsonl``: different people (different seeds), never mixed."""
    write_jsonl(examples(n_train, seed), out / "train.jsonl")
    write_jsonl(examples(n_test, seed + 1), out / "test.jsonl")


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1] if len(sys.argv) > 1 else "artefacts/slm/data"))

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


def write(kind: str, sheet: _Sheet, rng: np.random.Generator) -> tuple[Choice, str]:
    """The target reply for a question of ``kind`` on ``sheet``: a chart choice and 2-3 sentences."""
    p = lambda *o: _pick(rng, o)  # noqa: E731
    cats = [_split(x) for x in sheet.get("Top spending this month")]
    payees = [_split(x) for x in sheet.get("Top payees this month")]
    month = sheet.flow("This month so far")
    payday = sheet.flow("Since payday")
    fees = sheet.get("Fees this month")
    budgets = sheet.get("Budgets this month")

    if kind in ("where", "where_payday", "share"):
        if not cats:
            return _none(), p(
                "I am not sure: no spending is recorded this month yet.",
                "There is no spending recorded this month yet, so I am not sure where it went.",
            )
        period = "since_payday" if kind == "where_payday" and payday else "this_month"
        chart = "donut" if kind == "share" else "bar"
        (n1, v1) = cats[0]
        s = [p(f"Most of your spending this month went to {n1}, at {v1}.",
               f"{n1} is your biggest spend this month: {v1}.",
               f"Your largest category this month is {n1} with {v1}.")]  # fmt: skip
        if len(cats) > 1:
            n2, v2 = cats[1]
            s.append(p(f"Next is {n2} at {v2}.", f"After that comes {n2}, {v2}.",
                       f"{n2} follows with {v2}."))  # fmt: skip
        if kind == "where_payday" and payday:
            s.append(f"Since payday you have spent {payday[2]} and {payday[1]} came in.")
        elif month:
            s.append(p(f"In all you have spent {month[2]} so far this month.",
                       f"Your total spending this month is {month[2]}."))  # fmt: skip
        return Choice(chart, "top_categories", period), " ".join(s[:3])

    if kind == "payees":
        if not payees:
            return _none(), "I am not sure: no payments to named payees are recorded this month."
        n1, v1 = payees[0]
        s = [p(f"You paid {n1} the most this month, {v1}.",
               f"{n1} got the most from you this month: {v1}.")]  # fmt: skip
        if len(payees) > 1:
            n2, v2 = payees[1]
            s.append(p(f"Next is {n2} with {v2}.", f"Then {n2}, at {v2}."))
        return Choice("bar", "top_payees", "this_month"), " ".join(s)

    if kind == "spend_less":
        if not cats:
            return _none(), (
                "I am not sure yet: no spending is recorded this month. "
                "Once there is, I can show where to cut back."
            )
        n1, v1 = cats[0]
        s = [p(f"Your biggest spend this month is {n1} at {v1}, so that is the first place to look.",
               f"Start with {n1}: it is {v1} this month, more than anything else.",
               f"{n1} takes the most, {v1} this month, so small cuts there help most.")]  # fmt: skip
        over = [b for b in budgets if " over by " in b]
        if over:
            name, rest = over[0].split(": planned ", 1)
            by = rest.rsplit("over by ", 1)[1]
            s.append(f"You are over your {name} budget by {by}, so try to hold back there.")
        elif len(cats) > 1:
            n2, v2 = cats[1]
            s.append(p(f"{n2} at {v2} is next.", f"Then look at {n2}, which is {v2}."))
        if fees and fees[0] != "KES 0" and len(s) < 3:
            s.append(
                f"You also paid {fees[0]} in fees; sending larger amounts less often can cut them."
            )
        return Choice("bar", "top_categories", "this_month"), " ".join(s[:3])

    if kind == "budget":
        if not budgets:
            return _none(), p(
                "I am not sure: you have not set any budgets for this month.",
                "There are no budgets set this month, so I am not sure. You can add one in Plan.",
            )
        s = []
        over = [b for b in budgets if " over by " in b]
        under = [b for b in budgets if " over by " not in b]
        if over:
            name, rest = over[0].split(": planned ", 1)
            planned = rest.partition(", spent ")[0]
            by = rest.rsplit("over by ", 1)[1]
            s.append(
                f"You are over your {name} budget: you planned {planned} and are over by {by}."
            )
            if len(over) > 1:
                s.append(f"{over[1].split(': planned ')[0]} is over too.")
        else:
            s.append(p("You are within all your budgets so far this month.",
                       "Good news: no budget is over so far this month."))  # fmt: skip
        if under:
            name, rest = under[0].split(": planned ", 1)
            left = rest.rsplit("left ", 1)[1]
            s.append(f"{name} still has {left} left.")
        return Choice("bar", "budget_vs_spent", "this_month"), " ".join(s[:3])

    if kind == "trend":
        trend = sheet.get("Last 3 months")
        if len(trend) < 3:
            return _none("last_3_months"), _pick(rng, _NOT_SURE)
        vals = [re.match(r"(.*): money in (.*), spent (.*)$", t) for t in trend]
        if not all(vals):
            return _none("last_3_months"), _pick(rng, _NOT_SURE)
        ms = [(v.group(1), v.group(3)) for v in vals if v]
        s = [
            f"You spent {ms[0][1]} in {ms[0][0]}, {ms[1][1]} in {ms[1][0]} and {ms[2][1]} in {ms[2][0]}."
        ]
        if month:
            s.append(f"So far this month you have spent {month[2]}.")
        return Choice("line", "months_trend", "last_3_months"), " ".join(s)

    if kind in ("weeks", "weeks_payday"):
        use = payday if kind == "weeks_payday" and payday else month
        period = "since_payday" if use is payday and payday else "this_month"
        if use is None:
            return _none(), _pick(rng, _NOT_SURE)
        label = "Since payday" if period == "since_payday" else "This month"
        s = [f"{label}, {use[1]} came in and you spent {use[2]}."]
        left = use[3]
        if left.startswith("-"):
            s.append(f"That leaves you short by {left[1:]}, so spending is ahead of money in.")
        else:
            s.append(f"That leaves {left}.")
        s.append("The chart shows it week by week.")
        return Choice("bar", "in_out_weeks", period), " ".join(s)

    if kind == "fees":
        f = fees[0] if fees else "KES 0"
        if f == "KES 0":
            return _none(), "You paid no fees this month. Nice work keeping charges at zero."
        s = [p(f"You paid {f} in fees and charges this month.",
               f"Fees and charges came to {f} this month.")]  # fmt: skip
        s.append(p("Sending money in fewer, larger amounts and paying tills directly can cut them.",
                   "Paying by till or paybill, rather than many small transfers, usually costs less."))  # fmt: skip
        return Choice("donut", "fees", "this_month"), " ".join(s)

    if kind == "left":
        bal = sheet.get("Balances now")
        if not bal:
            if month:
                return _none(), (
                    f"I don't see your account balances, but this month {month[1]} came in "
                    f"and you spent {month[2]}, which leaves {month[3]}."
                )
            return _none(), _pick(rng, _NOT_SURE)
        day = _split(bal[0])[1]
        s = [f"You have {day} in your day-to-day accounts."]
        sav = [b for b in bal if b.startswith("Savings accounts: ")]
        if sav:
            s.append(f"Your savings accounts hold {_split(sav[0])[1]}.")
        return _none(), " ".join(s)

    if kind == "bills":
        bills = sheet.get("Upcoming bills")
        if not bills:
            return _none(), "I don't see any upcoming bills in your records."
        s = [f"Your next bill is {bills[0]}."]
        if len(bills) > 1:
            s.append(f"After that: {bills[1]}.")
        return _none(), " ".join(s)

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

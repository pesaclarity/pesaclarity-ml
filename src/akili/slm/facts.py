# ruff: noqa: PLR0915
"""The fact sheet the app gives the phone language model, written exactly as the app writes it.

A port of the app's ``SlmFacts.of`` (pesaclarity-app,
``lib/src/features/slm_experiment/domain/slm_facts.dart``). The input is the app's, reduced to
what the facts read: each ledger entry's date, what it adds to income and to spending (the app's
``incomeOf`` and ``spendingOf``), its fee, its category's name (``None`` for not sorted) and its
payee. ``tests/test_slm_facts.py`` checks this gives the app's text, word for word, on the cases
in ``tests/fixtures/slm_facts_parity.json``, which the app's own test writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

MAX_CHARS = 4200
TOP_COUNT = 5

#: The system text, exactly the app's ``slmSystemPrompt``.
SYSTEM_PROMPT = (
    "You help one person understand their own money. Use only the facts "
    "below; they were worked out by the app and are correct. First write "
    'one line of JSON that picks a chart: {"chart":"bar|line|donut|none",'
    '"data":"top_categories|top_payees|in_out_weeks|budget_vs_spent|'
    'months_trend|fees|none","period":"since_payday|this_month|'
    'last_3_months"}. Then answer in 2 or 3 short, plain sentences. Do not '
    "add, subtract or invent any number: quote amounts exactly as written. "
    "If the facts do not answer the question, say you are not sure and pick "
    '"none".'
)

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class Entry:
    day: date
    income_minor: int = 0
    spending_minor: int = 0
    fee_minor: int = 0
    category: str | None = None
    payee: str | None = None


@dataclass(frozen=True)
class Budget:
    name: str
    planned_minor: int
    used_minor: int


@dataclass(frozen=True)
class Line:
    name: str
    amount_minor: int
    day: date | None = None
    savings: bool = False


@dataclass(frozen=True)
class FactsInput:
    today: date
    entries: list[Entry]
    last_payday: date | None = None
    balances: list[Line] = field(default_factory=list)
    bills: list[Line] = field(default_factory=list)
    budgets: list[Budget] = field(default_factory=list)
    fees_this_month_minor: int | None = None


def money(minor: int) -> str:
    """The app's ``formatMoney`` for KES: ``KES 1,200``, ``KES 1,200.50``, ``-KES 30``."""
    a = abs(minor)
    major, cents = divmod(a, 100)
    s = ("-" if minor < 0 else "") + f"KES {major:,}"
    return s + (f".{cents:02d}" if cents else "")


def fmt_date(d: date) -> str:
    return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def month_name(year: int, month: int) -> str:
    return f"{_MONTHS[month - 1]} {year}"


def add_months(year: int, month: int, n: int) -> tuple[int, int]:
    k = year * 12 + (month - 1) + n
    return k // 12, k % 12 + 1


@dataclass
class Section:
    title: str
    lines: list[str]


def sections_text(sections: list[Section]) -> str:
    out: list[str] = []
    for s in sections:
        out.append(f"{s.title}:")
        out.extend(f"- {line}" for line in s.lines)
    return "\n".join(out)


def _top(m: dict[str, int]) -> list[str]:
    items = sorted(((k, v) for k, v in m.items() if v > 0), key=lambda kv: -kv[1])
    return [f"{k}: {money(v)}" for k, v in items[:TOP_COUNT]]


def facts(inp: FactsInput) -> list[Section]:
    """The fact sheet's sections, as ``SlmFacts.of`` builds them, capped to ``MAX_CHARS``."""
    today = inp.today
    first = today.replace(day=1)
    end = today + timedelta(days=1)

    def within(e: Entry, a: date, b: date) -> bool:
        return a <= e.day < b

    this_month = [e for e in inp.entries if within(e, first, end)]
    sections = [Section("Today", [fmt_date(today)])]

    flows: list[str] = []

    def flow(label: str, lst: list[Entry]) -> None:
        i = sum(e.income_minor for e in lst)
        o = sum(e.spending_minor for e in lst)
        flows.append(f"{label}: money in {money(i)}, spent {money(o)}, left {money(i - o)}")

    flow(f"This month so far ({month_name(today.year, today.month)})", this_month)
    pd = inp.last_payday
    if pd is not None and not pd > today:
        flow(f"Since payday ({fmt_date(pd)})", [e for e in inp.entries if within(e, pd, end)])
    sections.append(Section("Money in and out", flows))

    by_cat: dict[str, int] = {}
    by_payee: dict[str, int] = {}
    fees = 0
    for e in this_month:
        fees += e.fee_minor
        spent = e.spending_minor
        if spent == 0:
            continue
        name = "Not sorted" if e.category is None else e.category
        by_cat[name] = by_cat.get(name, 0) + spent
        payee = e.payee.strip() if e.payee is not None else None
        if payee:
            by_payee[payee] = by_payee.get(payee, 0) + spent
    cats = _top(by_cat)
    if cats:
        sections.append(Section("Top spending this month", cats))
    payees = _top(by_payee)
    if payees:
        sections.append(Section("Top payees this month", payees))
    sections.append(
        Section(
            "Fees this month",
            [money(inp.fees_this_month_minor if inp.fees_this_month_minor is not None else fees)],
        )
    )

    if inp.budgets:
        budgets = sorted(inp.budgets, key=lambda b: -(b.used_minor - b.planned_minor))
        lines = []
        for b in budgets[:TOP_COUNT]:
            tail = (
                f"over by {money(b.used_minor - b.planned_minor)}"
                if b.used_minor > b.planned_minor
                else f"left {money(b.planned_minor - b.used_minor)}"
            )
            lines.append(
                f"{b.name}: planned {money(b.planned_minor)}, spent {money(b.used_minor)}, {tail}"
            )
        sections.append(Section("Budgets this month", lines))

    bills = sorted(
        (b for b in inp.bills if b.day is None or not b.day < today),
        key=lambda b: b.day or today,
    )
    if bills:
        sections.append(
            Section(
                "Upcoming bills",
                [
                    f"{b.name}: {money(b.amount_minor)}"
                    + ("" if b.day is None else f", due {fmt_date(b.day)}")
                    for b in bills[:TOP_COUNT]
                ],
            )
        )

    if inp.balances:
        day_to_day = sum(b.amount_minor for b in inp.balances if not b.savings)
        savings = sum(b.amount_minor for b in inp.balances if b.savings)
        lines = [f"Day-to-day accounts: {money(day_to_day)}"]
        if any(b.savings for b in inp.balances):
            lines.append(f"Savings accounts: {money(savings)}")
        lines += [
            f"{b.name}{' (savings)' if b.savings else ''}: {money(b.amount_minor)}"
            for b in inp.balances[:6]
        ]
        sections.append(Section("Balances now", lines))

    trend = []
    for i in range(3, 0, -1):
        y, m = add_months(today.year, today.month, -i)
        lst = [e for e in inp.entries if e.day.year == y and e.day.month == m]
        trend.append(
            f"{month_name(y, m)}: money in {money(sum(e.income_minor for e in lst))}, "
            f"spent {money(sum(e.spending_minor for e in lst))}"
        )
    sections.append(Section("Last 3 months", trend))

    # Drop lines from the end of the longest sections until the text fits.
    while len(sections_text(sections)) > MAX_CHARS:
        longest = sections[0]
        for s in sections[1:]:
            longest = longest if len(longest.lines) >= len(s.lines) else s
        if len(longest.lines) <= 1:
            break
        longest.lines.pop()
    return sections


def facts_text(inp: FactsInput) -> str:
    return sections_text(facts(inp))


def system_text(facts_str: str) -> str:
    """The app's ``slmSystem``: the rules, then the facts."""
    return f"{SYSTEM_PROMPT}\n\nFacts:\n{facts_str}"


def from_fixture(case: dict) -> FactsInput:  # type: ignore[type-arg]
    """A case of ``slm_facts_parity.json`` as input."""
    d = date.fromisoformat
    return FactsInput(
        today=d(case["today"]),
        last_payday=d(case["last_payday"]) if case["last_payday"] else None,
        entries=[
            Entry(
                d(e["date"]),
                e["income_minor"],
                e["spending_minor"],
                e["fee_minor"],
                e["category"],
                e["payee"],
            )
            for e in case["entries"]
        ],
        balances=[
            Line(b["name"], b["amount_minor"], savings=b["savings"]) for b in case["balances"]
        ],
        bills=[
            Line(b["name"], b["amount_minor"], d(b["date"]) if b["date"] else None)
            for b in case["bills"]
        ],
        budgets=[Budget(b["name"], b["planned_minor"], b["used_minor"]) for b in case["budgets"]],
        fees_this_month_minor=case["fees_this_month_minor"],
    )

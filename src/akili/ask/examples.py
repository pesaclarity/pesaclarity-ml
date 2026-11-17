"""Conformance examples for Ask, for the app's Dart tests.

``akili ask-examples`` writes ``resources/ask_examples.v1.json``; a test fails if the file drifts
from the code. It holds:

* ``questions``: hand-written questions with the intent and slot spans a good reader gives, and
  the exact model input (the message reader's encoding). The input must match exactly; the
  intent and slots are what the shipped model should mostly get right, not a bit-exact test.
* ``fallbacks``: facts, their token ids, the kind's fallback sentence and the sentence filled.
* ``checker``: reply templates with the checker's verdict.
* ``detokenise``: reply tokens and the joined template.

Everything is invented.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path
from typing import Any

from akili.ask.schema import (
    ASK_INTENTS,
    ASK_SCHEMA_VERSION,
    SLOT_LABELS,
    capitalise,
    check_reply,
    detokenise,
    facts,
    fallback,
    fill,
)
from akili.reader.tokens import encode

EXAMPLE_FILE = f"ask_examples.v{ASK_SCHEMA_VERSION}.json"

# (intent, question with [text](slot) marks)
_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("spent", "How much did I spend on [food](category) [last month](period)?"),
    ("spent", "nimetumia pesa ngapi kwa [fare](category) [wiki hii](period)"),
    ("spent", "how much have i spent at [Naivas](payee) [since June](period)"),
    ("spent", "what did [school fees](category) cost me [this year](period)"),
    ("spent", "how much did i spend [last month](period)"),
    ("received", "how much came in [this month](period)?"),
    ("received", "how much was paid into my [Equity](account) [in March](period)"),
    ("received", "my [salary](category) [last month](period)"),
    ("sent_to", "how much have I sent to [John Kamau](payee) [this year](period)?"),
    ("sent_to", "how much did i send [my mum](payee) [last week](period)"),
    ("received_from", "how much did [Mary Wanjiku](payee) send me [in October](period)"),
    ("received_from", "has [Otieno](payee) paid me"),
    ("count_payments", "how many times did i pay [Java House](payee) [last month](period)"),
    ("count_payments", "how often do i spend on [airtime](category)"),
    ("biggest_payment", "what was my biggest payment [in March](period)?"),
    ("biggest_payment", "largest expense on [rent](category) [2025](period)"),
    ("top_payees", "who do i pay the most [this month](period)"),
    ("top_payees", "my top shops"),
    ("top_categories", "where does my money go?"),
    ("top_categories", "pesa yangu inaenda wapi [mwezi uliopita](period)"),
    ("trend", "is my [transport](category) spending going up"),
    ("trend", "how has my [electricity](category) spending changed [this year](period)"),
    ("compare", "did i spend more [this month](period) than [last month](period)"),
    ("compare", "compare [September](period) with [October](period)"),
    ("bills_due", "what bills are due [this week](period)?"),
    ("bills_due", "when is my [water](category) bill due"),
    ("subscriptions", "what subscriptions do i have"),
    ("subscriptions", "am i still paying for [Netflix](payee)"),
    ("debts", "how much do i owe"),
    ("debts", "how much do i owe on [Fuliza](account)"),
    ("lent", "how much does [Brian](payee) owe me"),
    ("lent", "how much have i lent"),
    ("safe_to_spend", "how much can i spend [today](period)?"),
    ("safe_to_spend", "can i afford [2,000](amount) [this week](period)"),
    ("balance", "what is my [M-Pesa](account) balance"),
    ("balance", "nina pesa ngapi"),
    ("goals", "how is my [school fees](category) goal going"),
    ("goals", "how much have i saved for [new phone](category)"),
    ("income_sources", "where does my money come from"),
    ("income_sources", "who pays me the most [this year](period)"),
    ("sales", "how much did i sell [this week](period)"),
    ("sales", "how many [tomatoes](item) did i sell [yesterday](period)"),
    ("top_items", "what sells best [this month](period)"),
    ("top_items", "my best sellers"),
    ("profit", "did i make a profit [last month](period)"),
    ("profit", "faida [wiki hii](period)"),
    ("low_stock", "what am i running out of"),
    ("low_stock", "am i low on [sugar](item)"),
    ("who_owes_me", "which customers owe me"),
    ("who_owes_me", "who took [bread](item) on credit"),
    ("fees", "how much have i paid in fees [this month](period)"),
    ("fees", "[M-Pesa](account) charges [last month](period)"),
    ("refine", "and only [Naivas](payee)?"),
    ("refine", "what about [September](period)"),
    ("refine", "and [last year](period)?"),
    ("refine", "only [cash](account)"),
    ("refine", "just the [over 1,000](amount) ones"),
    ("hi", "Habari"),
    ("hi", "hello there"),
    ("thanks", "asante sana"),
    ("thanks", "thank you"),
    ("bye", "kwaheri"),
    ("bye", "that's all for now"),
    ("help", "what can you do?"),
    ("help", "show me some examples"),
    ("out_of_scope", "should i buy bitcoin"),
    ("out_of_scope", "tell me a joke"),
    ("out_of_scope", "who will win the election"),
)

_MARK = re.compile(r"\[([^\]]+)\]\(([a-z]+)\)")


def parse_marked(marked: str) -> tuple[str, list[dict[str, Any]]]:
    """``"spent on [food](category)"`` -> text and slot spans with character offsets."""
    text = ""
    spans: list[dict[str, Any]] = []
    pos = 0
    for m in _MARK.finditer(marked):
        text += marked[pos : m.start()]
        if m.group(2) not in SLOT_LABELS:
            raise ValueError(f"unknown slot {m.group(2)!r}")
        spans.append(
            {
                "slot": m.group(2),
                "text": m.group(1),
                "start": len(text),
                "end": len(text) + len(m.group(1)),
            }
        )
        text += m.group(1)
        pos = m.end()
    return text + marked[pos:], spans


# (kind, flags, given, values)
_FACTS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], dict[str, str]], ...] = (
    (
        "spent",
        ("cmp=less",),
        ("amount", "category", "period", "prev_period"),
        {
            "amount": "KES 4,250",
            "category": "food",
            "period": "last month",
            "prev_period": "the month before",
        },
    ),
    (
        "received",
        ("cmp=more", "level=peak"),
        ("amount", "period"),
        {"amount": "KES 52,000", "period": "this month"},
    ),
    (
        "sent_to",
        (),
        ("amount", "payee", "period"),
        {"amount": "KES 6,000", "payee": "John Kamau", "period": "this year"},
    ),
    ("received_from", (), ("amount", "payee"), {"amount": "KES 1,500", "payee": "Mary Wanjiku"}),
    ("count_payments", ("count=one",), ("count", "payee"), {"count": "1", "payee": "Java House"}),
    (
        "biggest_payment",
        (),
        ("amount", "payee", "date"),
        {"amount": "KES 18,000", "payee": "the landlord", "date": "1 March"},
    ),
    (
        "top_payees",
        (),
        ("list", "period"),
        {"list": "Naivas, Quickmart and Kenya Power", "period": "this month"},
    ),
    ("top_categories", (), ("list",), {"list": "rent, food and transport"}),
    (
        "trend",
        ("cmp=more",),
        ("category", "period", "percent"),
        {"category": "transport", "period": "the last three months", "percent": "12%"},
    ),
    (
        "compare",
        ("cmp=more",),
        ("amount", "prev_amount", "period", "prev_period"),
        {
            "amount": "KES 21,000",
            "prev_amount": "KES 18,500",
            "period": "this month",
            "prev_period": "last month",
        },
    ),
    ("bills_due", (), ("list", "date"), {"list": "Kenya Power and Zuku", "date": "Friday"}),
    (
        "subscriptions",
        (),
        ("list", "amount"),
        {"list": "Netflix and Spotify", "amount": "KES 1,500"},
    ),
    ("debts", (), ("amount",), {"amount": "KES 3,200"}),
    ("lent", (), ("amount", "payee"), {"amount": "KES 2,000", "payee": "Brian"}),
    ("safe_to_spend", ("state=tight",), ("amount", "days"), {"amount": "KES 800", "days": "4"}),
    ("balance", ("level=low",), ("amount", "account"), {"amount": "KES 340", "account": "M-Pesa"}),
    (
        "goals",
        ("state=ok",),
        ("goal", "amount", "target"),
        {"goal": "school fees", "amount": "KES 12,000", "target": "KES 30,000"},
    ),
    ("income_sources", (), ("list",), {"list": "salary and side hustle"}),
    (
        "sales",
        ("cmp=more",),
        ("amount", "period", "prev_period"),
        {"amount": "KES 9,400", "period": "this week", "prev_period": "last week"},
    ),
    ("top_items", (), ("list",), {"list": "bread, milk and sugar"}),
    (
        "profit",
        ("result=loss",),
        ("amount", "period"),
        {"amount": "KES 1,100", "period": "last month"},
    ),
    ("low_stock", (), ("list",), {"list": "sugar and cooking oil"}),
    ("who_owes_me", (), ("list", "amount"), {"list": "Mama Njeri and Juma", "amount": "KES 2,750"}),
    ("fees", (), ("amount", "period"), {"amount": "KES 410", "period": "this month"}),
    ("nodata", (), ("category", "period"), {"category": "betting", "period": "this year"}),
    ("clarify", (), (), {}),
    ("out_of_scope", (), (), {}),
    ("hi", (), ("name",), {"name": "Wanjiru"}),
    ("thanks", (), (), {}),
    ("bye", (), (), {}),
    ("help", (), (), {}),
)

# (template, kind, given)
_CHECKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("You spent {amount} on {category} {period}.", "spent", ("amount", "category", "period")),
    ("You spent {amount}.", "spent", ("amount", "category")),
    ("You spent {amount} on {category}.", "spent", ("amount",)),
    ("You spent 4,250 on food.", "spent", ("amount",)),
    ("You spent {amount} and {amount}.", "spent", ("amount",)),
    ("You spent a lot.", "spent", ("amount",)),
    ("Hello {name}!", "hi", ("name",)),
    ("Hello!", "hi", ()),
    ("kind=spent {amount}", "spent", ("amount",)),
    ("", "thanks", ()),
    ("{payee} sent you {amount}.", "received_from", ("amount", "payee")),
    ("{goal}: {amount} saved.", "goals", ("goal", "amount")),
)

_DETOKENISE: tuple[tuple[str, ...], ...] = (
    ("okay", ".", "you", "spent", "{amount}", "on", "{category}", "{period}", "."),
    ("{period}", ",", "you", "spent", "{amount}", "!", "nice", "!"),
    ("i", "could", "not", "find", "anything", ",", "i", "am", "sorry", "."),
    ("habari", "{name}", "!", "what", "would", "you", "like", "to", "know", "?"),
    ("bills", "due", ":", "{list}", "."),
)


def build_examples() -> dict[str, Any]:
    from akili.ask.writer import TOKEN_ID, VOCAB

    questions = []
    for intent, marked in _QUESTIONS:
        if intent not in ASK_INTENTS:
            raise ValueError(f"unknown intent {intent!r}")
        text, spans = parse_marked(marked)
        questions.append(
            {"question": text, "intent": intent, "slots": spans, "input": encode(text)}
        )
    fallbacks = []
    for kind, flags, given, values in _FACTS:
        tokens = facts(kind, flags, given)
        template = fallback(kind)
        fallbacks.append(
            {
                "kind": kind,
                "facts": tokens,
                "fact_ids": [TOKEN_ID[t] for t in tokens],
                "fallback": template,
                "values": values,
                "filled": capitalise(fill(template, values)),
            }
        )
    checks = []
    for template, kind, given in _CHECKS:
        c = check_reply(template, kind, given)
        checks.append(
            {
                "template": template,
                "kind": kind,
                "given": list(given),
                "ok": c.ok,
                "reason": c.reason,
            }
        )
    detok = [{"tokens": list(t), "text": detokenise(list(t))} for t in _DETOKENISE]
    return {
        "ask_schema_version": ASK_SCHEMA_VERSION,
        "intents": list(ASK_INTENTS),
        "slots": list(SLOT_LABELS),
        "vocab_size": len(VOCAB),
        "questions": questions,
        "fallbacks": fallbacks,
        "checker": checks,
        "detokenise": detok,
    }


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=True) + "\n"


def default_path() -> Path:
    return Path(str(resources.files("akili.resources").joinpath(EXAMPLE_FILE)))


def write_examples(path: Path | None = None) -> Path:
    target = path or default_path()
    target.write_text(render(build_examples()), encoding="utf-8")
    return target

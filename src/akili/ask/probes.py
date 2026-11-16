"""Two extra held-out sets for the ask reader, on the phrasings the first reader got wrong.

* :func:`hard_questions`: the *hard* set. Counts and sums for one payee, "how much to X",
  money in from someone, money in, "<account> balance", "how often", and look-alikes that must
  stay out of scope or help. Its templates are written here and never used in training, and its
  payees come from :data:`akili.ask.names.HELD_OUT`, whose words training never sees.
* :data:`APP_QUESTIONS`: the questions pesaclarity-app's tests ask (counts, money in, "X sent
  me", "<account> balance", names the app does not know), with the reading the app wants. They
  are scored on their own; their names are kept out of training (``names.EXCLUDED``).

Both score intent accuracy and *name exact-match*: of the questions that name a payee or an
account, the share whose payee and account spans the reader marks exactly, nothing more or
less. Everything is invented.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from akili.ask import names
from akili.ask.examples import parse_marked
from akili.ask.questions import (
    _P,
    ACCOUNT,
    PAYEE,
    Question,
    _Gen,
    _pieces,
    _roughen,
    assemble,
)
from akili.ask.schema import ASK_INTENTS, SLOT_LABELS
from akili.reader.tokens import tokenise

_PAID = "[paid|been paying|bought from|sent money to]"

# (group, intent, template). {name} is a held-out payee, {bank} a bank or Sacco as payee.
HARD_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("count", "count_payments", "how frequently have i " + _PAID + " {name} " + _P),
    ("count", "count_payments", "on how many [occasions|days] did i [pay|buy from] {name} " + _P),
    ("count", "count_payments", "count [the|my] [payments|times i paid] [to|at] {name} " + _P),
    ("count", "count_payments", "how many [times|payments] [went to|were made to] {name} " + _P),
    ("count", "count_payments", "how many times have i sent [money|cash|pesa] to {name}"),
    ("count", "count_payments", "how often [was i paying|have i been at] {name} " + _P),
    ("count", "count_payments", "[the|] number of [purchases|payments] [from|at] {name}"),
    ("count", "count_payments", "mara ngapi nimeenda {name} " + _P),
    ("sum", "spent", "what's the total i [paid|spent] [to|at|with] {name} " + _P),
    ("sum", "spent", "how much money went to {name} " + _P),
    ("sum", "spent", "how much [at|for] {name} " + _P),
    ("sum", "spent", "amount [paid|spent] [to|at] {name} " + _P),
    ("sum", "spent", "what did {name} cost me " + _P),
    ("sum", "spent", "how much [from|on] {paccount} [to|at] {name} " + _P),
    ("sum", "spent", "how much have i paid {name} [so far|altogether|in total]"),
    ("from", "received_from", "how much has {name} [paid|sent] me " + _P),
    ("from", "received_from", "how much money did i get from {name} " + _P),
    ("from", "received_from", "what came in from {name} " + _P),
    ("from", "received_from", "any money from {name} " + _P),
    ("from", "received_from", "did i receive anything from {name} " + _P),
    ("from", "received_from", "has {name} sent me [money|pesa|anything] " + _P),
    ("from", "received_from", "how much has {bank} [sent|paid] me " + _P),
    ("from", "received_from", "money that {name} [sent|paid] me " + _P),
    ("from", "received_from", "{name} sent me [money|cash] " + _P),
    ("in", "received", "how much money came in " + _P),
    ("in", "received", "what did i receive " + _P),
    ("in", "received", "total income " + _P),
    ("in", "received", "how much have i earned " + _P),
    ("in", "received", "what money came in " + _P),
    ("in", "received", "how much did i receive [in total|altogether]"),
    ("in", "received", "money received " + _P),
    ("balance", "balance", "how much is in my {paccount}"),
    ("balance", "balance", "{paccount} balance please"),
    ("balance", "balance", "what's on [my|] {paccount}"),
    ("balance", "balance", "show [me|] [my|] {paccount} balance"),
    ("balance", "balance", "how much money is on {paccount} right now"),
    ("balance", "balance", "tell me my {paccount} balance"),
    ("negative", "out_of_scope", "how often should i [pay rent|save|go to the gym]"),
    ("negative", "out_of_scope", "how many times has [harambee stars|gor mahia] won [the cup|it]"),
    ("negative", "out_of_scope", "who sent me [this|that] [video|message]"),
    ("negative", "out_of_scope", "how much is [rent|a house] in [kilimani|westlands|kisumu]"),
    ("negative", "out_of_scope", "how much does [a laptop|a cow|petrol] cost [these days|now|]"),
    ("negative", "out_of_scope", "send [my sister|my mum] [money|1000] [now|please|]"),
    ("negative", "help", "what can you tell me about [income|payments|balances]"),
    ("negative", "help", "explain how to ask about [payees|money in|my accounts]"),
)  # fmt: skip


def hard_questions(n: int = 2000, seed: int = 7) -> list[tuple[str, Question]]:
    """``n`` (group, question) pairs, groups uniform, templates uniform within a group."""
    g = _Gen(np.random.default_rng(seed), names.HELD_OUT)
    groups: dict[str, list[tuple[int, str, str]]] = {}
    for i, (group, intent, template) in enumerate(HARD_TEMPLATES):
        groups.setdefault(group, []).append((i, intent, template))
    keys = list(groups)
    out = []
    for _ in range(n):
        group = g.pick(keys)
        i, intent, template = groups[group][int(g.rng.integers(len(groups[group])))]
        text, labels = assemble(_roughen(g, _pieces(g, template), intent))
        out.append((group, Question(text, labels, ASK_INTENTS.index(intent), f"hard:{i}")))
    return out


# (intent, question with [text](slot) marks), from pesaclarity-app's tests.
_APP: tuple[tuple[str, str], ...] = (
    ("count_payments", "How many times did I pay [Example Grocers](payee) [this month](period)?"),
    ("count_payments", "how often do I pay [Example Grocers](payee)"),
    ("count_payments", "number of payments to [Example Grocers](payee)"),
    ("count_payments", "times I paid [Example Grocers](payee) [this month](period)"),
    ("count_payments", "How many times did I pay [Zebra Hardware](payee) [this month](period)?"),
    ("count_payments", "How many times did I pay [Example Nowhere](payee) [this month](period)?"),
    ("count_payments", "How many times did I pay [Naivas](payee) [in October](period)?"),
    ("count_payments", "How many payments [over 1000](amount) [this month](period)?"),
    ("spent", "How much did I pay [Example Grocers](payee) [this month](period)?"),
    ("spent", "how much to [Example Grocers](payee)"),
    ("spent", "how much to [john](payee)"),
    ("spent", "How much did I spend at [Example Grocers](payee)?"),
    ("spent", "How much did I spend at [Example Nowhere](payee) [this month](period)?"),
    ("spent", "How much did I spend on [food](category) at [Example Nowhere](payee)?"),
    ("spent", "What did I pay [Kenya Power](payee) [this year](period)?"),
    ("spent", "Spent at [Naivas](payee) [yesterday](period)"),
    ("received", "How much came in [this month](period)?"),
    ("received", "money in [this month](period)"),
    ("received", "how much did I receive"),
    ("received", "income [this month](period)"),
    ("received", "how much did I get"),
    ("received", "How much did I get [this month](period)?"),
    ("received", "How much did I receive [this month](period)?"),
    ("received", "How much came in [this week](period)?"),
    ("received_from", "How much did [KCB](payee) send me?"),
    ("received_from", "[KCB](payee) sent me how much"),
    ("received_from", "How much has [KCB](payee) sent me?"),
    ("received_from", "How much did I receive from [KCB](payee)?"),
    ("received_from", "Money received from [KCB](payee)"),
    ("received_from", "How much has [John Kamau](payee) sent me [this month](period)?"),
    ("received_from", "Did [Acme](payee) pay me [last month](period)?"),
    ("received_from", "What has [Example Sacco](payee) paid me?"),
    ("received_from", "How much did I receive from [Acme](payee)?"),
    ("received_from", "How much I got from [Mama Mboga](payee)"),
    ("received_from", "received from [Acme](payee)"),
    ("received_from", "what I got from [Acme](payee)"),
    ("received_from", "Money I got from [Acme](payee)"),
    ("received_from", "How much did [Acme](payee) send me?"),
    ("received_from", "How much has [Acme](payee) sent me [this month](period)?"),
    ("received_from", "What did [Acme](payee) pay me [this month](period)?"),
    ("received_from", "How much did I receive from [Example Nowhere](payee)?"),
    ("received_from", "Money received from [Acme](payee) [in 2025](period)"),
    ("sent_to", "How much did I send to [Example Nowhere](payee)?"),
    ("sent_to", "How much did I send to [Example Landlord](payee)?"),
    ("sent_to", "How much did I send to [John](payee)?"),
    ("sent_to", "How much did I send to [Mama Mboga](payee) [since June](period)?"),
    ("balance", "What's my [M-Pesa](account) balance?"),
    ("balance", "What's my [Equity](account) balance?"),
    ("balance", "what's my [m-pesa](account) balance"),
    ("balance", "[KCB](account) account balance"),
    ("balance", "What's my balance?"),
    ("balance", "How much money do I have?"),
    ("fees", "How much did I pay in charges on [KCB](account) [this month](period)?"),
    ("biggest_payment", "What was my biggest payment to [Example Nowhere](payee)?"),
    ("debts", "How much do I owe to [Example Lender](payee)?"),
    ("bills_due", "What bills are due for [Example Water](payee)?"),
    ("compare", "Compare my spending at [Example Nowhere](payee) with [last month](period)"),
    (
        "compare",
        "How much did I pay [John Otieno](payee) [this month](period) compared to "
        "[last month](period)?",
    ),
    ("trend", "How has my spending at [Example Nowhere](payee) changed?"),
    ("income_sources", "income sources from [Example Nowhere](payee)"),
    ("hi", "hi"),
    ("bye", "bye"),
    ("help", "What can I ask?"),
    ("out_of_scope", "What's the weather?"),
    ("out_of_scope", "Should I buy a car?"),
)


def _labels(text: str, spans: list[dict[str, Any]]) -> list[int]:
    return [
        next(
            (SLOT_LABELS.index(s["slot"]) for s in spans if s["start"] <= t.start < s["end"]),
            0,
        )
        for t in tokenise(text)
    ]


def app_questions() -> list[Question]:
    out = []
    for i, (intent, marked) in enumerate(_APP):
        text, spans = parse_marked(marked)
        out.append(Question(text, _labels(text, spans), ASK_INTENTS.index(intent), f"app:{i}"))
    return out


def _runs(labels: list[int], keep: set[int]) -> set[tuple[int, int, int]]:
    out, start = set(), None
    for i in range(len(labels) + 1):
        cur = labels[i] if i < len(labels) else -1
        prev = labels[i - 1] if i > 0 else -1
        if start is not None and cur != prev:
            out.add((prev, start, i))
            start = None
        if start is None and cur in keep:
            start = i
    return out


def name_spans(labels: list[int]) -> set[tuple[int, int, int]]:
    """Payee and account spans as (label, first token, end token)."""
    return _runs(labels, {PAYEE, ACCOUNT})


def score_probe(
    questions: list[Question], intents: np.ndarray, slots: np.ndarray, groups: list[str] | None
) -> dict[str, Any]:
    """Intent accuracy and name exact-match, overall and per group."""
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        n = len(q.slots)
        predicted = [int(v) for v in slots[i, :n].argmax(-1)]
        truth = name_spans(q.slots)
        tokens = tokenise(q.text)
        unseen = all(
            all(
                tokens[t].text.lower() in names.HELD_ONLY_WORDS
                for t in range(s, e)
                if tokens[t].text.isalpha()
            )
            for _, s, e in truth
        )
        rows.append(
            {
                "group": groups[i] if groups else ASK_INTENTS[q.intent],
                "unseen_words": bool(truth) and unseen,
                "intent_ok": int(intents[i].argmax()) == q.intent,
                "named": bool(truth),
                "names_ok": name_spans(predicted) == truth,
                "all_slots_ok": predicted == q.slots,
            }
        )

    def summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
        named = [r for r in rs if r["named"]]
        strict = [r for r in named if r["unseen_words"]]
        return {
            "questions": len(rs),
            "intent_accuracy": round(float(np.mean([r["intent_ok"] for r in rs])), 4),
            "named_questions": len(named),
            "name_exact_match": round(float(np.mean([r["names_ok"] for r in named])), 4)
            if named
            else None,
            "all_slots_exact_match": round(float(np.mean([r["all_slots_ok"] for r in rs])), 4),
            "unseen_word_names": len(strict),
            "unseen_word_name_exact_match": round(
                float(np.mean([r["names_ok"] for r in strict])), 4
            )
            if strict
            else None,
        }

    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    return {**summary(rows), "groups": {k: summary(v) for k, v in sorted(by_group.items())}}


def readings(
    questions: list[Question], intents: np.ndarray, slots: np.ndarray
) -> list[dict[str, Any]]:
    """What the reader made of each question: intent, its probability and the slot spans."""
    out = []
    for i, q in enumerate(questions):
        tokens = tokenise(q.text)
        labels = [int(v) for v in slots[i, : len(tokens)].argmax(-1)]
        spans = [
            {"slot": SLOT_LABELS[lab], "text": q.text[tokens[s].start : tokens[e - 1].end]}
            for lab, s, e in sorted(
                _runs(labels, set(range(1, len(SLOT_LABELS)))), key=lambda r: r[1]
            )
        ]
        k = int(intents[i].argmax())
        out.append(
            {
                "question": q.text,
                "expected": ASK_INTENTS[q.intent],
                "intent": ASK_INTENTS[k],
                "p": round(float(intents[i][k]), 3),
                "slots": spans,
            }
        )
    return out

"""The Ask contract shared by training and the app: intents, slots, the writer's fact grammar,
its word tokens, the reply checker and the fallback sentences.

The app reproduces this in Dart; ``docs/ASK.md`` is the language-neutral spec and
``resources/ask_examples.v1.json`` the conformance cases. Any change to the lists below or to
:func:`reply_tokens`, :func:`detokenise` or :func:`check_reply` must bump
:data:`ASK_SCHEMA_VERSION`. The ask reader's *input* is the message reader's encoding
(:func:`akili.reader.tokens.encode`) unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ASK_SCHEMA_VERSION = 1

# The reader's intents, in output order. Never reorder; append and bump the schema version.
ASK_INTENTS: tuple[str, ...] = (
    "spent",  # how much did I spend on food last month
    "received",  # how much came in this month
    "sent_to",  # how much have I sent to John Kamau
    "received_from",  # how much did Mary send me
    "count_payments",  # how many times did I pay Naivas
    "biggest_payment",  # what was my biggest payment in March
    "top_payees",  # who do I pay the most
    "top_categories",  # where does my money go
    "trend",  # is my food spending going up
    "compare",  # did I spend more this month than last month
    "bills_due",  # what bills are due this week
    "subscriptions",  # what subscriptions do I have
    "debts",  # how much do I owe
    "lent",  # how much do people owe me (personal)
    "safe_to_spend",  # how much can I spend today
    "balance",  # what is my M-Pesa balance
    "goals",  # how is my school fees goal going
    "income_sources",  # where does my money come from
    "sales",  # how much did I sell this week (business)
    "top_items",  # what sells best
    "profit",  # did I make a profit this month
    "low_stock",  # what is running out
    "who_owes_me",  # which customers owe me
    "fees",  # how much have I paid in transaction charges
    "refine",  # and only Naivas? / what about September
    "hi",
    "thanks",
    "bye",
    "help",  # what can you do
    "out_of_scope",  # advice, general knowledge, chit-chat
)

# Per-token slot labels, in output order. "other" is O.
SLOT_LABELS: tuple[str, ...] = (
    "other",
    "category",
    "payee",
    "period",
    "item",
    "account",
    "amount",
)

# The writer's reply kinds: every data intent (refine resolves to one of them in the app), then
# the conversational ones.
REPLY_KINDS: tuple[str, ...] = (
    *(i for i in ASK_INTENTS if i not in {"refine", "hi", "thanks", "bye", "help", "out_of_scope"}),
    "nodata",
    "clarify",
    "out_of_scope",
    "hi",
    "thanks",
    "bye",
    "help",
)

# Every value in a reply is one of these; the app fills them. Order is the canonical fact order.
PLACEHOLDERS: tuple[str, ...] = (
    "amount",  # money, formatted by the app: "KES 4,250"
    "prev_amount",  # the same measure in the earlier period
    "target",  # a goal's target amount
    "change",  # money difference: "KES 600"
    "percent",  # "12%"
    "count",  # a number of payments, bills, items
    "days",  # a number of days
    "category",
    "payee",  # a person, business or merchant
    "item",  # a business stock item
    "account",  # M-Pesa, Equity, cash
    "goal",  # a goal's name
    "period",  # "last month", "in March"
    "prev_period",
    "date",  # "Friday 14 November"
    "name",  # the person's first name
    "list",  # names joined by the app: "Naivas, Quickmart and Java House"
)

# Flags, in canonical fact order. At most one of each group.
FLAGS: tuple[str, ...] = (
    "cmp=more",
    "cmp=less",
    "cmp=same",
    "level=peak",  # the highest in a while
    "level=low",  # the lowest in a while
    "count=one",  # count is exactly one (or the list has one name)
    "state=ok",  # on track / plenty left
    "state=tight",  # little left / behind
    "state=over",  # over budget / spent past the safe amount
    "state=done",  # goal reached
    "result=loss",  # profit is negative; amount is the loss
)


@dataclass(frozen=True, slots=True)
class KindSpec:
    """What the facts of one reply kind may carry."""

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    fallback: str = ""


_CMP = ("cmp=more", "cmp=less", "cmp=same")
_STATE = ("state=ok", "state=tight", "state=over")

KINDS: dict[str, KindSpec] = {
    "spent": KindSpec(
        ("amount",),
        (
            "category",
            "payee",
            "period",
            "prev_period",
            "prev_amount",
            "change",
            "percent",
            "count",
            "list",
        ),
        (*_CMP, "level=peak", "level=low"),
        "You spent {amount}.",
    ),
    "received": KindSpec(
        ("amount",),
        ("period", "payee", "prev_period", "prev_amount", "change", "percent", "count", "account"),
        (*_CMP, "level=peak", "level=low"),
        "You received {amount}.",
    ),
    "sent_to": KindSpec(
        ("amount", "payee"),
        ("period", "count", "date", "prev_period", "prev_amount"),
        (*_CMP, "count=one"),
        "You sent {payee} {amount}.",
    ),
    "received_from": KindSpec(
        ("amount", "payee"),
        ("period", "count", "date", "prev_period", "prev_amount"),
        (*_CMP, "count=one"),
        "{payee} sent you {amount}.",
    ),
    "count_payments": KindSpec(
        ("count",),
        ("category", "payee", "period", "amount", "account"),
        ("count=one",),
        "Number of payments: {count}.",
    ),
    "biggest_payment": KindSpec(
        ("amount", "payee"),
        ("date", "category", "period"),
        (),
        "Your biggest payment was {amount} to {payee}.",
    ),
    "top_payees": KindSpec(
        ("list",),
        ("period", "payee", "amount", "count", "category"),
        ("count=one",),
        "You paid these the most: {list}.",
    ),
    "top_categories": KindSpec(
        ("list",),
        ("period", "category", "amount", "percent"),
        ("count=one",),
        "Most of your money went to: {list}.",
    ),
    "trend": KindSpec(
        ("period",),
        ("category", "payee", "amount", "prev_amount", "change", "percent"),
        (*_CMP, "level=peak", "level=low"),
        "Here is how it changed over {period}.",
    ),
    "compare": KindSpec(
        ("amount", "prev_amount", "period", "prev_period"),
        ("category", "payee", "change", "percent"),
        _CMP,
        "{amount} {period}, against {prev_amount} {prev_period}.",
    ),
    "bills_due": KindSpec(
        ("list",),
        ("count", "amount", "date", "period", "days"),
        ("count=one",),
        "Bills due: {list}.",
    ),
    "subscriptions": KindSpec(
        ("list",),
        ("count", "amount", "period"),
        ("count=one",),
        "Your subscriptions: {list}.",
    ),
    "debts": KindSpec(
        ("amount",),
        ("list", "count", "payee", "date", "days"),
        ("count=one",),
        "You owe {amount} in total.",
    ),
    "lent": KindSpec(
        ("amount",),
        ("list", "count", "payee", "date"),
        ("count=one",),
        "People owe you {amount} in total.",
    ),
    "safe_to_spend": KindSpec(
        ("amount",),
        ("days", "date", "period"),
        _STATE,
        "Safe to spend: {amount}.",
    ),
    "balance": KindSpec(
        ("amount",),
        ("account", "date"),
        ("level=low",),
        "Your balance is {amount}.",
    ),
    "goals": KindSpec(
        ("goal", "amount"),
        ("target", "percent", "days", "date", "change"),
        ("state=ok", "state=tight", "state=done"),
        "{goal}: {amount} saved.",
    ),
    "income_sources": KindSpec(
        ("list",),
        ("period", "payee", "amount", "percent"),
        ("count=one",),
        "Your money came from: {list}.",
    ),
    "sales": KindSpec(
        ("amount",),
        ("period", "count", "prev_period", "prev_amount", "change", "percent", "item"),
        (*_CMP, "level=peak", "level=low"),
        "Sales: {amount}.",
    ),
    "top_items": KindSpec(
        ("list",),
        ("item", "count", "period", "amount"),
        ("count=one",),
        "Your best sellers: {list}.",
    ),
    "profit": KindSpec(
        ("amount",),
        ("period", "prev_period", "prev_amount", "percent"),
        (*_CMP, "result=loss"),
        "Profit: {amount}.",
    ),
    "low_stock": KindSpec(
        ("list",),
        ("count", "item", "days"),
        ("count=one",),
        "Running low: {list}.",
    ),
    "who_owes_me": KindSpec(
        ("list", "amount"),
        ("count", "name", "payee", "date"),
        ("count=one",),
        "Customers owe you {amount}: {list}.",
    ),
    "fees": KindSpec(
        ("amount",),
        ("period", "count", "account", "prev_period", "prev_amount"),
        (*_CMP, "level=peak"),
        "You paid {amount} in charges.",
    ),
    "nodata": KindSpec(
        (),
        ("category", "payee", "period", "item", "account"),
        (),
        "I could not find anything for that.",
    ),
    "clarify": KindSpec(
        (),
        (),
        (),
        "Sorry, I did not get that. Try: how much did I spend on food last month?",
    ),
    "out_of_scope": KindSpec(
        (),
        (),
        (),
        "I can only answer questions about your own money in this app.",
    ),
    "hi": KindSpec((), ("name",), (), "Hello! Ask me about your money."),
    "thanks": KindSpec((), ("name",), (), "You are welcome."),
    "bye": KindSpec((), ("name",), (), "Goodbye."),
    "help": KindSpec(
        (),
        (),
        (),
        "Ask me things like: how much did I spend on food last month?",
    ),
}

# Flags that cannot appear together: pick at most one of each group.
FLAG_GROUPS: tuple[tuple[str, ...], ...] = (
    _CMP,
    ("level=peak", "level=low"),
    ("state=ok", "state=tight", "state=over", "state=done"),
)

# ---------------------------------------------------------------------------------------------
# Writer vocabulary

PAD, UNK, SEP, EOS = "<pad>", "<unk>", "<sep>", "<eos>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD, UNK, SEP, EOS)
PUNCTUATION: tuple[str, ...] = (".", ",", "!", "?", ":")
WRITER_LENGTH = 64

_REPLY_TOKEN = re.compile(r"\{[a-z_]+\}|[a-z']+|[.,!?:]")


def kind_token(kind: str) -> str:
    return f"kind={kind}"


def placeholder_token(name: str) -> str:
    return "{" + name + "}"


FACT_TOKENS: tuple[str, ...] = (*(kind_token(k) for k in REPLY_KINDS), *FLAGS)
PLACEHOLDER_TOKENS: tuple[str, ...] = tuple(placeholder_token(p) for p in PLACEHOLDERS)


def reply_tokens(text: str) -> list[str]:
    """Tokens of a reply template: placeholders, lower-case words (letters and apostrophes) and
    the single punctuation marks ``. , ! ? :``. Raises on anything else, so training text can
    never carry a digit or a stray character."""
    lowered = text.lower()
    tokens = _REPLY_TOKEN.findall(lowered)
    if "".join(tokens) != "".join(lowered.split()):
        raise ValueError(f"reply has characters outside the grammar: {text!r}")
    return tokens


def detokenise(tokens: list[str]) -> str:
    """Join reply tokens into a template: a space between tokens except before punctuation; the
    first letter of the reply and of the first word after ``. ! ?`` upper case; the word ``i``
    (and ``i'm``, ``i'll`` ...) upper case. Placeholders are left as they are."""
    out = ""
    capital = True
    for tok in tokens:
        word = tok
        if tok in PUNCTUATION:
            out += tok
            if tok in ".!?":
                capital = True
            continue
        if not tok.startswith("{"):
            if capital or tok == "i" or tok.startswith("i'"):
                word = tok[0].upper() + tok[1:]
            capital = False
        else:
            capital = False
        out += (" " if out else "") + word
    return out


def facts(
    kind: str, flags: list[str] | tuple[str, ...], given: list[str] | tuple[str, ...]
) -> list[str]:
    """The canonical fact tokens: the kind, then flags in :data:`FLAGS` order, then the given
    placeholders in :data:`PLACEHOLDERS` order."""
    if kind not in KINDS:
        raise ValueError(f"unknown reply kind {kind!r}")
    unknown = [f for f in flags if f not in FLAGS] + [p for p in given if p not in PLACEHOLDERS]
    if unknown:
        raise ValueError(f"unknown fact tokens {unknown}")
    return (
        [kind_token(kind)]
        + [f for f in FLAGS if f in flags]
        + [placeholder_token(p) for p in PLACEHOLDERS if p in given]
    )


@dataclass(frozen=True, slots=True)
class Check:
    ok: bool
    reason: str = ""


_PLACEHOLDER_IN_TEXT = re.compile(r"\{([a-z_]+)\}")


def check_reply(template: str, kind: str, given: list[str] | tuple[str, ...]) -> Check:  # noqa: PLR0911
    """The app's checker, on the detokenised template before filling. A reply passes when it is
    non-empty and at most 300 characters, has no digit, no ``<``, ``>``, ``=``, ``{`` or ``}``
    outside a placeholder, uses only placeholders it was given, each at most once, and uses
    every placeholder the kind requires. ``reason`` is the first failed rule: ``empty``,
    ``digit``, ``stray``, ``repeated``, ``not_given`` or ``missing``."""
    if not template.strip() or len(template) > 300:
        return Check(False, "empty")
    if any("0" <= c <= "9" for c in template):
        return Check(False, "digit")
    names = _PLACEHOLDER_IN_TEXT.findall(template)
    rest = _PLACEHOLDER_IN_TEXT.sub("", template)
    if any(c in rest for c in "<>={}"):
        return Check(False, "stray")
    if len(names) != len(set(names)):
        return Check(False, "repeated")
    extra = [n for n in names if n not in given]
    if extra:
        return Check(False, "not_given")
    missing = [r for r in KINDS[kind].required if r not in names]
    if missing:
        return Check(False, "missing")
    return Check(True)


def fill(template: str, values: dict[str, str]) -> str:
    """Put values into a checked template."""
    return _PLACEHOLDER_IN_TEXT.sub(lambda m: values[m.group(1)], template)


def capitalise(text: str) -> str:
    """After filling: upper-case the first character of the reply and the first character after
    each ``.``, ``!`` or ``?`` (and the spaces after it) when it is a letter, since a filled value
    can start a sentence ("last month, ..."). A sentence that starts with a digit stays as is."""
    out, capital = [], True
    for c in text:
        if capital and c != " ":
            out.append(c.upper() if c.isalpha() else c)
            capital = False
        else:
            out.append(c)
            if c in ".!?":
                capital = True
    return "".join(out)


def fallback(kind: str) -> str:
    """The fixed sentence for ``kind``; it uses only the kind's required placeholders."""
    return KINDS[kind].fallback


def ask_manifest() -> dict[str, object]:
    return {
        "ask_schema_version": ASK_SCHEMA_VERSION,
        "intents": list(ASK_INTENTS),
        "slots": list(SLOT_LABELS),
    }

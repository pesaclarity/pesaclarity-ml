"""Synthetic bank and wallet messages with every field labelled.

Messages are built from labelled pieces, so each token's label is known exactly. Institutions,
people, merchants and numbers are invented; nothing comes from real messages. Each message also
records its *style*: the wording and layout choices that made it. Evaluation holds whole styles
out, so the reported score is for wording the model never saw, which is what the review inbox
meets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from akili.reader.tokens import MAX_TOKENS, encode, tokenise


class Field(IntEnum):
    """Per-token labels."""

    OTHER = 0
    AMOUNT = 1
    COUNTERPARTY = 2
    DATE = 3
    TIME = 4
    BALANCE = 5
    REFERENCE = 6
    FEE = 7


class Kind(IntEnum):
    """What the whole message is."""

    MONEY_OUT = 0
    MONEY_IN = 1
    NOT_TRANSACTION = 2


FIELD_LABELS = [f.name.lower() for f in Field]
KIND_LABELS = [k.name.lower() for k in Kind]

Piece = tuple[str, Field]


@dataclass(frozen=True, slots=True)
class Message:
    text: str
    fields: list[int]
    """One label per token of :func:`tokenise`."""
    kind: Kind
    style: str


_INSTITUTIONS = [
    "Sample Sacco",
    "Demo Bank",
    "Example Microfinance",
    "Test Credit Union",
    "Mfano Bank",
    "Mock Wallet",
    "Placeholder Pay",
    "Kielelezo Sacco",
    "Specimen Bank",
    "Trial Money",
    "Dummy Savings",
    "Pretend Bank",
    "Mfano Pesa",
    "Sampuli Sacco",
    "Prototype Pay",
]
_FIRST = [
    "JOHN",
    "MARY",
    "AMINA",
    "PETER",
    "WANJIRU",
    "OTIENO",
    "FAITH",
    "BRIAN",
    "AISHA",
    "KEVIN",
    "GRACE",
    "DAVID",
    "ACHIENG",
    "MOSES",
    "NJERI",
    "HASSAN",
    "LYDIA",
    "SAMUEL",
]
_LAST = [
    "DOE",
    "MWANGI",
    "OCHIENG",
    "KAMAU",
    "WAFULA",
    "MUTUA",
    "ODHIAMBO",
    "KIPTOO",
    "NJOROGE",
    "ATIENO",
    "CHEBET",
    "OMONDI",
    "WANJALA",
    "MUSYOKA",
    "KARANJA",
    "NDUTA",
]
_MERCHANT_A = [
    "EXAMPLE",
    "SAMPLE",
    "DEMO",
    "TEST",
    "MFANO",
    "SUNRISE",
    "GREENLEAF",
    "BLUEWAVE",
    "CORNER",
    "CITY",
    "HILLTOP",
    "RIVERSIDE",
    "UMOJA",
    "BAHARI",
    "JAMII",
    "SOKO",
]
_MERCHANT_B = [
    "GROCERS",
    "SUPERMARKET",
    "PHARMACY",
    "HARDWARE",
    "BUTCHERY",
    "CAFE",
    "STORES",
    "FUEL STATION",
    "BOOKSHOP",
    "SALON",
    "ELECTRONICS",
    "WATER CO",
    "POWER",
    "SCHOOL",
    "HOSPITAL",
    "MOTORS",
    "MINIMART",
    "BAKERY",
]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class _Gen:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def pick(self, options: list[str]) -> str:
        return options[int(self.rng.integers(len(options)))]

    def chance(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def money(self, style: str, low: int = 10, high: int = 150_000) -> str:
        whole = int(np.exp(self.rng.uniform(np.log(max(low, 1)), np.log(high))))
        cents = int(self.rng.integers(100)) if style.endswith("c") else 0
        grouped = style.startswith("g")
        text = f"{whole:,}" if grouped else str(whole)
        if style.endswith("c") or style.endswith("z"):
            text += f".{cents:02d}"
        return text

    def person(self) -> str:
        name = f"{self.pick(_FIRST)} {self.pick(_LAST)}"
        return name if self.chance(0.6) else name.title()

    def merchant(self) -> str:
        name = f"{self.pick(_MERCHANT_A)} {self.pick(_MERCHANT_B)}"
        return name if self.chance(0.7) else name.title()

    def phone(self) -> str:
        return f"07{self.rng.integers(10, 99)}***{self.rng.integers(100, 999)}"

    def digits(self, n: int) -> str:
        return "".join(str(int(d)) for d in self.rng.integers(0, 10, n))

    def reference(self, style: str) -> str:
        if style == "alnum":
            letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
            chars = [letters[int(i)] for i in self.rng.integers(0, len(letters), 10)]
            for i in self.rng.choice(range(2, 10), 4, replace=False):
                chars[int(i)] = str(int(self.rng.integers(10)))
            return "".join(chars)
        return self.digits(int(self.rng.integers(8, 13)))

    def date(self, style: str) -> str:
        d, m, y = int(self.rng.integers(1, 29)), int(self.rng.integers(1, 13)), 2026
        return {
            "dmy2": f"{d}/{m}/{y % 100}",
            "dmy2z": f"{d:02d}/{m:02d}/{y % 100}",
            "dmy4": f"{d:02d}/{m:02d}/{y}",
            "dash": f"{d:02d}-{m:02d}-{y}",
            "iso": f"{y}-{m:02d}-{d:02d}",
            "words": f"{d} {_MONTHS[m - 1]} {y}",
            "dashwords": f"{d:02d}-{_MONTHS[m - 1].upper()}-{y % 100}",
        }[style]

    def time(self, style: str) -> str:
        h, mi = int(self.rng.integers(0, 24)), int(self.rng.integers(0, 60))
        if style == "12h":
            return f"{(h % 12) or 12}:{mi:02d} {'AM' if h < 12 else 'PM'}"
        if style == "secs":
            return f"{h:02d}:{mi:02d}:{int(self.rng.integers(60)):02d}"
        return f"{h:02d}:{mi:02d}"


_CURRENCIES = ["KES ", "Ksh ", "KSh ", "Kshs. ", "KES", "Ksh."]

LAYOUTS = [
    "account",  # Your a/c has been debited KES 500 to JOHN DOE on ...
    "amount-first",  # KES 500 sent to JOHN DOE on ...
    "party-first",  # JOHN DOE has sent you KES 500 on ...
    "you-have",  # You have received KES 500 from JOHN DOE on ...
    "noun",  # Payment of KES 500 to EXAMPLE STORES on ...
    "verb-first",  # Paid KES 500 to EXAMPLE STORES on ...
    "key-value",  # Amount: KES 500 Merchant: EXAMPLE STORES Date: ...
]
"""Sentence layouts. Evaluation holds each one out in turn."""
# (words before the amount, words between amount and party, kind of party)
_OUT_VERBS = [
    ("debited", "to", "person"),
    ("debited with", "for payment to", "merchant"),
    ("charged", "at", "merchant"),
    ("debited", "at", "merchant"),
    ("used to send", "to", "person"),
    ("debited", "for withdrawal at agent", "merchant"),
    ("debited", "for a purchase at", "merchant"),
    ("debited", "towards", "merchant"),
]
_IN_VERBS = [
    ("credited with", "from", "person"),
    ("credited", "by", "person"),
    ("credited with", "being a refund from", "merchant"),
    ("credited with", "salary from", "merchant"),
    ("credited", "as a reversal from", "merchant"),
    ("credited with", "deposited by", "person"),
]
_BALANCES = [
    "Avl bal",
    "Available balance",
    "New balance is",
    "Bal:",
    "Your balance is",
    "A/C bal",
    "Account balance",
]
_REFS = ["Ref", "Ref:", "Ref.", "Trx ID", "Transaction ID:", "Txn", "Reference"]
_FEES = ["Transaction cost", "Charges:", "Fee", "Charge", "Cost"]


@dataclass(slots=True)
class _Draft:
    """The choices behind one transaction message."""

    out: bool
    before: str
    between: str
    money: str
    prefix: str
    layout: str
    amount: str
    party: str
    date: str
    time: str
    ref: str
    account: str
    institution: str
    style: str


def _draft(g: _Gen) -> _Draft:
    out = g.chance(0.6)
    verbs = _OUT_VERBS if out else _IN_VERBS
    before, between, party_kind = verbs[int(g.rng.integers(len(verbs)))]
    cur = g.pick(_CURRENCIES)
    money = g.pick(["gc", "c", "g", "gz", "plain"])
    date_style = g.pick(["dmy2", "dmy2z", "dmy4", "dash", "iso", "words", "dashwords"])
    time_style = g.pick(["24h", "12h", "secs"])
    layout = g.pick(LAYOUTS)
    prefix = g.pick(["bank:", "dear", "none", "ref-confirmed"])
    if layout == "key-value" and prefix == "ref-confirmed":
        prefix = "bank:"
    ref_style = g.pick(["alnum", "digits"])
    amount = cur + g.money(money)
    if g.chance(0.15):
        amount = g.money(money) + " " + cur.strip().rstrip(".")
    if party_kind == "person":
        party = g.person() + (f" {g.phone()}" if g.chance(0.4) else "")
    else:
        party = g.merchant()
        if g.chance(0.25):
            party = f"{g.pick(['Paybill', 'Till', 'Acc'])} {g.digits(6)} {party}"
    style = "|".join(
        [
            f"{before}/{between}",
            cur.strip(),
            money,
            date_style,
            time_style,
            layout,
            prefix,
            ref_style,
        ]
    )
    return _Draft(
        out=out,
        before=before,
        between=between,
        money=money,
        prefix=prefix,
        layout=layout,
        amount=amount,
        party=party,
        date=g.date(date_style),
        time=g.time(time_style),
        ref=g.reference(ref_style),
        account=g.pick(
            [
                f"a/c ***{g.digits(4)}",
                f"account ending {g.digits(4)}",
                f"A/C XXXX{g.digits(4)}",
                "account",
                "wallet",
            ]
        ),
        institution=g.pick(_INSTITUTIONS),
        style=style,
    )


def _opening(g: _Gen, d: _Draft) -> list[Piece]:
    if d.prefix == "bank:":
        return [(f"{d.institution}:", Field.OTHER)]
    if d.prefix == "dear":
        return [(g.pick(["Dear Customer,", "Dear Member,", "Hello,"]), Field.OTHER)]
    if d.prefix == "ref-confirmed":
        return [(d.ref, Field.REFERENCE), ("Confirmed.", Field.OTHER)]
    return []


def _party_first(g: _Gen, d: _Draft) -> list[Piece]:
    verb = (
        g.pick(["was paid", "has received", "received"])
        if d.out
        else g.pick(["has sent you", "sent you", "has paid you", "paid you"])
    )
    pieces: list[Piece] = [
        (d.party, Field.COUNTERPARTY),
        (verb, Field.OTHER),
        (d.amount, Field.AMOUNT),
    ]
    if d.out and g.chance(0.5):
        pieces.append(("from your account", Field.OTHER))
    return pieces


def _you_have(g: _Gen, d: _Draft) -> list[Piece]:
    word = g.pick(["sent", "paid", "transferred"]) if d.out else "received"
    return [
        ("You have", Field.OTHER),
        (word, Field.OTHER),
        (d.amount, Field.AMOUNT),
        ("to" if d.out else "from", Field.OTHER),
        (d.party, Field.COUNTERPARTY),
    ]


def _amount_first(g: _Gen, d: _Draft) -> list[Piece]:
    link = (
        g.pick(["sent to", "paid to", "spent at", "withdrawn at"])
        if d.out
        else g.pick(["received from", "credited from", "deposited by"])
    )
    return [(d.amount, Field.AMOUNT), (link, Field.OTHER), (d.party, Field.COUNTERPARTY)]


def _noun(g: _Gen, d: _Draft) -> list[Piece]:
    noun = (
        g.pick(["Payment of", "Transfer of", "Withdrawal of", "Purchase of"])
        if d.out
        else g.pick(["Deposit of", "Transfer of", "Receipt of", "Credit of"])
    )
    pieces: list[Piece] = [
        (noun, Field.OTHER),
        (d.amount, Field.AMOUNT),
        (g.pick(["to", "at"]) if d.out else g.pick(["from", "by"]), Field.OTHER),
        (d.party, Field.COUNTERPARTY),
    ]
    if g.chance(0.5):
        pieces.append((g.pick(["successful", "completed", "confirmed"]), Field.OTHER))
    return pieces


def _verb_first(g: _Gen, d: _Draft) -> list[Piece]:
    verb = (
        g.pick(["Paid", "Sent", "Spent", "Transferred"])
        if d.out
        else g.pick(["Received", "Got", "Credited"])
    )
    return [
        (verb, Field.OTHER),
        (d.amount, Field.AMOUNT),
        (g.pick(["to", "at"]) if d.out else "from", Field.OTHER),
        (d.party, Field.COUNTERPARTY),
    ]


def _account(g: _Gen, d: _Draft) -> list[Piece]:
    return [
        (f"Your {d.account}", Field.OTHER),
        ("has been" if g.chance(0.5) else "was", Field.OTHER),
        (d.before, Field.OTHER),
        (d.amount, Field.AMOUNT),
        (d.between, Field.OTHER),
        (d.party, Field.COUNTERPARTY),
    ]


def _key_value(g: _Gen, d: _Draft) -> list[Piece]:
    party_key = g.pick(["To:", "Merchant:", "Payee:", "Beneficiary:"]) if d.out else "From:"
    pairs: list[list[Piece]] = [
        [(g.pick(["Amount:", "Amt:", "Value:"]), Field.OTHER), (d.amount, Field.AMOUNT)],
        [(party_key, Field.OTHER), (d.party, Field.COUNTERPARTY)],
    ]
    g.rng.shuffle(pairs)
    alert = g.pick(["Txn alert:", "Transaction:", "Debit alert." if d.out else "Credit alert."])
    pieces: list[Piece] = [(alert, Field.OTHER)]
    pieces += [p for pair in pairs for p in pair]
    pieces += [("Date:", Field.OTHER), (d.date, Field.DATE)]
    if g.chance(0.7):
        pieces.append((d.time, Field.TIME))
    return pieces


_BODIES: dict[str, Callable[[_Gen, _Draft], list[Piece]]] = {
    "account": _account,
    "amount-first": _amount_first,
    "party-first": _party_first,
    "you-have": _you_have,
    "noun": _noun,
    "verb-first": _verb_first,
    "key-value": _key_value,
}


def _when(g: _Gen, d: _Draft) -> list[Piece]:
    when: list[Piece] = [("on", Field.OTHER), (d.date, Field.DATE)]
    if g.chance(0.8):
        when += [("at" if g.chance(0.5) else "", Field.OTHER), (d.time, Field.TIME)]
    return when


def _transaction(g: _Gen) -> tuple[list[Piece], Kind, str]:
    d = _draft(g)
    opening = _opening(g, d)
    body = _BODIES[d.layout](g, d)
    if d.layout != "key-value":
        when = _when(g, d)
        if g.chance(0.25):
            # "On 29/10/26 at 14:05, ..." in place of a trailing date.
            body = [(when[0][0].capitalize(), Field.OTHER), *when[1:], (",", Field.OTHER), *body]
        else:
            body += when
    pieces = [*opening, *body, (".", Field.OTHER), *_closing(g, d)]
    return pieces, Kind.MONEY_OUT if d.out else Kind.MONEY_IN, d.style


def _closing(g: _Gen, d: _Draft) -> list[Piece]:
    pieces: list[Piece] = []
    if g.chance(0.5) and d.prefix != "ref-confirmed":
        pieces += [(g.pick(_REFS), Field.OTHER), (d.ref, Field.REFERENCE), (".", Field.OTHER)]
    if d.out and g.chance(0.4):
        pieces += [
            (g.pick(_FEES), Field.OTHER),
            (g.pick(_CURRENCIES) + g.money(d.money, 1, 300), Field.FEE),
            (".", Field.OTHER),
        ]
    if g.chance(0.75):
        pieces += [
            (g.pick(_BALANCES), Field.OTHER),
            (g.pick(_CURRENCIES) + g.money(d.money, 10, 400_000), Field.BALANCE),
            (".", Field.OTHER),
        ]
    if g.chance(0.2):
        pieces.append(
            (
                g.pick(["Thank you.", "Call 0800 000 000 for help.", "Never share your PIN."]),
                Field.OTHER,
            )
        )
    return pieces


def _not_transaction(g: _Gen) -> tuple[list[Piece], Kind, str]:
    kind = g.pick(["otp", "promo", "balance", "failed", "limit", "statement"])
    institution = g.pick(_INSTITUTIONS)
    cur = g.pick(_CURRENCIES)
    if kind == "otp":
        pieces: list[Piece] = [
            (g.pick(["Your verification code is", "Use", "OTP:"]), Field.OTHER),
            (g.digits(6), Field.OTHER),
            (
                g.pick(["to log in. Do not share it.", ". It expires in 5 minutes.", ""]),
                Field.OTHER,
            ),
        ]
    elif kind == "promo":
        pieces = [
            (f"{institution}:", Field.OTHER),
            (
                g.pick(["Get a loan of up to", "Save now and earn on balances above", "Win up to"]),
                Field.OTHER,
            ),
            (cur + g.money("g", 1000, 500_000), Field.OTHER),
            (g.pick(["today. Dial *123#.", ". Terms apply.", "this month!"]), Field.OTHER),
        ]
    elif kind == "balance":
        pieces = [
            (f"{institution}:", Field.OTHER),
            (g.pick(_BALANCES), Field.OTHER),
            (cur + g.money("gc", 10, 400_000), Field.BALANCE),
            ("as at", Field.OTHER),
            (g.date("dmy4"), Field.DATE),
            (".", Field.OTHER),
        ]
    elif kind == "failed":
        pieces = [
            (
                g.pick(
                    [
                        "Transaction failed.",
                        "Your request could not be completed.",
                        "Payment declined.",
                    ]
                ),
                Field.OTHER,
            ),
            (g.pick(["Insufficient funds.", "Try again later.", "Wrong PIN."]), Field.OTHER),
        ]
    elif kind == "limit":
        pieces = [
            (f"{institution}:", Field.OTHER),
            ("Your loan limit is now", Field.OTHER),
            (cur + g.money("g", 500, 100_000), Field.OTHER),
            (".", Field.OTHER),
        ]
    else:
        pieces = [
            (f"{institution}:", Field.OTHER),
            ("Your e-statement for", Field.OTHER),
            (g.pick(_MONTHS) + " 2026", Field.OTHER),
            ("has been sent to your email.", Field.OTHER),
        ]
    return pieces, Kind.NOT_TRANSACTION, f"none|{kind}"


_FILLER = [
    "successfully",
    "via app",
    "kindly note",
    "now",
    "today",
    "through mobile banking",
    "please",
    "-",
    "|",
]


def _with_filler(g: _Gen, pieces: list[Piece]) -> list[Piece]:
    """Sprinkle unlabelled words between pieces, so position alone never decides a label."""
    out: list[Piece] = []
    for piece in pieces:
        out.append(piece)
        if piece[1] is Field.OTHER and g.chance(0.06):
            out.append((g.pick(_FILLER), Field.OTHER))
    return out


def _assemble(pieces: list[Piece]) -> tuple[str, list[tuple[int, int, Field]]]:
    text = ""
    spans: list[tuple[int, int, Field]] = []
    for piece, label in pieces:
        if not piece:
            continue
        glue = "" if piece in (".", ",") or piece.startswith(".") or not text else " "
        text += glue
        spans.append((len(text), len(text) + len(piece), label))
        text += piece
    return text, spans


def label_tokens(text: str, spans: list[tuple[int, int, Field]]) -> list[int]:
    """The label of the piece each token starts in."""
    labels = []
    for token in tokenise(text):
        label = next((lab for s, e, lab in spans if s <= token.start < e), Field.OTHER)
        labels.append(int(label))
    return labels


def generate(n: int, seed: int = 0, not_transaction_rate: float = 0.2) -> list[Message]:
    g = _Gen(np.random.default_rng(seed))
    build: list[Callable[[_Gen], tuple[list[Piece], Kind, str]]] = [_transaction, _not_transaction]
    out = []
    for _ in range(n):
        maker = build[1] if g.chance(not_transaction_rate) else build[0]
        pieces, kind, style = maker(g)
        text, spans = _assemble(_with_filler(g, pieces))
        out.append(Message(text, label_tokens(text, spans), kind, style))
    return out


def split_by_style(
    messages: list[Message], holdout_fraction: float, seed: int = 0, strict: bool = False
) -> tuple[list[Message], list[Message]]:
    """Split so whole wording families are unseen in training. By default a family is a verb,
    layout and opening together, so held-out messages combine familiar parts in new ways. With
    ``strict``, a family is the sentence layout alone (amount first, payee first, "You have",
    and so on): held-out messages are built in a way training never showed."""

    def family(style: str) -> str:
        parts = style.split("|")
        return (parts[5] if len(parts) > 6 else style) if strict else _family(style)

    families = sorted({family(m.style) for m in messages})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(families, max(1, int(len(families) * holdout_fraction)), replace=False))
    train = [m for m in messages if family(m.style) not in held]
    test = [m for m in messages if family(m.style) in held]
    return train, test


def _family(style: str) -> str:
    parts = style.split("|")
    return "|".join(parts[:1] + parts[5:7]) if len(parts) > 6 else style


def to_arrays(
    messages: list[Message],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inputs ``[n, 2 * MAX_TOKENS]``, field labels ``[n, MAX_TOKENS]`` (-1 on padding) and kind
    labels ``[n]``."""
    x = np.array([encode(m.text) for m in messages], dtype=np.int32)
    y = np.full((len(messages), MAX_TOKENS), -1, dtype=np.int32)
    for i, m in enumerate(messages):
        labels = m.fields[:MAX_TOKENS]
        y[i, : len(labels)] = labels
    k = np.array([int(m.kind) for m in messages], dtype=np.int32)
    return x, y, k

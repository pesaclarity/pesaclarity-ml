"""Synthetic questions for the ask reader, with an intent and a slot label on every token.

Questions are built from templates. A template is plain words with ``[a|b|]`` choices (an empty
choice drops the part), ``<word>`` choices from the shared :data:`LEXICON` and ``{slot}`` holes
filled from the value lists below, so each token's label is known exactly. Every question
records its template; evaluation holds whole templates out, so the reported score is for
phrasings the model never saw. Each intent's cue phrases also get short templates of their own,
so a held-out template is a new arrangement of known words. Questions are then roughened the way
people type on a phone: lower case, no question mark, small typos, a "hey" or "pls".

Version 2 adds templates for the phrasings the first reader got wrong in the app (counts and
sums for one payee, "how much to X", "X sent me", money in, "<account> balance", "how often",
and look-alikes that are out of scope), and, for training, invented payee and account names from
:mod:`akili.ask.names` in most payee and account holes. Version 1 templates keep their ids,
folds and, without a name pool, their exact questions, so their held-out score compares.

Everything is invented. Payees are well-known Kenyan chains and utilities, common first names
and surnames, or made-up names; there are no phone numbers or real messages.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from akili.ask import names
from akili.ask.schema import ASK_INTENTS, SLOT_LABELS
from akili.reader.tokens import MAX_TOKENS, encode, tokenise

T = TypeVar("T")

CATEGORY = SLOT_LABELS.index("category")
PAYEE = SLOT_LABELS.index("payee")
PERIOD = SLOT_LABELS.index("period")
ITEM = SLOT_LABELS.index("item")
ACCOUNT = SLOT_LABELS.index("account")
AMOUNT = SLOT_LABELS.index("amount")


@dataclass(frozen=True, slots=True)
class Question:
    text: str
    slots: list[int]
    """One label per token of :func:`akili.reader.tokens.tokenise`."""
    intent: int
    template: str
    """``intent:index``: the template the question came from."""


CATEGORIES = [
    "food",
    "groceries",
    "transport",
    "fare",
    "matatu",
    "airtime",
    "data",
    "data bundles",
    "rent",
    "school fees",
    "fees for the kids",
    "electricity",
    "power",
    "tokens",
    "water",
    "chama",
    "shopping",
    "clothes",
    "entertainment",
    "eating out",
    "fuel",
    "petrol",
    "health",
    "hospital",
    "medicine",
    "church",
    "tithe",
    "savings",
    "loans",
    "betting",
    "gas",
    "cooking gas",
    "internet",
    "wifi",
    "chakula",
    "salon",
    "kids",
    "family support",
    "gifts",
    "drinks",
    "lunch",
    "boda",
    "boda boda",
    "uber rides",
    "household",
    "toiletries",
    "furniture",
    "insurance",
    "nhif",
    "sha",
    "stock",
    "supplies",
    "repairs",
    "mama mboga",
    "sherehe",
    "black tax",
    "takeaway",
    "snacks",
    "bills",
    "utilities",
    "dstv",
    "subscriptions",
    "rent and bills",
    "books",
    "uniform",
    "travel",
    "holiday",
    "car",
    "garage",
]
PAYEES = [
    "Naivas",
    "Quickmart",
    "Carrefour",
    "Chandarana",
    "Java House",
    "Kenya Power",
    "KPLC",
    "Nairobi Water",
    "Safaricom",
    "Zuku",
    "Uber",
    "Bolt",
    "Little Cab",
    "KFC",
    "Artcaffe",
    "Total",
    "Shell",
    "Rubis",
    "Glovo",
    "Jumia",
    "Netflix",
    "Showmax",
    "DStv",
    "GOtv",
    "Spotify",
    "Kilimall",
    "Cleanshelf",
    "Magunas",
    "Tuskys",
    "Pizza Inn",
    "Chicken Inn",
    "Galitos",
    "John Kamau",
    "Mary Wanjiku",
    "Peter Otieno",
    "Grace Muthoni",
    "Faith Achieng",
    "Brian Kiprop",
    "Kevin Mwangi",
    "Amina Hassan",
    "David Ochieng",
    "Lucy Njeri",
    "Mama Njeri",
    "Baba Otieno",
    "mum",
    "my mum",
    "dad",
    "my brother",
    "my sister",
    "my landlord",
    "the landlord",
    "mama mboga",
    "the fundi",
    "the school",
    "the shop",
    "the chama",
    "Wanjiru",
    "Otieno",
    "Kamau",
    "Achieng",
    "Mutua",
    "Juma",
    "Halima",
    "Moses",
    "Esther",
    "Wafula",
    "Chebet",
    "Njoroge",
    "my wife",
    "my husband",
    "the cleaner",
    "the mechanic",
    "Joseph Mutiso",
]
PERIODS = [
    "last month",
    "this month",
    "this week",
    "last week",
    "today",
    "yesterday",
    "in March",
    "in september",
    "in Sept",
    "since June",
    "2025",
    "in 2025",
    "last year",
    "this year",
    "the last 3 months",
    "the past two weeks",
    "over the weekend",
    "on Monday",
    "last friday",
    "in january 2026",
    "between May and July",
    "so far this month",
    "since january",
    "mwezi uliopita",
    "wiki hii",
    "jana",
    "leo",
    "mwezi huu",
    "in December",
    "in Feb",
    "last 30 days",
    "the past 6 months",
    "since 1st March",
    "on 15th",
    "this morning",
    "last night",
    "this quarter",
    "last quarter",
    "over christmas",
    "during easter",
    "in the last week",
    "past month",
    "this past week",
    "the last two months",
    "every month",
    "in august",
    "in april",
    "in may",
    "in october",
    "in november",
    "last weekend",
    "since payday",
    "before payday",
    "this term",
    "last term",
    "the year so far",
]
ITEMS = [
    "tomatoes",
    "sukuma wiki",
    "bread",
    "milk",
    "sugar",
    "eggs",
    "unga",
    "maize flour",
    "cooking oil",
    "soap",
    "rice",
    "onions",
    "airtime cards",
    "sodas",
    "mandazi",
    "chapati",
    "beans",
    "cabbages",
    "potatoes",
    "bananas",
    "avocados",
    "sweets",
    "biscuits",
    "salt",
    "tea leaves",
    "kales",
    "spinach",
    "omena",
    "fish",
    "chicken",
    "mangoes",
    "water bottles",
    "charcoal",
    "paraffin",
    "nappies",
    "pads",
    "matches",
    "detergent",
    "juice",
    "yoghurt",
]
ACCOUNTS = [
    "M-Pesa",
    "mpesa",
    "MPESA",
    "Equity",
    "KCB",
    "Co-op",
    "co-op bank",
    "cash",
    "Airtel Money",
    "my bank",
    "Fuliza",
    "NCBA",
    "Family Bank",
    "Absa",
    "Stanbic",
    "the till",
    "M-Shwari",
    "KCB M-Pesa",
    "Pochi",
    "pochi la biashara",
    "my sacco",
    "the sacco",
    "Equitel",
    "DTB",
    "I&M",
    "my savings account",
    "T-Kash",
    "my card",
    "Loop",
    "Hustler Fund",
]
AMOUNTS = [
    "over 1,000",
    "more than 500",
    "above 5k",
    "under 200",
    "less than 2,000 bob",
    "between 1,000 and 5,000",
    "over ksh 10,000",
    "above 2,500",
    "below 100",
    "over 50k",
    "bigger than 3,000",
    "over kes 1,500",
    "under 1k",
    "more than 20,000",
    "less than 50 bob",
]
INCOME_CATEGORIES = [
    "salary",
    "business income",
    "freelance",
    "side hustle",
    "family support",
    "allowance",
    "bursary",
    "gifts",
    "rent income",
    "chama payout",
    "tips",
    "commission",
    "pension",
    "farm sales",
    "wages",
    "overtime",
]
MONEY = [
    "1,500",
    "2k",
    "ksh 3,000",
    "500 bob",
    "kes 1,200",
    "10,000",
    "800",
    "5k",
    "ksh 250",
    "20k",
    "4,500",
    "kes 15,000",
    "300 bob",
    "a thousand",
    "two thousand",
]
GOALS = [
    "school fees",
    "new phone",
    "emergency fund",
    "rent",
    "land",
    "car",
    "wedding",
    "holiday",
    "laptop",
    "business",
    "december",
    "christmas",
    "house",
    "fees",
    "chama",
    "shop stock",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "Jan", "Feb", "Sept", "Oct", "Nov", "Dec", "september",
    "october", "march", "august", "december", "june",
]  # fmt: skip
"""Bare month names: a period with no preposition ("compare September with October")."""

_VALUES: dict[str, tuple[list[str], int]] = {
    "category": (CATEGORIES, CATEGORY),
    "goal": (GOALS, CATEGORY),
    "incat": (INCOME_CATEGORIES, CATEGORY),
    "payee": (PAYEES, PAYEE),
    "period": (PERIODS, PERIOD),
    "period2": (PERIODS, PERIOD),
    "month": (MONTHS, PERIOD),
    "month2": (MONTHS, PERIOD),
    "item": (ITEMS, ITEM),
    "account": (ACCOUNTS, ACCOUNT),
    "amount": (AMOUNTS, AMOUNT),
    "money": (MONEY, AMOUNT),
}

V2_SHARE = 0.4
V4_WEIGHT = 0.5
"""How often a version 4 template is kept when drawn (else one of the intent's earlier ones is
drawn instead): version 4 adds phrasings without moving the reader far from the rest."""
"""With a name pool, the share of an intent's questions drawn from its version 2 and 3
templates (on top of their share of the rest)."""

BANK_PAYEES = [b for b in names.BANKS if b != "Cash"]
WIDE_ACCOUNTS = [*ACCOUNTS, *names.BANKS, "cash", "my cash", "CASH", "Co-operative Bank"]

_P = "[{period}|]"

# Shared word choices, written <name> in templates. Many templates share them, so a held-out
# template is a new way of putting familiar words together, as with the message reader.
LEXICON: dict[str, list[str]] = {
    "much": [
        "how much",
        "how much money",
        "how much pesa",
        "what amount",
        "how much cash",
        "how much doh",
        "hw much",
        "how mch",
    ],
    "did": ["did i", "have i", "i", "do i", "did we"],
    "spend": ["spend", "spent", "use", "used", "pay", "paid", "blow", "waste"],
    "spending": ["spending", "expenses", "costs", "expenditure", "outgoings", "matumizi"],
    "receive": ["receive", "received", "get", "got", "earn", "earned", "make", "made"],
    "send": ["send", "sent", "transfer", "transferred", "give", "gave", "forward", "tuma"],
    "count": [
        "how many times",
        "how many",
        "how often",
        "number of times",
        "mara ngapi",
        "count of",
        "how frequently",
    ],
    "payments": ["payments", "transactions", "purchases", "times", "trips", "malipo"],
    "biggest": [
        "biggest",
        "largest",
        "highest",
        "most expensive",
        "single biggest",
        "top",
        "heaviest",
    ],
    "payment": ["payment", "expense", "transaction", "spend", "purchase", "bill"],
    "who": [
        "who",
        "which people",
        "which person",
        "which shops",
        "where",
        "which places",
        "which merchants",
    ],
    "most": ["the most", "most", "the biggest share", "the highest amount", "zaidi"],
    "top": ["top", "main", "biggest", "leading", "major"],
    "money": ["money", "pesa", "cash", "doh", "salary", "income", "budget"],
    "go": ["go", "goes", "went", "gone", "end up", "disappear"],
    "categories": ["categories", "things", "areas", "category", "kinds of things"],
    "change": [
        "going up",
        "going down",
        "rising",
        "dropping",
        "increasing",
        "decreasing",
        "changing",
        "growing",
        "falling",
        "climbing",
        "trending",
    ],
    "changed": [
        "changed",
        "gone up",
        "gone down",
        "grown",
        "dropped",
        "moved",
        "shifted",
        "increased",
        "decreased",
    ],
    "trend": ["trend", "over time", "month by month", "pattern", "history", "graph"],
    "than": ["than", "compared to", "compared with", "vs", "versus", "against", "or"],
    "compare": ["compare", "comparison of", "difference between", "contrast"],
    "more": ["more", "less", "higher", "lower", "better", "worse", "cheaper"],
    "bills": ["bills", "payments due", "bill payments", "dues", "bill", "standing bills"],
    "due": ["due", "coming up", "pending", "upcoming", "to pay", "outstanding", "next"],
    "subs": [
        "subscriptions",
        "recurring payments",
        "monthly charges",
        "standing orders",
        "regular payments",
        "subscription",
        "subs",
        "monthly plans",
    ],
    "owe": ["owe", "still owe", "have to pay back", "need to repay", "must repay"],
    "loans": ["loans", "debts", "loan", "debt", "madeni", "credit"],
    "lend": ["lent", "lend", "loaned", "gave out", "borrowed out", "advanced"],
    "owes_me": [
        "owe me",
        "owes me",
        "owe me money",
        "owes me money",
        "has my money",
        "still owes me",
    ],
    "afford": ["afford", "manage", "handle", "spare"],
    "safe": [
        "safe to spend",
        "left to spend",
        "free to spend",
        "my spending limit",
        "budget left",
        "spendable",
        "allowance left",
    ],
    "left": ["left", "remaining", "remains", "available", "imebaki"],
    "balance": ["balance", "bal", "salio", "account balance"],
    "saved": ["saved", "put aside", "set aside", "saved up", "kept", "stashed"],
    "goal": ["goal", "target", "savings goal", "savings", "plan", "fund"],
    "progress": ["progress", "status", "how far", "update"],
    "income": ["income", "earnings", "money in", "salary", "pay", "mapato"],
    "from_where": ["come from", "comes from", "came from", "coming from", "is from"],
    "sources": ["sources", "source", "payers", "payer", "clients"],
    "sales": ["sales", "takings", "revenue", "sells", "mauzo"],
    "sell": ["sell", "sold", "sale", "move", "moved", "uza"],
    "shop": [
        "the shop",
        "the business",
        "my shop",
        "the kiosk",
        "my business",
        "the duka",
        "the stall",
        "my biashara",
    ],
    "bestsell": [
        "sells best",
        "sold the most",
        "is selling fast",
        "moves fastest",
        "best seller",
        "best sellers",
        "top sellers",
        "sells most",
        "popular",
    ],
    "profit": ["profit", "profits", "margin", "faida", "net income", "earnings after costs"],
    "low": [
        "running low",
        "running out",
        "almost finished",
        "finishing",
        "low",
        "out of stock",
        "inaisha",
        "nearly done",
    ],
    "stock": ["stock", "items", "products", "goods", "bidhaa", "inventory"],
    "customers": ["customers", "clients", "wateja", "buyers", "people"],
    "credit": [
        "on credit",
        "owing",
        "haven't paid",
        "have not paid",
        "unpaid",
        "with deni",
        "not paid",
    ],
    "fees": [
        "fees",
        "charges",
        "transaction costs",
        "transaction charges",
        "withdrawal fees",
        "gharama",
        "sending costs",
        "tariffs",
    ],
    "show": ["show", "show me", "list", "give me", "tell me", "display", "nionyeshe"],
    "greet": [
        "hi",
        "hello",
        "hey",
        "habari",
        "sasa",
        "niaje",
        "mambo",
        "hujambo",
        "yo",
        "howdy",
        "good morning",
        "good afternoon",
        "good evening",
        "morning",
        "vipi",
        "hey there",
        "hello there",
        "salaam",
    ],
    "buddy": ["boss", "buda", "msee", "friend", "akili", "there", "mate", "bro"],
    "thanks": [
        "thanks",
        "thank you",
        "thx",
        "asante",
        "asante sana",
        "shukran",
        "cheers",
        "ty",
        "thank u",
        "nashukuru",
        "much appreciated",
        "thanks a lot",
    ],
    "good": ["great", "nice", "cool", "ok", "perfect", "awesome", "poa", "sawa", "safi", "good"],
    "bye": [
        "bye",
        "goodbye",
        "see you",
        "later",
        "kwaheri",
        "tutaonana",
        "ciao",
        "good night",
        "see ya",
        "bye bye",
        "usiku mwema",
        "lala salama",
        "catch you later",
    ],
    "done": ["that's all", "i'm done", "nothing else", "that is all", "we're done", "enough"],
    "help": [
        "help",
        "what can you do",
        "how does this work",
        "how do i use this",
        "what do you do",
        "unaweza kufanya nini",
        "instructions",
        "guide me",
    ],
    "ask": ["ask", "ask you", "say", "type", "query"],
    "examples": ["examples", "example questions", "ideas", "suggestions", "sample questions"],
    "topic": [
        "bitcoin",
        "crypto",
        "the weather",
        "football",
        "the election",
        "politics",
        "the news",
        "a joke",
        "the dollar rate",
        "inflation",
        "the stock market",
        "forex",
        "arsenal",
        "the president",
        "a recipe",
        "my horoscope",
        "love",
        "history",
        "the capital of kenya",
        "maths homework",
        "a song",
        "the cbk rate",
        "movies",
    ],
    "advice": [
        "invest",
        "get rich",
        "make money fast",
        "get a loan",
        "buy shares",
        "buy land",
        "start a business",
        "save more",
        "buy bitcoin",
        "quit my job",
        "bet",
        "choose a bank",
        "pay less tax",
    ],
    "should": [
        "should i",
        "is it wise to",
        "do you think i should",
        "is it good to",
        "would you",
        "can you advise me to",
        "nishauri kama",
    ],
}

# Each intent's cue phrases, also used in short templates of their own (see :func:`_cue_templates`),
# so a held-out template tests a new arrangement of known words rather than unknown words.
CUES: dict[str, list[str]] = {
    "spent": [
        "nimechoma doh ngapi",
        "imekula pesa ngapi",
        "what did it cost",
        "what's the damage",
        "went out",
        "outgoing money",
        "nililipa ngapi",
        "nimelipa ngapi",
        "how much have i splashed",
        "my total cost at",
        "outflow",
        "expenses on",
        "how much has it taken",
        "how much did i use",
        "how much did i blow",
        "spend per week",
        "spend every month",
        "how much did i spend",
        "spending",
        "nimetumia ngapi",
        "what did i spend",
        "my expenses",
        "how much went on",
        "total spent",
        "matumizi yangu",
        "spent on",
        "amount spent",
        "what did it cost me",
    ],
    "received": [
        "imeingia",
        "pesa imeingia ngapi",
        "nilipokea ngapi",
        "landed in my account",
        "deposited",
        "incoming money",
        "inflow",
        "mshahara umeingia",
        "take home pay",
        "came my way",
        "did my salary come in",
        "brought in",
        "has come into",
        "came into my account",
        "how much was paid into",
        "money that came in",
        "how much came in",
        "my income",
        "money received",
        "nimepokea pesa ngapi",
        "how much did i get paid",
        "total received",
        "deposits",
        "what came in",
        "money in",
        "mapato yangu",
        "how much did i earn",
    ],
    "sent_to": [
        "nimetuma ngapi",
        "nilituma",
        "sent over to",
        "have i given",
        "got from me",
        "did i send home",
        "sent to family",
        "money went from me to",
        "how much did i send",
        "money sent to",
        "nimetumia pesa ngapi",
        "transfers to",
        "sends to",
        "what did i give",
        "how much have i transferred",
        "total sent to",
    ],
    "received_from": [
        "how much came from",
        "did he pay me",
        "sent me money",
        "been sent by",
        "did he send me",
        "did she pay me",
        "how much did they give me",
        "paid me via",
        "has she sent",
        "received from",
        "how much did they send me",
        "money from",
        "amenitumia pesa ngapi",
        "payments from",
        "what did i get from",
        "has paid me",
        "total from",
        "sent me",
    ],
    "count_payments": [
        "mara ngapi",
        "how many times did i send",
        "number of times",
        "how often do i visit",
        "how many transactions to",
        "how many times",
        "how many payments",
        "how often",
        "mara ngapi",
        "number of transactions",
        "count my payments",
        "how many transactions",
    ],
    "biggest_payment": [
        "biggest amount i paid",
        "largest amount",
        "highest amount i spent",
        "my top payment",
        "biggest payment",
        "largest expense",
        "most expensive purchase",
        "highest transaction",
        "biggest single payment",
        "my largest spend",
        "where did i make my biggest payment",
    ],
    "top_payees": [
        "who do i pay the most",
        "top payees",
        "where do i shop most",
        "who gets most of my money",
        "biggest recipients",
        "top shops",
        "which merchant takes most",
    ],
    "top_categories": [
        "which categories take most",
        "the bulk of my money",
        "where is my money going",
        "inaenda wapi",
        "category breakdown",
        "rank my categories",
        "what takes up most of my budget",
        "spending per category",
        "my top category",
        "biggest category",
        "which category takes most",
        "where does my money go",
        "top categories",
        "spending by category",
        "pesa yangu inaenda wapi",
        "breakdown of my spending",
        "what eats my money",
        "biggest expense categories",
        "what do i spend most on",
    ],
    "trend": [
        "am i spending more",
        "going up lately",
        "than usual",
        "is it going up",
        "trend",
        "over time",
        "has it changed",
        "month by month",
        "getting more expensive",
        "spending more lately",
        "is it rising",
        "spending less these days",
    ],
    "compare": [
        "compare",
        "versus",
        "compared to",
        "more than",
        "less than",
        "difference between",
        "vs",
    ],
    "bills_due": [
        "when do i pay",
        "next payment to",
        "bills due",
        "upcoming bills",
        "what do i need to pay",
        "nina bills gani",
        "what is due",
        "when is it due",
        "next bills",
        "pending bills",
        "when do i pay next",
    ],
    "subscriptions": [
        "how much do i pay for subscriptions",
        "subscriptions a month",
        "cost of subscriptions",
        "total subscriptions",
        "monthly plans",
        "subscriptions",
        "recurring payments",
        "monthly charges",
        "standing orders",
        "what am i paying for every month",
        "do i have any subscriptions",
        "costing me every month",
    ],
    "debts": [
        "is my loan cleared",
        "paid off my loan",
        "is my fuliza paid",
        "how much do i owe",
        "my loans",
        "my debts",
        "nadaiwa ngapi",
        "outstanding loans",
        "loan repayment",
        "do i owe anyone",
        "total debt",
        "madeni yangu",
    ],
    "lent": [
        "namdai",
        "ananiwia",
        "has he refunded",
        "who borrowed money from me",
        "friends loans",
        "does he still owe me",
        "ameregesha pesa",
        "amerudisha",
        "how much did i lend",
        "did he return my money",
        "has she repaid",
        "money i gave out",
        "who owes me personally",
        "how much have i lent",
        "money i lent out",
        "nadai ngapi",
        "has he paid me back",
        "people i lent money to",
        "who did i lend money to",
        "how much am i owed by friends",
        "returned my money",
    ],
    "safe_to_spend": [
        "how much can i use",
        "how much can i spend today",
        "until payday",
        "can i spend",
        "safe to spend",
        "how much can i spend",
        "left to spend",
        "nimebaki na pesa ngapi",
        "can i afford",
        "am i within budget",
        "budget left",
        "will i make it to payday",
        "okay to spend",
    ],
    "balance": [
        "balance",
        "salio",
        "how much do i have now",
        "nina pesa ngapi",
        "what is left in my account",
        "account balance",
        "check balance",
    ],
    "goals": [
        "how much more do i need",
        "to reach my goal",
        "goal status",
        "my goal",
        "savings goal",
        "how much have i saved",
        "am i on track",
        "goal progress",
        "when will i reach my target",
        "nimeweka akiba ngapi",
        "how close am i",
    ],
    "income_sources": [
        "who pays me the most",
        "my income sources",
        "sources of income",
        "where does my money come from",
        "income sources",
        "who pays me most",
        "pesa yangu inatoka wapi",
        "main source of income",
        "top payers",
        "breakdown of my income",
        "split of income",
    ],
    "sales": [
        "came into the shop",
        "how much did the shop take",
        "sales of",
        "mauzo ya",
        "how is it selling",
        "sales",
        "how much did i sell",
        "takings",
        "revenue",
        "nimeuza ngapi",
        "mauzo",
        "how much did the shop make",
        "how many did i sell",
    ],
    "top_items": [
        "best sellers",
        "what sells best",
        "top selling products",
        "bidhaa gani inauzwa sana",
        "inauzwa sana",
        "what do customers buy most",
        "what is moving fastest",
        "most popular items",
    ],
    "profit": [
        "profit",
        "did i make a profit",
        "faida",
        "making a loss",
        "did the shop lose money",
        "after costs",
        "net profit",
        "am i in profit",
        "did the business make money",
    ],
    "low_stock": [
        "do i need to restock",
        "buy more stock",
        "reorder",
        "what should i restock",
        "running low",
        "running out",
        "low stock",
        "restock",
        "almost finished",
        "inaisha",
        "imeisha",
        "out of stock",
        "stock levels",
    ],
    "who_owes_me": [
        "customers still owing",
        "customers with debts",
        "credit given to customers",
        "took goods on credit",
        "wateja ambao hawajalipa",
        "nani ana deni",
        "outstanding customer balances",
        "owed by customers",
        "credit book",
        "haven't cleared their deni",
        "which customers owe me",
        "customers on credit",
        "debtors",
        "my debtors",
        "unpaid invoices",
        "nani hajalipa",
        "deni za wateja",
        "who hasn't paid",
        "who owes the shop",
        "debtors list",
    ],
    "fees": [
        "charged me",
        "how much has it charged me",
        "bank charges",
        "fees",
        "charges",
        "transaction costs",
        "withdrawal fees",
        "gharama za mpesa",
        "how much did i pay in fees",
        "what did charges cost me",
        "lost to fees",
    ],
    "refine": [
        "and",
        "what about",
        "how about",
        "only",
        "just",
        "same for",
        "excluding",
        "without",
        "except",
        "and then",
    ],
    "hi": [
        "hi",
        "hello",
        "habari",
        "habari yako",
        "niaje",
        "mambo",
        "good morning",
        "sasa",
        "hey there",
        "how are you",
    ],
    "thanks": [
        "poa",
        "sawa sana",
        "nice thanks",
        "cool thanks",
        "good boss",
        "thanks",
        "thank you",
        "asante",
        "asante sana",
        "that helped",
        "that was helpful",
        "poa sana",
        "safi sana",
        "great thanks",
    ],
    "bye": [
        "talk later",
        "see you tomorrow",
        "see you soon",
        "catch you later",
        "bye",
        "goodbye",
        "kwaheri",
        "see you",
        "that's all",
        "i'm done",
        "nothing else",
        "lala salama",
        "see you later",
    ],
    "help": [
        "what are you for",
        "what are you able to do",
        "how do i ask",
        "what questions can you answer",
        "what kind of questions",
        "help",
        "what can you do",
        "examples",
        "show me some examples",
        "how does this work",
        "what can i ask",
        "give me ideas",
        "menu",
    ],
    "out_of_scope": [
        "write an email",
        "write a poem",
        "what's the weather",
        "teach me how to",
        "help me with my homework",
        "take a loan from tala",
        "who is the president",
        "the dollar rate",
        "which movie should i watch",
        "you are funny",
        "call my mum",
        "price of fuel",
        "interest rates",
        "translate this",
        "define this word",
        "will it rain",
        "who won the match",
        "what do you think about",
        "tell me about the news",
        "cook ugali",
        "marry me",
        "who are you",
        "who made you",
        "tell me a joke",
        "the weather",
        "what should i do with my life",
        "who will win",
        "recommend a phone",
        "should i invest",
        "are you human",
        "bitcoin",
        "politics",
        "football",
    ],
}

TEMPLATES: dict[str, list[str]] = {
    "spent": [
        "<much> <did> <spend> on {category} " + _P,
        "<much> [have i|did i] <spend> [on|for] {category}",
        "<much> <did> <spend> {period}",
        "what did i <spend> on {category} " + _P,
        "total <spending> [on|for] {category} " + _P,
        "{category} <spending> " + _P,
        "<much> went [to|on] {category} " + _P,
        "<much> did i <spend> at {payee} " + _P,
        "nimetumia [pesa|doh|] ngapi [kwa|on] {category} " + _P,
        "[what|how much] [is|was] my [total|] <spending> " + _P,
        "<much> <did> <spend> [on|for] {category} [from|on|using] {account} " + _P,
        "<show> my {category} <spending> " + _P,
        "{period} <much> <did> <spend> [on {category}|]",
        "amount <spend> on {category} " + _P,
        "<much> [has gone|went] [on|into] {category} " + _P,
        "<much> <did> <spend> on {category} [a month|every month|per week|weekly]",
        "what did {category} cost me " + _P,
        "<much> did i <spend> on [payments|things|stuff] {amount} " + _P,
        "my <spending> [at|with] {payee} " + _P,
        "<much> [has|did] {category} [taken|take|eaten] " + _P,
        "[total|] <spend> at {payee} " + _P,
        "{category} [this|] [total|amount] " + _P + " [please|pls|]",
        "i want to know <much> i <spend> [on|for] {category} " + _P,
        "<much> <did> <spend> [in total|altogether|overall|] {period}",
        "what [have i|did i] <spend> [in total|altogether|so far] " + _P,
        "my total <spending> {period}",
        "<much> [have i|did i] <spend> [so far|this far|already]",
    ],
    "received": [
        "<much> <did> <receive> {period}",
        "<much> came in " + _P,
        "[total|my] <income> " + _P,
        "<much> [was|is] my <income> " + _P,
        "what did i <receive> {period}",
        "<much> was paid [in|into] [my|] {account} " + _P,
        "<much> [has come|came] into {account} " + _P,
        "[money|cash|pesa] [in|received] " + _P,
        "<much> did i get paid " + _P,
        "nimepokea [pesa|doh|] ngapi " + _P,
        "what came in " + _P,
        "<show> [me|] [my|] <income> " + _P,
        "total [received|<income>] " + _P,
        "<much> <did> <receive> [on|in|via] {account} " + _P,
        "what [was|is] my total <income> [for|] {period}",
        "deposits [into|to] {account} " + _P,
        "<much> [money|] [have i|did i] [received|got] in total " + _P,
        "<much> {incat} <did> <receive> " + _P,
        "[my|] {incat} [income|money|] " + _P,
        "<much> came in from {incat} " + _P,
        "what did {incat} bring in " + _P,
    ],
    "sent_to": [
        "<much> <did> <send> [to|] {payee} " + _P,
        "what <did> <send> {payee} " + _P,
        "total <send> to {payee} " + _P,
        "<much> [do|did] i <send> {payee} [every month|monthly|a month|]",
        "money [sent|i sent|transferred] to {payee} " + _P,
        "nimetumia {payee} [pesa|doh|] ngapi " + _P,
        "when did i last <send> [money|pesa|] to {payee}",
        "{payee} <much> [have i|did i] <send> " + _P,
        "<much> <did> <send> {payee} [through|on|via] {account}",
        "did i <send> {payee} [money|anything|] {period}",
        "transfers to {payee} " + _P,
        "<show> [me|] what i <send> {payee} " + _P,
        "[my|] sends to {payee} " + _P,
    ],
    "received_from": [
        "<much> [did|has] {payee} [send|sent|given|transferred] me " + _P,
        "<much> <did> [received|receive|got|get] from {payee} " + _P,
        "what has {payee} [sent|given|paid] me " + _P,
        "[money|pesa|cash] from {payee} " + _P,
        "total from {payee} " + _P,
        "did {payee} [send|pay] [me|] [the|] [money|pesa|cash|] " + _P,
        "{payee} amenitumia [pesa|doh|] ngapi " + _P,
        "has {payee} paid me " + _P,
        "<much> came from {payee} " + _P,
        "<much> [have i|did i] been [sent|paid] by {payee}",
        "<much> did {payee} [pay|give|send] me [on|via|through] {account}",
        "payments from {payee} " + _P,
        "<show> [me|] what {payee} [sent|paid] me " + _P,
    ],
    "count_payments": [
        "<count> <did> [pay|paid|send to|sent to|buy from] {payee} " + _P,
        "<count> <payments> " + _P,
        "<count> <did> <spend> on {category} " + _P,
        "number of <payments> [to|at] {payee} " + _P,
        "count [my|] <payments> [to|at|for] {payee}",
        "<count> do i [pay|buy from|go to] {payee}",
        "<count> {category} <payments> " + _P,
        "<count> [nimelipa|nimetuma] {payee} " + _P,
        "<count> <payments> [were|are] {amount} " + _P,
        "<count> <payments> [on|from] {account} " + _P,
        "<count> <did> bought {category} " + _P,
        "<count> <payments> did i [do|make] {period}",
        "<count> <did> paid for {category} at {payee}",
    ],
    "biggest_payment": [
        "what [was|is] my <biggest> <payment> " + _P,
        "<biggest> <payment> [on {category}|] " + _P,
        "what did i <spend> the most on in one [go|payment] " + _P,
        "[my|the] <biggest> [thing|purchase] i [bought|paid for] " + _P,
        "which <payment> was the <biggest> " + _P,
        "<show> [me|] my <biggest> <payment> " + _P,
        "where did i [make|do] my <biggest> <payment> " + _P,
        "<biggest> amount i [paid|sent|spent] " + _P,
        "what was the <biggest> [{category}|] <payment> " + _P,
        "my <biggest> [single|one] <payment> [at {payee}|] " + _P,
        "the <biggest> <payment> [from|on] {account} " + _P,
    ],
    "top_payees": [
        "<who> [do i|did i] [pay|send money to|spend on] <most> " + _P,
        "[my|] <top> [payees|merchants|shops|recipients] " + _P,
        "where do i [shop|buy] <most> " + _P,
        "which [shop|business|merchant|supermarket] [gets|got|takes] <most> [of|] my <money> " + _P,
        "who [gets|got|is getting] <most> [of|] my <money> " + _P,
        "<show> [the|] people i [pay|send money to] <most>",
        "<who> [have i|did i] <send> <most> [money|] [to|] " + _P,
        "<top> [five|three|5|10] [payees|places|shops] " + _P,
        "<who> [have i|did i] paid <most> " + _P,
        "<biggest> [payees|recipients] " + _P,
    ],
    "top_categories": [
        "where does my <money> <go> " + _P,
        "[what|which] <categories> do i <spend> <most> on " + _P,
        "[my|] <top> [spending|] <categories> " + _P,
        "where [did|has] my <money> <go> " + _P,
        "what am i spending on <most> " + _P,
        "pesa yangu inaenda wapi " + _P,
        "[breakdown|split] of my <spending> " + _P,
        "<show> [me|] [my|] <spending> by category " + _P,
        "what eats [most of|] my <money> " + _P,
        "<biggest> [<spending>|spending areas] " + _P,
        "on what do i <spend> <most> " + _P,
        "which [category|<categories>] [takes|took] <most> [of|] [my|] <money> " + _P,
        "what [is|was] my <top> [category|expense category] " + _P,
    ],
    "trend": [
        "is my {category} [<spending>|bill] <change>",
        "[has|is] my <spending> [<changed>|<change>] " + _P,
        "how has my {category} [<spending>|] <changed> " + _P,
        "{category} <trend> " + _P,
        "am i spending more on {category} [lately|these days|recently|than usual]",
        "is {category} getting more expensive [lately|]",
        "how [is|has] my <spending> [<change>|<changed>] " + _P,
        "<show> [me|] the <trend> [for|of] {category} " + _P,
        "[<spending>|<income>] <trend> " + _P,
        "has my <income> <changed> " + _P,
        "am i spending less [these days|now|lately]",
        "is my <spending> <change>",
        "how [are|have] my {category} costs [<change>|<changed>] over {period}",
        "[are|is] my sales <change> " + _P,
    ],
    "compare": [
        "did i <spend> <more> {period} <than> {period2}",
        "<compare> {period} [with|and|to|vs] {period2}",
        "{period} <than> {period2} [<spending>|]",
        "how does {period} compare [with|to] {period2}",
        "was {period} <more> <than> {period2}",
        "<spending> {period} <than> {period2}",
        "did i <spend> <more> on {category} {period} <than> {period2}",
        "is my {category} [<spending>|] <more> {period} <than> {period2}",
        "<compare> {period} and {period2} [<spending>|]",
        "did i <receive> <more> {period} <than> {period2}",
        "{category} {period} <than> {period2}",
        "<compare> my {category} [<spending>|] {period} [and|with] {period2}",
        "[are|were] sales <more> {period} <than> {period2}",
        "<more> <than> {period2} [overall|]",
    ],
    "bills_due": [
        "what <bills> [are|do i have] <due> " + _P,
        "[any|are there] <bills> <due> " + _P,
        "when is [my|the] {category} [bill|payment] due",
        "<due> <bills>",
        "what do i need to pay " + _P,
        "when do i pay {payee} [next|]",
        "is {category} due " + _P,
        "<show> [my|] <bills> [due|] " + _P,
        "what's <due> " + _P,
        "nina <bills> gani " + _P,
        "remind me [what|which <bills>] i have to pay " + _P,
        "when is {payee} due",
        "which <bills> are <due> " + _P,
    ],
    "subscriptions": [
        "what <subs> do i have",
        "<show> [my|] <subs>",
        "[which|what] [services|apps|<subs>] am i paying for [every month|monthly|]",
        "<much> do i pay for <subs> [a month|every month|]",
        "am i [still|] paying for {payee} [every month|monthly|]",
        "[my|] [monthly|recurring] <subs>",
        "do i have any <subs>",
        "what <subs> [do i have|go out] [every month|]",
        "<much> is {payee} [costing me|charging me] [every month|monthly|]",
        "[which|what] <subs> [can i cancel|am i paying]",
        "total [cost of|] <subs> " + _P,
    ],
    "debts": [
        "<much> do i <owe> [in total|altogether|]",
        "[what|which] <loans> do i have",
        "<much> [is left|do i <owe>] on my <loans> [from {payee}|]",
        "<much> do i <owe> {payee}",
        "[my|] [outstanding|] <loans>",
        "<much> do i <owe> on {account}",
        "when is my [next|] <loans> [repayment|payment] due",
        "do i <owe> anyone [money|]",
        "nadaiwa [pesa|] ngapi",
        "total <loans> [outstanding|remaining|]",
        "<much> <loans> do i have [on {account}|]",
        "is my {account} <loans> [cleared|paid off]",
    ],
    "lent": [
        "who <owes_me>",
        "<much> do [people|friends|family] <owes_me>",
        "<much> have i <lend> [to {payee}|] " + _P,
        "<much> does {payee} <owes_me>",
        "[money|loans] i <lend>",
        "who [have i lent|did i lend] money [to|]",
        "has {payee} paid [me|] back",
        "<much> am i owed [by friends|by family|]",
        "nadai [pesa|] ngapi",
        "[did|has] {payee} [return|returned|repaid] [my|the] money",
        "money owed to me [by {payee}|]",
        "<show> [the|] people i <lend> [money|] to",
    ],
    "safe_to_spend": [
        "<much> can i [spend|use] [{period}|until payday]",
        "[what's|what is] [my|] <safe>",
        "can i <afford> [to spend|] {money} " + _P,
        "am i [over|within] [my|] budget " + _P,
        "<much> [do i have|is] <left> to spend " + _P,
        "[is it|am i] okay to spend {money} " + _P,
        "<much> [do i have|is] <left> [for|until] [the month|payday|the week]",
        "nimebaki na pesa ngapi " + _P,
        "what's my [budget|<safe>] <left> " + _P,
        "can i <afford> {category} " + _P,
        "<much> is <safe> " + _P,
        "will i make it to [payday|month end] [with my money|]",
    ],
    "balance": [
        "what is my [{account}|] <balance>",
        "[{account}|my] <balance>",
        "<much> [do i have|is there] [in|on] [my|] {account}",
        "<much> do i have [now|right now|]",
        "check [my|] [{account}|] <balance>",
        "nina pesa ngapi [kwa {account}|]",
        "what's [left|remaining] in [my|] {account}",
        "[total|current] <balance> [across|in] [all my|my] accounts",
        "<balance> [la|on|for] {account}",
        "what's my <balance> [on {account}|] [now|today|]",
        "<much> is [on|in] {account} [now|]",
    ],
    "goals": [
        "how is my {goal} <goal> [going|doing]",
        "[am i on track|how close am i] [for|to|with] [my|] {goal} [<goal>|]",
        "<much> have i <saved> [for {goal}|] " + _P,
        "[my|] <goal> <progress>",
        "<show> [my|] <goal> <progress>",
        "when will i reach [my|] {goal} <goal>",
        "<much> more do i need [for|to reach] [my|] {goal} [<goal>|]",
        "nimeweka [akiba|pesa] ngapi [ya {goal}|]",
        "<progress> [on|for|with] [the|my] {goal} [<goal>|]",
        "[have i|did i] [reach|reached|hit] my {goal} <goal>",
        "<much> [is in|have i got in] my {goal} <goal>",
    ],
    "income_sources": [
        "where does my <income> <from_where>",
        "[what are|<show>] my <income> <sources>",
        "who [pays|paid|sent] me <most> " + _P,
        "where [did|does|is] my <income> <from_where> " + _P,
        "<top> <income> <sources> " + _P,
        "who [are|is] my <top> <sources> " + _P,
        "[breakdown|split] of [my|] <income> " + _P,
        "pesa yangu inatoka wapi " + _P,
        "is most of my <income> from {incat} [or {incat}|]",
        "which [client|customer|person] [pays|paid] me <most> " + _P,
        "where [is|was] my money <from_where> " + _P,
    ],
    "sales": [
        "<much> did i <sell> " + _P,
        "[total|my] <sales> " + _P,
        "how [were|are|was] [my|] <sales> " + _P,
        "<much> did <shop> [make|take in|bring in] " + _P,
        "<much> {item} did i <sell> " + _P,
        "<sales> [for|from|of] {item} " + _P,
        "what [are|were] [the|my] <sales> " + _P,
        "nimeuza ngapi " + _P,
        "<much> came [in|into] <shop> " + _P,
        "how many {item} did i <sell> " + _P,
        "<sales> [on|via|through] {account} " + _P,
        "<much> has <shop> [sold|made] " + _P,
    ],
    "top_items": [
        "what <bestsell> " + _P,
        "[my|] [best|top] [sellers|selling <stock>] " + _P,
        "which <stock> [sell|sold] the most " + _P,
        "[most|best] [popular|sold] <stock> " + _P,
        "what do <customers> buy <most> " + _P,
        "which [product|item] [makes|brings] [me|] the most money " + _P,
        "what's moving [fastest|the most] " + _P,
        "<stock> gani inauzwa sana " + _P,
        "[top|best] selling [product|item|<stock>] " + _P,
        "what [is|was] my [best seller|<bestsell>] " + _P,
        "which <stock> <bestsell> at <shop> " + _P,
    ],
    "profit": [
        "did i make [a|any] <profit> " + _P,
        "[what's|what is|how much is] my <profit> " + _P,
        "<much> <profit> [did i make|did <shop> make|have i made] " + _P,
        "[am i|is <shop>] making [money|a <profit>|a loss] " + _P,
        "<profit> [on|for|from] {item} " + _P,
        "[<profit>|loss] " + _P,
        "<much> did i [make|earn] after [costs|expenses] " + _P,
        "<profit> [yangu|ya biashara|] " + _P,
        "did <shop> [lose|make] money " + _P,
        "[am i|was i] in [<profit>|loss] " + _P,
    ],
    "low_stock": [
        "what am i <low> [of|on|]",
        "[which|what] <stock> [are|is] <low>",
        "[low|remaining] <stock>",
        "do i need to [restock|buy more <stock>|reorder]",
        "<much> {item} [is left|do i have left|remains]",
        "[am i|are we] [out of|low on] {item}",
        "what [should|do] i restock",
        "<stock> [levels|check|count]",
        "nini [inaisha|imeisha] [kwa duka|]",
        "is {item} <low>",
        "what <stock> [is|has] [finished|run out]",
    ],
    "who_owes_me": [
        "<customers> who [still|] owe [me|the shop|<shop>]",
        "which <customers> [<owes_me>|are <credit>]",
        "<customers> <credit>",
        "who [hasn't paid|has not paid] [me|] [for {item}|] [yet|]",
        "[unpaid|outstanding] [customer|] [invoices|credit|debts from customers]",
        "<much> do [my|] <customers> <owe> [me|<shop>|]",
        "[my|the] debtors [list|]",
        "who took {item} <credit>",
        "deni za <customers>",
        "nani hajalipa [deni|]",
        "who owes <shop>",
        "<show> <customers> who <owes_me>",
    ],
    "fees": [
        "<much> [have i paid|did i pay] in <fees> " + _P,
        "{account} <fees> " + _P,
        "<much> [is|was] [the|] <fees> " + _P,
        "[total|] <fees> " + _P,
        "<much> [goes|went] to <fees> " + _P,
        "<much> has {account} charged me " + _P,
        "what [did|do] <fees> cost me " + _P,
        "<fees> za [mpesa|kutuma] " + _P,
        "[how much|what] [did|do] i lose to <fees> " + _P,
        "<show> [me|] my <fees> " + _P,
    ],
    "refine": [
        "and [only|just] {payee}",
        "[what about|how about|and] {period}",
        "and {category}",
        "[only|just] {category}",
        "same for {category}",
        "[what about|and] {payee}",
        "[and|what about] {period} [then|]",
        "and [on|in|from] {account}",
        "only {account}",
        "and [for|with] {item}",
        "[same|the same] [thing|] for {period}",
        "[ok|okay|and] [now|] [for|] {category} " + _P,
        "{period} [then|instead|too]",
        "[excluding|without|except] {category}",
        "[just|only] [the|] {amount} [ones|payments|]",
        "how about [at|for] {payee}",
        "and {period} [for {category}|]",
    ],
    "hi": [
        "<greet>",
        "<greet> <buddy>",
        "habari [yako|za asubuhi|za leo|ya leo|]",
        "<greet> how are you",
        "how are you [doing|today|]",
        "<greet> [<greet>|]",
    ],
    "thanks": [
        "<thanks>",
        "<thanks> [a lot|so much|very much|again|<buddy>]",
        "<good> <thanks>",
        "that [helps|helped|was helpful] [<thanks>|]",
        "<good> [sana|<buddy>]",
        "<thanks> for the help",
    ],
    "bye": [
        "<bye>",
        "[ok|okay|alright|<good>] <bye>",
        "<done> [for now|today|<bye>|]",
        "[talk|see you] [later|tomorrow|soon]",
        "<bye> <buddy>",
        "<bye> [for now|then|]",
    ],
    "help": [
        "<help>",
        "what [can|should] i <ask> [you|]",
        "what [questions|things] can you answer",
        "[show me|give me] [some|] <examples>",
        "[how do i|how can i] <ask> [a question|something]",
        "what are you [for|able to do]",
        "[menu|options|commands]",
        "what kind of questions [can i <ask>|do you know]",
        "<help> [please|<buddy>]",
    ],
    "out_of_scope": [
        "<should> <advice>",
        "how [do i|can i|to] <advice>",
        "tell me [about|] <topic>",
        "what do you [think|know] about <topic>",
        "[talk|chat] [to me|] about <topic>",
        "who [will win|won] the [election|match|league]",
        "[write|send] [an email|a message|an sms] to my [boss|landlord]",
        "[recommend|suggest] a [good|cheap] [phone|restaurant|school]",
        "[i'm|i am] [bored|sad|tired|hungry]",
        "who [are you|made you|created you]",
        "[translate|define] this word",
        "[can you|] [play|sing] [music|a song]",
        "which [bank|sacco] [is|has] the [best|lowest] interest",
        "how [do i|to] cook [pilau|ugali|githeri]",
        "will it rain [tomorrow|today]",
        "[i love you|marry me|are you human]",
        "[explain|what is] <topic>",
        "what should i do with my life",
    ],
}

_CUE_TEMPLATES = (
    "<cue_{intent}>",
    "<cue_{intent}> [{{period}}|]",
    "[tell me|show me|please|so|] <cue_{intent}>",
)
_NO_PERIOD = {"hi", "thanks", "bye", "help", "out_of_scope"}


def _cue_templates() -> None:
    """Add each intent's cue lexicon and three short templates that use it."""
    for intent, cues in CUES.items():
        LEXICON[f"cue_{intent}"] = cues
        extra = [t.format(intent=intent) for t in _CUE_TEMPLATES]
        if intent in _NO_PERIOD:
            extra[1] = f"<cue_{intent}> [<cue_{intent}>|]"
        if intent == "refine":
            extra = ["<cue_refine> {category}", "<cue_refine> {payee}", "<cue_refine> {period}"]
        TEMPLATES[intent].extend(extra)


# More phrasings, many of them Sheng and Swahili mixed with English, for the intents that are
# easiest to confuse.
LEXICON["cash"] = ["pesa", "doh", "ganji", "chapaa", "mulla", "cash", "money", "dough", "munde"]
LEXICON["burn"] = ["choma", "chomea", "kula", "maliza", "tumia", "blow", "burn"]
LEXICON["just_say"] = ["", "", "manze", "buda", "si", "ebu", "aki", "yaani", "kwani"]

_EXTRA_TEMPLATES: dict[str, list[str]] = {
    "spent": [
        "<just_say> nime<burn> <cash> ngapi [kwa|on] {category} " + _P,
        "<just_say> nime<burn> <cash> ngapi " + _P,
        "how much <cash> did i <burn> on {category} " + _P,
        "{category} imekula <cash> ngapi " + _P,
        "{category} ilikula how much " + _P,
        "<much> [money|<cash>] [have i|did i] [put|thrown|lost] into {category} " + _P,
        "what's the damage on {category} " + _P,
        "tell me [what|how much] {category} cost [me|us] " + _P,
        "give me my {category} [total|figure|number] " + _P,
        "[sum|total] of [my|] {category} <payments> " + _P,
        "<much> [went out|left my account|left] [for|on] {category} " + _P,
        "<much> went out " + _P,
        "<much> [have i|did i] spent [using|on|from] {account} " + _P,
        "<much> did {account} payments [to {payee}|for {category}] come to " + _P,
        "<much> [have i|did i] [splashed|dropped|blown] at {payee} " + _P,
        "{payee} [has|have] taken [how much|<much>] [from me|of my <cash>] " + _P,
        "<much> [do i|did i] [usually|normally|] <spend> at {payee} [a week|a month|]",
        "what [is|was] [my|the] [total|] [cost|bill] at {payee} " + _P,
        "outgoing [money|payments|<cash>] " + _P,
        "[all|total] [my|] expenses [for|on] {category} " + _P,
        "<much> did i <spend> on {category} [payments|purchases] {amount} " + _P,
        "<much> [went|has gone] on things {amount} " + _P,
        "[amount|money] [gone|used] on {category} [and {category}|] " + _P,
        "matumizi ya {category} " + _P,
        "nililipa ngapi [kwa|for] {category} " + _P,
        "nimelipa ngapi [kwa|to|at] {payee} " + _P,
        "<much> of my <cash> went to {category} " + _P,
        "what did i <spend> at {payee} [on {category}|] " + _P,
        "[my|] [total|] outflow " + _P,
        "how much [did|have] [we|i] used for {category} " + _P,
    ],
    "received": [
        "<much> <cash> [imeingia|iliingia|imekuja] " + _P,
        "nilipokea <cash> ngapi " + _P,
        "<cash> [imeingia|iliingia] ngapi [kwa {account}|] " + _P,
        "<much> [did|has] [money|<cash>] [come|landed|arrived] [in|into] [my|] {account} " + _P,
        "<much> [got|was] deposited " + _P,
        "<much> [have i|did i] [been paid|received in total] " + _P,
        "what [was|is] my [total|] [take home|pay|<income>] " + _P,
        "<much> [money|<cash>] [have i|did i] [pulled in|brought in|made] " + _P,
        "<much> [money|<cash>] [came|landed] [my way|to me] " + _P,
        "incoming [money|payments|<cash>] " + _P,
        "[sum|total] of [everything|all] i received " + _P,
        "<much> [did|has] my {incat} [come to|paid] " + _P,
        "did [my|the] {incat} [come in|land|arrive] " + _P,
        "[was|has] my {incat} [paid|deposited] [into {account}|] " + _P,
        "<much> did i receive [through|via|on] {account} " + _P,
        "[my|] inflow " + _P,
        "<much> [money|<cash>] [have i|did i] got in total " + _P,
        "what did i [bring in|earn|make] [in total|altogether|]  " + _P,
        "[money|<cash>] [received|in] on {account} " + _P,
        "mshahara [umeingia|imeingia|] " + _P,
    ],
    "sent_to": [
        "nimetuma <cash> ngapi [kwa|to] {payee} " + _P,
        "nilituma {payee} [<cash>|] ngapi " + _P,
        "<much> [have i|did i] [sent|sent over|sent across] to {payee} [in total|altogether|]",
        "[total|sum] [of|] [money|<cash>] [i've|i have] sent {payee} " + _P,
        "<much> [money|<cash>] [have i|did i] [given|sent] to {payee} " + _P,
        "what [is|was] the total i [sent|gave|transferred] [to|] {payee} " + _P,
        "<much> [did|does] {payee} get from me " + _P,
        "<much> has {payee} [received|gotten|got] from me " + _P,
        "what [have i|did i] [sent|forwarded] to {payee} [via|on|through] {account} " + _P,
        "<much> have i [sent home|sent to family|supported] {payee} " + _P,
        "<much> [did|do] i [send|give] {payee} [for upkeep|as support|for fees|]",
        "<much> [money|<cash>] went from me to {payee} " + _P,
    ],
    "count_payments": [
        "<count> [did|have] i [sent|paid] {payee} [money|] " + _P,
        "<count> [have i|did i] [transacted|transferred|sent money] " + _P,
        "[how many|number of] times {payee} [got|was paid|received money] " + _P,
        "<count> [did|do] i [visit|shop at|go to] {payee} " + _P,
        "<count> [payments|transactions] [have i made|did i make] to {payee} " + _P,
        "nimelipa {payee} mara ngapi " + _P,
        "nimetuma {payee} mara ngapi " + _P,
        "<count> <payments> [for|on] {category} " + _P,
        "<count> <payments> [were|are] [for|on] {category} " + _P,
        "<count> [times|transactions] [using|on|through] {account} " + _P,
        "in [how many|how much] [payments|transactions] did i [pay|send] {payee}",
        "<count> <payments> [{amount}|] [to|at] {payee} " + _P,
    ],
    "top_categories": [
        "[which|what] [categories|areas|things] [take|eat|swallow] [most of|] my <money> " + _P,
        "what [do i|did i] <spend> [the bulk|most|a lot] of my <money> on " + _P,
        "where [is|was] my <money> going " + _P,
        "where [does|did] [all|most of] [my|the] <money> <go> " + _P,
        "<cash> yangu [inaenda|ilienda|inaishia] wapi " + _P,
        "[split|breakdown] of [my|] <spending> [by|per] category " + _P,
        "[category|categories] breakdown " + _P,
        "what [are|were] my [biggest|largest|main] [spending categories|expense areas] " + _P,
        "rank my <categories> [by spending|by cost|] " + _P,
        "what [takes|took] [up|] [most|the biggest part] of my [budget|<money>] " + _P,
        "which [area|category] [do i|did i] <spend> [the|] most [on|in] " + _P,
        "[top|main] [things|areas] [i spend on|my money goes to] " + _P,
        "what am i [blowing|burning|using] [most|my <money>] on " + _P,
        "spending [per|by] category " + _P,
    ],
    "out_of_scope": [
        "[what's|what is] [the|] [weather|time|news] [like|] [today|now|]",
        "<should> [buy|sell|get] [a car|land|a plot|shares|crypto|safaricom shares]",
        "[how|where] [can|do] i [get a job|find work|apply for a loan|open a bank account]",
        "[which|what] [is the best|is a good] [phone|bank|sacco|school|car] [to buy|]",
        "[do you|can you] [know|tell me] [a joke|a story|a riddle|about <topic>]",
        "[write|compose] [a poem|a letter|a story|an essay] [for me|]",
        "[who|what] is [the president|ruto|raila|the governor] [of kenya|]",
        "[what|who] [won|is winning] [the game|the match|the league|the election] [yesterday|]",
        "[help me|can you] [with|do] my [homework|assignment|cv]",
        "[how are|what are] [interest rates|treasury bills|mmf rates] [now|today|]",
        "[i need|give me] [advice|tips] on [love|life|my career|investing]",
        "[can|should] i [take|borrow] a loan from [tala|branch|my sacco|the bank]",
        "[tell|teach] me [how to|] [cook|dance|drive|code]",
        "[what|which] [movie|song|show] should i watch",
        "[you are|you're] [stupid|useless|funny|smart|cool]",
        "[what is|how much is] [the|a] [dollar|pound|euro] [in shillings|today|rate]",
        "[nipe|niambie] [story|joke|habari za <topic>]",
        "[can you|please] [call|text|message] [my mum|my boss|someone]",
        "[how old|where] are you",
        "[tell me|what about] [the|] [price of|cost of] [fuel|unga|bread] in kenya",
    ],
    "lent": [
        "namdai {payee} [pesa|<cash>|] ngapi",
        "{payee} [ananiwia|anadaiwa na mimi] [pesa|<cash>] ngapi",
        "<much> [did|have] i lend {payee} " + _P,
        "<much> [money|<cash>] [is|does] {payee} [still|] [owe|owing|owes] me",
        "has {payee} [refunded|paid back|returned] [the loan|my <cash>|what i lent]",
        "[list|show] [everyone|people|friends] who [borrowed|took] [money|<cash>] from me",
        "who [borrowed|took] [money|<cash>] from me [and hasn't paid|]",
        "<much> [money|<cash>] [have i|did i] [lent|loaned|given out] to friends " + _P,
        "[personal|friends'|family] loans [i gave|owed to me]",
        "[total|sum] [i've|i have] lent [out|to people|]",
        "[is|does] {payee} [still|] [owe|owes] me [anything|money]",
        "<much> do my [friends|relatives|people] <owes_me>",
        "[when|did] {payee} [pay|paid] back [the money|my loan|]",
        "{payee} [ameregesha|amerudisha] [pesa|<cash>] [yangu|]",
    ],
    "who_owes_me": [
        "which <customers> [are|is] [still|] owing [me|the shop|]",
        "[list|show] [all|] [the|] <customers> [on credit|with debts|who owe]",
        "<much> [credit|deni] [is|have i] [out|given] [to customers|to clients|]",
        "who [bought|took] [goods|stock|{item}] [on credit|kwa deni] [and hasn't paid|]",
        "wateja [ambao hawajalipa|wasiolipa]",
        "nani [ana|bado ana] deni [ya duka|ya {item}|]",
        "[outstanding|unpaid] [customer|client] [balances|debts|accounts]",
        "[total|sum] [owed|owing] by <customers>",
        "<much> [money|<cash>] [is|are] <customers> [owing|holding]",
        "<customers> [who|that] [still|] owe [the shop|the business|<shop>|me]",
        "who [hasn't|has not|haven't] [cleared|paid] [their|his|her] [deni|debt|credit|tab]",
        "[credit|deni] [book|list|ledger] [for the shop|]",
        "[which|what] [clients|wateja] [have|haven't] [cleared|paid] [their|] [credit|deni]",
    ],
    "balance": [
        "<much> [money|<cash>] [is|remains|imebaki] [on|in] {account} [now|]",
        "[what's|show] [my|the] {account} [account|wallet|] [balance|bal|salio]",
        "<much> [have i got|is left|do i have] [on|in] {account}",
        "{account} [iko na|ina] <cash> ngapi",
        "[check|see] [how much is|what's] [on|in] [my|] {account}",
        "[balance|bal] [ya|of|on] {account} [please|]",
    ],
    "sales": [
        "<much> {item} [have i|did we|did i] [sold|moved] " + _P,
        "<much> [money|<cash>] [did|has] {item} [brought in|made] " + _P,
        "[sales|mauzo] ya {item} " + _P,
        "<sales> [through|on|via] {account} [for {item}|] " + _P,
        "<much> did <shop> [sell|make] [on|via|through] {account} " + _P,
        "[daily|weekly|monthly|] <sales> [total|figure|] " + _P,
        "how [is|was] {item} [selling|doing] " + _P,
    ],
    "low_stock": [
        "[do i|we] [still|] have {item} [in stock|left|]",
        "<much> {item} [stock|] [remains|is remaining|is left] [in the shop|]",
        "{item} [imeisha|inaisha|iko] [kweli|]",
        "[when|will] [does|] {item} run out",
        "[check|show] [stock|inventory] [for|of] {item}",
        "which <stock> [should|do] i [restock|buy] [first|soon|today]",
    ],
    "refine": [
        "[what about|and|how about] [on|from|via] {account} [then|]",
        "[and|only] [the|those] {amount} [ones|payments|]",
        "[and|what about] {item} [then|too|]",
        "[and|but] [only|just] [from|via|on] {account}",
        "[same|the same] [but|] for {payee}",
        "[same|the same] [but|] for {category}",
        "[and|now] [for|at] {payee} {period}",
        "{payee} [only|then|pia]",
        "{category} [pia|too|also]",
        "na {category}",
        "na {payee} je",
        "na {period} je",
    ],
}
for _intent, _extra in _EXTRA_TEMPLATES.items():
    TEMPLATES[_intent].extend(_extra)

_cue_templates()

# The templates above are version 1 of the generator: ml's held-out folds and their scores are
# defined on them, so they keep their ids and fold assignment (see :func:`folds`).
V1_TEMPLATE_COUNTS: dict[str, int] = {intent: len(TEMPLATES[intent]) for intent in ASK_INTENTS}

# Version 2: the phrasings the first reader got wrong in the app, with invented names from
# :mod:`akili.ask.names` ({name} is a made-up payee, {bank} a bank, wallet or Sacco named as the
# payee). Counts and sums for one payee, "how much to X", money in from someone, money in,
# "<account> balance", "how often", and questions that only look like them.
LEXICON["paid_to"] = [
    "pay",
    "paid",
    "send money to",
    "sent money to",
    "buy from",
    "bought from",
    "shop at",
    "shopped at",
    "go to",
    "transact with",
    "lipa",
]
LEXICON["sent_me"] = [
    "sent me",
    "send me",
    "paid me",
    "pay me",
    "given me",
    "give me",
    "transferred to me",
    "deposited for me",
    "refunded me",
]
LEXICON["got_from"] = [
    "got from",
    "get from",
    "received from",
    "receive from",
    "gotten from",
    "was paid by",
    "was sent by",
    "collected from",
]
LEXICON["came_in"] = [
    "came in",
    "has come in",
    "come in",
    "came into my accounts",
    "landed",
    "was received",
    "did i receive",
    "have i received",
    "did i get",
    "did i earn",
    "did i make",
    "entered",
]
LEXICON["how_often"] = [
    "how often",
    "how many times",
    "how frequently",
    "number of times",
    "how many payments",
    "the number of payments",
    "number of payments",
    "how many transactions",
    "mara ngapi",
    "count of payments",
]

_HARD_TEMPLATES: dict[str, list[str]] = {
    "count_payments": [
        "<how_often> [did i|do i|have i] <paid_to> {name} " + _P,
        "<how_often> [did i|do i|have i] <paid_to> {payee} " + _P,
        "times i [paid|sent money to|bought from|went to] {name} " + _P,
        "[number of|no of|count of|total] [payments|transactions|purchases|times] [to|at|with|for] "
        "{name} " + _P,
        "<how_often> [to|at] {name} " + _P,
        "[how many|the number of] [payments|transactions|purchases] [went|have gone|did i make] "
        "[to|at] {name} " + _P,
        "<just_say> mara ngapi [nimelipa|nimetuma|nililipa|nimenunua kwa] {name} " + _P,
        "{name} [how many times|how often|mara ngapi] " + _P,
        "<how_often> [have i|did i] used {name} " + _P,
        "<count> [payments|transactions] [did i make|have i made] [on|from|using] {account} " + _P,
    ],
    "spent": [
        "<much> [did i|have i|do i] [pay|paid|give|gave] {name} " + _P,
        "<much> [did i|have i] [spend|spent|used] [at|with] {name} " + _P,
        "<much> [to|at] {name} " + _P,
        "<much> [went|has gone|goes] [to|at] {name} " + _P,
        "[total|sum] [paid|payments|spent|spend] [to|at] {name} " + _P,
        "<much> <cash> [did i|have i] [pay|paid|put] [to|into|at] {name} " + _P,
        "<much> [did i|have i] paid {name} [on|via|from|using] {account} " + _P,
        "nimelipa {name} [pesa|<cash>|] ngapi " + _P,
        "what [did|have] i [paid|pay|spent at] {name} " + _P,
        "{name} [total|amount|payments] " + _P,
        "{name} [payments|purchases|bills|spending] [total|so far|] " + _P,
        "<much> [money|<cash>|] [did|has] {name} [taken|get|got] from me " + _P,
        "[payments|money|<cash>] [from|on|via] {account} [to|at] {name} " + _P,
        "<much> [did i|have i] [pay|paid|spend|spent] [at|to] {name} [so far|altogether|in total|"
        "now|this time|already]",
        "[what|how much] [has|did] {name} cost me [so far|altogether|in total|] " + _P,
        "<much> [from|via|on|using] {account} [to|at] {name} " + _P,
    ],
    "received_from": [
        "<much> [did|has|have] {name} [send|sent|pay|paid|give|given] me " + _P,
        "<much> [did|has] {bank} [send|sent|pay|paid] me " + _P,
        "{name} <sent_me> [how much|<much>|] " + _P,
        "{bank} <sent_me> [how much|<much>|] " + _P,
        "did {name} [send|pay|give] me [anything|money|pesa|<cash>|] " + _P,
        "[has|did] {name} [sent|paid|send|pay] [me|] [yet|already|back|] " + _P,
        "what [has|did] {name} [send|sent|paid|pay|given] me " + _P,
        "<much> [did i|have i|i] <got_from> {name} " + _P,
        "what [did i|have i|i] <got_from> {name} " + _P,
        "[money|cash|payments|<cash>|deposits] [i got|received|got|that came|in] from {name} " + _P,
        "<much> [came in|came|has come] from {name} " + _P,
        "<much> [did i|have i] <got_from> {bank} " + _P,
        "[income|earnings|money in] from {name} " + _P,
        "{name} [ametuma|alituma|amenitumia|alinitumia|amenilipa] [pesa|<cash>|] ngapi " + _P,
        "[what|anything|everything|money] [that|] came [in|] from {name} " + _P,
        "[what|the money|the cash|the payments] {name} [sent|paid|gave|transferred] me " + _P,
        "[the amount|the cash|the payments|everything] [that|which] {name} [gave|transferred|"
        "deposited|has sent] [me|to me] " + _P,
        "{name} [paid|gave|transferred|deposited] [me|to me] [money|cash|something|] " + _P,
        "{bank} [paid|transferred|deposited] [me|to me|into my account] [money|cash|] " + _P,
        "[any|some] [money|<cash>|payment|payments] [from|by] {name} [yet|] " + _P,
        "[did|have] i [get|got|receive|received] [anything|money|something] from {name} " + _P,
        "<much> {name} <sent_me> " + _P,
    ],
    "received": [
        "<much> [money|<cash>|] <came_in> " + _P,
        "[money|cash|<cash>] in " + _P,
        "[my|total|] [income|earnings] " + _P,
        "[what|how much] [did i get|i got|i received|have i got] [in total|altogether|] " + _P,
        "<much> [did i|have i] [receive|received|get|got|gotten|earn|earned] " + _P,
        "[total|my|all] [money|<cash>|payments] [received|in|coming in] " + _P,
        "<much> [is|was] [coming|came] in " + _P,
        "<much> [income|money in] [did i have|have i had|was there] " + _P,
    ],
    "balance": [
        "{account} [balance|bal|salio]",
        "[my|the] {account} [balance|bal|salio] [now|today|please|]",
        "<much> [is|money is|is left|is there|do i have|have i got|remains] [on|in] [my|the|] "
        "{account} [account|wallet|]",
        "[how much is|what's|whats|what is] [on|in] [my|] {account} [now|]",
        "[balance|bal|salio] [on|in|of|for|ya] {account}",
        "what do i have [on|in] [my|] {account}",
        "<just_say> {account} [iko na|ina|imebaki na] [pesa|<cash>|] ngapi",
        "<much> [money|<cash>] [is|do i have] [on|in] {account}",
        "<much> [money|<cash>|] [is|do i have] [on|in] [my|] {account} [right now|now|today|"
        "currently|at the moment|please|pls]",
        "[show|tell|give] [me|] [my|the] {account} [balance|bal] [please|pls|now|]",
        "[show|check|see|view] {account} [balance|bal|salio] [please|now|]",
        "[my|] {cashacct} [balance|bal|salio] [please|now|]",
        "<much> [is|do i have|is left] [in|on] [my|] {cashacct} [now|right now|]",
        "what's [left|remaining] [in|on] {cashacct}",
    ],
    "out_of_scope": [
        "how often [does it rain|should i exercise|do matatus come|is the news on|do people "
        "save] [in nairobi|here|]",
        "how many times [has arsenal won|did kenya win|should i eat|can i retake the exam]",
        "[send me|tell me|give me] [a joke|a story|a song|a recipe]",
        "how much [is|does] [a ticket to mombasa|bread|a plot in ruiru|a bitcoin|a dollar|an "
        "iphone|a car] [cost|go for|today|]",
        "who sent me [this app|that message|this link|a virus]",
        "[what is|tell me] the [price|share price|value] of [kcb|equity|safaricom|co-op] shares",
        "[please|can you] [send|transfer|pay] [money|pesa|500 bob] [to my mum|for me|to my friend]",
        "how [many|much] [people|money] [live in kenya|does the government have|does a doctor "
        "earn]",
        "[what is|what's] the price of [unga|sugar|a phone|land|maize] [in kenya|today|now|]",
        "how much [do|does|will] [phones|cars|houses|school shoes|a goat] cost [in kenya|now|]",
        "[transfer|send|pay|move] [money|500|1000|cash] [to my brother|to someone|for me|to my "
        "landlord] [now|please|]",
        "can you [send|move|withdraw] money [for me|to my mum|from my account]",
        "how many times [did|has] [kenya|the team|my team] [win|won|lose|lost] [this year|]",
        "who [sent|gave] me [this phone|that link|these messages]",
        "how often do you think i should [save|budget|eat out|travel]",
        "how [regularly|frequently] should i [pay my bills|buy shares|top up|service my car]",
        "should i pay [rent|school fees|my loan] [weekly|monthly|early|in cash]",
        "[what is|what's] the [average|usual] rent in [lavington|ruaka|thika|nakuru|town]",
        "[rent|land|plot] prices [in|around] [ruaka|thika|nakuru|syokimau]",
        "how much [does|do] [rent|a plot|a bedsitter] go for in [ruaka|thika|nakuru|town]",
        "[send|give] [my brother|my dad|my friend|my wife] [some money|cash|2000] [today|for me|]",
        "[top up|buy] airtime for [my mum|me|my brother] [now|please|]",
    ],
    "help": [
        "what can i ask about [payments|income|my money|payees|balances|shops]",
        "how do i ask about [a payee|a shop|my income|balances|payments]",
        "[can i|how can i] ask [you|] about [money in|payments to someone|my accounts]",
        "what [questions|things] [about money|about my accounts|] can i ask",
        "what [can|do] you know about [money in|payments|my spending|balances]",
        "tell me what you can do with [payments|income|my accounts]",
    ],
}
for _intent, _extra in _HARD_TEMPLATES.items():
    TEMPLATES[_intent].extend(_extra)

V2_TEMPLATE_COUNTS: dict[str, int] = {intent: len(TEMPLATES[intent]) for intent in ASK_INTENTS}

# Version 3: the misses of the version 2 reader. Charges on an account followed by a period
# ("charges on KCB this month": the account ends where the period starts), "how much to X",
# "what did X cost me" (spending, not charges) and "has X paid me" (money in, not a loan repaid),
# with the look-alikes that keep their own intent ("what did charges cost me", "has X paid me
# back"), and bare month names as periods.
_V3_TEMPLATES: dict[str, list[str]] = {
    "fees": [
        "<much> [have i paid|did i pay|did i lose|have i lost] in <fees> [on|at|with|from|for|via] "
        "{account} " + _P,
        "<much> [did i pay|have i paid|did i lose] in <fees> [on|at|with|from] {account} {period}",
        "[how much|what] [did|have] i [pay|paid] [in|for] <fees> [on|at] {account} {period}",
        "<fees> [on|at|for|from|with] {account} " + _P,
        "[my|the|total|] <fees> [on|at|for] {account} {period}",
        "<much> [in|for|on] <fees> [on|at|from|with] {account} " + _P,
        "<much> [has|did] {account} [charge|charged] me [in fees|in charges|for transactions|] "
        + _P,
        "nimelipa [charges|gharama|fees|<fees>] ngapi [kwa|on|na] {account} " + _P,
        "[my|the|] {account} [bank charges|transaction charges|fees|charges|withdrawal fees] "
        "{period}",
        "what did [the|my|] {account} [charges|fees|transaction costs] cost me " + _P,
        "what did <fees> [on|at] {account} cost me " + _P,
        "what [did|have] <fees> cost me " + _P,
    ],
    "spent": [
        "<much> to {name} " + _P,
        "[how much|hw much|how mch|how much money|how much pesa] to {name} [pls|please|]",
        "[how much|hw much|how much money] to {payee} " + _P,
        "<much> [from|on|via] {account} to {name} " + _P,
        "what did {payee} cost me " + _P,
        "what did {name} cost me [so far|altogether|in total|this time|]",
        "what has {name} cost me " + _P,
        "[how much|what] did [shopping at|buying from|going to|paying] {name} cost me " + _P,
        "how much did {name} cost me " + _P,
        "what did {category} [at|from] {name} cost me " + _P,
        "{name} cost me [how much|<much>] " + _P,
    ],
    "received_from": [
        "has {name} paid me " + _P,
        "has {name} [paid|sent] me [yet|already|] " + _P,
        "has {payee} paid me [yet|already|anything|] " + _P,
        "[has|did] {name} [pay|paid] me [anything|money|<cash>|] " + _P,
        "{name} [amenilipa|ameshanilipa|amenitumia pesa|alinilipa] " + _P,
        "has {bank} paid me " + _P,
        "[did|has] {name} pay me [for the goods|for the work|for the job|] " + _P,
    ],
    "lent": [
        "has {name} paid [me|] back [the loan|my money|what i lent|]",
        "has {name} [repaid|refunded] [me|my loan|the money i lent]",
        "[did|has] {name} [pay|paid] [me|] back " + _P,
        "{name} [amerudisha|ameregesha] [pesa|<cash>|deni] [yangu|]",
    ],
    # A bare month is a period too: "what about September", "compare September with October".
    "refine": [
        "[what about|how about|and|and for] {month} [then|]",
    ],
    "compare": [
        "<compare> {month} [with|and|to|against|vs] {month2}",
        "[how does|how did] {month} compare [with|to] {month2}",
    ],
}
for _intent, _extra in _V3_TEMPLATES.items():
    TEMPLATES[_intent].extend(_extra)

V3_TEMPLATE_COUNTS: dict[str, int] = {intent: len(TEMPLATES[intent]) for intent in ASK_INTENTS}

# Version 4: the patterns the PyTorch v3 reader was weakest on. Money in from a name given in
# few words ("received from X", "got from X", "was paid by X"), with sends to a name as the
# look-alike; a payee inside compare, trend, biggest-payment, debt, bill and income-source
# questions ("compare my spending at X with last month"); names that are one word ({short}) or
# whose words include plain ones ({long}: "Example Corner",
# "Golden Butchery"), so they come out whole; and amounts and amount filters written many ways
# ({amt}: "over 1000", "ksh 2,500 and above", "at least 1.5k"; {sum}: "2,500", "kes 1,200"),
# next to a period or a name.
LEXICON["got_in"] = [
    "received",
    "got",
    "recieved",
    "gotten",
    "collected",
    "was paid",
    "got paid",
    "was sent",
    "pokea",
]
LEXICON["paid_by"] = ["paid by", "sent by", "paid from", "sent from", "transferred from"]
LEXICON["sent_out"] = ["sent", "paid", "transferred", "sent money", "gave", "forwarded"]
LEXICON["vs"] = ["with", "to", "against", "and", "vs", "versus"]
LEXICON["few"] = ["2", "3", "5", "10", "20", "three", "five", "ten"]
_V4_TEMPLATES: dict[str, list[str]] = {
    # Money paid back is a loan repaid, whatever the name ("did X pay me back").
    "lent": [
        "[did|has] {short} [return|returned|refund|refunded|give back|given back] [my money|the "
        "money|the loan|what i lent]",
        "[did|has] [{short}|{long}] [paid|sent|brought] [the|my] [money|loan|cash] back [yet|"
        "already|]",
        "[have i got|did i get|have i received] [my money|the loan|my cash] back from {short} "
        "[yet|]",
        "is {short} [done|finished|through] paying [me|] back",
    ],
    "received_from": [
        "<got_in> from {short} " + _P,
        "<got_in> from {long} " + _P,
        "<got_in> from {name} [so far|altogether|in total|]",
        "[i|] [was|got] <paid_by> {short} " + _P,
        "[what|how much|everything|all] [i|i've|i have] <got_in> from {short} " + _P,
        "[what|how much|everything|all] [i|i've|i have] <got_in> from {long} " + _P,
        "from {short} [how much|what] [did i get|came in|did i receive|have i received] " + _P,
        "{period} <got_in> from [{short}|{long}]",
        "[incoming|received] [money|payments|transfers|deposits|<cash>] from [{short}|{long}] "
        + _P,
        "[any|all|the] [transfers|deposits|payments] [in|that came in] from {long} " + _P,
        "[received|got] [from|by] {short} [how much|what|<much>] " + _P,
        "[money|payments] {short} [sent|paid|transferred] [to me|me|my way] " + _P,
    ],
    # Look-alikes: money out to the same kinds of names.
    "sent_to": [
        "<sent_out> to {short} " + _P,
        "<sent_out> to {long} [so far|altogether|in total|] " + _P,
        "[what|how much|everything|all] [i|i've|i have] <sent_out> to {short} " + _P,
        "{period} <sent_out> to [{short}|{long}]",
        "[outgoing|sent] [money|transfers|payments|<cash>] to [{short}|{long}] " + _P,
        "to {short} [how much|what] [did i send|went out|have i sent] " + _P,
    ],
    "compare": [
        "<compare> [my|] <spending> [at|with] {long} <vs> {period}",
        "<compare> [my|] <spending> [at|with] {short} <vs> {period}",
        "<compare> [my|] [payments|<spending>|what i spent] [at|to] {name} {period} <vs> {period2}",
        "[how does|how did] my <spending> [at|with] [{long}|{short}] {period} compare [with|to] "
        "{period2}",
        "did i <spend> <more> at [{long}|{short}] {period} <than> {period2}",
        "[{long}|{short}] {period} <than> {period2}",
        "<compare> [{long}|{short}] [spending|payments|] <vs> {period}",
        "[is|was] my <spending> at {long} <more> {period} <than> {period2}",
        "<compare> [my|] {category} [<spending>|] at {long} <vs> {period}",
    ],
    "trend": [
        "how has my <spending> [at|with] [{long}|{short}] <changed> " + _P,
        "[is|has] my <spending> at {long} [<change>|<changed>] " + _P,
        "[how are|how have] [my|] [payments|<spending>] [to|at] {long} [<change>|<changed>] "
        "[lately|over time|]",
        "{long} [spending|payments] <trend> " + _P,
        "[am i|have i been] spending more at {short} [lately|these days|recently|]",
        "<show> [me|] [the|my] <trend> [for|of|at] [{long}|{short}] " + _P,
        "has [what i pay|my spending] [at|to] {long} <changed> " + _P,
    ],
    "biggest_payment": [
        "what was my <biggest> <payment> [to|at] {long} " + _P,
        "<biggest> <payment> [to|at|with] [{short}|{long}] " + _P,
        "[my|the] <biggest> [payment|transfer|purchase] [i made|i sent|ever] [to|at] {short} " + _P,
        "<biggest> <payment> {amt} " + _P,
        # Small numbers that are counts, not amounts.
        "[my|the] [top|] <few> <biggest> [payments|purchases|transactions] " + _P,
    ],
    "debts": [
        "<much> [do i|do i still] <owe> [to|] [{long}|{short}]",
        "[what is|what's] my [debt|loan|balance] [with|at|to] {long}",
        "[am i|do i] [still|] owing {short} [anything|money|]",
    ],
    "bills_due": [
        "[when is|is] [my|the] {long} [bill|payment] [due|coming up]",
        "<bills> [for|from|to] [{long}|{short}] <due> " + _P,
        "[any|is there a] [bill|payment] [from|for] {short} <due> " + _P,
    ],
    "income_sources": [
        "[how much of my|what share of my] <income> [comes|came] from {long} " + _P,
        "[is|was] [{long}|{short}] [my|a] [main|biggest|top] [source of income|income source|"
        "payer] " + _P,
        "[my|] [income|money|earnings] sources [besides|other than|apart from|like] {long} " + _P,
        "<income> [sources|source|payers] [from|including|with] [{long}|{short}] " + _P,
    ],
    "count_payments": [
        "<how_often> [did i|have i] [pay|spend|send] {amt} " + _P,
        "<count> <payments> {amt} {period}",
        "[how many|number of] [payments|purchases|transfers] [of|worth|for] {amt} [at|to] "
        "[{long}|{short}] " + _P,
        "{period} [how many|count the] [payments|transactions] [went|were] {amt}",
        "[payments|transactions|purchases] {amt} [how many|count] " + _P,
        "how many [visits|trips|purchases|payments] to [{long}|{short}] " + _P,
        "<count> [did i|have i] [buy|pay for|spend on] [stuff|things|items] {amt} " + _P,
        "[did i|have i] [pay|paid|visit|visited] {long} [more than|over|at least] <few> times "
        + _P,
    ],
    "spent": [
        "[total|sum] of [all|] [my|] [payments|purchases|transactions] {amt} " + _P,
        "<much> [went|was spent] [in|on] [payments|purchases] [worth|of|that were] {amt} " + _P,
        "{period} <much> did i <spend> [in|on] [payments|purchases] {amt}",
        "[my|] [spending|purchases] {amt} [at {long}|on {category}|] " + _P,
        "<much> [did i|have i] [spent|paid] at [{long}|{short}] [in|on] [payments|purchases] {amt}",
        "<much> [did i|have i] [spent|paid|used] on [items|goods|shopping|anything|orders] {amt} "
        + _P,
        "[what|how much] [went|did i pay] [for|on] [items|goods|orders|purchases] {amt} " + _P,
        "<much> [did i|have i] <spend> on [stuff|things|small things|odd things|everything|bits] "
        "{amt} " + _P,
        # A category where a name could stand: not a payee, and not charges.
        "[what|how much] did [my|the|] {category} [bills|payments|purchases|] cost me " + _P,
        "<much> [did|has] [my|] {category} cost [me|us] [in total|so far|altogether|] " + _P,
    ],
    "received": [
        "<much> came in [as|in] [payments|deposits|transfers] {amt} " + _P,
        "[deposits|payments i received|money in] {amt} " + _P,
        "{period} <much> [came in|did i receive] [in|as] [deposits|payments] {amt}",
    ],
    "top_payees": [
        "[who|which payees|which shops] [got|received] [payments|transfers] {amt} [from me|] " + _P,
        "<who> [did i|do i] [pay|send] {amt} [to|] " + _P,
    ],
    "safe_to_spend": [
        "[can i|could i|is it okay to] [spend|pay|use] {sum} [at {long}|on {category}|] " + _P,
        "[will|would] {sum} [be|leave me] [okay|fine|short] " + _P,
    ],
    "refine": [
        "[filter|limit it|narrow it] to [payments|transactions|ones] {amt}",
        "[now|same but|but] [only|just] {amt}",
        "{amt} [only|pia|too]",
        "[now|ok] [do|show] [{long}|{short}] instead",
        "[what about|how about] [{long}|{short}] [instead|then|too]",
    ],
}
for _intent, _extra in _V4_TEMPLATES.items():
    TEMPLATES[_intent].extend(_extra)

# Small numbers that count things ("my 5 biggest payments", "top 3 shops"): never amounts.
_V4_COUNTS: dict[str, list[str]] = {
    "top_payees": [
        "[show|list|give me] [my|the] <few> [biggest|main|top] [payees|shops|people i pay] " + _P,
    ],
    "top_categories": [
        "[show|list|give me] [my|the] <few> [biggest|main|top] [categories|spending areas] " + _P,
    ],
    "top_items": ["[my|the] <few> [best selling|top selling|best] [items|products|goods] " + _P],
}
for _intent, _extra in _V4_COUNTS.items():
    TEMPLATES[_intent].extend(_extra)

# Amounts for {amt} and {sum}: a value from AMOUNT_VALUES written one of AMOUNT_FORMS ways,
# and for {amt} a filter around it (a word before, a word after, or a range).
AMOUNT_VALUES = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 750, 800, 1000, 1200, 1500, 2000, 2500, 3000,
    3500, 4000, 4500, 5000, 6000, 7500, 8000, 10000, 12000, 15000, 20000, 25000, 30000, 40000,
    50000, 75000, 100000,
]  # fmt: skip
AMOUNT_BEFORE = [
    "over", "above", "more than", "greater than", "bigger than", "exceeding", "at least",
    "under", "below", "less than", "at most", "not more than", "up to", "zaidi ya", "chini ya",
    "juu ya",
]  # fmt: skip
AMOUNT_AFTER = ["and above", "or more", "and below", "or less", "plus", "and over", "na zaidi"]
AMOUNT_RANGE = [("between", "and"), ("from", "to"), ("", "to"), ("", "-")]
AMOUNT_CURRENCY = ["ksh", "kes", "Ksh", "KES", "sh", "ksh.", "Kshs", "kshs"]
AMOUNT_UNITS = ["bob", "shillings", "/=", "/-", "bob only"]


def _money_text(g: _Gen, value: int, bare: bool = True) -> str:
    """``value`` as people type it: "1000", "1,000", "1k", "1.5K", "ksh 1,000", "1000 bob".
    Without ``bare``, never bare digits, which read as a count or a year ("top 10", "2025")."""
    plain, comma = str(value), f"{value:,}"
    forms = [plain, comma, comma] if bare or value < 1000 else [comma, comma]
    if value >= 1000 and value % 500 == 0:
        k = f"{value / 1000:g}"
        forms += [f"{k}k", f"{k}k", f"{k}K", f"{k} k"]
    number = g.pick(forms)
    if not bare and number.isdigit():
        # A bare number alone reads as a count or a year ("top 10", "2025"), not money.
        currency = g.pick(AMOUNT_CURRENCY)
        return f"{currency} {number}"
    r = g.rng.random()
    if r < 0.3:
        currency = g.pick(AMOUNT_CURRENCY)
        return f"{currency}{number}" if g.chance(0.3) else f"{currency} {number}"
    if r < 0.42 and not number.lower().endswith("k"):
        unit = g.pick(AMOUNT_UNITS)
        return f"{number}{unit}" if unit[0] == "/" else f"{number} {unit}"
    return number


def _amount(g: _Gen, bound: bool) -> str:
    """An amount ({sum}) or, with ``bound``, an amount filter ({amt})."""
    value = g.pick(AMOUNT_VALUES)
    if not bound:
        return _money_text(g, value, bare=False)
    r = g.rng.random()
    if r < 0.6:
        return f"{g.pick(AMOUNT_BEFORE)} {_money_text(g, value)}"
    if r < 0.8:
        return f"{_money_text(g, value)} {g.pick(AMOUNT_AFTER)}"
    upper = g.pick([v for v in AMOUNT_VALUES if v > value] or [value * 2])
    start, middle = AMOUNT_RANGE[int(g.rng.integers(len(AMOUNT_RANGE)))]
    text = f"{_money_text(g, value, bool(start))} {middle} {_money_text(g, upper, bool(start))}"
    return f"{start} {text}" if start else text


_PREFIXES = [
    "hey",
    "hi",
    "pls",
    "please",
    "eh",
    "sasa",
    "so",
    "ok",
    "niambie",
    "tell me",
    "quick one,",
    "akili",
    "yo",
    "boss",
    "hey akili,",
]
_SUFFIXES = ["please", "pls", "bana", "boss", "tafadhali", "asap", "kindly"]
_FILLER = [
    "the",
    "my",
    "please",
    "now",
    "really",
    "exactly",
    "actually",
    "so",
    "like",
    "ati",
    "eh",
    "kwani",
    "um",
    "hmm",
    "pls",
    "total",
]
_NO_PREFIX = {"hi", "thanks", "bye"}
_CHOICE = re.compile(r"\[([^\]]*)\]")
_HOLE = re.compile(r"\{([a-z0-9]+)\}")
_WORDS = re.compile(r"<([a-z_]+)>")
_GLUE = {"?", ",", ".", "!"}


class _Gen:
    def __init__(self, rng: np.random.Generator, pool: names.NamePool | None = None) -> None:
        self.rng = rng
        self.pool = pool
        """With a pool, payee and account holes often take its invented names instead of the
        short lists above ("wide" names). Without one, version 1 questions come out exactly as
        they always did."""

    def pick(self, options: Sequence[T]) -> T:
        return options[int(self.rng.integers(len(options)))]

    def chance(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def typo(self, word: str) -> str:
        if len(word) < 4 or not word.isalpha():
            return word
        i = int(self.rng.integers(1, len(word) - 1))
        kind = int(self.rng.integers(3))
        if kind == 0:
            return word[:i] + word[i + 1 :]
        if kind == 1:
            return word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]
        return word[:i] + word[i] + word[i:]


def _choose(g: _Gen, template: str) -> str:
    text = _CHOICE.sub(lambda m: g.pick(m.group(1).split("|")), template)
    return _WORDS.sub(lambda m: g.pick(LEXICON[m.group(1)]), text)


def _pieces(g: _Gen, template: str) -> list[tuple[str, int]]:
    """Literal words (label 0) and filled holes, in order."""
    text = _choose(g, template)
    out: list[tuple[str, int]] = []
    pos = 0
    for m in _HOLE.finditer(text):
        out += [(w, 0) for w in text[pos : m.start()].split()]
        out.append(_fill(g, m.group(1)))
        pos = m.end()
    out += [(w, 0) for w in text[pos:].split()]
    return out


def _fill(g: _Gen, hole: str) -> tuple[str, int]:  # noqa: PLR0911
    pool = g.pool or names.TRAIN
    if hole == "name":
        return pool.name(g.rng), PAYEE
    if hole == "bank":
        return (pool.sacco(g.rng) if g.chance(0.15) else g.pick(BANK_PAYEES)), PAYEE
    if hole == "short":
        return pool.short_name(g.rng), PAYEE
    if hole == "long":
        return pool.long_name(g.rng), PAYEE
    if hole in ("amt", "sum"):
        return _amount(g, hole == "amt"), AMOUNT
    if hole == "cashacct":
        return g.pick(["cash", "Cash", "CASH", "my cash"]), ACCOUNT
    if hole == "paccount":
        return (pool.sacco(g.rng) if g.chance(0.15) else g.pick(names.PROBE_ACCOUNTS)), ACCOUNT
    if g.pool is not None and hole == "payee" and g.chance(0.65):
        return pool.name(g.rng), PAYEE
    if g.pool is not None and hole == "account" and g.chance(0.6):
        if g.chance(0.7):
            return g.pick(names.PROBE_ACCOUNTS), ACCOUNT
        return (pool.sacco(g.rng) if g.chance(0.2) else g.pick(WIDE_ACCOUNTS)), ACCOUNT
    values, label = _VALUES[hole]
    return g.pick(values), label


def _roughen(g: _Gen, pieces: list[tuple[str, int]], intent: str) -> list[tuple[str, int]]:
    casing = g.rng.random()
    out: list[tuple[str, int]] = []
    if intent not in _NO_PREFIX and g.chance(0.12):
        out.append((g.pick(_PREFIXES), 0))
    for piece, label in pieces:
        if label == 0 and len(pieces) > 2 and g.chance(0.04):
            continue
        if label == 0 and g.chance(0.04):
            out.append((g.pick(_FILLER), 0))
        words = piece.split(" ")
        if g.chance(0.05 if label == 0 else 0.02):
            j = int(g.rng.integers(len(words)))
            words[j] = g.typo(words[j])
        text = " ".join(words)
        if casing < 0.55:
            text = text.lower()
        elif casing < 0.6:
            text = text.upper()
        elif casing < 0.7 and label == 0:
            text = " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
        out.append((text, label))
    if intent not in _NO_PREFIX and g.chance(0.06):
        out.append((g.pick(_SUFFIXES), 0))
    if casing >= 0.55 and out and out[0][0][:1].islower():
        out[0] = (out[0][0][0].upper() + out[0][0][1:], out[0][1])
    end = g.rng.random()
    if end < 0.4:
        out.append(("?", 0))
    elif end < 0.44:
        out.append(("??", 0))
    elif end < 0.47 and intent in {"hi", "thanks", "bye"}:
        out.append(("!", 0))
    return out


def assemble(pieces: list[tuple[str, int]]) -> tuple[str, list[int]]:
    """Join pieces with single spaces (punctuation glued to the word before it) and label each
    token with the piece it starts in."""
    text = ""
    spans: list[tuple[int, int, int]] = []
    for piece, label in pieces:
        if not piece:
            continue
        glue = "" if not text or piece[0] in _GLUE else " "
        text += glue
        spans.append((len(text), len(text) + len(piece), label))
        text += piece
    labels = [next((lab for s, e, lab in spans if s <= t.start < e), 0) for t in tokenise(text)]
    return text, labels


def template_ids() -> list[str]:
    return [f"{intent}:{i}" for intent in ASK_INTENTS for i in range(len(TEMPLATES[intent]))]


def _is_v4(tid: str) -> bool:
    intent, i = tid.split(":")
    return int(i) >= V3_TEMPLATE_COUNTS[intent]


def generate(
    n: int,
    seed: int = 0,
    templates: set[str] | None = None,
    pool: names.NamePool | None = None,
) -> list[Question]:
    """``n`` questions, intents uniform, templates uniform within an intent. With
    ``templates``, only those template ids are used. With ``pool``, payees and accounts are
    often invented names from it (training uses :data:`akili.ask.names.TRAIN`)."""
    g = _Gen(np.random.default_rng(seed), pool)
    pools = {
        intent: [
            (f"{intent}:{i}", t)
            for i, t in enumerate(TEMPLATES[intent])
            if templates is None or f"{intent}:{i}" in templates
        ]
        for intent in ASK_INTENTS
    }
    intents = [i for i in ASK_INTENTS if pools[i]]
    # With invented names, version 2 and 3 templates (the hard phrasings) come up more often;
    # version 4 ones as often as version 1's, so they add phrasings without crowding out the rest.
    newer = {
        intent: [
            p
            for p in pools[intent]
            if V1_TEMPLATE_COUNTS[intent] <= int(p[0].split(":")[1]) < V3_TEMPLATE_COUNTS[intent]
        ]
        for intent in intents
    }
    older = {intent: [p for p in pools[intent] if not _is_v4(p[0])] for intent in intents}
    out = []
    for _ in range(n):
        intent = g.pick(intents)
        pool_of = pools[intent]
        if g.pool is not None and newer[intent] and g.chance(V2_SHARE):
            pool_of = newer[intent]
        tid, template = pool_of[int(g.rng.integers(len(pool_of)))]
        if older[intent] and _is_v4(tid) and not g.chance(V4_WEIGHT):
            tid, template = older[intent][int(g.rng.integers(len(older[intent])))]
        text, labels = assemble(_roughen(g, _pieces(g, template), intent))
        out.append(Question(text, labels, ASK_INTENTS.index(intent), tid))
    return out


def folds(k: int, seed: int = 0) -> list[set[str]]:
    """Split every intent's templates into ``k`` held-out sets, so each template is held out
    exactly once and every fold holds some of every intent out. Each version's templates fall
    in the same folds as before later versions were added (one permutation per version and
    intent), so their scores compare across versions."""
    rng = np.random.default_rng(seed)
    out: list[set[str]] = [set() for _ in range(k)]
    for intent in ASK_INTENTS:
        order = rng.permutation(V1_TEMPLATE_COUNTS[intent])
        for j, i in enumerate(order):
            out[j % k].add(f"{intent}:{int(i)}")
    for step, (counts, ends) in enumerate(
        [
            (V1_TEMPLATE_COUNTS, V2_TEMPLATE_COUNTS),
            (V2_TEMPLATE_COUNTS, V3_TEMPLATE_COUNTS),
            (V3_TEMPLATE_COUNTS, None),
        ]
    ):
        rng = np.random.default_rng(seed + 1 + step)
        for intent in ASK_INTENTS:
            first = counts[intent]
            last = ends[intent] if ends is not None else len(TEMPLATES[intent])
            order = rng.permutation(last - first)
            start = int(rng.integers(k))
            for j, i in enumerate(order):
                out[(start + j) % k].add(f"{intent}:{first + int(i)}")
    return out


def v1_template_ids() -> set[str]:
    return {f"{intent}:{i}" for intent in ASK_INTENTS for i in range(V1_TEMPLATE_COUNTS[intent])}


def v2_template_ids() -> set[str]:
    return {
        f"{intent}:{i}"
        for intent in ASK_INTENTS
        for i in range(V1_TEMPLATE_COUNTS[intent], V2_TEMPLATE_COUNTS[intent])
    }


def v3_template_ids() -> set[str]:
    return {
        f"{intent}:{i}"
        for intent in ASK_INTENTS
        for i in range(V2_TEMPLATE_COUNTS[intent], V3_TEMPLATE_COUNTS[intent])
    }


def v4_template_ids() -> set[str]:
    return {
        f"{intent}:{i}"
        for intent in ASK_INTENTS
        for i in range(V3_TEMPLATE_COUNTS[intent], len(TEMPLATES[intent]))
    }


def to_arrays(questions: list[Question]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inputs ``[n, 2 * MAX_TOKENS]`` (the message reader's encoding), slot labels
    ``[n, MAX_TOKENS]`` (-1 on padding) and intent labels ``[n]``."""
    x = np.array([encode(q.text) for q in questions], dtype=np.int32)
    y = np.full((len(questions), MAX_TOKENS), -1, dtype=np.int32)
    for i, q in enumerate(questions):
        labels = q.slots[:MAX_TOKENS]
        y[i, : len(labels)] = labels
    k = np.array([q.intent for q in questions], dtype=np.int32)
    return x, y, k

"""Made-up payee, merchant and account names for the ask reader's synthetic questions.

The first reader learned the few dozen payees in :mod:`akili.ask.questions` by heart and left
most other names untagged ("how many times did I pay Example Grocers" came back with no payee).
These pools are large and invented, so the reader has to tag a name from its place in the
question, not from the word itself: Kenyan-sounding words built from syllables, business words
("Hardware", "Butchery"), "Example"/"Sample"/"Demo" names and people's names, one to four words
long, in mixed case.

There are two pools. :data:`TRAIN` feeds training; :data:`HELD_OUT` only the hard held-out
set (:mod:`akili.ask.probes`). Its syllable words are its own, so no held-out name is one the
reader has seen. Its business words, prefixes and tails are the common ones training also uses
("Hardware", "Example", "Ltd") plus as many again that training never sees ("Emporium",
"Mock", "EA"), since real merchants share such words. :data:`HELD_ONLY_WORDS` are the words
only the held-out pool has; the probe also scores the names made of those alone. Nothing here
is a real person, business or message.
"""

from __future__ import annotations

import numpy as np

_ONSETS = [
    "b", "ch", "d", "g", "h", "j", "k", "l", "m", "mb", "n", "nd", "ng", "nj", "ny",
    "r", "s", "sh", "t", "w", "y", "z",
]  # fmt: skip
_VOWELS = ["a", "e", "i", "o", "u", "a", "i", "o"]

# Business words, split between the pools. "Grocers" is a training word, but the app's own
# test names (such as "Example Grocers") are never generated whole: see EXCLUDED.
_TRAIN_BUSINESS = [
    "Hardware", "Grocers", "Stores", "Supermarket", "Butchery", "Pharmacy", "Wines", "Enterprises",
    "Traders", "Agencies", "Motors", "Electronics", "Bakery", "Hotel", "Cafe", "Salon", "Academy",
    "Clinic", "Kiosk", "Mart", "Ventures", "Ltd", "Investments", "Garage", "Boutique", "Cyber",
    "Water", "Gas", "Dairy", "Fresh", "Movers", "Farm", "Lodge", "Spares", "Textiles", "Furniture",
    "Chemist", "Supplies", "Shop", "Butchers", "Hub", "Services", "Wholesalers", "Agrovet",
]  # fmt: skip
_HELD_BUSINESS = [
    "Emporium", "Outfitters", "Depot", "Foods", "Distributors", "Fashions", "Opticians",
    "Bookshop", "Studio", "Restaurant", "Pub", "Dispensary", "Logistics", "Builders", "Printers",
    "Barbers", "Collections", "Solutions", "Cereals", "Laundry", "Bistro", "Florist", "Tailors",
    "Imports",
]  # fmt: skip
_TRAIN_PREFIXES = ["Example", "Sample", "Demo"]
_HELD_PREFIXES = ["Test", "Fake", "Mock"]
_TITLES = ["Mama", "Baba", "Mzee", "Auntie", "Uncle", "Dr", "Madam", "Cucu"]
_TRAIN_TAILS = ["Ltd", "Limited", "& Sons", "Kenya", "Co"]
_HELD_TAILS = ["Group", "& Daughters", "EA", "Intl"]

# Banks, wallets and Saccos as the app names them. They are accounts ("my KCB balance") and,
# in money-in questions, payees too ("how much did KCB send me").
BANKS = [
    "M-Pesa", "Mpesa", "KCB", "Equity", "Equity Bank", "Co-op", "Co-op Bank", "Airtel Money",
    "Loop", "I&M", "I&M Bank", "Standard Chartered", "StanChart", "Cash", "NCBA", "Absa",
    "Family Bank", "DTB", "Stanbic", "M-Shwari", "KCB M-Pesa", "Fuliza", "T-Kash",
]  # fmt: skip

# The accounts the app names most, each as people type it.
PROBE_ACCOUNTS = [
    "M-Pesa", "mpesa", "MPESA", "m-pesa", "KCB", "kcb", "Equity", "equity", "Equity Bank",
    "Co-op", "co-op", "Co-operative Bank", "coop bank", "Airtel Money", "airtel money", "Loop",
    "loop", "I&M", "i&m", "I&M Bank", "Standard Chartered", "standard chartered", "StanChart",
    "Cash", "cash",
]  # fmt: skip

# The app's own test names (pesaclarity-app test/), kept out of both pools so the app's
# phrasings stay a held-out check.
EXCLUDED = {
    "example grocers", "zebra hardware", "example nowhere", "example sacco", "example lender",
    "example water", "example landlord", "acme", "zebras", "john otieno", "mama mboga",
}  # fmt: skip


def _words(rng: np.random.Generator, n: int, avoid: set[str]) -> list[str]:
    out: list[str] = []
    seen = set(avoid)
    while len(out) < n:
        w = ""
        for _ in range(int(rng.integers(2, 4))):
            w += _ONSETS[int(rng.integers(len(_ONSETS)))] + _VOWELS[int(rng.integers(8))]
        if rng.random() < 0.15:
            w += "n"
        if len(w) >= 4 and w not in seen:
            seen.add(w)
            out.append(w.capitalize())
    return out


# Name shapes: w a syllable word, b a business word, p a prefix ("Example"), t a title
# ("Mama"), x a tail ("Ltd"). "wb" is twice as likely as the others.
_FORMS = ["w", "ww", "wb", "wb", "pb", "pw", "tw", "pwb", "wbx", "pwbx", "wwb", "www"]


class NamePool:
    """Invented names drawn from one pool's words."""

    def __init__(
        self,
        words: list[str],
        business: list[str],
        prefixes: list[str],
        tails: list[str],
        own: set[str] | None = None,
    ) -> None:
        self.words = words
        self.business = business
        self.prefixes = prefixes
        self.tails = tails
        self.own = own
        """When set, every name has at least one of these words, so no other pool makes it."""

    def _base(self, rng: np.random.Generator) -> str:
        source = {"w": self.words, "b": self.business, "p": self.prefixes, "t": _TITLES}
        source["x"] = self.tails
        form = _FORMS[int(rng.integers(len(_FORMS)))]
        return " ".join(source[c][int(rng.integers(len(source[c])))] for c in form)

    def name(self, rng: np.random.Generator) -> str:
        """One name, one to four words, cased the way people type it."""
        while True:
            base = self._base(rng)
            lower = base.lower()
            if lower in EXCLUDED or len(base.split()) > 4:
                continue
            if self.own is None or any(w in self.own for w in lower.split()):
                break
        r = rng.random()
        if r < 0.55:
            return base
        if r < 0.82:
            return base.lower()
        if r < 0.9:
            return base.upper()
        return " ".join(w.lower() if rng.random() < 0.5 else w for w in base.split())

    def word(self, rng: np.random.Generator) -> str:
        """A one-word name ("Wekesa", "wekesa"): short names read like the one-word cues
        around them ("what did charges cost me"), so they get questions of their own."""
        w = self.words[int(rng.integers(len(self.words)))]
        r = rng.random()
        return w if r < 0.45 else w.lower() if r < 0.9 else w.upper()

    def sacco(self, rng: np.random.Generator) -> str:
        w = self.words[int(rng.integers(len(self.words)))]
        p = self.prefixes[int(rng.integers(len(self.prefixes)))]
        name = f"{w} Sacco" if rng.random() < 0.7 else f"{p} Sacco"
        return name.lower() if rng.random() < 0.4 else name


def _pools() -> tuple[NamePool, NamePool, set[str]]:
    train_words = _words(np.random.default_rng(17), 600, set())
    held_words = _words(np.random.default_rng(29), 250, {w.lower() for w in train_words})
    held_only = {
        w.lower()
        for group in (held_words, _HELD_BUSINESS, _HELD_PREFIXES, _HELD_TAILS)
        for phrase in group
        for w in phrase.split()
        if w.isalpha()
    }
    return (
        NamePool(train_words, _TRAIN_BUSINESS, _TRAIN_PREFIXES, _TRAIN_TAILS),
        NamePool(
            held_words,
            _HELD_BUSINESS + _TRAIN_BUSINESS,
            _HELD_PREFIXES + _TRAIN_PREFIXES,
            _HELD_TAILS + _TRAIN_TAILS,
            own=held_only,
        ),
        held_only,
    )


TRAIN, HELD_OUT, HELD_ONLY_WORDS = _pools()

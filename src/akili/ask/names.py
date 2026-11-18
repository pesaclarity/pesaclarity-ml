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

Version 4 adds two shapes to the training pool only (the held-out pool is unchanged):
:meth:`NamePool.short_name`, a single word (a syllable word or an invented word that starts
with a vowel), and :meth:`NamePool.long_name`, two to four words in which plain words
("Corner", "Golden", "Sunrise") stand beside prefixes, syllable words and business words, so a
name whose second word is an ordinary word still reads whole.
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
# Plain words that real merchants put in their names (version 4, training only). None is a word
# of the held-out pool, of the app's test names or of a date, amount or question word.
_TRAIN_PLAIN = [
    "Corner", "Point", "Junction", "Plaza", "Centre", "Express", "Quick", "Prime", "Royal",
    "Golden", "Sunrise", "Sunset", "Green", "Blue", "Star", "City", "Valley", "Hill", "Gate",
    "Bridge", "Rock", "River", "Lake", "Park", "Garden", "Palm", "Cedar", "Acacia", "Baobab",
    "Jacaranda", "Eagle", "Lion", "Simba", "Tembo", "Twiga", "Nyota", "Upendo", "Amani",
    "Baraka", "Neema", "Faraja", "Tumaini", "Uhuru", "Umoja", "Pamoja", "Bidii", "Fanaka",
    "Mwangaza", "Classic", "Modern", "Urban", "Metro", "Global", "United", "Pioneer",
    "Summit", "Crown", "Diamond", "Pearl", "Silver", "Ocean", "Coast", "Highway", "Square",
    "Alpha", "Omega", "Delta", "Nova", "Apex", "Zenith", "Horizon", "Harmony", "Liberty",
    "Victory", "Unity", "Elite", "Choice", "Village", "Island", "Meadow", "Orchard",
    "Spring", "Lighthouse", "Anchor", "Compass", "Falcon", "Kestrel", "Phoenix", "Rhino",
]  # fmt: skip
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


def _vowel_words(rng: np.random.Generator, n: int, avoid: set[str]) -> list[str]:
    """Invented brand-like words that start with a vowel ("Abeko", "Imaro"), four to six
    letters, which the syllable words (all starting with a consonant) never are."""
    out: list[str] = []
    seen = set(avoid)
    while len(out) < n:
        w = "aeiou"[int(rng.integers(5))]
        for _ in range(int(rng.integers(1, 3))):
            w += _ONSETS[int(rng.integers(len(_ONSETS)))] + _VOWELS[int(rng.integers(8))]
        if rng.random() < 0.2:
            w += "n"
        if 4 <= len(w) <= 6 and w not in seen:
            seen.add(w)
            out.append(w.capitalize())
    return out


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
# Words the questions themselves use that an invented vowel word could spell ("unga" is flour).
# A test checks that no version 4 name word is a word of a template, the lexicon or a value list.
QUESTION_WORDS = {"unga", "ebu", "aki", "ati", "ile", "uza", "ya"}
# Version 4's longer shapes, with c a plain word ("Corner"): "pc" is "Example Corner".
LONG_FORMS = ["pc", "pc", "cb", "wc", "cw", "cc", "pcb", "ccb", "wcb", "pwc", "tc", "cbx"]
# Version 4's one-word shapes: w a syllable word, v a vowel word. Plain words stay out of
# one-word names: alone, in lower case, they read like a category ("fuel", "rent").
SHORT_FORMS = ["w", "v", "v"]


class NamePool:
    """Invented names drawn from one pool's words."""

    def __init__(
        self,
        words: list[str],
        business: list[str],
        prefixes: list[str],
        tails: list[str],
        *,
        own: set[str] | None = None,
        plain: list[str] | None = None,
        vowel_words: list[str] | None = None,
    ) -> None:
        self.words = words
        self.business = business
        self.prefixes = prefixes
        self.tails = tails
        self.own = own
        """When set, every name has at least one of these words, so no other pool makes it."""
        self.plain = plain or []
        """Plain words for :meth:`long_name` and :meth:`short_name` (version 4)."""
        self.vowel_words = vowel_words or []

    def _base(self, rng: np.random.Generator, forms: list[str] = _FORMS) -> str:
        source = {"w": self.words, "b": self.business, "p": self.prefixes, "t": _TITLES}
        source["x"] = self.tails
        source["c"] = self.plain
        source["v"] = self.vowel_words
        form = forms[int(rng.integers(len(forms)))]
        return " ".join(source[c][int(rng.integers(len(source[c])))] for c in form)

    def name(self, rng: np.random.Generator) -> str:
        """One name, one to four words, cased the way people type it."""
        return self._name(rng, _FORMS)

    def long_name(self, rng: np.random.Generator) -> str:
        """A name of two to four words with a plain word in it ("Example Corner", "Golden
        Butchery"), cased as :meth:`name` (version 4; needs ``plain``)."""
        return self._name(rng, LONG_FORMS)

    def short_name(self, rng: np.random.Generator) -> str:
        """A one-word name (a syllable word or a vowel word), cased as
        :meth:`name` (version 4; needs ``plain`` and ``vowel_words``)."""
        return self._name(rng, SHORT_FORMS)

    def _name(self, rng: np.random.Generator, forms: list[str]) -> str:
        while True:
            base = self._base(rng, forms)
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
    avoid = {w for name in EXCLUDED for w in name.split()} | QUESTION_WORDS
    vowel_words = _vowel_words(np.random.default_rng(41), 120, avoid)
    return (
        NamePool(
            train_words,
            _TRAIN_BUSINESS,
            _TRAIN_PREFIXES,
            _TRAIN_TAILS,
            plain=_TRAIN_PLAIN,
            vowel_words=vowel_words,
        ),
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

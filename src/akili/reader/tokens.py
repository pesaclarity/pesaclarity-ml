"""Message tokens and the reader's input, shared by training and the app.

The app reproduces this exactly in Dart; ``docs/READER.md`` is the language-neutral spec and
``resources/reader_vectors.v1.json`` the conformance vectors. Everything is ASCII-only, so the
result never depends on locale or Unicode tables. Any change to :func:`encode` must bump
:data:`READER_SCHEMA_VERSION`.

Input: one int32 vector of length ``2 * MAX_TOKENS``: the token ids, then the shape ids. Id 0 is
padding in both halves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

READER_SCHEMA_VERSION = 1
MAX_TOKENS = 96
HASH_BUCKETS = 4096

# Numbers (with separators); words, which keep digits only when a letter follows them (so
# "SK29XY7" stays whole but "KES1,250.00" splits into "KES" and "1,250.00"); masks; then any
# single visible character. Spelled out so no engine applies Unicode semantics.
_TOKEN = re.compile(
    r"[0-9]+(?:[.,:/-][0-9]+)*"
    r"|[A-Za-z]+(?:[0-9]+[A-Za-z][A-Za-z0-9]*)?"
    r"|\*+[0-9]*"
    r"|[^\t\n\v\f\r ]"
)
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193


class Shape(IntEnum):
    """What a token looks like. 0 is padding."""

    PAD = 0
    UPPER = 1  # KES, EXAMPLE
    LOWER = 2  # debited
    TITLE = 3  # Sample
    MIXED = 4  # letters and digits, such as SK29XY7
    INTEGER = 5  # 1250
    DECIMAL = 6  # 1250.00
    GROUPED = 7  # 1,250 or 1,250.00
    DATE = 8  # 29/10/26, 29-10-2026, 2026-10-29
    TIME = 9  # 14:05 or 14:05:33
    MASK = 10  # ***1234
    PUNCT = 11  # any other single character


N_SHAPES = len(Shape)


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int


def tokenise(message: str) -> list[Token]:
    """Every token in ``message``, with its character offsets."""
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(message)]


def _is_digits(text: str) -> bool:
    return text != "" and all("0" <= c <= "9" for c in text)


def _number_shape(text: str) -> Shape:
    if _is_digits(text):
        return Shape.INTEGER
    if ":" in text:
        return Shape.TIME
    if "/" in text or "-" in text:
        return Shape.DATE
    return Shape.GROUPED if "," in text else Shape.DECIMAL


def _word_shape(text: str) -> Shape:
    if any("0" <= c <= "9" for c in text):
        return Shape.MIXED
    if text.isupper():
        return Shape.UPPER if len(text) > 1 else Shape.TITLE
    return Shape.TITLE if "A" <= text[0] <= "Z" else Shape.LOWER


def shape_of(text: str) -> Shape:
    first = text[0]
    if "0" <= first <= "9":
        return _number_shape(text)
    if first == "*":
        return Shape.MASK
    if ("a" <= first <= "z") or ("A" <= first <= "Z"):
        return _word_shape(text)
    return Shape.PUNCT


def fnv1a_32(text: str) -> int:
    value = _FNV_OFFSET
    for byte in text.encode("ascii", "replace"):
        value ^= byte
        value = (value * _FNV_PRIME) & 0xFFFFFFFF
    return value


def hash_key(text: str) -> str:
    """What a token hashes as: numbers and masks by their shape only, never their value, so the
    model learns layouts rather than amounts; words in ASCII lower case."""
    shape = shape_of(text)
    if shape in (Shape.INTEGER, Shape.DECIMAL, Shape.GROUPED, Shape.DATE, Shape.TIME):
        return "#" + "".join("9" if "0" <= c <= "9" else c for c in text)
    if shape is Shape.MASK:
        return "#mask"
    return text.translate(_ASCII_LOWER)


def token_id(text: str) -> int:
    """1 .. HASH_BUCKETS - 1; 0 is padding."""
    return 1 + fnv1a_32(hash_key(text)) % (HASH_BUCKETS - 1)


def encode(message: str) -> list[int]:
    """The model input for ``message``: ``MAX_TOKENS`` token ids then ``MAX_TOKENS`` shape ids.
    Tokens past ``MAX_TOKENS`` are dropped."""
    tokens = tokenise(message)[:MAX_TOKENS]
    ids = [token_id(t.text) for t in tokens]
    shapes = [int(shape_of(t.text)) for t in tokens]
    pad = MAX_TOKENS - len(tokens)
    return ids + [0] * pad + shapes + [0] * pad


def reader_manifest() -> dict[str, Any]:
    return {
        "reader_schema_version": READER_SCHEMA_VERSION,
        "max_tokens": MAX_TOKENS,
        "hash_buckets": HASH_BUCKETS,
        "input": {"name": "message", "dtype": "int32", "shape": [-1, 2 * MAX_TOKENS]},
    }

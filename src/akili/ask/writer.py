"""The ask writer's vocabulary and encoding. The writer itself is frozen as shipped.

The app runs a small causal (decoder-only) transformer over word tokens, ``ask-writer.weights``
(pesaclarity/docs, decision 0023: frozen until Ask LM replaces it). Its input is the fact
tokens, then ``<sep>``, then the reply so far; it predicts the next reply token. The app decodes
one token at a time (``docs/ASK.md``), fills the placeholders and runs
:func:`akili.ask.schema.check_reply`; on a failure it shows the kind's fixed fallback sentence.

The Keras code that trained it, its TFLite export and the step that read its weights out of
``ask-writer.tflite`` were removed with TensorFlow (decision 0023, stage 5): it can no longer
be retrained from this branch. They are at the ``tflite-archive`` tag, and the TFLite file the
shipped weights came from is in that tag's release. What stays is what the app's conformance
cases (:mod:`akili.ask.examples`) and the reply checks need: the vocabulary, in the order the
shipped file's ids follow, and the training examples' encoding.
"""

from __future__ import annotations

import numpy as np

from akili.ask.replies import ALL_FRAGMENT_POOLS, Example
from akili.ask.schema import (
    EOS,
    FACT_TOKENS,
    PLACEHOLDER_TOKENS,
    PUNCTUATION,
    SEP,
    SPECIAL_TOKENS,
    WRITER_LENGTH,
    reply_tokens,
)


def build_vocab() -> list[str]:
    """Special tokens, fact tokens, placeholders, punctuation, then every word the fragments use,
    sorted. Ids are positions in this list."""
    words: set[str] = set()
    for pool in ALL_FRAGMENT_POOLS:
        for text in pool:
            body = " ".join(w for w in text.split(" ") if not w.startswith("@"))
            words.update(
                t for t in reply_tokens(body) if not t.startswith("{") and t not in PUNCTUATION
            )
    return [*SPECIAL_TOKENS, *FACT_TOKENS, *PLACEHOLDER_TOKENS, *PUNCTUATION, *sorted(words)]


VOCAB: list[str] = build_vocab()
TOKEN_ID: dict[str, int] = {t: i for i, t in enumerate(VOCAB)}


def encode_example(e: Example) -> list[int]:
    """``WRITER_LENGTH + 1`` ids: facts, ``<sep>``, reply, ``<eos>``, padding."""
    ids = [TOKEN_ID[t] for t in e.facts.tokens()] + [TOKEN_ID[SEP]]
    ids += [TOKEN_ID[t] for t in e.reply] + [TOKEN_ID[EOS]]
    if len(ids) > WRITER_LENGTH:
        raise ValueError(f"example longer than {WRITER_LENGTH}: {e}")
    return ids + [0] * (WRITER_LENGTH + 1 - len(ids))


def to_arrays(examples: list[Example]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inputs and next-token targets ``[n, WRITER_LENGTH]``, and weights that keep only the
    reply and ``<eos>`` targets."""
    seq = np.array([encode_example(e) for e in examples], dtype=np.int32)
    x, y = seq[:, :-1], seq[:, 1:]
    sep = (x == TOKEN_ID[SEP]).argmax(axis=1)
    pos = np.arange(WRITER_LENGTH)[None, :]
    w = ((pos >= sep[:, None]) & (y != 0)).astype("float32")
    return x, y, w

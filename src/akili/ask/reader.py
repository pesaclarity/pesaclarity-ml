"""The Ask reader's config, data pipeline and scores, shared by its trainers.

The input is the message reader's encoding unchanged (token hashes then shapes), so the app's
tokeniser serves both. The network has the message reader's shape: token and shape embeddings,
two 1-D convolutions and a dilated one, then a slot label per token and, from the pooled tokens,
the intent. Version 2 widens the convolutions (96 filters) and gives the slot head two small
convolutions of its own, so a name keeps one label from its first word to its last. About
240k parameters (the budget is 400 KB).

Training swaps some token ids for random hash buckets in every epoch (:func:`swap_tokens`): most
of the payee and account tokens, a few of the rest. An unknown name hashes to a bucket some
other word trained, so the reader learns to tag names from their place in the question.

Evaluation holds templates out (see :func:`akili.ask.questions.folds`): each fold trains
without a fifth of every intent's templates (:func:`fold_questions`) and scores the exported
model on them. Slots score as whole spans: a slot counts only when every token of it, and
nothing else, carries its label. The exported model is also scored on :mod:`akili.ask.probes`
(:func:`probe_scores`).

The Keras trainer and its TFLite export are :mod:`akili.ask.keras_reader`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from akili.ask import names, probes
from akili.ask.questions import (
    ACCOUNTS,
    PAYEES,
    Question,
    folds,
    generate,
    template_ids,
    to_arrays,
    v1_template_ids,
    v2_template_ids,
)
from akili.ask.schema import ASK_INTENTS, SLOT_LABELS
from akili.reader.tokens import HASH_BUCKETS, MAX_TOKENS, token_id, tokenise

MANIFEST_FILE = "ask-reader-manifest.json"
REPORT_FILE = "ask-reader-report.json"

# A model as the scores see it: inputs ``int32[n, 192]`` to (intent ``[n, intents]``, slots
# ``[n, MAX_TOKENS, slots]``), as the exported file runs them.
Run = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True, slots=True)
class AskReaderConfig:
    questions: int = 120_000
    folds: int = 5
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.003
    token_dim: int = 24
    shape_dim: int = 8
    filters: int = 96
    slot_filters: int = 64
    dropout: float = 0.1
    seed: int = 0
    max_size_bytes: int = 400_000
    name_dropout: float = 0.7
    """Share of payee and account tokens whose id is swapped for a random bucket in each epoch,
    so a name reads from its place in the question, not its word."""
    word_dropout: float = 0.03
    """The same for every other token."""


def _spans(labels: np.ndarray) -> set[tuple[int, int, int]]:
    """Maximal runs of the same non-other label, as (label, start, end)."""
    spans = set()
    start = None
    for i in range(len(labels) + 1):
        current = int(labels[i]) if i < len(labels) else -1
        previous = int(labels[i - 1]) if i > 0 else -1
        if start is not None and current != previous:
            if previous > 0:
                spans.add((previous, start, i))
            start = None
        if start is None and current > 0:
            start = i
    return spans


def slot_scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-slot precision, recall and F1 over exact spans, plus ``all`` (micro). Padding (-1)
    is ignored."""
    k = len(SLOT_LABELS)
    tp, fp, fn = np.zeros(k), np.zeros(k), np.zeros(k)
    for t_row, p_row in zip(truth, predicted, strict=True):
        n = int((t_row >= 0).sum())
        true_spans = _spans(t_row[:n])
        pred_spans = _spans(p_row[:n])
        for label, *_ in true_spans & pred_spans:
            tp[label] += 1
        for label, *_ in pred_spans - true_spans:
            fp[label] += 1
        for label, *_ in true_spans - pred_spans:
            fn[label] += 1

    def score(t: float, f_p: float, f_n: float) -> dict[str, float]:
        p = t / max(t + f_p, 1)
        r = t / max(t + f_n, 1)
        return {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(2 * p * r / max(p + r, 1e-9)), 4),
            "support": int(t + f_n),
        }

    out = {SLOT_LABELS[i]: score(tp[i], fp[i], fn[i]) for i in range(1, k)}
    out["all"] = score(tp[1:].sum(), fp[1:].sum(), fn[1:].sum())
    return out


def scores(run: Run, questions: list[Question]) -> dict[str, Any]:
    """Intent accuracy (overall and per intent) and slot scores of ``run`` on ``questions``."""
    x, y, k = to_arrays(questions)
    intent_p, slots_p = run(x)
    predicted = intent_p.argmax(-1)
    per_intent = {
        name: round(float((predicted[k == i] == i).mean()), 4)
        for i, name in enumerate(ASK_INTENTS)
        if (k == i).any()
    }
    return {
        "intent_accuracy": round(float((predicted == k).mean()), 4),
        "per_intent_accuracy": per_intent,
        "slots": slot_scores(y, slots_p.argmax(-1)),
        "questions": len(questions),
    }


def probe_scores(run: Run) -> dict[str, Any]:
    """Scores of ``run`` on the probe sets (see :mod:`akili.ask.probes`): (b) the hard held-out
    set, (c) the app's questions, the known misses with their variants and look-alikes, and
    the conformance file's questions, with the reading of each question of the last three."""
    hard = probes.hard_questions()
    qs = [q for _, q in hard]
    intent_p, slots_p = run(to_arrays(qs)[0])
    app = probes.app_questions()
    intent_a, slots_a = run(to_arrays(app)[0])
    misses = probes.miss_questions()
    mq = [q for _, q in misses]
    intent_m, slots_m = run(to_arrays(mq)[0])
    conf = probes.conformance_questions()
    intent_c, slots_c = run(to_arrays(conf)[0])
    return {
        "hard_held_out": probes.score_probe(qs, intent_p, slots_p, [g for g, _ in hard]),
        "app_questions": {
            **probes.score_probe(app, intent_a, slots_a, None),
            "readings": probes.readings(app, intent_a, slots_a),
        },
        "misses": {
            **probes.score_probe(mq, intent_m, slots_m, [g for g, _ in misses]),
            "readings": probes.readings(mq, intent_m, slots_m),
        },
        "conformance_questions": {
            **probes.score_probe(conf, intent_c, slots_c, None),
            "readings": probes.readings(conf, intent_c, slots_c),
        },
    }


_NAME_LABELS = [SLOT_LABELS.index(s) for s in ("payee", "account")]
# Words of the version 1 payee and account lists (well-known chains, banks, "my mum"). They
# keep their ids as other words do, so the reader still knows them by name.
_KNOWN_NAME_IDS = sorted(
    {token_id(t.text) for name in [*PAYEES, *ACCOUNTS] for t in tokenise(name)}
)


def swap_tokens(
    x: np.ndarray, y: np.ndarray, cfg: AskReaderConfig, rng: np.random.Generator
) -> np.ndarray:
    """A copy of ``x`` with some token ids (never shapes or padding) swapped for random
    buckets: ``name_dropout`` of the payee and account tokens (but for the words of the well-known
    names the reader should know by name), ``word_dropout`` of the rest. An unknown name hashes
    to a bucket some other word trained, so this is what the app sees."""
    out = x.copy()
    tokens = out[:, :MAX_TOKENS]
    named = np.isin(y, _NAME_LABELS) & ~np.isin(tokens, _KNOWN_NAME_IDS)
    p = np.where(named, cfg.name_dropout, cfg.word_dropout)
    swap = (rng.random(y.shape) < p) & (y >= 0)
    tokens[swap] = rng.integers(1, HASH_BUCKETS, size=int(swap.sum()))
    return out


def swap_rng(cfg: AskReaderConfig) -> np.random.Generator:
    """The generator of a training run's token swaps (one draw of :func:`swap_tokens` per
    epoch)."""
    return np.random.default_rng(cfg.seed + 7)


def training_questions(cfg: AskReaderConfig) -> list[Question]:
    """The questions the released model trains on: every template, invented names."""
    return generate(cfg.questions, seed=cfg.seed, pool=names.TRAIN)


def seen_questions(cfg: AskReaderConfig) -> list[Question]:
    """Questions from the templates the model trained on, scored as ``seen_templates``."""
    return generate(2000, seed=cfg.seed + 999)


def fold_questions(
    cfg: AskReaderConfig, i: int
) -> tuple[list[Question], dict[str, list[Question] | None]]:
    """Fold ``i``: the questions to train on (the other folds' templates) and the held-out
    ones, as ``v1``, ``v2`` and ``v3`` (the templates of each version; ``None`` when the fold
    holds none). The version 1 templates are scored on the very questions version 1 was scored
    on (same folds, seeds and value lists); the version 2 and 3 ones on their own, with
    invented names."""
    all_ids = set(template_ids())
    v1, v2 = v1_template_ids(), v2_template_ids()
    held = folds(cfg.folds, seed=cfg.seed)[i]
    train = generate(
        cfg.questions, seed=cfg.seed + 100 + i, templates=all_ids - held, pool=names.TRAIN
    )
    test: dict[str, list[Question] | None] = {
        "v1": generate(cfg.questions // 5, seed=cfg.seed + 200 + i, templates=held & v1)
    }
    for key, ids, seed in (("v2", held & v2, 300), ("v3", held - v1 - v2, 400)):
        test[key] = (
            generate(cfg.questions // 20, seed=cfg.seed + seed + i, templates=ids, pool=names.TRAIN)
            if ids
            else None
        )
    return train, test


def summarise_folds(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
    """The held-out score from each fold's ``fold_scores`` (one per trainer)."""
    results = [f["v1"] for f in per_fold]
    per_intent = {
        name: round(float(np.mean([r["per_intent_accuracy"][name] for r in results])), 4)
        for name in ASK_INTENTS
    }

    def later(key: str) -> dict[str, Any] | None:
        rs = [f[key] for f in per_fold if f.get(key) is not None]
        if not rs:
            return None
        return {
            "mean_intent_accuracy": round(float(np.mean([r["intent_accuracy"] for r in rs])), 4),
            "mean_slot_f1": round(float(np.mean([r["slots"]["all"]["f1"] for r in rs])), 4),
            "questions": int(sum(r["questions"] for r in rs)),
        }

    return {
        "mean_intent_accuracy": round(float(np.mean([r["intent_accuracy"] for r in results])), 4),
        "mean_slot_f1": {
            name: round(float(np.mean([r["slots"][name]["f1"] for r in results])), 4)
            for name in [*SLOT_LABELS[1:], "all"]
        },
        "mean_per_intent_accuracy": per_intent,
        "folds": results,
        "v2": later("v2"),
        "v3": later("v3"),
    }

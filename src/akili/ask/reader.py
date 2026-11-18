"""The Ask reader: its config, data pipeline and scores, and the PyTorch trainer that releases it.

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

The release path (pesaclarity/docs, decision 0023; needs the [torch] extra): :func:`fit` trains
:class:`akili.ask.reader_model.AskReader`, ``torch.onnx`` exports it and rayo-convert quantises
it to int8 (:func:`export_module`), and every score is of that ``ask-reader.weights`` file as
rayo runs it (:func:`train_ask_reader`). The Keras trainer and its TFLite export it replaces
are :mod:`akili.ask.keras_reader`, until TensorFlow is removed (stage 5).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from akili._optional import require
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
    v3_template_ids,
    v4_template_ids,
)
from akili.ask.reader_weights import WEIGHTS_FILE
from akili.ask.schema import ASK_INTENTS, ASK_SCHEMA_VERSION, SLOT_LABELS
from akili.reader.tokens import HASH_BUCKETS, MAX_TOKENS, reader_manifest, token_id, tokenise

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
    # The PyTorch trainer (fit): the Keras recipe (the learning rate falls along a cosine to
    # cosine_floor of itself over the run), plus one change, which lifts the held-out template
    # scores (docs/ASK.md): the intent's and the slots' targets smoothed by 0.05. 0.0 gives the
    # Keras recipe.
    cosine_floor: float = 0.05
    intent_label_smoothing: float = 0.05
    slot_label_smoothing: float = 0.05
    threads: int = 1  # torch's CPU threads: fixed, so a run is repeatable
    # How far rayo's int8 file may be from the torch model's float probabilities on the parity
    # questions (docs/ASK.md#parity): the largest difference per output, the 99.9th percentile
    # of the differences, and the gap between torch's two best probabilities beyond which the
    # labels must agree. Decision 0023 proposed 1e-2; the trained reader's int8 layers move
    # 99.9% of probabilities by less than 0.05 and a few, between two close labels, by up to
    # 0.15, so two labels whose probabilities are less than about twice that apart can swap.
    parity_tolerance_intent: float = 0.1
    parity_tolerance_slots: float = 0.25
    parity_tolerance_p99_9: float = 0.06
    parity_label_margin: float = 0.2


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
    ones, as ``v1`` to ``v4`` (the templates of each version; ``None`` when the fold holds
    none). The version 1 templates are scored on the very questions version 1 was scored on
    (same folds, seeds and value lists); the later ones on their own, with invented names, and
    each version's on the same questions as when it was added."""
    all_ids = set(template_ids())
    v1 = v1_template_ids()
    held = folds(cfg.folds, seed=cfg.seed)[i]
    train = generate(
        cfg.questions, seed=cfg.seed + 100 + i, templates=all_ids - held, pool=names.TRAIN
    )
    test: dict[str, list[Question] | None] = {
        "v1": generate(cfg.questions // 5, seed=cfg.seed + 200 + i, templates=held & v1)
    }
    later = (
        ("v2", held & v2_template_ids(), 300),
        ("v3", held & v3_template_ids(), 400),
        ("v4", held & v4_template_ids(), 500),
    )
    for key, ids, seed in later:
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
        "v4": later("v4"),
    }


# The torch path (needs the [torch] extra): what is released. torch trains the reader,
# torch.onnx exports it and rayo-convert quantises it to int8 (akili.export.onnx); every score
# is of that file as rayo runs it.

PARITY_QUESTIONS = 1000  # generated with the next seed, beside the probe sets

# How far a slot output reaches: the slots at token p are a function of the embeddings of
# tokens p - 10 to p + 10 (conv_1, conv_2 and conv_3 reach 1, 2 and 3 tokens; slot_context
# and slot_span 2 and 2), and the intent pools the question's own tokens only. Training runs
# each batch on its longest question and this many padding tokens after it, which gives the
# very outputs, loss and gradients of all 96 positions, in a third of the time.
RECEPTIVE_FIELD = 10


def _lengths(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The number of tokens of each question (labels and shapes both mark them)."""
    lengths: np.ndarray = np.maximum((y >= 0).sum(1), (x[:, MAX_TOKENS:] > 0).sum(1))
    return lengths


def fit(cfg: AskReaderConfig, questions: list[Question]) -> Any:
    """An :class:`~akili.ask.reader_model.AskReader` trained on ``questions`` with the Keras
    reader's recipe: Adam (Keras' epsilon) on a cosine decay to ``cosine_floor`` of the rate,
    the batch shuffled each epoch, the token swaps drawn afresh each epoch from the same
    generator as the Keras trainer's (:func:`swap_tokens`), cross-entropy of the intent plus
    cross-entropy per token (padding weighted 0, averaged over every position, as Keras'
    ``sum_over_batch_size``), and the Keras network's dropout; with ``cfg``'s label smoothing
    on top. Seeded and deterministic."""
    from akili.ask.reader_model import AskReader, split
    from akili.export.onnx import deterministic

    torch = require("torch", "torch")
    deterministic(cfg.seed, cfg.threads)
    model = AskReader(
        token_dim=cfg.token_dim,
        shape_dim=cfg.shape_dim,
        filters=cfg.filters,
        slot_filters=cfg.slot_filters,
        dropout=cfg.dropout,
    )
    x, y, k = to_arrays(questions)
    rng = swap_rng(cfg)
    lengths = torch.from_numpy(_lengths(x, y).astype(np.int64))
    labels = torch.from_numpy(np.maximum(y, 0).astype(np.int64))
    weight = torch.from_numpy((y >= 0).astype(np.float32))
    intents = torch.from_numpy(k.astype(np.int64))
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-7)
    steps = cfg.epochs * -(-len(x) // cfg.batch_size)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, steps, eta_min=cfg.learning_rate * cfg.cosine_floor
    )
    order = torch.Generator().manual_seed(cfg.seed)
    ce = torch.nn.functional.cross_entropy
    model.train()
    for _ in range(cfg.epochs):
        swapped = swap_tokens(x, y, cfg, rng)
        tokens, shapes = (torch.from_numpy(a.astype(np.int64)) for a in split(swapped))
        for batch in torch.randperm(len(x), generator=order).split(cfg.batch_size):
            n = min(MAX_TOKENS, int(lengths[batch].max()) + RECEPTIVE_FIELD)
            slots, intent = model.logits(tokens[batch, :n], shapes[batch, :n])
            per_token = ce(
                slots.reshape(-1, len(SLOT_LABELS)),
                labels[batch, :n].reshape(-1),
                reduction="none",
                label_smoothing=cfg.slot_label_smoothing,
            )
            loss = (per_token * weight[batch, :n].reshape(-1)).sum() / (len(batch) * MAX_TOKENS)
            loss = loss + ce(intent, intents[batch], label_smoothing=cfg.intent_label_smoothing)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
    model.eval()
    return model


def export_module(model: Any) -> Any:
    """``model`` through ONNX and rayo-convert to its int8 ``ask-reader.weights``
    (:class:`akili.export.onnx.Export`)."""
    from akili.ask.reader_model import (
        INPUTS,
        LAYERS,
        ONNX_FILE,
        OUTPUTS,
        RAYO_OPTIONS,
        required_tensors,
    )
    from akili.export.onnx import weights_of_module

    torch = require("torch", "torch")
    example = {name: torch.zeros((1, MAX_TOKENS), dtype=torch.int64) for name in INPUTS}
    return weights_of_module(
        model,
        example,
        OUTPUTS,
        RAYO_OPTIONS,
        layers=LAYERS,
        required=required_tensors(),
        onnx_name=ONNX_FILE,
    )


def runner(done: Any, rayo_dir: str | Path | None) -> Any:
    """The converted file on rayo's C library when ``rayo_dir`` is a rayo checkout, else on
    rayo-convert's NumPy reference (the same int8 meaning, which rayo is checked against)."""
    from akili.ask.reader_model import RAYO_OPTIONS
    from akili.export.onnx import RayoRunner, ReferenceRunner

    if rayo_dir:
        return RayoRunner(done.data, rayo_dir)
    return ReferenceRunner(done.onnx, RAYO_OPTIONS)


def run_converted(converted: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The converted file's intent ``[n, intents]`` and slots ``[n, 96, slots]`` for inputs
    ``x``, a question at a time, as the app runs it (rayo's inputs: tokens, then shapes)."""
    from akili.ask.reader_model import split

    intents, slots = [], []
    for row in np.asarray(x).astype(np.int32):
        tokens, shapes = split(row)
        s, i = converted.run({"tokens": tokens, "shapes": shapes}, {"T": MAX_TOKENS})
        slots.append(np.asarray(s).reshape(MAX_TOKENS, len(SLOT_LABELS)))
        intents.append(np.asarray(i).reshape(len(ASK_INTENTS)))
    return np.stack(intents), np.stack(slots)


def converted_run(converted: Any) -> Run:
    """:func:`run_converted` on ``converted`` as a :data:`Run`."""
    return lambda x: run_converted(converted, x)


def torch_outputs(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The torch model's float intent ``[n, intents]`` and slots ``[n, 96, slots]``."""
    from akili.ask.reader_model import split

    torch = require("torch", "torch")
    model.eval()
    intents, slots = [], []
    with torch.no_grad():
        for start in range(0, len(x), 1000):
            tokens, shapes = (
                torch.from_numpy(a.astype(np.int64)) for a in split(x[start : start + 1000])
            )
            s, i = model(tokens, shapes)
            slots.append(s.numpy())
            intents.append(i.numpy())
    return np.concatenate(intents), np.concatenate(slots)


def fold_scores(cfg: AskReaderConfig, i: int, rayo_dir: str | Path | None = None) -> dict[str, Any]:
    """Fold ``i`` (:func:`fold_questions`): train on the other folds' templates, export, and
    score the converted int8 file on this fold's, as ``v1`` to ``v4``. A fold can run
    in a process of its own; :func:`summarise_folds` joins them."""
    train, test = fold_questions(cfg, i)
    run = converted_run(runner(export_module(fit(cfg, train)), rayo_dir))
    return {key: scores(run, qs) if qs is not None else None for key, qs in test.items()}


def _start_folds(cfg: AskReaderConfig, rayo_dir: str | Path | None, jobs: int) -> Any:
    """The held-out folds started in ``jobs`` processes of their own (``None`` for one job):
    the caller can train the released model meanwhile. Each trains on ``cfg.threads``
    threads, so the scores do not depend on ``jobs``."""
    if jobs <= 1:
        return None
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    pool = ProcessPoolExecutor(
        min(jobs, cfg.folds), mp_context=multiprocessing.get_context("spawn")
    )
    return pool, [pool.submit(fold_scores, cfg, i, rayo_dir) for i in range(cfg.folds)]


def _join_folds(cfg: AskReaderConfig, rayo_dir: str | Path | None, started: Any) -> dict[str, Any]:
    if started is None:
        return summarise_folds([fold_scores(cfg, i, rayo_dir) for i in range(cfg.folds)])
    pool, futures = started
    try:
        return summarise_folds([f.result() for f in futures])
    finally:
        pool.shutdown(cancel_futures=True)


def evaluate_unseen_templates(
    cfg: AskReaderConfig, rayo_dir: str | Path | None = None, jobs: int = 1
) -> dict[str, Any]:
    """Each fold: train on the other folds' templates, score on this fold's; ``jobs`` folds
    at a time, each in a process of its own (the scores do not depend on ``jobs``)."""
    return _join_folds(cfg, rayo_dir, _start_folds(cfg, rayo_dir, jobs))


def parity_questions(cfg: AskReaderConfig) -> dict[str, list[Question]]:
    """The questions parity is checked on: :data:`PARITY_QUESTIONS` generated with the next
    seed, and the probe sets."""
    return {
        "generated": generate(PARITY_QUESTIONS, seed=cfg.seed + 1, pool=names.TRAIN),
        "hard_held_out": [q for _, q in probes.hard_questions()],
        "app_questions": probes.app_questions(),
        "misses": [q for _, q in probes.miss_questions()],
        "conformance_questions": probes.conformance_questions(),
    }


def check_parity(
    model: Any, done: Any, cfg: AskReaderConfig, rayo_dir: str | Path | None = None
) -> dict[str, Any]:
    """The torch model's float probabilities against the converted int8 file on
    :func:`parity_questions`: the largest difference per output and its 99.9th percentile
    (within ``cfg``'s tolerances), and the labels (each question's intent, each token's slot),
    which must agree wherever torch's two best probabilities are more than
    ``cfg.parity_label_margin`` apart. Also the
    NumPy reference the app's Dart reader copies (:func:`akili.ask.reader_weights.forward`)
    against the converted file, on the app, misses and conformance questions."""
    from akili.ask.reader_weights import forward

    sets = parity_questions(cfg)
    x, y, _ = to_arrays([q for qs in sets.values() for q in qs])
    want = torch_outputs(model, x)
    converted = runner(done, rayo_dir)
    got = run_converted(converted, x)
    tolerance = {"intent": cfg.parity_tolerance_intent, "slots": cfg.parity_tolerance_slots}
    out: dict[str, Any] = {
        "questions": {name: len(qs) for name, qs in sets.items()},
        "runner": "rayo" if rayo_dir else "rayo-convert reference",
        "tolerance": tolerance,
        "p99_9_tolerance": cfg.parity_tolerance_p99_9,
        "label_margin": cfg.parity_label_margin,
    }
    within = True
    for name, w, g in zip(("intent", "slots"), want, got, strict=True):
        diff = np.abs(g - w).max(-1)
        top = np.sort(w, axis=-1)
        clear = top[..., -1] - top[..., -2] > cfg.parity_label_margin
        same = w.argmax(-1) == g.argmax(-1)
        if name == "slots":  # the questions' tokens, not the padding
            diff, clear, same = diff[y >= 0], clear[y >= 0], same[y >= 0]
        entry = {
            "max_abs": round(float(diff.max()), 6),
            "p99_9_abs": round(float(np.quantile(diff, 0.999)), 6),
            "over_1e-2": int((diff > 1e-2).sum()),
            "labels": int(diff.size),
            "labels_same": int(same.sum()),
            "labels_compared": int(clear.sum()),
            "labels_compared_same": int(same[clear].sum()),
        }
        out[name] = entry
        within &= entry["max_abs"] <= tolerance[name]
        within &= entry["p99_9_abs"] <= cfg.parity_tolerance_p99_9
        within &= entry["labels_compared"] == entry["labels_compared_same"]
    names_ = ("app_questions", "misses", "conformance_questions")
    xd, yd, _ = to_arrays([q for name in names_ for q in sets[name]])
    gi, gs = run_converted(converted, xd)
    worst = 0.0
    for row, ti, ts, labels in zip(xd, gi, gs, yd, strict=True):
        di, ds = forward(done.weights, row)
        worst = max(worst, float(np.abs(di - ti).max()), float(np.abs(ds - ts)[labels >= 0].max()))
    out["dart_reference"] = {"questions": len(xd), "max_abs": round(worst, 7)}
    out["within"] = bool(within)
    return out


def _versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in ("torch", "onnx", "rayo-convert")}


def manifest(data: bytes, onnx: bytes) -> dict[str, Any]:
    """``ask-reader-manifest.json`` for the weights file ``data`` exported from ``onnx``. There
    is no TFLite file: ``source`` names the ONNX file the weights were converted from."""
    from akili.ask.reader_model import ONNX_FILE
    from akili.ask.reader_weights import manifest_fields

    return {
        **manifest_fields(data),
        "source": {
            "file": ONNX_FILE,
            "sha256": hashlib.sha256(onnx).hexdigest(),
            "size_bytes": len(onnx),
        },
        "ask_schema_version": ASK_SCHEMA_VERSION,
        **{k: v for k, v in reader_manifest().items() if k != "input"},
        "input": {
            "name": "question",
            "dtype": "int32",
            "shape": [-1, 2 * MAX_TOKENS],
            "encoding": "the message reader's encode(), unchanged (docs/READER.md)",
        },
        "rayo_inputs": {
            "tokens": {"dtype": "int32", "shape": [MAX_TOKENS], "from": "question[0:96]"},
            "shapes": {"dtype": "int32", "shape": [MAX_TOKENS], "from": "question[96:192]"},
        },
        "outputs": {
            "intent": {"shape": [-1, len(ASK_INTENTS)], "labels": list(ASK_INTENTS)},
            "slots": {"shape": [-1, MAX_TOKENS, len(SLOT_LABELS)], "labels": list(SLOT_LABELS)},
        },
        "trained_on": "synthetic questions only",
    }


def train_ask_reader(
    cfg: AskReaderConfig,
    out_dir: str | Path,
    evaluate: bool = True,
    rayo_dir: str | Path | None = None,
    jobs: int = 1,
) -> dict[str, Any]:
    """Train on every template and export ``ask-reader.weights`` (int8, with its rayo graph),
    the ONNX file it was converted from and the manifest; score that file on the templates it
    trained on and the probe sets, then (unless ``evaluate`` is off) score the recipe on
    held-out templates, ``jobs`` folds at a time. Fails if the weights are over the size budget
    or the converted file strays from the torch model by more than the tolerance. Returns the
    report.

    ``rayo_dir`` (by default ``RAYO_DIR``): a rayo checkout whose C library runs the scores;
    without one, rayo-convert's NumPy reference runs them."""
    from akili.ask.reader_model import ONNX_FILE, parameter_count
    from akili.export.onnx import OPSET
    from akili.export.size import ModelTooLargeError

    rayo = rayo_dir if rayo_dir is not None else os.environ.get("RAYO_DIR") or None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # With more than one job the folds train in processes of their own while this one trains
    # the model to export; that model is the same with or without the evaluation.
    started = _start_folds(cfg, rayo, jobs) if evaluate else None
    try:
        model = fit(cfg, training_questions(cfg))
        done = export_module(model)
        data = done.data
        if len(data) > cfg.max_size_bytes:
            raise ModelTooLargeError(
                f"{WEIGHTS_FILE} is {len(data)} bytes; the budget is {cfg.max_size_bytes}"
            )
        parity_report = check_parity(model, done, cfg, rayo)
        run = converted_run(runner(done, rayo))
        seen = scores(run, seen_questions(cfg))
        probed = probe_scores(run)
    except BaseException:
        if started is not None:
            started[0].shutdown(cancel_futures=True)
        raise
    unseen = _join_folds(cfg, rayo, started) if evaluate else None
    report = {
        "config": asdict(cfg),
        "parameters": parameter_count(model),
        "weights_size_bytes": len(data),
        "weights_sha256": hashlib.sha256(data).hexdigest(),
        "templates": len(template_ids()),
        "scored_on": f"{WEIGHTS_FILE} (int8) on "
        + ("rayo's C library" if rayo else "rayo-convert's reference"),
        "unseen_templates": unseen,
        "seen_templates": seen,
        **probed,
        "parity": parity_report,
        "export": {"onnx_opset": OPSET, "onnx_exporter": "torchscript", **_versions()},
        "note": "Scores are on synthetic questions; real questions will score lower.",
    }
    # The report is written either way, so a failed run can be read; the files only when the
    # converted file is within the tolerance.
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not parity_report["within"]:
        raise ValueError(f"{WEIGHTS_FILE} strays from the torch model: {parity_report}")
    (out / WEIGHTS_FILE).write_bytes(data)
    (out / ONNX_FILE).write_bytes(done.onnx)
    doc = manifest(data, done.onnx)
    (out / MANIFEST_FILE).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return report

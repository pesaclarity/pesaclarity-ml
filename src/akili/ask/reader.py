"""Train, evaluate and export the ask reader (needs the [train] extra).

The input is the message reader's encoding unchanged (token hashes then shapes), so the app's
tokeniser serves both. The network has the message reader's shape: token and shape embeddings,
two 1-D convolutions and a dilated one, then a slot label per token and, from the pooled tokens,
the intent. Version 2 widens the convolutions (96 filters) and gives the slot head two small
convolutions of its own, so a name keeps one label from its first word to its last. About
240k parameters; the quantised TFLite file is about 270 KB (the budget is 400 KB).

Training swaps some token ids for random hash buckets in every epoch (:func:`swap_tokens`): most
of the payee and account tokens, a few of the rest. An unknown name hashes to a bucket some
other word trained, so the reader learns to tag names from their place in the question.

Evaluation holds templates out (see :func:`akili.ask.questions.folds`): each fold trains
without a fifth of every intent's templates and scores the TFLite model on them. Slots score
as whole spans: a slot counts only when every token of it, and nothing else, carries its label.
The exported model is also scored on :mod:`akili.ask.probes`.

Beside the TFLite file goes ``ask-reader.weights``, its tensors read back out of it in the plain
format of docs/WEIGHTS.md (:mod:`akili.ask.reader_weights`): the app runs the reader in Dart.
"""

from __future__ import annotations

import gc
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
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
)
from akili.ask.reader_weights import WEIGHTS_FILE, manifest_fields, read_tflite
from akili.ask.schema import ASK_INTENTS, ASK_SCHEMA_VERSION, SLOT_LABELS
from akili.reader.tokens import (
    HASH_BUCKETS,
    MAX_TOKENS,
    N_SHAPES,
    reader_manifest,
    token_id,
    tokenise,
)

MODEL_FILE = "ask-reader.tflite"
MANIFEST_FILE = "ask-reader-manifest.json"
REPORT_FILE = "ask-reader-report.json"


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


def build_ask_reader(cfg: AskReaderConfig) -> Any:
    keras = require("keras", "train")
    ops = keras.ops
    keras.utils.set_random_seed(cfg.seed)

    inputs = keras.Input(shape=(2 * MAX_TOKENS,), dtype="int32", name="question")
    tokens = inputs[:, :MAX_TOKENS]
    shapes = inputs[:, MAX_TOKENS:]
    x = keras.layers.Concatenate()(
        [
            keras.layers.Embedding(HASH_BUCKETS, cfg.token_dim, name="token_embedding")(tokens),
            keras.layers.Embedding(N_SHAPES, cfg.shape_dim, name="shape_embedding")(shapes),
        ]
    )
    emb = keras.layers.SpatialDropout1D(cfg.dropout)(x)
    x = keras.layers.Conv1D(cfg.filters, 3, padding="same", activation="relu", name="conv_1")(emb)
    y = keras.layers.Conv1D(cfg.filters, 5, padding="same", activation="relu", name="conv_2")(x)
    x = keras.layers.Add()([x, y])
    y = keras.layers.Conv1D(
        cfg.filters, 3, padding="same", dilation_rate=3, activation="relu", name="conv_3"
    )(x)
    x = keras.layers.Add()([x, y])
    # The token itself next to its context, so a known name keeps its slot in a new phrasing.
    t = keras.layers.Concatenate()([x, emb])
    t = keras.layers.Dense(cfg.slot_filters, activation="relu", name="slot_hidden")(t)
    # Neighbours' slot features, so a name keeps one label from its first word to its last.
    u = keras.layers.Conv1D(
        cfg.slot_filters, 5, padding="same", activation="relu", name="slot_context"
    )(t)
    t = keras.layers.Add()([t, u])
    u = keras.layers.Conv1D(
        cfg.slot_filters, 3, padding="same", dilation_rate=2, activation="relu", name="slot_span"
    )(t)
    t = keras.layers.Add()([t, u])
    slots = keras.layers.Dense(len(SLOT_LABELS), activation="softmax", name="slots")(t)

    mask = ops.expand_dims(ops.cast(ops.greater(shapes, 0), "float32"), -1)
    n = ops.maximum(ops.sum(mask, axis=1), 1.0)
    pooled = keras.layers.Concatenate()(
        [
            ops.max(x * mask - (1.0 - mask) * 1e4, axis=1),
            ops.sum(x * mask, axis=1) / n,
            # A bag of the words, which keeps working when a phrasing puts them in a new order.
            ops.sum(emb * mask, axis=1) / n,
        ]
    )
    h = keras.layers.Dropout(cfg.dropout)(pooled)
    h = keras.layers.Dense(64, activation="relu", name="intent_hidden")(h)
    intent = keras.layers.Dense(len(ASK_INTENTS), activation="softmax", name="intent")(h)
    return keras.Model(inputs, {"intent": intent, "slots": slots}, name="akili_ask_reader")


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


def to_tflite(model: Any) -> bytes:
    tf = require("tensorflow", "train")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return bytes(converter.convert())


def run_tflite(path: Path, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(intent ``[n, intents]``, slots ``[n, MAX_TOKENS, slots]``) from the exported file."""
    tf = require("tensorflow", "train")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        interpreter = tf.lite.Interpreter(model_path=str(path))
    inp = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(inp["index"], list(x.shape))
    interpreter.allocate_tensors()
    interpreter.set_tensor(inp["index"], x.astype(np.int32))
    interpreter.invoke()
    outputs = {d["shape_signature"].size: d for d in interpreter.get_output_details()}
    intent = interpreter.get_tensor(outputs[2]["index"])
    slots = interpreter.get_tensor(outputs[3]["index"])
    return np.asarray(intent), np.asarray(slots)


def probe_scores(path: Path) -> dict[str, Any]:
    """Scores of the TFLite file at ``path`` on the probe sets (see :mod:`akili.ask.probes`):
    (b) the hard held-out set, (c) the app's questions, the known misses with their variants
    and look-alikes, and the conformance file's questions, with the reading of each question
    of the last three."""
    hard = probes.hard_questions()
    qs = [q for _, q in hard]
    x, _, _ = to_arrays(qs)
    intent_p, slots_p = run_tflite(path, x)
    app = probes.app_questions()
    xa, _, _ = to_arrays(app)
    intent_a, slots_a = run_tflite(path, xa)
    misses = probes.miss_questions()
    mq = [q for _, q in misses]
    xm, _, _ = to_arrays(mq)
    intent_m, slots_m = run_tflite(path, xm)
    conf = probes.conformance_questions()
    xc, _, _ = to_arrays(conf)
    intent_c, slots_c = run_tflite(path, xc)
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


def _fit(cfg: AskReaderConfig, questions: list[Question]) -> Any:
    keras = require("keras", "train")
    x, y, k = to_arrays(questions)
    rng = np.random.default_rng(cfg.seed + 7)
    model = build_ask_reader(cfg)
    steps = cfg.epochs * -(-len(k) // cfg.batch_size)
    model.compile(
        optimizer=keras.optimizers.Adam(
            keras.optimizers.schedules.CosineDecay(cfg.learning_rate, steps, alpha=0.05)
        ),
        loss={
            "intent": keras.losses.SparseCategoricalCrossentropy(),
            "slots": keras.losses.SparseCategoricalCrossentropy(),
        },
        jit_compile=True,
    )
    for _ in range(cfg.epochs):
        model.fit(
            swap_tokens(x, y, cfg, rng),
            {"intent": k, "slots": np.maximum(y, 0)},
            sample_weight={
                "intent": np.ones(len(k), "float32"),
                "slots": (y >= 0).astype("float32"),
            },
            epochs=1,
            batch_size=cfg.batch_size,
            verbose=0,
        )
    return model


def score(model: Any, questions: list[Question], path: Path) -> dict[str, Any]:
    """Scores of ``model`` exported as TFLite, since that is what the app runs."""
    x, y, k = to_arrays(questions)
    path.write_bytes(to_tflite(model))
    try:
        intent_p, slots_p = run_tflite(path, x)
    finally:
        path.unlink()
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


def fold_scores(cfg: AskReaderConfig, i: int, work: Path) -> dict[str, Any]:
    """Fold ``i``: train on the other folds' templates and score on this fold's, as ``v1``,
    ``v2`` and ``v3`` (the templates of each version; ``None`` when the fold holds none). A
    fold can run in a process of its own; :func:`summarise_folds` joins them."""
    all_ids = set(template_ids())
    v1, v2 = v1_template_ids(), v2_template_ids()
    held = folds(cfg.folds, seed=cfg.seed)[i]
    train = generate(
        cfg.questions, seed=cfg.seed + 100 + i, templates=all_ids - held, pool=names.TRAIN
    )
    model = _fit(cfg, train)
    del train
    out: dict[str, Any] = {
        "v1": score(
            model,
            generate(cfg.questions // 5, seed=cfg.seed + 200 + i, templates=held & v1),
            work / "held-out.tflite",
        )
    }
    for key, ids, seed in (("v2", held & v2, 300), ("v3", held - v1 - v2, 400)):
        out[key] = (
            score(
                model,
                generate(
                    cfg.questions // 20, seed=cfg.seed + seed + i, templates=ids, pool=names.TRAIN
                ),
                work / "held-out.tflite",
            )
            if ids
            else None
        )
    # Free the fold's model before the next: six trainings in one process otherwise hold
    # several GB.
    del model
    require("keras", "train").backend.clear_session()
    gc.collect()
    return out


def summarise_folds(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
    """The held-out score from each fold's :func:`fold_scores`."""
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


def evaluate_unseen_templates(cfg: AskReaderConfig, work: Path) -> dict[str, Any]:
    """Each fold: train on the other folds' templates, score on this fold's. The version 1
    templates are scored on the very questions version 1 was scored on (same folds, seeds and
    value lists), so ``mean_intent_accuracy`` compares across versions; the version 2 and 3
    templates are scored on their own, with invented names, as ``v2`` and ``v3`` (version 2
    on the very questions it was scored on before version 3)."""
    return summarise_folds([fold_scores(cfg, i, work) for i in range(cfg.folds)])


def ask_reader_manifest(blob: bytes, weights: bytes) -> dict[str, Any]:
    """The manifest for the TFLite file ``blob`` and its plain ``weights``
    (:mod:`akili.ask.reader_weights`)."""
    return {
        "model_file": MODEL_FILE,
        "model_sha256": hashlib.sha256(blob).hexdigest(),
        "model_size_bytes": len(blob),
        "ask_schema_version": ASK_SCHEMA_VERSION,
        **{k: v for k, v in reader_manifest().items() if k != "input"},
        "input": {
            "name": "question",
            "dtype": "int32",
            "shape": [-1, 2 * MAX_TOKENS],
            "encoding": "the message reader's encode(), unchanged (docs/READER.md)",
        },
        "outputs": {
            "intent": {"shape": [-1, len(ASK_INTENTS)], "labels": list(ASK_INTENTS)},
            "slots": {"shape": [-1, MAX_TOKENS, len(SLOT_LABELS)], "labels": list(SLOT_LABELS)},
        },
        **manifest_fields(weights),
        "trained_on": "synthetic questions only",
    }


def train_ask_reader(
    cfg: AskReaderConfig, out_dir: str | Path, evaluate: bool = True
) -> dict[str, Any]:
    """Train on every template and export TFLite, its plain weights (read out of the TFLite
    file, for the app's Dart reader) and their manifest, score it on the probe sets, then
    (unless ``evaluate`` is off) score the recipe on held-out templates. Returns the report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The model to export first, so it is the same with or without the evaluation.
    model = _fit(cfg, generate(cfg.questions, seed=cfg.seed, pool=names.TRAIN))
    blob = to_tflite(model)
    if len(blob) > cfg.max_size_bytes:
        raise ValueError(f"ask reader is {len(blob)} bytes, over {cfg.max_size_bytes}")
    (out / MODEL_FILE).write_bytes(blob)
    weights = read_tflite(out / MODEL_FILE).write(out / WEIGHTS_FILE).read_bytes()
    seen = score(model, generate(2000, seed=cfg.seed + 999), out / "seen.tflite")
    probed = probe_scores(out / MODEL_FILE)
    unseen = evaluate_unseen_templates(cfg, out) if evaluate else None

    report = {
        "config": asdict(cfg),
        "parameters": int(model.count_params()),
        "model_size_bytes": len(blob),
        "templates": len(template_ids()),
        "unseen_templates": unseen,
        "seen_templates": seen,
        **probed,
        "note": "Scores are on synthetic questions; real questions will score lower.",
    }
    manifest = ask_reader_manifest(blob, weights)
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

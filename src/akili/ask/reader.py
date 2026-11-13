"""Train, evaluate and export the ask reader (needs the [train] extra).

The input is the message reader's encoding unchanged (token hashes then shapes), so the app's
tokeniser serves both. The network has the message reader's shape: token and shape embeddings,
two 1-D convolutions and a dilated one, then a slot label per token and, from the pooled tokens,
the intent. About 185k parameters; the quantised TFLite file is about 230 KB.

Evaluation holds templates out (see :func:`akili.ask.questions.folds`): each fold trains
without a third of every intent's templates and scores the TFLite model on them. Slots score
as whole spans: a slot counts only when every token of it, and nothing else, carries its label.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from akili._optional import require
from akili.ask.questions import Question, folds, generate, template_ids, to_arrays
from akili.ask.schema import ASK_INTENTS, ASK_SCHEMA_VERSION, SLOT_LABELS
from akili.reader.tokens import HASH_BUCKETS, MAX_TOKENS, N_SHAPES, reader_manifest

MODEL_FILE = "ask-reader.tflite"
MANIFEST_FILE = "ask-reader-manifest.json"
REPORT_FILE = "ask-reader-report.json"


@dataclass(frozen=True, slots=True)
class AskReaderConfig:
    questions: int = 40_000
    folds: int = 5
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.003
    token_dim: int = 32
    shape_dim: int = 8
    filters: int = 64
    dropout: float = 0.1
    seed: int = 0
    max_size_bytes: int = 400_000


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
    t = keras.layers.Dense(cfg.filters, activation="relu", name="slot_hidden")(t)
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


def _fit(cfg: AskReaderConfig, questions: list[Question]) -> Any:
    keras = require("keras", "train")
    x, y, k = to_arrays(questions)
    model = build_ask_reader(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss={
            "intent": keras.losses.SparseCategoricalCrossentropy(),
            "slots": keras.losses.SparseCategoricalCrossentropy(),
        },
        jit_compile=True,
    )
    model.fit(
        x,
        {"intent": k, "slots": np.maximum(y, 0)},
        sample_weight={"intent": np.ones(len(k), "float32"), "slots": (y >= 0).astype("float32")},
        epochs=cfg.epochs,
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


def evaluate_unseen_templates(cfg: AskReaderConfig, work: Path) -> dict[str, Any]:
    """Each fold: train on two thirds of every intent's templates, score on the rest."""
    all_ids = set(template_ids())
    results = []
    for i, held in enumerate(folds(cfg.folds, seed=cfg.seed)):
        train = generate(cfg.questions, seed=cfg.seed + 100 + i, templates=all_ids - held)
        test = generate(cfg.questions // 5, seed=cfg.seed + 200 + i, templates=held)
        results.append(score(_fit(cfg, train), test, work / "held-out.tflite"))
    per_intent = {
        name: round(float(np.mean([r["per_intent_accuracy"][name] for r in results])), 4)
        for name in ASK_INTENTS
    }
    return {
        "mean_intent_accuracy": round(float(np.mean([r["intent_accuracy"] for r in results])), 4),
        "mean_slot_f1": {
            name: round(float(np.mean([r["slots"][name]["f1"] for r in results])), 4)
            for name in [*SLOT_LABELS[1:], "all"]
        },
        "mean_per_intent_accuracy": per_intent,
        "folds": results,
    }


def ask_reader_manifest(blob: bytes) -> dict[str, Any]:
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
        "trained_on": "synthetic questions only",
    }


def train_ask_reader(
    cfg: AskReaderConfig, out_dir: str | Path, evaluate: bool = True
) -> dict[str, Any]:
    """Score on held-out templates (unless ``evaluate`` is off), then train on every template
    and export TFLite with its manifest. Returns the report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    unseen = evaluate_unseen_templates(cfg, out) if evaluate else None

    model = _fit(cfg, generate(cfg.questions, seed=cfg.seed))
    blob = to_tflite(model)
    if len(blob) > cfg.max_size_bytes:
        raise ValueError(f"ask reader is {len(blob)} bytes, over {cfg.max_size_bytes}")
    (out / MODEL_FILE).write_bytes(blob)
    seen = score(model, generate(2000, seed=cfg.seed + 999), out / "seen.tflite")

    report = {
        "config": asdict(cfg),
        "parameters": int(model.count_params()),
        "model_size_bytes": len(blob),
        "templates": len(template_ids()),
        "unseen_templates": unseen,
        "seen_templates": seen,
        "note": "Scores are on synthetic questions; real questions will score lower.",
    }
    manifest = ask_reader_manifest(blob)
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

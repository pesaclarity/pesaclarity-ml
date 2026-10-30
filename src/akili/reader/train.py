"""Train, evaluate and export the message reader (needs the [train] extra).

The network: token and shape embeddings, two 1-D convolutions that see two tokens either side,
a dilated convolution for a whole clause, then a label per token and, from the pooled tokens,
the message kind. About 160k parameters; the quantised TFLite file is about 200 KB.

Evaluation holds each sentence layout out of training in turn (see
:func:`evaluate_unseen_layouts`) and scores whole fields: a field counts as read only when every
token of it, and nothing else, carries its label.
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
from akili.reader.synthetic import (
    FIELD_LABELS,
    KIND_LABELS,
    LAYOUTS,
    Field,
    generate,
    to_arrays,
)
from akili.reader.tokens import HASH_BUCKETS, MAX_TOKENS, N_SHAPES, reader_manifest

MODEL_FILE = "reader.tflite"
MANIFEST_FILE = "reader-manifest.json"
REPORT_FILE = "reader-report.json"


@dataclass(frozen=True, slots=True)
class ReaderConfig:
    messages: int = 30_000
    holdout_fraction: float = 0.2
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.003
    token_dim: int = 32
    shape_dim: int = 8
    filters: int = 64
    seed: int = 0
    max_size_bytes: int = 512_000


def build_reader(cfg: ReaderConfig) -> Any:
    keras = require("keras", "train")
    ops = keras.ops
    keras.utils.set_random_seed(cfg.seed)

    inputs = keras.Input(shape=(2 * MAX_TOKENS,), dtype="int32", name="message")
    tokens = inputs[:, :MAX_TOKENS]
    shapes = inputs[:, MAX_TOKENS:]
    x = keras.layers.Concatenate()(
        [
            keras.layers.Embedding(HASH_BUCKETS, cfg.token_dim, name="token_embedding")(tokens),
            keras.layers.Embedding(N_SHAPES, cfg.shape_dim, name="shape_embedding")(shapes),
        ]
    )
    x = keras.layers.Conv1D(cfg.filters, 5, padding="same", activation="relu", name="conv_1")(x)
    y = keras.layers.Conv1D(cfg.filters, 5, padding="same", activation="relu", name="conv_2")(x)
    x = keras.layers.Add()([x, y])
    # Dilated: each token sees about a dozen tokens either side, enough for a whole clause.
    y = keras.layers.Conv1D(
        cfg.filters, 3, padding="same", dilation_rate=4, activation="relu", name="conv_3"
    )(x)
    x = keras.layers.Add()([x, y])
    fields = keras.layers.Dense(len(FIELD_LABELS), activation="softmax", name="fields")(x)

    mask = ops.expand_dims(ops.cast(ops.greater(shapes, 0), "float32"), -1)
    pooled = ops.max(x * mask - (1.0 - mask) * 1e4, axis=1)
    k = keras.layers.Dense(32, activation="relu", name="kind_hidden")(pooled)
    kind = keras.layers.Dense(len(KIND_LABELS), activation="softmax", name="kind")(k)
    return keras.Model(inputs, {"fields": fields, "kind": kind}, name="akili_reader")


def _spans(labels: np.ndarray) -> set[tuple[int, int, int]]:
    """Maximal runs of the same non-OTHER label, as (label, start, end)."""
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


def field_scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-field precision, recall and F1 over exact spans. Padding (-1) is ignored."""
    tp = np.zeros(len(FIELD_LABELS))
    fp = np.zeros(len(FIELD_LABELS))
    fn = np.zeros(len(FIELD_LABELS))
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
    out = {}
    for f in Field:
        if f is Field.OTHER:
            continue
        p = tp[f] / max(tp[f] + fp[f], 1)
        r = tp[f] / max(tp[f] + fn[f], 1)
        out[f.name.lower()] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(2 * p * r / max(p + r, 1e-9)), 4),
            "support": int(tp[f] + fn[f]),
        }
    return out


def _run_tflite(path: Path, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    fields = interpreter.get_tensor(outputs[3]["index"])
    kind = interpreter.get_tensor(outputs[2]["index"])
    return np.asarray(fields), np.asarray(kind)


def _score(model: Any, x: np.ndarray, y: np.ndarray, k: np.ndarray, path: Path) -> dict[str, Any]:
    """Scores of ``model`` exported as TFLite, since that is what the app runs."""
    tf = require("tensorflow", "train")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    path.write_bytes(bytes(converter.convert()))
    try:
        fields_p, kind_p = _run_tflite(path, x)
    finally:
        path.unlink()
    return {
        "kind_accuracy": round(float((kind_p.argmax(-1) == k).mean()), 4),
        "fields": field_scores(y, fields_p.argmax(-1)),
    }


def _fit(cfg: ReaderConfig, messages: list[Any]) -> Any:
    keras = require("keras", "train")
    x, y, k = to_arrays(messages)
    model = build_reader(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss={
            "fields": keras.losses.SparseCategoricalCrossentropy(),
            "kind": keras.losses.SparseCategoricalCrossentropy(),
        },
    )
    model.fit(
        x,
        {"fields": np.maximum(y, 0), "kind": k},
        sample_weight={"fields": (y >= 0).astype("float32"), "kind": np.ones(len(k), "float32")},
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
    )
    return model


def _layout(style: str) -> str | None:
    parts = style.split("|")
    return parts[5] if len(parts) > 6 else None


def evaluate_unseen_layouts(cfg: ReaderConfig, messages: list[Any], work: Path) -> dict[str, Any]:
    """Hold each sentence layout out in turn: train without it, score the TFLite model on it.
    This is the honest measure for wording the review inbox has never seen."""
    per_layout = {}
    for layout in LAYOUTS:
        train = [m for m in messages if _layout(m.style) != layout]
        test = [m for m in messages if _layout(m.style) == layout]
        xt, yt, kt = to_arrays(test)
        per_layout[layout] = _score(_fit(cfg, train), xt, yt, kt, work / "held-out.tflite")
    fields = [f.name.lower() for f in Field if f is not Field.OTHER]
    mean = {
        name: round(
            float(
                np.mean(
                    [
                        r["fields"][name]["f1"]
                        for r in per_layout.values()
                        if r["fields"][name]["support"]
                    ]
                )
            ),
            4,
        )
        for name in fields
    }
    return {
        "mean_f1": mean,
        "mean_kind_accuracy": round(
            float(np.mean([r["kind_accuracy"] for r in per_layout.values()])), 4
        ),
        "per_layout": per_layout,
    }


def train_reader(cfg: ReaderConfig, out_dir: str | Path, evaluate: bool = True) -> dict[str, Any]:
    """Score on held-out layouts (unless ``evaluate`` is off), then train on every message and
    export TFLite with its manifest. Returns the report."""
    tf = require("tensorflow", "train")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    messages = generate(cfg.messages, seed=cfg.seed)
    unseen = evaluate_unseen_layouts(cfg, messages, out) if evaluate else None

    model = _fit(cfg, messages)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    blob = bytes(converter.convert())
    if len(blob) > cfg.max_size_bytes:
        raise ValueError(f"reader is {len(blob)} bytes, over {cfg.max_size_bytes}")
    (out / MODEL_FILE).write_bytes(blob)

    report = {
        "config": asdict(cfg),
        "unseen_layouts": unseen,
        "model_size_bytes": len(blob),
        "note": "Scores are on synthetic messages; real messages will score lower.",
    }
    manifest = {
        "model_file": MODEL_FILE,
        "model_sha256": hashlib.sha256(blob).hexdigest(),
        "model_size_bytes": len(blob),
        **reader_manifest(),
        "outputs": {
            "fields": {"shape": [-1, MAX_TOKENS, len(FIELD_LABELS)], "labels": FIELD_LABELS},
            "kind": {"shape": [-1, len(KIND_LABELS)], "labels": KIND_LABELS},
        },
        "trained_on": "synthetic messages only",
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

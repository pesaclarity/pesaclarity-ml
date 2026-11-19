"""The Keras Ask reader and its TFLite export (needs the [train] extra).

The network is :func:`build_ask_reader`; the config, data, token swaps and scores are
:mod:`akili.ask.reader`'s. Beside the TFLite file goes ``ask-reader.weights``, its tensors read
back out of it in the plain format of docs/WEIGHTS.md (:mod:`akili.ask.reader_weights`): the
app runs the reader on rayo, or in Dart.
"""

from __future__ import annotations

import gc
import hashlib
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from akili._optional import require
from akili.ask.questions import Question, template_ids, to_arrays
from akili.ask.reader import (
    MANIFEST_FILE,
    REPORT_FILE,
    AskReaderConfig,
    fold_questions,
    seen_questions,
    summarise_folds,
    swap_rng,
    swap_tokens,
    training_questions,
)
from akili.ask.reader import probe_scores as _probe_scores
from akili.ask.reader import scores as _scores
from akili.ask.reader_weights import WEIGHTS_FILE, manifest_fields, read_tflite
from akili.ask.schema import ASK_INTENTS, ASK_SCHEMA_VERSION, SLOT_LABELS
from akili.reader.tokens import HASH_BUCKETS, MAX_TOKENS, N_SHAPES, reader_manifest

MODEL_FILE = "ask-reader.tflite"

__all__ = [
    "MANIFEST_FILE",
    "MODEL_FILE",
    "REPORT_FILE",
    "ask_reader_manifest",
    "build_ask_reader",
    "evaluate_unseen_templates",
    "fold_scores",
    "probe_scores",
    "run_tflite",
    "score",
    "to_tflite",
    "train_ask_reader",
]


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
    """:func:`akili.ask.reader.probe_scores` of the TFLite file at ``path``."""
    return _probe_scores(lambda x: run_tflite(path, x))


def _fit(cfg: AskReaderConfig, questions: list[Question]) -> Any:
    keras = require("keras", "train")
    x, y, k = to_arrays(questions)
    rng = swap_rng(cfg)
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
    """Scores of ``model`` exported as TFLite, since that is what the app ran."""
    path.write_bytes(to_tflite(model))
    try:
        return _scores(lambda x: run_tflite(path, x), questions)
    finally:
        path.unlink()


def fold_scores(cfg: AskReaderConfig, i: int, work: Path) -> dict[str, Any]:
    """Fold ``i`` (:func:`akili.ask.reader.fold_questions`): train on the other folds'
    templates and score on this fold's, as ``v1`` to ``v4``. A fold can run in a
    process of its own; :func:`akili.ask.reader.summarise_folds` joins them."""
    train, test = fold_questions(cfg, i)
    model = _fit(cfg, train)
    del train
    out: dict[str, Any] = {
        key: score(model, qs, work / "held-out.tflite") if qs is not None else None
        for key, qs in test.items()
    }
    # Free the fold's model before the next: six trainings in one process otherwise hold
    # several GB.
    del model
    require("keras", "train").backend.clear_session()
    gc.collect()
    return out


def evaluate_unseen_templates(cfg: AskReaderConfig, work: Path) -> dict[str, Any]:
    """Each fold: train on the other folds' templates, score on this fold's."""
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
    file) and their manifest, score it on the probe sets, then (unless ``evaluate`` is off)
    score the recipe on held-out templates. Returns the report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The model to export first, so it is the same with or without the evaluation.
    model = _fit(cfg, training_questions(cfg))
    blob = to_tflite(model)
    if len(blob) > cfg.max_size_bytes:
        raise ValueError(f"ask reader is {len(blob)} bytes, over {cfg.max_size_bytes}")
    (out / MODEL_FILE).write_bytes(blob)
    weights = read_tflite(out / MODEL_FILE).write(out / WEIGHTS_FILE).read_bytes()
    seen = score(model, seen_questions(cfg), out / "seen.tflite")
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

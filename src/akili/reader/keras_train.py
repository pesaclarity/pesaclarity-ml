"""The Keras message reader, exported as TFLite (needs the [train] extra). Not the release path.

The message reader moved to PyTorch (:mod:`akili.reader.train`, pesaclarity/docs decision
0023); this is the Keras model it replaced, kept so the old export can be reproduced until
TensorFlow is removed (stage 5). ``akili reader --keras`` runs it.

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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from akili._optional import require
from akili.reader.synthetic import (
    FIELD_LABELS,
    KIND_LABELS,
    LAYOUTS,
    generate,
    to_arrays,
)
from akili.reader.tokens import HASH_BUCKETS, MAX_TOKENS, N_SHAPES, reader_manifest
from akili.reader.train import (
    MANIFEST_FILE,
    REPORT_FILE,
    ReaderConfig,
    field_scores,
    layout_of,
    mean_scores,
)

MODEL_FILE = "reader.tflite"
__all__ = ["MANIFEST_FILE", "MODEL_FILE", "REPORT_FILE", "ReaderConfig", "train_reader"]


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


def evaluate_unseen_layouts(cfg: ReaderConfig, messages: list[Any], work: Path) -> dict[str, Any]:
    """Hold each sentence layout out in turn: train without it, score the TFLite model on it.
    This is the honest measure for wording the review inbox has never seen."""
    per_layout = {}
    for layout in LAYOUTS:
        train = [m for m in messages if layout_of(m.style) != layout]
        test = [m for m in messages if layout_of(m.style) == layout]
        xt, yt, kt = to_arrays(test)
        per_layout[layout] = _score(_fit(cfg, train), xt, yt, kt, work / "held-out.tflite")
    return mean_scores(per_layout)


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

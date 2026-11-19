"""The Ask reader's plain weights: its layers, the Keras model's file read out, and the forward
pass the app's Dart reader copies.

The app runs the Ask reader on rayo (in Dart where rayo cannot load) from
``ask-reader.weights`` (format: [docs/WEIGHTS.md](../../../docs/WEIGHTS.md)). Since decision
0023 stage 3 the file is the PyTorch model's (:mod:`akili.ask.reader_model`), quantised to
int8 by rayo-convert from its ONNX graph (:func:`akili.ask.reader.export_module`). The Keras
model's (:mod:`akili.ask.keras_reader`) is read out of its ``ask-reader.tflite``
(:func:`read_tflite`), so it is byte for byte the int8 and float32 tensors TFLite ran; both
have the same layers, tensors and layout.

:func:`forward` is the reference forward pass on those weights, in NumPy, the int8 layers run
the way TFLite's hybrid kernels (and rayo) run them: it gives TFLite's outputs to about 1e-5
on the Keras file, and rayo's to about 1e-4 on the torch one. The app's Dart code is a copy
of it.

Reading a TFLite file is rayo-convert's (pesaclarity/rayo, a pinned dependency): it maps the
TFLite graph to rayo's ops and gives its tensors as they are in the file, with no TensorFlow.
:func:`forward` needs only NumPy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from rayo_convert import Options

from akili.export.rayo import weights_of
from akili.export.weights import FORMAT_VERSION, Weights, read_weights
from akili.reader.tokens import MAX_TOKENS

MODEL_NAME = "ask-reader"
WEIGHTS_FILE = "ask-reader.weights"

# The layers in the order they run, as in build_ask_reader. The tensors of layer ``x`` are
# ``x.table`` (embedding) or ``x.kernel`` and ``x.bias``.
LAYERS: list[dict[str, Any]] = [
    {"name": "token_embedding", "kind": "embedding"},
    {"name": "shape_embedding", "kind": "embedding"},
    {"name": "conv_1", "kind": "conv1d", "size": 3, "dilation": 1, "activation": "relu"},
    {"name": "conv_2", "kind": "conv1d", "size": 5, "dilation": 1, "activation": "relu"},
    {"name": "conv_3", "kind": "conv1d", "size": 3, "dilation": 3, "activation": "relu"},
    {"name": "slot_hidden", "kind": "dense", "activation": "relu"},
    {"name": "slot_context", "kind": "conv1d", "size": 5, "dilation": 1, "activation": "relu"},
    {"name": "slot_span", "kind": "conv1d", "size": 3, "dilation": 2, "activation": "relu"},
    {"name": "slots", "kind": "dense", "activation": "softmax"},
    {"name": "intent_hidden", "kind": "dense", "activation": "relu"},
    {"name": "intent", "kind": "dense", "activation": "softmax"},
]


# How rayo-convert presents the reader to rayo: the encoding's two halves as two inputs (token
# ids, then shape ids, each [T]), the slots before the intent.
RAYO_OPTIONS = Options(
    name=MODEL_NAME,
    split_inputs={"question": [("tokens", MAX_TOKENS), ("shapes", MAX_TOKENS)]},
    outputs=["slots", "intent"],
)


def required_tensors() -> list[str]:
    """The tensors of :data:`LAYERS`, each layer's ``.table`` or ``.kernel`` and ``.bias``."""
    return [
        f"{layer['name']}.{part}"
        for layer in LAYERS
        for part in (("table",) if layer["kind"] == "embedding" else ("kernel", "bias"))
    ]


def run_weights(data: bytes, rayo_dir: str | Path | None = None) -> Any:
    """A runner of the weights file ``data`` for :func:`akili.ask.reader.probe_scores`: on
    rayo's C library when ``rayo_dir`` is a rayo checkout, else on :func:`forward`."""
    from akili.export.onnx import RayoRunner

    if rayo_dir:
        from akili.ask.reader import converted_run

        return converted_run(RayoRunner(data, rayo_dir))
    w = read_weights(data)

    def run(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outs = [forward(w, row) for row in np.asarray(x)]
        return np.stack([i for i, _ in outs]), np.stack([s for _, s in outs])

    return run


def read_tflite(path: str | Path) -> Weights:
    """The weights of the ask reader exported at ``path``, with the graph rayo runs them with,
    both from rayo-convert (:mod:`akili.export.rayo`)."""
    weights, _ = weights_of(path, RAYO_OPTIONS, LAYERS, required_tensors())
    return weights


def manifest_fields(blob: bytes) -> dict[str, Any]:
    """The manifest's fields for the weights file ``blob``."""
    return {
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": hashlib.sha256(blob).hexdigest(),
        "weights_size_bytes": len(blob),
        "weights_format_version": FORMAT_VERSION,
    }


def export_weights(model_path: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Write ``ask-reader.weights`` from the TFLite file at ``model_path`` (into ``out_dir``,
    by default beside it) and, when ``ask-reader-manifest.json`` there is the model's, add the
    weights' fields to it. Returns those fields."""
    from akili.ask.reader import MANIFEST_FILE

    out = Path(out_dir) if out_dir is not None else Path(model_path).parent
    weights = read_tflite(model_path)
    blob = weights.to_bytes()
    out.mkdir(parents=True, exist_ok=True)
    (out / WEIGHTS_FILE).write_bytes(blob)
    fields = manifest_fields(blob)
    manifest_path = out / MANIFEST_FILE
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_sha256") == weights.source_sha256:
            manifest.update(fields)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return fields


def _round(x: Any) -> Any:
    """std::round: halves away from zero, where Python's round and np.round go to even."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _quantised(x: np.ndarray) -> np.ndarray:
    """``x``'s round trip through int8 as TFLite's hybrid kernels quantise a float input on the
    fly: one scale and zero point for the whole of ``x``, the range widened to include 0
    (tensor_utils AsymmetricQuantizeFloats)."""
    lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    if lo == hi:
        return np.zeros_like(x)
    scale = (hi - lo) / 255.0
    zp_min, zp_max = -128 - lo / scale, 127 - hi / scale
    zp = zp_min if 128 + abs(lo / scale) < 127 + abs(hi / scale) else zp_max
    zero = -128 if zp <= -128 else 127 if zp >= 127 else int(_round(zp))
    s = np.float32(scale)
    inv = np.float32(1.0) / s
    q = np.clip(_round(np.float32(zero) + (x * np.float64(inv)).astype(np.float32)), -128, 127)
    out: np.ndarray = (q - zero) * np.float64(s)
    return out


def forward(w: Weights, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(intent ``[intents]``, slots ``[MAX_TOKENS, slots]``) for one question's input
    (``int32[2 * MAX_TOKENS]``), as TFLite runs the model."""
    ids = np.asarray(ids)
    t = MAX_TOKENS
    mask = (ids[t:] > 0).astype(np.float32)[:, None]
    n = max(float(mask.sum()), 1.0)
    emb = np.concatenate(
        [
            w.tensors["token_embedding.table"].real()[ids[:t]].astype(np.float64),
            w.tensors["shape_embedding.table"].real()[ids[t:]].astype(np.float64),
        ],
        axis=1,
    )
    spec = {layer["name"]: layer for layer in w.layers}

    def conv(x: np.ndarray, name: str) -> np.ndarray:
        kernel = w.tensors[name + ".kernel"]
        k = kernel.real().astype(np.float64)  # [out, 1, size, in]
        size, dilation = int(spec[name]["size"]), int(spec[name]["dilation"])
        if kernel.scales is not None:
            # TFLite runs a dilated convolution as SPACE_TO_BATCH_ND: each of the `dilation`
            # phases (rows i with i % dilation == p) is a batch row of its own, quantised alone.
            x = x.copy()
            for p in range(dilation):
                x[p::dilation] = _quantised(x[p::dilation])
        pad = dilation * (size - 1) // 2
        xp = np.pad(x, ((pad, pad), (0, 0)))
        y = np.zeros((t, k.shape[0]))
        for j in range(size):
            y += xp[j * dilation : j * dilation + t] @ k[:, 0, j, :].T
        out: np.ndarray = np.maximum(y + w.tensors[name + ".bias"].real(), 0)
        return out

    def dense(x: np.ndarray, name: str) -> np.ndarray:
        kernel = w.tensors[name + ".kernel"]
        rows = np.atleast_2d(x)
        if kernel.scales is not None:
            rows = np.stack([_quantised(r) for r in rows])
        y: np.ndarray = rows @ kernel.real().astype(np.float64).T + w.tensors[name + ".bias"].real()
        if spec[name]["activation"] == "relu":
            y = np.maximum(y, 0)
        else:
            e = np.exp(y - y.max(axis=-1, keepdims=True))
            y = e / e.sum(axis=-1, keepdims=True)
        out: np.ndarray = y.reshape(*x.shape[:-1], -1)
        return out

    x = conv(emb, "conv_1")
    x = x + conv(x, "conv_2")
    x = x + conv(x, "conv_3")
    s = dense(np.concatenate([x, emb], axis=1), "slot_hidden")
    s = s + conv(s, "slot_context")
    s = s + conv(s, "slot_span")
    slots = dense(s, "slots")
    pooled = np.concatenate(
        [
            (x * mask - (1 - mask) * 1e4).max(axis=0),
            (x * mask).sum(axis=0) / n,
            (emb * mask).sum(axis=0) / n,
        ]
    )
    intent = dense(dense(pooled, "intent_hidden"), "intent")
    return intent, slots

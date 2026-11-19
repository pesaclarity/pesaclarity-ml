"""The Ask reader's plain weights: its layers, the Keras model's file read out, and the forward
pass the app's Dart reader copies.

The app runs the Ask reader on rayo (in Dart where rayo cannot load) from
``ask-reader.weights`` (format: [docs/WEIGHTS.md](../../../docs/WEIGHTS.md)). Since decision
0023 stage 3 the file is the PyTorch model's (:mod:`akili.ask.reader_model`), quantised to
int8 by rayo-convert from its ONNX graph (:func:`akili.ask.reader.export_module`). The Keras
model's (:mod:`akili.ask.keras_reader`) is read out of its ``ask-reader.tflite``
(:func:`read_tflite`), so it is byte for byte the int8 and float32 tensors TFLite ran; both
have the same layers, tensors and layout.

:func:`forward` is the reference forward pass on those weights, in NumPy, run as rayo runs
the file: float32 activations, each operation in rayo's order, and the int8 layers quantising
their inputs on the fly as TFLite's hybrid kernels do. Every int8 layer so sees rayo's very
inputs, so no rounding falls the other way, and the outputs match rayo's to about 1e-7 (the
Keras file's TFLite outputs to about 1e-5). The app's Dart code is a copy of it.

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


_F = np.float32


def _quantise_rows(x: np.ndarray) -> tuple[np.ndarray, np.float32, int]:
    """``x`` (float32) quantised to int8 as rayo does it (kernels/quantize.c, TFLite's
    AsymmetricQuantizeFloats): one scale and zero point for all of ``x``, the range widened to
    include 0, the scale and zero point worked out in double and the scale kept as float32, then
    ``zero + x * (1 / scale)`` in float32 rounded half away from zero. Returns (int values,
    scale, zero point)."""
    lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    if lo == hi:
        return np.zeros(x.shape, dtype=np.int64), _F(1.0), 0
    scale = (hi - lo) / 255.0
    zp_min, zp_max = -128.0 - lo / scale, 127.0 - hi / scale
    zp = zp_min if 128.0 + abs(lo / scale) < 127.0 + abs(hi / scale) else zp_max
    if zp <= -128.0:
        zero = -128
    elif zp >= 127.0:
        zero = 127
    else:
        zero = int(np.floor(zp + 0.5)) if zp >= 0 else int(np.ceil(zp - 0.5))
    sf = _F(scale)
    inv = _F(1.0) / sf
    v = (_F(zero) + x.astype(_F) * inv).astype(np.float64)  # each step in float32
    v = np.clip(v, -128.5, 127.5)
    q = np.where(v >= 0, np.trunc(v + 0.5), np.trunc(v - 0.5)).astype(np.int64)
    return np.clip(q, -128, 127), sf, zero


def _finish(
    acc: np.ndarray, zero: np.ndarray, scale: np.ndarray, kernel: Any, bias: Any, *, relu: bool
) -> np.ndarray:
    """rayo's rayo_linear_finish_i8: ``bias + float(acc - zero * rowsum) * (scale * wscale)``,
    each operation in float32, then the ReLU."""
    w8 = kernel.value.reshape(kernel.value.shape[0], -1).astype(np.int64)
    rowsum = w8.sum(axis=1)
    wscale = np.asarray(kernel.scales, dtype=_F).reshape(-1)
    if wscale.size == 1:
        wscale = np.full(w8.shape[0], wscale[0], dtype=_F)
    exact = (acc - zero[:, None] * rowsum[None, :]).astype(_F)
    y: np.ndarray = (
        np.asarray(bias, dtype=_F) + exact * (scale[:, None] * wscale[None, :])
    ).astype(_F)
    return np.maximum(y, _F(0)) if relu else y


def _sum_rows(x: np.ndarray) -> np.ndarray:
    """The sum over axis 0 in float32, in order (rayo's reduce)."""
    out: np.ndarray = x[0].astype(_F).copy()
    for row in x[1:]:
        out = (out + row.astype(_F)).astype(_F)
    return out


def _softmax(y: np.ndarray) -> np.ndarray:
    """rayo's softmax: exp and the sum in double, scaled by a float32."""
    e = np.exp((y - y.max(axis=-1, keepdims=True)).astype(_F).astype(np.float64)).astype(_F)
    inv = (1.0 / e.astype(np.float64).sum(axis=-1, keepdims=True)).astype(_F)
    out: np.ndarray = (e * inv).astype(_F)
    return out


def _embed(w: Weights, ids: np.ndarray) -> np.ndarray:
    """The token embedding (int8 rows times their scale, in float32) beside the shape one."""
    t = MAX_TOKENS
    table = w.tensors["token_embedding.table"]
    if table.scales is not None:
        scales = np.asarray(table.scales, dtype=_F).reshape(-1)
        rows = table.value[ids[:t]].astype(np.int64) - int(table.zero_point or 0)
        scale = scales[0] if scales.size == 1 else scales[ids[:t], None]
        tokens = (rows.astype(_F) * scale).astype(_F)
    else:
        tokens = table.value[ids[:t]].astype(_F)
    shapes = w.tensors["shape_embedding.table"].real().astype(_F)[ids[t:]]
    return np.concatenate([tokens, shapes], axis=1)


def _conv(w: Weights, layer: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """rayo's conv1d (ops/conv1d.c): each of the ``dilation`` phases (positions with
    ``i % dilation == p``) quantised alone, padding taps at the zero point, one int8 product."""
    name = layer["name"]
    kernel, bias = w.tensors[name + ".kernel"], w.tensors[name + ".bias"].real()
    size, dilation = int(layer["size"]), int(layer["dilation"])
    t, c = x.shape
    half = (size - 1) // 2 * dilation
    if kernel.scales is None:
        k = kernel.real().astype(np.float64)
        xp = np.pad(x.astype(np.float64), ((half, half), (0, 0)))
        y = np.zeros((t, k.shape[0]))
        for j in range(size):
            y += xp[j * dilation : j * dilation + t] @ k[:, 0, j, :].T
        out: np.ndarray = np.maximum((y + bias).astype(_F), _F(0))
        return out
    q = np.zeros((t, c), dtype=np.int64)
    scale = np.zeros(t, dtype=_F)
    zero = np.zeros(t, dtype=np.int64)
    for p in range(min(dilation, t)):
        q[p::dilation], scale[p::dilation], zero[p::dilation] = _quantise_rows(x[p::dilation])
    cols = np.empty((t, size, c), dtype=np.int64)
    for j in range(size):
        src = np.arange(t) + j * dilation - half
        inside = ((src >= 0) & (src < t))[:, None]
        cols[:, j] = np.where(inside, q[np.clip(src, 0, t - 1)], zero[:, None])
    w8 = kernel.value.reshape(kernel.value.shape[0], -1).astype(np.int64)
    return _finish(cols.reshape(t, -1) @ w8.T, zero, scale, kernel, bias, relu=True)


def _dense(w: Weights, layer: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """rayo's dense (ops/dense.c): each row quantised alone for an int8 kernel; then the ReLU
    or the softmax."""
    name = layer["name"]
    kernel, bias = w.tensors[name + ".kernel"], w.tensors[name + ".bias"].real()
    rows = np.atleast_2d(x).astype(_F)
    relu = layer["activation"] == "relu"
    if kernel.scales is not None:
        parts = [_quantise_rows(r) for r in rows]
        q = np.stack([p[0] for p in parts])
        scale = np.array([p[1] for p in parts], dtype=_F)
        zero = np.array([p[2] for p in parts], dtype=np.int64)
        y = _finish(q @ kernel.value.astype(np.int64).T, zero, scale, kernel, bias, relu=relu)
    else:
        product = (rows.astype(np.float64) @ kernel.real().astype(np.float64).T).astype(_F)
        y = (product + np.asarray(bias, dtype=_F)).astype(_F)
        if relu:
            y = np.maximum(y, _F(0))
    if not relu:
        y = _softmax(y)
    out: np.ndarray = y.reshape(*x.shape[:-1], -1)
    return out


def forward(w: Weights, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(intent ``[intents]``, slots ``[MAX_TOKENS, slots]``) for one question's input
    (``int32[2 * MAX_TOKENS]``), as rayo runs the file: every activation in float32 and each
    operation in rayo's order, so every int8 layer quantises rayo's very inputs. The intent
    matches rayo's bit for bit; the slots, whose last layer is a float32 product, to about
    1e-7. pesaclarity-app's Dart reader is a copy of this."""
    ids = np.asarray(ids)
    spec = {layer["name"]: layer for layer in w.layers}
    emb = _embed(w, ids)
    x = _conv(w, spec["conv_1"], emb)
    x = (x + _conv(w, spec["conv_2"], x)).astype(_F)
    x = (x + _conv(w, spec["conv_3"], x)).astype(_F)
    s = _dense(w, spec["slot_hidden"], np.concatenate([x, emb], axis=1))
    s = (s + _conv(w, spec["slot_context"], s)).astype(_F)
    s = (s + _conv(w, spec["slot_span"], s)).astype(_F)
    slots = _dense(w, spec["slots"], s)
    # The pooling, as the graph's mul, sub, reduce and div nodes run it.
    mask = (ids[MAX_TOKENS:] > 0).astype(_F)[:, None]
    n = np.maximum(_sum_rows(mask), _F(1.0))
    masked = (x * mask).astype(_F)
    off = ((_F(1.0) - mask) * _F(1e4)).astype(_F)
    pooled = np.concatenate(
        [
            (masked - off).astype(_F).max(axis=0),
            (_sum_rows(masked) / n).astype(_F),
            (_sum_rows((emb * mask).astype(_F)) / n).astype(_F),
        ]
    )
    intent = _dense(w, spec["intent"], _dense(w, spec["intent_hidden"], pooled))
    return intent.astype(np.float64), slots.astype(np.float64)

"""The ask reader's plain weights, read out of its exported TFLite file.

The app runs the ask reader in Dart, not TFLite, and reads its weights from
``ask-reader.weights`` (format: [docs/WEIGHTS.md](../../../docs/WEIGHTS.md)). They are read
out of the very ``ask-reader.tflite`` that :mod:`akili.ask.reader` exports, so they are byte
for byte the int8 and float32 tensors TFLite runs, not a second conversion of the Keras model.

:func:`forward` is the reference forward pass on those weights, in NumPy: the layers of
:func:`akili.ask.reader.build_ask_reader` run the way TFLite runs this dynamic-range-quantised
model, so it gives TFLite's outputs to about 1e-5. The app's Dart code is a copy of it.

Reading the file needs the [train] extra (TensorFlow's interpreter); :func:`forward` needs only
NumPy.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from akili._optional import require
from akili.export.weights import FORMAT_VERSION, Tensor, Weights
from akili.reader.tokens import MAX_TOKENS

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


def rayo_graph() -> dict[str, Any]:
    """The graph rayo (the app's native runtime) runs these weights with: LAYERS wired as
    build_ask_reader wires them, each int8 layer run as TFLite runs it (docs/WEIGHTS.md,
    "The rayo graph"). Inputs are the encoding's two halves, token ids and shape ids."""

    def conv(x: str, name: str, out: str) -> dict[str, Any]:
        spec = next(layer for layer in LAYERS if layer["name"] == name)
        return {
            "op": "conv1d",
            "inputs": [x, f"{name}.kernel", f"{name}.bias"],
            "output": out,
            "size": spec["size"],
            "dilation": spec["dilation"],
            "activation": "relu",
            "input_quantization": "tensor_phases",
        }

    def dense(x: str, name: str, out: str, activation: str) -> dict[str, Any]:
        return {
            "op": "dense",
            "inputs": [x, f"{name}.kernel", f"{name}.bias"],
            "output": out,
            "activation": activation,
            "input_quantization": "rows",
        }

    def op(kind: str, inputs: list[str], out: str, **attrs: Any) -> dict[str, Any]:
        return {"op": kind, "inputs": inputs, "output": out, **attrs}

    nodes = [
        op("gather", ["token_embedding.table", "tokens"], "tok"),
        op("gather", ["shape_embedding.table", "shapes"], "shp"),
        op("concat", ["tok", "shp"], "emb"),
        op("compare", ["shapes"], "mask", comparison="greater", value=0),
        conv("emb", "conv_1", "c1"),
        conv("c1", "conv_2", "c2"),
        op("add", ["c1", "c2"], "x2"),
        conv("x2", "conv_3", "c3"),
        op("add", ["x2", "c3"], "x"),
        op("concat", ["x", "emb"], "xe"),
        dense("xe", "slot_hidden", "sh", "relu"),
        conv("sh", "slot_context", "sc"),
        op("add", ["sh", "sc"], "s2"),
        conv("s2", "slot_span", "ss"),
        op("add", ["s2", "ss"], "s3"),
        dense("s3", "slots", "slot_logits", "none"),
        op("softmax", ["slot_logits"], "slots"),
        op("masked_reduce", ["x", "mask"], "pmax", reduction="max", fill=-10000),
        op("masked_reduce", ["x", "mask"], "pmean", reduction="mean"),
        op("masked_reduce", ["emb", "mask"], "emean", reduction="mean"),
        op("concat", ["pmax", "pmean"], "p1"),
        op("concat", ["p1", "emean"], "pooled"),
        dense("pooled", "intent_hidden", "ih", "relu"),
        dense("ih", "intent", "intent_logits", "none"),
        op("softmax", ["intent_logits"], "intent"),
    ]
    return {
        "version": 1,
        "symbols": {"T": MAX_TOKENS},
        "inputs": [
            {"name": "tokens", "dtype": "int32", "shape": ["T"]},
            {"name": "shapes", "dtype": "int32", "shape": ["T"]},
        ],
        "nodes": nodes,
        "outputs": [{"name": "slots", "node": "slots"}, {"name": "intent", "node": "intent"}],
    }


def _tensor(detail: dict[str, Any], value: np.ndarray) -> Tensor:
    if value.dtype.name != "int8":
        return Tensor(value.astype(np.float32))
    q = detail["quantization_parameters"]
    scales = np.asarray(q["scales"], dtype=np.float32)
    zero_points = np.asarray(q["zero_points"])
    if zero_points.size and np.any(zero_points != zero_points.flat[0]):
        raise ValueError(f"{detail['name']}: zero points differ by channel")
    return Tensor(
        value.copy(),
        scales,
        zero_point=int(zero_points.flat[0]) if zero_points.size else 0,
        axis=int(q["quantized_dimension"]) if scales.size > 1 else None,
    )


def read_tflite(path: str | Path) -> Weights:
    """The weights of the ask reader exported at ``path``."""
    tf = require("tensorflow", "train")
    blob = Path(path).read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    by_index = {d["index"]: d for d in details}
    ops = interpreter._get_ops_details()  # the graph's op list (private API)
    tensors: dict[str, Tensor] = {}

    for layer in LAYERS:
        name = layer["name"]
        prefix = f"akili_ask_reader_1/{name}_1/"
        if layer["kind"] == "embedding":
            tables = [
                d
                for d in details
                if d["name"].startswith(prefix + "GatherV2/ReadVariableOp") and len(d["shape"]) == 2
            ]
            if len(tables) != 1:
                raise ValueError(f"{name}: {len(tables)} tables in the graph")
            d = tables[0]
            tensors[name + ".table"] = _tensor(d, interpreter.get_tensor(d["index"]))
            continue
        # The convolution or dense op whose kernel is a constant named after the layer.
        layer_ops = [
            o
            for o in ops
            if o["op_name"] in ("CONV_2D", "FULLY_CONNECTED")
            and by_index[o["inputs"][1]]["name"].startswith(prefix)
        ]
        if len(layer_ops) != 1:
            raise ValueError(f"{name}: {len(layer_ops)} kernels in the graph")
        op = layer_ops[0]
        d = by_index[int(op["inputs"][1])]
        tensors[name + ".kernel"] = _tensor(d, interpreter.get_tensor(d["index"]))
        # The bias is the op's own, or (dilated convolutions, Tensordot denses) added after it
        # in a separate ADD; the two never both hold one, but summing them is safe.
        biases = []
        if op["inputs"][2] >= 0:
            biases.append(interpreter.get_tensor(int(op["inputs"][2])))
        for d in details:
            if (
                d["name"] in (prefix + "Squeeze", prefix + "BiasAdd/ReadVariableOp")
                and len(d["shape"]) == 1
            ):
                biases.append(interpreter.get_tensor(d["index"]))
        if not biases or any(b.dtype.name != "float32" for b in biases):
            raise ValueError(f"{name}: no float32 bias in the graph")
        tensors[name + ".bias"] = Tensor(np.sum(biases, axis=0, dtype=np.float32))

    return Weights(
        model="ask-reader",
        source_file=Path(path).name,
        source_sha256=hashlib.sha256(blob).hexdigest(),
        source_size_bytes=len(blob),
        layers=LAYERS,
        tensors=tensors,
        graph=rayo_graph(),
    )


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

"""The ask writer's plain weights, read out of its exported TFLite file, for rayo.

The app runs the ask writer with rayo, its own native runtime (pesaclarity/rayo), and reads its
weights from ``ask-writer.weights`` (format: [docs/WEIGHTS.md](../../../docs/WEIGHTS.md), with
the graph rayo runs them with). They are read out of the very ``ask-writer.tflite`` that
:mod:`akili.ask.writer` exports, so they are the int8 and float32 values TFLite runs.

The graph decodes word by word: each run takes the new tokens at positions ``P`` .. ``P+T-1``,
keeps every layer's keys and values in a cache, and gives the logits after the last token.

:func:`forward` is the reference on those weights, in NumPy: the layers of
:func:`akili.ask.writer.build_writer` run the way TFLite's built-in kernels run this
dynamic-range-quantised model, to about 1e-5. Reading the file needs the [train] extra
(TensorFlow's interpreter); :func:`forward` needs only NumPy.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from akili._optional import require
from akili.ask.reader_weights import _quantised, _tensor
from akili.export.weights import FORMAT_VERSION, Tensor, Weights

WEIGHTS_FILE = "ask-writer.weights"
_PREFIX = "akili_ask_writer_1/"


def layers(n_layers: int) -> list[dict[str, Any]]:
    """The layers in the order they run, as in build_writer."""
    out: list[dict[str, Any]] = [
        {"name": "token_embedding", "kind": "embedding"},
        {"name": "position_embedding", "kind": "embedding"},
    ]
    for i in range(n_layers):
        out += [
            {"name": f"ln_{i}_a", "kind": "layer_norm"},
            {"name": f"attention_{i}", "kind": "causal_self_attention"},
            {"name": f"ln_{i}_b", "kind": "layer_norm"},
            {"name": f"ff_{i}_in", "kind": "dense", "activation": "gelu_erf"},
            {"name": f"ff_{i}_out", "kind": "dense"},
        ]
    return [
        *out,
        {"name": "ln_final", "kind": "layer_norm"},
        {"name": "logits", "kind": "dense"},
    ]


def rayo_graph(n_layers: int, heads: int, dim: int, length: int) -> dict[str, Any]:
    """The graph rayo runs the writer with, one step of a word-by-word decode."""

    def op(kind: str, inputs: list[str], out: str, **attrs: Any) -> dict[str, Any]:
        return {"op": kind, "inputs": inputs, "output": out, **attrs}

    def dense(x: str, name: str, out: str, **attrs: Any) -> dict[str, Any]:
        return op(
            "dense", [x, f"{name}.kernel", f"{name}.bias"], out, input_quantization="rows", **attrs
        )

    def norm(x: str, name: str, out: str) -> dict[str, Any]:
        return op("layer_norm", [x, f"{name}.gamma", f"{name}.beta"], out, epsilon=1e-5)

    nodes = [
        op("range", [], "pos", start="P", size="T"),
        op("gather", ["token_embedding.table", "tokens"], "tok"),
        op("gather", ["position_embedding.table", "pos"], "at"),
        op("add", ["tok", "at"], "x0"),
    ]
    x = "x0"
    for i in range(n_layers):
        a = f"attention_{i}"
        nodes += [
            norm(x, f"ln_{i}_a", f"a{i}"),
            dense(f"a{i}", f"{a}.query", f"q{i}"),
            dense(f"a{i}", f"{a}.key", f"k{i}"),
            dense(f"a{i}", f"{a}.value", f"v{i}"),
            op("cache_write", [f"keys_{i}", f"k{i}"], f"ks{i}", at="P"),
            op("cache_write", [f"values_{i}", f"v{i}"], f"vs{i}", at="P"),
            op("attention", [f"q{i}", f"ks{i}", f"vs{i}"], f"h{i}", heads=heads, causal=1),
            dense(f"h{i}", f"{a}.out", f"o{i}"),
            op("add", [x, f"o{i}"], f"y{i}"),
            norm(f"y{i}", f"ln_{i}_b", f"b{i}"),
            dense(f"b{i}", f"ff_{i}_in", f"f{i}", activation="gelu_erf"),
            dense(f"f{i}", f"ff_{i}_out", f"g{i}"),
            op("add", [f"y{i}", f"g{i}"], f"x{i + 1}"),
        ]
        x = f"x{i + 1}"
    nodes += [
        # Every op from here works on each position alone, so only the last one is needed.
        op("last_row", [x], "last"),
        norm("last", "ln_final", "final"),
        dense("final", "logits", "logits"),
    ]
    return {
        "version": 1,
        "symbols": {"T": length, "P": length - 1},
        # P + T is at most the length: plan for the longest run, the whole sequence at once.
        "plan": {"P": 0},
        "inputs": [{"name": "tokens", "dtype": "int32", "shape": ["T"]}],
        "states": [
            {"name": f"{kind}_{i}", "dtype": "float32", "shape": [length, dim]}
            for i in range(n_layers)
            for kind in ("keys", "values")
        ],
        "nodes": nodes,
        "outputs": [{"name": "logits", "node": "logits"}],
    }


def _rows(t: Tensor, start: int, stop: int) -> Tensor:
    """Rows ``start`` .. ``stop - 1`` of a [out, in] kernel or an [out] bias."""
    if t.scales is None:
        return Tensor(t.value[start:stop].copy())
    scales = t.scales if t.axis is None else t.scales[start:stop]
    return Tensor(t.value[start:stop].copy(), scales.copy(), t.zero_point, t.axis)


def read_tflite(path: str | Path, n_layers: int, heads: int) -> Weights:
    """The weights of the ask writer exported at ``path``. Its attention's one ``qkv`` dense
    layer becomes three (``query``, ``key``, ``value``): the same rows, so the same products."""
    tf = require("tensorflow", "train")
    blob = Path(path).read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        interpreter = tf.lite.Interpreter(model_content=blob)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    by_name = {d["name"]: d for d in details}
    by_index = {d["index"]: d for d in details}
    ops = interpreter._get_ops_details()  # the graph's op list (private API)
    tensors: dict[str, Tensor] = {}

    def named(name: str) -> Tensor:
        d = by_name.get(_PREFIX + name)
        if d is None:
            raise ValueError(f"no tensor {name} in the graph")
        return _tensor(d, interpreter.get_tensor(d["index"]))

    tensors["token_embedding.table"] = named("token_embedding_1/GatherV2/ReadVariableOp")
    position = named("position_embedding_1/strided_slice")
    tensors["position_embedding.table"] = Tensor(position.value[0].copy())
    length, dim = tensors["position_embedding.table"].value.shape
    for layer in layers(n_layers):
        if layer["kind"] == "layer_norm":
            # TFLite: x * (rsqrt(var + eps) * gamma) + (beta - mean * ...): gamma in Reshape.
            name = layer["name"]
            tensors[name + ".gamma"] = Tensor(named(name + "_1/Reshape").value.reshape(-1).copy())
            tensors[name + ".beta"] = Tensor(named(name + "_1/Reshape_1").value.reshape(-1).copy())
    for o in ops:
        if o["op_name"] != "FULLY_CONNECTED":
            continue
        kernel_detail = by_index[int(o["inputs"][1])]
        bias_detail = by_index[int(o["inputs"][2])]
        layer = kernel_detail["name"][len(_PREFIX) :].split("/Tensordot")[0]
        layer = layer.replace("_1/", ".").removesuffix("_1")  # attention_0.qkv, ff_0_in, ...
        kernel = _tensor(kernel_detail, interpreter.get_tensor(kernel_detail["index"]))
        bias = Tensor(interpreter.get_tensor(bias_detail["index"]).reshape(-1).astype(np.float32))
        if layer.endswith(".qkv"):
            base = layer.removesuffix(".qkv")
            for j, part in enumerate(("query", "key", "value")):
                tensors[f"{base}.{part}.kernel"] = _rows(kernel, j * dim, (j + 1) * dim)
                tensors[f"{base}.{part}.bias"] = _rows(bias, j * dim, (j + 1) * dim)
        else:
            tensors[layer + ".kernel"] = kernel
            tensors[layer + ".bias"] = bias
    return Weights(
        model="ask-writer",
        source_file=Path(path).name,
        source_sha256=hashlib.sha256(blob).hexdigest(),
        source_size_bytes=len(blob),
        layers=layers(n_layers),
        tensors=tensors,
        graph=rayo_graph(n_layers, heads, dim, length),
    )


def manifest_fields(blob: bytes) -> dict[str, Any]:
    """The manifest's fields for the weights file ``blob``."""
    return {
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": hashlib.sha256(blob).hexdigest(),
        "weights_size_bytes": len(blob),
        "weights_format_version": FORMAT_VERSION,
    }


def export_weights(
    model_path: str | Path, out_dir: str | Path | None = None, *, n_layers: int, heads: int
) -> dict[str, Any]:
    """Write ``ask-writer.weights`` from the TFLite file at ``model_path`` (into ``out_dir``,
    by default beside it) and, when ``ask-writer-manifest.json`` there is the model's, add the
    weights' fields to it. Returns those fields."""
    from akili.ask.writer import MANIFEST_FILE

    out = Path(out_dir) if out_dir is not None else Path(model_path).parent
    weights = read_tflite(model_path, n_layers, heads)
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


def forward(w: Weights, tokens: np.ndarray, heads: int) -> np.ndarray:
    """Logits ``[len(tokens), vocab]`` for every position, as TFLite's built-in kernels run the
    writer: each dense layer's input quantised to int8 per row first."""
    t = {name: (x.real() if x.scales is not None else x.value) for name, x in w.tensors.items()}
    real = {name: np.asarray(v, dtype=np.float64) for name, v in t.items()}
    n = len(tokens)
    x = real["token_embedding.table"][tokens] + real["position_embedding.table"][:n]

    def norm(x: np.ndarray, name: str) -> np.ndarray:
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        out: np.ndarray = (x - mu) / np.sqrt(var + 1e-5) * real[name + ".gamma"] + real[
            name + ".beta"
        ]
        return out

    def dense(x: np.ndarray, name: str) -> np.ndarray:
        q = np.stack([_quantised(r) for r in x])
        out: np.ndarray = q @ real[name + ".kernel"].T + real[name + ".bias"]
        return out

    gelu = np.vectorize(lambda v: 0.5 * v * (1 + erf(v / sqrt(2))))
    dim = x.shape[1]
    hd = dim // heads
    causal = np.tril(np.ones((n, n))) > 0
    n_layers = sum(1 for layer in w.layers if layer["kind"] == "causal_self_attention")
    for i in range(n_layers):
        a = norm(x, f"ln_{i}_a")
        q, k, v = (dense(a, f"attention_{i}.{p}") for p in ("query", "key", "value"))
        h = np.zeros_like(q)
        for j in range(heads):
            s = slice(j * hd, (j + 1) * hd)
            sc = np.where(causal, q[:, s] @ k[:, s].T / np.sqrt(hd), -1e9)
            e = np.exp(sc - sc.max(-1, keepdims=True))
            h[:, s] = (e / e.sum(-1, keepdims=True)) @ v[:, s]
        x = x + dense(h, f"attention_{i}.out")
        b = norm(x, f"ln_{i}_b")
        x = x + dense(gelu(dense(b, f"ff_{i}_in")), f"ff_{i}_out")
    out: np.ndarray = dense(norm(x, "ln_final"), "logits")
    return out

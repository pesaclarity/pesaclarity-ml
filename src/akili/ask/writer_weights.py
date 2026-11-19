"""The ask writer's plain weights, read out of its exported TFLite file, for rayo.

The app runs the ask writer with rayo, its own native runtime (pesaclarity/rayo), and reads its
weights from ``ask-writer.weights`` (format: [docs/WEIGHTS.md](../../../docs/WEIGHTS.md), with
the graph rayo runs them with). They are read out of the very ``ask-writer.tflite`` that
:mod:`akili.ask.writer` exports, so they are the int8 and float32 values TFLite runs.

rayo-convert (pesaclarity/rayo, a pinned dependency) reads the file and writes the graph: it
recognises the TFLite graph's causal self-attention and rewrites it for a key/value cache, so
the graph decodes word by word: each run takes the new tokens at positions ``P`` ..
``P+T-1``, keeps every layer's keys and values in a cache, and gives the logits after the last
token. No TensorFlow is needed.

:func:`forward` is the reference on those weights, in NumPy: the layers of
:func:`akili.ask.writer.build_writer` run the way TFLite's built-in kernels run this
dynamic-range-quantised model, to about 1e-5. It needs only NumPy.
"""

from __future__ import annotations

import hashlib
import json
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from rayo_convert import Options

from akili.ask.reader_weights import _quantised
from akili.export.rayo import weights_of
from akili.export.weights import FORMAT_VERSION, Weights

WEIGHTS_FILE = "ask-writer.weights"


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


# How rayo-convert presents the writer to rayo: the logits under their name. The attention is
# rewritten for a key/value cache, so rayo decodes word by word (rayo docs/CONVERTING.md).
RAYO_OPTIONS = Options(name="ask-writer", rename={"output_0": "logits"})


def _required(n_layers: int) -> list[str]:
    out = []
    for layer in layers(n_layers):
        name, kind = layer["name"], layer["kind"]
        if kind == "embedding":
            out.append(f"{name}.table")
        elif kind == "layer_norm":
            out += [f"{name}.gamma", f"{name}.beta"]
        elif kind == "causal_self_attention":
            out += [
                f"{name}.{p}.{t}"
                for p in ("query", "key", "value", "out")
                for t in ("kernel", "bias")
            ]
        else:
            out += [f"{name}.kernel", f"{name}.bias"]
    return out


def read_tflite(path: str | Path, n_layers: int, heads: int) -> Weights:
    """The weights of the ask writer exported at ``path``, with the graph rayo decodes them
    with, both from rayo-convert (:mod:`akili.export.rayo`). Its attention's one ``qkv`` dense
    layer becomes three (``query``, ``key``, ``value``): the same rows, so the same products.
    Raises ValueError unless the graph decodes with a cache in ``n_layers`` attention layers
    of ``heads`` heads."""
    weights, result = weights_of(path, RAYO_OPTIONS, layers(n_layers), _required(n_layers))
    attention = [n for n in result.graph["nodes"] if n["op"] == "attention"]
    if (
        result.mode != "decode"
        or len(attention) != n_layers
        or any(n["heads"] != heads or n["causal"] != 1 for n in attention)
    ):
        raise ValueError(
            f"{Path(path).name}: rayo-convert did not give {n_layers} cached attention layers "
            f"of {heads} heads: " + "; ".join(result.notes)
        )
    return weights


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

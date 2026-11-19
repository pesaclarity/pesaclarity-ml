"""A model's plain weights and rayo graph, from rayo-convert's conversion of its ONNX file.

rayo (pesaclarity/rayo), the app's native runtime, ships ``rayo-convert``: it reads the ONNX
file a torch model was exported to (:mod:`akili.export.onnx`), maps its ops to rayo's,
quantises the weights to int8 and writes the graph that runs them (rayo's docs/CONVERTING.md).
It is a pinned dependency (pyproject.toml, by rayo commit).

This module turns its result into :class:`akili.export.weights.Weights`: the same tensors
under the names rayo-convert gives them (the layers' own: ``conv_1.kernel``,
``hidden.bias``), the graph, and this repository's ``layers`` list, which the app's Dart reader
checks.
"""

from __future__ import annotations

import hashlib
from typing import Any

from rayo_convert import Options, Result

from akili.export.weights import Tensor, Weights


def weights_from_result(
    blob: bytes,
    file_name: str,
    result: Result,
    *,
    options: Options,
    layers: list[dict[str, Any]],
    required: list[str],
) -> Weights:
    """``Weights`` for rayo-convert's ``result`` of the model file ``blob`` (named
    ``file_name``: the ONNX file a torch model was exported to)."""
    tensors: dict[str, Tensor] = {}
    for w in result.weights:
        if w.scales is None:
            tensors[w.name] = Tensor(w.data)
        else:
            tensors[w.name] = Tensor(w.data, w.scales, zero_point=w.zero_point, axis=w.axis)
    missing = [n for n in required if n not in tensors]
    if missing:
        raise ValueError(
            f"{file_name}: rayo-convert gave no {', '.join(missing)}; "
            "is this the model these layers describe?"
        )
    return Weights(
        model=str(options.name),
        source_file=file_name,
        source_sha256=hashlib.sha256(blob).hexdigest(),
        source_size_bytes=len(blob),
        layers=layers,
        tensors=tensors,
        graph=result.graph,
    )

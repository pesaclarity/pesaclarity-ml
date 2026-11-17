"""A model's plain weights and rayo graph, converted from its TFLite file by rayo-convert.

rayo (pesaclarity/rayo), the app's native runtime, ships ``rayo-convert``: it reads a TFLite
file, maps its ops to rayo's, keeps the int8 weights as they are in the file, and writes the
graph that runs them (rayo's docs/CONVERTING.md). It is a pinned dependency (pyproject.toml,
by rayo commit) and needs neither TensorFlow nor TFLite's interpreter, so exporting the weights
of an exported model needs only the core install.

This module turns its result into :class:`akili.export.weights.Weights`: the same tensors
under the names rayo-convert gives them (the Keras layers': ``conv_1.kernel``,
``attention_0.query.bias``), the graph, and this repository's ``layers`` list, which the app's
Dart reader checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rayo_convert import Options, Result, convert

from akili.export.weights import Tensor, Weights


def convert_tflite(path: str | Path, options: Options) -> tuple[bytes, Result]:
    """The TFLite file at ``path`` and rayo-convert's conversion of it."""
    blob = Path(path).read_bytes()
    return blob, convert(blob, Path(path).name, options)


def weights_of(
    path: str | Path,
    options: Options,
    layers: list[dict[str, Any]],
    required: list[str],
) -> tuple[Weights, Result]:
    """``Weights`` for the TFLite file at ``path``, converted with ``options``; every tensor
    named in ``required`` must be among the converted ones (the layers this repository says
    the model has)."""
    blob, result = convert_tflite(path, options)
    tensors: dict[str, Tensor] = {}
    for w in result.weights:
        if w.scales is None:
            tensors[w.name] = Tensor(w.data)
        else:
            tensors[w.name] = Tensor(w.data, w.scales, zero_point=w.zero_point, axis=w.axis)
    missing = [n for n in required if n not in tensors]
    if missing:
        raise ValueError(
            f"{Path(path).name}: rayo-convert gave no {', '.join(missing)}; "
            "is this the model these layers describe?"
        )
    weights = Weights(
        model=str(options.name),
        source_file=Path(path).name,
        source_sha256=hashlib.sha256(blob).hexdigest(),
        source_size_bytes=len(blob),
        layers=layers,
        tensors=tensors,
        graph=result.graph,
    )
    return weights, result

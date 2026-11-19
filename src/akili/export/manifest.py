"""The model manifest, ``model-manifest.json``, written next to every exported categoriser.

The app reads it before loading a model and refuses one whose feature schema version or
feature config it does not implement, or whose file hash does not match.

The PyTorch categoriser's (:func:`write_weights_manifest`) names its rayo ``categoriser.weights``
(``weights_file``, ``weights_sha256``, ``weights_size_bytes``, ``weights_format_version``) and
the ONNX file it was converted from (``source``), as the message reader's and the Ask reader's
manifests do, and the dense layers the app may fine-tune (``fine_tune``). (The Keras
categoriser's named a TFLite file, ``model_file``, and its trainable copy, ``trainable``; both
went with TensorFlow in decision 0023, stage 5.)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from akili.config import FeatureConfig
from akili.features import feature_manifest

MANIFEST_NAME = "model-manifest.json"


def write_weights_manifest(
    weights_path: Path,
    onnx: bytes,
    onnx_name: str,
    features: FeatureConfig,
    fine_tune: dict[str, Any] | None = None,
) -> Path:
    """Write ``model-manifest.json`` beside ``weights_path`` (a ``.weights`` file converted from
    the ONNX file ``onnx``, named ``onnx_name``) and return its path."""
    from akili.export.weights import FORMAT_VERSION, read_weights

    data = weights_path.read_bytes()
    graph = read_weights(data).graph or {}
    doc: dict[str, Any] = {
        "weights_file": weights_path.name,
        "weights_sha256": hashlib.sha256(data).hexdigest(),
        "weights_size_bytes": len(data),
        "weights_format_version": FORMAT_VERSION,
        "source": {
            "file": onnx_name,
            "sha256": hashlib.sha256(onnx).hexdigest(),
            "size_bytes": len(onnx),
        },
        **feature_manifest(features),
        "rayo_inputs": {
            i["name"]: {"dtype": i["dtype"], "shape": i["shape"]} for i in graph.get("inputs", [])
        },
        "outputs": [o["name"] for o in graph.get("outputs", [])],
    }
    if fine_tune is not None:
        doc["fine_tune"] = fine_tune
    path = weights_path.with_name(MANIFEST_NAME)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path

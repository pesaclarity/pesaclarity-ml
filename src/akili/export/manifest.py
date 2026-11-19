"""The model manifest, ``model-manifest.json``, written next to every exported categoriser.

The app reads it before loading a model and refuses one whose feature schema version or
feature config it does not implement, or whose file hash does not match.

The PyTorch categoriser's (:func:`write_weights_manifest`) names its rayo ``categoriser.weights``
(``weights_file``, ``weights_sha256``, ``weights_size_bytes``, ``weights_format_version``) and
the ONNX file it was converted from (``source``), as the message reader's and the Ask reader's
manifests do, and the dense layers the app may fine-tune (``fine_tune``). The Keras one's
(:func:`write_model_manifest`, ``--keras``) names the TFLite file (``model_file``) and the
trainable TFLite copy (``trainable``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from akili.config import FeatureConfig, FineTuneConfig
from akili.export.size import ExportedModel
from akili.export.trainable import SIGNATURES
from akili.features import feature_manifest

MANIFEST_NAME = "model-manifest.json"


@dataclass(frozen=True)
class TrainableManifest:
    """The trainable model beside the main one, and how the app should fine-tune it."""

    exported: ExportedModel
    weight_shapes: dict[str, list[int]]
    fine_tune: FineTuneConfig


def write_model_manifest(
    exported: ExportedModel, features: FeatureConfig, trainable: TrainableManifest | None = None
) -> Path:
    """Write ``model-manifest.json`` beside ``exported.path`` and return its path."""
    doc: dict[str, object] = {
        "model_file": exported.path.name,
        "model_sha256": exported.sha256,
        "model_size_bytes": exported.size_bytes,
        **feature_manifest(features),
    }
    if trainable is not None:
        doc["trainable"] = {
            "file": trainable.exported.path.name,
            "sha256": trainable.exported.sha256,
            "size_bytes": trainable.exported.size_bytes,
            "signatures": list(SIGNATURES),
            "weights": trainable.weight_shapes,
            "fine_tune": trainable.fine_tune.model_dump(),
        }
    path = exported.path.with_name(MANIFEST_NAME)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


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

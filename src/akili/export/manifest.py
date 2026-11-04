"""The model manifest written next to every exported TFLite file.

The app reads it before loading a model and refuses one whose feature schema version or
feature config it does not implement, or whose file hash does not match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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

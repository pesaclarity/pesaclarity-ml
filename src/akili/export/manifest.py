"""The model manifest written next to every exported TFLite file.

The app reads it before loading a model and refuses one whose feature schema version or
feature config it does not implement, or whose file hash does not match.
"""

from __future__ import annotations

import json
from pathlib import Path

from akili.config import FeatureConfig
from akili.export.size import ExportedModel
from akili.features import feature_manifest

MANIFEST_NAME = "model-manifest.json"


def write_model_manifest(exported: ExportedModel, features: FeatureConfig) -> Path:
    """Write ``model-manifest.json`` beside ``exported.path`` and return its path."""
    doc = {
        "model_file": exported.path.name,
        "model_sha256": exported.sha256,
        "model_size_bytes": exported.size_bytes,
        **feature_manifest(features),
    }
    path = exported.path.with_name(MANIFEST_NAME)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path

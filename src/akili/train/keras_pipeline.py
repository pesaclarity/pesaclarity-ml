"""The Keras categoriser as a :class:`~akili.train.backend.Backend` (needs the [train] extra).

Kept for ``akili train --keras`` (and ``audit``, ``simulate``) until TensorFlow is removed
(decision 0023, stage 5). It trains the Keras model (:mod:`akili.keras_model`) with Adam or
Keras DP-SGD (:mod:`akili.train.keras_dp_sgd`), exports ``categoriser.tflite`` (dynamic-range
int8) and the trainable ``categoriser-train.tflite`` with the manifest, and the audit scores the
TFLite files.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from akili._optional import require
from akili.config import ExperimentConfig
from akili.export.manifest import TrainableManifest, write_model_manifest
from akili.export.size import ExportedModel
from akili.export.tflite import export_tflite, run_tflite, to_tflite_bytes
from akili.federated.client import LocalTrainer
from akili.federated.updates import Weights
from akili.keras_model import build_categoriser
from akili.train.backend import Exported, FitOutput, FloatArray, IntArray

MODEL_FILE = "categoriser.tflite"


class KerasBackend:
    name = "keras"

    def fit(
        self,
        cfg: ExperimentConfig,
        x: IntArray,
        y: IntArray,
        *,
        users: npt.ArrayLike,
        seed: int,
        n_classes: int,
        noise_seed: int | None = None,
    ) -> FitOutput:
        from akili.train.keras_dp_sgd import train_dp_sgd
        from akili.train.pipeline import privacy_report

        keras = require("keras", "train")
        model = build_categoriser(cfg.features, cfg.model, n_classes, seed=seed)
        if cfg.dp.enabled:
            stats = train_dp_sgd(
                model, x, y, cfg.training, cfg.dp, users=users, noise_seed=noise_seed
            )
            return FitOutput(
                model=model, losses=[stats.final_loss], privacy=privacy_report(cfg, stats)
            )
        model.compile(
            optimizer=keras.optimizers.Adam(cfg.training.learning_rate),
            loss="sparse_categorical_crossentropy",
        )
        history = model.fit(
            x, y, epochs=cfg.training.epochs, batch_size=cfg.training.batch_size, verbose=0
        )
        return FitOutput(model=model, losses=[float(v) for v in history.history["loss"]])

    def predict(self, model: Any, x: IntArray) -> FloatArray:
        return np.asarray(model.predict(x, verbose=0), dtype=np.float32)

    def export(self, model: Any, cfg: ExperimentConfig, run_dir: Path) -> Exported:
        exported = export_tflite(model, run_dir / MODEL_FILE, cfg.export)
        files = [exported.path]
        metrics = {"model.size_bytes": float(exported.size_bytes)}
        trainable: TrainableManifest | None = None
        if cfg.export.trainable:
            from akili.export.trainable import TRAINABLE_NAME, export_trainable, weight_shapes

            trainable = TrainableManifest(
                exported=export_trainable(model, run_dir / TRAINABLE_NAME, cfg.export),
                weight_shapes=weight_shapes(model),
                fine_tune=cfg.export.fine_tune,
            )
            metrics["model.trainable_size_bytes"] = float(trainable.exported.size_bytes)
            files.append(trainable.exported.path)
        files.append(write_model_manifest(exported, cfg.features, trainable))
        return Exported(model=exported, files=files, metrics=metrics)

    def run_exported(self, exported: ExportedModel, x: IntArray) -> FloatArray:
        return run_tflite(exported.path, x)

    def shipped_predict(self, model: Any, cfg: ExperimentConfig, x: IntArray) -> FloatArray:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shadow.tflite"
            # The converter prints a SavedModel summary per call; keep the CLI output readable.
            with contextlib.redirect_stdout(io.StringIO()):
                path.write_bytes(to_tflite_bytes(model, quantize=cfg.export.quantize))
            return run_tflite(path, x)

    def new_model(self, cfg: ExperimentConfig, n_classes: int, seed: int) -> Any:
        return build_categoriser(cfg.features, cfg.model, n_classes, seed=seed)

    def get_weights(self, model: Any) -> Weights:
        return [np.asarray(w, dtype=np.float32) for w in model.get_weights()]

    def set_weights(self, model: Any, weights: Weights) -> None:
        model.set_weights(weights)

    def local_trainer(self, model: Any) -> Callable[[IntArray, IntArray, int], LocalTrainer]:
        from akili.federated.keras_trainer import KerasTrainer, SharedKerasModel

        shared = SharedKerasModel(model)

        def make(x: IntArray, y: IntArray, seed: int) -> LocalTrainer:
            return KerasTrainer(shared, x, y, seed=seed)

        return make

"""Centralised training pipeline: synthetic data, train (plain or DP-SGD), export, track."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from akili._optional import require
from akili.categories import default_categories
from akili.config import ExperimentConfig
from akili.export.size import ExportedModel
from akili.export.tflite import export_tflite
from akili.features import encode_batch
from akili.model import build_categoriser
from akili.privacy.accounting import PrivacyReport, dp_sgd_report
from akili.synthetic import generate
from akili.tracking import Tracker
from akili.train.dp_sgd import train_dp_sgd


@dataclass(frozen=True)
class TrainResult:
    accuracy: float
    exported: ExportedModel
    privacy: PrivacyReport | None
    run_dir: Path
    run_id: str | None


def run_centralised(cfg: ExperimentConfig) -> TrainResult:
    """Train on synthetic data and write the TFLite model and reports to ``cfg.run_dir``."""
    keras = require("keras", "train")
    cats = default_categories()
    batch = encode_batch(generate(cfg.data, cats), cfg.features, cats.labels)

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(batch.labels))
    n_val = max(1, int(len(order) * cfg.training.validation_fraction))
    val, train = order[:n_val], order[n_val:]
    x_train, y_train = batch.features[train], batch.labels[train]
    x_val, y_val = batch.features[val], batch.labels[val]

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_categoriser(cfg.features, cfg.model, len(cats.labels), seed=cfg.seed)
    privacy: PrivacyReport | None = None

    with Tracker(cfg.tracking, cfg.name) as tracker:
        tracker.set_tags({"track": cfg.track.value, "dp": str(cfg.dp.enabled).lower()})
        tracker.log_params(cfg.flat_params())

        if cfg.dp.enabled:
            stats = train_dp_sgd(model, x_train, y_train, cfg.training, cfg.dp, seed=cfg.seed)
            privacy = dp_sgd_report(
                n_examples=len(y_train),
                batch_size=cfg.training.batch_size,
                epochs=cfg.training.epochs,
                noise_multiplier=cfg.dp.noise_multiplier,
                l2_norm_clip=cfg.dp.l2_norm_clip,
                delta=cfg.dp.target_delta,
            )
            if stats.steps != privacy.steps:
                raise RuntimeError("DP-SGD steps do not match the privacy accounting")
            tracker.log_metrics({"train.final_loss": stats.final_loss})
            tracker.log_privacy(privacy)
            (run_dir / "privacy-report.json").write_text(
                privacy.model_dump_json(indent=2), encoding="utf-8"
            )
        else:
            model.compile(
                optimizer=keras.optimizers.Adam(cfg.training.learning_rate),
                loss="sparse_categorical_crossentropy",
            )
            history = model.fit(
                x_train,
                y_train,
                epochs=cfg.training.epochs,
                batch_size=cfg.training.batch_size,
                verbose=0,
            )
            for epoch, loss in enumerate(history.history["loss"]):
                tracker.log_metrics({"train.loss": float(loss)}, step=epoch)

        probs = model.predict(x_val, verbose=0)
        accuracy = float((probs.argmax(axis=1) == y_val).mean())
        tracker.log_metrics({"val.accuracy": accuracy})

        exported = export_tflite(model, run_dir / "categoriser.tflite", cfg.export)
        tracker.log_metrics({"model.size_bytes": float(exported.size_bytes)})
        tracker.log_artifact(exported.path)

        labels_path = run_dir / "labels.json"
        labels_path.write_text(
            json.dumps({"version": cats.version, "labels": list(cats.labels)}, indent=2),
            encoding="utf-8",
        )
        tracker.log_artifact(labels_path)

        return TrainResult(
            accuracy=accuracy,
            exported=exported,
            privacy=privacy,
            run_dir=run_dir,
            run_id=tracker.run_id,
        )

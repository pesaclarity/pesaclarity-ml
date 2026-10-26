"""Centralised training pipeline: synthetic data, train (plain or DP-SGD), export, audit, track."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from akili._optional import require
from akili.categories import default_categories
from akili.config import ExperimentConfig
from akili.export.manifest import write_model_manifest
from akili.export.size import ExportedModel
from akili.export.tflite import export_tflite
from akili.features import encode_batch
from akili.model import build_categoriser
from akili.privacy.accounting import PrivacyReport, dp_sgd_report
from akili.privacy.audit import AuditReport
from akili.synthetic import generate
from akili.tracking import Tracker
from akili.train.dp_sgd import train_dp_sgd

IntArray = npt.NDArray[np.int32]


@dataclass(frozen=True)
class TrainResult:
    accuracy: float
    exported: ExportedModel
    privacy: PrivacyReport | None
    run_dir: Path
    run_id: str | None
    audit: AuditReport | None = None


@dataclass(frozen=True)
class FitOutput:
    model: Any
    losses: list[float]
    """Per-epoch training loss without DP; the final loss only with DP-SGD."""
    dp_steps: int | None


def split_indices(n: int, cfg: ExperimentConfig) -> tuple[npt.NDArray[np.int64], ...]:
    """(train, validation) indices; validation examples are never trained on."""
    order = np.random.default_rng(cfg.seed).permutation(n)
    n_val = max(1, int(n * cfg.training.validation_fraction))
    return order[n_val:], order[:n_val]


def fit_categoriser(
    cfg: ExperimentConfig, x: IntArray, y: IntArray, *, seed: int, n_classes: int
) -> FitOutput:
    """Build and train one categoriser exactly as the config says (plain or DP-SGD).

    The membership-inference audit trains its shadow models with this too, so they match the
    target's training procedure.
    """
    keras = require("keras", "train")
    model = build_categoriser(cfg.features, cfg.model, n_classes, seed=seed)
    if cfg.dp.enabled:
        stats = train_dp_sgd(model, x, y, cfg.training, cfg.dp, seed=seed)
        return FitOutput(model=model, losses=[stats.final_loss], dp_steps=stats.steps)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.training.learning_rate),
        loss="sparse_categorical_crossentropy",
    )
    history = model.fit(
        x, y, epochs=cfg.training.epochs, batch_size=cfg.training.batch_size, verbose=0
    )
    return FitOutput(model=model, losses=[float(v) for v in history.history["loss"]], dp_steps=None)


def run_centralised(
    cfg: ExperimentConfig, *, progress: Callable[[str], None] | None = None
) -> TrainResult:
    """Train on synthetic data and write the TFLite model and reports to ``cfg.run_dir``.

    With ``cfg.audit.enabled`` the trained model is then attacked (see
    :mod:`akili.privacy.mia`) and ``audit-report.json`` is written next to it.
    """
    cats = default_categories()
    batch = encode_batch(generate(cfg.data, cats), cfg.features, cats.labels)

    train, val = split_indices(len(batch.labels), cfg)
    x_train, y_train = batch.features[train], batch.labels[train]
    x_val, y_val = batch.features[val], batch.labels[val]

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    privacy: PrivacyReport | None = None
    audit: AuditReport | None = None

    with Tracker(cfg.tracking, cfg.name) as tracker:
        tracker.set_tags({"track": cfg.track.value, "dp": str(cfg.dp.enabled).lower()})
        tracker.log_params(cfg.flat_params())

        fit = fit_categoriser(cfg, x_train, y_train, seed=cfg.seed, n_classes=len(cats.labels))
        model = fit.model
        if cfg.dp.enabled:
            privacy = dp_sgd_report(
                n_examples=len(y_train),
                batch_size=cfg.training.batch_size,
                epochs=cfg.training.epochs,
                noise_multiplier=cfg.dp.noise_multiplier,
                l2_norm_clip=cfg.dp.l2_norm_clip,
                delta=cfg.dp.target_delta,
            )
            if fit.dp_steps != privacy.steps:
                raise RuntimeError("DP-SGD steps do not match the privacy accounting")
            tracker.log_metrics({"train.final_loss": fit.losses[-1]})
            tracker.log_privacy(privacy)
            (run_dir / "privacy-report.json").write_text(
                privacy.model_dump_json(indent=2), encoding="utf-8"
            )
        else:
            for epoch, loss in enumerate(fit.losses):
                tracker.log_metrics({"train.loss": loss}, step=epoch)

        probs = model.predict(x_val, verbose=0)
        accuracy = float((probs.argmax(axis=1) == y_val).mean())
        tracker.log_metrics({"val.accuracy": accuracy})

        exported = export_tflite(model, run_dir / "categoriser.tflite", cfg.export)
        tracker.log_metrics({"model.size_bytes": float(exported.size_bytes)})
        tracker.log_artifact(exported.path)
        tracker.log_artifact(write_model_manifest(exported, cfg.features))

        labels_path = run_dir / "labels.json"
        labels_path.write_text(
            json.dumps({"version": cats.version, "labels": list(cats.labels)}, indent=2),
            encoding="utf-8",
        )
        tracker.log_artifact(labels_path)

        if cfg.audit.enabled:
            from akili.privacy.mia import run_audit

            audit = run_audit(
                cfg,
                batch.features,
                batch.labels,
                train,
                val,
                target=model,
                exported=exported.path,
                epsilon=privacy.epsilon if privacy else None,
                progress=progress,
            )
            audit_path = run_dir / "audit-report.json"
            audit_path.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
            tracker.log_audit(audit)

        return TrainResult(
            accuracy=accuracy,
            exported=exported,
            privacy=privacy,
            run_dir=run_dir,
            run_id=tracker.run_id,
            audit=audit,
        )

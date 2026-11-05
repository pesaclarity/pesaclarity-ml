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
from akili.export.manifest import TrainableManifest, write_model_manifest
from akili.export.size import ExportedModel
from akili.export.tflite import export_tflite
from akili.features import encode_batch
from akili.model import build_categoriser
from akili.privacy.accounting import PrivacyReport, PrivacyUnit, dp_sgd_report
from akili.privacy.audit import AuditReport
from akili.privacy.ledger import BudgetTotal, Ledger
from akili.synthetic import generate
from akili.tracking import Tracker
from akili.train.dp_sgd import DPTrainingStats, train_dp_sgd

IntArray = npt.NDArray[np.int32]


@dataclass(frozen=True)
class TrainResult:
    accuracy: float
    exported: ExportedModel
    privacy: PrivacyReport | None
    run_dir: Path
    run_id: str | None
    audit: AuditReport | None = None
    budget: BudgetTotal | None = None
    """Everything recorded in the ledger for this data, this run and its audit included."""


@dataclass(frozen=True)
class FitOutput:
    model: Any
    losses: list[float]
    """Per-epoch training loss without DP; the final loss only with DP-SGD."""
    privacy: PrivacyReport | None = None


def split_indices(
    users: npt.ArrayLike, cfg: ExperimentConfig
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """(train, validation) row indices. Whole people are held out, so no one in validation was
    trained on, and validation measures accuracy for new people."""
    ids = np.asarray(users)
    people = np.unique(ids)
    order = np.random.default_rng(cfg.seed).permutation(len(people))
    n_val = max(1, int(len(people) * cfg.training.validation_fraction))
    held_out = np.isin(ids, people[order[:n_val]])
    return np.flatnonzero(~held_out), np.flatnonzero(held_out)


def privacy_report(cfg: ExperimentConfig, stats: DPTrainingStats) -> PrivacyReport:
    """The accountant's budget for a DP-SGD run, checked against what training really did."""
    per_step = cfg.dp.users_per_step if cfg.dp.unit is PrivacyUnit.USER else cfg.training.batch_size
    report = dp_sgd_report(
        population=stats.population,
        per_step=per_step,
        epochs=cfg.training.epochs,
        noise_multiplier=cfg.dp.noise_multiplier,
        l2_norm_clip=cfg.dp.l2_norm_clip,
        noise_seeded=stats.noise_seeded,
        unit=stats.unit,
        delta=cfg.dp.target_delta,
    )
    if stats.steps != report.steps:
        raise RuntimeError("DP-SGD steps do not match the privacy accounting")
    if stats.sampling_rate != report.sampling_rate:
        raise RuntimeError("DP-SGD sampling rate does not match the privacy accounting")
    return report


def fit_categoriser(
    cfg: ExperimentConfig,
    x: IntArray,
    y: IntArray,
    *,
    users: npt.ArrayLike,
    seed: int,
    n_classes: int,
    noise_seed: int | None = None,
) -> FitOutput:
    """Build and train one categoriser exactly as the config says (plain or DP-SGD).

    ``seed`` sets the starting weights only. DP noise comes from OS entropy unless
    ``noise_seed`` is given. The membership-inference audit trains its shadow models with this
    too, so they match the target's training procedure.
    """
    keras = require("keras", "train")
    model = build_categoriser(cfg.features, cfg.model, n_classes, seed=seed)
    if cfg.dp.enabled:
        stats = train_dp_sgd(model, x, y, cfg.training, cfg.dp, users=users, noise_seed=noise_seed)
        return FitOutput(model=model, losses=[stats.final_loss], privacy=privacy_report(cfg, stats))
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.training.learning_rate),
        loss="sparse_categorical_crossentropy",
    )
    history = model.fit(
        x, y, epochs=cfg.training.epochs, batch_size=cfg.training.batch_size, verbose=0
    )
    return FitOutput(model=model, losses=[float(v) for v in history.history["loss"]])


def _write_budget(
    cfg: ExperimentConfig, ledger: Ledger, privacy: PrivacyReport | None, tracker: Tracker
) -> BudgetTotal | None:
    """The data's total after this run, written to ``privacy-budget.json`` for the gate."""
    budget = ledger.total(cfg.dataset_id, privacy.unit) if privacy else None
    if budget is not None:
        tracker.log_metrics({"privacy.budget_epsilon": budget.epsilon})
        (cfg.run_dir / "privacy-budget.json").write_text(
            budget.model_dump_json(indent=2), encoding="utf-8"
        )
    return budget


def run_centralised(
    cfg: ExperimentConfig, *, progress: Callable[[str], None] | None = None
) -> TrainResult:
    """Train on synthetic data and write the TFLite model and reports to ``cfg.run_dir``.

    With ``cfg.audit.enabled`` the trained model is then attacked (see
    :mod:`akili.privacy.mia`) and ``audit-report.json`` is written next to it.
    """
    cats = default_categories()
    txs = generate(cfg.data, cats)
    batch = encode_batch(txs, cfg.features, cats.labels)
    users = np.array([tx.user_id or f"row-{i}" for i, tx in enumerate(txs)])

    train, val = split_indices(users, cfg)
    x_train, y_train = batch.features[train], batch.labels[train]
    x_val, y_val = batch.features[val], batch.labels[val]

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    privacy: PrivacyReport | None = None
    audit: AuditReport | None = None
    ledger = Ledger(cfg.ledger_path)

    with Tracker(cfg.tracking, cfg.name) as tracker:
        tracker.set_tags({"track": cfg.track.value, "dp": str(cfg.dp.enabled).lower()})
        tracker.log_params(cfg.flat_params())

        fit = fit_categoriser(
            cfg,
            x_train,
            y_train,
            users=users[train],
            seed=cfg.seed,
            n_classes=len(cats.labels),
            noise_seed=cfg.dp.noise_seed,
        )
        model = fit.model
        if fit.privacy is not None:
            privacy = fit.privacy
            ledger.record(cfg.dataset_id, privacy, run=cfg.name, run_id=tracker.run_id)
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
        trainable: TrainableManifest | None = None
        if cfg.export.trainable:
            from akili.export.trainable import TRAINABLE_NAME, export_trainable, weight_shapes

            trainable = TrainableManifest(
                exported=export_trainable(model, run_dir / TRAINABLE_NAME, cfg.export),
                weight_shapes=weight_shapes(model),
                fine_tune=cfg.export.fine_tune,
            )
            tracker.log_metrics(
                {"model.trainable_size_bytes": float(trainable.exported.size_bytes)}
            )
            tracker.log_artifact(trainable.exported.path)
        tracker.log_artifact(write_model_manifest(exported, cfg.features, trainable))

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
                users=users,
                target=model,
                exported=exported.path,
                epsilon=privacy.epsilon if privacy else None,
                ledger=ledger,
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
            budget=_write_budget(cfg, ledger, privacy, tracker),
        )

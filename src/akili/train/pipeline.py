"""Centralised training pipeline: synthetic data, train (plain or DP-SGD), export, audit, track.

Written for a :class:`~akili.train.backend.Backend`: the PyTorch categoriser
(:mod:`akili.train.torch_backend`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from akili._optional import require
from akili.categories import default_categories
from akili.config import ExperimentConfig
from akili.export.size import ExportedModel
from akili.features import encode_batch
from akili.privacy.accounting import PrivacyReport, PrivacyUnit, dp_sgd_report
from akili.privacy.audit import AuditReport
from akili.privacy.ledger import BudgetTotal, Ledger
from akili.synthetic import generate
from akili.tracking import Tracker
from akili.train.backend import Backend, FitOutput, accuracy
from akili.train.dp_sgd import DPTrainingStats

__all__ = ["FitOutput", "TrainResult", "backend_for", "run_centralised"]


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
    backend: str = "torch"
    scores: dict[str, Any] = field(default_factory=dict)
    """Validation accuracy of the trained model and of the shipped file, and what the export
    measured (``export``)."""


def backend_for() -> Backend:
    """The PyTorch backend (the [torch] extra)."""
    require("torch", "torch")
    from akili.train.torch_backend import TorchBackend

    return TorchBackend()


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
    cfg: ExperimentConfig,
    *,
    progress: Callable[[str], None] | None = None,
    backend: Backend | None = None,
) -> TrainResult:
    """Train on synthetic data and write the model that ships and its reports to
    ``cfg.run_dir``.

    With ``cfg.audit.enabled`` the trained model is then attacked (see
    :mod:`akili.privacy.mia`) and ``audit-report.json`` is written next to it.
    """
    impl = backend or backend_for()
    say = progress or (lambda _msg: None)
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
        tracker.set_tags(
            {"track": cfg.track.value, "dp": str(cfg.dp.enabled).lower(), "backend": impl.name}
        )
        tracker.log_params(cfg.flat_params())

        say(f"training ({impl.name})")
        fit = impl.fit(
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

        val_accuracy = accuracy(impl.predict(model, x_val), y_val)
        tracker.log_metrics({"val.accuracy": val_accuracy})

        say("exporting")
        exported = impl.export(model, cfg, run_dir)
        tracker.log_metrics(exported.metrics)
        for path in exported.files:
            tracker.log_artifact(path)
        # The same people scored on the file that ships, as the app runs it.
        shipped_accuracy = accuracy(impl.run_exported(exported.model, x_val), y_val)
        tracker.log_metrics({"val.accuracy_shipped": shipped_accuracy})

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
                exported=exported.model,
                epsilon=privacy.epsilon if privacy else None,
                ledger=ledger,
                progress=progress,
                backend=impl,
            )
            audit_path = run_dir / "audit-report.json"
            audit_path.write_text(audit.model_dump_json(indent=2), encoding="utf-8")
            tracker.log_audit(audit)

        return TrainResult(
            accuracy=val_accuracy,
            exported=exported.model,
            privacy=privacy,
            run_dir=run_dir,
            run_id=tracker.run_id,
            audit=audit,
            budget=_write_budget(cfg, ledger, privacy, tracker),
            backend=impl.name,
            scores={
                "val_accuracy": val_accuracy,
                "val_accuracy_shipped": shipped_accuracy,
                "export": exported.report,
            },
        )

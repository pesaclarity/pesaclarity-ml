"""Federated simulation on synthetic users with the Keras trainer (needs the [train] extra)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from akili.categories import default_categories
from akili.config import ExperimentConfig
from akili.export.manifest import write_model_manifest
from akili.export.size import ExportedModel
from akili.export.tflite import export_tflite
from akili.features import encode_batch
from akili.federated.client import CategoriserClient, Consent
from akili.federated.keras_trainer import KerasTrainer, SharedKerasModel
from akili.federated.simulation import SimulationResult, simulate
from akili.model import build_categoriser
from akili.synthetic import commercial_opt_ins, generate
from akili.tracking import Tracker
from akili.types import Transaction


@dataclass(frozen=True)
class FederatedRunResult:
    simulation: SimulationResult
    accuracy: float
    exported: ExportedModel
    run_dir: Path


def run_federated(cfg: ExperimentConfig) -> FederatedRunResult:
    """Simulate federated training; held-out synthetic users measure accuracy."""
    if cfg.federated is None:
        raise ValueError("config has no federated section")
    fed = cfg.federated
    cats = default_categories()
    by_user: dict[str, list[Transaction]] = defaultdict(list)
    for tx in generate(cfg.data, cats):
        by_user[tx.user_id or "unknown"].append(tx)

    users = sorted(by_user)
    rng = np.random.default_rng(cfg.seed)
    n_holdout = max(1, int(len(users) * cfg.training.validation_fraction))
    holdout = set(rng.choice(users, size=n_holdout, replace=False).tolist())
    train_users = [u for u in users if u not in holdout]

    opted_in = commercial_opt_ins(cfg.data, train_users)
    consents = {
        u: Consent(improve_app=True, contribute_licensed_models=u in opted_in) for u in train_users
    }
    model = build_categoriser(cfg.features, cfg.model, len(cats.labels), seed=cfg.seed)
    shared = SharedKerasModel(model)
    encoded = {u: encode_batch(by_user[u], cfg.features, cats.labels) for u in train_users}

    def make_client(cid: str) -> CategoriserClient:
        data = encoded[cid]
        seed = int(rng.integers(1 << 31))
        trainer = KerasTrainer(shared, data.features, data.labels, seed=seed)
        return CategoriserClient(trainer, consents[cid], fed, seed=seed)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with Tracker(cfg.tracking, cfg.name) as tracker:
        tracker.set_tags({"track": cfg.track.value, "mode": "federated-simulation"})
        tracker.log_params(cfg.flat_params())
        sim = simulate(cfg, model.get_weights(), consents, make_client)
        for log in sim.rounds:
            tracker.log_metrics(
                {"round.participants": log.participants, "round.update_norm": log.mean_update_norm},
                step=log.round,
            )
        model.set_weights(sim.weights)
        if sim.privacy is not None:
            tracker.log_privacy(sim.privacy)
            (run_dir / "privacy-report.json").write_text(
                sim.privacy.model_dump_json(indent=2), encoding="utf-8"
            )

        held = encode_batch(
            [tx for u in sorted(holdout) for tx in by_user[u]], cfg.features, cats.labels
        )
        probs = model.predict(held.features, verbose=0)
        accuracy = float((probs.argmax(axis=1) == held.labels).mean())
        tracker.log_metrics({"val.accuracy": accuracy})
        exported = export_tflite(model, run_dir / "categoriser.tflite", cfg.export)
        tracker.log_artifact(exported.path)
        tracker.log_artifact(write_model_manifest(exported, cfg.features))

    return FederatedRunResult(simulation=sim, accuracy=accuracy, exported=exported, run_dir=run_dir)

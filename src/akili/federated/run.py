"""Federated simulation on synthetic users, with the PyTorch trainer.

The weights go between the simulated server and its clients as NumPy arrays in the order and
layout Keras' ``get_weights()`` gave, as between pesaclarity-backend's aggregator and the phones
(:meth:`akili.train.backend.Backend.get_weights`)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from akili.categories import default_categories
from akili.config import ExperimentConfig
from akili.export.size import ExportedModel
from akili.features import encode_batch
from akili.federated.client import CategoriserClient, Consent
from akili.federated.simulation import SimulationResult, simulate
from akili.synthetic import commercial_opt_ins, generate
from akili.tracking import Tracker
from akili.train.backend import Backend, accuracy
from akili.types import Transaction


@dataclass(frozen=True)
class FederatedRunResult:
    simulation: SimulationResult
    accuracy: float
    exported: ExportedModel
    run_dir: Path
    shipped_accuracy: float | None = None
    """Held-out accuracy of the file that ships, as the app runs it."""
    backend: str = "torch"


def run_federated(cfg: ExperimentConfig, backend: Backend | None = None) -> FederatedRunResult:
    """Simulate federated training; held-out synthetic users measure accuracy."""
    from akili.train.pipeline import backend_for

    impl = backend or backend_for()
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
    model = impl.new_model(cfg, len(cats.labels), cfg.seed)
    trainer_for = impl.local_trainer(model)
    encoded = {u: encode_batch(by_user[u], cfg.features, cats.labels) for u in train_users}

    def make_client(cid: str) -> CategoriserClient:
        data = encoded[cid]
        seed = int(rng.integers(1 << 31))
        trainer = trainer_for(data.features, data.labels, seed)
        return CategoriserClient(trainer, consents[cid], fed, seed=seed)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    with Tracker(cfg.tracking, cfg.name) as tracker:
        tracker.set_tags(
            {"track": cfg.track.value, "mode": "federated-simulation", "backend": impl.name}
        )
        tracker.log_params(cfg.flat_params())
        sim = simulate(cfg, impl.get_weights(model), consents, make_client)
        for log in sim.rounds:
            tracker.log_metrics(
                {"round.participants": log.participants, "round.update_norm": log.mean_update_norm},
                step=log.round,
            )
        impl.set_weights(model, sim.weights)
        if sim.privacy is not None:
            tracker.log_privacy(sim.privacy)
            (run_dir / "privacy-report.json").write_text(
                sim.privacy.model_dump_json(indent=2), encoding="utf-8"
            )

        held = encode_batch(
            [tx for u in sorted(holdout) for tx in by_user[u]], cfg.features, cats.labels
        )
        val_accuracy = accuracy(impl.predict(model, held.features), held.labels)
        tracker.log_metrics({"val.accuracy": val_accuracy})
        # The federated model is not fine-tuned on the phone from this export.
        export_cfg = cfg.model_copy(
            update={"export": cfg.export.model_copy(update={"trainable": False})}
        )
        exported = impl.export(model, export_cfg, run_dir)
        tracker.log_metrics(exported.metrics)
        for path in exported.files:
            tracker.log_artifact(path)
        shipped_accuracy = accuracy(impl.run_exported(exported.model, held.features), held.labels)
        tracker.log_metrics({"val.accuracy_shipped": shipped_accuracy})

    return FederatedRunResult(
        simulation=sim,
        accuracy=val_accuracy,
        exported=exported.model,
        run_dir=run_dir,
        shipped_accuracy=shipped_accuracy,
        backend=impl.name,
    )

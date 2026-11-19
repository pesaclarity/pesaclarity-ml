"""The federated simulation with the PyTorch trainer: it learns, and the clients send the
weights in the order and layout pesaclarity-backend's aggregator holds (Keras' get_weights()
order, dense kernels [in, out]), whichever framework trains them. Needs the [torch] extra."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")

from akili.config import (
    ExperimentConfig,
    FederatedConfig,
    SyntheticDataConfig,
    TrackingConfig,
)
from akili.federated.client import CategoriserClient, Consent
from akili.federated.run import run_federated
from akili.model import WEIGHTS_FILE
from akili.train.torch_backend import TorchBackend

pytestmark = pytest.mark.torch


def test_federated_simulation_learns_without_noise(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        name="tiny-fl",
        output_dir=tmp_path,
        data=SyntheticDataConfig(n_users=150, transactions_per_user=30),
        federated=FederatedConfig(
            rounds=15,
            clients_per_round=30,
            local_epochs=2,
            local_learning_rate=1.0,
            update_clip_norm=3.0,
            update_noise_multiplier=0.0,
        ),
        tracking=TrackingConfig(enabled=False),
    )
    result = run_federated(cfg)
    assert result.backend == "torch"
    assert result.simulation.eligible_clients == 120
    assert result.accuracy > 0.2
    assert result.shipped_accuracy == pytest.approx(result.accuracy, abs=0.02)
    assert result.exported.path.name == WEIGHTS_FILE
    assert result.exported.path.exists()
    assert [w.shape for w in result.simulation.weights][6:8] == [(36, 32), (32,)]


def test_a_client_sends_the_keras_layout() -> None:
    """One client's round: the global weights in, a clipped update out, both as the server's
    arrays."""
    backend = TorchBackend()
    cfg = ExperimentConfig(name="x", federated=FederatedConfig(update_noise_multiplier=0.0))
    model = backend.new_model(cfg, 27, seed=0)
    start = backend.get_weights(model)
    rng = np.random.default_rng(0)
    x = np.c_[rng.integers(1, 4096, (40, 8)), np.zeros((40, 5), int)].astype(np.int32)
    y = rng.integers(0, 27, 40).astype(np.int32)
    trainer = backend.local_trainer(model)(x, y, 1)
    assert cfg.federated is not None
    client = CategoriserClient(trainer, Consent(), cfg.federated, seed=1)
    new, n, metrics = client.fit(start, {"local_epochs": 1, "expected_clients": 1})
    assert n == 1
    assert [a.shape for a in new] == [a.shape for a in start]
    assert all(a.dtype == np.float32 for a in new)
    update = np.sqrt(sum(float(((a - b) ** 2).sum()) for a, b in zip(new, start, strict=True)))
    assert 0 < update <= cfg.federated.update_clip_norm + 1e-5
    assert metrics["update_norm"] > 0

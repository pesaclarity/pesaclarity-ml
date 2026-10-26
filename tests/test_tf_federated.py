"""Keras trainer and Flower adapter. Skipped unless the extras are installed."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from akili.config import (
    ExperimentConfig,
    FederatedConfig,
    SyntheticDataConfig,
    TrackingConfig,
)
from akili.federated.client import CategoriserClient, Consent
from akili.federated.run import run_federated

pytestmark = pytest.mark.tf


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
    assert result.simulation.eligible_clients == 120
    assert result.accuracy > 0.2
    assert result.exported.path.exists()


@pytest.mark.fl
def test_flower_adapter_wraps_client() -> None:
    pytest.importorskip("flwr")
    import flwr.client
    import flwr.clientapp

    from akili.federated.flower import client_app, to_numpy_client

    class Trainer:
        num_examples = 1

        def get_weights(self) -> list[np.ndarray]:
            return [np.zeros(2, np.float32)]

        def set_weights(self, weights: list[np.ndarray]) -> None: ...

        def fit(self, epochs: int, batch_size: int, learning_rate: float) -> dict[str, float]:
            return {}

        def evaluate(self) -> tuple[float, dict[str, float]]:
            return 0.0, {}

    core = CategoriserClient(Trainer(), Consent(), FederatedConfig())
    adapted = to_numpy_client(core)
    assert isinstance(adapted, flwr.client.NumPyClient)
    assert adapted.get_properties({}) == {"app": True, "commercial": False}
    assert adapted.fit([np.zeros(2, np.float32)], {"track": "commercial"})[1] == 0
    assert isinstance(client_app(lambda ctx: core), flwr.clientapp.ClientApp)

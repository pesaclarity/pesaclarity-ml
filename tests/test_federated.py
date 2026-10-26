import math

import numpy as np
import pytest

from akili.config import ExperimentConfig, FederatedConfig
from akili.federated.client import CategoriserClient, Consent
from akili.federated.simulation import CohortTooSmallError, simulate
from akili.federated.updates import (
    Weights,
    add_gaussian_noise,
    client_noise_std,
    clip_update,
    global_norm,
    mean_update,
)
from akili.tracks import Track


class FakeTrainer:
    """Moves every weight one step towards ``target``."""

    def __init__(self, target: float, n: int = 10) -> None:
        self.target = target
        self.n = n
        self.weights: Weights = [np.zeros(3, np.float32), np.zeros((2, 2), np.float32)]
        self.fit_calls = 0

    @property
    def num_examples(self) -> int:
        return self.n

    def get_weights(self) -> Weights:
        return [w.copy() for w in self.weights]

    def set_weights(self, weights: Weights) -> None:
        self.weights = [np.asarray(w, np.float32).copy() for w in weights]

    def fit(self, epochs: int, batch_size: int, learning_rate: float) -> dict[str, float]:
        self.fit_calls += 1
        self.weights = [w + learning_rate * (self.target - w) for w in self.weights]
        return {"loss": 0.0}

    def evaluate(self) -> tuple[float, dict[str, float]]:
        return float(np.mean([np.mean(np.abs(w - self.target)) for w in self.weights])), {}


FED = FederatedConfig(
    rounds=5,
    clients_per_round=100,
    local_learning_rate=0.5,
    update_clip_norm=10.0,
    update_noise_multiplier=0.0,
    min_cohort_size=100,
)


def zeros() -> Weights:
    return FakeTrainer(0).get_weights()


def test_clip_update_bounds_norm() -> None:
    update = [np.full(4, 3.0, np.float32), np.full(5, 4.0, np.float32)]
    clipped = clip_update(update, 1.0)
    assert global_norm(clipped) == pytest.approx(1.0, rel=1e-5)
    small = [np.array([0.1], np.float32)]
    assert clip_update(small, 1.0)[0][0] == pytest.approx(0.1)
    assert clip_update([np.zeros(2, np.float32)], 1.0)[0].tolist() == [0, 0]


def test_client_noise_sums_to_aggregate_noise() -> None:
    std = client_noise_std(clip_norm=2.0, noise_multiplier=1.5, expected_clients=100)
    assert std * math.sqrt(100) == pytest.approx(3.0)
    rng = np.random.default_rng(0)
    noisy = add_gaussian_noise([np.zeros(100_000, np.float32)], 0.5, rng)
    assert float(np.std(noisy[0])) == pytest.approx(0.5, rel=0.02)


def test_mean_update_divides_by_expected_cohort() -> None:
    ups = [[np.ones(2, np.float32)], [np.ones(2, np.float32)]]
    assert mean_update(ups, expected_clients=4)[0].tolist() == [0.5, 0.5]
    with pytest.raises(ValueError, match="no updates"):
        mean_update([], 1)


def test_client_declines_tracks_without_consent() -> None:
    trainer = FakeTrainer(1.0)
    client = CategoriserClient(trainer, Consent(), FED, seed=0)
    weights, n, metrics = client.fit(zeros(), {"track": "commercial"})
    assert n == 0
    assert metrics["declined"] is True
    assert trainer.fit_calls == 0
    assert all(np.all(w == 0) for w in weights)
    assert client.get_properties({}) == {"app": True, "commercial": False}


def test_client_returns_clipped_update_counted_once() -> None:
    fed = FED.model_copy(update={"update_clip_norm": 0.1})
    client = CategoriserClient(FakeTrainer(1.0, n=500), Consent(), fed, seed=0)
    weights, n, metrics = client.fit(zeros(), {"track": "app"})
    assert n == 1
    assert global_norm(weights) == pytest.approx(0.1, rel=1e-5)
    assert float(metrics["update_norm"]) > 0.1


def test_client_with_no_data_skips() -> None:
    client = CategoriserClient(FakeTrainer(1.0, n=0), Consent(), FED)
    assert client.fit(zeros(), {"track": "app"})[1] == 0


def _consents(n: int, commercial_every: int = 0) -> dict[str, Consent]:
    return {
        f"c{i}": Consent(
            contribute_licensed_models=bool(commercial_every) and i % commercial_every == 0
        )
        for i in range(n)
    }


def test_simulation_moves_towards_clients() -> None:
    cfg = ExperimentConfig(name="sim", federated=FED)
    consents = _consents(200)
    result = simulate(
        cfg, zeros(), consents, lambda cid: CategoriserClient(FakeTrainer(1.0), consents[cid], FED)
    )
    assert len(result.rounds) == FED.rounds
    assert result.eligible_clients == 200
    assert all(log.participants > 0 for log in result.rounds)
    assert 0.5 < float(np.mean(result.weights[0])) < 1.0
    assert result.privacy is not None
    assert math.isinf(result.privacy.epsilon)


def test_simulation_enforces_cohort_minimum() -> None:
    cfg = ExperimentConfig(name="sim", federated=FED)
    with pytest.raises(CohortTooSmallError, match="minimum cohort is 100"):
        simulate(cfg, zeros(), _consents(50), lambda cid: pytest.fail("no client should train"))


def test_commercial_simulation_only_uses_opted_in_clients() -> None:
    fed = FED.model_copy(update={"min_cohort_size": 1000, "update_noise_multiplier": 1.0})
    cfg = ExperimentConfig(name="sim", track=Track.COMMERCIAL, federated=fed)
    consents = _consents(5000, commercial_every=4)
    trained: set[str] = set()

    def make(cid: str) -> CategoriserClient:
        trained.add(cid)
        return CategoriserClient(FakeTrainer(1.0), consents[cid], fed, seed=1)

    result = simulate(cfg, zeros(), consents, make)
    assert result.eligible_clients == 1250
    assert trained
    assert all(consents[c].contribute_licensed_models for c in trained)
    assert result.privacy is not None
    assert result.privacy.epsilon < float("inf")

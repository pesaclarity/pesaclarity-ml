"""In-process federated simulation of DP-FedAvg rounds against :class:`CategoriserClient`.

This mirrors what the aggregator in pesaclarity-backend does (Poisson client sampling, cohort
minimum, unweighted mean of clipped and noised updates) without needing Flower or a network,
so the client code can be tested and tuned locally.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from akili.config import ExperimentConfig
from akili.federated.client import CategoriserClient, Consent
from akili.federated.updates import Weights, add, mean_update, subtract
from akili.privacy.accounting import PrivacyReport, dp_fedavg_report

ClientFactory = Callable[[str], CategoriserClient]


class CohortTooSmallError(RuntimeError):
    pass


@dataclass
class RoundLog:
    round: int
    participants: int
    declined: int
    mean_update_norm: float


@dataclass
class SimulationResult:
    weights: Weights
    rounds: list[RoundLog] = field(default_factory=list)
    privacy: PrivacyReport | None = None
    eligible_clients: int = 0


def simulate(
    cfg: ExperimentConfig,
    initial_weights: Weights,
    consents: Mapping[str, Consent],
    make_client: ClientFactory,
) -> SimulationResult:
    """Run ``cfg.federated.rounds`` rounds on the clients whose consent allows ``cfg.track``."""
    if cfg.federated is None:
        raise ValueError("config has no federated section")
    fed = cfg.federated
    eligible = sorted(cid for cid, c in consents.items() if c.allows(cfg.track))
    minimum = cfg.min_cohort_size
    if len(eligible) < minimum:
        raise CohortTooSmallError(
            f"{len(eligible)} clients consented to the {cfg.track.value} track; "
            f"the minimum cohort is {minimum}"
        )

    expected = min(fed.clients_per_round, len(eligible))
    q = expected / len(eligible)
    rng = np.random.default_rng(cfg.seed)
    weights = [np.asarray(w, dtype=np.float32) for w in initial_weights]
    result = SimulationResult(weights=weights, eligible_clients=len(eligible))
    round_config = {
        "track": cfg.track.value,
        "expected_clients": expected,
        "local_epochs": fed.local_epochs,
        "local_batch_size": fed.local_batch_size,
        "local_learning_rate": fed.local_learning_rate,
    }

    for r in range(fed.rounds):
        sampled = [cid for cid in eligible if rng.random() < q]
        updates: list[Weights] = []
        norms: list[float] = []
        declined = 0
        for cid in sampled:
            new_weights, n, metrics = make_client(cid).fit(weights, {**round_config, "round": r})
            if n == 0:
                declined += 1
                continue
            updates.append(subtract(new_weights, weights))
            norms.append(float(metrics.get("update_norm", 0.0)))
        if updates:
            weights = add(weights, mean_update(updates, expected))
        result.rounds.append(
            RoundLog(
                round=r,
                participants=len(updates),
                declined=declined,
                mean_update_norm=float(np.mean(norms)) if norms else 0.0,
            )
        )

    result.weights = weights
    result.privacy = dp_fedavg_report(
        n_clients=len(eligible),
        clients_per_round=expected,
        rounds=fed.rounds,
        noise_multiplier=fed.update_noise_multiplier,
        clip_norm=fed.update_clip_norm,
        delta=cfg.dp.target_delta,
    )
    return result

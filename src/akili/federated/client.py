"""Federated client with the Flower ``NumPyClient`` method shapes, usable without Flower.

Each participating user contributes one clipped, noised update and reports ``num_examples=1``,
so a stock FedAvg on the server computes the unweighted, user-level mean that the privacy
accounting assumes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from akili.config import FederatedConfig
from akili.federated.updates import (
    Weights,
    add,
    add_gaussian_noise,
    client_noise_std,
    clip_update,
    global_norm,
    subtract,
)
from akili.tracks import Track, policy_for

Scalar = bool | bytes | float | int | str
ConfigDict = Mapping[str, Scalar]


class LocalTrainer(Protocol):
    """Trains the categoriser on one user's confirmed transactions."""

    @property
    def num_examples(self) -> int: ...

    def get_weights(self) -> Weights: ...

    def set_weights(self, weights: Weights) -> None: ...

    def fit(self, epochs: int, batch_size: int, learning_rate: float) -> dict[str, float]: ...

    def evaluate(self) -> tuple[float, dict[str, float]]: ...


@dataclass(frozen=True)
class Consent:
    """The user's choices from the first-run privacy screen."""

    improve_app: bool = True
    contribute_licensed_models: bool = False

    def allows(self, track: Track) -> bool:
        key = policy_for(track).consent_key
        return bool(getattr(self, key))


class CategoriserClient:
    def __init__(
        self,
        trainer: LocalTrainer,
        consent: Consent,
        cfg: FederatedConfig,
        seed: int | None = None,
    ) -> None:
        self.trainer = trainer
        self.consent = consent
        self.cfg = cfg
        # Real clients must draw noise from OS entropy; a seed is for reproducible simulation only.
        self._rng = np.random.default_rng(seed)

    def get_properties(self, config: ConfigDict) -> dict[str, Scalar]:
        return {t.value: self.consent.allows(t) for t in Track}

    def get_parameters(self, config: ConfigDict) -> Weights:
        return self.trainer.get_weights()

    def fit(
        self, parameters: Weights, config: ConfigDict
    ) -> tuple[Weights, int, dict[str, Scalar]]:
        """Train locally and return global weights plus a clipped, noised update."""
        track = Track(str(config.get("track", Track.APP.value)))
        if not self.consent.allows(track):
            return [np.asarray(p, dtype=np.float32) for p in parameters], 0, {"declined": True}
        if self.trainer.num_examples == 0:
            return [np.asarray(p, dtype=np.float32) for p in parameters], 0, {"empty": True}

        self.trainer.set_weights(parameters)
        metrics = self.trainer.fit(
            epochs=int(config.get("local_epochs", self.cfg.local_epochs)),
            batch_size=int(config.get("local_batch_size", self.cfg.local_batch_size)),
            learning_rate=float(config.get("local_learning_rate", self.cfg.local_learning_rate)),
        )
        update = subtract(self.trainer.get_weights(), parameters)
        raw_norm = global_norm(update)
        update = clip_update(update, self.cfg.update_clip_norm)
        expected = int(config.get("expected_clients", self.cfg.clients_per_round))
        std = client_noise_std(
            self.cfg.update_clip_norm, self.cfg.update_noise_multiplier, expected
        )
        update = add_gaussian_noise(update, std, self._rng)
        out: dict[str, Scalar] = {"update_norm": raw_norm, **metrics}
        return add(parameters, update), 1, out

    def evaluate(
        self, parameters: Weights, config: ConfigDict
    ) -> tuple[float, int, dict[str, Scalar]]:
        self.trainer.set_weights(parameters)
        loss, metrics = self.trainer.evaluate()
        return loss, self.trainer.num_examples, dict(metrics)

"""Model-update maths for DP-FedAvg: clipping, client-side noise, averaging."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Weights = list[npt.NDArray[np.float32]]


def subtract(a: Sequence[npt.NDArray[np.float32]], b: Sequence[npt.NDArray[np.float32]]) -> Weights:
    return [np.asarray(x - y, dtype=np.float32) for x, y in zip(a, b, strict=True)]


def add(a: Sequence[npt.NDArray[np.float32]], b: Sequence[npt.NDArray[np.float32]]) -> Weights:
    return [np.asarray(x + y, dtype=np.float32) for x, y in zip(a, b, strict=True)]


def global_norm(update: Sequence[npt.NDArray[np.float32]]) -> float:
    return math.sqrt(sum(float(np.sum(np.square(u, dtype=np.float64))) for u in update))


def clip_update(update: Sequence[npt.NDArray[np.float32]], clip_norm: float) -> Weights:
    """Scale the update so its global L2 norm is at most ``clip_norm``."""
    norm = global_norm(update)
    scale = min(1.0, clip_norm / norm) if norm > 0 else 1.0
    return [np.asarray(u * scale, dtype=np.float32) for u in update]


def client_noise_std(clip_norm: float, noise_multiplier: float, expected_clients: int) -> float:
    """Per-client noise so the securely aggregated sum carries N(0, (clip * z)^2).

    With secure aggregation the server only sees the sum, so each of k clients adds 1/sqrt(k) of
    the noise. If fewer than k clients report, the sum is under-noised: the server must drop
    rounds below the cohort minimum.
    """
    return clip_norm * noise_multiplier / math.sqrt(expected_clients)


def add_gaussian_noise(
    update: Sequence[npt.NDArray[np.float32]], std: float, rng: np.random.Generator
) -> Weights:
    if std == 0:
        return [np.asarray(u, dtype=np.float32) for u in update]
    return [np.asarray(u + rng.normal(0.0, std, u.shape), dtype=np.float32) for u in update]


def mean_update(
    updates: Sequence[Sequence[npt.NDArray[np.float32]]], expected_clients: float
) -> Weights:
    """Sum of client updates divided by the expected cohort size (the DP-FedAvg estimator)."""
    if not updates:
        raise ValueError("no updates to average")
    total = [np.zeros_like(u, dtype=np.float32) for u in updates[0]]
    for update in updates:
        total = add(total, update)
    return [np.asarray(t / expected_clients, dtype=np.float32) for t in total]

"""DP-SGD's bookkeeping, shared by the PyTorch loop (:mod:`akili.train.torch_dp_sgd`) and the
Keras one kept for ``--keras`` (:mod:`akili.train.keras_dp_sgd`): what a run reports, its steps
and how each sampled person's examples are picked.

With ``unit: user`` (the default) each step samples people, not transactions: every sampled
person's examples give one gradient (the mean over their examples), which is clipped to the
norm C before noise is added (McMahan et al. 2018). Adding or removing one person then changes
the noisy sum by at most C, so the budget covers everything that person contributed. With
``unit: example`` each transaction is clipped on its own, which protects one transaction only.

Noise and sampling are drawn from operating-system entropy unless ``dp.noise_seed`` is set. The
entropy-drawn seed is never returned or logged: anyone with it could regenerate the noise and
remove it. A fixed seed is for reproducible synthetic runs only, and the release gate refuses
any report that used one.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from akili.privacy.accounting import PrivacyUnit


@dataclass(frozen=True)
class DPTrainingStats:
    unit: PrivacyUnit
    population: int
    """People (user unit) or examples (example unit) that could be sampled."""
    steps: int
    """Steps actually taken, counted in the loop; the accountant must be given the same."""
    sampling_rate: float
    """Poisson sampling rate actually used; the accountant must be given the same."""
    mean_batch_size: float
    """Mean people (user unit) or examples (example unit) per step."""
    final_loss: float
    noise_seeded: bool
    """True when the noise came from ``dp.noise_seed`` rather than OS entropy."""


def steps_per_epoch(population: int, per_step: int) -> int:
    """Must match :func:`akili.privacy.accounting.dp_sgd_report`."""
    return max(1, round(population / per_step))


def group_by_user(users: npt.ArrayLike) -> list[npt.NDArray[np.int64]]:
    """Row indices of each person's examples, one array per person, in order of first id."""
    _, first, inverse = np.unique(np.asarray(users), return_index=True, return_inverse=True)
    rows = np.argsort(inverse, kind="stable").astype(np.int64)
    groups = np.split(rows, np.cumsum(np.bincount(inverse))[:-1])
    return [groups[k] for k in np.argsort(first)]


def user_rows(
    groups: list[npt.NDArray[np.int64]],
    chosen: npt.NDArray[np.int64],
    cap: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Row indices ``[people, length]`` of each chosen person's examples (at most ``cap``,
    picked at random), padded by repeating their last row, and a mask that is 0 on padding."""
    rows = [
        rng.choice(groups[k], size=cap, replace=False) if len(groups[k]) > cap else groups[k]
        for k in chosen
    ]
    length = max(len(r) for r in rows)
    idx = np.stack([np.pad(r, (0, length - len(r)), mode="edge") for r in rows])
    mask = np.stack([np.arange(length) < len(r) for r in rows]).astype(np.float32)
    return idx, mask


def noise_seed_or_entropy(noise_seed: int | None) -> tuple[int, bool]:
    """The seed noise and sampling are drawn from, and whether it was fixed: ``noise_seed``,
    or 63 bits of OS entropy that are never returned to the caller's logs."""
    if noise_seed is not None:
        return noise_seed, True
    return secrets.randbits(63), False

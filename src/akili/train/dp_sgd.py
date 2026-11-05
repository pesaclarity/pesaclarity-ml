"""DP-SGD: clipping and Gaussian noise with Poisson sampling, per person or per example.

tensorflow-privacy is not used because its last release (0.9.0) supports neither Python 3.12 nor
Keras 3. This is the same algorithm (Abadi et al. 2016) in a small custom loop, and the budget is
computed with ``dp-accounting``, the accountant tensorflow-privacy itself uses. Revisit if
tensorflow-privacy ships a compatible release.

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
from typing import Any

import numpy as np
import numpy.typing as npt

from akili._optional import require
from akili.config import DPConfig, TrainingConfig
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


class _Steps:
    """Compiled DP-SGD steps for one model: per example, per person, and noise only."""

    def __init__(
        self,
        model: Any,
        *,
        width: int,
        training: TrainingConfig,
        dp: DPConfig,
        expected: float,
        noise: Any,
    ) -> None:
        tf = require("tensorflow", "train")
        keras = require("keras", "train")
        clip = float(dp.l2_norm_clip)
        noise_std = clip * float(dp.noise_multiplier)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction="none")
        optimizer = keras.optimizers.SGD(learning_rate=training.learning_rate, momentum=0.9)
        variables = model.trainable_variables
        optimizer.build(variables)

        def clipped(tape: Any, loss: Any) -> Any:
            grads = [tf.convert_to_tensor(g) for g in tape.gradient(loss, variables)]
            out, _ = tf.clip_by_global_norm(grads, clip)
            return out

        def example_grad(args: Any) -> Any:
            xi, yi = args
            with tf.GradientTape() as tape:
                loss = loss_fn(yi[None], model(xi[None], training=True))
            return clipped(tape, loss)

        def user_grad(args: Any) -> Any:
            xu, yu, mu = args
            with tf.GradientTape() as tape:
                losses = loss_fn(yu, model(xu, training=True))
                loss = tf.reduce_sum(losses * mu) / tf.maximum(tf.reduce_sum(mu), 1.0)
            return clipped(tape, loss)

        def apply(per_unit: Any) -> None:
            noisy = [
                (tf.reduce_sum(g, axis=0) + noise.normal(tf.shape(v), stddev=noise_std)) / expected
                for g, v in zip(per_unit, variables, strict=True)
            ]
            optimizer.apply(noisy, variables)

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, width], tf.int32),
                tf.TensorSpec([None], tf.int32),
            ],
            reduce_retracing=True,
        )
        def example_step(xb: Any, yb: Any) -> Any:
            apply(tf.vectorized_map(example_grad, (xb, yb)))
            return tf.reduce_mean(loss_fn(yb, model(xb, training=False)))

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, None, width], tf.int32),
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, None], tf.float32),
            ],
            reduce_retracing=True,
        )
        def user_step(xb: Any, yb: Any, mb: Any) -> Any:
            apply(tf.vectorized_map(user_grad, (xb, yb, mb)))
            losses = loss_fn(
                tf.reshape(yb, [-1]), model(tf.reshape(xb, [-1, width]), training=False)
            )
            m = tf.reshape(mb, [-1])
            return tf.reduce_sum(losses * m) / tf.maximum(tf.reduce_sum(m), 1.0)

        @tf.function
        def noise_only_step() -> None:
            optimizer.apply(
                [noise.normal(tf.shape(v), stddev=noise_std) / expected for v in variables],
                variables,
            )

        self.example = example_step
        self.user = user_step
        self.noise_only = noise_only_step


def train_dp_sgd(
    model: Any,
    x: npt.NDArray[np.int32],
    y: npt.NDArray[np.int32],
    training: TrainingConfig,
    dp: DPConfig,
    *,
    users: npt.ArrayLike | None = None,
    noise_seed: int | None = None,
) -> DPTrainingStats:
    """Train ``model`` in place with DP-SGD.

    ``users`` gives each row's person and is required for ``dp.unit: user``. ``noise_seed``
    makes the noise and sampling reproducible; leave it None for anything but synthetic data.
    """
    tf = require("tensorflow", "train")
    if dp.unit is PrivacyUnit.USER:
        if users is None:
            raise ValueError("user-level DP-SGD needs each example's user")
        groups = group_by_user(users)
        population, per_step = len(groups), dp.users_per_step
    else:
        groups = []
        population, per_step = len(x), training.batch_size
    q = min(1.0, per_step / population)

    seeded = noise_seed is not None
    seed = noise_seed if seeded else secrets.randbits(63)
    rng = np.random.default_rng(seed)
    steps = _Steps(
        model,
        width=x.shape[1],
        training=training,
        dp=dp,
        expected=q * population,
        noise=tf.random.Generator.from_seed(seed),
    )

    sizes: list[int] = []
    last_loss = float("nan")
    for _ in range(training.epochs * steps_per_epoch(population, per_step)):
        chosen = np.flatnonzero(rng.random(population) < q)
        sizes.append(len(chosen))
        if len(chosen) == 0:
            steps.noise_only()
        elif dp.unit is PrivacyUnit.USER:
            idx, mask = user_rows(groups, chosen, dp.max_examples_per_user, rng)
            last_loss = float(steps.user(x[idx], y[idx], mask))
        else:
            last_loss = float(steps.example(x[chosen], y[chosen]))
    return DPTrainingStats(
        unit=dp.unit,
        population=population,
        steps=len(sizes),
        sampling_rate=q,
        mean_batch_size=float(np.mean(sizes)),
        final_loss=last_loss,
        noise_seeded=seeded,
    )

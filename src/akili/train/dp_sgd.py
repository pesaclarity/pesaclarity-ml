"""DP-SGD: per-example clipping and Gaussian noise, with Poisson sampling.

tensorflow-privacy is not used because its last release (0.9.0) supports neither Python 3.12 nor
Keras 3. This is the same algorithm (Abadi et al. 2016) in a small custom loop, and the budget is
computed with ``dp-accounting``, the accountant tensorflow-privacy itself uses. Revisit if
tensorflow-privacy ships a compatible release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from akili._optional import require
from akili.config import DPConfig, TrainingConfig


@dataclass(frozen=True)
class DPTrainingStats:
    steps: int
    mean_batch_size: float
    final_loss: float


def steps_per_epoch(n_examples: int, batch_size: int) -> int:
    """Must match :func:`akili.privacy.accounting.dp_sgd_report`."""
    return max(1, round(n_examples / batch_size))


def train_dp_sgd(
    model: Any,
    x: npt.NDArray[np.int32],
    y: npt.NDArray[np.int32],
    training: TrainingConfig,
    dp: DPConfig,
    *,
    seed: int = 0,
) -> DPTrainingStats:
    """Train ``model`` in place with DP-SGD."""
    tf = require("tensorflow", "train")
    keras = require("keras", "train")

    n = len(x)
    q = min(1.0, training.batch_size / n)
    expected_batch = q * n
    clip = float(dp.l2_norm_clip)
    noise_std = clip * float(dp.noise_multiplier)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction="none")
    optimizer = keras.optimizers.SGD(learning_rate=training.learning_rate, momentum=0.9)
    variables = model.trainable_variables
    optimizer.build(variables)

    def clipped_grad(args: Any) -> Any:
        xi, yi = args
        with tf.GradientTape() as tape:
            loss = loss_fn(yi[None], model(xi[None], training=True))
        grads = [tf.convert_to_tensor(g) for g in tape.gradient(loss, variables)]
        clipped, _ = tf.clip_by_global_norm(grads, clip)
        return clipped

    width = x.shape[1]

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, width], tf.int32),
            tf.TensorSpec([None], tf.int32),
        ],
        reduce_retracing=True,
    )
    def step(xb: Any, yb: Any) -> Any:
        per_example = tf.vectorized_map(clipped_grad, (xb, yb))
        noisy = [
            (tf.reduce_sum(g, axis=0) + tf.random.normal(tf.shape(v), stddev=noise_std))
            / expected_batch
            for g, v in zip(per_example, variables, strict=True)
        ]
        optimizer.apply(noisy, variables)
        return tf.reduce_mean(loss_fn(yb, model(xb, training=False)))

    @tf.function
    def noise_only_step() -> None:
        noise = [
            tf.random.normal(tf.shape(v), stddev=noise_std) / expected_batch for v in variables
        ]
        optimizer.apply(noise, variables)

    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    total_steps = training.epochs * steps_per_epoch(n, training.batch_size)
    sizes: list[int] = []
    last_loss = float("nan")
    for _ in range(total_steps):
        idx = np.flatnonzero(rng.random(n) < q)
        sizes.append(len(idx))
        if len(idx) == 0:
            noise_only_step()
            continue
        last_loss = float(step(x[idx], y[idx]))
    return DPTrainingStats(
        steps=total_steps, mean_batch_size=float(np.mean(sizes)), final_loss=last_loss
    )

"""Keras implementation of :class:`LocalTrainer` (needs the [train] extra)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from akili._optional import require
from akili.federated.updates import Weights


class SharedKerasModel:
    """One model, optimiser and compiled train step reused by every simulated client.

    Plain SGD keeps no optimiser state, so sharing it across clients is safe; each client sets
    the global weights before training.
    """

    def __init__(self, model: Any) -> None:
        tf = require("tensorflow", "train")
        keras = require("keras", "train")
        self.model = model
        self.optimizer = keras.optimizers.SGD(learning_rate=0.1)
        self.optimizer.build(model.trainable_variables)
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        width = int(model.inputs[0].shape[-1])

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, width], tf.int32),
                tf.TensorSpec([None], tf.int32),
            ]
        )
        def step(x: Any, y: Any) -> Any:
            with tf.GradientTape() as tape:
                loss = loss_fn(y, model(x, training=True))
            grads = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply(grads, model.trainable_variables)
            return loss

        self._step = step

    def train_step(
        self, x: npt.NDArray[np.int32], y: npt.NDArray[np.int32], learning_rate: float
    ) -> float:
        self.optimizer.learning_rate.assign(learning_rate)
        return float(self._step(x, y))


class KerasTrainer:
    """Local training on one client's data."""

    def __init__(
        self,
        shared: SharedKerasModel,
        x: npt.NDArray[np.int32],
        y: npt.NDArray[np.int32],
        seed: int = 0,
    ) -> None:
        self.shared = shared
        self.x = x
        self.y = y
        self._rng = np.random.default_rng(seed)

    @property
    def num_examples(self) -> int:
        return len(self.y)

    def get_weights(self) -> Weights:
        return [np.asarray(w, dtype=np.float32) for w in self.shared.model.get_weights()]

    def set_weights(self, weights: Weights) -> None:
        self.shared.model.set_weights(weights)

    def fit(self, epochs: int, batch_size: int, learning_rate: float) -> dict[str, float]:
        loss = float("nan")
        for _ in range(epochs):
            order = self._rng.permutation(len(self.y))
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                loss = self.shared.train_step(self.x[idx], self.y[idx], learning_rate)
        return {"loss": loss}

    def evaluate(self) -> tuple[float, dict[str, float]]:
        probs = np.asarray(self.shared.model(self.x, training=False))
        loss = float(-np.mean(np.log(probs[np.arange(len(self.y)), self.y] + 1e-7)))
        accuracy = float((probs.argmax(axis=1) == self.y).mean())
        return loss, {"accuracy": accuracy}

"""What the centralised pipeline, the audit and the federated simulation need from a framework.

The categoriser trains in PyTorch (:mod:`akili.train.torch_backend`) and, until TensorFlow is
removed (decision 0023, stage 5), in Keras (:mod:`akili.train.keras_pipeline`, ``--keras``).
Both give the same things, so the pipeline (:mod:`akili.train.pipeline`), the
membership-inference audit (:mod:`akili.privacy.mia`) and the federated simulation
(:mod:`akili.federated.run`) are written once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from akili.config import ExperimentConfig
from akili.export.size import ExportedModel
from akili.federated.client import LocalTrainer
from akili.federated.updates import Weights
from akili.privacy.accounting import PrivacyReport

IntArray = npt.NDArray[np.int32]
FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class FitOutput:
    model: Any
    losses: list[float]
    """Per-epoch training loss without DP; the final loss only with DP-SGD."""
    privacy: PrivacyReport | None = None


@dataclass(frozen=True)
class Exported:
    """What an export wrote: the file that ships, and the other files beside it."""

    model: ExportedModel
    files: list[Path] = field(default_factory=list)
    """Every file written (the shipped one, its manifest and anything else), for tracking."""
    metrics: dict[str, float] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    """Anything the export measured, such as the shipped file's parity with the trained model."""


class Backend(Protocol):
    """One framework's categoriser."""

    @property
    def name(self) -> str:
        """``torch`` or ``keras``."""
        ...

    def fit(
        self,
        cfg: ExperimentConfig,
        x: IntArray,
        y: IntArray,
        *,
        users: npt.ArrayLike,
        seed: int,
        n_classes: int,
        noise_seed: int | None = None,
    ) -> FitOutput:
        """Build and train one categoriser exactly as the config says (plain or DP-SGD).

        ``seed`` sets the starting weights (and dropout) only. DP noise and sampling come
        from OS entropy unless ``noise_seed`` is given."""
        ...

    def predict(self, model: Any, x: IntArray) -> FloatArray:
        """The trained model's probabilities ``[n, classes]``."""
        ...

    def export(self, model: Any, cfg: ExperimentConfig, run_dir: Path) -> Exported:
        """Write the file that ships and its manifest to ``run_dir``; refuse a file over the
        size budget."""
        ...

    def run_exported(self, exported: ExportedModel, x: IntArray) -> FloatArray:
        """The shipped file's probabilities, run as the app runs it."""
        ...

    def shipped_predict(self, model: Any, cfg: ExperimentConfig, x: IntArray) -> FloatArray:
        """``model`` exported as the release would be, then run: how the audit scores its
        shadow models, the same way as the target."""
        ...

    def new_model(self, cfg: ExperimentConfig, n_classes: int, seed: int) -> Any:
        """An untrained categoriser, for the federated simulation."""
        ...

    def get_weights(self, model: Any) -> Weights:
        """The weights as the federated protocol carries them: Keras' order and layout."""
        ...

    def set_weights(self, model: Any, weights: Weights) -> None: ...

    def local_trainer(self, model: Any) -> Callable[[IntArray, IntArray, int], LocalTrainer]:
        """A factory of one client's trainer (its data and seed) sharing ``model``."""
        ...


def accuracy(probs: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    return float((np.asarray(probs).argmax(axis=1) == np.asarray(labels)).mean())

"""The PyTorch categoriser as a :class:`~akili.train.backend.Backend` (export needs the [torch]
extra, for onnx).

Training (:func:`fit`): the Keras recipe in torch. Without DP, Adam (Keras' epsilon) on
cross-entropy, the rows shuffled each epoch, dropout on; with DP, :mod:`akili.train.torch_dp_sgd`.
Seeded and deterministic (one thread, torch's deterministic algorithms), so the same code, torch
version and machine give the same model.

Export (:func:`export_categoriser`): ``torch.onnx`` at opset 17, then rayo-convert's int8
(:mod:`akili.export.onnx`), to ``categoriser.weights``, the file the app runs, with the ONNX
file it came from and ``model-manifest.json``. The export fails if the file is over
``export.max_size_bytes`` or strays from the torch model by more than
``export.parity_tolerance`` (:func:`check_parity`). Everything the release reports about the
model is scored on that file as rayo runs it, or with no rayo checkout on rayo-convert's NumPy
reference, which gives the same int8 meaning and which rayo is checked against.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from akili.categories import default_categories
from akili.config import ExperimentConfig
from akili.export.size import ExportedModel, ModelTooLargeError, describe
from akili.federated.client import LocalTrainer
from akili.federated.updates import Weights
from akili.model import (
    INPUT,
    LAYERS,
    ONNX_FILE,
    OUTPUT,
    RAYO_OPTIONS,
    WEIGHTS_FILE,
    Categoriser,
    build_categoriser,
    example_input,
    keras_layout,
    parameter_count,
    required_tensors,
    set_keras_layout,
)
from akili.train.backend import Exported, FitOutput, FloatArray, IntArray

THREADS = 1  # torch's CPU threads: fixed, so a run is repeatable


def rayo_dir() -> str | None:
    """A rayo checkout whose C library runs the converted file (``RAYO_DIR``), if any."""
    return os.environ.get("RAYO_DIR") or None


def _deterministic(seed: int) -> None:
    from akili.export.onnx import deterministic

    deterministic(seed, THREADS)


def fit(
    cfg: ExperimentConfig,
    x: IntArray,
    y: IntArray,
    *,
    users: npt.ArrayLike,
    seed: int,
    n_classes: int,
    noise_seed: int | None = None,
) -> FitOutput:
    """A :class:`~akili.model.Categoriser` trained as ``cfg`` says, plain or with DP-SGD."""
    from akili.train.pipeline import privacy_report
    from akili.train.torch_dp_sgd import train_dp_sgd

    _deterministic(seed)
    model = build_categoriser(cfg.features, cfg.model, n_classes, seed=seed)
    if cfg.dp.enabled:
        stats = train_dp_sgd(model, x, y, cfg.training, cfg.dp, users=users, noise_seed=noise_seed)
        return FitOutput(model=model, losses=[stats.final_loss], privacy=privacy_report(cfg, stats))
    t = cfg.training
    xt = torch.from_numpy(np.asarray(x, dtype=np.int64))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    optimiser = torch.optim.Adam(model.parameters(), lr=t.learning_rate, eps=1e-7)
    schedule = cosine(optimiser, t.lr_schedule, t.epochs * -(-len(yt) // t.batch_size))
    order = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for _ in range(t.epochs):
        total, count = 0.0, 0
        perm: Any = torch.randperm(len(yt), generator=order)
        for batch in perm.split(t.batch_size):
            loss: Any = torch.nn.functional.cross_entropy(model.logits(xt[batch]), yt[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if schedule is not None:
                schedule.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        losses.append(total / count)
    model.eval()
    return FitOutput(model=model, losses=losses)


def cosine(optimiser: Any, schedule: str, steps: int) -> Any:
    """The learning-rate schedule ``training.lr_schedule`` names over ``steps`` steps, or None
    for a constant rate."""
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, max(1, steps))
    return None


class WeightAverage:
    """An exponential moving average of ``model``'s weights (``dp.weight_average``),
    updated after every step and loaded into the model at the end; a no-op at decay 0."""

    def __init__(self, model: Any, decay: float) -> None:
        self.model = model
        self.decay = decay
        self.steps = 0
        self.average = (
            {k: v.detach().clone() for k, v in model.named_parameters()} if decay > 0 else {}
        )

    def update(self) -> None:
        if not self.decay:
            return
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        with torch.no_grad():
            for k, v in self.model.named_parameters():
                self.average[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def load(self) -> None:
        if not self.decay:
            return
        with torch.no_grad():
            for k, v in self.model.named_parameters():
                v.copy_(self.average[k])


def predict(model: Any, x: IntArray, batch: int = 4096) -> FloatArray:
    model.eval()
    xt = torch.from_numpy(np.asarray(x, dtype=np.int64))
    with torch.no_grad():
        out = [model(xt[i : i + batch]) for i in range(0, len(xt), batch)]
    if not out:
        return np.zeros((0, model.category.out_features), np.float32)
    return np.asarray(torch.cat(out).numpy(), dtype=np.float32)


class Converted:
    """A converted ``categoriser.weights`` as the app runs it: on rayo's C library when there
    is a rayo checkout (``rayo_dir``), else rayo-convert's reference on the ONNX file it came
    from."""

    def __init__(self, weights: bytes, onnx: bytes, rayo: str | None) -> None:
        from akili.export.onnx import RayoRunner, ReferenceRunner

        self.runner: Any = (
            RayoRunner(weights, rayo) if rayo else ReferenceRunner(onnx, RAYO_OPTIONS)
        )
        self.on = "rayo" if rayo else "rayo-convert reference"

    def run(self, x: IntArray) -> FloatArray:
        rows = [
            np.asarray(self.runner.run({INPUT: row})[0], dtype=np.float32).reshape(-1)
            for row in np.asarray(x, dtype=np.int64)
        ]
        if not rows:
            return np.zeros((0, 0), np.float32)
        return np.stack(rows)


def to_weights(model: Categoriser, cfg: ExperimentConfig) -> Any:
    """``model`` through ONNX and rayo-convert to its int8 ``categoriser.weights``
    (:class:`akili.export.onnx.Export`)."""
    from akili.export.onnx import weights_of_module

    return weights_of_module(
        model,
        example_input(cfg.features),
        [OUTPUT],
        RAYO_OPTIONS,
        layers=LAYERS,
        required=required_tensors(),
        onnx_name=ONNX_FILE,
    )


def parity_inputs(cfg: ExperimentConfig) -> tuple[IntArray, list[str]]:
    """What the shipped file is checked on: ``export.parity_rows`` transactions generated with
    the next data seed, and the app's feature vectors (``feature_vectors.v1.json``) when their
    feature config is this model's. Returns the rows and the vectors' ids."""
    from akili.features import encode_batch
    from akili.synthetic import generate
    from akili.vectors import default_path

    cats = default_categories()
    users = max(1, -(-cfg.export.parity_rows // cfg.data.transactions_per_user))
    data = cfg.data.model_copy(update={"seed": cfg.data.seed + 1, "n_users": users})
    rows = encode_batch(generate(data, cats), cfg.features, cats.labels).features
    rows = rows[: cfg.export.parity_rows]
    doc = json.loads(default_path().read_text(encoding="utf-8"))
    ids: list[str] = []
    if doc["feature_config"] == cfg.features.model_dump():
        vectors = [c for c in doc["cases"] if "features" in c.get("expected", {})]
        ids = [c["id"] for c in vectors]
        if vectors:
            rows = np.concatenate(
                [rows, np.asarray([c["expected"]["features"] for c in vectors], np.int32)]
            )
    return rows, ids


def check_parity(model: Categoriser, converted: Converted, cfg: ExperimentConfig) -> dict[str, Any]:
    """The torch model's float probabilities against the converted int8 file on
    :func:`parity_inputs`: the largest difference (within ``export.parity_tolerance``), its
    99.9th percentile (within ``export.parity_tolerance_p99_9``) and the labels, which must
    agree wherever torch's two best probabilities are more than ``export.parity_label_margin``
    apart (closer ones may rightly swap)."""
    x, ids = parity_inputs(cfg)
    want = predict(model, x)
    got = converted.run(x)
    tolerance = cfg.export.parity_tolerance
    margin = cfg.export.parity_label_margin
    diff = np.abs(got - want).max(-1)
    top = np.sort(want, axis=-1)
    clear = top[:, -1] - top[:, -2] > margin
    same = want.argmax(-1) == got.argmax(-1)
    vectors = len(ids)
    report: dict[str, Any] = {
        "rows": len(x) - vectors,
        "vectors": vectors,
        "runner": converted.on,
        "tolerance": tolerance,
        "tolerance_p99_9": cfg.export.parity_tolerance_p99_9,
        "label_margin": margin,
        "max_abs": round(float(diff.max()), 6),
        "p99_9_abs": round(float(np.quantile(diff, 0.999)), 6),
        "over_1e-3": int((diff > 1e-3).sum()),
        "labels": len(x),
        "labels_same": int(same.sum()),
        "labels_compared": int(clear.sum()),
        "labels_compared_same": int(same[clear].sum()),
        "vectors_max_abs": round(float(diff[len(x) - vectors :].max()), 6) if vectors else None,
    }
    report["within"] = bool(
        report["max_abs"] <= tolerance
        and report["p99_9_abs"] <= cfg.export.parity_tolerance_p99_9
        and report["labels_compared"] == report["labels_compared_same"]
    )
    return report


def _versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in ("torch", "onnx", "rayo-convert")}


def export_categoriser(
    model: Categoriser, cfg: ExperimentConfig, run_dir: Path, *, parity: bool = True
) -> Exported:
    """Write ``categoriser.weights``, ``categoriser.onnx`` and ``model-manifest.json`` (with the
    fine-tuning block when ``export.trainable``) to ``run_dir``, and ``export-report.json``
    with the parity check. Fails over the size budget or outside the parity tolerance."""
    from akili.export.fine_tune import fine_tune_block
    from akili.export.manifest import write_weights_manifest
    from akili.export.onnx import OPSET

    done = to_weights(model, cfg)
    data = done.data
    if len(data) > cfg.export.max_size_bytes:
        raise ModelTooLargeError(
            f"{WEIGHTS_FILE} is {len(data)} bytes; the budget is {cfg.export.max_size_bytes}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    rayo = rayo_dir()
    report: dict[str, Any] = {
        "weights_file": WEIGHTS_FILE,
        "weights_size_bytes": len(data),
        "weights_sha256": hashlib.sha256(data).hexdigest(),
        "parameters": parameter_count(model),
        "export": {"onnx_opset": OPSET, "onnx_exporter": "torchscript", **_versions()},
    }
    if parity:
        report["parity"] = check_parity(model, Converted(data, done.onnx, rayo), cfg)
    report_path = run_dir / "export-report.json"
    # The report is written either way, so a failed export can be read; the files only when
    # the converted file is within the tolerance.
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if parity and not report["parity"]["within"]:
        raise ValueError(f"{WEIGHTS_FILE} strays from the torch model: {report['parity']}")
    weights_path = run_dir / WEIGHTS_FILE
    weights_path.write_bytes(data)
    (run_dir / ONNX_FILE).write_bytes(done.onnx)
    fine_tune = fine_tune_block(model, cfg.export.fine_tune) if cfg.export.trainable else None
    manifest = write_weights_manifest(weights_path, done.onnx, ONNX_FILE, cfg.features, fine_tune)
    exported = describe(weights_path)
    metrics = {"model.size_bytes": float(exported.size_bytes)}
    if parity:
        metrics["model.parity_max_abs"] = float(report["parity"]["max_abs"])
    return Exported(
        model=exported,
        files=[weights_path, run_dir / ONNX_FILE, manifest, report_path],
        metrics=metrics,
        report=report,
    )


class _TorchTrainer:
    """Local training on one client's data, with plain SGD on the shared model (no optimiser
    state, so sharing it across clients is safe; each client sets the global weights first)."""

    def __init__(self, model: Categoriser, x: IntArray, y: IntArray, seed: int) -> None:
        self.model = model
        self.x = torch.from_numpy(np.asarray(x, dtype=np.int64))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    @property
    def num_examples(self) -> int:
        return len(self.y)

    def get_weights(self) -> Weights:
        return keras_layout(self.model)

    def set_weights(self, weights: Weights) -> None:
        set_keras_layout(self.model, weights)

    def fit(self, epochs: int, batch_size: int, learning_rate: float) -> dict[str, float]:
        torch.manual_seed(self._seed)  # dropout, per client
        optimiser = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        loss = float("nan")
        self.model.train()
        for _ in range(epochs):
            order = self._rng.permutation(len(self.y))
            for start in range(0, len(order), batch_size):
                idx = torch.from_numpy(order[start : start + batch_size])
                step: Any = torch.nn.functional.cross_entropy(
                    self.model.logits(self.x[idx]), self.y[idx]
                )
                optimiser.zero_grad()
                step.backward()
                optimiser.step()
                loss = float(step)
        self.model.eval()
        return {"loss": loss}

    def evaluate(self) -> tuple[float, dict[str, float]]:
        probs = predict(self.model, self.x.numpy())
        y = self.y.numpy()
        loss = float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-7)))
        return loss, {"accuracy": float((probs.argmax(axis=1) == y).mean())}


class TorchBackend:
    name = "torch"

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
        return fit(cfg, x, y, users=users, seed=seed, n_classes=n_classes, noise_seed=noise_seed)

    def predict(self, model: Any, x: IntArray) -> FloatArray:
        return predict(model, x)

    def export(self, model: Any, cfg: ExperimentConfig, run_dir: Path) -> Exported:
        return export_categoriser(model, cfg, run_dir)

    def run_exported(self, exported: ExportedModel, x: IntArray) -> FloatArray:
        onnx = exported.path.with_name(ONNX_FILE).read_bytes()
        return Converted(exported.path.read_bytes(), onnx, rayo_dir()).run(x)

    def shipped_predict(self, model: Any, cfg: ExperimentConfig, x: IntArray) -> FloatArray:
        done = to_weights(model, cfg)
        return Converted(done.data, done.onnx, rayo_dir()).run(x)

    def new_model(self, cfg: ExperimentConfig, n_classes: int, seed: int) -> Any:
        _deterministic(seed)
        return build_categoriser(cfg.features, cfg.model, n_classes, seed=seed)

    def get_weights(self, model: Any) -> Weights:
        return keras_layout(model)

    def set_weights(self, model: Any, weights: Weights) -> None:
        set_keras_layout(model, weights)

    def local_trainer(self, model: Any) -> Callable[[IntArray, IntArray, int], LocalTrainer]:
        def make(x: IntArray, y: IntArray, seed: int) -> LocalTrainer:
            return _TorchTrainer(model, x, y, seed)

        return make

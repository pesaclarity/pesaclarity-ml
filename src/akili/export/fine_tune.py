"""Fine-tuning the categoriser on the phone: the dense layers' SGD step, its spec and vectors.

The app fine-tunes the categoriser on the person's own confirmations without anything leaving
the phone (docs/FINE-TUNING.md). The Keras export did it with a second TFLite file holding
hand-written gradients (``categoriser-train.tflite``, removed with TensorFlow in stage 5; at
the ``tflite-archive`` tag). The PyTorch export ships no second file (decision 0023): the step
is written down here, with conformance vectors whose reference torch computes, and the app's
own code (Dart, or rayo once it has a backward pass) implements it.

The step trains ``hidden`` and ``category`` (:data:`TRAINED`) and keeps everything before them
frozen: ``f``, what ``hidden`` reads, comes from the shipped ``categoriser.weights`` as rayo
runs it (the masked mean of the token embeddings and the five context embeddings). On a batch
of ``n`` rows ``f`` with labels ``y`` and a learning rate ``lr``::

    z1 = f W1^T + b1,  h = relu(z1),  z2 = h W2^T + b2,  p = softmax(z2)
    loss = mean over the batch of -log p[y]                       (reported, before the update)
    g2 = (p - onehot(y)) / n,   dW2 = g2^T h,   db2 = sum of g2's rows
    gh = (g2 W2) * (z1 > 0),    dW1 = gh^T f,   db1 = sum of gh's rows
    W1 -= lr dW1,  b1 -= lr db1,  W2 -= lr dW2,  b2 -= lr db2     (all from the weights before)

``W1`` is ``hidden.kernel`` ``[hidden, in]`` and ``W2`` ``category.kernel`` ``[classes, hidden]``,
in rayo's layout. They start as the shipped file's values (an int8 kernel dequantised) and are
kept as float32 once tuned. No dropout on the phone.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import torch

from akili.config import FineTuneConfig

SPEC_VERSION = 1
"""Version of the step above. The app refuses a manifest's ``fine_tune.spec_version`` it does
not implement."""
TRAINED = ("hidden.kernel", "hidden.bias", "category.kernel", "category.bias")
"""The tensors the step changes, in the order it lists them."""
VECTOR_FILE = f"fine_tune_vectors.v{SPEC_VERSION}.json"
TOLERANCE = 1e-5
"""Float32 implementations must give each loss and weight within this of the vectors."""


def fine_tune_block(module: Any, settings: FineTuneConfig) -> dict[str, Any]:
    """The manifest's ``fine_tune`` block for the categoriser ``module``."""
    shapes = {
        "hidden.kernel": list(module.hidden.weight.shape),
        "hidden.bias": list(module.hidden.bias.shape),
        "category.kernel": list(module.category.weight.shape),
        "category.bias": list(module.category.bias.shape),
    }
    return {
        "spec_version": SPEC_VERSION,
        "trains": list(TRAINED),
        "input": "hidden",
        "weights": shapes,
        "settings": settings.model_dump(),
    }


def step(
    weights: dict[str, np.ndarray], f: np.ndarray, y: np.ndarray, lr: float
) -> tuple[float, dict[str, np.ndarray]]:
    """One step of the spec, from torch's automatic gradients in float32: the loss before the
    update and the four tensors after it."""
    w = {k: torch.tensor(np.asarray(v, np.float32), requires_grad=True) for k, v in weights.items()}
    ft = torch.tensor(np.asarray(f, np.float32))
    yt = torch.tensor(np.asarray(y, np.int64))
    h = torch.relu(ft @ w["hidden.kernel"].T + w["hidden.bias"])
    logits = h @ w["category.kernel"].T + w["category.bias"]
    loss = torch.nn.functional.cross_entropy(logits, yt)
    grads = torch.autograd.grad(loss, [w[k] for k in TRAINED])
    after = {
        k: (w[k] - float(np.float32(lr)) * g).detach().numpy()
        for k, g in zip(TRAINED, grads, strict=True)
    }
    return float(loss.detach()), after


def manual_step(
    weights: dict[str, np.ndarray], f: np.ndarray, y: np.ndarray, lr: float
) -> tuple[float, dict[str, np.ndarray]]:
    """The same step written out as the spec gives it, in float64 (what the vectors are
    checked against)."""
    w1, b1, w2, b2 = (np.asarray(weights[k], np.float64) for k in TRAINED)
    x = np.asarray(f, np.float64)
    n = len(y)
    z1 = x @ w1.T + b1
    h = np.maximum(z1, 0.0)
    z2 = h @ w2.T + b2
    z2 = z2 - z2.max(axis=1, keepdims=True)
    p = np.exp(z2) / np.exp(z2).sum(axis=1, keepdims=True)
    onehot = np.eye(w2.shape[0])[np.asarray(y)]
    loss = float(-np.log(p[np.arange(n), y]).mean())
    g2 = (p - onehot) / n
    gh = (g2 @ w2) * (z1 > 0)
    grads = (gh.T @ x, gh.sum(0), g2.T @ h, g2.sum(0))
    return loss, {k: (v - lr * g) for k, v, g in zip(TRAINED, (w1, b1, w2, b2), grads, strict=True)}


def _case(
    case_id: str,
    note: str,
    sizes: tuple[int, int, int],
    batches: list[tuple[int, float]],
    *,
    seed: int,
    dead_unit: bool = False,
) -> dict[str, Any]:
    """A case: random float32 weights and rows of the given sizes (in, hidden, classes), then
    one step per batch (rows, learning rate), each from the weights the last one left."""
    rng = np.random.default_rng(seed)
    n_in, n_hidden, n_classes = sizes
    w = {
        "hidden.kernel": rng.normal(0, 0.5, (n_hidden, n_in)),
        "hidden.bias": rng.normal(0, 0.1, n_hidden),
        "category.kernel": rng.normal(0, 0.5, (n_classes, n_hidden)),
        "category.bias": rng.normal(0, 0.1, n_classes),
    }
    if dead_unit:  # the first hidden unit is off for every row, so its weights must not move
        w["hidden.bias"][0] = -100.0
    w = {k: v.astype(np.float32) for k, v in w.items()}
    start = {k: v.tolist() for k, v in w.items()}
    steps = []
    for rows, lr in batches:
        f = rng.normal(0, 1, (rows, n_in)).astype(np.float32)
        y = rng.integers(0, n_classes, rows)
        loss, w = step(w, f, y, lr)
        steps.append(
            {
                "f": f.tolist(),
                "y": y.tolist(),
                "lr": lr,
                "loss": loss,
                "after": {k: v.tolist() for k, v in w.items()},
            }
        )
    return {"id": case_id, "note": note, "sizes": list(sizes), "weights": start, "steps": steps}


def build_vectors() -> dict[str, Any]:
    cases = [
        _case("one_row", "a batch of one", (6, 4, 3), [(1, 0.05)], seed=1),
        _case("batch", "five rows, two with the same label", (6, 4, 3), [(5, 0.05)], seed=2),
        _case(
            "dead_unit",
            "a hidden unit off for every row keeps its kernel row and bias",
            (6, 4, 3),
            [(4, 0.1)],
            seed=3,
            dead_unit=True,
        ),
        _case(
            "three_steps",
            "three batches in a row, each step from the weights the last one left",
            (5, 3, 4),
            [(2, 0.5), (3, 0.5), (1, 0.5)],
            seed=4,
        ),
    ]
    return {
        "spec_version": SPEC_VERSION,
        "spec": "docs/FINE-TUNING.md",
        "trains": list(TRAINED),
        "layouts": {
            "hidden.kernel": "[hidden, in]",
            "category.kernel": "[classes, hidden]",
            "f": "[rows, in]",
        },
        "tolerance": TOLERANCE,
        "cases": cases,
    }


def render(doc: dict[str, Any]) -> str:
    """Canonical text of the vector file: ASCII-only JSON, two-space indent, trailing newline."""
    return json.dumps(doc, indent=2, ensure_ascii=True) + "\n"


def default_path() -> Path:
    """The committed vector file inside the source tree."""
    return Path(str(resources.files("akili.resources").joinpath(VECTOR_FILE)))


def write_vectors(path: str | Path | None = None) -> Path:
    out = Path(path) if path is not None else default_path()
    out.write_text(render(build_vectors()), encoding="utf-8", newline="\n")
    return out


def differences(committed: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Where ``fresh`` (rebuilt) differs from ``committed``: any field, or a number by more
    than a float32 rounding (the reference is torch's, whose last bit may differ between
    machines)."""
    out: list[str] = []

    def walk(path: str, a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                out.append(f"{path}: fields {sorted(a)} against {sorted(b)}")
                return
            for k in a:
                walk(f"{path}.{k}", a[k], b[k])
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append(f"{path}: {len(a)} items against {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(f"{path}[{i}]", x, y)
        elif isinstance(a, float) or isinstance(b, float):
            if not np.isclose(float(a), float(b), rtol=1e-6, atol=1e-7):
                out.append(f"{path}: {a} against {b}")
        elif a != b:
            out.append(f"{path}: {a!r} against {b!r}")

    walk("", committed, fresh)
    return out

"""Train, evaluate and export the message reader in PyTorch (needs the [torch] extra).

The network is :class:`akili.reader.model.Reader`: token and shape embeddings, two 1-D
convolutions that see two tokens either side, a dilated convolution for a whole clause, then a
label per token and, from the pooled tokens, the message kind. 179,627 parameters; the int8
``reader.weights`` file rayo runs is about 188 KB.

The release path (pesaclarity/docs, decision 0023): torch trains it, ``torch.onnx`` exports it
and rayo-convert quantises it to int8 (:mod:`akili.export.onnx`); every score is of that file
as rayo runs it. Evaluation holds each sentence layout out of training in turn (see
:func:`evaluate_unseen_layouts`) and scores whole fields: a field counts as read only when
every token of it, and nothing else, carries its label.

The Keras reader this replaces stays in :mod:`akili.reader.keras_train` until TensorFlow is
removed (stage 5).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from akili._optional import require
from akili.reader.synthetic import (
    FIELD_LABELS,
    KIND_LABELS,
    LAYOUTS,
    Field,
    Message,
    generate,
    to_arrays,
)
from akili.reader.tokens import MAX_TOKENS, reader_manifest

MANIFEST_FILE = "reader-manifest.json"
REPORT_FILE = "reader-report.json"

PARITY_MESSAGES = 1000  # generated with the next seed


@dataclass(frozen=True, slots=True)
class ReaderConfig:
    messages: int = 30_000
    holdout_fraction: float = 0.2
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.003
    token_dim: int = 32
    shape_dim: int = 8
    filters: int = 64
    seed: int = 0
    max_size_bytes: int = 512_000
    # Two changes from the Keras reader's recipe, which lift every unseen-layout score
    # (docs/READER.md): the learning rate falls to 0 along a cosine over the run, and the
    # targets are smoothed by 0.1. "constant" and 0.0 give the Keras recipe.
    lr_schedule: str = "cosine"
    label_smoothing: float = 0.1
    threads: int = 1  # torch's CPU threads: fixed, so a run is repeatable
    # The largest difference between the torch model's float probabilities and rayo's int8
    # file on the parity messages and the reader vectors, per output. Decision 0023 proposed
    # 1e-2; on the trained reader 99.9% of tokens are within 8.6e-3, but a token whose two
    # best fields are close moves by up to 1.6e-2, and the kind of the vector of 96 times
    # "word", which the reader is unsure of (0.66 against 0.29), by 1.17e-2: int8 shifts its
    # log-probabilities no more than a generated message's (3.5e-2 against 2.9e-2), but an
    # unsure softmax turns that into twice the difference (docs/READER.md).
    parity_tolerance_fields: float = 2e-2
    parity_tolerance_kind: float = 2e-2


def _spans(labels: np.ndarray) -> set[tuple[int, int, int]]:
    """Maximal runs of the same non-OTHER label, as (label, start, end)."""
    spans = set()
    start = None
    for i in range(len(labels) + 1):
        current = int(labels[i]) if i < len(labels) else -1
        previous = int(labels[i - 1]) if i > 0 else -1
        if start is not None and current != previous:
            if previous > 0:
                spans.add((previous, start, i))
            start = None
        if start is None and current > 0:
            start = i
    return spans


def field_scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-field precision, recall and F1 over exact spans. Padding (-1) is ignored."""
    tp = np.zeros(len(FIELD_LABELS))
    fp = np.zeros(len(FIELD_LABELS))
    fn = np.zeros(len(FIELD_LABELS))
    for t_row, p_row in zip(truth, predicted, strict=True):
        n = int((t_row >= 0).sum())
        true_spans = _spans(t_row[:n])
        pred_spans = _spans(p_row[:n])
        for label, *_ in true_spans & pred_spans:
            tp[label] += 1
        for label, *_ in pred_spans - true_spans:
            fp[label] += 1
        for label, *_ in true_spans - pred_spans:
            fn[label] += 1
    out = {}
    for f in Field:
        if f is Field.OTHER:
            continue
        p = tp[f] / max(tp[f] + fp[f], 1)
        r = tp[f] / max(tp[f] + fn[f], 1)
        out[f.name.lower()] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(2 * p * r / max(p + r, 1e-9)), 4),
            "support": int(tp[f] + fn[f]),
        }
    return out


def scores(
    fields_p: np.ndarray, kind_p: np.ndarray, y: np.ndarray, k: np.ndarray
) -> dict[str, Any]:
    return {
        "kind_accuracy": round(float((kind_p.argmax(-1) == k).mean()), 4),
        "fields": field_scores(y, fields_p.argmax(-1)),
    }


def layout_of(style: str) -> str | None:
    parts = style.split("|")
    return parts[5] if len(parts) > 6 else None


def mean_scores(per_layout: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The mean over held-out layouts of each field's F1 (where it occurs) and of kind."""
    fields = [f.name.lower() for f in Field if f is not Field.OTHER]
    mean = {
        name: round(
            float(
                np.mean(
                    [
                        r["fields"][name]["f1"]
                        for r in per_layout.values()
                        if r["fields"][name]["support"]
                    ]
                )
            ),
            4,
        )
        for name in fields
    }
    return {
        "mean_f1": mean,
        "mean_kind_accuracy": round(
            float(np.mean([r["kind_accuracy"] for r in per_layout.values()])), 4
        ),
        "per_layout": per_layout,
    }


# The torch path.


def _feeds(x: np.ndarray) -> list[dict[str, np.ndarray]]:
    from akili.reader.model import split

    out = []
    for row in x.astype(np.int64):
        tokens, shapes = split(row)
        out.append({"tokens": tokens, "shapes": shapes})
    return out


def fit(cfg: ReaderConfig, messages: list[Message]) -> Any:
    """A :class:`~akili.reader.model.Reader` trained on ``messages`` with the Keras reader's
    recipe: Adam (Keras' epsilon), the batch shuffled each epoch, cross-entropy per token
    (padding weighted 0, averaged over every position, as Keras' ``sum_over_batch_size``)
    plus cross-entropy of the kind; with ``cfg``'s learning-rate schedule and label
    smoothing on top. Seeded and deterministic."""
    from akili.export.onnx import deterministic
    from akili.reader.model import Reader, split

    torch = require("torch", "torch")
    deterministic(cfg.seed, cfg.threads)
    model = Reader(cfg.token_dim, cfg.shape_dim, cfg.filters)
    x, y, k = to_arrays(messages)
    tokens, shapes = (torch.from_numpy(a.astype(np.int64)) for a in split(x))
    labels = torch.from_numpy(np.maximum(y, 0).astype(np.int64))
    weight = torch.from_numpy((y >= 0).astype(np.float32))
    kinds = torch.from_numpy(k.astype(np.int64))
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-7)
    steps = cfg.epochs * -(-len(x) // cfg.batch_size)
    schedule = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
        if cfg.lr_schedule == "cosine"
        else None
    )
    order = torch.Generator().manual_seed(cfg.seed)
    ce = torch.nn.functional.cross_entropy
    smooth = cfg.label_smoothing
    model.train()
    for _ in range(cfg.epochs):
        for batch in torch.randperm(len(x), generator=order).split(cfg.batch_size):
            fields, kind = model.logits(tokens[batch], shapes[batch])
            per_token = ce(
                fields.reshape(-1, len(FIELD_LABELS)),
                labels[batch].reshape(-1),
                reduction="none",
                label_smoothing=smooth,
            )
            loss = (per_token * weight[batch].reshape(-1)).sum() / per_token.numel()
            loss = loss + ce(kind, kinds[batch], label_smoothing=smooth)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if schedule is not None:
                schedule.step()
    model.eval()
    return model


def export_module(model: Any) -> Any:
    """``model`` through ONNX and rayo-convert to its int8 ``reader.weights``
    (:class:`akili.export.onnx.Export`)."""
    from akili.export.onnx import weights_of_module
    from akili.reader.model import (
        INPUTS,
        LAYERS,
        ONNX_FILE,
        OUTPUTS,
        RAYO_OPTIONS,
        required_tensors,
    )

    torch = require("torch", "torch")
    example = {name: torch.zeros((1, MAX_TOKENS), dtype=torch.int64) for name in INPUTS}
    return weights_of_module(
        model,
        example,
        OUTPUTS,
        RAYO_OPTIONS,
        layers=LAYERS,
        required=required_tensors(),
        onnx_name=ONNX_FILE,
    )


def runner(done: Any, rayo_dir: str | Path | None) -> Any:
    """The converted file on rayo's C library when ``rayo_dir`` is a rayo checkout, else on
    rayo-convert's NumPy reference (the same int8 meaning, which rayo is checked against)."""
    from akili.export.onnx import RayoRunner, ReferenceRunner
    from akili.reader.model import RAYO_OPTIONS

    if rayo_dir:
        return RayoRunner(done.data, rayo_dir)
    return ReferenceRunner(done.onnx, RAYO_OPTIONS)


def run_converted(converted: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The converted file's fields ``[n, 96, 8]`` and kind ``[n, 3]`` for inputs ``x``."""
    fields, kinds = [], []
    for feeds in _feeds(x):
        f, k = converted.run(feeds, {"T": MAX_TOKENS})
        fields.append(np.asarray(f).reshape(MAX_TOKENS, len(FIELD_LABELS)))
        kinds.append(np.asarray(k).reshape(len(KIND_LABELS)))
    return np.stack(fields), np.stack(kinds)


def evaluate_unseen_layouts(
    cfg: ReaderConfig, messages: list[Message], rayo_dir: str | Path | None = None
) -> dict[str, Any]:
    """Hold each sentence layout out in turn: train without it, score the converted int8 file
    on it. This is the honest measure for wording the review inbox has never seen."""
    per_layout = {}
    for layout in LAYOUTS:
        train = [m for m in messages if layout_of(m.style) != layout]
        test = [m for m in messages if layout_of(m.style) == layout]
        xt, yt, kt = to_arrays(test)
        converted = runner(export_module(fit(cfg, train)), rayo_dir)
        per_layout[layout] = scores(*run_converted(converted, xt), yt, kt)
    return mean_scores(per_layout)


def torch_outputs(model: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The torch model's float fields ``[n, 96, 8]`` and kind ``[n, 3]`` for inputs ``x``."""
    from akili.reader.model import split

    torch = require("torch", "torch")
    model.eval()
    with torch.no_grad():
        tokens, shapes = (torch.from_numpy(a.astype(np.int64)) for a in split(x))
        fields, kind = model(tokens, shapes)
    return fields.numpy(), kind.numpy()


def vector_inputs() -> tuple[list[str], np.ndarray]:
    """The ids and inputs ``[n, 192]`` of the app's reader vectors
    (``reader_vectors.v1.json``): the edge cases the generator does not write, such as an
    empty message and one of 120 words cut to 96 tokens."""
    from akili.reader import vectors

    cases = json.loads(vectors.default_path().read_text(encoding="utf-8"))["cases"]
    x = np.asarray([c["expected"]["input"] for c in cases], dtype=np.int64)
    return [c["id"] for c in cases], x


def check_parity(
    model: Any, done: Any, cfg: ReaderConfig, rayo_dir: str | Path | None = None
) -> dict[str, Any]:
    """The torch model's float probabilities against the converted int8 file on
    :data:`PARITY_MESSAGES` messages generated with the next seed and on the reader vectors'
    messages (:func:`vector_inputs`): the largest difference per output (within ``cfg``'s
    tolerance for each), and the labels (each token's field, each message's kind), which
    must agree wherever torch's two best probabilities are more than twice the fields'
    tolerance apart. A generated message's fields are compared on its tokens; a vector's on
    its tokens too (a shape id above 0), as it has no field labels. ``vectors`` gives each
    vector's largest difference."""
    x, y, _ = to_arrays(generate(PARITY_MESSAGES, seed=cfg.seed + 1))
    ids, xv = vector_inputs()
    converted = runner(done, rayo_dir)
    tolerance = {"fields": cfg.parity_tolerance_fields, "kind": cfg.parity_tolerance_kind}
    margin = 2 * cfg.parity_tolerance_fields
    out: dict[str, Any] = {
        "messages": PARITY_MESSAGES,
        "vectors": len(ids),
        "runner": "rayo" if rayo_dir else "rayo-convert reference",
        "tolerance": tolerance,
        "label_margin": margin,
    }
    tokens = np.concatenate([y >= 0, xv[:, MAX_TOKENS:] > 0])
    want = [
        np.concatenate(p)
        for p in zip(torch_outputs(model, x), torch_outputs(model, xv), strict=True)
    ]
    got = [
        np.concatenate(p)
        for p in zip(run_converted(converted, x), run_converted(converted, xv), strict=True)
    ]
    within = True
    per_vector: dict[str, dict[str, float]] = {i: {} for i in ids}
    for name, w, g in zip(("fields", "kind"), want, got, strict=True):
        diff = np.abs(g - w).max(-1)
        top = np.sort(w, axis=-1)
        clear = top[..., -1] - top[..., -2] > margin
        same = w.argmax(-1) == g.argmax(-1)
        for i, d, keep in zip(ids, diff[len(x) :], tokens[len(x) :], strict=True):
            if name == "kind":
                per_vector[i][name] = round(float(d), 6)
            else:  # the message's tokens; none for the empty one
                per_vector[i][name] = round(float(d[keep].max()), 6) if keep.any() else 0.0
        if name == "fields":  # the message's tokens, not the padding
            diff, clear, same = diff[tokens], clear[tokens], same[tokens]
        entry = {
            "max_abs": round(float(diff.max()), 6),
            "p99_9_abs": round(float(np.quantile(diff, 0.999)), 6),
            "over_1e-2": int((diff > 1e-2).sum()),
            "labels": int(diff.size),
            "labels_same": int(same.sum()),
            "labels_compared": int(clear.sum()),
            "labels_compared_same": int(same[clear].sum()),
        }
        out[name] = entry
        within &= entry["max_abs"] <= tolerance[name]
        within &= entry["labels_compared"] == entry["labels_compared_same"]
    out["per_vector"] = per_vector
    out["within"] = bool(within)
    return out


def _versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in ("torch", "onnx", "rayo-convert")}


def manifest(data: bytes, onnx: bytes) -> dict[str, Any]:
    """``reader-manifest.json`` for the weights file ``data`` exported from ``onnx``."""
    from akili.export.weights import FORMAT_VERSION
    from akili.reader.model import ONNX_FILE, WEIGHTS_FILE

    return {
        "weights_file": WEIGHTS_FILE,
        "weights_sha256": hashlib.sha256(data).hexdigest(),
        "weights_size_bytes": len(data),
        "weights_format_version": FORMAT_VERSION,
        "source": {
            "file": ONNX_FILE,
            "sha256": hashlib.sha256(onnx).hexdigest(),
            "size_bytes": len(onnx),
        },
        **reader_manifest(),
        "rayo_inputs": {
            "tokens": {"dtype": "int32", "shape": [MAX_TOKENS], "from": "message[0:96]"},
            "shapes": {"dtype": "int32", "shape": [MAX_TOKENS], "from": "message[96:192]"},
        },
        "outputs": {
            "fields": {"shape": [-1, MAX_TOKENS, len(FIELD_LABELS)], "labels": FIELD_LABELS},
            "kind": {"shape": [-1, len(KIND_LABELS)], "labels": KIND_LABELS},
        },
        "trained_on": "synthetic messages only",
    }


def train_reader(
    cfg: ReaderConfig,
    out_dir: str | Path,
    evaluate: bool = True,
    rayo_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score on held-out layouts (unless ``evaluate`` is off), then train on every message and
    export ``reader.weights`` (int8, with its rayo graph), the ONNX file it was converted from
    and the manifest. Fails if the weights are over the size budget or the converted file
    strays from the torch model by more than the tolerance. Returns the report.

    ``rayo_dir`` (by default ``RAYO_DIR``): a rayo checkout whose C library runs the scores;
    without one, rayo-convert's NumPy reference runs them."""
    from akili.export.onnx import OPSET
    from akili.export.size import ModelTooLargeError
    from akili.reader.model import ONNX_FILE, WEIGHTS_FILE, parameter_count

    rayo = rayo_dir if rayo_dir is not None else os.environ.get("RAYO_DIR") or None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    messages = generate(cfg.messages, seed=cfg.seed)
    unseen = evaluate_unseen_layouts(cfg, messages, rayo) if evaluate else None

    model = fit(cfg, messages)
    done = export_module(model)
    data = done.data
    if len(data) > cfg.max_size_bytes:
        raise ModelTooLargeError(
            f"{WEIGHTS_FILE} is {len(data)} bytes; the budget is {cfg.max_size_bytes}"
        )
    parity_report = check_parity(model, done, cfg, rayo)
    report = {
        "config": asdict(cfg),
        "unseen_layouts": unseen,
        "scored_on": f"{WEIGHTS_FILE} (int8) on "
        + ("rayo's C library" if rayo else "rayo-convert's reference"),
        "parameters": parameter_count(model),
        "weights_size_bytes": len(data),
        "weights_sha256": hashlib.sha256(data).hexdigest(),
        "parity": parity_report,
        "export": {"onnx_opset": OPSET, "onnx_exporter": "torchscript", **_versions()},
        "note": "Scores are on synthetic messages; real messages will score lower.",
    }
    # The report is written either way, so a failed run can be read; the files only when the
    # converted file is within the tolerance.
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not parity_report["within"]:
        raise ValueError(f"{WEIGHTS_FILE} strays from the torch model: {parity_report}")
    (out / WEIGHTS_FILE).write_bytes(data)
    (out / ONNX_FILE).write_bytes(done.onnx)
    doc = manifest(data, done.onnx)
    (out / MANIFEST_FILE).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return report

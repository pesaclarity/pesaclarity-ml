"""The message reader's training config and scores, shared by its trainers.

Evaluation holds each sentence layout out of training in turn and scores whole fields: a field
counts as read only when every token of it, and nothing else, carries its label. The Keras
trainer and TFLite export are :mod:`akili.reader.keras_train`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from akili.reader.synthetic import (
    FIELD_LABELS,
    Field,
)

MANIFEST_FILE = "reader-manifest.json"
REPORT_FILE = "reader-report.json"


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

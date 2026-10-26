"""Run the membership-inference audit against a trained categoriser (needs the [train] extra).

Setup, following Carlini et al. 2022 ("Membership Inference Attacks From First Principles"):

1. **Candidates.** Up to ``audit.max_candidates`` held-out validation examples (non-members) and
   as many random training examples (members).
2. **Target scores.** The target model, by default its exported TFLite file, gives each candidate
   the probability of its true label.
3. **Shadow models.** ``audit.shadow_models`` models are trained with the target's exact
   procedure (same config, plain or DP-SGD) on as many examples as the target saw: each
   candidate is IN for exactly half of the shadows, and the rest is filled from the target's
   non-candidate data. This is a strong attacker who knows the training procedure and most of the
   data, which is the point of an audit.
4. **Attacks.** ``loss`` (threshold on the target's loss), ``lira-online`` (IN and OUT shadows)
   and ``lira-offline`` (OUT shadows only); see :mod:`akili.privacy.audit`.

Everything is synthetic; nothing here touches real data.
"""

from __future__ import annotations

import contextlib
import gc
import io
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from akili.config import ExperimentConfig
from akili.export.tflite import run_tflite, to_tflite_bytes
from akili.privacy.audit import (
    AuditLimits,
    AuditReport,
    AuditResult,
    build_report,
    lira_scores,
    logit_confidence,
    loss_scores,
    result_from_scores,
)

IndexArray = npt.NDArray[np.int64]


def effective_limits(cfg: ExperimentConfig) -> AuditLimits:
    """The track's limits, tightened (never loosened) by the config's ``audit`` section."""
    a = cfg.audit
    return AuditLimits.from_policy(cfg.policy).tightened(
        max_auc=a.max_attack_auc,
        max_advantage=a.max_attack_advantage,
        max_tpr_at_1pct_fpr=a.max_attack_tpr_at_1pct_fpr,
        max_tpr_at_0_1pct_fpr=a.max_attack_tpr_at_0_1pct_fpr,
        min_examples=a.min_audit_examples,
    )


def choose_candidates(
    train: IndexArray, val: IndexArray, max_candidates: int, rng: np.random.Generator
) -> tuple[IndexArray, IndexArray]:
    """Equal numbers of members (from ``train``) and non-members (from ``val``)."""
    m = min(len(train), len(val), max_candidates)
    members = rng.choice(train, size=m, replace=False)
    non_members = rng.choice(val, size=m, replace=False)
    return members, non_members


def shadow_inclusion(k: int, n: int, rng: np.random.Generator) -> npt.NDArray[np.bool_]:
    """``[k, n]`` mask with each column IN for exactly ``k / 2`` shadows, chosen at random."""
    ranks = np.argsort(rng.random((k, n)), axis=0)
    return np.asarray(ranks < k // 2)


def _predict(model: Any, x: npt.NDArray[np.int32], exported: bool) -> npt.NDArray[np.float32]:
    if not exported:
        return np.asarray(model.predict(x, verbose=0), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shadow.tflite"
        # The converter prints a SavedModel summary per call; keep the CLI output readable.
        with contextlib.redirect_stdout(io.StringIO()):
            path.write_bytes(to_tflite_bytes(model, quantize=True))
        return run_tflite(path, x)


def run_audit(
    cfg: ExperimentConfig,
    x: npt.NDArray[np.int32],
    y: npt.NDArray[np.int32],
    train: IndexArray,
    val: IndexArray,
    *,
    target: Any,
    exported: Path | None = None,
    epsilon: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> AuditReport:
    """Attack ``target`` (trained on ``x[train]``) and check the results against the limits."""
    from akili.train.pipeline import fit_categoriser

    say = progress or (lambda _msg: None)
    audit = cfg.audit
    n_classes = int(target.output_shape[-1])
    rng = np.random.default_rng(audit.seed)
    members, non_members = choose_candidates(train, val, audit.max_candidates, rng)
    candidates = np.concatenate([members, non_members])
    is_member = np.r_[np.ones(len(members), bool), np.zeros(len(non_members), bool)]
    xc, yc = x[candidates], y[candidates]

    use_tflite = audit.score_exported and exported is not None
    if use_tflite:
        assert exported is not None
        target_probs = run_tflite(exported, xc)
    else:
        target_probs = np.asarray(target.predict(xc, verbose=0), dtype=np.float32)

    k = audit.shadow_models
    in_mask = shadow_inclusion(k, len(candidates), rng)
    candidate_set = set(candidates.tolist())
    rest = np.array([i for i in range(len(y)) if i not in candidate_set], dtype=np.int64)
    fill = len(train) - len(members)
    if fill > len(rest):
        raise ValueError("not enough non-candidate data to fill the shadow training sets")

    shadow_stats = np.zeros((k, len(candidates)), dtype=np.float64)
    for s in range(k):
        say(f"shadow model {s + 1}/{k}")
        data = np.concatenate([candidates[in_mask[s]], rng.choice(rest, fill, replace=False)])
        shadow = fit_categoriser(
            cfg, x[data], y[data], seed=audit.seed + s + 1, n_classes=n_classes
        )
        shadow_stats[s] = logit_confidence(_predict(shadow.model, xc, use_tflite), yc)
        del shadow
        gc.collect()

    target_stat = logit_confidence(target_probs, yc)
    scores = {
        "loss": loss_scores(target_probs, yc),
        "lira-online": lira_scores(target_stat, shadow_stats, in_mask, online=True),
        "lira-offline": lira_scores(target_stat, shadow_stats, in_mask, online=False),
    }
    results: list[AuditResult] = [
        result_from_scores(name, s[is_member], s[~is_member]) for name, s in scores.items()
    ]
    return build_report(
        cfg.name, results, effective_limits(cfg), dp=cfg.dp.enabled, epsilon=epsilon
    )


def quick_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """A small, CPU-friendly version of ``cfg`` for CI and tests (seconds to a minute or two).

    Too few candidates to meet any track's ``min_audit_examples``, so a quick audit never passes
    the gate; it checks that the attack runs and moves in the right direction.
    """
    return cfg.model_copy(
        update={
            "name": f"{cfg.name}-audit-quick",
            "data": cfg.data.model_copy(update={"n_users": 40, "transactions_per_user": 40}),
            "audit": cfg.audit.model_copy(
                update={"enabled": True, "shadow_models": 4, "max_candidates": 300}
            ),
        }
    )

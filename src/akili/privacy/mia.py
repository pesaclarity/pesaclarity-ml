"""Run the membership-inference audit against a trained categoriser.

The shadow models are trained, and every model scored, by the categoriser's framework (a
:class:`~akili.train.backend.Backend`: PyTorch, or Keras with ``--keras``).

Setup, following Carlini et al. 2022 ("Membership Inference Attacks From First Principles"):

1. **Candidates.** Up to ``audit.max_candidates`` held-out validation examples (non-members) and
   as many random training examples (members).
2. **Target scores.** The target model, by default the file that ships (PyTorch: the rayo int8
   ``categoriser.weights``; Keras: its TFLite file), gives each candidate the probability of
   its true label.
3. **Shadow models.** ``audit.shadow_models`` models are trained with the target's exact
   procedure (same config, plain or DP-SGD) on as many examples as the target saw: each
   candidate is IN for exactly half of the shadows, and the rest is filled from the target's
   non-candidate data. With ``audit.score_exported`` each shadow is exported and scored the same
   way as the target. This is a strong attacker who knows the training procedure and most of the
   data, which is the point of an audit.
4. **Attacks.** ``loss`` (threshold on the target's loss), ``lira-online`` (IN and OUT shadows)
   and ``lira-offline`` (OUT shadows only); see :mod:`akili.privacy.audit`.

Everything is synthetic; nothing here touches real data.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from akili.config import ExperimentConfig
from akili.export.size import ExportedModel
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
from akili.privacy.ledger import Ledger, Purpose

if TYPE_CHECKING:
    from akili.train.backend import Backend

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


def run_audit(
    cfg: ExperimentConfig,
    x: npt.NDArray[np.int32],
    y: npt.NDArray[np.int32],
    train: IndexArray,
    val: IndexArray,
    *,
    users: npt.ArrayLike,
    target: Any,
    exported: ExportedModel | None = None,
    epsilon: float | None = None,
    ledger: Ledger | None = None,
    progress: Callable[[str], None] | None = None,
    backend: Backend | None = None,
) -> AuditReport:
    """Attack ``target`` (trained on ``x[train]``) and check the results against the limits.

    The shadow models train on the same records, held-out ones included, so with DP each one
    is recorded in ``ledger`` as an audit run on this data. On real data, audit a separate
    stand-in dataset instead, or the audit spends the budget the release needs.
    """
    from akili.train.pipeline import backend_for

    impl = backend or backend_for()
    say = progress or (lambda _msg: None)
    audit = cfg.audit
    rng = np.random.default_rng(audit.seed)
    members, non_members = choose_candidates(train, val, audit.max_candidates, rng)
    candidates = np.concatenate([members, non_members])
    is_member = np.r_[np.ones(len(members), bool), np.zeros(len(non_members), bool)]
    xc, yc = x[candidates], y[candidates]

    shipped = audit.score_exported and exported is not None
    if exported is not None and shipped:
        target_probs = impl.run_exported(exported, xc)
    else:
        target_probs = impl.predict(target, xc)
    n_classes = int(target_probs.shape[-1])

    k = audit.shadow_models
    in_mask = shadow_inclusion(k, len(candidates), rng)
    candidate_set = set(candidates.tolist())
    rest = np.array([i for i in range(len(y)) if i not in candidate_set], dtype=np.int64)
    fill = len(train) - len(members)
    if fill > len(rest):
        raise ValueError("not enough non-candidate data to fill the shadow training sets")

    people = np.asarray(users)
    noise_seed = cfg.dp.noise_seed
    shadow_stats = np.zeros((k, len(candidates)), dtype=np.float64)
    for s in range(k):
        say(f"shadow model {s + 1}/{k}")
        data = np.concatenate([candidates[in_mask[s]], rng.choice(rest, fill, replace=False)])
        shadow = impl.fit(
            cfg,
            x[data],
            y[data],
            users=people[data],
            seed=audit.seed + s + 1,
            n_classes=n_classes,
            noise_seed=None if noise_seed is None else noise_seed + s + 1,
        )
        if ledger is not None and shadow.privacy is not None:
            ledger.record(
                cfg.dataset_id,
                shadow.privacy,
                run=f"{cfg.name}/shadow-{s + 1}",
                purpose=Purpose.AUDIT,
            )
        probs = (
            impl.shipped_predict(shadow.model, cfg, xc)
            if shipped
            else impl.predict(shadow.model, xc)
        )
        shadow_stats[s] = logit_confidence(probs, yc)
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

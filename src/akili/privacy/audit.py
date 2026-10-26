"""Membership-inference audit: attack scores, metrics and limits (core, numpy only).

An attack gives each candidate example a score, higher meaning "more likely trained on". The
audit compares the scores of real members (the training set) with held-out non-members and
reports how well they separate:

* **AUC**: probability that a random member scores above a random non-member; 0.5 is chance.
* **Advantage**: best TPR minus FPR over all thresholds (Yeom et al. 2018); 0 is chance.
* **TPR at low FPR** (0.1% and 1%): how many members an attacker identifies with confidence
  while almost never accusing a non-member (Carlini et al. 2022). An attack can have an AUC near
  0.5 yet expose a few examples reliably; this is the number that catches it.

Two attacks are implemented, both scored from the per-example probability of the true label:

* :func:`loss_scores`: the loss/confidence-threshold attack (Yeom et al.). Needs only the target.
* :func:`lira_scores`: the likelihood-ratio attack, LiRA (Carlini et al., "Membership Inference
  Attacks From First Principles", 2022), with shadow models trained by the same procedure on
  data that does (IN) or does not (OUT) contain each candidate. The online variant uses both;
  the offline variant uses only OUT shadows.

Training the target and shadow models needs TensorFlow; see :mod:`akili.privacy.mia`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from akili.tracks import TrackPolicy

FloatArray = npt.NDArray[np.float64]

LOW_FPRS: tuple[float, float] = (0.001, 0.01)
"""False-positive rates at which the audit reports the attack's true-positive rate."""

THRESHOLDS_ISSUE = "pesaclarity/pesaclarity-ml#3"
"""Where the provisional audit limits are being reviewed."""


class AuditResult(BaseModel):
    """One attack's result against one model."""

    model_config = ConfigDict(frozen=True)

    attack: str
    auc: float = Field(ge=0, le=1)
    advantage: float = Field(ge=0, le=1)
    n_members: int = Field(ge=0)
    n_non_members: int = Field(ge=0)
    tpr_at_0_1pct_fpr: float | None = Field(default=None, ge=0, le=1)
    """True-positive rate at a false-positive rate of at most 0.1%."""
    tpr_at_1pct_fpr: float | None = Field(default=None, ge=0, le=1)
    """True-positive rate at a false-positive rate of at most 1%."""
    auc_ci95: tuple[float, float] | None = None
    """Approximate 95% interval for the AUC (Hanley and McNeil 1982); for reading, not gating."""
    placeholder: bool = False
    """True when no real attack ran. The gate never passes a placeholder."""


class AuditLimits(BaseModel):
    """Pass/fail limits for an audit. A ``None`` limit is not checked."""

    model_config = ConfigDict(frozen=True)

    max_auc: float = Field(ge=0.5, le=1)
    max_advantage: float = Field(ge=0, le=1)
    max_tpr_at_1pct_fpr: float | None = Field(default=None, ge=0, le=1)
    max_tpr_at_0_1pct_fpr: float | None = Field(default=None, ge=0, le=1)
    min_examples: int = Field(default=1, ge=1)
    """Minimum members and non-members each; below it the low-FPR numbers mean little."""
    provisional: bool = False
    """True while the limits await review (:data:`THRESHOLDS_ISSUE`)."""

    @classmethod
    def from_policy(cls, policy: TrackPolicy) -> AuditLimits:
        return cls(
            max_auc=policy.max_attack_auc,
            max_advantage=policy.max_attack_advantage,
            max_tpr_at_1pct_fpr=policy.max_attack_tpr_at_1pct_fpr,
            max_tpr_at_0_1pct_fpr=policy.max_attack_tpr_at_0_1pct_fpr,
            min_examples=policy.min_audit_examples,
            provisional=policy.provisional,
        )

    def tightened(self, **limits: float | int | None) -> AuditLimits:
        """Apply stricter limits; a looser or ``None`` value leaves the current limit in place."""
        update: dict[str, float | int] = {}
        for key, value in limits.items():
            if value is None:
                continue
            current = getattr(self, key)
            if key == "min_examples":
                update[key] = max(int(current), int(value))
            else:
                update[key] = value if current is None else min(current, value)
        return self.model_copy(update=update)


class AuditReport(BaseModel):
    """Every attack's result for one model, with the limits it was checked against."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    dp: bool
    epsilon: float | None = None
    results: tuple[AuditResult, ...]
    limits: AuditLimits
    passed: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...] = ()


# --- Metrics -------------------------------------------------------------------------------


def _pair(member_scores: npt.ArrayLike, non_member_scores: npt.ArrayLike) -> tuple[FloatArray, ...]:
    m = np.asarray(member_scores, dtype=np.float64).ravel()
    n = np.asarray(non_member_scores, dtype=np.float64).ravel()
    if m.size == 0 or n.size == 0:
        raise ValueError("need at least one member and one non-member score")
    if not (np.isfinite(m).all() and np.isfinite(n).all()):
        raise ValueError("attack scores must be finite")
    return m, n


def roc_curve(
    member_scores: npt.ArrayLike, non_member_scores: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """(FPR, TPR) at every distinct threshold, "score >= t means member", from (0, 0) to (1, 1)."""
    m, n = _pair(member_scores, non_member_scores)
    scores = np.concatenate([m, n])
    is_member = np.concatenate([np.ones(m.size), np.zeros(n.size)])
    order = np.argsort(-scores, kind="stable")
    scores, is_member = scores[order], is_member[order]
    tp = np.cumsum(is_member)
    fp = np.cumsum(1 - is_member)
    # Keep only the last index of each run of equal scores: thresholds sit between values.
    last = np.r_[scores[1:] != scores[:-1], True]
    tpr = np.r_[0.0, tp[last] / m.size]
    fpr = np.r_[0.0, fp[last] / n.size]
    return fpr, tpr


def roc_auc(member_scores: npt.ArrayLike, non_member_scores: npt.ArrayLike) -> float:
    """AUC of "higher score means member", via ranks (Mann-Whitney U; ties count half)."""
    m, n = _pair(member_scores, non_member_scores)
    scores = np.concatenate([m, n])
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    # Average ranks over ties.
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    u = ranks[: m.size].sum() - m.size * (m.size + 1) / 2.0
    return float(u / (m.size * n.size))


def auc_interval(auc: float, n_members: int, n_non_members: int) -> tuple[float, float]:
    """Approximate 95% interval for an AUC (Hanley and McNeil 1982), clipped to [0, 1]."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_members - 1) * (q1 - auc * auc)
        + (n_non_members - 1) * (q2 - auc * auc)
    ) / (n_members * n_non_members)
    half = 1.96 * math.sqrt(max(var, 0.0))
    return max(0.0, auc - half), min(1.0, auc + half)


def membership_advantage(member_scores: npt.ArrayLike, non_member_scores: npt.ArrayLike) -> float:
    """Best |TPR - FPR| over all thresholds (Yeom et al.), in [0, 1]."""
    fpr, tpr = roc_curve(member_scores, non_member_scores)
    return float(np.abs(tpr - fpr).max())


def tpr_at_fpr(
    member_scores: npt.ArrayLike, non_member_scores: npt.ArrayLike, max_fpr: float
) -> float:
    """Highest TPR at any threshold whose FPR is at most ``max_fpr``.

    With ``n`` non-members the smallest non-zero FPR is ``1/n``; below that this is the TPR at
    zero false positives, which is still meaningful but noisy.
    """
    fpr, tpr = roc_curve(member_scores, non_member_scores)
    return float(tpr[fpr <= max_fpr + 1e-12].max())


def result_from_scores(
    attack: str, member_scores: npt.ArrayLike, non_member_scores: npt.ArrayLike
) -> AuditResult:
    m, n = _pair(member_scores, non_member_scores)
    auc = roc_auc(m, n)
    low, mid = LOW_FPRS
    return AuditResult(
        attack=attack,
        auc=auc,
        advantage=membership_advantage(m, n),
        n_members=int(m.size),
        n_non_members=int(n.size),
        tpr_at_0_1pct_fpr=tpr_at_fpr(m, n, low),
        tpr_at_1pct_fpr=tpr_at_fpr(m, n, mid),
        auc_ci95=auc_interval(auc, int(m.size), int(n.size)),
    )


# --- Attack scores -------------------------------------------------------------------------

_EPS = 1e-12


def true_label_probs(probs: npt.ArrayLike, labels: npt.ArrayLike) -> FloatArray:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return np.asarray(p[np.arange(len(y)), y], dtype=np.float64)


def loss_scores(probs: npt.ArrayLike, labels: npt.ArrayLike) -> FloatArray:
    """Loss/confidence-threshold attack: score = minus the cross-entropy loss = log p(y|x).

    Thresholding the loss and thresholding the true-label confidence give the same ranking.
    """
    return np.log(np.clip(true_label_probs(probs, labels), _EPS, 1.0))


def logit_confidence(probs: npt.ArrayLike, labels: npt.ArrayLike) -> FloatArray:
    """LiRA's statistic: log p(y|x) - log(1 - p(y|x)), close to Gaussian across models.

    Computed as log p_y - log(sum of the other probabilities) to stay finite near p_y = 1.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0)
    y = np.asarray(labels, dtype=np.int64)
    rows = np.arange(len(y))
    py = p[rows, y]
    rest = p.sum(axis=1) - py
    return np.asarray(np.log(py) - np.log(np.clip(rest, _EPS, None)), dtype=np.float64)


def _gaussian_logpdf(x: FloatArray, mean: FloatArray, var: FloatArray) -> FloatArray:
    return np.asarray(-0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var), dtype=np.float64)


def lira_scores(
    target: npt.ArrayLike,
    shadow: npt.ArrayLike,
    in_mask: npt.ArrayLike,
    *,
    online: bool = True,
    global_variance: bool | None = None,
    min_var: float = 1e-4,
) -> FloatArray:
    """LiRA membership scores.

    ``target`` is the target model's :func:`logit_confidence` for each of ``n`` candidates;
    ``shadow`` is ``[k, n]``, the same statistic from ``k`` shadow models; ``in_mask[k, i]`` is
    true when shadow ``k`` trained on candidate ``i``.

    Online LiRA scores each candidate by the log likelihood ratio of the target's statistic under
    the IN and OUT Gaussians fitted from the shadows. Offline LiRA uses only the OUT shadows and
    scores by how far above the OUT mean the target's statistic lies (a one-sided test).

    With few shadows per candidate the per-example variance is unreliable, so by default a single
    variance pooled over all candidates is used below 32 shadows, as Carlini et al. recommend.
    """
    t = np.asarray(target, dtype=np.float64)
    s = np.asarray(shadow, dtype=np.float64)
    mask = np.asarray(in_mask, dtype=bool)
    if s.ndim != 2 or s.shape != mask.shape or s.shape[1] != t.size:
        raise ValueError("shadow and in_mask must be [k, n] with n = len(target)")
    out = ~mask
    if not out.any(axis=0).all():
        raise ValueError("every candidate needs at least one OUT shadow model")
    if online and not mask.any(axis=0).all():
        raise ValueError("online LiRA needs at least one IN shadow model per candidate")
    if global_variance is None:
        global_variance = s.shape[0] < 32

    def stats(sel: npt.NDArray[np.bool_]) -> tuple[FloatArray, FloatArray]:
        count = sel.sum(axis=0)
        mean = np.where(sel, s, 0.0).sum(axis=0) / count
        dev = np.where(sel, (s - mean) ** 2, 0.0)
        if global_variance:
            var = np.full_like(mean, dev.sum() / max(sel.sum() - sel.shape[1], 1))
        else:
            var = dev.sum(axis=0) / np.maximum(count - 1, 1)
        return mean, np.maximum(var, min_var)

    mean_out, var_out = stats(out)
    if not online:
        return np.asarray((t - mean_out) / np.sqrt(var_out), dtype=np.float64)
    mean_in, var_in = stats(mask)
    return _gaussian_logpdf(t, mean_in, var_in) - _gaussian_logpdf(t, mean_out, var_out)


# --- Limits --------------------------------------------------------------------------------


def audit_failures(results: Iterable[AuditResult], limits: AuditLimits) -> list[str]:
    """Every way the results break the limits; empty when all attacks pass."""
    failures: list[str] = []
    found = False
    for r in results:
        found = True
        name = r.attack
        if r.placeholder:
            failures.append(f"membership-inference audit {name!r} is a placeholder")
            continue
        if min(r.n_members, r.n_non_members) < limits.min_examples:
            failures.append(
                f"{name}: audit used {r.n_members} members and {r.n_non_members} non-members, "
                f"below the minimum {limits.min_examples} each"
            )
        if r.auc > limits.max_auc:
            failures.append(f"{name}: attack AUC {r.auc:.3f} exceeds the limit {limits.max_auc}")
        if r.advantage > limits.max_advantage:
            failures.append(
                f"{name}: attack advantage {r.advantage:.3f} exceeds the limit "
                f"{limits.max_advantage}"
            )
        checks = (
            ("1%", r.tpr_at_1pct_fpr, limits.max_tpr_at_1pct_fpr),
            ("0.1%", r.tpr_at_0_1pct_fpr, limits.max_tpr_at_0_1pct_fpr),
        )
        for label, value, limit in checks:
            if limit is None:
                continue
            if value is None:
                failures.append(f"{name}: no TPR at {label} FPR reported")
            elif value > limit:
                failures.append(f"{name}: TPR at {label} FPR {value:.4f} exceeds the limit {limit}")
    if not found:
        failures.append("no membership-inference audit")
    return failures


def provisional_note(limits: AuditLimits) -> tuple[str, ...]:
    if not limits.provisional:
        return ()
    return (f"membership-inference limits are provisional pending review in {THRESHOLDS_ISSUE}",)


def build_report(
    model_name: str,
    results: Sequence[AuditResult],
    limits: AuditLimits,
    *,
    dp: bool,
    epsilon: float | None = None,
) -> AuditReport:
    failures = audit_failures(results, limits)
    return AuditReport(
        model_name=model_name,
        dp=dp,
        epsilon=epsilon,
        results=tuple(results),
        limits=limits,
        passed=not failures,
        failures=tuple(failures),
        notes=provisional_note(limits),
    )

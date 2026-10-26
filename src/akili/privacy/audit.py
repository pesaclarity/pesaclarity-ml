"""Membership-inference audit interface.

An attack scores how likely each example was in the training set; the audit reports how well
those scores separate real members from held-out non-members. The release gate then compares
the result with the track's thresholds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from akili.types import Transaction


class AuditResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    attack: str
    auc: float = Field(ge=0, le=1)
    advantage: float = Field(ge=0, le=1)
    n_members: int = Field(ge=0)
    n_non_members: int = Field(ge=0)
    placeholder: bool = False
    """True when no real attack ran; the gate never passes a placeholder."""


@runtime_checkable
class MembershipInferenceAttack(Protocol):
    name: str

    def run(
        self,
        model: object,
        members: Sequence[Transaction],
        non_members: Sequence[Transaction],
    ) -> AuditResult: ...


def roc_auc(member_scores: npt.ArrayLike, non_member_scores: npt.ArrayLike) -> float:
    """AUC of 'higher score means member', via the Mann-Whitney U statistic (ties count half)."""
    m = np.asarray(member_scores, dtype=np.float64)
    n = np.asarray(non_member_scores, dtype=np.float64)
    if m.size == 0 or n.size == 0:
        raise ValueError("need at least one member and one non-member score")
    greater = (m[:, None] > n[None, :]).sum()
    ties = (m[:, None] == n[None, :]).sum()
    return float((greater + 0.5 * ties) / (m.size * n.size))


def membership_advantage(member_scores: npt.ArrayLike, non_member_scores: npt.ArrayLike) -> float:
    """Best TPR minus FPR over all thresholds (Yeom et al.), in [0, 1]."""
    m = np.asarray(member_scores, dtype=np.float64)
    n = np.asarray(non_member_scores, dtype=np.float64)
    thresholds = np.unique(np.concatenate([m, n]))
    best = 0.0
    for t in thresholds:
        tpr = float((m >= t).mean())
        fpr = float((n >= t).mean())
        best = max(best, abs(tpr - fpr))
    return best


def result_from_scores(
    attack: str, member_scores: npt.ArrayLike, non_member_scores: npt.ArrayLike
) -> AuditResult:
    m = np.asarray(member_scores)
    n = np.asarray(non_member_scores)
    return AuditResult(
        attack=attack,
        auc=roc_auc(m, n),
        advantage=membership_advantage(m, n),
        n_members=int(m.size),
        n_non_members=int(n.size),
    )


class PlaceholderAttack:
    """Stand-in until a real attack lands; its result always fails the release gate.

    TODO(privacy): implement a loss-threshold attack and LiRA with shadow models against the
    exported TFLite model, using held-out synthetic and opted-in evaluation data on the server.
    """

    name = "placeholder"

    def run(
        self,
        model: object,
        members: Sequence[Transaction],
        non_members: Sequence[Transaction],
    ) -> AuditResult:
        return AuditResult(
            attack=self.name,
            auc=0.5,
            advantage=0.0,
            n_members=len(members),
            n_non_members=len(non_members),
            placeholder=True,
        )

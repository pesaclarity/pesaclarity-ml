"""Release gate: the spec's release conditions, checked against evidence for one model."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from akili.privacy.accounting import PrivacyReport, PrivacyUnit
from akili.privacy.audit import AuditLimits, AuditResult, audit_failures, provisional_note
from akili.privacy.ledger import BudgetTotal
from akili.tracks import Track, TrackPolicy, policy_for


class ReleaseEvidence(BaseModel):
    """Everything the gate needs to decide whether a model may leave the server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    track: Track
    privacy: PrivacyReport | None = None
    budget: BudgetTotal | None = None
    """The dataset's total across every recorded run (``privacy-budget.json``), which must
    include the release run itself."""
    audit: AuditResult | tuple[AuditResult, ...] | None = None
    """One attack's result, or every attack's (``results`` of ``audit-report.json``); the gate
    checks each against the track limits, so the strongest attack decides."""
    cohort_size: int = Field(default=0, ge=0)
    secure_aggregation: bool = False
    dpia_filed: bool = False
    legal_review: bool = False
    model_size_bytes: int | None = Field(default=None, ge=0)
    max_model_size_bytes: int | None = Field(default=None, ge=1)


class GateDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    track: Track
    passed: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...] = ()


def evaluate_release(evidence: ReleaseEvidence, policy: TrackPolicy | None = None) -> GateDecision:
    """Check evidence against the track policy; every unmet condition is listed."""
    p = policy or policy_for(evidence.track)
    failures: list[str] = []

    if evidence.privacy is None:
        failures.append("no DP-SGD privacy report: train with DP enabled")
    else:
        failures.extend(_privacy_failures(evidence.privacy, evidence.budget, p))

    if not evidence.secure_aggregation:
        failures.append("updates were not combined with secure aggregation")
    if evidence.cohort_size < p.min_cohort_size:
        failures.append(
            f"cohort of {evidence.cohort_size} is below the minimum {p.min_cohort_size}"
        )

    limits = AuditLimits.from_policy(p)
    audits: tuple[AuditResult, ...] = (
        (evidence.audit,) if isinstance(evidence.audit, AuditResult) else evidence.audit or ()
    )
    failures.extend(audit_failures(audits, limits))

    if not evidence.dpia_filed:
        failures.append("no data protection impact assessment filed with the ODPC")
    if p.requires_legal_review and not evidence.legal_review:
        failures.append("no legal review by a Kenyan data protection lawyer")

    if (
        evidence.max_model_size_bytes is not None
        and evidence.model_size_bytes is not None
        and evidence.model_size_bytes > evidence.max_model_size_bytes
    ):
        failures.append(
            f"model is {evidence.model_size_bytes} bytes, over the "
            f"{evidence.max_model_size_bytes}-byte budget"
        )

    return GateDecision(
        model_name=evidence.model_name,
        track=evidence.track,
        passed=not failures,
        failures=tuple(failures),
        notes=provisional_note(limits),
    )


def _budget_failures(epsilon: float, delta: float, population: int, p: TrackPolicy) -> list[str]:
    out: list[str] = []
    if epsilon > p.max_epsilon:
        out.append(f"epsilon {epsilon:.3g} exceeds the track limit {p.max_epsilon}")
    if delta > p.max_delta:
        out.append(f"delta {delta:.3g} exceeds the track limit {p.max_delta:.3g}")
    # A delta of 1/n or more permits mechanisms that publish an individual's record outright,
    # so the fixed track limit alone is not enough once the population is large.
    if delta * population >= 1:
        out.append(f"delta {delta:.3g} is not below 1/population (1/{population})")
    return out


def _privacy_failures(
    report: PrivacyReport, budget: BudgetTotal | None, p: TrackPolicy
) -> list[str]:
    out = _budget_failures(report.epsilon, report.delta, report.population, p)
    if report.unit is not PrivacyUnit.USER:
        out.append(
            f"the budget is per {report.unit.value}, not per person: someone with many "
            "transactions is far less protected; train with dp.unit user or DP-FedAvg"
        )
    if report.noise_seeded:
        out.append(
            "the DP noise came from a fixed seed, so anyone with the seed could remove it: "
            "retrain without dp.noise_seed"
        )
    if budget is None:
        out.append("no privacy budget total for the dataset: attach privacy-budget.json")
        return out
    if budget.unit is not report.unit:
        out.append(
            f"the budget total is per {budget.unit.value}, the model per {report.unit.value}"
        )
    if budget.epsilon < report.epsilon:
        out.append("the budget total is below the model's own epsilon, so it misses this run")
    out.extend(
        f"dataset total ({budget.runs} runs): {f}"
        for f in _budget_failures(budget.epsilon, budget.delta, budget.population, p)
    )
    return out


def load_evidence(path: str | Path) -> ReleaseEvidence:
    return ReleaseEvidence.model_validate_json(Path(path).read_text(encoding="utf-8"))

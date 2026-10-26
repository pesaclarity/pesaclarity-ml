"""Release gate: the spec's release conditions, checked against evidence for one model."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from akili.privacy.accounting import PrivacyReport
from akili.privacy.audit import AuditLimits, AuditResult, audit_failures, provisional_note
from akili.tracks import Track, TrackPolicy, policy_for


class ReleaseEvidence(BaseModel):
    """Everything the gate needs to decide whether a model may leave the server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    track: Track
    privacy: PrivacyReport | None = None
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
        if evidence.privacy.epsilon > p.max_epsilon:
            failures.append(
                f"epsilon {evidence.privacy.epsilon:.3g} exceeds the track limit {p.max_epsilon}"
            )
        if evidence.privacy.delta > p.max_delta:
            failures.append(
                f"delta {evidence.privacy.delta:.3g} exceeds the track limit {p.max_delta:.3g}"
            )

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


def load_evidence(path: str | Path) -> ReleaseEvidence:
    return ReleaseEvidence.model_validate_json(Path(path).read_text(encoding="utf-8"))

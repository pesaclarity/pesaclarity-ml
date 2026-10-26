import json
from pathlib import Path

import pytest

from akili.privacy.accounting import dp_fedavg_report
from akili.privacy.audit import (
    AuditResult,
    MembershipInferenceAttack,
    PlaceholderAttack,
    membership_advantage,
    result_from_scores,
    roc_auc,
)
from akili.privacy.gate import ReleaseEvidence, evaluate_release, load_evidence
from akili.tracks import Track

PRIVACY = dp_fedavg_report(
    n_clients=50_000,
    clients_per_round=1000,
    rounds=50,
    noise_multiplier=1.5,
    clip_norm=1.0,
    delta=1e-8,
)
GOOD_AUDIT = AuditResult(attack="loss", auc=0.51, advantage=0.02, n_members=500, n_non_members=500)


def evidence(**overrides: object) -> ReleaseEvidence:
    base: dict[str, object] = {
        "model_name": "categoriser",
        "track": Track.COMMERCIAL,
        "privacy": PRIVACY,
        "audit": GOOD_AUDIT,
        "cohort_size": 5000,
        "secure_aggregation": True,
        "dpia_filed": True,
        "legal_review": True,
        "model_size_bytes": 80_000,
        "max_model_size_bytes": 512_000,
    }
    return ReleaseEvidence.model_validate({**base, **overrides})


def test_auc_and_advantage() -> None:
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert roc_auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert membership_advantage([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert membership_advantage([0.5], [0.5]) == 0.0
    with pytest.raises(ValueError, match="at least one"):
        roc_auc([], [0.1])


def test_result_from_scores() -> None:
    result = result_from_scores("loss", [0.9, 0.4], [0.1, 0.5])
    assert result.auc == 0.75
    assert result.n_members == 2
    assert not result.placeholder


def test_placeholder_attack_satisfies_protocol() -> None:
    attack = PlaceholderAttack()
    assert isinstance(attack, MembershipInferenceAttack)
    assert attack.run(object(), [], []).placeholder


def test_complete_evidence_passes() -> None:
    decision = evaluate_release(evidence())
    assert decision.passed, decision.failures


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"privacy": None}, "no DP-SGD privacy report"),
        ({"privacy": PRIVACY.model_copy(update={"epsilon": 9.0})}, "epsilon 9"),
        ({"privacy": PRIVACY.model_copy(update={"delta": 1e-5})}, "delta"),
        ({"secure_aggregation": False}, "secure aggregation"),
        ({"cohort_size": 999}, "below the minimum 1000"),
        ({"audit": None}, "no membership-inference audit"),
        ({"audit": PlaceholderAttack().run(None, [], [])}, "placeholder"),
        ({"audit": GOOD_AUDIT.model_copy(update={"auc": 0.7})}, "attack AUC"),
        ({"audit": GOOD_AUDIT.model_copy(update={"advantage": 0.2})}, "attack advantage"),
        ({"dpia_filed": False}, "impact assessment"),
        ({"legal_review": False}, "legal review"),
        ({"model_size_bytes": 600_000}, "byte budget"),
    ],
)
def test_each_unmet_condition_fails(overrides: dict[str, object], reason: str) -> None:
    decision = evaluate_release(evidence(**overrides))
    assert not decision.passed
    assert any(reason in f for f in decision.failures), decision.failures


def test_app_track_does_not_need_legal_review() -> None:
    decision = evaluate_release(evidence(track=Track.APP, legal_review=False))
    assert decision.passed, decision.failures


def test_load_evidence(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence().model_dump(mode="json")), encoding="utf-8")
    assert load_evidence(path) == evidence()

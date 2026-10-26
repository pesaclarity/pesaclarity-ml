"""Membership-inference metrics, LiRA scoring and limits (core install, no TensorFlow)."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from akili.cli import format_audit, main
from akili.config import AuditConfig
from akili.privacy.audit import (
    AuditLimits,
    AuditResult,
    auc_interval,
    audit_failures,
    build_report,
    lira_scores,
    logit_confidence,
    loss_scores,
    membership_advantage,
    result_from_scores,
    roc_auc,
    roc_curve,
    tpr_at_fpr,
)
from akili.tracks import Track, policy_for

RNG = np.random.default_rng(0)


def brute_auc(m: np.ndarray, n: np.ndarray) -> float:
    return float(((m[:, None] > n[None, :]) + 0.5 * (m[:, None] == n[None, :])).mean())


def test_auc_matches_pairwise_definition_with_ties() -> None:
    m = RNG.integers(0, 5, 200).astype(float)
    n = RNG.integers(0, 4, 150).astype(float)
    assert roc_auc(m, n) == pytest.approx(brute_auc(m, n))


def test_roc_curve_and_tpr_at_fpr() -> None:
    fpr, tpr = roc_curve([3, 2, 1], [2.5, 0, -1, -2])
    assert fpr[0] == tpr[0] == 0
    assert fpr[-1] == tpr[-1] == 1
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)
    assert tpr_at_fpr([3, 2, 1], [2.5, 0, -1, -2], 0.0) == pytest.approx(1 / 3)
    assert tpr_at_fpr([3, 2, 1], [2.5, 0, -1, -2], 0.25) == pytest.approx(1.0)
    assert membership_advantage([3, 2, 1], [2.5, 0, -1, -2]) == pytest.approx(0.75)


def test_metrics_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        roc_auc([], [0.1])
    with pytest.raises(ValueError, match="finite"):
        roc_auc([np.nan], [0.1])


def test_chance_attack_reads_as_chance() -> None:
    m, n = RNG.normal(size=5000), RNG.normal(size=5000)
    r = result_from_scores("noise", m, n)
    assert r.auc == pytest.approx(0.5, abs=0.02)
    assert r.advantage < 0.05
    assert r.tpr_at_1pct_fpr is not None
    assert r.tpr_at_1pct_fpr < 0.03
    assert r.auc_ci95 is not None
    assert r.auc_ci95[0] < 0.5 < r.auc_ci95[1]


def test_auc_interval_narrows_with_more_examples() -> None:
    lo1, hi1 = auc_interval(0.6, 100, 100)
    lo2, hi2 = auc_interval(0.6, 2000, 2000)
    assert lo1 < lo2 < 0.6 < hi2 < hi1


def test_loss_and_logit_scores() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    labels = np.array([0, 0, 1])
    np.testing.assert_allclose(loss_scores(probs, labels), np.log([0.9, 0.2, 0.5]))
    np.testing.assert_allclose(logit_confidence(probs, labels), np.log([9.0, 0.25, 1.0]))
    assert np.isfinite(logit_confidence(np.array([[1.0, 0.0]]), np.array([0]))).all()


def _shadow_world(
    n: int, k: int, effect: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-example base confidence; training on an example adds ``effect`` to it."""
    base = rng.normal(0, 2, n)
    in_mask = np.argsort(rng.random((k, n)), axis=0) < k // 2
    shadow = base + effect * in_mask + rng.normal(0, 0.3, (k, n))
    is_member = np.arange(n) < n // 2
    target = base + effect * is_member + rng.normal(0, 0.3, n)
    return target, shadow, in_mask, is_member


@pytest.mark.parametrize("online", [True, False])
def test_lira_finds_memorisation_that_the_loss_attack_misses(online: bool) -> None:
    rng = np.random.default_rng(1)
    target, shadow, in_mask, is_member = _shadow_world(2000, 16, effect=1.0, rng=rng)
    lira = lira_scores(target, shadow, in_mask, online=online)
    r = result_from_scores("lira", lira[is_member], lira[~is_member])
    loss = result_from_scores("loss", target[is_member], target[~is_member])
    assert r.auc > 0.9
    assert r.auc > loss.auc + 0.2
    assert r.tpr_at_1pct_fpr is not None
    assert r.tpr_at_1pct_fpr > 0.3


def test_lira_is_chance_when_training_changes_nothing() -> None:
    rng = np.random.default_rng(2)
    target, shadow, in_mask, is_member = _shadow_world(2000, 8, effect=0.0, rng=rng)
    for online in (True, False):
        s = lira_scores(target, shadow, in_mask, online=online)
        assert roc_auc(s[is_member], s[~is_member]) == pytest.approx(0.5, abs=0.04)


def test_lira_validates_shapes_and_coverage() -> None:
    with pytest.raises(ValueError, match=r"\[k, n\]"):
        lira_scores(np.zeros(3), np.zeros((2, 4)), np.zeros((2, 4), bool))
    with pytest.raises(ValueError, match="OUT shadow"):
        lira_scores(np.zeros(2), np.zeros((2, 2)), np.ones((2, 2), bool))
    with pytest.raises(ValueError, match="IN shadow"):
        lira_scores(np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2), bool))
    offline = lira_scores(np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2), bool), online=False)
    assert offline.shape == (2,)


def test_limits_come_from_track_and_only_tighten() -> None:
    app = AuditLimits.from_policy(policy_for(Track.APP))
    commercial = AuditLimits.from_policy(policy_for(Track.COMMERCIAL))
    assert app.provisional
    assert commercial.max_auc <= app.max_auc
    assert commercial.min_examples >= app.min_examples
    tight = app.tightened(
        max_auc=0.55, max_advantage=0.5, min_examples=10, max_tpr_at_1pct_fpr=None
    )
    assert tight.max_auc == 0.55
    assert tight.max_advantage == app.max_advantage
    assert tight.min_examples == app.min_examples
    assert tight.max_tpr_at_1pct_fpr == app.max_tpr_at_1pct_fpr


def test_audit_failures_and_report() -> None:
    limits = AuditLimits(max_auc=0.6, max_advantage=0.1, max_tpr_at_1pct_fpr=0.05, min_examples=10)
    ok = AuditResult(
        attack="loss", auc=0.55, advantage=0.05, n_members=10, n_non_members=10, tpr_at_1pct_fpr=0
    )
    assert audit_failures([ok], limits) == []
    assert audit_failures([], limits) == ["no membership-inference audit"]
    bad = ok.model_copy(update={"attack": "lira", "auc": 0.7, "n_members": 5})
    failures = audit_failures([ok, bad], limits)
    assert len(failures) == 2
    assert all(f.startswith("lira:") for f in failures)
    report = build_report("m", [ok, bad], limits, dp=True, epsilon=4.0)
    assert not report.passed
    assert report.notes == ()
    text = format_audit(report)
    assert "epsilon 4" in text
    assert "FAIL" in text


def test_audit_config_needs_even_shadow_count() -> None:
    assert AuditConfig().shadow_models % 2 == 0
    with pytest.raises(ValidationError, match="even"):
        AuditConfig(shadow_models=3)


def test_cli_gate_reads_an_audit_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    limits = AuditLimits.from_policy(policy_for(Track.APP))
    leaky = AuditResult(
        attack="lira-online",
        auc=0.8,
        advantage=0.4,
        n_members=2000,
        n_non_members=2000,
        tpr_at_0_1pct_fpr=0.1,
        tpr_at_1pct_fpr=0.2,
    )
    report_path = tmp_path / "audit-report.json"
    report_path.write_text(build_report("m", [leaky], limits, dp=False).model_dump_json())
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"model_name": "m", "track": "app"}), encoding="utf-8")
    assert main(["gate", "--evidence", str(evidence), "--audit", str(report_path)]) == 1
    decision = json.loads(capsys.readouterr().out)
    assert any("lira-online: attack AUC 0.800" in f for f in decision["failures"])
    assert not any(f == "no membership-inference audit" for f in decision["failures"])

"""Membership-inference audit end to end. Skipped unless the [train] extra is installed."""

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")

from akili.cli import main
from akili.config import (
    AuditConfig,
    ExperimentConfig,
    ModelConfig,
    SyntheticDataConfig,
    TrackingConfig,
    TrainingConfig,
)
from akili.privacy.audit import AuditReport
from akili.privacy.mia import quick_config
from akili.train.pipeline import run_centralised

pytestmark = pytest.mark.tf


def tracking(tmp_path: Path) -> TrackingConfig:
    return TrackingConfig(
        uri=f"sqlite:///{tmp_path}/mlruns/mlflow.db",
        artifact_root=tmp_path / "mlruns" / "artifacts",
        experiment="akili-test",
    )


def test_pipeline_runs_audit_and_logs_it(tmp_path: Path) -> None:
    import mlflow

    cfg = ExperimentConfig(
        name="tiny-audit",
        output_dir=tmp_path / "artefacts",
        data=SyntheticDataConfig(n_users=20, transactions_per_user=20, payee_style="varied"),
        training=TrainingConfig(epochs=2, batch_size=32),
        audit=AuditConfig(enabled=True, shadow_models=4, max_candidates=50),
        tracking=tracking(tmp_path),
    )
    result = run_centralised(cfg)
    assert result.audit is not None
    report = AuditReport.model_validate_json(
        (result.run_dir / "audit-report.json").read_text(encoding="utf-8")
    )
    assert report == result.audit
    assert [r.attack for r in report.results] == ["loss", "lira-online", "lira-offline"]
    assert all(r.n_members == r.n_non_members == 50 for r in report.results)
    assert not report.passed  # 50 candidates is far below any track's minimum
    assert report.limits.provisional

    run = mlflow.get_run(result.run_id)
    assert "audit.lira-online.auc" in run.data.metrics
    assert "audit.loss.tpr_at_0_1pct_fpr" in run.data.metrics
    assert run.data.tags["audit.passed"] == "false"
    assert run.data.tags["audit.limits_provisional"] == "true"


def test_lira_detects_a_memorising_model(tmp_path: Path) -> None:
    """Sanity check of attack power: no DP, no dropout, many epochs on a small varied dataset."""
    cfg = ExperimentConfig(
        name="memorise",
        output_dir=tmp_path / "artefacts",
        data=SyntheticDataConfig(
            n_users=30, transactions_per_user=40, payee_style="varied", label_noise=0.1
        ),
        model=ModelConfig(dropout=0.0),
        training=TrainingConfig(epochs=30, batch_size=32, learning_rate=0.01),
        audit=AuditConfig(enabled=True, shadow_models=4, max_candidates=240, score_exported=False),
        tracking=TrackingConfig(enabled=False),
    )
    report = run_centralised(cfg).audit
    assert report is not None
    best = max(report.results, key=lambda r: r.auc)
    assert best.auc > 0.6, report.results
    assert any("exceeds the limit" in f for f in report.failures)


def test_cli_quick_audit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "tiny.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "tiny",
                "output_dir": str(tmp_path / "artefacts"),
                "training": {"epochs": 1},
                "tracking": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    assert quick_config(ExperimentConfig(name="x")).audit.shadow_models == 4
    args = ["audit", "--quick", "--shadow-models", "2", "--config", str(path)]
    assert main(args) == 1  # a quick audit is too small to pass
    out = capsys.readouterr().out
    assert "tiny-audit-quick (no DP)" in out
    assert "lira-online" in out
    report = json.loads(
        (tmp_path / "artefacts" / "tiny-audit-quick" / "audit-report.json").read_text()
    )
    assert report["passed"] is False
    assert main([*args, "--report-only"]) == 0

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from akili.cli import main
from akili.config import ExperimentConfig
from akili.export.size import ModelTooLargeError, check_size


def test_check_size(tmp_path: Path) -> None:
    path = tmp_path / "m.tflite"
    path.write_bytes(b"x" * 100)
    assert check_size(path, 100).size_bytes == 100
    with pytest.raises(ModelTooLargeError, match="budget is 99"):
        check_size(path, 99)


def test_cli_synth(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "tx.jsonl"
    assert main(["synth", "--out", str(out), "--users", "2", "--per-user", "5"]) == 0
    assert len(out.read_text().splitlines()) == 10
    assert "10 synthetic" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("config", "code"),
    [
        ("dp-sgd", 0),
        ("federated-app", 0),
        ("federated-commercial", 0),
        ("smoke", 1),
        ("federated-smoke", 1),
    ],
)
def test_cli_epsilon_checks_track_budget(config: str, code: int) -> None:
    root = Path(__file__).resolve().parents[1]
    assert main(["epsilon", "--config", str(root / "configs" / f"{config}.yaml")]) == code


def test_cli_gate_refuses_incomplete_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"model_name": "m", "track": "app"}), encoding="utf-8")
    assert main(["gate", "--evidence", str(path)]) == 1
    decision = json.loads(capsys.readouterr().out)
    assert decision["passed"] is False
    assert len(decision["failures"]) >= 4


def test_config_fixture(load_repo_config: Callable[[str], ExperimentConfig]) -> None:
    assert load_repo_config("smoke").dp.enabled is False

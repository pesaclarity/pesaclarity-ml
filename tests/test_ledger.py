from pathlib import Path

import pytest

from akili.config import ExperimentConfig, SyntheticDataConfig
from akili.privacy.accounting import PrivacyReport, PrivacyUnit, dp_sgd_report
from akili.privacy.ledger import Ledger, Purpose, synthetic_dataset_id


def report(
    epochs: int = 8, unit: PrivacyUnit = PrivacyUnit.USER, delta: float = 1e-6
) -> PrivacyReport:
    return dp_sgd_report(
        population=2000,
        per_step=100,
        epochs=epochs,
        noise_multiplier=1.5,
        l2_norm_clip=1.0,
        noise_seeded=False,
        unit=unit,
        delta=delta,
    )


def test_empty_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.entries() == []
    assert ledger.total("data", PrivacyUnit.USER) is None


def test_one_run_totals_its_own_epsilon(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    r = report()
    ledger.record("data", r, run="dp-sgd", run_id="abc")
    total = ledger.total("data", PrivacyUnit.USER)
    assert total is not None
    assert total.runs == 1
    assert total.epsilon == pytest.approx(r.epsilon)
    assert ledger.entries()[0].run_id == "abc"


def test_every_run_adds_up(tmp_path: Path) -> None:
    """Two runs spend more than one, less than twice (RDP composition), and the same as one run
    twice as long."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record("data", report(), run="first")
    ledger.record("data", report(), run="again")
    total = ledger.total("data", PrivacyUnit.USER)
    assert total is not None
    assert report().epsilon < total.epsilon < 2 * report().epsilon
    assert total.epsilon == pytest.approx(report(epochs=16).epsilon)


def test_audits_count_and_datasets_and_units_are_kept_apart(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record("data", report(), run="dp-sgd")
    ledger.record("data", report(), run="dp-sgd/shadow-1", purpose=Purpose.AUDIT)
    ledger.record("other", report(), run="dp-sgd")
    ledger.record("data", report(unit=PrivacyUnit.EXAMPLE), run="per-example")
    total = ledger.total("data", PrivacyUnit.USER)
    assert total is not None
    assert total.runs == 2
    assert [e.purpose for e in ledger.entries("data")][:2] == [Purpose.TRAINING, Purpose.AUDIT]
    other = ledger.total("other", PrivacyUnit.USER)
    assert other is not None
    assert other.runs == 1
    examples = ledger.total("data", PrivacyUnit.EXAMPLE)
    assert examples is not None
    assert examples.runs == 1


def test_total_uses_the_smallest_delta(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record("data", report(delta=1e-5), run="a")
    ledger.record("data", report(delta=1e-7), run="b")
    total = ledger.total("data", PrivacyUnit.USER)
    assert total is not None
    assert total.delta == 1e-7


def test_dataset_id_follows_the_data_settings() -> None:
    a = ExperimentConfig(name="a", data=SyntheticDataConfig(n_users=10))
    b = ExperimentConfig(name="b", data=SyntheticDataConfig(n_users=10))
    c = ExperimentConfig(name="c", data=SyntheticDataConfig(n_users=11))
    assert a.dataset_id == b.dataset_id != c.dataset_id
    assert a.dataset_id == synthetic_dataset_id(a.data.model_dump_json())
    assert a.ledger_path == a.output_dir / "privacy-ledger.jsonl"

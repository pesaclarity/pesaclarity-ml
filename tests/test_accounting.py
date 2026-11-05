import math

import pytest

from akili.privacy.accounting import (
    Mechanism,
    PrivacyReport,
    PrivacyUnit,
    default_delta,
    dp_fedavg_report,
    dp_sgd_report,
    epsilon_for,
)


def test_known_value() -> None:
    # Abadi et al. style setting: q=0.01, sigma=1.1, 1000 steps, delta=1e-5.
    assert epsilon_for(0.01, 1.1, 1000, 1e-5) == pytest.approx(1.71, abs=0.05)


def test_more_noise_spends_less() -> None:
    assert epsilon_for(0.01, 2.0, 1000, 1e-5) < epsilon_for(0.01, 1.0, 1000, 1e-5)


def test_more_steps_spend_more() -> None:
    assert epsilon_for(0.01, 1.1, 2000, 1e-5) > epsilon_for(0.01, 1.1, 1000, 1e-5)


def test_edge_cases() -> None:
    assert epsilon_for(0.1, 1.0, 0, 1e-5) == 0.0
    assert math.isinf(epsilon_for(0.1, 0.0, 10, 1e-5))
    with pytest.raises(ValueError, match="sampling_rate"):
        epsilon_for(0.0, 1.0, 10, 1e-5)


def test_default_delta_is_below_one_over_n() -> None:
    assert default_delta(10_000) < 1 / 10_000
    assert default_delta(10) == 1e-5


def test_dp_sgd_report_is_user_level_by_default() -> None:
    report = dp_sgd_report(
        population=2000,
        per_step=100,
        epochs=8,
        noise_multiplier=1.5,
        l2_norm_clip=1.0,
        noise_seeded=False,
        delta=1e-6,
    )
    assert report.mechanism is Mechanism.DP_SGD
    assert report.unit is PrivacyUnit.USER
    assert report.population == 2000
    assert report.steps == 160
    assert report.sampling_rate == pytest.approx(0.05)
    assert not report.noise_seeded
    assert 0 < report.epsilon < 3


def test_dp_sgd_report_per_example() -> None:
    report = dp_sgd_report(
        population=10_000,
        per_step=100,
        epochs=2,
        noise_multiplier=1.1,
        l2_norm_clip=1.0,
        noise_seeded=True,
        unit=PrivacyUnit.EXAMPLE,
    )
    assert report.unit is PrivacyUnit.EXAMPLE
    assert report.steps == 200
    assert report.sampling_rate == pytest.approx(0.01)
    assert 0 < report.epsilon < 2


def test_a_report_without_the_seed_flag_counts_as_seeded() -> None:
    """Reports written before the flag existed all used a fixed seed."""
    report = dp_fedavg_report(
        n_clients=2000,
        clients_per_round=200,
        rounds=30,
        noise_multiplier=1.0,
        clip_norm=3.0,
        noise_seeded=False,
    )
    old = report.model_dump(mode="json")
    del old["noise_seeded"]
    assert PrivacyReport.model_validate(old).noise_seeded


def test_infinite_epsilon_survives_json() -> None:
    report = dp_fedavg_report(
        n_clients=200,
        clients_per_round=20,
        rounds=3,
        noise_multiplier=0.0,
        clip_norm=1.0,
        noise_seeded=True,
    )
    assert math.isinf(PrivacyReport.model_validate_json(report.model_dump_json()).epsilon)


def test_dp_fedavg_report_is_user_level() -> None:
    report = dp_fedavg_report(
        n_clients=2000,
        clients_per_round=200,
        rounds=30,
        noise_multiplier=1.0,
        clip_norm=3.0,
        noise_seeded=False,
        delta=1e-6,
    )
    assert report.unit is PrivacyUnit.USER
    assert report.steps == 30
    assert report.delta == 1e-6
    assert report.epsilon > 0

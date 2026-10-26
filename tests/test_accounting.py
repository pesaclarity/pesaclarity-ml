import math

import pytest

from akili.privacy.accounting import (
    Mechanism,
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


def test_dp_sgd_report() -> None:
    report = dp_sgd_report(
        n_examples=10_000, batch_size=100, epochs=2, noise_multiplier=1.1, l2_norm_clip=1.0
    )
    assert report.mechanism is Mechanism.DP_SGD
    assert report.unit is PrivacyUnit.EXAMPLE
    assert report.steps == 200
    assert report.sampling_rate == pytest.approx(0.01)
    assert 0 < report.epsilon < 2


def test_dp_fedavg_report_is_user_level() -> None:
    report = dp_fedavg_report(
        n_clients=2000,
        clients_per_round=200,
        rounds=30,
        noise_multiplier=1.0,
        clip_norm=3.0,
        delta=1e-6,
    )
    assert report.unit is PrivacyUnit.USER
    assert report.steps == 30
    assert report.delta == 1e-6
    assert report.epsilon > 0

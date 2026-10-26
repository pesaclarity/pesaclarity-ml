from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from akili.config import ExperimentConfig, FederatedConfig
from akili.tracks import Track, policy_for

CONFIGS = sorted(p.stem for p in (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


def test_tracks_match_the_spec_defaults() -> None:
    app, commercial = policy_for(Track.APP), policy_for(Track.COMMERCIAL)
    assert app.default_opt_in is True
    assert commercial.default_opt_in is False
    assert commercial.licensable
    assert not app.licensable
    assert commercial.max_epsilon < app.max_epsilon
    assert commercial.min_cohort_size >= 1000
    assert commercial.requires_legal_review


@pytest.mark.parametrize("name", CONFIGS)
def test_repo_configs_load(name: str, load_repo_config: Callable[[str], ExperimentConfig]) -> None:
    cfg = load_repo_config(name)
    assert cfg.name == name


def test_commercial_track_requires_dp() -> None:
    with pytest.raises(ValidationError, match="DP-SGD"):
        ExperimentConfig(name="x", track=Track.COMMERCIAL)


def test_commercial_track_requires_secure_aggregation() -> None:
    with pytest.raises(ValidationError, match="secure aggregation"):
        ExperimentConfig(
            name="x", track=Track.COMMERCIAL, federated=FederatedConfig(secure_aggregation=False)
        )


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "x", "trainning": {}})


def test_cohort_minimum_never_drops_below_track_floor() -> None:
    cfg = ExperimentConfig(name="x", federated=FederatedConfig(min_cohort_size=5))
    assert cfg.min_cohort_size == policy_for(Track.APP).min_cohort_size


def test_flat_params_are_dotted() -> None:
    params = ExperimentConfig(name="x").flat_params()
    assert params["dp.enabled"] is False
    assert params["track"] == "app"

"""Experiment configuration, loaded from YAML files in ``configs/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from akili.tracks import Track, TrackPolicy, policy_for


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticDataConfig(_Strict):
    n_users: int = Field(default=200, ge=1)
    transactions_per_user: int = Field(default=60, ge=1)
    label_noise: float = Field(default=0.02, ge=0, lt=1)
    commercial_opt_in_rate: float = Field(default=0.2, ge=0, le=1)
    seed: int = 7


class FeatureConfig(_Strict):
    max_tokens: int = Field(default=8, ge=1, le=32)
    hash_buckets: int = Field(default=4096, ge=16, le=65536)


class ModelConfig(_Strict):
    token_embedding_dim: int = Field(default=16, ge=2)
    hidden_units: int = Field(default=32, ge=4)
    dropout: float = Field(default=0.1, ge=0, lt=1)


class TrainingConfig(_Strict):
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)


class DPConfig(_Strict):
    """DP-SGD settings. ``target_delta`` defaults to below 1/n at accounting time."""

    enabled: bool = False
    l2_norm_clip: float = Field(default=1.0, gt=0)
    noise_multiplier: float = Field(default=1.1, ge=0)
    target_delta: float | None = Field(default=None, gt=0, lt=1)


class FederatedConfig(_Strict):
    """Client-side federated settings; the aggregator in pesaclarity-backend mirrors these."""

    rounds: int = Field(default=10, ge=1)
    clients_per_round: int = Field(default=50, ge=1)
    local_epochs: int = Field(default=1, ge=1)
    local_batch_size: int = Field(default=16, ge=1)
    local_learning_rate: float = Field(default=0.05, gt=0)
    update_clip_norm: float = Field(default=1.0, gt=0)
    update_noise_multiplier: float = Field(default=1.0, ge=0)
    secure_aggregation: bool = True
    min_cohort_size: int | None = Field(default=None, ge=1)


class ExportConfig(_Strict):
    max_size_bytes: int = Field(default=1_000_000, ge=1)
    quantize: bool = True


class TrackingConfig(_Strict):
    """Local MLflow by default: a SQLite file and an artefact folder under ``./mlruns``.

    MLflow 3 no longer accepts the old ``./mlruns`` file store without an opt-out, so the
    default is a single local SQLite file instead; no server is needed either way.
    """

    uri: str = "sqlite:///mlruns/mlflow.db"
    artifact_root: Path = Path("mlruns/artifacts")
    experiment: str = "akili"
    enabled: bool = True


class ExperimentConfig(_Strict):
    name: str
    track: Track = Track.APP
    seed: int = 42
    output_dir: Path = Path("artefacts")
    data: SyntheticDataConfig = SyntheticDataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    dp: DPConfig = DPConfig()
    federated: FederatedConfig | None = None
    export: ExportConfig = ExportConfig()
    tracking: TrackingConfig = TrackingConfig()

    @model_validator(mode="after")
    def _commercial_needs_privacy(self) -> Self:
        if self.track is Track.COMMERCIAL:
            if self.federated is None and not self.dp.enabled:
                raise ValueError("commercial track needs DP-SGD enabled")
            if self.federated is not None and not self.federated.secure_aggregation:
                raise ValueError("commercial track needs secure aggregation")
        return self

    @property
    def policy(self) -> TrackPolicy:
        return policy_for(self.track)

    @property
    def min_cohort_size(self) -> int:
        """Configured cohort minimum, never below the track's floor."""
        floor = self.policy.min_cohort_size
        if self.federated is None or self.federated.min_cohort_size is None:
            return floor
        return max(floor, self.federated.min_cohort_size)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.name

    def flat_params(self) -> dict[str, Any]:
        """Flattened ``section.key`` parameters for experiment tracking."""
        out: dict[str, Any] = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(f"{prefix}.{k}" if prefix else k, v)
            else:
                out[prefix] = value

        walk("", self.model_dump(mode="json"))
        return out


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from YAML."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(raw)

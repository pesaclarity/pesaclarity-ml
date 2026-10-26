"""App and commercial training tracks, loaded from ``resources/tracks.yaml``."""

from __future__ import annotations

from enum import StrEnum
from functools import cache
from importlib import resources

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Track(StrEnum):
    APP = "app"
    COMMERCIAL = "commercial"


class TrackPolicy(BaseModel):
    """Consent and release thresholds for one training track."""

    model_config = ConfigDict(frozen=True)

    description: str
    consent_key: str
    default_opt_in: bool
    licensable: bool
    max_epsilon: float = Field(gt=0)
    max_delta: float = Field(gt=0, lt=1)
    min_cohort_size: int = Field(ge=1)
    max_attack_auc: float = Field(ge=0.5, le=1)
    max_attack_advantage: float = Field(ge=0, le=1)
    max_attack_tpr_at_1pct_fpr: float | None = Field(default=None, ge=0, le=1)
    max_attack_tpr_at_0_1pct_fpr: float | None = Field(default=None, ge=0, le=1)
    min_audit_examples: int = Field(default=1000, ge=1)
    provisional: bool = True
    """Membership-inference limits await review (pesaclarity/pesaclarity-ml#3)."""
    requires_legal_review: bool


@cache
def track_policies() -> dict[Track, TrackPolicy]:
    """Load the packaged track policies."""
    text = resources.files("akili.resources").joinpath("tracks.yaml").read_text("utf-8")
    raw = yaml.safe_load(text)
    return {Track(k): TrackPolicy.model_validate(v) for k, v in raw.items()}


def policy_for(track: Track) -> TrackPolicy:
    return track_policies()[track]

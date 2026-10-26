from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from akili.config import ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


@pytest.fixture
def load_repo_config() -> Callable[[str], ExperimentConfig]:
    def _load(name: str) -> ExperimentConfig:
        return load_config(CONFIG_DIR / f"{name}.yaml")

    return _load

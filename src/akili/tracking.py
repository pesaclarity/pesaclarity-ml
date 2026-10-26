"""MLflow tracking (needs the [train] extra). Defaults to local files under ./mlruns."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from akili._optional import require
from akili.config import TrackingConfig
from akili.privacy.accounting import PrivacyReport


class Tracker:
    """Thin wrapper over one MLflow run; every method is a no-op when tracking is disabled."""

    def __init__(self, cfg: TrackingConfig, run_name: str) -> None:
        self._cfg = cfg
        self._run_name = run_name
        self._mlflow: Any = None
        self.run_id: str | None = None

    def __enter__(self) -> Self:
        if not self._cfg.enabled:
            return self
        mlflow = require("mlflow", "train")
        if self._cfg.uri.startswith("sqlite:///"):
            Path(self._cfg.uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self._cfg.uri)
        if mlflow.get_experiment_by_name(self._cfg.experiment) is None:
            mlflow.create_experiment(
                self._cfg.experiment,
                artifact_location=self._cfg.artifact_root.resolve().as_uri(),
            )
        mlflow.set_experiment(self._cfg.experiment)
        run = mlflow.start_run(run_name=self._run_name)
        self._mlflow = mlflow
        self.run_id = str(run.info.run_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
            self._mlflow = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(dict(metrics), step=step)

    def log_privacy(self, report: PrivacyReport) -> None:
        """Log the budget as metrics and the full report as an artefact."""
        if self._mlflow is not None:
            self._mlflow.log_metrics(
                {"privacy.epsilon": report.epsilon, "privacy.delta": report.delta}
            )
            self._mlflow.log_dict(report.model_dump(mode="json"), "privacy-report.json")

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        if self._mlflow is not None:
            self._mlflow.set_tags(dict(tags))

"""Privacy budget ledger: every DP run on a dataset, and what they spend together.

Each run on the same data spends budget again, so a model's own epsilon understates what the
data has given up. The ledger keeps one JSON line per run (training, audit shadow models,
tuning) and composes them all with the RDP accountant. That is tighter than adding the epsilons
and still a valid bound. Runs are only composed with runs of the same privacy unit.

The gate checks the dataset's total, not just the release model's epsilon.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import dp_accounting
from pydantic import BaseModel, ConfigDict, Field

from akili.privacy.accounting import RDP_ORDERS, PrivacyReport, PrivacyUnit


class Purpose(StrEnum):
    TRAINING = "training"
    AUDIT = "audit"
    TUNING = "tuning"


class LedgerEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str
    run: str
    purpose: Purpose
    recorded_at: datetime
    report: PrivacyReport
    run_id: str | None = None


class BudgetTotal(BaseModel):
    """What every recorded run on one dataset spends together, for one privacy unit."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="constants")

    dataset: str
    unit: PrivacyUnit
    runs: int = Field(ge=0)
    epsilon: float = Field(ge=0)
    delta: float = Field(gt=0, lt=1)
    population: int = Field(ge=1)
    """The smallest population among the runs, so 1/population is the strictest."""
    accountant: str = "rdp"


def synthetic_dataset_id(settings_json: str) -> str:
    """A stable name for a synthetic dataset, from its generator settings."""
    return "synthetic-" + hashlib.sha256(settings_json.encode("utf-8")).hexdigest()[:12]


def compose(entries: list[LedgerEntry], delta: float) -> float:
    """Epsilon at ``delta`` of every entry's mechanism composed together."""
    accountant = dp_accounting.rdp.RdpAccountant(orders=list(RDP_ORDERS))
    for e in entries:
        r = e.report
        if r.steps == 0:
            continue
        if r.noise_multiplier <= 0:
            return float("inf")
        event = dp_accounting.PoissonSampledDpEvent(
            r.sampling_rate, dp_accounting.GaussianDpEvent(r.noise_multiplier)
        )
        accountant.compose(event, r.steps)
    return float(accountant.get_epsilon(delta))


class Ledger:
    """An append-only JSON-lines file of :class:`LedgerEntry`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(
        self,
        dataset: str,
        report: PrivacyReport,
        *,
        run: str,
        purpose: Purpose = Purpose.TRAINING,
        run_id: str | None = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            dataset=dataset,
            run=run,
            purpose=purpose,
            recorded_at=datetime.now(UTC),
            report=report,
            run_id=run_id,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        return entry

    def entries(self, dataset: str | None = None) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        out = [
            LedgerEntry.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [e for e in out if dataset is None or e.dataset == dataset]

    def total(
        self, dataset: str, unit: PrivacyUnit, delta: float | None = None
    ) -> BudgetTotal | None:
        """The dataset's total for ``unit``, at ``delta`` (default: the smallest delta any run
        used, which gives the largest, so safest, epsilon). None if nothing is recorded."""
        runs = [e for e in self.entries(dataset) if e.report.unit is unit]
        if not runs:
            return None
        d = delta if delta is not None else min(e.report.delta for e in runs)
        return BudgetTotal(
            dataset=dataset,
            unit=unit,
            runs=len(runs),
            epsilon=compose(runs, d),
            delta=d,
            population=min(e.report.population for e in runs),
        )

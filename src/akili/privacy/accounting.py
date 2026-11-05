"""Privacy budget accounting with Google's ``dp-accounting`` RDP accountant.

Both mechanisms are a Poisson-subsampled Gaussian composed over many steps: DP-SGD per person
(q = people per step / people) or per example (q = batch / examples), and DP-FedAvg per person
(q = clients per round / eligible clients).
"""

from __future__ import annotations

from enum import StrEnum

import dp_accounting
from pydantic import BaseModel, ConfigDict, Field

# Integer and a few fractional orders; the very small fractional orders in the library default
# often fail to converge for large sampling rates and only add log noise.
RDP_ORDERS: tuple[float, ...] = (
    1.75,
    2.0,
    2.25,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    *map(float, range(5, 64)),
    128.0,
    256.0,
    512.0,
)


class Mechanism(StrEnum):
    DP_SGD = "dp-sgd"
    DP_FEDAVG = "dp-fedavg"


class PrivacyUnit(StrEnum):
    EXAMPLE = "example"
    USER = "user"


class PrivacyReport(BaseModel):
    """The privacy budget published with a model release."""

    # Without DP noise epsilon is infinite; keep it infinite in JSON rather than null.
    model_config = ConfigDict(frozen=True, ser_json_inf_nan="constants")

    mechanism: Mechanism
    unit: PrivacyUnit
    epsilon: float = Field(ge=0)
    delta: float = Field(gt=0, lt=1)
    noise_multiplier: float = Field(ge=0)
    l2_norm_clip: float = Field(gt=0)
    sampling_rate: float = Field(gt=0, le=1)
    steps: int = Field(ge=0)
    population: int = Field(ge=1)
    accountant: str = "rdp"
    noise_seeded: bool = True
    """True when the noise came from a fixed seed, which anyone holding it could use to remove
    the noise. Missing means seeded, so older reports are not mistaken for safe ones."""


def default_delta(population: int) -> float:
    """A delta comfortably below 1/population, the usual rule of thumb."""
    return min(1e-5, 1.0 / (10 * population))


def epsilon_for(sampling_rate: float, noise_multiplier: float, steps: int, delta: float) -> float:
    """Epsilon of a Poisson-subsampled Gaussian mechanism composed ``steps`` times."""
    if not 0 < sampling_rate <= 1:
        raise ValueError("sampling_rate must be in (0, 1]")
    if steps == 0:
        return 0.0
    if noise_multiplier <= 0:
        return float("inf")
    accountant = dp_accounting.rdp.RdpAccountant(orders=list(RDP_ORDERS))
    event = dp_accounting.PoissonSampledDpEvent(
        sampling_rate, dp_accounting.GaussianDpEvent(noise_multiplier)
    )
    accountant.compose(event, steps)
    return float(accountant.get_epsilon(delta))


def dp_sgd_report(
    *,
    population: int,
    per_step: int,
    epochs: int,
    noise_multiplier: float,
    l2_norm_clip: float,
    noise_seeded: bool,
    unit: PrivacyUnit = PrivacyUnit.USER,
    delta: float | None = None,
) -> PrivacyReport:
    """Budget for centralised DP-SGD with Poisson sampling.

    For ``unit: user``, ``population`` is the number of people and ``per_step`` the expected
    people per step; for ``unit: example``, examples and the batch size.
    """
    q = min(1.0, per_step / population)
    steps = epochs * max(1, round(population / per_step))
    d = delta if delta is not None else default_delta(population)
    return PrivacyReport(
        mechanism=Mechanism.DP_SGD,
        unit=unit,
        epsilon=epsilon_for(q, noise_multiplier, steps, d),
        delta=d,
        noise_multiplier=noise_multiplier,
        l2_norm_clip=l2_norm_clip,
        sampling_rate=q,
        steps=steps,
        population=population,
        noise_seeded=noise_seeded,
    )


def dp_fedavg_report(
    *,
    n_clients: int,
    clients_per_round: int,
    rounds: int,
    noise_multiplier: float,
    clip_norm: float,
    noise_seeded: bool,
    delta: float | None = None,
) -> PrivacyReport:
    """User-level budget for DP-FedAvg (clipped updates, Gaussian noise on the aggregate).

    ``noise_seeded`` is False only when every client drew its noise from OS entropy."""
    q = min(1.0, clients_per_round / n_clients)
    d = delta if delta is not None else default_delta(n_clients)
    return PrivacyReport(
        mechanism=Mechanism.DP_FEDAVG,
        unit=PrivacyUnit.USER,
        epsilon=epsilon_for(q, noise_multiplier, rounds, d),
        delta=d,
        noise_multiplier=noise_multiplier,
        l2_norm_clip=clip_norm,
        sampling_rate=q,
        steps=rounds,
        population=n_clients,
        noise_seeded=noise_seeded,
    )

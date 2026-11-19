"""Privacy budget accounting with Opacus's accountants (Meta, Apache-2.0; decision 0023).

Both mechanisms are a Poisson-subsampled Gaussian composed over many steps: DP-SGD per person
(q = people per step / people) or per example (q = batch / examples), and DP-FedAvg per person
(q = clients per round / eligible clients).

The gate checks the RDP epsilon (:func:`epsilon_for`): Opacus's ``RDPAccountant`` with this
repository's integer orders (:data:`RDP_ORDERS`) and the RDP to (epsilon, delta) conversion of
Balle et al. 2020. Every report also gives the tighter figure from Opacus's ``PRVAccountant``
(:func:`pld_epsilon_for`, Gopi et al. 2021), as ``epsilon_pld``: the field keeps the name it
had when Google's ``dp-accounting`` PLD accountant computed it, and holds PRV's upper bound.
How the two libraries' figures compare is in :mod:`akili.privacy.compare` and
docs/DP-SGD-REVIEW.md.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

SIGNED_OFF_RDP_ORDERS: tuple[float, ...] = (
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
"""The orders dp-accounting was given for the figures signed off in pesaclarity-ml#3: integers
and a few fractions (the very small fractional orders in its default often fail to converge for
large sampling rates and only add log noise)."""

RDP_ORDERS: tuple[float, ...] = tuple(o for o in SIGNED_OFF_RDP_ORDERS if o.is_integer())
"""The orders the gate's figure uses: the integers among :data:`SIGNED_OFF_RDP_ORDERS`.

At an integer order both libraries sum the same positive terms and agree to rounding. At a
fractional one Opacus sums Mironov et al.'s alternating series exactly, where dp-accounting
bounds it by the sum of the terms' magnitudes, so Opacus's figure can come out below the one
signed off (by up to 0.05% under the app track's limits, 5% far above them; docs/DP-SGD-REVIEW.md).
Leaving the fractional orders out makes the figure never lower than dp-accounting's, and the
same at every config in ``configs/``, where the best order is an integer."""

RDP_ACCOUNTANT = "opacus-rdp"
PRV_ACCOUNTANT = "opacus-prv"
PRV_EPS_ERROR = 1e-3
"""Opacus PRV's allowed error in epsilon. Its default is 1e-2; PRV reports the upper end of
its interval, so a smaller error gives a tighter figure that is still a bound."""

Mechanisms = Iterable[tuple[float, float, int]]
"""(sampling rate, noise multiplier, steps) of each Poisson-subsampled Gaussian composed."""


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
    accountant: str = RDP_ACCOUNTANT
    """Which accountant gave ``epsilon``: ``opacus-rdp``, or ``rdp`` (dp-accounting's) in
    reports written before decision 0023's stage 4."""
    epsilon_pld: float | None = Field(default=None, ge=0)
    """The same budget from a tighter accountant: Opacus's PRV accountant (``tight_accountant``),
    dp-accounting's PLD in older reports. Published alongside; the gate uses the RDP
    ``epsilon``, which is the safer upper bound."""
    tight_accountant: str | None = None
    """Which accountant gave ``epsilon_pld`` (``opacus-prv``); missing in older reports, whose
    figure is dp-accounting's PLD."""
    noise_seeded: bool = True
    """True when the noise came from a fixed seed, which anyone holding it could use to remove
    the noise. Missing means seeded, so older reports are not mistaken for safe ones."""


def default_delta(population: int) -> float:
    """A delta comfortably below 1/population, the usual rule of thumb."""
    return min(1e-5, 1.0 / (10 * population))


def _check(sampling_rate: float) -> None:
    if not 0 < sampling_rate <= 1:
        raise ValueError("sampling_rate must be in (0, 1]")


def _history(mechanisms: Mechanisms) -> list[tuple[float, float, int]] | None:
    """Opacus's history, ``(noise multiplier, sampling rate, steps)`` per mechanism, without
    the ones that took no step; None when one without noise did (epsilon is infinite)."""
    out = []
    for q, sigma, steps in mechanisms:
        _check(q)
        if steps == 0:
            continue
        if sigma <= 0:
            return None
        out.append((float(sigma), float(q), int(steps)))
    return out


def composed_epsilon(
    mechanisms: Mechanisms, delta: float, orders: tuple[float, ...] = RDP_ORDERS
) -> float:
    """Epsilon at ``delta`` of Poisson-subsampled Gaussians composed, from Opacus's
    ``RDPAccountant`` with ``orders`` (by default :data:`RDP_ORDERS`)."""
    from opacus.accountants import RDPAccountant

    history = _history(mechanisms)
    if history is None:
        return float("inf")
    if not history:
        return 0.0
    accountant = RDPAccountant()
    accountant.history = history
    with warnings.catch_warnings():
        # Opacus warns when the best order is the first or last; the bound holds either way
        # (dp-accounting returned the same figure silently).
        warnings.simplefilter("ignore", UserWarning)
        return float(accountant.get_epsilon(delta=delta, alphas=list(orders)))


def composed_prv_epsilon(mechanisms: Mechanisms, delta: float) -> float:
    """The same from Opacus's ``PRVAccountant``: the upper end of its estimate, within
    :data:`PRV_EPS_ERROR`."""
    from opacus.accountants import PRVAccountant

    history = _history(mechanisms)
    if history is None:
        return float("inf")
    if not history:
        return 0.0
    accountant = PRVAccountant()
    accountant.history = history
    return float(accountant.get_epsilon(delta, eps_error=PRV_EPS_ERROR))


def epsilon_for(sampling_rate: float, noise_multiplier: float, steps: int, delta: float) -> float:
    """Epsilon of a Poisson-subsampled Gaussian mechanism composed ``steps`` times (RDP)."""
    return composed_epsilon([(sampling_rate, noise_multiplier, steps)], delta)


def pld_epsilon_for(
    sampling_rate: float, noise_multiplier: float, steps: int, delta: float
) -> float:
    """Epsilon of the same mechanism from the tighter accountant (Opacus's PRV; the name is
    kept from dp-accounting's PLD accountant, which it replaces)."""
    return composed_prv_epsilon([(sampling_rate, noise_multiplier, steps)], delta)


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
        epsilon_pld=pld_epsilon_for(q, noise_multiplier, steps, d),
        tight_accountant=PRV_ACCOUNTANT,
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
        epsilon_pld=pld_epsilon_for(q, noise_multiplier, rounds, d),
        tight_accountant=PRV_ACCOUNTANT,
        delta=d,
        noise_multiplier=noise_multiplier,
        l2_norm_clip=clip_norm,
        sampling_rate=q,
        steps=rounds,
        population=n_clients,
        noise_seeded=noise_seeded,
    )

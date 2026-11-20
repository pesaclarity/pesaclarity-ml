"""Opacus's privacy budgets beside the figures Google's ``dp-accounting`` gave, which were
signed off in pesaclarity-ml#3 (Opacus computes them since decision 0023, stage 4).

``accountants.v1.json`` (``akili accountants``) records, for the same mechanisms:

- ``cases``: every DP config in ``configs/`` (as ``akili epsilon`` plans it), the example in
  docs/DP-SGD-REVIEW.md, and a grid of sampling rates, noise multipliers, steps and deltas;
  each with Opacus's RDP epsilon at the integer orders the gate uses (``rdp``), at the orders
  signed off (:data:`~akili.privacy.accounting.SIGNED_OFF_RDP_ORDERS`,
  ``rdp_signed_off_orders``) and PRV, beside dp-accounting's RDP and PLD epsilon;
- ``ledgers``: the entries of recorded privacy ledgers and their totals, composed by Opacus's
  RDP accountant, beside dp-accounting's;
- ``limits``: for each config's and each grid point's sampling rate and noise multiplier, and
  each track, the most steps whose RDP epsilon stays within the track's limits, by each. The
  gate admits no run the signed-off figure refused when Opacus's never allows more steps.

dp-accounting was removed with TensorFlow (stage 5). Its columns (``dp_accounting``,
``dp_accounting_max_steps``) are the record of the signed-off figures, as dp-accounting 0.6.0
computed them at the ``tflite-archive`` tag: they are read from the committed file and never
recomputed. Only Opacus's are. A case or ledger added since has no dp-accounting column.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from importlib import metadata, resources
from itertools import product
from pathlib import Path
from typing import Any

from akili.config import ExperimentConfig, load_config
from akili.privacy.accounting import (
    PRV_EPS_ERROR,
    RDP_ORDERS,
    SIGNED_OFF_RDP_ORDERS,
    PrivacyReport,
    PrivacyUnit,
    composed_epsilon,
    composed_prv_epsilon,
    dp_fedavg_report,
    dp_sgd_report,
)
from akili.tracks import Track, policy_for

FILE = "accountants.v1.json"
DIGITS = 6  # epsilons are recorded to six decimals
GRID: dict[str, tuple[float, ...]] = {
    "sampling_rate": (0.001, 0.01, 0.05, 0.1),
    "noise_multiplier": (0.8, 1.0, 1.5, 2.0),
    "steps": (100, 1000),
    "delta": (1e-6, 1e-7),
}
# docs/DP-SGD-REVIEW.md's example (check 3): n = 9,600, q = 0.0133, 375 steps, sigma 1.1.
REVIEW_EXAMPLE = {"sampling_rate": 128 / 9600, "noise_multiplier": 1.1, "steps": 375, "delta": 1e-6}
MAX_STEPS = 1_000_000

Mechanism = tuple[float, float, int]


def default_path() -> Path:
    return Path(str(resources.files("akili.resources").joinpath(FILE)))


def planned_report(cfg: ExperimentConfig, population: int | None = None) -> PrivacyReport | None:
    """The budget ``cfg`` would spend, with the population estimated from its data settings
    unless given (``akili epsilon``); None without DP."""
    trained = 1 - cfg.training.validation_fraction
    if cfg.federated is not None:
        fed = cfg.federated
        n = int(cfg.data.n_users * trained)
        if cfg.track is Track.COMMERCIAL:
            n = int(n * cfg.data.commercial_opt_in_rate)
        n_clients = population or max(1, n)
        return dp_fedavg_report(
            n_clients=n_clients,
            clients_per_round=min(fed.clients_per_round, n_clients),
            rounds=fed.rounds,
            noise_multiplier=fed.update_noise_multiplier,
            clip_norm=fed.update_clip_norm,
            noise_seeded=False,
            delta=cfg.dp.target_delta,
        )
    if not cfg.dp.enabled:
        return None
    users = cfg.dp.unit is PrivacyUnit.USER
    n = population or int(
        cfg.data.n_users * trained
        if users
        else cfg.data.n_users * trained * cfg.data.transactions_per_user
    )
    return dp_sgd_report(
        population=n,
        per_step=cfg.dp.users_per_step if users else cfg.training.batch_size,
        epochs=cfg.training.epochs,
        noise_multiplier=cfg.dp.noise_multiplier,
        l2_norm_clip=cfg.dp.l2_norm_clip,
        noise_seeded=cfg.dp.noise_seed is not None,
        unit=cfg.dp.unit,
        delta=cfg.dp.target_delta,
    )


def _opacus_signed_off_orders(mechanisms: Sequence[Mechanism], delta: float) -> float:
    """Opacus's RDP with the orders signed off, fractional ones included."""
    return composed_epsilon(mechanisms, delta, orders=SIGNED_OFF_RDP_ORDERS)


def _r(x: float) -> float:
    return round(x, DIGITS)


def _max_steps(epsilon: Any, q: float, sigma: float, delta: float, limit: float) -> int:
    """The most steps (up to :data:`MAX_STEPS`) whose epsilon is within ``limit``."""
    lo, hi = 0, MAX_STEPS
    if epsilon([(q, sigma, hi)], delta) <= limit:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if epsilon([(q, sigma, mid)], delta) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def _cases(configs: Path | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(configs.glob("*.yaml")) if configs else []:
        report = planned_report(load_config(path))
        if report is None or report.noise_multiplier <= 0:
            continue
        cases.append(
            {
                "id": f"config:{path.stem}",
                "track": load_config(path).track.value,
                "sampling_rate": report.sampling_rate,
                "noise_multiplier": report.noise_multiplier,
                "steps": report.steps,
                "delta": report.delta,
            }
        )
    cases.append({"id": "dp-sgd-review-example", **REVIEW_EXAMPLE})
    for q, sigma, steps, delta in product(*GRID.values()):
        cases.append(
            {
                "id": f"grid:q={q}:sigma={sigma}:steps={steps}:delta={delta}",
                "sampling_rate": q,
                "noise_multiplier": sigma,
                "steps": int(steps),
                "delta": delta,
            }
        )
    return cases


def _ledger(path: Path) -> list[dict[str, Any]]:
    """Each dataset and unit of a recorded ledger file, with its entries' mechanisms."""
    from akili.privacy.ledger import Ledger

    ledger = Ledger(path)
    out = []
    entries = ledger.entries()
    for dataset in sorted({e.dataset for e in entries}):
        for unit in sorted({e.report.unit for e in entries if e.dataset == dataset}):
            runs = [e for e in entries if e.dataset == dataset and e.report.unit is unit]
            out.append(
                {
                    "id": f"ledger:{dataset}:{unit.value}",
                    "delta": min(e.report.delta for e in runs),
                    "entries": [
                        {
                            "run": e.run,
                            "purpose": e.purpose.value,
                            "sampling_rate": e.report.sampling_rate,
                            "noise_multiplier": e.report.noise_multiplier,
                            "steps": e.report.steps,
                            "delta": e.report.delta,
                            "recorded_epsilon": e.report.epsilon,
                        }
                        for e in runs
                    ],
                }
            )
    return out


def _mechanism(c: dict[str, Any]) -> Mechanism:
    return (float(c["sampling_rate"]), float(c["noise_multiplier"]), int(c["steps"]))


def build(
    configs: Path | None = Path("configs"),
    ledgers: Sequence[Path] = (),
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The comparison. Cases come from ``configs`` and the grid, ledgers from the ledger files
    (or, for each that ``previous`` holds and is not given again, from ``previous``).
    dp-accounting's figures, the signed-off record, are taken from ``previous`` for every case,
    ledger and limit it holds; Opacus's are computed."""
    before = {c["id"]: c for c in (previous or {}).get("cases", [])}
    before_ledgers = {g["id"]: g for g in (previous or {}).get("ledgers", [])}
    before_limits = {(x["id"], x["track"]): x for x in (previous or {}).get("limits", [])}

    def recorded(now: dict[str, Any], then: dict[str, Any] | None, keys: Sequence[str]) -> Any:
        """The signed-off figure of ``then``, if the record has one for the same mechanism as
        ``now`` (a config whose settings changed keeps no figure it was not signed off with)."""
        if then is None or any(now.get(k) != then.get(k) for k in keys):
            return None
        return then.get("dp_accounting", then.get("dp_accounting_max_steps"))

    def column(figure: Any) -> dict[str, Any]:
        return {} if figure is None else {"dp_accounting": figure}

    mechanism = ("sampling_rate", "noise_multiplier", "steps", "delta")

    cases = []
    for c in _cases(configs) if configs else list(before.values()):
        q, sigma, steps = _mechanism(c)
        d = float(c["delta"])
        prior = recorded(c, before.get(c["id"]), mechanism)
        cases.append(
            {
                **{k: c[k] for k in c if k not in ("dp_accounting", "opacus")},
                **column(prior),
                "opacus": {
                    "rdp": _r(composed_epsilon([(q, sigma, steps)], d)),
                    "rdp_signed_off_orders": _r(_opacus_signed_off_orders([(q, sigma, steps)], d)),
                    "prv": _r(composed_prv_epsilon([(q, sigma, steps)], d)),
                },
            }
        )

    groups: dict[str, dict[str, Any]] = {}
    for path in ledgers:
        for g in _ledger(path):
            groups[g["id"]] = g
    for gid, g in before_ledgers.items():
        groups.setdefault(gid, {k: g[k] for k in ("id", "delta", "entries")})
    out_ledgers = []
    for g in groups.values():
        mechs = [_mechanism(e) for e in g["entries"]]
        d = float(g["delta"])
        prior = recorded(g, before_ledgers.get(g["id"]), ("delta", "entries"))
        out_ledgers.append(
            {
                **{k: v for k, v in g.items() if k not in ("dp_accounting", "opacus")},
                **column(prior),
                "opacus": {
                    "rdp": _r(composed_epsilon(mechs, d)),
                    "rdp_signed_off_orders": _r(_opacus_signed_off_orders(mechs, d)),
                },
            }
        )

    limits = []
    pairs: dict[tuple[float, float], list[str]] = {}
    for c in cases:
        if c["id"].startswith("config:"):
            pairs.setdefault(_mechanism(c)[:2], []).append(c["id"])
    for c in cases:
        if c["id"].startswith("grid:"):
            q, sigma, _ = _mechanism(c)
            pairs.setdefault((q, sigma), [f"grid:q={q}:sigma={sigma}"])
    for (q, sigma), ids in pairs.items():
        c = {"id": ",".join(ids)}
        for track in Track:
            p = policy_for(track)
            prior = recorded(
                {
                    "sampling_rate": q,
                    "noise_multiplier": sigma,
                    "max_delta": p.max_delta,
                    "max_epsilon": p.max_epsilon,
                },
                before_limits.get((c["id"], track.value)),
                ("sampling_rate", "noise_multiplier", "max_delta", "max_epsilon"),
            )
            limits.append(
                {
                    "id": c["id"],
                    "track": track.value,
                    "max_epsilon": p.max_epsilon,
                    "max_delta": p.max_delta,
                    "sampling_rate": q,
                    "noise_multiplier": sigma,
                    **({} if prior is None else {"dp_accounting_max_steps": prior}),
                    "opacus_max_steps": _max_steps(
                        composed_epsilon, q, sigma, p.max_delta, p.max_epsilon
                    ),
                    "opacus_signed_off_orders_max_steps": _max_steps(
                        _opacus_signed_off_orders, q, sigma, p.max_delta, p.max_epsilon
                    ),
                }
            )

    versions = dict((previous or {}).get("versions", {}))
    versions["opacus"] = metadata.version("opacus")
    return {
        "note": (
            "Privacy budgets of the same mechanisms from Opacus (decision 0023, stage 4), beside "
            "dp-accounting's, the figures signed off in pesaclarity-ml#3. dp-accounting was "
            "removed in stage 5: its columns are the record from dp-accounting 0.6.0 (tag "
            "tflite-archive) and are not recomputed. Regenerate Opacus's with "
            "`akili accountants`."
        ),
        "versions": versions,
        "rdp_orders": list(RDP_ORDERS),
        "signed_off_rdp_orders": list(SIGNED_OFF_RDP_ORDERS),
        "prv_eps_error": PRV_EPS_ERROR,
        "pld_value_discretization_interval": 1e-4,
        "cases": cases,
        "ledgers": out_ledgers,
        "limits": limits,
    }


def differences(doc: dict[str, Any]) -> dict[str, Any]:
    """The largest differences between the libraries over the finite figures that have a
    signed-off column: Opacus's RDP against dp-accounting's (relative), PRV against PLD
    (absolute), and whether the gate admits the same steps."""
    cases = [c for c in doc["cases"] if "dp_accounting" in c]
    ledgers = [g for g in doc["ledgers"] if "dp_accounting" in g]
    limits = [x for x in doc["limits"] if "dp_accounting_max_steps" in x]
    finite = [c for c in cases if 0 < c["dp_accounting"]["rdp"] < math.inf]

    def rel(key: str) -> list[float]:
        return [
            (c["opacus"][key] - c["dp_accounting"]["rdp"]) / c["dp_accounting"]["rdp"]
            for c in finite
        ]

    rdp, same_orders = rel("rdp"), rel("rdp_signed_off_orders")
    tight = [
        c["opacus"]["prv"] - c["dp_accounting"]["pld"]
        for c in cases
        if c["dp_accounting"]["pld"] < math.inf
    ]
    return {
        "rdp_relative_min": min(rdp),
        "rdp_relative_max": max(rdp),
        "rdp_signed_off_orders_relative_min": min(same_orders),
        "rdp_signed_off_orders_relative_max": max(same_orders),
        "configs_same": all(
            c["opacus"]["rdp"] == c["dp_accounting"]["rdp"]
            for c in cases
            if c["id"].startswith("config:")
        ),
        "prv_minus_pld_min": min(tight),
        "prv_minus_pld_max": max(tight),
        "ledgers_relative_max": max(
            (
                abs(g["opacus"]["rdp"] - g["dp_accounting"]["rdp"]) / g["dp_accounting"]["rdp"]
                for g in ledgers
            ),
            default=0.0,
        ),
        "limits_never_more_steps": all(
            x["opacus_max_steps"] <= x["dp_accounting_max_steps"] for x in limits
        ),
        "limits_same": all(x["opacus_max_steps"] == x["dp_accounting_max_steps"] for x in limits),
        "limits_signed_off_orders_more_steps": [
            f"{x['id']} {x['track']}: {x['opacus_signed_off_orders_max_steps']} against "
            f"{x['dp_accounting_max_steps']}"
            for x in limits
            if x["opacus_signed_off_orders_max_steps"] > x["dp_accounting_max_steps"]
        ],
    }


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=True) + "\n"


def mismatches(committed: dict[str, Any], fresh: dict[str, Any], rel: float = 1e-5) -> list[str]:
    """Where ``fresh`` (recomputed) differs from ``committed`` beyond ``rel``."""
    out: list[str] = []

    def close(a: float, b: float) -> bool:
        return a == b or abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)

    for kind in ("cases", "ledgers"):
        old = {c["id"]: c for c in committed.get(kind, [])}
        for c in fresh.get(kind, []):
            o = old.get(c["id"])
            if o is None:
                out.append(f"{c['id']}: not in the committed file")
                continue
            for lib in ("dp_accounting", "opacus"):
                if (lib in c) != (lib in o):
                    out.append(f"{c['id']} {lib}: in one file only")
                    continue
                for k, v in c.get(lib, {}).items():
                    if not close(float(v), float(o[lib][k])):
                        out.append(f"{c['id']} {lib} {k}: {o[lib][k]} committed, {v} now")
    old_limits = {(x["id"], x["track"]): x for x in committed.get("limits", [])}
    for x in fresh.get("limits", []):
        o = old_limits.get((x["id"], x["track"]))
        for k in (
            "dp_accounting_max_steps",
            "opacus_max_steps",
            "opacus_signed_off_orders_max_steps",
        ):
            if o is None or o.get(k) != x.get(k):
                was = o and o.get(k)
                out.append(f"{x['id']} {x['track']} {k}: {was} committed, {x.get(k)} now")
    return out

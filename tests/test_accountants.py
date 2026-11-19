"""dp-accounting against Opacus (akili.privacy.compare, accountants.v1.json): the committed
figures are what the accountants give, and Opacus's figure, which the gate checks, is never
below the one signed off in pesaclarity-ml#3, is the same at every config, and admits no more
steps at the track limits. dp-accounting's own columns are recomputed while it is installed
(until stage 5); after that the file is their record."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from akili._optional import has_module
from akili.config import load_config
from akili.privacy import compare
from akili.privacy.accounting import RDP_ORDERS, SIGNED_OFF_RDP_ORDERS, composed_epsilon

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

DOC = json.loads(compare.default_path().read_text(encoding="utf-8"))
ROUNDING = 2e-6  # the file keeps six decimals


def _subset(ids: set[str]) -> dict[str, object]:
    return {
        **DOC,
        "cases": [c for c in DOC["cases"] if c["id"] in ids],
        "limits": [x for x in DOC["limits"] if x["id"].startswith("config:")],
    }


def test_the_file_covers_every_dp_config() -> None:
    want = {
        f"config:{p.stem}"
        for p in CONFIG_DIR.glob("*.yaml")
        if (r := compare.planned_report(load_config(p))) is not None and r.noise_multiplier > 0
    }
    assert want <= {c["id"] for c in DOC["cases"]}
    assert {"config:dp-sgd", "config:federated-app", "config:federated-commercial"} <= want
    assert len([c for c in DOC["cases"] if c["id"].startswith("grid:")]) == 64
    assert DOC["ledgers"], "no recorded ledger in the comparison"


def test_opacus_gives_the_committed_figures() -> None:
    """The configs, the review's example and the ledgers, recomputed (dp-accounting's too while
    it is installed): the grid is checked by `akili accountants --check`."""
    keep = {c["id"] for c in DOC["cases"] if not c["id"].startswith("grid:")}
    committed = _subset(keep)
    fresh = compare.build(None, [], committed)
    assert compare.mismatches(committed, fresh) == []


def test_the_gate_figure_is_never_below_the_signed_off_one() -> None:
    for c in [*DOC["cases"], *DOC["ledgers"]]:
        old, new = c["dp_accounting"]["rdp"], c["opacus"]["rdp"]
        assert new >= old - ROUNDING, c["id"]
    for c in DOC["cases"]:
        if c["id"].startswith("config:") or c["id"] == "dp-sgd-review-example":
            assert c["opacus"]["rdp"] == c["dp_accounting"]["rdp"], c["id"]
        assert c["opacus"]["prv"] >= c["dp_accounting"]["pld"] - ROUNDING, c["id"]
        assert c["opacus"]["prv"] <= c["opacus"]["rdp"] + ROUNDING, c["id"]


def test_the_limits_admit_no_more_steps() -> None:
    for x in DOC["limits"]:
        assert x["opacus_max_steps"] <= x["dp_accounting_max_steps"], x
        if x["id"].startswith("config:"):
            assert x["opacus_max_steps"] > 0 or x["dp_accounting_max_steps"] == 0, x
    d = compare.differences(DOC)
    assert d["limits_never_more_steps"]
    assert d["configs_same"]


def test_the_gate_uses_the_integer_orders() -> None:
    """With the fractional orders Opacus's exact series can come out below dp-accounting's
    bound; the file shows where, and the gate's orders leave them out."""
    assert all(o.is_integer() for o in RDP_ORDERS)
    assert set(RDP_ORDERS) == {o for o in SIGNED_OFF_RDP_ORDERS if o.is_integer()}
    below = [
        c
        for c in DOC["cases"]
        if c["opacus"]["rdp_signed_off_orders"] < c["dp_accounting"]["rdp"] - ROUNDING
    ]
    assert below  # why the fractional orders are left out
    c = below[0]
    mech = [(c["sampling_rate"], c["noise_multiplier"], c["steps"])]
    assert composed_epsilon(mech, c["delta"]) >= c["dp_accounting"]["rdp"] - ROUNDING


@pytest.mark.skipif(not has_module("dp_accounting"), reason="dp-accounting is not installed")
def test_dp_accounting_gives_its_committed_figures() -> None:
    """Some of the grid, the slowest cases left to `akili accountants --check`."""
    keep = {c["id"] for c in DOC["cases"] if c["id"].endswith("steps=100:delta=1e-06")}
    committed = {**_subset(keep), "ledgers": []}
    fresh = compare.build(None, [], committed)
    assert compare.mismatches({"cases": committed["cases"]}, {"cases": fresh["cases"]}) == []


def test_planned_report_is_what_epsilon_prints(capsys: pytest.CaptureFixture[str]) -> None:
    from akili.cli import main

    assert main(["epsilon", "--config", str(CONFIG_DIR / "dp-sgd.yaml")]) == 0
    printed = json.loads(capsys.readouterr().out)
    report = compare.planned_report(load_config(CONFIG_DIR / "dp-sgd.yaml"))
    assert report is not None
    assert printed["epsilon"] == pytest.approx(report.epsilon)
    assert printed["accountant"] == "opacus-rdp"
    assert printed["tight_accountant"] == "opacus-prv"
    assert math.isfinite(printed["epsilon_pld"])
    assert compare.planned_report(load_config(CONFIG_DIR / "smoke.yaml")) is None


def test_mismatches_are_named(tmp_path: Path) -> None:
    changed = json.loads(json.dumps(DOC))
    changed["cases"][0]["opacus"]["rdp"] += 0.01
    changed["limits"][0]["opacus_max_steps"] += 1
    out = compare.mismatches(DOC, changed)
    assert len(out) == 2
    assert DOC["cases"][0]["id"] in out[0]

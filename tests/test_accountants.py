"""Opacus against the dp-accounting figures signed off in pesaclarity-ml#3 (akili.privacy.compare,
accountants.v1.json): the committed Opacus figures are what Opacus gives, and its figure, which
the gate checks, is never below the signed-off one, is the same at every config, and admits no
more steps at the track limits. dp-accounting was removed in stage 5: the file is the record of
its columns, kept as they were signed off."""

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

# The privacy accountant, Opacus, is in the [torch] extra (and [train]).
needs_opacus = pytest.mark.skipif(not has_module("opacus"), reason="needs the [torch] extra")


def _subset(ids: set[str]) -> dict[str, object]:
    return {
        **DOC,
        "cases": [c for c in DOC["cases"] if c["id"] in ids],
        "limits": [x for x in DOC["limits"] if x["id"].startswith("config:")],
    }


@pytest.mark.torch
@needs_opacus
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


@pytest.mark.torch
@needs_opacus
def test_opacus_gives_the_committed_figures() -> None:
    """The configs, the review's example and the ledgers, recomputed, with the signed-off columns
    kept as they are: the grid is checked by `akili accountants --check`."""
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


@pytest.mark.torch
@needs_opacus
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


def test_every_signed_off_figure_is_kept() -> None:
    """The record has a dp-accounting column for every case, ledger and limit it was signed
    off with: none was dropped when dp-accounting was removed."""
    assert DOC["versions"]["dp-accounting"] == "0.6.0"
    assert all({"rdp", "pld"} <= set(c["dp_accounting"]) for c in DOC["cases"])
    assert all("rdp" in g["dp_accounting"] for g in DOC["ledgers"])
    assert all("dp_accounting_max_steps" in x for x in DOC["limits"])


@pytest.mark.torch
@needs_opacus
def test_a_new_case_has_no_signed_off_column() -> None:
    """A case the record does not hold gets Opacus's figures only; the comparison leaves it out."""
    case = {k: v for k, v in DOC["cases"][0].items() if k != "dp_accounting"}
    previous = {**DOC, "cases": [case], "ledgers": [], "limits": []}
    fresh = compare.build(None, [], previous)
    assert "dp_accounting" not in fresh["cases"][0]
    assert set(fresh["cases"][0]["opacus"]) == {"rdp", "rdp_signed_off_orders", "prv"}
    assert compare.mismatches({"cases": [case]}, {"cases": fresh["cases"]}) == []
    assert all("dp_accounting_max_steps" not in x for x in fresh["limits"])
    assert compare.differences({**fresh, "cases": DOC["cases"]})["limits_never_more_steps"]


@pytest.mark.torch
@needs_opacus
def test_a_changed_mechanism_keeps_no_signed_off_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config whose settings change keeps its id; the figure signed off for the old settings
    is not carried over to the new ones."""
    case = next(c for c in DOC["cases"] if c["id"] == "config:dp-sgd")
    plain = {k: v for k, v in case.items() if k not in ("dp_accounting", "opacus")}
    previous = {**DOC, "ledgers": []}

    def build(c: dict[str, object]) -> dict[str, object]:
        monkeypatch.setattr(compare, "_cases", lambda _configs: [c])
        return compare.build(CONFIG_DIR, [], previous)

    kept = build(plain)
    assert kept["cases"][0]["dp_accounting"] == case["dp_accounting"]
    moved = build({**plain, "steps": case["steps"] + 1})
    assert "dp_accounting" not in moved["cases"][0]
    other = build({**plain, "noise_multiplier": case["noise_multiplier"] + 0.1})
    assert "dp_accounting" not in other["cases"][0]
    assert all("dp_accounting_max_steps" not in x for x in other["limits"])


@pytest.mark.torch
@needs_opacus
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

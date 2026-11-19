"""The categoriser's fine-tuning step (akili.export.fine_tune, docs/FINE-TUNING.md): the
committed vectors are what the code gives, torch's automatic gradients match the step as the
spec writes it, and the manifest block names the tensors the step trains. Needs PyTorch (the
[torch] extra)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from akili.categories import default_categories
from akili.config import FeatureConfig, FineTuneConfig, ModelConfig
from akili.export import fine_tune
from akili.model import build_categoriser

pytestmark = pytest.mark.torch

DOC = json.loads(fine_tune.default_path().read_text(encoding="utf-8"))


def test_committed_vectors_match_the_code() -> None:
    assert fine_tune.differences(DOC, fine_tune.build_vectors()) == [], (
        "fine-tuning vectors are out of date: run `uv run akili fine-tune-vectors`"
    )


def test_the_vectors_follow_the_written_step() -> None:
    """Each step of every case, from the weights the last one left, by the spec's formulas
    in float64: the loss before the update and every weight after it."""
    assert DOC["spec_version"] == fine_tune.SPEC_VERSION
    assert DOC["trains"] == list(fine_tune.TRAINED)
    for case in DOC["cases"]:
        w = {k: np.asarray(v) for k, v in case["weights"].items()}
        for step in case["steps"]:
            loss, after = fine_tune.manual_step(w, np.asarray(step["f"]), step["y"], step["lr"])
            assert loss == pytest.approx(step["loss"], abs=DOC["tolerance"])
            for name in fine_tune.TRAINED:
                np.testing.assert_allclose(after[name], step["after"][name], atol=DOC["tolerance"])
            w = {k: np.asarray(v) for k, v in step["after"].items()}


def test_a_dead_unit_keeps_its_weights() -> None:
    case = next(c for c in DOC["cases"] if c["id"] == "dead_unit")
    before, after = case["weights"], case["steps"][0]["after"]
    assert after["hidden.kernel"][0] == before["hidden.kernel"][0]
    assert after["hidden.bias"][0] == before["hidden.bias"][0]
    assert after["category.bias"] != before["category.bias"]


def test_the_manifest_block() -> None:
    n = len(default_categories().labels)
    m = build_categoriser(FeatureConfig(), ModelConfig(), n)
    block = fine_tune.fine_tune_block(m, FineTuneConfig())
    assert block["trains"] == list(fine_tune.TRAINED)
    assert block["input"] == "hidden"
    assert block["weights"] == {
        "hidden.kernel": [32, 36],
        "hidden.bias": [32],
        "category.kernel": [n, 32],
        "category.bias": [n],
    }
    assert block["settings"] == FineTuneConfig().model_dump()


def test_differences_names_what_moved() -> None:
    changed = json.loads(json.dumps(DOC))
    changed["cases"][0]["steps"][0]["loss"] += 1e-3
    changed["cases"][1]["note"] = "other"
    out = fine_tune.differences(DOC, changed)
    assert len(out) == 2
    assert ".cases[0].steps[0].loss" in out[0]

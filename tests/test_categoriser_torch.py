"""The categoriser in PyTorch (akili.model, akili.train.torch_backend): the Keras layer names,
order and layout kept (for the weights file and the federated protocol), the int8
``categoriser.weights`` and its manifest, determinism, the parity check, the size budget and
the pipeline end to end; on rayo's C library when ``RAYO_DIR`` is a rayo checkout, else on
rayo-convert's reference.

Small runs; the released figures are in docs/CATEGORISER.md. Needs the [torch] extra (onnx,
MLflow).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("mlflow")

from akili import model as cm
from akili.categories import default_categories
from akili.config import (
    DPConfig,
    ExperimentConfig,
    ExportConfig,
    FeatureConfig,
    ModelConfig,
    SyntheticDataConfig,
    TrackingConfig,
    TrainingConfig,
)
from akili.export import fine_tune
from akili.export.size import ModelTooLargeError
from akili.export.weights import read_weights
from akili.features import encode_batch, feature_width
from akili.privacy.accounting import PrivacyUnit
from akili.synthetic import generate
from akili.train import torch_backend as tb
from akili.train.pipeline import TrainResult, run_centralised

pytestmark = pytest.mark.torch

N_CLASSES = len(default_categories().labels)


def tiny(tmp_path: Path, **update: object) -> ExperimentConfig:
    cfg = ExperimentConfig(
        name="tiny",
        output_dir=tmp_path / "artefacts",
        data=SyntheticDataConfig(n_users=30, transactions_per_user=20),
        training=TrainingConfig(epochs=2, batch_size=32),
        export=ExportConfig(max_size_bytes=512_000, parity_rows=300),
        tracking=TrackingConfig(
            uri=f"sqlite:///{tmp_path}/mlruns/mlflow.db",
            artifact_root=tmp_path / "mlruns" / "artifacts",
            experiment="akili-test",
        ),
    )
    return cfg.model_copy(update=update)


def test_layers_keep_the_keras_names_and_count() -> None:
    m = cm.build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES)
    names = [name for name, _ in m.named_children()]
    assert names == [*(layer["name"] for layer in cm.LAYERS[:-1]), "dropout", "category"]
    # The Keras categoriser's count: 4096 x 16 + (12 + 2 + 6 + 7 + 3) x 4 + 36 x 32 + 32
    # + 32 x classes + classes.
    assert cm.parameter_count(m) == 4096 * 16 + 30 * 4 + 36 * 32 + 32 + 33 * N_CLASSES
    x = torch.zeros((3, feature_width(FeatureConfig())), dtype=torch.int64)
    p = m.eval()(x)
    assert p.shape == (3, N_CLASSES)
    np.testing.assert_allclose(p.sum(1).detach().numpy(), 1.0, rtol=1e-6)


def test_padding_tokens_do_not_count() -> None:
    """The mean is over the tokens that are not 0, as the Keras model's masked mean."""
    m = cm.build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES).eval()
    row = torch.tensor([[5, 9, 0, 0, 0, 0, 0, 0, 3, 1, 2, 4, 0]])
    body = m.body(row)[0, :16]
    want = (m.token_embedding.weight[5] + m.token_embedding.weight[9]) / 2
    torch.testing.assert_close(body, want)
    empty = m.body(torch.tensor([[0] * 8 + [3, 1, 2, 4, 0]]))[0, :16]
    assert not empty.any()


def test_keras_layout_round_trip() -> None:
    """The federated protocol's arrays: Keras' get_weights() order, dense kernels [in, out]."""
    m = cm.build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES, seed=1)
    w = cm.keras_layout(m)
    assert [a.shape for a in w] == [
        (4096, 16),
        (12, 4),
        (2, 4),
        (6, 4),
        (7, 4),
        (3, 4),
        (36, 32),
        (32,),
        (32, N_CLASSES),
        (N_CLASSES,),
    ]
    np.testing.assert_array_equal(w[6], m.hidden.weight.detach().numpy().T)
    other = cm.build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES, seed=2)
    cm.set_keras_layout(other, w)
    for a, b in zip(m.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="shape"):
        cm.set_keras_layout(other, [*w[:6], w[6].T, *w[7:]])


@pytest.fixture(scope="module")
def dp_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[ExperimentConfig, TrainResult]:
    cfg = tiny(tmp_path_factory.mktemp("dp"), dp=DPConfig(enabled=True, users_per_step=4))
    return cfg, run_centralised(cfg)


def test_pipeline_writes_the_weights_and_manifest(
    dp_run: tuple[ExperimentConfig, TrainResult],
) -> None:
    cfg, result = dp_run
    out = result.run_dir
    assert result.backend == "torch"
    assert result.exported.path == out / cm.WEIGHTS_FILE
    data = (out / cm.WEIGHTS_FILE).read_bytes()
    onnx = (out / cm.ONNX_FILE).read_bytes()
    manifest = json.loads((out / "model-manifest.json").read_text())
    assert manifest["weights_file"] == cm.WEIGHTS_FILE
    assert manifest["weights_sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["weights_size_bytes"] == len(data) == result.exported.size_bytes
    assert manifest["source"] == {
        "file": cm.ONNX_FILE,
        "sha256": hashlib.sha256(onnx).hexdigest(),
        "size_bytes": len(onnx),
    }
    assert manifest["input"] == {"name": "features", "dtype": "int32", "shape": [-1, 13]}
    assert manifest["rayo_inputs"] == {"features": {"dtype": "int32", "shape": [13]}}
    assert manifest["outputs"] == ["category"]
    assert "model_file" not in manifest  # no TFLite file for this model any more
    assert "trainable" not in manifest
    block = manifest["fine_tune"]
    assert block["spec_version"] == fine_tune.SPEC_VERSION
    assert block["trains"] == list(fine_tune.TRAINED)
    assert block["weights"]["hidden.kernel"] == [32, 36]  # [out, in], as in the weights file
    assert block["settings"] == cfg.export.fine_tune.model_dump()
    assert (out / "labels.json").exists()
    assert not (out / "categoriser.tflite").exists()


def test_the_weights_file(dp_run: tuple[ExperimentConfig, TrainResult]) -> None:
    _, result = dp_run
    out = result.run_dir
    w = read_weights((out / cm.WEIGHTS_FILE).read_bytes())
    assert w.model == "categoriser"
    assert w.source_file == cm.ONNX_FILE
    assert w.layers == cm.LAYERS
    t = w.tensors
    assert set(cm.required_tensors()) <= set(t)
    assert t["token_embedding.table"].value.dtype == np.int8
    assert t["token_embedding.table"].axis is None  # one scale for the table
    assert t["hidden.kernel"].value.dtype == np.int8
    assert t["hidden.kernel"].value.shape == (32, 36)
    assert t["hidden.kernel"].axis == 0  # one scale per output
    for i in range(5):  # under 1,024 values: float32, as TFLite kept them
        assert t[f"context_{i}.table"].value.dtype == np.float32
    assert [i["name"] for i in (w.graph or {})["inputs"]] == ["features"]


def test_the_run_is_tracked(dp_run: tuple[ExperimentConfig, TrainResult]) -> None:
    import mlflow

    _, result = dp_run
    out = result.run_dir
    data = (out / cm.WEIGHTS_FILE).read_bytes()
    report = json.loads((out / "export-report.json").read_text())
    parity = report["parity"]
    assert parity["within"]
    assert parity["vectors"] == 60  # feature_vectors.v1.json
    # Two noisy epochs leave the model unsure of every row, so no label is compared here;
    # test_on_rayo compares a trained one's.
    assert parity["labels_compared"] == parity["labels_compared_same"]
    # The file's accuracy is the trained model's to within int8's rounding.
    assert result.scores["val_accuracy_shipped"] == pytest.approx(result.accuracy, abs=0.05)

    assert result.privacy is not None
    assert result.privacy.unit is PrivacyUnit.USER
    assert result.privacy.accountant == "opacus-rdp"
    assert result.privacy.tight_accountant == "opacus-prv"
    assert result.budget is not None
    assert result.budget.runs == 1
    run = mlflow.get_run(result.run_id)
    assert run.data.tags["backend"] == "torch"
    assert run.data.metrics["model.size_bytes"] == len(data)
    assert "val.accuracy_shipped" in run.data.metrics
    assert run.data.metrics["privacy.epsilon"] == pytest.approx(result.privacy.epsilon)


def test_training_and_export_are_deterministic(tmp_path: Path) -> None:
    cfg = tiny(tmp_path)
    txs = generate(cfg.data, default_categories())
    b = encode_batch(txs, cfg.features, default_categories().labels)
    users = np.array([tx.user_id for tx in txs])
    runs = [
        tb.to_weights(
            tb.fit(cfg, b.features, b.labels, users=users, seed=3, n_classes=N_CLASSES).model, cfg
        )
        for _ in range(2)
    ]
    assert runs[0].onnx == runs[1].onnx
    assert runs[0].sha256 == runs[1].sha256


def test_the_export_refuses_a_file_over_budget(tmp_path: Path) -> None:
    cfg = tiny(tmp_path, export=ExportConfig(max_size_bytes=1_000))
    m = cm.build_categoriser(cfg.features, cfg.model, N_CLASSES)
    with pytest.raises(ModelTooLargeError, match=r"categoriser\.weights is \d+ bytes"):
        tb.export_categoriser(m, cfg, tmp_path / "out")


def test_the_export_refuses_a_file_outside_the_tolerance(tmp_path: Path) -> None:
    cfg = tiny(tmp_path, export=ExportConfig(parity_tolerance=1e-9, parity_rows=50))
    m = cm.build_categoriser(cfg.features, cfg.model, N_CLASSES)
    with pytest.raises(ValueError, match="strays from the torch model"):
        tb.export_categoriser(m, cfg, tmp_path / "out")
    assert (tmp_path / "out" / "export-report.json").exists()  # so the failure can be read
    assert not (tmp_path / "out" / cm.WEIGHTS_FILE).exists()


def test_parity_inputs_are_new_rows_and_the_feature_vectors(tmp_path: Path) -> None:
    cfg = tiny(tmp_path)
    x, ids = tb.parity_inputs(cfg)
    assert len(ids) == 60
    assert len(x) == cfg.export.parity_rows + 60
    other = cfg.model_copy(update={"features": FeatureConfig(max_tokens=6)})
    x, ids = tb.parity_inputs(other)  # the vectors are for the default feature config only
    assert ids == []
    assert x.shape == (cfg.export.parity_rows, 11)


def test_on_rayo(tmp_path: Path) -> None:
    """The file as the app's rayo runs it: rayo-convert's reference of the same int8 graph to
    about 1e-6, and torch's labels."""
    d = os.environ.get("RAYO_DIR")
    if not d or not (Path(d) / "src" / "model.c").exists():
        pytest.skip("RAYO_DIR: no rayo checkout to build rayo's C library from")
    cfg = tiny(tmp_path)
    txs = generate(cfg.data, default_categories())
    b = encode_batch(txs, cfg.features, default_categories().labels)
    users = np.array([tx.user_id for tx in txs])
    model = tb.fit(cfg, b.features, b.labels, users=users, seed=0, n_classes=N_CLASSES).model
    done = tb.to_weights(model, cfg)
    x = b.features[:200]
    rayo = tb.Converted(done.data, done.onnx, d).run(x)
    ref = tb.Converted(done.data, done.onnx, None).run(x)
    assert np.abs(rayo - ref).max() <= 1e-5
    want = tb.predict(model, x)
    assert np.abs(rayo - want).max() <= cfg.export.parity_tolerance
    assert (rayo.argmax(1) == want.argmax(1)).mean() > 0.99


def test_fine_tuning_learns_the_persons_choices(tmp_path: Path) -> None:
    """The spec's step (akili.export.fine_tune) on what ``hidden`` reads moves a category the
    person files elsewhere and keeps the rest."""
    cfg = tiny(tmp_path, data=SyntheticDataConfig(n_users=30, transactions_per_user=40))
    txs = generate(cfg.data, default_categories())
    b = encode_batch(txs, cfg.features, default_categories().labels)
    users = np.array([tx.user_id for tx in txs])
    model = tb.fit(cfg, b.features, b.labels, users=users, seed=0, n_classes=N_CLASSES).model
    x, y = b.features[:300], b.labels[:300]
    with torch.no_grad():
        f = model.body(torch.from_numpy(x.astype(np.int64))).numpy()
    common = int(np.bincount(y).argmax())
    other = (common + 1) % N_CLASSES
    theirs = np.where(y == common, other, y)
    moved = y == common
    w = {
        "hidden.kernel": model.hidden.weight.detach().numpy(),
        "hidden.bias": model.hidden.bias.detach().numpy(),
        "category.kernel": model.category.weight.detach().numpy(),
        "category.bias": model.category.bias.detach().numpy(),
    }
    tune = cfg.export.fine_tune
    for _ in range(tune.epochs):
        for i in range(0, len(x), tune.batch_size):
            _, w = fine_tune.step(
                w, f[i : i + tune.batch_size], theirs[i : i + tune.batch_size], tune.learning_rate
            )
    h = np.maximum(f @ w["hidden.kernel"].T + w["hidden.bias"], 0)
    predicted = (h @ w["category.kernel"].T + w["category.bias"]).argmax(1)
    assert float((predicted[moved] == other).mean()) > 0.8
    kept_before = float((tb.predict(model, x[~moved]).argmax(1) == y[~moved]).mean())
    assert float((predicted[~moved] == y[~moved]).mean()) >= kept_before - 0.02

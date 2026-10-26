"""Model, DP-SGD, TFLite export and tracking. Skipped unless the [train] extra is installed."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")

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
from akili.export.size import ModelTooLargeError
from akili.export.tflite import export_tflite, run_tflite
from akili.features import encode_batch, feature_width
from akili.model import build_categoriser
from akili.synthetic import generate
from akili.train.dp_sgd import train_dp_sgd
from akili.train.pipeline import run_centralised

pytestmark = pytest.mark.tf

N_CLASSES = len(default_categories().labels)


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    txs = generate(SyntheticDataConfig(n_users=30, transactions_per_user=40))
    batch = encode_batch(txs, FeatureConfig(), default_categories().labels)
    return batch.features, batch.labels


def test_model_is_small_and_outputs_probabilities(data: tuple[np.ndarray, np.ndarray]) -> None:
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES)
    assert model.inputs[0].shape[-1] == feature_width(FeatureConfig())
    assert model.count_params() < 100_000
    probs = model.predict(data[0][:4], verbose=0)
    assert probs.shape == (4, N_CLASSES)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_dp_sgd_learns_and_matches_accounting(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES, seed=1)
    stats = train_dp_sgd(
        model,
        x,
        y,
        TrainingConfig(epochs=4, batch_size=64, learning_rate=0.5),
        DPConfig(enabled=True, noise_multiplier=0.5),
        seed=1,
    )
    assert stats.steps == 4 * round(len(y) / 64)
    accuracy = float((model.predict(x, verbose=0).argmax(axis=1) == y).mean())
    assert accuracy > 3 / N_CLASSES


def test_tflite_matches_keras_and_respects_budget(
    tmp_path: Path, data: tuple[np.ndarray, np.ndarray]
) -> None:
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES)
    exported = export_tflite(model, tmp_path / "m.tflite", ExportConfig(quantize=False))
    x = data[0][:8]
    np.testing.assert_allclose(run_tflite(exported.path, x), model.predict(x, verbose=0), atol=1e-5)
    quantised = export_tflite(model, tmp_path / "q.tflite", ExportConfig(quantize=True))
    assert quantised.size_bytes < exported.size_bytes / 2
    with pytest.raises(ModelTooLargeError):
        export_tflite(model, tmp_path / "big.tflite", ExportConfig(max_size_bytes=1_000))


def test_pipeline_tracks_dp_run(tmp_path: Path) -> None:
    import mlflow

    cfg = ExperimentConfig(
        name="tiny-dp",
        output_dir=tmp_path / "artefacts",
        data=SyntheticDataConfig(n_users=20, transactions_per_user=20),
        training=TrainingConfig(epochs=1, batch_size=32, learning_rate=0.5),
        dp=DPConfig(enabled=True),
        tracking=TrackingConfig(
            uri=f"sqlite:///{tmp_path}/mlruns/mlflow.db",
            artifact_root=tmp_path / "mlruns" / "artifacts",
            experiment="akili-test",
        ),
    )
    result = run_centralised(cfg)
    assert result.privacy is not None
    assert (result.run_dir / "privacy-report.json").exists()
    assert (result.run_dir / "labels.json").exists()
    run = mlflow.get_run(result.run_id)
    assert run.data.metrics["privacy.epsilon"] == pytest.approx(result.privacy.epsilon)
    assert run.data.tags["track"] == "app"

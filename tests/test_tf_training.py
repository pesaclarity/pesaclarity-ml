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


def _one_step(
    x: np.ndarray,
    y: np.ndarray,
    *,
    clip: float,
    noise_multiplier: float,
    batch_size: int | None = None,
    seed: int = 0,
) -> tuple[list[np.ndarray], float]:
    """One DP-SGD step from a fresh model; ``batch_size`` defaults to q = 1 (every example).

    Returns the gradient the optimiser applied, recovered from the weights (on the first step
    SGD with momentum moves each weight by exactly ``-learning_rate * gradient``), and the
    realised batch size.
    """
    model = build_categoriser(FeatureConfig(), ModelConfig(dropout=0.0), N_CLASSES, seed=3)
    before = [v.numpy().copy() for v in model.trainable_variables]
    stats = train_dp_sgd(
        model,
        x,
        y,
        TrainingConfig(epochs=1, batch_size=batch_size or len(y), learning_rate=1.0),
        DPConfig(enabled=True, l2_norm_clip=clip, noise_multiplier=noise_multiplier),
        seed=seed,
    )
    assert stats.steps == 1
    applied = [b - v.numpy() for b, v in zip(before, model.trainable_variables, strict=True)]
    return applied, stats.mean_batch_size


@pytest.mark.parametrize("clip", [0.05, 1.0])
def test_dp_sgd_clips_each_example_over_all_variables(
    data: tuple[np.ndarray, np.ndarray], clip: float
) -> None:
    """The applied gradient is the sum of per-example gradients, each clipped to ``clip`` in
    global L2 norm over every trainable variable, divided by the expected batch size."""
    import keras
    import tensorflow as tf

    x, y = data[0][:16], data[1][:16]
    applied, _ = _one_step(x, y, clip=clip, noise_multiplier=0.0)

    reference = build_categoriser(FeatureConfig(), ModelConfig(dropout=0.0), N_CLASSES, seed=3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction="none")
    expected = [np.zeros(v.shape, np.float64) for v in reference.trainable_variables]
    for i in range(len(y)):
        with tf.GradientTape() as tape:
            loss = loss_fn(y[i : i + 1], reference(x[i : i + 1], training=True))
        grads = [
            tf.convert_to_tensor(g).numpy()
            for g in tape.gradient(loss, reference.trainable_variables)
        ]
        norm = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
        assert norm > 0.05  # so the small clip really clips every example
        scale = min(1.0, clip / norm)
        for k, g in enumerate(grads):
            expected[k] += g * scale
    for got, want in zip(applied, expected, strict=True):
        np.testing.assert_allclose(got, want / len(y), atol=1e-6 * max(clip, 1.0))


def test_dp_sgd_noise_std_is_multiplier_times_clip(data: tuple[np.ndarray, np.ndarray]) -> None:
    """Noise on the clipped sum has std ``noise_multiplier * clip`` per coordinate, and the sum is
    divided by the expected batch size q * n, not the realised one, as Poisson accounting assumes.

    The noise is so large that the clipped gradients (norm at most ``clip`` each) are negligible.
    """
    x, y = data[0][:16], data[1][:16]
    clip, sigma, expected_batch = 0.5, 2000.0, 12  # q = 0.75, one step
    applied, realised = _one_step(
        x, y, clip=clip, noise_multiplier=sigma, batch_size=expected_batch, seed=4
    )
    assert realised != expected_batch  # so dividing by the realised size would show
    noise_on_sum = np.concatenate([g.ravel() for g in applied]) * expected_batch
    assert noise_on_sum.size > 50_000
    assert noise_on_sum.std() == pytest.approx(sigma * clip, rel=0.02)
    assert abs(noise_on_sum.mean()) < 0.02 * sigma * clip


def test_dp_sgd_matches_what_the_accountant_is_told(data: tuple[np.ndarray, np.ndarray]) -> None:
    from akili.privacy.accounting import dp_sgd_report

    x, y = data
    training = TrainingConfig(epochs=3, batch_size=50, learning_rate=0.5)
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES, seed=1)
    stats = train_dp_sgd(model, x, y, training, DPConfig(enabled=True), seed=1)
    report = dp_sgd_report(
        n_examples=len(y),
        batch_size=training.batch_size,
        epochs=training.epochs,
        noise_multiplier=1.1,
        l2_norm_clip=1.0,
    )
    assert stats.steps == report.steps
    assert stats.sampling_rate == report.sampling_rate
    # Poisson sampling: the realised batch size varies around q * n.
    assert stats.mean_batch_size == pytest.approx(training.batch_size, rel=0.15)


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
    assert (result.run_dir / "model-manifest.json").exists()
    run = mlflow.get_run(result.run_id)
    assert run.data.metrics["privacy.epsilon"] == pytest.approx(result.privacy.epsilon)
    assert run.data.tags["track"] == "app"

"""The trainable TFLite export used for fine-tuning on the phone. Needs the [train] extra."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from akili.categories import default_categories
from akili.config import (
    ExportConfig,
    FeatureConfig,
    FineTuneConfig,
    ModelConfig,
    SyntheticDataConfig,
)
from akili.export.size import ModelTooLargeError
from akili.export.trainable import WEIGHTS, TrainableRunner, export_trainable, weight_shapes
from akili.features import encode_batch
from akili.model import build_categoriser
from akili.synthetic import generate

pytestmark = pytest.mark.tf

N_CLASSES = len(default_categories().labels)


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    txs = generate(SyntheticDataConfig(n_users=10, transactions_per_user=30))
    batch = encode_batch(txs, FeatureConfig(), default_categories().labels)
    return batch.features, batch.labels


@pytest.fixture(scope="module")
def model():  # type: ignore[no-untyped-def]
    return build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES)


@pytest.fixture
def runner(model, tmp_path: Path) -> TrainableRunner:  # type: ignore[no-untyped-def]
    exported = export_trainable(model, tmp_path / "t.tflite", ExportConfig())
    return TrainableRunner(exported.path)


def test_has_the_four_signatures(runner: TrainableRunner) -> None:
    assert set(runner.signatures) == {"train", "infer", "save", "restore"}
    assert set(runner.signatures["train"]["inputs"]) == {"x", "y", "lr"}
    assert set(runner.signatures["save"]["outputs"]) == set(WEIGHTS)


def test_infer_matches_keras(runner, model, data) -> None:  # type: ignore[no-untyped-def]
    x = data[0][:16]
    np.testing.assert_allclose(runner.infer(x), model.predict(x, verbose=0), atol=1e-5)


def test_one_step_matches_automatic_gradients(runner, model, data) -> None:  # type: ignore[no-untyped-def]
    import tensorflow as tf

    x, y = data[0][:16], data[1][:16]
    hidden, category = model.get_layer("hidden"), model.get_layer("category")
    variables = [hidden.kernel, hidden.bias, category.kernel, category.bias]
    with tf.GradientTape() as tape:
        probs = model(x, training=False)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, probs))
    grads = tape.gradient(loss, [v.value for v in variables])

    before = runner.save()
    reported = runner.train(x, y, learning_rate=1.0)
    after = runner.save()
    assert reported == pytest.approx(float(loss), rel=1e-4)
    for name, grad in zip(WEIGHTS, grads, strict=True):
        np.testing.assert_allclose(before[name] - after[name], grad.numpy(), atol=1e-5)


def test_fine_tuning_learns_the_persons_choices(tmp_path: Path) -> None:
    import keras

    txs = generate(SyntheticDataConfig(n_users=30, transactions_per_user=40))
    batch = encode_batch(txs, FeatureConfig(), default_categories().labels)
    trained = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES)
    trained.compile(optimizer=keras.optimizers.Adam(0.01), loss="sparse_categorical_crossentropy")
    trained.fit(batch.features, batch.labels, epochs=5, batch_size=64, verbose=0)
    x, y = batch.features[:300], batch.labels[:300]
    runner = TrainableRunner(export_trainable(trained, tmp_path / "t.tflite", ExportConfig()).path)

    # The person files their most common category under another one.
    common = int(np.bincount(y).argmax())
    other = (common + 1) % N_CLASSES
    theirs = np.where(y == common, other, y)
    moved = y == common

    tune = FineTuneConfig()
    for _ in range(tune.epochs):
        for i in range(0, len(x), tune.batch_size):
            runner.train(
                x[i : i + tune.batch_size], theirs[i : i + tune.batch_size], tune.learning_rate
            )
    predicted = runner.infer(x).argmax(axis=1)
    assert float((predicted[moved] == other).mean()) > 0.8
    kept_before = float((trained.predict(x[~moved], verbose=0).argmax(axis=1) == y[~moved]).mean())
    assert float((predicted[~moved] == y[~moved]).mean()) >= kept_before - 0.02


def test_save_and_restore_round_trip(runner, data) -> None:  # type: ignore[no-untyped-def]
    x, y = data[0][:32], data[1][:32]
    original = runner.save()
    before = runner.infer(x)
    for _ in range(20):
        runner.train(x, y, 0.5)
    tuned = runner.save()
    assert not np.allclose(runner.infer(x), before)
    runner.restore(original)
    np.testing.assert_allclose(runner.infer(x), before, atol=1e-6)
    runner.restore(tuned)
    np.testing.assert_allclose(runner.save()["w2"], tuned["w2"])


def test_weight_shapes_match_the_signatures(runner, model) -> None:  # type: ignore[no-untyped-def]
    saved = runner.save()
    assert {k: list(v.shape) for k, v in saved.items()} == weight_shapes(model)


def test_respects_the_size_budget(model, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ModelTooLargeError):
        export_trainable(model, tmp_path / "big.tflite", ExportConfig(max_size_bytes=1_000))


def test_fine_tune_examples_must_be_in_order() -> None:
    with pytest.raises(ValueError, match="min_examples"):
        ExportConfig(fine_tune=FineTuneConfig(min_examples=50, max_examples=10))

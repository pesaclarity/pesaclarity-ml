"""Model, DP-SGD, TFLite export and tracking. Skipped unless the [train] extra is installed."""

import json
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
from akili.privacy.accounting import PrivacyUnit
from akili.synthetic import generate
from akili.train.dp_sgd import train_dp_sgd
from akili.train.pipeline import run_centralised

pytestmark = pytest.mark.tf

N_CLASSES = len(default_categories().labels)


EXAMPLES = DPConfig(enabled=True, unit=PrivacyUnit.EXAMPLE)


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    txs = generate(SyntheticDataConfig(n_users=30, transactions_per_user=40))
    batch = encode_batch(txs, FeatureConfig(), default_categories().labels)
    return batch.features, batch.labels


@pytest.fixture(scope="module")
def people() -> np.ndarray:
    """Each row's person in ``data``: 30 people with 40 transactions each."""
    txs = generate(SyntheticDataConfig(n_users=30, transactions_per_user=40))
    return np.array([tx.user_id for tx in txs])


def test_model_is_small_and_outputs_probabilities(data: tuple[np.ndarray, np.ndarray]) -> None:
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES)
    assert model.inputs[0].shape[-1] == feature_width(FeatureConfig())
    assert model.count_params() < 100_000
    probs = model.predict(data[0][:4], verbose=0)
    assert probs.shape == (4, N_CLASSES)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_dp_sgd_learns_per_example(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES, seed=1)
    stats = train_dp_sgd(
        model,
        x,
        y,
        TrainingConfig(epochs=4, batch_size=64, learning_rate=0.5),
        EXAMPLES.model_copy(update={"noise_multiplier": 0.5}),
        noise_seed=1,
    )
    assert stats.unit is PrivacyUnit.EXAMPLE
    assert stats.steps == 4 * round(len(y) / 64)
    accuracy = float((model.predict(x, verbose=0).argmax(axis=1) == y).mean())
    assert accuracy > 3 / N_CLASSES


def test_dp_sgd_learns_per_person(data: tuple[np.ndarray, np.ndarray], people: np.ndarray) -> None:
    x, y = data
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES, seed=1)
    stats = train_dp_sgd(
        model,
        x,
        y,
        TrainingConfig(epochs=20, learning_rate=0.5),
        DPConfig(enabled=True, users_per_step=15, noise_multiplier=0.3),
        users=people,
        noise_seed=1,
    )
    assert stats.unit is PrivacyUnit.USER
    assert stats.population == 30
    assert stats.steps == 20 * 2
    accuracy = float((model.predict(x, verbose=0).argmax(axis=1) == y).mean())
    assert accuracy > 3 / N_CLASSES


def test_user_level_needs_users(data: tuple[np.ndarray, np.ndarray]) -> None:
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES)
    with pytest.raises(ValueError, match="each example's user"):
        train_dp_sgd(model, *data, TrainingConfig(), DPConfig(enabled=True))


def _one_step(
    x: np.ndarray,
    y: np.ndarray,
    *,
    clip: float,
    noise_multiplier: float,
    batch_size: int | None = None,
    noise_seed: int | None = 0,
    users: np.ndarray | None = None,
) -> tuple[list[np.ndarray], float]:
    """One DP-SGD step from a fresh model; ``batch_size`` defaults to q = 1 (every example).
    With ``users``, one step per person instead, with q = 1 (every person).

    Returns the gradient the optimiser applied, recovered from the weights (on the first step
    SGD with momentum moves each weight by exactly ``-learning_rate * gradient``), and the
    realised batch size.
    """
    model = build_categoriser(FeatureConfig(), ModelConfig(dropout=0.0), N_CLASSES, seed=3)
    before = [v.numpy().copy() for v in model.trainable_variables]
    unit = PrivacyUnit.EXAMPLE if users is None else PrivacyUnit.USER
    n_people = 1 if users is None else len(np.unique(users))
    stats = train_dp_sgd(
        model,
        x,
        y,
        TrainingConfig(epochs=1, batch_size=batch_size or len(y), learning_rate=1.0),
        DPConfig(
            enabled=True,
            unit=unit,
            users_per_step=n_people,
            l2_norm_clip=clip,
            noise_multiplier=noise_multiplier,
        ),
        users=users,
        noise_seed=noise_seed,
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
        x, y, clip=clip, noise_multiplier=sigma, batch_size=expected_batch, noise_seed=4
    )
    assert realised != expected_batch  # so dividing by the realised size would show
    noise_on_sum = np.concatenate([g.ravel() for g in applied]) * expected_batch
    assert noise_on_sum.size > 50_000
    assert noise_on_sum.std() == pytest.approx(sigma * clip, rel=0.02)
    assert abs(noise_on_sum.mean()) < 0.02 * sigma * clip


@pytest.mark.parametrize("clip", [0.05, 1.0])
def test_dp_sgd_clips_each_persons_mean_gradient(
    data: tuple[np.ndarray, np.ndarray], people: np.ndarray, clip: float
) -> None:
    """Per person, the applied gradient is the sum over people of the gradient of their mean
    loss, each clipped to ``clip`` over every variable, divided by the expected people."""
    import keras
    import tensorflow as tf

    rows = np.flatnonzero(np.isin(people, np.unique(people)[:4]))[::4]  # 4 people, 10 rows each
    x, y, users = data[0][rows], data[1][rows], people[rows]
    applied, _ = _one_step(x, y, clip=clip, noise_multiplier=0.0, users=users)

    reference = build_categoriser(FeatureConfig(), ModelConfig(dropout=0.0), N_CLASSES, seed=3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    expected = [np.zeros(v.shape, np.float64) for v in reference.trainable_variables]
    for person in np.unique(users):
        mine = users == person
        with tf.GradientTape() as tape:
            loss = loss_fn(y[mine], reference(x[mine], training=True))
        grads = [
            tf.convert_to_tensor(g).numpy()
            for g in tape.gradient(loss, reference.trainable_variables)
        ]
        norm = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
        assert norm > 0.05  # so the small clip really clips every person
        for k, g in enumerate(grads):
            expected[k] += g * min(1.0, clip / norm)
    for got, want in zip(applied, expected, strict=True):
        np.testing.assert_allclose(got, want / 4, atol=1e-6 * max(clip, 1.0))


def test_dp_sgd_caps_each_persons_examples(people: np.ndarray) -> None:
    from akili.train.dp_sgd import group_by_user, user_rows

    groups = group_by_user(people)
    assert len(groups) == 30
    assert all(len(g) == 40 for g in groups)
    idx, mask = user_rows(groups, np.array([0, 3]), 25, np.random.default_rng(0))
    assert idx.shape == (2, 25)
    assert mask.sum() == 50
    assert set(people[idx[0]]) == {people[groups[0][0]]}
    assert len(set(idx[1])) == 25  # sampled without replacement


def test_noise_comes_from_os_entropy_unless_seeded(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data[0][:16], data[1][:16]
    a, _ = _one_step(x, y, clip=1.0, noise_multiplier=1.0, noise_seed=None)
    b, _ = _one_step(x, y, clip=1.0, noise_multiplier=1.0, noise_seed=None)
    c, _ = _one_step(x, y, clip=1.0, noise_multiplier=1.0, noise_seed=7)
    d, _ = _one_step(x, y, clip=1.0, noise_multiplier=1.0, noise_seed=7)
    assert not np.allclose(a[0], b[0])
    for got, want in zip(c, d, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("unit", [PrivacyUnit.USER, PrivacyUnit.EXAMPLE])
def test_dp_sgd_matches_what_the_accountant_is_told(
    data: tuple[np.ndarray, np.ndarray], people: np.ndarray, unit: PrivacyUnit
) -> None:
    from akili.privacy.accounting import dp_sgd_report

    x, y = data
    training = TrainingConfig(epochs=3, batch_size=50, learning_rate=0.5)
    dp = DPConfig(enabled=True, unit=unit, users_per_step=6)
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES, seed=1)
    stats = train_dp_sgd(model, x, y, training, dp, users=people)
    per_person = unit is PrivacyUnit.USER
    report = dp_sgd_report(
        population=30 if per_person else len(y),
        per_step=6 if per_person else 50,
        epochs=training.epochs,
        noise_multiplier=1.1,
        l2_norm_clip=1.0,
        noise_seeded=False,
        unit=unit,
    )
    assert not stats.noise_seeded
    assert stats.population == report.population
    assert stats.steps == report.steps
    assert stats.sampling_rate == report.sampling_rate
    # Poisson sampling: the realised batch size varies around q * n.
    assert stats.mean_batch_size == pytest.approx(6 if per_person else 50, rel=0.25)


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
        dp=DPConfig(enabled=True, users_per_step=4),
        tracking=TrackingConfig(
            uri=f"sqlite:///{tmp_path}/mlruns/mlflow.db",
            artifact_root=tmp_path / "mlruns" / "artifacts",
            experiment="akili-test",
        ),
    )
    result = run_centralised(cfg)
    assert result.privacy is not None
    assert result.privacy.unit is PrivacyUnit.USER
    assert result.privacy.population == 16  # 4 of the 20 people are held out
    assert not result.privacy.noise_seeded
    assert (result.run_dir / "privacy-report.json").exists()
    assert (result.run_dir / "labels.json").exists()
    manifest = json.loads((result.run_dir / "model-manifest.json").read_text())
    trainable = manifest["trainable"]
    assert (result.run_dir / trainable["file"]).exists()
    assert trainable["signatures"] == ["train", "infer", "save", "restore"]
    assert trainable["fine_tune"] == cfg.export.fine_tune.model_dump()
    run = mlflow.get_run(result.run_id)
    assert run.data.metrics["model.trainable_size_bytes"] == trainable["size_bytes"]
    assert run.data.metrics["privacy.epsilon"] == pytest.approx(result.privacy.epsilon)
    assert run.data.tags["track"] == "app"
    assert result.budget is not None
    assert result.budget.runs == 1
    assert result.budget.epsilon == pytest.approx(result.privacy.epsilon)

    again = run_centralised(cfg)
    assert again.budget is not None
    assert again.budget.runs == 2
    assert again.budget.epsilon > result.budget.epsilon
    saved = json.loads((again.run_dir / "privacy-budget.json").read_text())
    assert saved["runs"] == 2
    assert saved["dataset"] == cfg.dataset_id

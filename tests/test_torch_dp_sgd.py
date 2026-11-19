"""DP-SGD in PyTorch (akili.train.torch_dp_sgd): the checks of docs/DP-SGD-REVIEW.md, measured
on the torch loop. Per-example and per-person clipping over every weight together (against
gradients computed one at a time), noise of standard deviation sigma x C on the sum divided by
the expected batch, Poisson batches, the steps the accountant is told, and noise from OS
entropy unless seeded. Needs PyTorch and Opacus: the [torch] extra (or [train], which has both).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("opacus")

from akili.categories import default_categories  # noqa: E402
from akili.config import (  # noqa: E402
    DPConfig,
    FeatureConfig,
    ModelConfig,
    SyntheticDataConfig,
    TrainingConfig,
)
from akili.features import encode_batch  # noqa: E402
from akili.model import build_categoriser  # noqa: E402
from akili.privacy.accounting import PrivacyUnit, dp_sgd_report  # noqa: E402
from akili.synthetic import generate  # noqa: E402
from akili.train.dp_sgd import group_by_user, user_rows  # noqa: E402
from akili.train.torch_dp_sgd import train_dp_sgd  # noqa: E402

pytestmark = pytest.mark.torch

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


def _accuracy(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(x.astype(np.int64))).numpy()
    return float((p.argmax(1) == y).mean())


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
    assert _accuracy(model, x, y) > 3 / N_CLASSES


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
    assert _accuracy(model, x, y) > 3 / N_CLASSES


def test_user_level_needs_users(data: tuple[np.ndarray, np.ndarray]) -> None:
    model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES)
    with pytest.raises(ValueError, match="each example's user"):
        train_dp_sgd(model, *data, TrainingConfig(), DPConfig(enabled=True))


def _weights(model: torch.nn.Module) -> list[np.ndarray]:
    return [p.detach().numpy().astype(np.float64).copy() for p in model.parameters()]


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
    before = _weights(model)
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
            weight_average=0.0,  # the step's own weights, to read the gradient off them
        ),
        users=users,
        noise_seed=noise_seed,
    )
    assert stats.steps == 1
    applied = [b - a for b, a in zip(before, _weights(model), strict=True)]
    return applied, stats.mean_batch_size


def _gradient(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """The gradient of the mean cross-entropy over ``x``, by ordinary backprop."""
    model.zero_grad()
    logits = model.logits(torch.from_numpy(x.astype(np.int64)))
    torch.nn.functional.cross_entropy(logits, torch.from_numpy(y.astype(np.int64))).backward()
    return [p.grad.numpy().astype(np.float64).copy() for p in model.parameters()]


def _clipped_sum(groups: list[list[np.ndarray]], clip: float) -> list[np.ndarray]:
    total = [np.zeros_like(g) for g in groups[0]]
    for grads in groups:
        norm = np.sqrt(sum(float((g**2).sum()) for g in grads))
        assert norm > 0.05  # so the small clip really clips every one
        for k, g in enumerate(grads):
            total[k] += g * min(1.0, clip / norm)
    return total


@pytest.mark.parametrize("clip", [0.05, 1.0])
def test_dp_sgd_clips_each_example_over_all_variables(
    data: tuple[np.ndarray, np.ndarray], clip: float
) -> None:
    """The applied gradient is the sum of per-example gradients, each clipped to ``clip`` in
    global L2 norm over every weight, divided by the expected batch size."""
    x, y = data[0][:16], data[1][:16]
    applied, _ = _one_step(x, y, clip=clip, noise_multiplier=0.0)
    reference = build_categoriser(FeatureConfig(), ModelConfig(dropout=0.0), N_CLASSES, seed=3)
    expected = _clipped_sum(
        [_gradient(reference, x[i : i + 1], y[i : i + 1]) for i in range(len(y))], clip
    )
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
    loss, each clipped to ``clip`` over every weight, divided by the expected people."""
    rows = np.flatnonzero(np.isin(people, np.unique(people)[:4]))[::4]  # 4 people, 10 rows each
    x, y, users = data[0][rows], data[1][rows], people[rows]
    applied, _ = _one_step(x, y, clip=clip, noise_multiplier=0.0, users=users)
    reference = build_categoriser(FeatureConfig(), ModelConfig(dropout=0.0), N_CLASSES, seed=3)
    expected = _clipped_sum(
        [_gradient(reference, x[users == p], y[users == p]) for p in np.unique(users)], clip
    )
    for got, want in zip(applied, expected, strict=True):
        np.testing.assert_allclose(got, want / 4, atol=1e-6 * max(clip, 1.0))


def test_dp_sgd_caps_each_persons_examples(people: np.ndarray) -> None:
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


def test_the_weight_average_changes_only_the_released_weights(
    data: tuple[np.ndarray, np.ndarray], people: np.ndarray
) -> None:
    """The average is post-processing: the steps, sampling and accounting are those of the run
    that releases its last step."""
    x, y = data
    runs = []
    for average in (0.0, 0.9):
        model = build_categoriser(FeatureConfig(), ModelConfig(), N_CLASSES, seed=1)
        training = TrainingConfig(epochs=2, learning_rate=0.5)
        dp = DPConfig(enabled=True, users_per_step=10, weight_average=average)
        runs.append((train_dp_sgd(model, x, y, training, dp, users=people, noise_seed=5), model))
    (plain, a), (averaged, b) = runs
    assert plain.steps == averaged.steps
    assert plain.sampling_rate == averaged.sampling_rate
    assert plain.mean_batch_size == averaged.mean_batch_size  # the same Poisson draws
    assert not np.allclose(a.hidden.weight.detach().numpy(), b.hidden.weight.detach().numpy())

"""DP-SGD in PyTorch: per-person (or per-example) gradients with ``torch.func``, clipped across
every layer together, Gaussian noise, Poisson sampling (decision 0023).

The algorithm of Abadi et al. 2016, per person as McMahan et al. 2018 (:mod:`akili.train.dp_sgd`
describes the privacy unit, sampling and seeding). Opacus clips per example, and the unit here is
one person, so the gradients come from ``torch.func``: ``vmap`` of ``grad`` over the sampled
people, each person's loss the mean over their examples. Each step:

1. every person (or example) joins the batch independently with probability q (Poisson);
2. each one's gradient over all trainable weights together is scaled to an L2 norm of at most
   ``dp.l2_norm_clip`` (C);
3. the clipped gradients are summed, Gaussian noise with standard deviation
   ``dp.noise_multiplier`` x C is added to every coordinate, and the sum is divided by the
   expected batch q x n, not the realised one; an empty batch still takes a noise-only step;
4. SGD with momentum 0.9 applies it (momentum only post-processes the noisy gradient), at a
   constant learning rate.

The released weights are an exponential moving average over the steps (``dp.weight_average``;
0 releases the last step's), which smooths out the noise and, being computed from the noisy
steps alone, costs no privacy.

Opacus's accountant turns q, the noise multiplier and the steps counted here into the budget
(:mod:`akili.privacy.accounting`). Dropout stays on while training, a fresh mask per person.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.func import functional_call, grad, vmap

from akili.config import DPConfig, TrainingConfig
from akili.privacy.accounting import PrivacyUnit
from akili.train.dp_sgd import (
    DPTrainingStats,
    group_by_user,
    noise_seed_or_entropy,
    steps_per_epoch,
    user_rows,
)


class _Logits(torch.nn.Module):  # type: ignore[misc,unused-ignore]
    """The categoriser's scores before the softmax, as a module ``functional_call`` runs."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Any) -> Any:
        return self.model.logits(x)


class _Steps:
    """DP-SGD steps for one model: per person, per example and noise only."""

    def __init__(
        self,
        model: Any,
        training: TrainingConfig,
        dp: DPConfig,
        expected: float,
        noise: torch.Generator,
    ) -> None:
        self.model = model
        self.clip = float(dp.l2_norm_clip)
        self.noise_std = self.clip * float(dp.noise_multiplier)
        self.expected = float(expected)
        self.noise = noise
        net = _Logits(model)
        self.params = dict(net.named_parameters())
        self.optimizer = torch.optim.SGD(
            self.params.values(), lr=training.learning_rate, momentum=0.9
        )
        self.average: Any = None
        ce = torch.nn.functional.cross_entropy

        def person_loss(params: dict[str, Any], x: Any, y: Any, m: Any) -> Any:
            logits = functional_call(net, params, (x,))
            losses = ce(logits, y, reduction="none")
            return (losses * m).sum() / torch.clamp(m.sum(), min=1.0)

        def example_loss(params: dict[str, Any], x: Any, y: Any) -> Any:
            logits = functional_call(net, params, (x[None],))
            return ce(logits, y[None])

        # Dropout draws a fresh mask for each person or example.
        self._per_person = vmap(grad(person_loss), in_dims=(None, 0, 0, 0), randomness="different")
        self._per_example = vmap(grad(example_loss), in_dims=(None, 0, 0), randomness="different")

    def _apply(self, grads: dict[str, Any] | None) -> None:
        """Clip each unit's gradient (the leading axis of ``grads``) over every weight
        together, sum, add the noise, divide by the expected batch and step."""
        if grads is not None:
            flat = torch.cat([g.reshape(g.shape[0], -1) for g in grads.values()], 1)
            norms = flat.norm(dim=1)
            scale = torch.clamp(self.clip / torch.clamp(norms, min=1e-12), max=1.0)
        for name, p in self.params.items():
            total = (
                torch.zeros_like(p)
                if grads is None
                else torch.tensordot(scale, grads[name], dims=([0], [0]))
            )
            noise = torch.normal(0.0, self.noise_std, p.shape, generator=self.noise)
            p.grad = (total + noise) / self.expected
        self.optimizer.step()
        if self.average is not None:
            self.average.update()

    def _loss(self, x: Any, y: Any, m: Any) -> float:
        self.model.eval()
        with torch.no_grad():
            losses = torch.nn.functional.cross_entropy(self.model.logits(x), y, reduction="none")
        self.model.train()
        return float((losses * m).sum() / torch.clamp(m.sum(), min=1.0))

    def person(self, x: Any, y: Any, m: Any) -> float:
        """One step on people ``[people, length, width]``, their labels and padding mask."""
        params = {k: v.detach() for k, v in self.params.items()}
        self._apply(self._per_person(params, x, y, m))
        width = x.shape[-1]
        return self._loss(x.reshape(-1, width), y.reshape(-1), m.reshape(-1))

    def example(self, x: Any, y: Any) -> float:
        params = {k: v.detach() for k, v in self.params.items()}
        self._apply(self._per_example(params, x, y))
        return self._loss(x, y, torch.ones(len(y)))

    def noise_only(self) -> None:
        self._apply(None)


def train_dp_sgd(
    model: Any,
    x: npt.NDArray[np.int32],
    y: npt.NDArray[np.int32],
    training: TrainingConfig,
    dp: DPConfig,
    *,
    users: npt.ArrayLike | None = None,
    noise_seed: int | None = None,
) -> DPTrainingStats:
    """Train ``model`` (a :class:`akili.model.Categoriser`) in place with DP-SGD.

    ``users`` gives each row's person and is required for ``dp.unit: user``. ``noise_seed``
    makes the noise and sampling reproducible; leave it None for anything but synthetic data.
    """
    if dp.unit is PrivacyUnit.USER:
        if users is None:
            raise ValueError("user-level DP-SGD needs each example's user")
        groups = group_by_user(users)
        population, per_step = len(groups), dp.users_per_step
    else:
        groups = []
        population, per_step = len(x), training.batch_size
    q = min(1.0, per_step / population)

    seed, seeded = noise_seed_or_entropy(noise_seed)
    rng = np.random.default_rng(seed)
    noise = torch.Generator().manual_seed(seed)
    from akili.train.torch_backend import WeightAverage

    steps = _Steps(model, training, dp, q * population, noise)
    planned = training.epochs * steps_per_epoch(population, per_step)
    steps.average = WeightAverage(model, dp.weight_average)
    xt = torch.from_numpy(np.asarray(x, dtype=np.int64))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))

    model.train()
    sizes: list[int] = []
    last_loss = float("nan")
    for _ in range(planned):
        chosen = np.flatnonzero(rng.random(population) < q)
        sizes.append(len(chosen))
        if len(chosen) == 0:
            steps.noise_only()
        elif dp.unit is PrivacyUnit.USER:
            idx, mask = user_rows(groups, chosen, dp.max_examples_per_user, rng)
            index = torch.from_numpy(idx)
            last_loss = steps.person(xt[index], yt[index], torch.from_numpy(mask))
        else:
            index = torch.from_numpy(chosen)
            last_loss = steps.example(xt[index], yt[index])
    steps.average.load()
    model.eval()
    return DPTrainingStats(
        unit=dp.unit,
        population=population,
        steps=len(sizes),
        sampling_rate=q,
        mean_batch_size=float(np.mean(sizes)),
        final_loss=last_loss,
        noise_seeded=seeded,
    )

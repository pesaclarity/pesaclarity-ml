"""The categoriser as a PyTorch module, and how it is presented to rayo.

The network the Keras categoriser had (:mod:`akili.keras_model`), layer for layer and under the
same names, so the weights keep their names (``token_embedding.table``, ``hidden.kernel``), the
federated protocol its arrays and the release its figures:

- ``token_embedding``: the payee's hashed tokens (``hash_buckets`` x ``token_embedding_dim``),
  averaged over the tokens that are not padding (id 0);
- ``context_0`` .. ``context_4``: 4-wide embeddings of the amount bucket, direction, hour bucket,
  weekday and month phase (:data:`akili.features.CONTEXT_LAYOUT`);
- ``hidden``: dense, ReLU, on the five context embeddings after the mean (dropout after it
  while training);
- ``category``: dense, softmax over the category set's labels.

About 70k parameters with the default config. The input is the app's one int32 row per
transaction, ``features`` ``[batch, max_tokens + 5]`` (:mod:`akili.features`); the exported
graph takes the same row (rayo slices it), so the app feeds rayo what it builds today.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from rayo_convert import Options
from torch import nn

from akili.config import FeatureConfig, ModelConfig
from akili.features import CONTEXT_CARDINALITIES, feature_width

CONTEXT_EMBEDDING_DIM = 4
MODEL_NAME = "categoriser"
WEIGHTS_FILE = "categoriser.weights"
ONNX_FILE = "categoriser.onnx"
INPUT = "features"
OUTPUT = "category"

# The layers in the order they run, as the weights file lists them (docs/WEIGHTS.md).
LAYERS: list[dict[str, Any]] = [
    {"name": "token_embedding", "kind": "embedding", "pooling": "masked_mean"},
    *({"name": f"context_{i}", "kind": "embedding"} for i in range(len(CONTEXT_CARDINALITIES))),
    {"name": "hidden", "kind": "dense", "activation": "relu"},
    {"name": "category", "kind": "dense", "activation": "softmax"},
]

# int8 weights quantised by rayo-convert from the float graph (decision 0023); tensors with
# fewer than 1,024 values (the context tables, a small ``category``) stay float32, as TFLite's
# converter kept them.
RAYO_OPTIONS = Options(name=MODEL_NAME, outputs=[OUTPUT], weights="int8")


def required_tensors() -> list[str]:
    return [
        f"{layer['name']}.{part}"
        for layer in LAYERS
        for part in (("table",) if layer["kind"] == "embedding" else ("kernel", "bias"))
    ]


class Categoriser(nn.Module):  # type: ignore[misc,unused-ignore]
    """The categoriser. ``forward`` gives the softmax (what the app runs); :meth:`logits` the
    scores before it (what training uses)."""

    def __init__(self, features: FeatureConfig, model: ModelConfig, n_classes: int) -> None:
        super().__init__()
        self.max_tokens = features.max_tokens
        self.token_embedding = nn.Embedding(features.hash_buckets, model.token_embedding_dim)
        for i, cardinality in enumerate(CONTEXT_CARDINALITIES):
            setattr(self, f"context_{i}", nn.Embedding(cardinality, CONTEXT_EMBEDDING_DIM))
        width = model.token_embedding_dim + CONTEXT_EMBEDDING_DIM * len(CONTEXT_CARDINALITIES)
        self.hidden = nn.Linear(width, model.hidden_units)
        self.dropout = nn.Dropout(model.dropout)
        self.category = nn.Linear(model.hidden_units, n_classes)
        self.reset_parameters()

    def contexts(self) -> list[nn.Embedding]:
        return [getattr(self, f"context_{i}") for i in range(len(CONTEXT_CARDINALITIES))]

    def reset_parameters(self) -> None:
        """Keras' initialisers, as the Keras categoriser had them: embeddings uniform in
        +-0.05, kernels Glorot uniform, biases zero."""
        with torch.no_grad():
            for emb in (self.token_embedding, *self.contexts()):
                nn.init.uniform_(emb.weight, -0.05, 0.05)
            for layer in (self.hidden, self.category):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def body(self, features: Any) -> Any:
        """What ``hidden`` reads: the masked mean of the token embeddings, then the five
        context embeddings, ``[batch, token_embedding_dim + 20]``."""
        n = self.max_tokens
        tokens = features[:, :n]
        mask = (tokens > 0).float().unsqueeze(-1)
        pooled = (self.token_embedding(tokens) * mask).sum(1) / torch.clamp(mask.sum(1), min=1.0)
        # Each column as a slice of one, not an index, so rayo reads it as a row lookup.
        parts = [pooled] + [
            emb(features[:, n + i : n + i + 1]).flatten(1) for i, emb in enumerate(self.contexts())
        ]
        return torch.cat(parts, -1)

    def logits(self, features: Any) -> Any:
        return self.category(self.dropout(torch.relu(self.hidden(self.body(features)))))

    def forward(self, features: Any) -> Any:
        return torch.softmax(self.logits(features), -1)


def build_categoriser(
    features: FeatureConfig, model: ModelConfig, n_classes: int, seed: int = 0
) -> Categoriser:
    """A categoriser with Keras' initialisers, seeded by ``seed``."""
    torch.manual_seed(seed)
    return Categoriser(features, model, n_classes)


def parameter_count(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# The federated protocol's layout: Keras' get_weights() order, a dense kernel [in, out].
def keras_layout(module: Categoriser) -> list[npt.NDArray[np.float32]]:
    """The weights as the Keras categoriser's ``get_weights()`` gives them: the six embedding
    tables, then each dense layer's kernel ``[in, out]`` and bias."""
    out = [
        _numpy(module.token_embedding.weight),
        *(_numpy(emb.weight) for emb in module.contexts()),
    ]
    for layer in (module.hidden, module.category):
        out += [_numpy(layer.weight).T.copy(), _numpy(layer.bias)]
    return out


def set_keras_layout(module: Categoriser, weights: list[npt.NDArray[np.float32]]) -> None:
    """Load weights in :func:`keras_layout`'s order and layout."""
    tables = [module.token_embedding, *module.contexts()]
    targets = [(emb.weight, False) for emb in tables]
    for layer in (module.hidden, module.category):
        targets += [(layer.weight, True), (layer.bias, False)]
    if len(weights) != len(targets):
        raise ValueError(f"{len(weights)} arrays for the categoriser's {len(targets)}")
    with torch.no_grad():
        for (param, transposed), w in zip(targets, weights, strict=True):
            value = np.asarray(w, dtype=np.float32)
            value = value.T if transposed else value
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"shape {value.shape} for a weight of {tuple(param.shape)}")
            param.copy_(torch.from_numpy(np.ascontiguousarray(value)))


def _numpy(t: Any) -> npt.NDArray[np.float32]:
    return np.asarray(t.detach().cpu().numpy(), dtype=np.float32)


def example_input(features: FeatureConfig) -> dict[str, Any]:
    """One row of zeros, for the export."""
    return {INPUT: torch.zeros((1, feature_width(features)), dtype=torch.int64)}

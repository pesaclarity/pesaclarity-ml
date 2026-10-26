"""The categoriser network (needs the [train] extra).

One int32 input of shape ``[batch, max_tokens + 5]`` (see :mod:`akili.features`) and a softmax
over the category set's targets. About 70k parameters with the default config, so the TFLite
file stays well under the size budget for 2 GB RAM phones.
"""

from __future__ import annotations

from typing import Any

from akili._optional import require
from akili.config import FeatureConfig, ModelConfig
from akili.features import CONTEXT_CARDINALITIES, feature_width

CONTEXT_EMBEDDING_DIM = 4


def build_categoriser(
    features: FeatureConfig, model: ModelConfig, n_classes: int, seed: int = 0
) -> Any:
    """Build an uncompiled Keras model."""
    keras = require("keras", "train")
    ops = keras.ops
    keras.utils.set_random_seed(seed)

    inputs = keras.Input(shape=(feature_width(features),), dtype="int32", name="features")
    tokens = inputs[:, : features.max_tokens]
    token_emb = keras.layers.Embedding(
        features.hash_buckets, model.token_embedding_dim, name="token_embedding"
    )(tokens)
    mask = ops.expand_dims(ops.cast(ops.greater(tokens, 0), "float32"), -1)
    pooled = ops.sum(token_emb * mask, axis=1) / ops.maximum(ops.sum(mask, axis=1), 1.0)

    parts = [pooled]
    for i, cardinality in enumerate(CONTEXT_CARDINALITIES):
        column = inputs[:, features.max_tokens + i]
        emb = keras.layers.Embedding(cardinality, CONTEXT_EMBEDDING_DIM, name=f"context_{i}")
        parts.append(emb(column))

    x = keras.layers.Concatenate()(parts)
    x = keras.layers.Dense(model.hidden_units, activation="relu", name="hidden")(x)
    if model.dropout:
        x = keras.layers.Dropout(model.dropout, seed=seed)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax", name="category")(x)
    return keras.Model(inputs, outputs, name="akili_categoriser")

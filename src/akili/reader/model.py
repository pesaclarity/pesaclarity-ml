"""The message reader as a PyTorch module, and how it is presented to rayo (needs [torch]).

The network the Keras reader had (``akili.reader.keras_train``), layer for layer and under the same
names, so the weights file keeps its tensors (``conv_1.kernel``, ``token_embedding.table``),
their layout and the ``layers`` list:

- ``token_embedding`` (4096 x 32) and ``shape_embedding`` (12 x 8), concatenated;
- ``conv_1`` and ``conv_2``: 1-D convolutions of 64 filters, width 5, "same" padding, ReLU;
  ``conv_2``'s output is added to its input;
- ``conv_3``: width 3, dilated by 4 (about a dozen tokens either side), ReLU, added likewise;
- ``fields``: a softmax per token over :data:`~akili.reader.synthetic.FIELD_LABELS`;
- ``kind_hidden`` (32, ReLU) and ``kind`` (softmax over the kinds) on the tokens' masked max.

The app's input is one int32 vector per message, ``[batch, 192]``: token ids, then shape ids
(docs/READER.md). The module takes its two halves, ``tokens`` and ``shapes`` (:func:`split`),
so the exported graph has the two inputs rayo runs, as the Ask reader's has.
"""

from __future__ import annotations

from typing import Any

from rayo_convert import Options

from akili._optional import require
from akili.reader.synthetic import FIELD_LABELS, KIND_LABELS
from akili.reader.tokens import HASH_BUCKETS, MAX_TOKENS, N_SHAPES

require("torch", "torch")  # or say which extra installs it

import torch  # noqa: E402
from torch import nn  # noqa: E402

MODEL_NAME = "reader"
WEIGHTS_FILE = "reader.weights"
ONNX_FILE = "reader.onnx"

# The layers in the order they run, as the weights file lists them (docs/WEIGHTS.md).
LAYERS: list[dict[str, Any]] = [
    {"name": "token_embedding", "kind": "embedding"},
    {"name": "shape_embedding", "kind": "embedding"},
    {"name": "conv_1", "kind": "conv1d", "size": 5, "dilation": 1, "activation": "relu"},
    {"name": "conv_2", "kind": "conv1d", "size": 5, "dilation": 1, "activation": "relu"},
    {"name": "conv_3", "kind": "conv1d", "size": 3, "dilation": 4, "activation": "relu"},
    {"name": "fields", "kind": "dense", "activation": "softmax"},
    {"name": "kind_hidden", "kind": "dense", "activation": "relu"},
    {"name": "kind", "kind": "dense", "activation": "softmax"},
]

# The fields before the kind; int8 weights quantised by rayo-convert from the float graph
# (decision 0023).
RAYO_OPTIONS = Options(name=MODEL_NAME, outputs=["fields", "kind"], weights="int8")
INPUTS = ("tokens", "shapes")
OUTPUTS = ("fields", "kind")


def required_tensors() -> list[str]:
    return [
        f"{layer['name']}.{part}"
        for layer in LAYERS
        for part in (("table",) if layer["kind"] == "embedding" else ("kernel", "bias"))
    ]


def split(message: Any) -> tuple[Any, Any]:
    """The token ids and the shape ids of inputs ``[..., 192]`` (NumPy or torch)."""
    return message[..., :MAX_TOKENS], message[..., MAX_TOKENS:]


class Reader(nn.Module):  # type: ignore[misc,unused-ignore]
    """The message reader. ``forward`` gives the two softmaxes (what the app runs);
    :meth:`logits` the scores before them (what training uses)."""

    def __init__(
        self,
        token_dim: int = 32,
        shape_dim: int = 8,
        filters: int = 64,
        kind_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(HASH_BUCKETS, token_dim)
        self.shape_embedding = nn.Embedding(N_SHAPES, shape_dim)
        width = token_dim + shape_dim
        self.conv_1 = nn.Conv1d(width, filters, 5, padding=2)
        self.conv_2 = nn.Conv1d(filters, filters, 5, padding=2)
        self.conv_3 = nn.Conv1d(filters, filters, 3, padding=4, dilation=4)
        self.fields = nn.Linear(filters, len(FIELD_LABELS))
        self.kind_hidden = nn.Linear(filters, kind_hidden)
        self.kind = nn.Linear(kind_hidden, len(KIND_LABELS))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Keras' initialisers, as the Keras reader had them: embeddings uniform in
        +-0.05, kernels Glorot uniform, biases zero."""
        with torch.no_grad():
            for emb in (self.token_embedding, self.shape_embedding):
                nn.init.uniform_(emb.weight, -0.05, 0.05)
            for layer in (
                self.conv_1,
                self.conv_2,
                self.conv_3,
                self.fields,
                self.kind_hidden,
                self.kind,
            ):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _conv(self, conv: Any, x: Any) -> Any:
        return torch.relu(conv(x.transpose(1, 2))).transpose(1, 2)

    def logits(self, tokens: Any, shapes: Any) -> tuple[Any, Any]:
        x = torch.cat([self.token_embedding(tokens), self.shape_embedding(shapes)], -1)
        x = self._conv(self.conv_1, x)
        x = x + self._conv(self.conv_2, x)
        x = x + self._conv(self.conv_3, x)
        fields = self.fields(x)
        mask = (shapes > 0).float().unsqueeze(-1)
        pooled = (x * mask - (1.0 - mask) * 1e4).max(1).values
        kind = self.kind(torch.relu(self.kind_hidden(pooled)))
        return fields, kind

    def forward(self, tokens: Any, shapes: Any) -> tuple[Any, Any]:
        fields, kind = self.logits(tokens, shapes)
        return torch.softmax(fields, -1), torch.softmax(kind, -1)


def parameter_count(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))

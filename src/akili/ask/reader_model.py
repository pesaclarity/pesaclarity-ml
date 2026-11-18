"""The Ask reader as a PyTorch module, and how it is presented to rayo (needs [torch]).

The network the Keras Ask reader had (:mod:`akili.ask.keras_reader`), layer for layer and under
the same names, so the weights file keeps its tensors (``conv_1.kernel``,
``token_embedding.table``), their layout and the ``layers`` list the app's Dart reader checks:

- ``token_embedding`` (4096 x 24) and ``shape_embedding`` (12 x 8), concatenated, with spatial
  dropout (whole channels) in training;
- ``conv_1`` (96 filters, width 3), ``conv_2`` (width 5, added to its input) and ``conv_3``
  (width 3, dilated by 3, added likewise), all "same" padding and ReLU;
- the slot head: ``slot_hidden`` (64, ReLU) on each token's context and embedding,
  ``slot_context`` (width 5) and ``slot_span`` (width 3, dilated by 2), each added to its
  input, then ``slots``, a softmax per token over :data:`~akili.ask.schema.SLOT_LABELS`;
- the intent head on the tokens' masked max and mean and the mean of their embeddings:
  dropout in training, ``intent_hidden`` (64, ReLU), ``intent`` (softmax over the intents).

The app's input is one int32 vector per question, ``[batch, 192]``: token ids, then shape ids
(docs/ASK.md). The module takes its two halves, ``tokens`` and ``shapes`` (:func:`split`), so
the exported graph has the two inputs rayo runs (the app feeds them so already), with no
``Slice`` for rayo-convert to split.
"""

from __future__ import annotations

from typing import Any

from rayo_convert import Options

from akili._optional import require
from akili.ask.reader_weights import LAYERS, MODEL_NAME, required_tensors
from akili.ask.schema import ASK_INTENTS, SLOT_LABELS
from akili.reader.tokens import HASH_BUCKETS, MAX_TOKENS, N_SHAPES

require("torch", "torch")  # or say which extra installs it

import torch  # noqa: E402
from torch import nn  # noqa: E402

ONNX_FILE = "ask-reader.onnx"

# The slots before the intent, as the app's rayo session reads them; int8 weights quantised by
# rayo-convert from the float graph (decision 0023).
RAYO_OPTIONS = Options(name=MODEL_NAME, outputs=["slots", "intent"], weights="int8")
INPUTS = ("tokens", "shapes")
OUTPUTS = ("slots", "intent")

__all__ = [
    "INPUTS",
    "LAYERS",
    "ONNX_FILE",
    "OUTPUTS",
    "RAYO_OPTIONS",
    "AskReader",
    "parameter_count",
    "required_tensors",
    "split",
]


def split(question: Any) -> tuple[Any, Any]:
    """The token ids and the shape ids of inputs ``[..., 192]`` (NumPy or torch)."""
    return question[..., :MAX_TOKENS], question[..., MAX_TOKENS:]


class AskReader(nn.Module):  # type: ignore[misc,unused-ignore]
    """The Ask reader. ``forward`` gives the two softmaxes, slots then intent (what the app
    runs); :meth:`logits` the scores before them (what training uses)."""

    def __init__(
        self,
        *,
        token_dim: int = 24,
        shape_dim: int = 8,
        filters: int = 96,
        slot_filters: int = 64,
        dropout: float = 0.1,
        intent_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(HASH_BUCKETS, token_dim)
        self.shape_embedding = nn.Embedding(N_SHAPES, shape_dim)
        width = token_dim + shape_dim
        self.conv_1 = nn.Conv1d(width, filters, 3, padding=1)
        self.conv_2 = nn.Conv1d(filters, filters, 5, padding=2)
        self.conv_3 = nn.Conv1d(filters, filters, 3, padding=3, dilation=3)
        self.slot_hidden = nn.Linear(filters + width, slot_filters)
        self.slot_context = nn.Conv1d(slot_filters, slot_filters, 5, padding=2)
        self.slot_span = nn.Conv1d(slot_filters, slot_filters, 3, padding=2, dilation=2)
        self.slots = nn.Linear(slot_filters, len(SLOT_LABELS))
        self.intent_hidden = nn.Linear(2 * filters + width, intent_hidden)
        self.intent = nn.Linear(intent_hidden, len(ASK_INTENTS))
        # Keras' SpatialDropout1D (whole channels of a question) and Dropout; identity in eval,
        # so not in the exported graph.
        self.spatial_dropout = nn.Dropout1d(dropout)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Keras' initialisers, as the Keras reader had them: embeddings uniform in +-0.05,
        kernels Glorot uniform, biases zero."""
        with torch.no_grad():
            for emb in (self.token_embedding, self.shape_embedding):
                nn.init.uniform_(emb.weight, -0.05, 0.05)
            for layer in (
                self.conv_1,
                self.conv_2,
                self.conv_3,
                self.slot_hidden,
                self.slot_context,
                self.slot_span,
                self.slots,
                self.intent_hidden,
                self.intent,
            ):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    @staticmethod
    def _conv(conv: Any, x: Any) -> Any:
        return torch.relu(conv(x.transpose(1, 2))).transpose(1, 2)

    def logits(self, tokens: Any, shapes: Any) -> tuple[Any, Any]:
        emb = torch.cat([self.token_embedding(tokens), self.shape_embedding(shapes)], -1)
        emb = self.spatial_dropout(emb.transpose(1, 2)).transpose(1, 2)
        x = self._conv(self.conv_1, emb)
        x = x + self._conv(self.conv_2, x)
        x = x + self._conv(self.conv_3, x)
        t = torch.relu(self.slot_hidden(torch.cat([x, emb], -1)))
        t = t + self._conv(self.slot_context, t)
        t = t + self._conv(self.slot_span, t)
        slots = self.slots(t)
        mask = (shapes > 0).float().unsqueeze(-1)
        n = torch.clamp(mask.sum(1), min=1.0)
        pooled = torch.cat(
            [
                (x * mask - (1.0 - mask) * 1e4).max(1).values,
                (x * mask).sum(1) / n,
                (emb * mask).sum(1) / n,
            ],
            -1,
        )
        h = torch.relu(self.intent_hidden(self.dropout(pooled)))
        return slots, self.intent(h)

    def forward(self, tokens: Any, shapes: Any) -> tuple[Any, Any]:
        slots, intent = self.logits(tokens, shapes)
        return torch.softmax(slots, -1), torch.softmax(intent, -1)


def parameter_count(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))

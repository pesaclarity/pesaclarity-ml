"""The Ask reader in PyTorch (akili.ask.reader_model, the torch path of akili.ask.reader): the
Keras layer names and the weights file's layout kept, the int8 ``ask-reader.weights`` and its
manifest, the training shortcut, determinism, and the converted file against torch and the
Dart reader's reference on rayo's C library when ``RAYO_DIR`` is a rayo checkout.

Small runs (few questions, two epochs); the released figures are in docs/ASK.md. Needs the
[torch] extra.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from akili.ask import names, questions  # noqa: E402
from akili.ask import reader as ar  # noqa: E402
from akili.ask import reader_model as rm  # noqa: E402
from akili.ask.reader_weights import WEIGHTS_FILE, forward, run_weights  # noqa: E402
from akili.ask.schema import ASK_INTENTS, SLOT_LABELS  # noqa: E402
from akili.export.weights import read_weights  # noqa: E402
from akili.reader.tokens import MAX_TOKENS, encode  # noqa: E402

pytestmark = pytest.mark.torch

# Held to the released reader's tolerances (AskReaderConfig).
SMALL = ar.AskReaderConfig(questions=1500, epochs=2)

# The tensors of the Keras reader's ask-reader.weights (pesaclarity-ml#51, which the app ships):
# dtype, shape and the axis of the scales (None: one scale; "-": float32).
KERAS_LAYOUT = {
    "token_embedding.table": ("int8", (4096, 24), None),
    "shape_embedding.table": ("float32", (12, 8), "-"),
    "conv_1.kernel": ("int8", (96, 1, 3, 32), 0),
    "conv_2.kernel": ("int8", (96, 1, 5, 96), 0),
    "conv_3.kernel": ("int8", (96, 1, 3, 96), 0),
    "slot_hidden.kernel": ("int8", (64, 128), 0),
    "slot_context.kernel": ("int8", (64, 1, 5, 64), 0),
    "slot_span.kernel": ("int8", (64, 1, 3, 64), 0),
    "slots.kernel": ("float32", (7, 64), "-"),
    "intent_hidden.kernel": ("int8", (64, 224), 0),
    "intent.kernel": ("int8", (30, 64), 0),
}


def _rayo_dir() -> str | None:
    d = os.environ.get("RAYO_DIR")
    return d if d and (Path(d) / "src" / "model.c").exists() else None


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    out = tmp_path_factory.mktemp("ask-reader")
    report = ar.train_ask_reader(SMALL, out, evaluate=False, rayo_dir=_rayo_dir() or "")
    return out, report


def test_layers_keep_the_keras_names() -> None:
    m = rm.AskReader()
    layers = [
        n for n, c in m.named_children() if not isinstance(c, torch.nn.Dropout | torch.nn.Dropout1d)
    ]
    assert layers == [layer["name"] for layer in rm.LAYERS]
    assert rm.parameter_count(m) == 239_589  # the Keras reader's count
    for layer in rm.LAYERS:
        if layer["kind"] == "conv1d":
            conv = getattr(m, layer["name"])
            assert conv.kernel_size == (layer["size"],)
            assert conv.dilation == (layer["dilation"],)


def test_training_on_the_longest_question_is_exact() -> None:
    """fit runs each batch on its longest question plus RECEPTIVE_FIELD padding tokens: the
    same outputs on the questions' tokens as all 96 positions, and the same gradients."""
    torch.manual_seed(0)
    m = rm.AskReader()
    x, y, k = questions.to_arrays(questions.generate(64, seed=3, pool=names.TRAIN))
    tokens, shapes = (torch.from_numpy(a.astype(np.int64)) for a in rm.split(x))
    n = int(ar._lengths(x, y).max()) + ar.RECEPTIVE_FIELD
    assert n < MAX_TOKENS
    labels = torch.from_numpy(np.maximum(y, 0).astype(np.int64))
    weight = torch.from_numpy((y >= 0).astype(np.float32))
    m.eval()  # no dropout, so the two runs compare

    def loss(upto: int) -> torch.Tensor:
        slots, intent = m.logits(tokens[:, :upto], shapes[:, :upto])
        ce = torch.nn.functional.cross_entropy
        per = ce(
            slots.reshape(-1, len(SLOT_LABELS)), labels[:, :upto].reshape(-1), reduction="none"
        )
        total = (per * weight[:, :upto].reshape(-1)).sum() / (len(x) * MAX_TOKENS)
        return total + ce(intent, torch.from_numpy(k.astype(np.int64)))

    grads = []
    for upto in (MAX_TOKENS, n):
        m.zero_grad()
        loss(upto).backward()
        grads.append([p.grad.clone() for p in m.parameters()])
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    with torch.no_grad():
        full = m.logits(tokens, shapes)
        cut = m.logits(tokens[:, :n], shapes[:, :n])
    mask = torch.from_numpy(y[:, :n] >= 0)
    torch.testing.assert_close(full[0][:, :n][mask], cut[0][mask], rtol=0, atol=1e-5)
    torch.testing.assert_close(full[1], cut[1], rtol=0, atol=1e-5)


def test_weights_and_manifest(trained: tuple[Path, dict[str, object]]) -> None:
    out, report = trained
    data = (out / WEIGHTS_FILE).read_bytes()
    onnx = (out / rm.ONNX_FILE).read_bytes()
    manifest = json.loads((out / ar.MANIFEST_FILE).read_text())
    assert manifest["weights_file"] == WEIGHTS_FILE
    assert manifest["weights_sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["weights_size_bytes"] == len(data) == report["weights_size_bytes"]
    assert manifest["weights_format_version"] == 1
    assert manifest["source"]["file"] == rm.ONNX_FILE
    assert manifest["source"]["sha256"] == hashlib.sha256(onnx).hexdigest()
    assert manifest["outputs"]["intent"]["labels"] == list(ASK_INTENTS)
    assert manifest["outputs"]["slots"]["labels"] == list(SLOT_LABELS)
    assert manifest["input"]["shape"] == [-1, 192]
    assert manifest["ask_schema_version"] == 1
    assert "model_file" not in manifest  # no TFLite file for this model any more
    assert len(data) < 400_000

    w = read_weights(data)
    assert w.model == "ask-reader"
    assert w.source_file == rm.ONNX_FILE
    assert w.layers == rm.LAYERS
    t = w.tensors
    assert set(rm.required_tensors()) <= set(t)
    assert sum(t[n].value.size for n in rm.required_tensors()) == report["parameters"]
    for name, (dtype, shape, axis) in KERAS_LAYOUT.items():
        assert t[name].value.dtype == np.dtype(dtype), name
        assert t[name].value.shape == shape, name
        if axis != "-":
            assert t[name].scales is not None, name
            assert t[name].axis == axis, name
    graph = w.graph or {}
    assert [i["name"] for i in graph["inputs"]] == ["tokens", "shapes"]
    assert [o["name"] for o in graph["outputs"]] == ["slots", "intent"]
    assert graph["symbols"] == {"T": MAX_TOKENS}

    parity = report["parity"]
    assert isinstance(parity, dict)
    assert parity["within"]
    for name in ("intent", "slots"):
        assert parity[name]["labels_compared"] == parity[name]["labels_compared_same"]
    assert parity["slots"]["labels_compared"] > 0  # two epochs leave few intents sure
    assert parity["dart_reference"]["max_abs"] < 1e-4
    assert {"hard_held_out", "app_questions", "misses", "conformance_questions"} <= set(report)


def test_training_is_deterministic() -> None:
    qs = questions.generate(300, seed=4, pool=names.TRAIN)
    cfg = ar.AskReaderConfig(questions=300, epochs=1)
    a = ar.export_module(ar.fit(cfg, qs))
    b = ar.export_module(ar.fit(cfg, qs))
    assert a.onnx == b.onnx
    assert a.sha256 == b.sha256


def test_input_is_the_question_split() -> None:
    row = np.asarray(encode("how much did I pay Naivas last month"))
    tokens, shapes = rm.split(row)
    assert list(tokens) == list(row[:MAX_TOKENS])
    assert list(shapes) == list(row[MAX_TOKENS:])


def test_on_rayo(trained: tuple[Path, dict[str, object]]) -> None:
    """The released file as the app runs it: rayo-convert's reference of the same int8 graph to
    about 1e-5, the NumPy forward pass the Dart reader copies within the app's 1e-4, and
    torch's labels."""
    d = _rayo_dir()
    if not d:
        pytest.skip("RAYO_DIR: no rayo checkout to build rayo's C library from")
    out, _ = trained
    from akili.export.onnx import RayoRunner, ReferenceRunner, parity

    data = (out / WEIGHTS_FILE).read_bytes()
    onnx = (out / rm.ONNX_FILE).read_bytes()
    x, y, _ = questions.to_arrays(questions.generate(30, seed=9, pool=names.TRAIN))
    rayo = ar.run_converted(RayoRunner(data, d), x)
    ref = ar.run_converted(ReferenceRunner(onnx, rm.RAYO_OPTIONS), x)
    for a, b in zip(rayo, ref, strict=True):
        assert np.abs(a - b).max() <= 1e-4
    w = read_weights(data)
    for row, labels, intent, slots in zip(x, y, *rayo, strict=True):
        di, ds = forward(w, row)
        assert np.abs(di - intent).max() < 1e-4
        assert np.abs(ds - slots)[labels >= 0].max() < 1e-4
    dart = run_weights(data)(x[:5])
    np.testing.assert_allclose(dart[0], rayo[0][:5], atol=1e-4)
    model = ar.fit(SMALL, ar.training_questions(SMALL))
    feeds = [dict(zip(rm.INPUTS, rm.split(row), strict=True)) for row in x.astype(np.int64)]
    p = parity(
        model,
        RayoRunner(data, d),
        feeds,
        rm.OUTPUTS,
        labels=rm.OUTPUTS,
        bind={"T": MAX_TOKENS},
        margin=SMALL.parity_label_margin,
    )
    assert p.rows == p.same > 0
    assert p.within(SMALL.parity_tolerance_slots), p.max_abs

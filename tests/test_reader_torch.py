"""The message reader in PyTorch (akili.reader.model, akili.reader.train): the Keras layer
names and layout kept, the int8 ``reader.weights`` and its manifest, determinism, and the
converted file against torch on rayo's C library when ``RAYO_DIR`` is a rayo checkout.

Small runs (few messages, two epochs); the released figures are in docs/READER.md. Needs the
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

from akili.export.weights import read_weights  # noqa: E402
from akili.reader import model as rm  # noqa: E402
from akili.reader import train as rt  # noqa: E402
from akili.reader.synthetic import FIELD_LABELS, KIND_LABELS, generate, to_arrays  # noqa: E402
from akili.reader.tokens import MAX_TOKENS, encode  # noqa: E402

pytestmark = pytest.mark.torch

# Two epochs leave the heads less sure than a trained model's; the released reader is held to
# ReaderConfig's tolerance.
SMALL = rt.ReaderConfig(
    messages=1200, epochs=2, parity_tolerance_fields=6e-2, parity_tolerance_kind=6e-2
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    out = tmp_path_factory.mktemp("reader")
    report = rt.train_reader(SMALL, out, evaluate=False, rayo_dir="")
    return out, report


def test_layers_keep_the_keras_names() -> None:
    m = rm.Reader()
    children = [name for name, _ in m.named_children()]
    assert children == [layer["name"] for layer in rm.LAYERS]
    assert rm.parameter_count(m) == 179_627
    convs = {layer["name"]: layer for layer in rm.LAYERS if layer["kind"] == "conv1d"}
    for name, layer in convs.items():
        conv = getattr(m, name)
        assert conv.kernel_size == (layer["size"],)
        assert conv.dilation == (layer["dilation"],)


def test_weights_and_manifest(trained: tuple[Path, dict[str, object]]) -> None:
    out, report = trained
    data = (out / rm.WEIGHTS_FILE).read_bytes()
    onnx = (out / rm.ONNX_FILE).read_bytes()
    manifest = json.loads((out / rt.MANIFEST_FILE).read_text())
    assert manifest["weights_file"] == rm.WEIGHTS_FILE
    assert manifest["weights_sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["weights_size_bytes"] == len(data) == report["weights_size_bytes"]
    assert manifest["source"]["sha256"] == hashlib.sha256(onnx).hexdigest()
    assert manifest["outputs"]["fields"]["labels"] == FIELD_LABELS
    assert manifest["outputs"]["kind"]["labels"] == KIND_LABELS
    assert manifest["input"] == {"name": "message", "dtype": "int32", "shape": [-1, 192]}
    assert "model_file" not in manifest  # no TFLite file for this model any more
    assert len(data) < 512_000

    w = read_weights(data)
    assert w.model == "reader"
    assert w.source_file == rm.ONNX_FILE
    assert w.layers == rm.LAYERS
    t = w.tensors
    assert set(rm.required_tensors()) <= set(t)
    for name in ("token_embedding.table", "conv_1.kernel", "conv_2.kernel", "conv_3.kernel"):
        assert t[name].value.dtype == np.int8, name
    assert t["conv_1.kernel"].value.shape == (64, 1, 5, 40)  # [out, 1, size, in]
    assert t["conv_3.kernel"].value.shape == (64, 1, 3, 64)
    assert t["fields.kernel"].value.shape == (len(FIELD_LABELS), 64)  # [out, in]
    graph = w.graph or {}
    assert [i["name"] for i in graph["inputs"]] == ["tokens", "shapes"]
    assert [o["name"] for o in graph["outputs"]] == ["fields", "kind"]

    parity = report["parity"]
    assert isinstance(parity, dict)
    assert parity["within"]
    for name in ("fields", "kind"):
        assert parity[name]["labels_compared"] == parity[name]["labels_compared_same"] > 0


def test_training_is_deterministic() -> None:
    messages = generate(300, seed=4)
    cfg = rt.ReaderConfig(messages=300, epochs=1)
    a = rt.export_module(rt.fit(cfg, messages))
    b = rt.export_module(rt.fit(cfg, messages))
    assert a.onnx == b.onnx
    assert a.sha256 == b.sha256


def test_input_is_the_message_split() -> None:
    row = np.asarray(encode("Paid KES 50 to DEMO CAFE on 29/10/26"))
    tokens, shapes = rm.split(row)
    assert list(tokens) == list(row[:MAX_TOKENS])
    assert list(shapes) == list(row[MAX_TOKENS:])


def test_on_rayo(trained: tuple[Path, dict[str, object]]) -> None:
    """The released file as the app's rayo runs it: torch's labels, and rayo-convert's
    reference of the same int8 graph to about 1e-5."""
    d = os.environ.get("RAYO_DIR")
    if not d or not (Path(d) / "src" / "model.c").exists():
        pytest.skip("RAYO_DIR: no rayo checkout to build rayo's C library from")
    out, _ = trained
    from akili.export.onnx import RayoRunner, ReferenceRunner, parity

    data = (out / rm.WEIGHTS_FILE).read_bytes()
    onnx = (out / rm.ONNX_FILE).read_bytes()
    x, _, _ = to_arrays(generate(40, seed=9))
    rayo = rt.run_converted(RayoRunner(data, d), x)
    ref = rt.run_converted(ReferenceRunner(onnx, rm.RAYO_OPTIONS), x)
    for a, b in zip(rayo, ref, strict=True):
        assert np.abs(a - b).max() <= 1e-4
    model = rt.fit(SMALL, generate(SMALL.messages, seed=SMALL.seed))
    p = parity(
        model,
        RayoRunner(data, d),
        rt._feeds(x),
        rm.OUTPUTS,
        labels=rm.OUTPUTS,
        bind={"T": MAX_TOKENS},
        margin=2 * SMALL.parity_tolerance_fields,
    )
    assert p.rows == p.same > 0
    assert p.within(SMALL.parity_tolerance_fields), p.max_abs

import struct
from pathlib import Path

import numpy as np
import pytest

from akili.export.weights import FORMAT_VERSION, MAGIC, Tensor, Weights, read_weights


def _weights() -> Weights:
    rng = np.random.default_rng(0)
    return Weights(
        model="test",
        source_file="test.tflite",
        source_sha256="0" * 64,
        source_size_bytes=123,
        layers=[{"name": "dense", "kind": "dense", "activation": "relu"}],
        tensors={
            "dense.kernel": Tensor(
                rng.integers(-127, 128, (3, 5), dtype=np.int8),
                np.array([0.5, 0.25, 0.125], dtype=np.float32),
                axis=0,
            ),
            "dense.bias": Tensor(np.array([1.5, -2.0, 0.0], dtype=np.float32)),
            "table": Tensor(
                np.array([[1, -2], [3, 4]], dtype=np.int8), np.array([0.1], dtype=np.float32)
            ),
        },
    )


def test_round_trip_is_exact() -> None:
    w = _weights()
    blob = w.to_bytes()
    back = read_weights(blob)
    assert back.to_bytes() == blob
    assert back.layers == w.layers
    assert back.source_sha256 == w.source_sha256
    for name, t in w.tensors.items():
        np.testing.assert_array_equal(back.tensors[name].value, t.value)
        np.testing.assert_array_equal(back.tensors[name].real(), t.real())
    np.testing.assert_array_equal(
        back.tensors["dense.kernel"].real()[1], w.tensors["dense.kernel"].value[1] * 0.25
    )
    np.testing.assert_allclose(back.tensors["table"].real(), [[0.1, -0.2], [0.3, 0.4]], rtol=1e-6)


def test_layout_is_as_specified() -> None:
    blob = _weights().to_bytes()
    assert blob[:4] == MAGIC == b"PCWT"
    version, header_len = struct.unpack_from("<II", blob, 4)
    assert version == FORMAT_VERSION == 1
    assert (12 + header_len) % 4 == 0
    # 3x5 int8 kernel padded to 16, 3 scales, 3 biases, 2x2 table, 1 scale.
    assert len(blob) == 12 + header_len + 16 + 12 + 12 + 4 + 4
    data = blob[12 + header_len :]
    assert struct.unpack_from("<3f", data, 16) == (0.5, 0.25, 0.125)


def test_refuses_what_does_not_fit() -> None:
    blob = _weights().to_bytes()
    with pytest.raises(ValueError, match="not a plain weights"):
        read_weights(b"XXXX" + blob[4:])
    with pytest.raises(ValueError, match="format version"):
        read_weights(blob[:4] + struct.pack("<I", 2) + blob[8:])
    with pytest.raises(ValueError, match="outside the data"):
        read_weights(blob[:-8])
    with pytest.raises(ValueError, match="after the last tensor"):
        read_weights(blob + b"\0\0\0\0")
    with pytest.raises(ValueError, match="int8 tensors need scales"):
        Tensor(np.zeros(3, dtype=np.int8))
    with pytest.raises(ValueError, match="scales for"):
        Tensor(np.zeros((2, 3), dtype=np.int8), np.ones(3, dtype=np.float32), axis=0)
    with pytest.raises(ValueError, match="not int8 or float32"):
        Tensor(np.zeros(3, dtype=np.float64))


def test_graph_and_metadata_round_trip() -> None:
    w = Weights(
        model="m",
        source_file="f",
        source_sha256="0" * 64,
        source_size_bytes=0,
        layers=[],
        tensors={"t": Tensor(np.zeros(2, np.float32))},
        graph={"version": 1},
        metadata={"a": 1},
    )
    back = read_weights(w.to_bytes())
    assert back.graph == {"version": 1}
    assert back.metadata == {"a": 1}
    plain = Weights(
        model="m",
        source_file="f",
        source_sha256="0" * 64,
        source_size_bytes=0,
        layers=[],
        tensors={"t": Tensor(np.zeros(2, np.float32))},
    )
    assert read_weights(plain.to_bytes()).graph is None


def _fake_conversion(
    monkeypatch: pytest.MonkeyPatch, names: list[str], mode: str = "decode", heads: int = 4
) -> list[object]:
    """rayo-convert's convert replaced by one that gives these tensors and a graph with three
    attention nodes; returns the options it was called with."""
    from rayo_convert import Result
    from rayo_convert.lower import Weight

    from akili.export import rayo

    seen: list[object] = []

    def convert(blob: bytes, name: str, options: object) -> Result:
        seen.append(options)
        weights = [
            Weight(
                n, np.arange(6, dtype=np.int8).reshape(2, 3), np.array([0.5, 0.25], np.float32), 0
            )
            if n.endswith("kernel")
            else Weight(n, np.ones(2, np.float32))
            for n in names
        ]
        nodes = [{"op": "attention", "heads": heads, "causal": 1} for _ in range(3)]
        return Result(b"", {"version": 1, "nodes": nodes}, weights, ["a note"], mode)

    monkeypatch.setattr(rayo, "convert", convert)
    return seen


def test_weights_from_rayo_convert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from akili.ask.reader_weights import LAYERS, RAYO_OPTIONS, read_tflite, required_tensors

    names = required_tensors()
    assert len(names) == 2 + 2 * (len(LAYERS) - 2)
    seen = _fake_conversion(monkeypatch, [*names, "const"])
    model = tmp_path / "ask-reader.tflite"
    model.write_bytes(b"tflite")
    w = read_tflite(model)
    assert seen == [RAYO_OPTIONS]
    assert RAYO_OPTIONS.split_inputs == {"question": [("tokens", 96), ("shapes", 96)]}
    assert RAYO_OPTIONS.outputs == ["slots", "intent"]
    assert w.model == "ask-reader"
    assert w.layers == LAYERS
    assert w.graph is not None
    assert w.source_size_bytes == 6
    k = w.tensors["conv_1.kernel"]
    assert k.value.dtype == np.int8
    assert k.axis == 0
    assert k.scales is not None
    assert w.tensors["const"].scales is None
    back = read_weights(w.to_bytes())
    assert set(back.tensors) == set(w.tensors)
    _fake_conversion(monkeypatch, names[1:])
    with pytest.raises(ValueError, match=r"rayo-convert gave no token_embedding\.table"):
        read_tflite(model)


def test_writer_needs_cached_attention(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from akili.ask.writer_weights import RAYO_OPTIONS, _required, read_tflite

    names = _required(3)
    assert "attention_2.query.kernel" in names
    assert "logits.bias" in names
    assert len(names) == 2 + 3 * (4 + 4 * 2 + 4) + 2 + 2
    model = tmp_path / "ask-writer.tflite"
    model.write_bytes(b"tflite")
    seen = _fake_conversion(monkeypatch, names)
    assert read_tflite(model, 3, 4).model == "ask-writer"
    assert seen == [RAYO_OPTIONS]
    assert RAYO_OPTIONS.rename == {"output_0": "logits"}
    _fake_conversion(monkeypatch, names, mode="sequence")
    with pytest.raises(ValueError, match=r"did not give 3 cached attention layers.*a note"):
        read_tflite(model, 3, 4)
    _fake_conversion(monkeypatch, names, heads=2)
    with pytest.raises(ValueError, match="of 4 heads"):
        read_tflite(model, 3, 4)

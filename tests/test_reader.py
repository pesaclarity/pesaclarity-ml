import json

import numpy as np
import pytest

from akili._optional import has_module
from akili.reader import vectors
from akili.reader.synthetic import (
    FIELD_LABELS,
    LAYOUTS,
    Field,
    Kind,
    generate,
    label_tokens,
    to_arrays,
)
from akili.reader.tokens import (
    HASH_BUCKETS,
    MAX_TOKENS,
    Shape,
    encode,
    hash_key,
    shape_of,
    token_id,
    tokenise,
)


def texts(message: str) -> list[str]:
    return [t.text for t in tokenise(message)]


def test_tokens_split_currency_but_keep_references() -> None:
    assert texts("Paid KES1,250.00 Ref SK29XY7") == ["Paid", "KES", "1,250.00", "Ref", "SK29XY7"]
    assert texts("Ksh.500 0712***678") == ["Ksh", ".", "500", "0712", "***678"]


def test_offsets_point_back_into_the_message() -> None:
    message = "Bal: KES 1,000.|Ref#77"
    for token in tokenise(message):
        assert message[token.start : token.end] == token.text


@pytest.mark.parametrize(
    ("text", "shape"),
    [
        ("1250", Shape.INTEGER),
        ("1250.00", Shape.DECIMAL),
        ("1,250", Shape.GROUPED),
        ("29/10/26", Shape.DATE),
        ("2026-10-29", Shape.DATE),
        ("14:05", Shape.TIME),
        ("***12", Shape.MASK),
        ("SK29XY7", Shape.MIXED),
        ("KES", Shape.UPPER),
        ("A", Shape.TITLE),
        ("Sacco", Shape.TITLE),
        ("debited", Shape.LOWER),
        (":", Shape.PUNCT),
    ],
)
def test_shapes(text: str, shape: Shape) -> None:
    assert shape_of(text) is shape


def test_numbers_hash_by_layout_not_value() -> None:
    assert hash_key("1,250.00") == hash_key("9,431.27") == "#9,999.99"
    assert token_id("1,250.00") == token_id("3,000.50")
    assert token_id("KES") == token_id("kes")
    assert all(1 <= token_id(w) < HASH_BUCKETS for w in ["a", "#", chr(0x2013)])


def test_input_is_padded_and_truncated() -> None:
    short = encode("Paid KES 50")
    assert len(short) == 2 * MAX_TOKENS
    assert short[3:MAX_TOKENS] == [0] * (MAX_TOKENS - 3)
    long = encode(" ".join(["word"] * (MAX_TOKENS + 10)))
    assert len(long) == 2 * MAX_TOKENS
    assert 0 not in long


def test_labels_follow_the_pieces() -> None:
    text = "Paid KES 50 to DEMO CAFE"
    spans = [(5, 11, Field.AMOUNT), (15, 24, Field.COUNTERPARTY)]
    assert label_tokens(text, spans) == [0, 1, 1, 0, 2, 2]


def test_synthetic_messages_cover_every_layout_and_kind() -> None:
    messages = generate(3000, seed=1)
    layouts = {m.style.split("|")[5] for m in messages if m.style.count("|") > 5}
    assert layouts == set(LAYOUTS)
    assert {m.kind for m in messages} == set(Kind)
    for m in messages:
        assert len(m.fields) == len(tokenise(m.text))
        if m.kind is not Kind.NOT_TRANSACTION:
            assert Field.AMOUNT in m.fields
            assert Field.COUNTERPARTY in m.fields


def test_synthetic_messages_are_reproducible() -> None:
    assert [m.text for m in generate(50, seed=3)] == [m.text for m in generate(50, seed=3)]


def test_arrays_mask_padding() -> None:
    x, y, k = to_arrays(generate(10, seed=2))
    assert x.shape == (10, 2 * MAX_TOKENS)
    assert y.shape == (10, MAX_TOKENS)
    assert k.shape == (10,)
    assert ((x[:, :MAX_TOKENS] == 0) == (y == -1)).all()


def test_vectors_file_matches_the_code() -> None:
    committed = vectors.default_path().read_text(encoding="utf-8")
    assert committed == vectors.render(vectors.build_vectors())
    doc = json.loads(committed)
    assert len(doc["cases"]) >= 8
    assert FIELD_LABELS[0] == "other"


def test_field_scores_count_whole_spans() -> None:
    from akili.reader.train import field_scores

    truth = np.array([[0, 1, 1, 0, 2, -1]])
    exact = field_scores(truth, np.array([[0, 1, 1, 0, 2, 0]]))
    assert exact["amount"]["f1"] == 1.0
    assert exact["counterparty"]["f1"] == 1.0
    partial = field_scores(truth, np.array([[0, 1, 0, 0, 2, 0]]))
    assert partial["amount"]["f1"] == 0.0


@pytest.mark.skipif(not has_module("tensorflow"), reason="needs the [train] extra")
def test_reader_trains_and_exports(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from akili.reader.train import MANIFEST_FILE, MODEL_FILE, ReaderConfig, train_reader

    report = train_reader(ReaderConfig(messages=1500, epochs=2), tmp_path, evaluate=False)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
    assert (tmp_path / MODEL_FILE).stat().st_size == manifest["model_size_bytes"]
    assert manifest["outputs"]["fields"]["labels"] == FIELD_LABELS
    assert report["model_size_bytes"] < 512_000

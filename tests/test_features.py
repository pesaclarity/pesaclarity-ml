from datetime import datetime

import pytest

from akili.config import FeatureConfig
from akili.features import (
    N_AMOUNT_BUCKETS,
    NUM_TOKEN,
    PHONE_TOKEN,
    amount_bucket,
    encode,
    encode_batch,
    feature_width,
    fnv1a_32,
    month_phase,
    token_ids,
    tokenise,
)
from akili.types import Direction, Transaction

CFG = FeatureConfig(max_tokens=6, hash_buckets=64)


def test_fnv1a_matches_reference_vectors() -> None:
    assert fnv1a_32("") == 0x811C9DC5
    assert fnv1a_32("a") == 0xE40C292C
    assert fnv1a_32("foobar") == 0xBF9CF968


def test_tokenise_masks_numbers_and_phones() -> None:
    assert tokenise("DEMO Fresh-Mart TILL 000123") == ["demo", "fresh", "mart", "till", NUM_TOKEN]
    assert tokenise("TEST PERSON 07 0700***007") == ["test", "person", NUM_TOKEN, PHONE_TOKEN]
    assert tokenise("PAYBILL 000400 ACC 0001") == ["paybill", NUM_TOKEN, "acc", NUM_TOKEN]


def test_token_ids_are_padded_and_in_range() -> None:
    ids = token_ids("demo grocers", CFG)
    assert len(ids) == CFG.max_tokens
    assert ids[2:] == [0] * 4
    assert all(1 <= i < CFG.hash_buckets for i in ids[:2])


def test_token_ids_truncate() -> None:
    assert len(token_ids("a b c d e f g h i", CFG)) == CFG.max_tokens


@pytest.mark.parametrize(
    ("amount", "bucket"),
    [(0, 0), (49.99, 0), (50, 1), (999, 4), (1_000, 5), (-1_000, 5), (10**7, N_AMOUNT_BUCKETS - 1)],
)
def test_amount_buckets(amount: float, bucket: int) -> None:
    assert amount_bucket(amount) == bucket


def test_month_phase() -> None:
    assert [month_phase(d) for d in (1, 5, 6, 24, 25, 31)] == [0, 0, 1, 1, 2, 2]


def test_encode_layout() -> None:
    tx = Transaction("DEMO GROCERS", 1_200, Direction.IN, datetime(2026, 3, 28, 13, 5))
    vec = encode(tx, CFG)
    assert len(vec) == feature_width(CFG)
    assert vec[CFG.max_tokens :] == [5, 1, 3, 5, 2]


def test_encode_batch_requires_labels() -> None:
    tx = Transaction("DEMO GROCERS", 100, Direction.OUT, datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="no target label"):
        encode_batch([tx], CFG, ["groceries"])
    labelled = Transaction("DEMO GROCERS", 100, Direction.OUT, datetime(2026, 1, 1), "groceries")
    batch = encode_batch([labelled], CFG, ["groceries"])
    assert batch.features.shape == (1, feature_width(CFG))
    assert batch.labels.tolist() == [0]

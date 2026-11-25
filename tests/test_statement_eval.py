import numpy as np

from akili.categories import default_categories
from akili.config import FeatureConfig
from akili.statement_eval import MARKERS, rows, score
from akili.synthetic import FAKE_MARKERS


def test_the_set_is_invented_and_unseen() -> None:
    data = rows()
    assert rows() == data
    assert len(data) > 150
    for r in data:
        words = r.payee.split()
        assert any(m in words for m in MARKERS), r.payee
        assert not any(m in words for m in FAKE_MARKERS), r.payee


def test_score_counts_per_category() -> None:
    labels = list(default_categories().labels)
    data = rows()
    want = np.zeros((len(data), len(labels)), np.float32)
    for i, r in enumerate(data):
        want[i, labels.index(r.category)] = 1
    report = score(lambda x: want[: len(x)], FeatureConfig())
    assert report["accuracy"] == 1.0
    assert report["wrong"] == []
    assert all(c["precision"] == 1.0 and c["recall"] == 1.0 for c in report["categories"].values())

    groceries = np.zeros_like(want)
    groceries[:, labels.index("groceries")] = 1
    report = score(lambda x: groceries[: len(x)], FeatureConfig())
    assert report["categories"]["groceries"]["recall"] == 1.0
    assert report["categories"]["fuel"]["recall"] == 0.0


def test_people_count_within_their_direction() -> None:
    labels = list(default_categories().labels)
    data = rows()
    assert sum(r.category == "transport" for r in data) >= 50
    assert sum(r.category == "salary" for r in data) >= 20
    assert any(r.payee.startswith("CUSTOMER TRANSFER FROM 0700***0") for r in data)
    got = np.zeros((len(data), len(labels)), np.float32)
    got[:, labels.index("gifts_received")] = 1
    people = score(lambda x: got[: len(x)], FeatureConfig())["people"]
    assert people["received_from_people"]["accuracy"] == 1.0
    assert people["sent_to_people"]["accuracy"] == 0.0

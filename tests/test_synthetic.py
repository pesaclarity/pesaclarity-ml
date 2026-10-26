import re
from pathlib import Path

from akili.config import SyntheticDataConfig
from akili.synthetic import FAKE_MARKERS, commercial_opt_ins, generate, read_jsonl, write_jsonl

CFG = SyntheticDataConfig(n_users=10, transactions_per_user=30, seed=3)
REAL_LOOKING_PHONE = re.compile(r"(?<![\d*])(?:\+?254|0)7\d{8}(?!\d)")


def test_generation_is_deterministic() -> None:
    assert generate(CFG) == generate(CFG)
    assert generate(CFG) != generate(CFG.model_copy(update={"seed": 4}))


def test_every_payee_is_obviously_fake() -> None:
    for tx in generate(CFG):
        assert tx.payee.split()[0] in FAKE_MARKERS, tx.payee
        assert not REAL_LOOKING_PHONE.search(tx.payee), tx.payee
        for kind in ("TILL", "PAYBILL"):
            if kind in tx.payee:
                number = tx.payee.split(kind, 1)[1].split()[0]
                assert number.startswith("000"), tx.payee
        if "***" in tx.payee:
            assert "0700***0" in tx.payee


def test_shape_and_labels() -> None:
    txs = generate(CFG)
    assert len(txs) == 300
    assert len({tx.user_id for tx in txs}) == 10
    assert all(tx.amount_minor >= 100 for tx in txs)
    assert all(tx.timestamp.utcoffset() is not None for tx in txs)
    assert all(tx.category for tx in txs)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    txs = generate(CFG)
    path = tmp_path / "tx.jsonl"
    write_jsonl(txs, path)
    assert list(read_jsonl(path)) == txs


def test_commercial_opt_ins_follow_rate() -> None:
    cfg = SyntheticDataConfig(commercial_opt_in_rate=0.25)
    users = [f"u{i}" for i in range(2000)]
    rate = len(commercial_opt_ins(cfg, users)) / len(users)
    assert 0.2 < rate < 0.3
    assert commercial_opt_ins(cfg, users) == commercial_opt_ins(cfg, reversed(users))

import json
from pathlib import Path
from typing import Any

import pytest

from akili.cli import main
from akili.config import FeatureConfig
from akili.export.manifest import write_model_manifest
from akili.export.size import describe
from akili.features import FEATURE_SCHEMA_VERSION, fnv1a_32
from akili.vectors import VECTOR_FILE, build_vectors, default_path, render

DOC = json.loads(default_path().read_text(encoding="utf-8"))


def test_committed_vectors_match_the_code() -> None:
    assert default_path().name == VECTOR_FILE
    assert default_path().read_text(encoding="utf-8") == render(build_vectors()), (
        "feature vectors are out of date: run `uv run akili vectors` and bump "
        "FEATURE_SCHEMA_VERSION if encode() output changed"
    )


def test_vectors_cover_the_edge_cases() -> None:
    cases = DOC["cases"]
    assert DOC["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert len(cases) >= 50
    inputs = [c["input"] for c in cases]
    assert {i["direction"] for i in inputs} == {"in", "out"}
    assert any(i["payee"] == "" for i in inputs)
    assert any(not i["payee"].isascii() for i in inputs)
    assert any(len(i["payee"]) > 500 for i in inputs)
    assert {0, 2**63 - 1} <= {i["amount_minor"] for i in inputs}
    widths = {len(c["expected"]["features"]) for c in cases}
    assert widths == {DOC["input"]["shape"][1]}


def _civil_from_days(z: int) -> tuple[int, int, int]:
    """Howard Hinnant's days-to-civil, as written in docs/FEATURES.md."""
    z += 719_468
    era = z // 146_097
    doe = z - era * 146_097
    yoe = (doe - doe // 1460 + doe // 36_524 - doe // 146_096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    day = doy - (153 * mp + 2) // 5 + 1
    month = mp + 3 if mp < 10 else mp - 9
    return yoe + era * 400 + (month <= 2), month, day


@pytest.mark.parametrize("case", DOC["cases"], ids=lambda c: c["id"])
def test_spec_integer_time_arithmetic_matches(case: dict[str, Any]) -> None:
    ms = case["input"]["timestamp_ms"]
    local_ms = ms + 3 * 3_600_000
    days = local_ms // 86_400_000
    hour = (local_ms - days * 86_400_000) // 3_600_000
    year, month, day = _civil_from_days(days)
    local = case["expected"]["local_time"]
    assert local.startswith(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}")
    features = case["expected"]["features"]
    assert features[-3] == hour // 4
    assert features[-2] == (days + 3) % 7
    assert features[-1] == (0 if day <= 5 else 2 if day >= 25 else 1)


def test_hash_checks() -> None:
    for check in DOC["hash_checks"]:
        assert fnv1a_32(check["text"]) == check["fnv1a_32"]


def test_vectors_for_another_config_differ_only_in_tokens() -> None:
    small = build_vectors(FeatureConfig(max_tokens=4, hash_buckets=64))
    assert small["input"]["shape"] == [-1, 9]
    for a, b in zip(small["cases"], DOC["cases"], strict=True):
        assert a["expected"]["features"][4:] == b["expected"]["features"][8:]


def test_cli_vectors_writes_and_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "v.json"
    assert main(["vectors", "--out", str(out), "--check"]) == 1
    assert main(["vectors", "--out", str(out)]) == 0
    assert main(["vectors", "--out", str(out), "--check"]) == 0
    assert main(["vectors", "--check"]) == 0
    assert "matches" in capsys.readouterr().out


def test_model_manifest_records_schema_and_hash(tmp_path: Path) -> None:
    model = tmp_path / "categoriser.tflite"
    model.write_bytes(b"not really a model")
    path = write_model_manifest(describe(model), FeatureConfig())
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent == tmp_path
    assert doc["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert doc["model_sha256"] == describe(model).sha256
    assert doc["input"] == {"name": "features", "dtype": "int32", "shape": [-1, 13]}
    assert len(doc["layout"]) == 13

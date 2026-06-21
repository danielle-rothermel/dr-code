"""Unit tests for AttemptRecord export."""

from __future__ import annotations

from pathlib import Path

from dr_code.datasets.export import read_attempts, write_attempts
from dr_code.datasets.pool_loader import load_pool_parquet

_FIXTURE_PARQUET = Path("tests/fixtures/pool/sample.parquet")


def test_parquet_round_trip(tmp_path: Path) -> None:
    records = load_pool_parquet(_FIXTURE_PARQUET)
    out = tmp_path / "attempts.parquet"
    write_attempts(records, out)
    loaded = read_attempts(out)
    assert len(loaded) == len(records)
    assert loaded[0].sample_id == records[0].sample_id
    assert loaded[0].provenance.pool_name == records[0].provenance.pool_name


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = load_pool_parquet(_FIXTURE_PARQUET)
    out = tmp_path / "attempts.jsonl"
    write_attempts(records, out)
    loaded = read_attempts(out)
    assert loaded[0].model_dump() == records[0].model_dump()

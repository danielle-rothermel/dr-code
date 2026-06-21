"""Unit tests for attempt summary stats."""

from __future__ import annotations

from pathlib import Path

from dr_code.datasets.pool_loader import load_pool_parquet
from dr_code.datasets.stats import summarize_attempts

_FIXTURE_PARQUET = Path("tests/fixtures/pool/sample.parquet")


def test_summarize_attempts_counts() -> None:
    records = load_pool_parquet(_FIXTURE_PARQUET)
    summary = summarize_attempts(records, sample_count=2)
    assert summary["record_count"] == 4
    assert summary["weighted_count"] == 4
    assert summary["unique_tasks"] == 2
    assert summary["source_breakdown"]["pool"] == 4
    assert len(summary["samples"]) == 2

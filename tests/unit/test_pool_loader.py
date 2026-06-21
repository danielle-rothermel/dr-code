"""Unit tests for pool loader."""

from __future__ import annotations

from pathlib import Path

from dr_code.datasets.pool_loader import (
    infer_task_id_from_dedup_path,
    load_pool_dedup_jsonl,
    load_pool_dedup_with_parquet,
    load_pool_parquet,
)

_FIXTURE_PARQUET = Path("tests/fixtures/pool/sample.parquet")
_FIXTURE_DEDUP = Path("tests/fixtures/pool/human_eval-0-decode-dedup.jsonl")


def test_load_pool_parquet_fixture() -> None:
    records = load_pool_parquet(_FIXTURE_PARQUET)
    assert len(records) == 4
    assert all(record.provenance.source.value == "pool" for record in records)
    assert records[0].entry_point == "has_close_elements"


def test_infer_task_id_from_dedup_filename() -> None:
    assert (
        infer_task_id_from_dedup_path("human_eval-0-decode-dedup.jsonl")
        == "HumanEval/0"
    )


def test_load_pool_dedup_jsonl_infers_task_id() -> None:
    records = load_pool_dedup_jsonl(_FIXTURE_DEDUP)
    assert len(records) == 3
    assert records[0].task_id == "HumanEval/0"
    assert records[0].provenance.occurrence_count == 42


def test_dedup_with_parquet_join_uses_richer_decoder_input() -> None:
    records = load_pool_dedup_with_parquet(_FIXTURE_DEDUP, _FIXTURE_PARQUET)
    matched = next(
        record
        for record in records
        if record.raw_output.startswith("```python")
    )
    assert matched.decoder_input.startswith("from typing import")
    assert matched.provenance.pool_name == "budget_dec_v0_size6"

    fallback = next(
        record
        for record in records
        if record.raw_output.endswith("return True\n")
    )
    assert fallback.decoder_input.startswith("from typing import")

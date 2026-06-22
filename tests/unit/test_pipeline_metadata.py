"""Unit tests for eval run metadata models."""

from __future__ import annotations

from pathlib import Path

from dr_code.models.attempts import AttemptProvenance, AttemptRecord, AttemptSource
from dr_code.pipeline.metadata import (
    EvalSeedSource,
    build_seed_metadata,
)


def _record(sample_id: str, task_id: str) -> AttemptRecord:
    return AttemptRecord(
        sample_id=sample_id,
        run_id="run-1",
        task_id=task_id,
        entry_point="fn",
        decoder_input="desc",
        raw_output="code",
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )


def test_seed_metadata_hash_preserves_order() -> None:
    metadata = build_seed_metadata(
        records=[_record("sample-a", "HumanEval/1"), _record("sample-b", "HumanEval/0")],
        source=EvalSeedSource.DUMP_DIR,
        source_path=Path("tests/fixtures/pool"),
        task_indices=[1, 0],
        limit_per_task=2,
    )
    reordered = build_seed_metadata(
        records=[_record("sample-b", "HumanEval/0"), _record("sample-a", "HumanEval/1")],
        source=EvalSeedSource.DUMP_DIR,
        source_path=Path("tests/fixtures/pool"),
        task_indices=[1, 0],
        limit_per_task=2,
    )

    assert metadata.expected_jobs == 2
    assert metadata.task_ids == ("HumanEval/0", "HumanEval/1")
    assert metadata.task_indices == (1, 0)
    assert metadata.source == EvalSeedSource.DUMP_DIR
    assert metadata.sample_ids_hash != reordered.sample_ids_hash

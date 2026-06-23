"""Unit tests for eval run metadata models."""

from __future__ import annotations

from pathlib import Path

from dr_code.models.attempts import AttemptProvenance, AttemptRecord, AttemptSource
from dr_code.pipeline.metadata import (
    EvalSeedSource,
    EvalRunMetadataStore,
    build_init_metadata,
    build_seed_metadata,
)


def _record(sample_id: str, task_id: str) -> AttemptRecord:
    return AttemptRecord(
        sample_id=sample_id,
        run_id="run-1",
        task_id=task_id,
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


def test_record_init_ignores_mongo_insert_id() -> None:
    class _Collection:
        def insert_one(self, document):
            document["_id"] = object()

    store = EvalRunMetadataStore.__new__(EvalRunMetadataStore)
    store._collection = _Collection()

    metadata = build_init_metadata(
        worker_spec="parse=1,test=1",
        workers_by_stage={"parse": 1, "test": 1},
    )

    stored = store.record_init(run_id="run-1", metadata=metadata)

    assert stored.run_id == "run-1"
    assert stored.init == metadata

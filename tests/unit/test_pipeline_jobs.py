"""Unit tests for pipeline job serialization."""

from __future__ import annotations

from dr_queues import JobEnvelope

from dr_code.models.attempts import AttemptProvenance, AttemptRecord, AttemptSource
from dr_code.pipeline.definition import PIPELINE_ID
from dr_code.pipeline.jobs import (
    attempt_from_job,
    attempt_to_payload,
    build_seed_jobs,
    stamp_run_id,
)


def _sample_record() -> AttemptRecord:
    return AttemptRecord(
        sample_id="abc123",
        run_id=None,
        task_id="HumanEval/0",
        entry_point="has_close_elements",
        decoder_input='def has_close_elements(numbers, threshold):\n    """doc"""\n',
        raw_output="def has_close_elements(numbers, threshold):\n    return False\n",
        provenance=AttemptProvenance(source=AttemptSource.POOL, occurrence_count=3),
    )


def test_attempt_payload_round_trip() -> None:
    record = _sample_record()
    payload = attempt_to_payload(record)
    job = JobEnvelope(
        run_id="run-1",
        lane="default",
        repeat=0,
        step_index=0,
        pipeline_id=PIPELINE_ID,
        payload=payload,
    )
    restored = attempt_from_job(job)
    assert restored == record


def test_stamp_run_id() -> None:
    record = _sample_record()
    stamped = stamp_run_id([record], "run-99")[0]
    assert stamped.run_id == "run-99"
    assert stamped.sample_id == record.sample_id


def test_build_seed_jobs_one_per_record() -> None:
    records = stamp_run_id([_sample_record(), _sample_record()], "run-1")
    jobs = build_seed_jobs(records, run_id="run-1")
    assert len(jobs) == 2
    assert jobs[0].repeat == 0
    assert jobs[1].repeat == 1
    assert attempt_from_job(jobs[0]).run_id == "run-1"

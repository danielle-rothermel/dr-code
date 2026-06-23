"""Unit tests for pipeline proof reporting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dr_queues import EventKind, PipelineEvent

from dr_code.models.attempts import AttemptProvenance, AttemptRecord, AttemptSource
from dr_code.models.outcomes import ParseOutcome, TestOutcome
from dr_code.pipeline.report import build_proof_report


def _attempt(sample_id: str, task_id: str = "HumanEval/0") -> AttemptRecord:
    return AttemptRecord(
        sample_id=sample_id,
        run_id="run-1",
        task_id=task_id,
        decoder_input="desc",
        raw_output="code",
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )


def _stage_events(
    job_id: str,
    stage: str,
    *,
    base: datetime,
    latency_seconds: float,
) -> list[PipelineEvent]:
    return [
        PipelineEvent(
            run_id="run-1",
            job_id=job_id,
            lane="default",
            stage=stage,
            event=EventKind.STAGE_STARTED,
            timestamp=base.isoformat(),
        ),
        PipelineEvent(
            run_id="run-1",
            job_id=job_id,
            lane="default",
            stage=stage,
            event=EventKind.STAGE_OUTPUT,
            timestamp=(base + timedelta(seconds=latency_seconds)).isoformat(),
        ),
    ]


def test_build_proof_report_timing_and_outcomes() -> None:
    base = datetime(2026, 6, 21, 12, 0, 0, tzinfo=UTC)
    events = [
        *_stage_events("job-a", "parse", base=base, latency_seconds=0.5),
        *_stage_events("job-a", "test", base=base, latency_seconds=2.0),
        PipelineEvent(
            run_id="run-1",
            job_id="job-a",
            lane="default",
            stage="terminal",
            event=EventKind.TERMINAL,
            timestamp=(base + timedelta(seconds=3)).isoformat(),
            payload={
                "payload": {
                    "attempt": {
                        "sample_id": "s1",
                        "task_id": "HumanEval/0",
                    },
                },
            },
        ),
    ]
    attempts = [_attempt("s1")]
    parse_outcomes = [
        ParseOutcome(
            sample_id="s1",
            run_id="run-1",
            task_id="HumanEval/0",
            parse_success=True,
        ),
    ]
    test_outcomes = [
        TestOutcome(
            sample_id="s1",
            run_id="run-1",
            task_id="HumanEval/0",
            parse_success=True,
            outcome_kind="tested",
            tests_ran=True,
            test_pass_rate=1.0,
            all_tests_passed=True,
        ),
    ]

    report = build_proof_report(
        run_id="run-1",
        attempts=attempts,
        events=events,
        parse_outcomes=parse_outcomes,
        test_outcomes=test_outcomes,
        expected_jobs=1,
        terminal_count=1,
        wall_seconds=3.0,
    )

    assert report.payload["complete"] is True
    assert report.payload["throughput"]["by_stage"]["parse"] == 2.0
    assert report.payload["throughput"]["by_stage"]["test"] == 0.5
    assert report.payload["outcomes"]["outcome_kind_counts"]["tested"] == 1
    assert report.payload["outcomes"]["missing_test_count"] == 0

"""Post-run export of pipeline artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from dr_queues import (
    EventKind,
    JobEnvelope,
    MongoRunStore,
    PipelineEvent,
    filter_run_events,
)

from dr_code.datasets.export import write_attempts
from dr_code.models.attempts import AttemptRecord
from dr_code.models.base import FrozenModel
from dr_code.models.outcomes import ParseOutcome, TestOutcome
from dr_code.pipeline.jobs import attempt_from_job
from dr_code.pipeline.report import build_proof_report

PARSE_STAGE = "parse"
TEST_STAGE = "test"


class RunExportPaths(FrozenModel):
    """Paths written by export_run_artifacts."""

    run_dir: Path
    attempts: Path
    parse_jsonl: Path
    test_jsonl: Path
    manifest: Path
    proof_report: Path | None = None


class EvalRunOutcomes(FrozenModel):
    """Parse and test outcomes reconstructed from a persisted eval run."""

    parse_outcomes: list[ParseOutcome]
    test_outcomes: list[TestOutcome]


def read_eval_run_outcomes(
    *,
    run_id: str,
    mongo_sink: MongoRunStore | None = None,
) -> EvalRunOutcomes:
    """Read parse and test outcomes from persisted dr-queues events."""
    sink = mongo_sink or MongoRunStore()
    owns_sink = mongo_sink is None
    try:
        events = filter_run_events(sink.read_by_run_id(run_id), run_id)
        terminals = [
            event for event in events if event.event == EventKind.TERMINAL
        ]
        return EvalRunOutcomes(
            parse_outcomes=_parse_outcomes_from_events(events),
            test_outcomes=_test_outcomes_from_terminals(terminals),
        )
    finally:
        if owns_sink:
            sink.close()


def export_run_artifacts(
    *,
    run_id: str,
    attempts: list[AttemptRecord] | None = None,
    mongo_sink: MongoRunStore | None = None,
    output_root: Path | str = Path("exports/runs"),
) -> RunExportPaths:
    """Write derived artifacts reconstructed from persisted run state."""
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    attempts_path = run_dir / "attempts.parquet"
    parse_path = run_dir / "parse.jsonl"
    test_path = run_dir / "test.jsonl"
    manifest_out = run_dir / "manifest.json"
    proof_report_path = run_dir / "proof_report.json"

    sink = mongo_sink or MongoRunStore()
    owns_sink = mongo_sink is None
    written_proof_report: Path | None = None
    try:
        manifest = sink.get_manifest(run_id)
        first_stage = manifest.stages[0].name
        reconstructed_attempts = attempts or _attempts_from_stage_jobs(
            sink,
            run_id=run_id,
            stage=first_stage,
        )
        write_attempts(reconstructed_attempts, attempts_path)

        events = filter_run_events(sink.read_by_run_id(run_id), run_id)
        parse_outcomes = _parse_outcomes_from_events(events)
        terminals = [
            event for event in events if event.event == EventKind.TERMINAL
        ]
        test_outcomes = _test_outcomes_from_terminals(terminals)
        _write_outcomes_jsonl(parse_path, parse_outcomes)
        _write_outcomes_jsonl(test_path, test_outcomes)
        manifest_out.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        expected_jobs = sink.expected_job_count(run_id)
        if terminals and len(terminals) >= expected_jobs:
            wall_seconds = wall_seconds_from_events(events)
            report = build_proof_report(
                run_id=run_id,
                attempts=reconstructed_attempts,
                events=events,
                parse_outcomes=parse_outcomes,
                test_outcomes=test_outcomes,
                expected_jobs=expected_jobs,
                terminal_count=len(terminals),
                wall_seconds=wall_seconds,
            )
            written_proof_report = report.write_json(proof_report_path)
    finally:
        if owns_sink:
            sink.close()

    return RunExportPaths(
        run_dir=run_dir,
        attempts=attempts_path,
        parse_jsonl=parse_path,
        test_jsonl=test_path,
        manifest=manifest_out,
        proof_report=written_proof_report,
    )


def _attempts_from_stage_jobs(
    sink: MongoRunStore,
    *,
    run_id: str,
    stage: str,
) -> list[AttemptRecord]:
    jobs = [
        JobEnvelope.model_validate(state.job)
        for state in sink.list_job_states(run_id, stage=stage)
    ]
    return [
        attempt_from_job(job)
        for job in sorted(jobs, key=lambda job: job.repeat)
    ]


def _parse_outcomes_from_events(
    events: list[PipelineEvent],
) -> list[ParseOutcome]:
    outcomes: list[ParseOutcome] = []
    for event in events:
        if event.event != EventKind.STAGE_OUTPUT or event.stage != PARSE_STAGE:
            continue
        raw = event.payload.get("step_record")
        if raw is not None:
            outcomes.append(ParseOutcome.model_validate(raw))
    return outcomes


def _test_outcomes_from_terminals(
    terminals: list[PipelineEvent],
) -> list[TestOutcome]:
    outcomes: list[TestOutcome] = []
    for event in terminals:
        job = JobEnvelope.model_validate(event.payload)
        raw = job.step_records.get(TEST_STAGE)
        if raw is not None:
            outcomes.append(TestOutcome.model_validate(raw))
    return outcomes


def _write_outcomes_jsonl(
    path: Path, outcomes: Sequence[ParseOutcome | TestOutcome]
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(outcome.model_dump_json())
            handle.write("\n")


def wall_seconds_from_events(events: list[PipelineEvent]) -> float:
    starts = [
        datetime.fromisoformat(event.timestamp)
        for event in events
        if event.event == EventKind.STAGE_STARTED
    ]
    terminals = [
        datetime.fromisoformat(event.timestamp)
        for event in events
        if event.event == EventKind.TERMINAL
    ]
    if not starts or not terminals:
        return 0.0
    return (max(terminals) - min(starts)).total_seconds()

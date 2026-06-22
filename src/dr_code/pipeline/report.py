"""Proof-run timing and outcome reporting."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dr_queues import EventKind, PipelineEvent, filter_run_events

from dr_code.models.attempts import AttemptRecord
from dr_code.models.outcomes import ParseOutcome, TestOutcome


@dataclass(frozen=True)
class ProofReport:
    """Structured proof report for a pipeline run."""

    run_id: str
    expected_jobs: int
    terminal_count: int
    payload: dict[str, Any]

    def write_json(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")
        return out


def build_proof_report(
    *,
    run_id: str,
    attempts: list[AttemptRecord],
    events: list[PipelineEvent],
    parse_outcomes: list[ParseOutcome],
    test_outcomes: list[TestOutcome],
    expected_jobs: int,
    terminal_count: int,
    wall_seconds: float,
) -> ProofReport:
    """Build timing and outcome report from events and exports."""
    filtered = filter_run_events(events, run_id)
    stage_timing = _stage_timing(filtered)
    job_task_ids = _job_task_ids(filtered, attempts)
    throughput = _throughput(
        stage_timing,
        job_task_ids,
        terminal_count=terminal_count,
        wall_seconds=wall_seconds,
    )
    outcomes = _outcome_summary(attempts, parse_outcomes, test_outcomes)
    payload = {
        "run_id": run_id,
        "expected_jobs": expected_jobs,
        "terminal_count": terminal_count,
        "complete": terminal_count >= expected_jobs,
        "wall_seconds": wall_seconds,
        "throughput": throughput,
        "outcomes": outcomes,
    }
    return ProofReport(
        run_id=run_id,
        expected_jobs=expected_jobs,
        terminal_count=terminal_count,
        payload=payload,
    )


def format_proof_summary(report: ProofReport) -> str:
    """Human-readable summary for stdout."""
    payload = report.payload
    lines = [
        f"run_id={report.run_id}",
        f"terminals={report.terminal_count}/{report.expected_jobs}",
        f"wall_seconds={payload['wall_seconds']:.1f}",
    ]
    overall = payload["throughput"]["overall_samples_per_second"]
    lines.append(f"overall_samples_per_second={overall:.3f}")
    for stage, rate in payload["throughput"]["by_stage"].items():
        lines.append(f"  {stage}_samples_per_second={rate:.3f}")
    lines.append("outcome_kind_counts:")
    for kind, count in sorted(payload["outcomes"]["outcome_kind_counts"].items()):
        lines.append(f"  {kind}={count}")
    return "\n".join(lines)


def _stage_timing(events: list[PipelineEvent]) -> dict[tuple[str, str], float]:
    """Map (job_id, stage) → handler latency seconds."""
    started: dict[tuple[str, str], datetime] = {}
    latency: dict[tuple[str, str], float] = {}
    for event in events:
        if event.event not in {EventKind.STAGE_STARTED, EventKind.STAGE_OUTPUT}:
            continue
        key = (event.job_id, event.stage)
        ts = datetime.fromisoformat(event.timestamp)
        if event.event == EventKind.STAGE_STARTED:
            started[key] = ts
        elif key in started:
            latency[key] = (ts - started[key]).total_seconds()
    return latency


def _job_task_ids(
    events: list[PipelineEvent],
    attempts: list[AttemptRecord],
) -> dict[str, str]:
    """Map job_id → task_id from terminal events or attempts index."""
    by_sample = {record.sample_id: record.task_id for record in attempts}
    job_tasks: dict[str, str] = {}
    for event in events:
        if event.event != EventKind.TERMINAL:
            continue
        payload = event.payload
        attempt = payload.get("payload", {}).get("attempt", {})
        task_id = attempt.get("task_id")
        sample_id = attempt.get("sample_id")
        if task_id is not None:
            job_tasks[event.job_id] = str(task_id)
        elif sample_id is not None and sample_id in by_sample:
            job_tasks[event.job_id] = by_sample[sample_id]
    return job_tasks


def _throughput(
    stage_timing: dict[tuple[str, str], float],
    job_task_ids: dict[str, str],
    *,
    terminal_count: int,
    wall_seconds: float,
) -> dict[str, Any]:
    by_stage_counts: Counter[str] = Counter()
    by_stage_seconds: dict[str, float] = defaultdict(float)
    by_task_stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_task_stage_seconds: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float),
    )

    for (job_id, stage), seconds in stage_timing.items():
        by_stage_counts[stage] += 1
        by_stage_seconds[stage] += seconds
        task_id = job_task_ids.get(job_id, "unknown")
        by_task_stage_counts[task_id][stage] += 1
        by_task_stage_seconds[task_id][stage] += seconds

    by_stage_rate = {
        stage: (
            by_stage_counts[stage] / by_stage_seconds[stage]
            if by_stage_seconds[stage] > 0
            else 0.0
        )
        for stage in by_stage_counts
    }
    by_task: dict[str, dict[str, float]] = {}
    for task_id, stage_counts in by_task_stage_counts.items():
        by_task[task_id] = {
            stage: (
                stage_counts[stage] / by_task_stage_seconds[task_id][stage]
                if by_task_stage_seconds[task_id][stage] > 0
                else 0.0
            )
            for stage in stage_counts
        }

    overall = terminal_count / wall_seconds if wall_seconds > 0 else 0.0
    return {
        "overall_samples_per_second": overall,
        "by_stage": by_stage_rate,
        "by_task": by_task,
    }


def _outcome_summary(
    attempts: list[AttemptRecord],
    parse_outcomes: list[ParseOutcome],
    test_outcomes: list[TestOutcome],
) -> dict[str, Any]:
    parse_by_id = {outcome.sample_id: outcome for outcome in parse_outcomes}
    test_by_id = {outcome.sample_id: outcome for outcome in test_outcomes}
    attempt_ids = {record.sample_id for record in attempts}

    outcome_kind_counts: Counter[str] = Counter()
    weighted_outcome_kind: Counter[str] = Counter()
    parse_success = 0
    tests_ran = 0
    all_passed = 0
    by_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "seeded": 0,
            "parse_success": 0,
            "tested": 0,
            "all_tests_passed": 0,
        },
    )

    for record in attempts:
        by_task[record.task_id]["seeded"] += 1
        parse = parse_by_id.get(record.sample_id)
        if parse is not None and parse.parse_success:
            parse_success += 1
            by_task[record.task_id]["parse_success"] += 1
        test = test_by_id.get(record.sample_id)
        if test is None:
            continue
        outcome_kind_counts[test.outcome_kind] += 1
        weighted_outcome_kind[test.outcome_kind] += (
            record.provenance.occurrence_count
        )
        if test.outcome_kind == "tested":
            by_task[record.task_id]["tested"] += 1
            tests_ran += 1
            if test.all_tests_passed:
                all_passed += 1
                by_task[record.task_id]["all_tests_passed"] += 1

    missing_test = sorted(attempt_ids - set(test_by_id))
    return {
        "attempt_count": len(attempts),
        "parse_success_count": parse_success,
        "tests_ran_count": tests_ran,
        "all_tests_passed_count": all_passed,
        "outcome_kind_counts": dict(outcome_kind_counts),
        "outcome_kind_counts_weighted": dict(weighted_outcome_kind),
        "by_task": dict(by_task),
        "missing_test_sample_ids": missing_test,
        "missing_test_count": len(missing_test),
    }

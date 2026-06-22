"""Live test-worker throughput tuning for in-flight pipeline runs."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from dr_queues import replace_stage_workers, stop_workers
from pymongo import MongoClient

from dr_code.pipeline.mongo import (
    EVAL_RESULTS_COLLECTION,
    _database_name,
    mongodb_url,
)
from dr_code.pipeline.runner import DEFAULT_HANDLERS_MODULE

PIPELINE_EVENTS_COLLECTION = "pipeline_events"
TEST_STAGE = "test"


@dataclass(frozen=True)
class SweepStepResult:
    """Throughput measurement for one worker count."""

    workers: int
    swapped: bool
    terminals_before: int
    terminals_after: int
    warmup_seconds: float
    measure_seconds: float
    samples_per_second: float
    infra_error_before: int
    infra_error_after: int
    infra_error_delta: int
    reliable: bool


@dataclass
class SweepReport:
    """Full sweep report."""

    run_id: str
    expected_jobs: int
    steps: list[SweepStepResult] = field(default_factory=list)
    best_workers: int = 0
    best_samples_per_second: float = 0.0
    stop_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "expected_jobs": self.expected_jobs,
            "best_workers": self.best_workers,
            "best_samples_per_second": self.best_samples_per_second,
            "stop_reason": self.stop_reason,
            "steps": [
                {
                    "workers": step.workers,
                    "swapped": step.swapped,
                    "terminals_before": step.terminals_before,
                    "terminals_after": step.terminals_after,
                    "warmup_seconds": step.warmup_seconds,
                    "measure_seconds": step.measure_seconds,
                    "samples_per_second": step.samples_per_second,
                    "infra_error_delta": step.infra_error_delta,
                    "reliable": step.reliable,
                }
                for step in self.steps
            ],
        }

    def write_json(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out


def count_terminals(run_id: str, *, mongo_url: str | None = None) -> int:
    """Count TERMINAL pipeline events for a run."""
    client = MongoClient(mongo_url or mongodb_url())
    try:
        database = client.get_database(_database_name(mongodb_url()))
        collection = database[PIPELINE_EVENTS_COLLECTION]
        return collection.count_documents(
            {"run_id": run_id, "event": "terminal"}
        )
    finally:
        client.close()


def count_infra_errors(run_id: str, *, mongo_url: str | None = None) -> int:
    """Count infra_error eval results for a run."""
    client = MongoClient(mongo_url or mongodb_url())
    try:
        database = client.get_database(_database_name(mongodb_url()))
        collection = database[EVAL_RESULTS_COLLECTION]
        return collection.count_documents(
            {"run_id": run_id, "outcome_kind": "infra_error"},
        )
    finally:
        client.close()


def count_stage_completions(
    run_id: str,
    stage: str,
    *,
    mongo_url: str | None = None,
) -> int:
    """Count stage_output events for a pipeline stage."""
    client = MongoClient(mongo_url or mongodb_url())
    try:
        database = client.get_database(_database_name(mongodb_url()))
        collection = database[PIPELINE_EVENTS_COLLECTION]
        return collection.count_documents(
            {"run_id": run_id, "stage": stage, "event": "stage_output"},
        )
    finally:
        client.close()


def measure_throughput(
    run_id: str,
    *,
    window_seconds: float,
    poll_interval: float = 5.0,
    stall_timeout: float = 120.0,
) -> tuple[float, int, int]:
    """Measure terminal throughput over a window; detect stalls."""
    start_count = count_terminals(run_id)
    started = time.monotonic()
    deadline = started + window_seconds
    last_count = start_count
    last_progress = started

    while time.monotonic() < deadline:
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
        current = count_terminals(run_id)
        if current > last_count:
            last_count = current
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress > stall_timeout:
            msg = (
                f"Terminal count stalled at {current} for "
                f"{stall_timeout:.0f}s during measure window"
            )
            raise RuntimeError(msg)

    end_count = count_terminals(run_id)
    elapsed = time.monotonic() - started
    rate = (end_count - start_count) / elapsed if elapsed > 0 else 0.0
    return rate, start_count, end_count


def replace_test_workers(
    run_id: str,
    workers: int,
    *,
    handlers_module: str = DEFAULT_HANDLERS_MODULE,
) -> subprocess.Popen[bytes]:
    """Hot-swap test stage workers."""
    return replace_stage_workers(
        run_id=run_id,
        stage=TEST_STAGE,
        workers=workers,
        handlers_module=handlers_module,
    )


def stop_parse_worker_if_idle(run_id: str, *, expected_jobs: int) -> bool:
    """Stop parse worker when parse stage is complete to free CPU."""
    parse_done = count_stage_completions(run_id, "parse")
    if parse_done < expected_jobs:
        return False
    return bool(stop_workers(run_id=run_id, stage="parse"))


def run_sweep(
    *,
    run_id: str,
    expected_jobs: int,
    start_workers: int = 2,
    multiplier: int = 2,
    window_seconds: float = 60.0,
    warmup_seconds: float = 15.0,
    max_workers: int = 16,
    stop_threshold: float = 0.10,
    min_samples_in_window: int = 5,
    handlers_module: str = DEFAULT_HANDLERS_MODULE,
    dry_run: bool = False,
    apply_best: bool = True,
    on_step: Callable[[SweepStepResult], None] | None = None,
) -> SweepReport:
    """Run N×multiplier sweep on live test workers."""
    report = SweepReport(run_id=run_id, expected_jobs=expected_jobs)
    workers = start_workers
    best_rate = -1.0
    best_workers = start_workers
    previous_rate: float | None = None
    worker_process: subprocess.Popen[bytes] | None = None

    stop_parse_worker_if_idle(run_id, expected_jobs=expected_jobs)

    while workers <= max_workers:
        swapped = False
        if worker_process is None and workers == start_workers:
            swapped = False
        elif not dry_run:
            worker_process = replace_test_workers(
                run_id,
                workers,
                handlers_module=handlers_module,
            )
            swapped = True
            time.sleep(1.0)
            if worker_process.poll() is not None:
                msg = f"Test worker exited immediately with code {worker_process.returncode}"
                raise RuntimeError(msg)

        if not dry_run and swapped and warmup_seconds > 0:
            time.sleep(warmup_seconds)

        infra_before = count_infra_errors(run_id) if not dry_run else 0

        if dry_run:
            rate, t_before, t_after = (
                0.0,
                count_terminals(run_id),
                count_terminals(run_id),
            )
            reliable = False
        else:
            rate, t_before, t_after = measure_throughput(
                run_id,
                window_seconds=window_seconds,
            )
            reliable = (t_after - t_before) >= min_samples_in_window

        infra_after = count_infra_errors(run_id) if not dry_run else 0
        step = SweepStepResult(
            workers=workers,
            swapped=swapped,
            terminals_before=t_before,
            terminals_after=t_after,
            warmup_seconds=warmup_seconds if swapped else 0.0,
            measure_seconds=window_seconds,
            samples_per_second=rate,
            infra_error_before=infra_before,
            infra_error_after=infra_after,
            infra_error_delta=infra_after - infra_before,
            reliable=reliable,
        )
        report.steps.append(step)
        if on_step is not None:
            on_step(step)

        if reliable and rate > best_rate:
            best_rate = rate
            best_workers = workers

        if previous_rate is not None and reliable:
            if rate < previous_rate:
                report.stop_reason = f"regression at workers={workers}"
                break
            if previous_rate > 0:
                gain = (rate - previous_rate) / previous_rate
                if gain < stop_threshold:
                    report.stop_reason = (
                        f"diminishing returns at workers={workers} "
                        f"(gain={gain:.1%})"
                    )
                    break

        previous_rate = rate if reliable else previous_rate
        next_workers = workers * multiplier
        if next_workers > max_workers:
            report.stop_reason = (
                report.stop_reason or f"reached max_workers={max_workers}"
            )
            break
        workers = next_workers

    if not report.stop_reason:
        report.stop_reason = "completed sweep range"

    report.best_workers = best_workers
    report.best_samples_per_second = best_rate if best_rate >= 0 else 0.0

    if apply_best and not dry_run and best_workers > 0:
        if worker_process is None or best_workers != workers:
            worker_process = replace_test_workers(
                run_id,
                best_workers,
                handlers_module=handlers_module,
            )
            time.sleep(1.0)
            if worker_process.poll() is not None:
                msg = (
                    f"Failed to apply best workers={best_workers}; "
                    f"exit code {worker_process.returncode}"
                )
                raise RuntimeError(msg)

    return report


def format_sweep_table(report: SweepReport) -> str:
    """Human-readable sweep summary."""
    lines = [
        f"run_id={report.run_id}",
        f"best_workers={report.best_workers} "
        f"best_samples_per_second={report.best_samples_per_second:.3f}",
        f"stop_reason={report.stop_reason}",
        "",
        "workers  swapped  rate/s   delta  reliable",
    ]
    for step in report.steps:
        delta = step.terminals_after - step.terminals_before
        lines.append(
            f"{step.workers:7d}  "
            f"{'yes' if step.swapped else 'no':5s}  "
            f"{step.samples_per_second:6.3f}  "
            f"{delta:5d}  "
            f"{'yes' if step.reliable else 'no'}"
        )
    return "\n".join(lines)

"""Post-run export of pipeline artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from dr_queues import EventKind, JobEnvelope, MongoEventSink, filter_run_events
from dr_queues.manifest import manifest_path, load_run_manifest

from dr_code.datasets.export import write_attempts
from dr_code.models.attempts import AttemptRecord
from dr_code.models.outcomes import ParseOutcome, TestOutcome


@dataclass(frozen=True)
class RunExportPaths:
    """Paths written by export_run_artifacts."""

    run_dir: Path
    attempts: Path
    parse_jsonl: Path
    test_jsonl: Path
    manifest: Path


def export_run_artifacts(
    *,
    run_id: str,
    attempts: list[AttemptRecord],
    mongo_sink: MongoEventSink | None = None,
    output_root: Path | str = Path("exports/runs"),
) -> RunExportPaths:
    """Write attempts, parse/test JSONL, and manifest copy for a run."""
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    attempts_path = run_dir / "attempts.parquet"
    parse_path = run_dir / "parse.jsonl"
    test_path = run_dir / "test.jsonl"
    manifest_out = run_dir / "manifest.json"

    write_attempts(attempts, attempts_path)

    sink = mongo_sink or MongoEventSink()
    owns_sink = mongo_sink is None
    try:
        events = filter_run_events(sink.read_by_run_id(run_id), run_id)
        terminals = [event for event in events if event.event == EventKind.TERMINAL]
        parse_outcomes, test_outcomes = _outcomes_from_terminals(terminals)
        _write_outcomes_jsonl(parse_path, parse_outcomes)
        _write_outcomes_jsonl(test_path, test_outcomes)
    finally:
        if owns_sink:
            sink.close()

    source_manifest = manifest_path(run_id)
    if source_manifest.is_file():
        shutil.copy2(source_manifest, manifest_out)

    return RunExportPaths(
        run_dir=run_dir,
        attempts=attempts_path,
        parse_jsonl=parse_path,
        test_jsonl=test_path,
        manifest=manifest_out,
    )


def _outcomes_from_terminals(
    terminals: list,
) -> tuple[list[ParseOutcome], list[TestOutcome]]:
    parse_outcomes: list[ParseOutcome] = []
    test_outcomes: list[TestOutcome] = []
    for event in terminals:
        job = JobEnvelope.model_validate(event.payload)
        parse_raw = job.step_records.get("parse")
        test_raw = job.step_records.get("test")
        if parse_raw is not None:
            parse_outcomes.append(ParseOutcome.model_validate(parse_raw))
        if test_raw is not None:
            test_outcomes.append(TestOutcome.model_validate(test_raw))
    return parse_outcomes, test_outcomes


def _write_outcomes_jsonl(path: Path, outcomes: list[ParseOutcome | TestOutcome]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(outcome.model_dump_json())
            handle.write("\n")

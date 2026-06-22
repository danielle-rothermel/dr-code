"""dr-queues handlers for parse and test stages."""

from __future__ import annotations

import traceback

from dr_queues import HandlerRegistry, JobEnvelope

from dr_code.models.outcomes import ParseOutcome, TestOutcome
from dr_code.parsing.adapter import parse_attempt as parse_attempt_record
from dr_code.pipeline.jobs import attempt_from_job
from dr_code.pipeline.mongo import EvalResultsSink
from dr_code.testing.adapter import missing_parse_outcome, test_parsed_sample

registry = HandlerRegistry()

_eval_sink: EvalResultsSink | None = None


def _eval_sink_instance() -> EvalResultsSink:
    global _eval_sink
    if _eval_sink is None:
        _eval_sink = EvalResultsSink()
    return _eval_sink


def _parse_outcome_from_job(job: JobEnvelope) -> ParseOutcome | None:
    raw = job.step_records.get("parse")
    if raw is None:
        return None
    return ParseOutcome.model_validate(raw)


@registry.register("parse_attempt")
def parse_attempt_handler(job: JobEnvelope) -> JobEnvelope:
    """Parse raw_output via code-eval and record ParseOutcome."""
    record = attempt_from_job(job)
    try:
        outcome = parse_attempt_record(record)
    except Exception as exc:
        outcome = ParseOutcome(
            sample_id=record.sample_id,
            run_id=record.run_id,
            task_id=record.task_id,
            parse_success=False,
            skip_reason=f"parse_handler_error: {type(exc).__name__}: {exc}",
        )
    if record.run_id is not None and outcome.run_id is None:
        outcome = outcome.model_copy(update={"run_id": record.run_id})
    job.step_records["parse"] = outcome.model_dump(mode="json")
    job.step_outputs["parse"] = {
        "parse_success": outcome.parse_success,
        "sample_id": outcome.sample_id,
        "task_id": outcome.task_id,
    }
    return job


@registry.register("run_tests")
def run_tests_handler(job: JobEnvelope) -> JobEnvelope:
    """Run HumanEval+ local fork tests or emit explicit skip/error outcome."""
    record = attempt_from_job(job)
    parse_outcome = _parse_outcome_from_job(job)
    if parse_outcome is None:
        outcome = missing_parse_outcome(record)
    else:
        try:
            outcome = test_parsed_sample(record, parse_outcome)
        except Exception as exc:
            outcome = TestOutcome(
                sample_id=record.sample_id,
                run_id=parse_outcome.run_id or record.run_id,
                task_id=record.task_id,
                parse_success=parse_outcome.parse_success,
                outcome_kind="internal_error",
                skipped=False,
                tests_ran=False,
                internal_error=(
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                ),
            )

    job.step_records["test"] = outcome.model_dump(mode="json")
    job.step_outputs["test"] = {
        "outcome_kind": outcome.outcome_kind,
        "sample_id": outcome.sample_id,
        "task_id": outcome.task_id,
        "all_tests_passed": outcome.all_tests_passed,
    }

    _eval_sink_instance().upsert_test_outcome(
        outcome,
        provenance_source=record.provenance.source.value,
        occurrence_count=record.provenance.occurrence_count,
    )
    return job

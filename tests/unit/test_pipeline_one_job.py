"""Unit tests for one-job generated-code evaluation."""

from __future__ import annotations

from dr_code.datasets.humaneval_loader import get_task
from dr_code.models.attempts import AttemptRecord, AttemptSource
from dr_code.models.outcomes import ParseOutcome, TestOutcome
from dr_code.pipeline import one_job
from dr_code.pipeline.one_job import OneJobEvalRequest, evaluate_generated_code


def test_evaluate_generated_code_wires_attempt_parse_and_test(monkeypatch) -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)

    def fake_parse_attempt(attempt: AttemptRecord) -> ParseOutcome:
        return ParseOutcome(
            sample_id=attempt.sample_id,
            run_id=attempt.run_id,
            task_id=attempt.task_id,
            parse_success=True,
            extracted_code=attempt.raw_output,
        )

    def fake_test_parsed_sample(
        attempt: AttemptRecord,
        parse_outcome: ParseOutcome,
        **kwargs: object,
    ) -> TestOutcome:
        assert kwargs["task"] == task
        return TestOutcome(
            sample_id=attempt.sample_id,
            run_id=parse_outcome.run_id,
            task_id=attempt.task_id,
            parse_success=parse_outcome.parse_success,
            outcome_kind="tested",
            tests_ran=True,
            all_tests_passed=True,
            test_pass_rate=1.0,
            selected_function_name="candidate",
        )

    monkeypatch.setattr(one_job, "parse_attempt", fake_parse_attempt)
    monkeypatch.setattr(one_job, "test_parsed_sample", fake_test_parsed_sample)

    result = evaluate_generated_code(
        OneJobEvalRequest(
            run_id="run-1",
            task_id=task.task_id,
            decoder_input="description",
            raw_output="def candidate(numbers, threshold):\n    return False\n",
            decode_model="demo/model",
            metadata={"candidate_id": "cand-1"},
        ),
        task=task,
    )

    assert result.attempt.run_id == "run-1"
    assert result.attempt.task_id == task.task_id
    assert result.attempt.provenance.source is AttemptSource.BOTTLENECK
    assert result.attempt.provenance.model == "demo/model"
    assert result.attempt.provenance.extra["candidate_id"] == "cand-1"
    assert result.parse_outcome.parse_success is True
    assert result.test_outcome.selected_function_name == "candidate"

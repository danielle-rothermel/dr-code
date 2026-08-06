from __future__ import annotations

import ast
from collections.abc import Sequence

import pytest
from dr_exec import Executor, ExecutorFailure
from pydantic import TypeAdapter, ValidationError

import dr_code.humaneval.scoring as scoring_module
from _executor_stubs import raising_executor
from _humaneval_builders import (
    _PARTIAL_RUNNER_PASSED_CASE_0,
    _stub_executor,
    _task,
)
from dr_code.humaneval import EvaluationCaseStatus, HumanEvalTask
from dr_code.humaneval.parsed_tests import HumanEvalTestCaseKind
from dr_code.humaneval.scoring import (
    CompletedScore,
    HarnessFailure,
    HumanEvalSubmissionRequest,
    HumanEvalSubmissionScore,
    SubmissionOutcome,
    evaluation_outcome,
    score_humaneval_submission,
    score_humaneval_submissions_batch,
)
from dr_code.metrics.engine.execution import (
    ExecutionCache,
    ExecutionOutcome,
    ExecutionRequest,
    InMemoryExecutionCache,
)
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationTaskResult,
)


def _partial_evaluation_result(task: HumanEvalTask) -> EvaluationTaskResult:
    return EvaluationTaskResult(
        task_id=task.task_id,
        entry_point=task.entry_point,
        function_names=[task.entry_point],
        total_cases=2,
        results=[
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id="case_0",
                function_name=task.entry_point,
                status=EvaluationCaseStatus.PASSED,
                test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            ),
        ],
    )


def test_evaluation_task_result_round_trips_through_its_own_dump() -> None:
    evaluation = _partial_evaluation_result(_task())
    payload = evaluation.model_dump()

    assert not {
        "best_function_name",
        "failures",
        "coverage_complete",
        "passed",
        "status_counts",
    } & set(payload)

    restored = EvaluationTaskResult.model_validate(payload)

    assert restored == evaluation
    assert restored.best_function_name == evaluation.best_function_name
    assert restored.coverage_complete == evaluation.coverage_complete
    assert restored.passed == evaluation.passed
    assert restored.status_counts == evaluation.status_counts
    assert restored.failures == evaluation.failures


def test_evaluation_task_result_round_trips_through_json() -> None:
    evaluation = _partial_evaluation_result(_task())

    restored = EvaluationTaskResult.model_validate_json(
        evaluation.model_dump_json()
    )

    assert restored == evaluation


def test_evaluation_task_summary_still_carries_the_readings() -> None:
    evaluation = _partial_evaluation_result(_task())
    summary = evaluation.to_summary()
    payload = summary.model_dump()

    assert payload["best_function_name"] == evaluation.best_function_name
    assert payload["passed"] == evaluation.passed
    assert payload["status_counts"] == evaluation.status_counts
    assert payload["failure_count"] == len(evaluation.failures)


def test_evaluation_outcome_reports_incomplete_for_partial_coverage() -> None:
    evaluation = _partial_evaluation_result(_task())

    assert evaluation.coverage_complete is False
    assert evaluation.failures == []
    assert evaluation_outcome(evaluation) is (
        SubmissionOutcome.EVALUATION_INCOMPLETE
    )


def test_evaluation_outcome_reports_tests_failed_when_case_fails() -> None:
    task = _task()
    evaluation = EvaluationTaskResult(
        task_id=task.task_id,
        entry_point=task.entry_point,
        function_names=[task.entry_point],
        total_cases=2,
        results=[
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id="case_0",
                function_name=task.entry_point,
                status=EvaluationCaseStatus.FAILED,
                message="bad",
                test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            ),
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id="case_1",
                function_name=task.entry_point,
                status=EvaluationCaseStatus.PASSED,
                test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            ),
        ],
    )

    assert evaluation_outcome(evaluation) is SubmissionOutcome.TESTS_FAILED


def test_score_humaneval_submission_reports_incomplete_runner_output() -> None:
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        executor=_stub_executor(stdout=_PARTIAL_RUNNER_PASSED_CASE_0),
    )

    assert isinstance(result, CompletedScore)
    assert result.outcome is SubmissionOutcome.EVALUATION_INCOMPLETE
    assert result.score == 0.0
    assert result.evaluation is not None
    assert result.evaluation.failures == []
    assert result.evaluation.coverage_complete is False


def test_batch_scorer_runs_all_planned_requests_in_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run_requests = scoring_module.run_requests
    calls: list[tuple[ExecutionRequest, ...]] = []
    cache = InMemoryExecutionCache()

    def recording_run_requests(
        requests: Sequence[ExecutionRequest],
        *,
        executor: Executor | None,
        cache: ExecutionCache,
    ) -> dict[ExecutionRequest, ExecutionOutcome]:
        calls.append(tuple(requests))
        assert cache is expected_cache
        return original_run_requests(
            requests,
            executor=executor,
            cache=cache,
        )

    expected_cache = cache
    monkeypatch.setattr(
        scoring_module,
        "run_requests",
        recording_run_requests,
    )
    raw_submissions = (
        " \n\t ",
        "def add_one(x):\n    return x + 1\n",
        "def candidate(x):\n    return x + 1\n",
    )

    results = score_humaneval_submissions_batch(
        tuple(
            HumanEvalSubmissionRequest(
                raw_submission=raw_submission,
                task=_task(),
            )
            for raw_submission in raw_submissions
        ),
        executor=_stub_executor(stdout=_PARTIAL_RUNNER_PASSED_CASE_0),
        execution_cache=cache,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert tuple(result.raw_submission for result in results) == (
        raw_submissions
    )
    assert all(isinstance(result, CompletedScore) for result in results)
    assert results[0].outcome is SubmissionOutcome.EMPTY_SUBMISSION


def test_batch_scorer_calls_execution_planning_once_for_empty_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[ExecutionRequest, ...]] = []

    def recording_run_requests(
        requests: Sequence[ExecutionRequest],
        *,
        executor: Executor | None,
        cache: ExecutionCache,
    ) -> dict[ExecutionRequest, ExecutionOutcome]:
        del executor, cache
        calls.append(tuple(requests))
        return {}

    monkeypatch.setattr(
        scoring_module,
        "run_requests",
        recording_run_requests,
    )

    assert score_humaneval_submissions_batch(()) == ()
    assert calls == [()]


def test_batch_scorer_preserves_unplanned_results_on_executor_failure() -> (
    None
):
    results = score_humaneval_submissions_batch(
        (
            HumanEvalSubmissionRequest(
                raw_submission=" \n\t ",
                task=_task(),
            ),
            HumanEvalSubmissionRequest(
                raw_submission="def add_one(x):\n    return x + 1\n",
                task=_task(),
            ),
        ),
        executor=raising_executor(ExecutorFailure("executor unavailable")),
    )

    assert isinstance(results[0], CompletedScore)
    assert results[0].outcome is SubmissionOutcome.EMPTY_SUBMISSION
    assert isinstance(results[1], HarnessFailure)
    assert results[1].cause.exception_type == "ExecutorFailure"


def test_score_humaneval_submission_returns_harness_failure() -> None:
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        executor=_stub_executor(stdout="not-json"),
    )

    assert isinstance(result, HarnessFailure)
    assert result.kind == "harness_failure"
    assert result.failure_class == "unknown"
    assert result.cause.exception_type == "JSONDecodeError"
    assert result.evaluation is not None
    assert result.evaluation.results[0].elapsed_seconds is not None


@pytest.mark.parametrize(
    ("runner_stdout", "expected_type"),
    (
        (_PARTIAL_RUNNER_PASSED_CASE_0, CompletedScore),
        ("not-json", HarnessFailure),
    ),
    ids=("completed", "harness-failure"),
)
def test_submission_score_variants_round_trip_with_accepted_code(
    runner_stdout: str,
    expected_type: type[CompletedScore] | type[HarnessFailure],
) -> None:
    adapter = TypeAdapter(HumanEvalSubmissionScore)
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        executor=_stub_executor(stdout=runner_stdout),
    )

    assert isinstance(result, expected_type)
    assert result.extraction is not None
    assert result.extraction.accepted_code is not None
    assert result.extraction.accepted_tree is not None

    payload = adapter.dump_json(result)
    restored = adapter.validate_json(payload)

    assert b"accepted_tree" not in payload
    assert restored == result
    assert restored.extraction is not None
    assert restored.extraction.accepted_tree is not None
    assert ast.dump(restored.extraction.accepted_tree) == ast.dump(
        result.extraction.accepted_tree
    )


@pytest.mark.parametrize(
    ("payload", "error_type", "error_context"),
    (
        (
            {"raw_submission": "def add_one(x):\n    return x + 1\n"},
            "union_tag_not_found",
            {"discriminator": "'kind'"},
        ),
        (
            {"kind": "pending"},
            "union_tag_invalid",
            {
                "discriminator": "'kind'",
                "tag": "pending",
                "expected_tags": "'completed', 'harness_failure'",
            },
        ),
    ),
    ids=("missing", "unknown"),
)
def test_submission_score_rejects_invalid_discriminator(
    payload: dict[str, object],
    error_type: str,
    error_context: dict[str, str],
) -> None:
    adapter = TypeAdapter(HumanEvalSubmissionScore)

    with pytest.raises(ValidationError) as exc_info:
        adapter.validate_python(payload)

    errors = exc_info.value.errors(include_url=False)
    assert len(errors) == 1
    assert errors[0]["type"] == error_type
    assert errors[0]["loc"] == ()
    assert errors[0]["ctx"] == error_context


def test_score_humaneval_submission_reports_executor_breakage() -> None:
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        executor=raising_executor(
            ExecutorFailure("the executor is unavailable")
        ),
    )

    assert isinstance(result, HarnessFailure)
    assert result.kind == "harness_failure"
    assert result.cause.exception_type == "ExecutorFailure"


def test_score_humaneval_submission_without_executor_is_harness_failure() -> (
    None
):
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
    )

    assert isinstance(result, HarnessFailure)
    assert result.cause.exception_type == "ExecutorFailure"
    assert "no executor" in result.cause.message


def test_score_humaneval_submission_reports_empty_submission() -> None:
    result = score_humaneval_submission(
        raw_submission=" \n\t ",
        task=_task(),
    )

    assert isinstance(result, CompletedScore)
    assert result.kind == "completed"
    assert result.raw_submission == " \n\t "
    assert result.extraction.raw_submission == " \n\t "
    assert result.outcome is SubmissionOutcome.EMPTY_SUBMISSION
    assert result.evaluation is None


def test_scoring_reports_extraction_failure_without_top_level_functions() -> (
    None
):
    result = score_humaneval_submission(
        raw_submission="x = 1\n",
        task=_task(),
    )

    assert result.outcome is SubmissionOutcome.EXTRACTION_FAILED


def test_score_humaneval_submission_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="raw_submission must be str"):
        score_humaneval_submission(
            raw_submission={"code": "def add_one(x):\n    return x + 1\n"},  # type: ignore[arg-type]
            task=_task(),
        )

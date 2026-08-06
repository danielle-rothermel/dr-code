from __future__ import annotations

import pytest

from _executor_stubs import (
    completed_execution,
    output_limit_executor,
    timeout_executor,
)
from _humaneval_builders import (
    _PARTIAL_RUNNER_PASSED_CASE_0,
    _input_result_test,
    _stub_executor,
    _task,
)
from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    ExecutionJob,
    FakeExecutor,
    FiniteDurationLimit,
)
from dr_code.humaneval import EvaluationCaseStatus, HumanEvalTask
from dr_code.humaneval.parsed_tests import HumanEvalTestCaseKind
from dr_code.humaneval.runner import (
    evaluate_humaneval_code,
    require_parsed_tests,
    run_subprocess_batch,
)
from dr_code.humaneval.scoring import (
    SubmissionOutcome,
    evaluation_outcome,
)
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationHarnessError,
)


def test_evaluation_passes_when_best_function_passes(
    local_executor: FakeExecutor,
) -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=(
            "def broken_helper(x):\n"
            "    return x\n"
            "\n"
            "def add_one(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
        executor=local_executor,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is True
    assert result.status_counts == {"passed": 2}
    assert result.failures == []
    summary = result.to_summary()
    assert summary.passed is True
    assert summary.best_function_name == "add_one"
    assert summary.failure_count == 0


def test_evaluation_prefers_entry_point_when_pass_counts_tie(
    local_executor: FakeExecutor,
) -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    return x + 1\n"
            "\n"
            "def also_add_one(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
        executor=local_executor,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is True


def test_evaluation_fails_when_best_function_does_not_pass_all_cases(
    local_executor: FakeExecutor,
) -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=(
            "def broken_helper(x):\n"
            "    return x\n"
            "\n"
            "def add_one(x):\n"
            "    return x + 1 if x == 1 else x\n"
        ),
        timeout_seconds=2.0,
        executor=local_executor,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is False
    assert result.status_counts == {"passed": 1, "failed": 1}


def test_evaluation_uses_highest_pass_count(
    local_executor: FakeExecutor,
) -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    return x\n"
            "\n"
            "def helper(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
        executor=local_executor,
    )

    assert result.best_function_name == "helper"
    assert result.passed is True
    assert result.status_counts == {"passed": 2}


def test_evaluate_humaneval_code_reports_timeout_per_case() -> None:
    candidate_code = "def add_one(x):\n    return x + 1\n"
    timeout_seconds = 0.2
    forwarded_jobs: list[ExecutionJob] = []

    def timeout_responder(job: ExecutionJob, cancellation: object) -> object:
        del cancellation
        forwarded_jobs.append(job)
        return completed_execution(
            job,
            outcome=BudgetExceededOutcome(axis=BudgetAxis.WALL_TIME),
        )

    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=candidate_code,
        timeout_seconds=timeout_seconds,
        executor=FakeExecutor(responder=timeout_responder),
    )

    assert result.passed is False
    assert result.status_counts == {"timeout": 2}
    assert result.results == [
        EvaluationCaseResult(
            task_id="HumanEval/fixture",
            case_id="case_0",
            function_name="add_one",
            status=EvaluationCaseStatus.TIMEOUT,
            message="Batch timed out after 0.2 seconds",
            test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            input_repr="[1]",
            expected_output_repr="2",
            elapsed_seconds=0.2,
            timeout_seconds=0.2,
        ),
        EvaluationCaseResult(
            task_id="HumanEval/fixture",
            case_id="case_1",
            function_name="add_one",
            status=EvaluationCaseStatus.TIMEOUT,
            message="Batch timed out after 0.2 seconds",
            test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            input_repr="[2]",
            expected_output_repr="3",
            elapsed_seconds=0.2,
            timeout_seconds=0.2,
        ),
    ]
    assert len(forwarded_jobs) == 1
    assert forwarded_jobs[0].target.request.payload == {
        "task_id": "HumanEval/fixture",
        "candidate_code": candidate_code,
        "support_code": "",
        "function_name": "add_one",
        "test_type": "input_result",
        "checks": [
            {
                "case_id": "case_0",
                "code": "assertion(candidate(*[1]), 2, 0.0)",
                "input_repr": "[1]",
                "expected_output_repr": "2",
                "expected_output_expr": None,
                "actual_output_expr": "candidate(*[1])",
            },
            {
                "case_id": "case_1",
                "code": "assertion(candidate(*[2]), 3, 0.0)",
                "input_repr": "[2]",
                "expected_output_repr": "3",
                "expected_output_expr": None,
                "actual_output_expr": "candidate(*[2])",
            },
        ],
    }
    wall_time = forwarded_jobs[0].budgets.wall_time
    assert isinstance(wall_time, FiniteDurationLimit)
    assert wall_time.max_ns == 200_000_000
    assert evaluation_outcome(result) is SubmissionOutcome.TIMED_OUT


def test_run_subprocess_batch_raises_for_malformed_runner_output() -> None:
    executor = _stub_executor(
        stdout=(
            '[{"case_id": "case_0", "status": "passed", "message": ""}, '
            '{"case_id": "case_1", "status": "nonsense"}]'
        ),
    )

    with pytest.raises(EvaluationHarnessError) as exc_info:
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
            executor=executor,
        )

    results = exc_info.value.case_results
    by_case_id = {result.case_id: result for result in results}
    assert set(by_case_id) == {"case_0", "case_1"}
    assert by_case_id["case_0"].status is EvaluationCaseStatus.PASSED
    assert by_case_id["case_1"].status is EvaluationCaseStatus.ERROR
    assert "Invalid runner output" in by_case_id["case_1"].message


def test_evaluation_incomplete_when_runner_returns_partial_results() -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        timeout_seconds=2.0,
        executor=_stub_executor(stdout=_PARTIAL_RUNNER_PASSED_CASE_0),
    )

    assert result.passed is False
    assert result.coverage_complete is False
    assert result.failures == []
    assert result.status_counts == {"passed": 1}


def test_run_subprocess_batch_scores_candidate_kill_as_error() -> None:
    results = run_subprocess_batch(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        function_name="add_one",
        timeout_seconds=2.0,
        executor=_stub_executor(stdout="", stderr="killed", returncode=-9),
    )

    assert len(results) == 2
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "ExecutionKilledError" in results[0].message
    assert "killed" in results[0].message


def test_run_subprocess_batch_scores_output_limit_as_candidate_error() -> None:
    results = run_subprocess_batch(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        function_name="add_one",
        timeout_seconds=2.0,
        executor=output_limit_executor(),
    )

    assert len(results) == 2
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "ExecutionOutputLimitError" in results[0].message


def test_run_subprocess_batch_scores_wall_time_budget_as_timeout() -> None:
    results = run_subprocess_batch(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        function_name="add_one",
        timeout_seconds=2.0,
        executor=timeout_executor(),
    )

    assert len(results) == 2
    assert all(
        result.status is EvaluationCaseStatus.TIMEOUT for result in results
    )


@pytest.mark.parametrize(
    "runner_stdout",
    [
        '[{"case_id": "case_0", "status": "passed", "message": ""},'
        ' {"case_id": "case_0", "status": "passed", "message": ""}]',
        '[{"case_id": "case_99", "status": "passed", "message": ""}]',
        '[{"case_id": "case_0", "status": "passed", "message": ""},'
        ' {"case_id": "case_1", "status": "passed", "message": ""},'
        ' {"case_id": "case_2", "status": "passed", "message": ""}]',
    ],
    ids=("duplicate", "unknown", "more_rows_than_cases"),
)
def test_run_subprocess_batch_rejects_invalid_case_ids(
    runner_stdout: str,
) -> None:
    with pytest.raises(
        EvaluationHarnessError,
        match="duplicate or unknown case ids",
    ):
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
            executor=_stub_executor(stdout=runner_stdout),
        )


def test_candidate_module_level_sys_exit_is_scored(
    local_executor: FakeExecutor,
) -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=(
            "import sys\nsys.exit(5)\ndef add_one(x):\n    return x + 1\n"
        ),
        timeout_seconds=2.0,
        executor=local_executor,
    )

    assert result.passed is False
    assert result.status_counts == {"error": 2}


def test_run_subprocess_batch_raises_for_nonzero_returncode() -> None:
    executor = _stub_executor(stdout="", stderr="runner crashed", returncode=1)

    with pytest.raises(EvaluationHarnessError) as exc_info:
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
            executor=executor,
        )

    results = exc_info.value.case_results
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "runner crashed" in results[0].message


def test_run_subprocess_batch_raises_for_invalid_json() -> None:
    with pytest.raises(EvaluationHarnessError) as exc_info:
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
            executor=_stub_executor(stdout="not-json"),
        )

    results = exc_info.value.case_results
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "Could not decode runner output" in results[0].message


def test_run_subprocess_batch_raises_for_non_list_json() -> None:
    with pytest.raises(EvaluationHarnessError) as exc_info:
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
            executor=_stub_executor(stdout='{"not": "a list"}'),
        )

    results = exc_info.value.case_results
    assert "expected a JSON list" in results[0].message


def test_run_subprocess_batch_fallback_case_id_is_harness_detail() -> None:
    executor = _stub_executor(stdout='[{"status": "passed", "message": ""}]')

    with pytest.raises(EvaluationHarnessError) as exc_info:
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
            executor=executor,
        )

    results = exc_info.value.case_results
    assert results[0].case_id == "case_0"


def test_require_parsed_tests_raises_when_missing() -> None:
    task = HumanEvalTask.model_construct(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=_input_result_test(),
        parsed_tests=None,
    )

    with pytest.raises(
        ValueError,
        match=r"HumanEvalTask\.parsed_tests is required",
    ):
        require_parsed_tests(task)

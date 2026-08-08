from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable

import pytest

from _executor_stubs import (
    raising_executor,
    scripted_executor,
)
from dr_exec import Executor, ExecutorFailure, FakeExecutor
from dr_code.humaneval.metric_operator import CodeTest, CodeTestSettings
from dr_code.humaneval.runner import (
    build_humaneval_batch_request,
    evaluate_humaneval_code,
    top_level_function_names,
)
from dr_code.humaneval.task import (
    EvaluationCaseStatus,
    HumanEvalTask,
    select_best_function_name,
)
from dr_code.metrics import (
    MetricName,
    MetricQuestion,
    MetricsDefinition,
    extract_metrics,
)
from dr_code.metrics.records import (
    MeasuredRecord,
    MetricRecord,
    OperatorFailureRecord,
)
from dr_code.trace import CodeArtifact, JsonArtifact, Trace, external_trace

_INPUT_RESULT_TEST = (
    "def check(candidate):\n"
    "    inputs = [(1,), (2,)]\n"
    "    results = [2, 3]\n"
    "    for inp, expected in zip(inputs, results):\n"
    "        assertion(candidate(*inp), expected)\n"
)


@pytest.fixture
def task() -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=_INPUT_RESULT_TEST,
    )


@pytest.fixture
def good_submission() -> str:
    return "def add_one(x):\n    return x + 1\n"


@pytest.fixture
def failing_submission() -> str:
    return "def add_one(x):\n    return x - 1\n"


def _code_test_trace(candidate_code: str, task: HumanEvalTask) -> Trace:
    code = CodeArtifact(source=candidate_code)
    return external_trace(
        {
            "input": code,
            "output": code,
            "task": JsonArtifact(payload=task.model_dump(mode="json")),
        }
    )


def _code_test_definition(
    timeout_seconds: float = 5.0,
) -> MetricsDefinition:
    return MetricsDefinition(
        definition_id="parity",
        version="1",
        questions=(
            MetricQuestion(
                metric=MetricName.CODE_TEST,
                on="input",
                settings={"timeout_seconds": timeout_seconds},
            ),
        ),
    )


def _extract_code_test(
    trace: Trace,
    *,
    executor: Executor,
) -> MetricRecord:
    records = extract_metrics(
        _code_test_definition(),
        trace,
        executor=executor,
    )
    assert len(records) == 1
    return records[0]


def _facts(record: MetricRecord) -> dict[str, object]:
    assert isinstance(record, MeasuredRecord), record
    return {fact.name: fact.value for fact in record.facts}


def _value(record: MetricRecord, key: str) -> object:
    return _facts(record)[key]


def _scripted_executor(
    *,
    stdout: str = "[]",
    stderr: str = "",
    returncode: int = 0,
) -> FakeExecutor:
    return scripted_executor(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


@pytest.fixture
def scripted() -> Callable[..., FakeExecutor]:
    return _scripted_executor


@pytest.fixture
def raising() -> Callable[[BaseException], FakeExecutor]:
    return raising_executor


def _partial_pass_runner_output(
    *,
    passed: tuple[str, ...] = ("case_0",),
    case_ids: tuple[str, ...] = ("case_0", "case_1"),
) -> str:
    payload = []
    for case_id in case_ids:
        status = (
            EvaluationCaseStatus.PASSED.value
            if case_id in passed
            else EvaluationCaseStatus.FAILED.value
        )
        payload.append(
            {
                "case_id": case_id,
                "status": status,
                "message": "",
                "input_repr": "[1]",
                "expected_output_repr": "2",
                "actual_output_repr": "2",
                "elapsed_seconds": 0.0,
                "timeout_seconds": None,
            }
        )
    return json.dumps({"kind": "case_results", "results": payload})


def test_code_test_imports_in_clean_interpreter() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from dr_code.humaneval.metric_operator import CodeTest",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_code_test_passing_counts_match_oracle(
    task: HumanEvalTask,
    good_submission: str,
    local_executor: FakeExecutor,
) -> None:
    oracle = evaluate_humaneval_code(
        task=task,
        candidate_code=good_submission,
        timeout_seconds=5.0,
        executor=local_executor,
    )
    record = _extract_code_test(
        _code_test_trace(good_submission, task),
        executor=local_executor,
    )

    assert _value(record, "total_cases") == oracle.total_cases
    assert _value(record, "passed_count") == oracle.status_counts.get(
        "passed", 0
    )
    assert _value(record, "failed_count") == oracle.status_counts.get(
        "failed", 0
    )
    assert _value(record, "error_count") == oracle.status_counts.get(
        "error", 0
    )
    assert _value(record, "timeout_count") == oracle.status_counts.get(
        "timeout", 0
    )
    assert _value(record, "coverage_complete") == oracle.coverage_complete
    assert _value(record, "function_count") == len(oracle.function_names)
    assert _value(record, "best_function_name") == oracle.best_function_name


def test_code_test_failing_counts_match_oracle(
    task: HumanEvalTask,
    failing_submission: str,
    local_executor: FakeExecutor,
) -> None:
    oracle = evaluate_humaneval_code(
        task=task,
        candidate_code=failing_submission,
        timeout_seconds=5.0,
        executor=local_executor,
    )
    record = _extract_code_test(
        _code_test_trace(failing_submission, task),
        executor=local_executor,
    )

    assert _value(record, "passed_count") == oracle.status_counts.get(
        "passed", 0
    )
    assert _value(record, "failed_count") == oracle.status_counts.get(
        "failed", 0
    )


def test_code_test_kill_attributed_to_candidate(
    task: HumanEvalTask,
    good_submission: str,
    scripted: Callable[..., FakeExecutor],
) -> None:
    record = _extract_code_test(
        _code_test_trace(good_submission, task),
        executor=scripted(returncode=-9, stdout="", stderr="killed"),
    )

    assert _value(record, "error_count") == _value(record, "total_cases")
    assert _value(record, "passed_count") == 0


def test_code_test_nonzero_runner_exit_is_operator_failure(
    task: HumanEvalTask,
    good_submission: str,
    scripted: Callable[..., FakeExecutor],
) -> None:
    record = _extract_code_test(
        _code_test_trace(good_submission, task),
        executor=scripted(
            returncode=5,
            stdout="",
            stderr="boom",
        ),
    )

    assert isinstance(record, OperatorFailureRecord)
    assert record.failure.failure_type == "EvaluationHarnessError"
    assert "runner subprocess exited nonzero" in record.failure.failure_message


def test_code_test_malformed_stdout_is_operator_failure(
    task: HumanEvalTask,
    good_submission: str,
    scripted: Callable[..., FakeExecutor],
) -> None:
    for bad_stdout in (
        "this is not json{",
        '{"not": "a list"}',
        '[{"case_id": "case_0"}]',
        '[{"case_id": "ghost", "status": "passed"}]',
    ):
        record = _extract_code_test(
            _code_test_trace(good_submission, task),
            executor=scripted(stdout=bad_stdout),
        )

        assert isinstance(record, OperatorFailureRecord), bad_stdout
        assert record.failure.failure_type == "EvaluationHarnessError"


def test_code_test_support_initialization_failure_is_operator_failure(
    good_submission: str,
    local_executor: FakeExecutor,
) -> None:
    task = HumanEvalTask(
        task_id="HumanEval/broken-support",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=("raise RuntimeError('support broke')\n" + _INPUT_RESULT_TEST),
    )

    record = _extract_code_test(
        _code_test_trace(good_submission, task),
        executor=local_executor,
    )

    assert isinstance(record, OperatorFailureRecord)
    assert record.failure.failure_type == "EvaluationHarnessError"
    assert "RuntimeError: support broke" in record.failure.failure_message


def test_code_test_candidate_setup_failure_remains_measured(
    task: HumanEvalTask,
    local_executor: FakeExecutor,
) -> None:
    record = _extract_code_test(
        _code_test_trace(
            "raise RuntimeError('candidate broke')\n"
            "def add_one(x):\n    return x + 1\n",
            task,
        ),
        executor=local_executor,
    )

    assert isinstance(record, MeasuredRecord)
    assert _value(record, "error_count") == _value(record, "total_cases")
    assert _value(record, "passed_count") == 0


def test_code_test_executor_failure_still_propagates(
    task: HumanEvalTask,
    good_submission: str,
    raising: Callable[[BaseException], FakeExecutor],
) -> None:
    with pytest.raises(ExecutorFailure):
        _extract_code_test(
            _code_test_trace(good_submission, task),
            executor=raising(ExecutorFailure("boundary broke")),
        )


def test_code_test_requests_are_the_canonical_batch_request(
    task: HumanEvalTask,
) -> None:
    candidate = (
        "def add_one(x):\n    return x + 1\ndef decoy(x):\n    return x - 1\n"
    )
    timeout_seconds = 5.0
    operator = CodeTest(CodeTestSettings(timeout_seconds=timeout_seconds))
    requests = operator.execution_requests(
        CodeArtifact(source=candidate),
        {"task": JsonArtifact(payload=task.model_dump(mode="json"))},
    )

    function_names = ["add_one", "decoy"]
    assert len(requests) == len(function_names)
    for request, function_name in zip(requests, function_names, strict=True):
        canonical = build_humaneval_batch_request(
            task=task,
            candidate_code=candidate,
            function_name=function_name,
            timeout_seconds=timeout_seconds,
        )
        assert request.input_json == canonical.input_json, function_name
        assert request.source == canonical.source, function_name
        assert request.timeout_seconds == canonical.timeout_seconds


def test_code_test_function_names_come_from_the_shared_rule(
    task: HumanEvalTask,
) -> None:
    candidate = (
        "def add_one(x):\n"
        "    return x + 1\n"
        "async def fetch(x):\n"
        "    return x\n"
        "def add_one(x):\n"
        "    return x + 2\n"
        "class Ignored:\n"
        "    def method(self):\n"
        "        return 0\n"
    )
    operator = CodeTest(CodeTestSettings())
    requests = operator.execution_requests(
        CodeArtifact(source=candidate),
        {"task": JsonArtifact(payload=task.model_dump(mode="json"))},
    )

    names = top_level_function_names(candidate)
    assert names == ["add_one", "fetch"]
    assert len(requests) == len(names)


def test_code_test_selection_is_the_task_selection_rule(
    task: HumanEvalTask,
    local_executor: FakeExecutor,
) -> None:
    candidate = (
        "def add_one(x):\n    return x - 1\ndef decoy(x):\n    return x + 1\n"
    )
    record = _extract_code_test(
        _code_test_trace(candidate, task),
        executor=local_executor,
    )
    oracle = evaluate_humaneval_code(
        task=task,
        candidate_code=candidate,
        timeout_seconds=5.0,
        executor=local_executor,
    )

    assert _value(record, "best_function_name") == "decoy"
    assert _value(record, "best_function_name") == oracle.best_function_name
    assert oracle.best_function_name == select_best_function_name(
        function_names=oracle.function_names,
        entry_point=task.entry_point,
        results=oracle.results,
    )


def test_code_test_best_function_is_mechanical_max_passes(
    task: HumanEvalTask,
    local_executor: FakeExecutor,
) -> None:
    candidate = (
        "def add_one(x):\n    return x + 1\ndef decoy(x):\n    return x - 1\n"
    )
    record = _extract_code_test(
        _code_test_trace(candidate, task),
        executor=local_executor,
    )

    assert _value(record, "best_function_name") == task.entry_point
    assert _value(record, "function_count") == 2
    assert "score" not in _facts(record)
    assert "outcome" not in _facts(record)


def test_code_test_partial_coverage_is_measured(
    task: HumanEvalTask,
    good_submission: str,
    scripted: Callable[..., FakeExecutor],
) -> None:
    incomplete_output = _partial_pass_runner_output(
        passed=("case_0",),
        case_ids=("case_0",),
    )
    record = _extract_code_test(
        _code_test_trace(good_submission, task),
        executor=scripted(stdout=incomplete_output),
    )

    assert isinstance(record, MeasuredRecord)
    assert _value(record, "passed_count") == 1
    assert _value(record, "failed_count") == 0
    assert _value(record, "coverage_complete") is False


def test_code_test_complete_coverage_with_failure_is_covered(
    task: HumanEvalTask,
    good_submission: str,
    scripted: Callable[..., FakeExecutor],
) -> None:
    complete_with_failure = _partial_pass_runner_output()
    record = _extract_code_test(
        _code_test_trace(good_submission, task),
        executor=scripted(stdout=complete_with_failure),
    )

    assert isinstance(record, MeasuredRecord)
    assert _value(record, "passed_count") == 1
    assert _value(record, "failed_count") == 1
    assert _value(record, "coverage_complete") is True

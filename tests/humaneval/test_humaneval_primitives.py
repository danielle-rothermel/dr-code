from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

import dr_code.humaneval as humaneval
from dr_code.code_analysis import validate_python_source
from dr_code.humaneval.code_extraction import apply_cleaning
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
)
from dr_code.humaneval.compression import (
    CompressionMethod,
    compression_metrics,
)
from dr_code.humaneval.parsed_code import ParsedCode, parse_code
from dr_code.humaneval.parsed_tests import (
    HumanEvalTestCaseKind,
    UnsupportedTestFormatError,
)
from dr_code.humaneval.sampling import sample_human_eval_tasks_from_rows
from dr_code.humaneval.scoring import (
    CompletedScore,
    HarnessFailure,
    SubmissionOutcome,
    evaluation_outcome,
    score_humaneval_submission,
)
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationHarnessError,
    EvaluationCaseStatus,
    EvaluationTaskResult,
    HumanEvalOverride,
    HumanEvalTask,
    apply_human_eval_override,
    evaluate_human_eval_code,
    parse_human_eval_dataset,
    parse_human_eval_tests,
    require_parsed_tests,
    run_subprocess_batch,
)


EXPECTED_HUMANEVAL_PUBLIC_API = {
    "AstMetricsPayload",
    "BEST_EFFORT_HUMANEVAL_PARSER_PROFILE",
    "BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID",
    "CodeExtractionResult",
    "CodeParserProfile",
    "CompletedScore",
    "DEFAULT_HUMANEVAL_SCORING_PROFILE",
    "DEFAULT_HUMANEVAL_TIMEOUT_SECONDS",
    "EvaluationCaseStatus",
    "EvaluationCaseSummary",
    "EvaluationTaskSummary",
    "HUMANEVAL_SCORING_PROFILE_ID",
    "HUMANEVAL_SCORING_PROFILE_VERSION",
    "HarnessFailure",
    "HarnessFailureCause",
    "HumanEvalScoringProfile",
    "HumanEvalSubmissionScore",
    "HumanEvalTask",
    "HumanEvalTaskTestMetricsPayload",
    "HumanEvalTestCaseKind",
    "MetricsPayload",
    "MetricsStagePayload",
    "NodeOutputMetricsSource",
    "PARSER_PROFILE_VERSION",
    "PythonLeakageMetricsPayload",
    "STRICT_FIELD_MARKER_PARSER_PROFILE",
    "STRICT_FIELD_MARKER_PARSER_PROFILE_ID",
    "SampledHumanEvalTask",
    "SubmissionOutcome",
    "TextMetricsPayload",
    "build_metrics_payload",
    "evaluation_aggregate_metrics",
    "extract_code_with_profile",
    "load_human_eval_rows",
    "parse_human_eval_dataset",
    "resolve_humaneval_scoring_profile",
    "resolve_parser_profile",
    "sample_human_eval_tasks",
    "sample_human_eval_tasks_from_rows",
    "score_humaneval_submission",
}


def _task(*, test: str | None = None) -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=test or _input_result_test(),
    )


def _row(task_id: str, offset: int) -> dict[str, str]:
    return {
        "task_id": task_id,
        "prompt": f"def f_{offset}(x):\n",
        "canonical_solution": f"    return x + {offset}\n",
        "entry_point": f"f_{offset}",
        "test": (
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            f"    results = [{1 + offset}]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    }


def _input_result_test() -> str:
    return (
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    results = [2, 3]\n"
        "    for inp, expected in zip(inputs, results):\n"
        "        assertion(candidate(*inp), expected)\n"
    )


def test_humaneval_public_api_is_curated() -> None:
    assert set(humaneval.__all__) == EXPECTED_HUMANEVAL_PUBLIC_API


class _CompletedProcessStub:
    def __init__(
        self,
        *,
        stdout: str,
        stderr: str = "",
        returncode: int = 0,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_sampling_from_rows_is_deterministic_and_indexed() -> None:
    rows = [_row(f"HumanEval/{index}", index) for index in range(5)]

    first = sample_human_eval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
    )
    second = sample_human_eval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
    )

    assert [sample.sample_index for sample in first] == [0, 1, 2]
    assert [sample.task.task_id for sample in first] == [
        sample.task.task_id for sample in second
    ]
    assert [sample.task.task_id for sample in first] == [
        "HumanEval/0",
        "HumanEval/2",
        "HumanEval/1",
    ]


def test_parse_input_result_tests_have_stable_case_ids() -> None:
    parsed = parse_human_eval_tests(_input_result_test())

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_RESULT
    assert [case.case_id for case in parsed.cases] == ["case_0", "case_1"]
    assert [case.kind for case in parsed.cases] == [
        HumanEvalTestCaseKind.INPUT_RESULT,
        HumanEvalTestCaseKind.INPUT_RESULT,
    ]
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[0].input_repr == "[1]"
    assert "candidate(*[1])" in checks[0].code

    summary = parsed.to_summary()
    assert summary.test_type is HumanEvalTestCaseKind.INPUT_RESULT
    assert [case.case_id for case in summary.cases] == ["case_0", "case_1"]
    assert summary.cases[0].input_repr == "[1]"
    assert "code" not in summary.cases[0].model_dump(mode="json")


def test_parse_oracle_tests_have_expected_expression_metadata() -> None:
    parsed = parse_human_eval_tests(
        "def ref(x):\n"
        "    return x + 1\n"
        "\n"
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    for inp in inputs:\n"
        "        assertion(candidate(*inp), ref(*inp))\n"
    )

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_ORACLE
    assert [case.case_id for case in parsed.cases] == ["case_0", "case_1"]
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[0].expected_output_expr == "ref(*[1])"


def test_parse_expression_tests_preserve_indexed_assertion() -> None:
    parsed = parse_human_eval_tests(
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    results = [2, 3]\n"
        "    for i, (inp, expected) in enumerate(zip(inputs, results)):\n"
        "        assert candidate(*inp) == expected\n"
    )

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_EXPRESSION
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[1].case_id == "case_1"
    assert "i = 1" in checks[1].code
    assert "assert candidate(*inp) == expected" in checks[1].code


def test_parsed_code_summary_excludes_runtime_ast() -> None:
    parsed = parse_code(
        display_title="fixture",
        code_str=(
            'def add_one(x: int) -> int:\n'
            '    """doc"""\n'
            '    return x + 1\n'
        ),
    )

    assert isinstance(parsed, ParsedCode)
    assert parsed.display_title == "fixture"
    assert parsed.signatures[0].function_name == "add_one"
    assert parsed.signatures[0].function_args[0].name == "x"
    dumped = parsed.model_dump(mode="json")
    assert "tree" not in dumped
    assert "doc" in dumped["comments"]


@pytest.mark.parametrize(
    ("source", "expected_fragment"),
    [
        ("```python\ndef add_one(x):\n    return x + 1\n```", "def add_one"),
        ("> def add_one(x):\n>     return x + 1", "def add_one"),
        (
            "    def add_one(x):\n        return x + 1\n",
            "def add_one",
        ),
        (
            "def add_one(x):\n"
            "    return x + 1\n"
            "print('trailing')\n",
            "return x + 1",
        ),
        (
            "def add_one(x):\n"
            "    return x + 1\n"
            "if __name__ == '__main__':\n"
            "    print(add_one(1))\n",
            "def add_one",
        ),
    ],
)
def test_apply_cleaning_extracts_known_submission_shapes(
    source: str,
    expected_fragment: str,
) -> None:
    candidates = apply_cleaning(source, apply_dedent=True)

    assert candidates
    assert expected_fragment in candidates[0]
    assert validate_python_source(candidates[0]).compile_ok
    assert "if __name__" not in candidates[0]
    assert "print('trailing')" not in candidates[0]


def test_evaluation_passes_when_best_function_passes() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def broken_helper(x):\n"
            "    return x\n"
            "\n"
            "def add_one(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is True
    assert result.status_counts == {"passed": 2}
    assert result.failures == []
    summary = result.to_summary()
    assert summary.passed is True
    assert summary.best_function_name == "add_one"
    assert summary.failure_count == 0


def test_evaluation_prefers_entry_point_when_pass_counts_tie() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    return x + 1\n"
            "\n"
            "def also_add_one(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is True


def test_evaluation_fails_when_best_function_does_not_pass_all_cases() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def broken_helper(x):\n"
            "    return x\n"
            "\n"
            "def add_one(x):\n"
            "    return x + 1 if x == 1 else x\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is False
    assert result.status_counts == {"passed": 1, "failed": 1}


def test_evaluation_uses_highest_pass_count() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    return x\n"
            "\n"
            "def helper(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.best_function_name == "helper"
    assert result.passed is True
    assert result.status_counts == {"passed": 2}


def test_evaluate_humaneval_code_reports_timeout_per_case() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    while True:\n"
            "        pass\n"
        ),
        timeout_seconds=0.2,
    )

    assert result.passed is False
    assert result.status_counts == {"timeout": 2}
    assert {case.case_id for case in result.results} == {"case_0", "case_1"}
    assert {case.timeout_seconds for case in result.results} == {0.2}
    assert evaluation_outcome(result) is SubmissionOutcome.TIMED_OUT


def test_run_subprocess_batch_raises_for_malformed_runner_output() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(
            stdout=(
                '[{"case_id": "case_0", "status": "passed", "message": ""}, '
                '{"case_id": "case_1", "status": "nonsense"}]'
            ),
        )

    with (
        patch("dr_code.humaneval.task.subprocess.run", fake_run),
        pytest.raises(EvaluationHarnessError) as exc_info,
    ):
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
        )

    results = exc_info.value.case_results
    by_case_id = {result.case_id: result for result in results}
    assert set(by_case_id) == {"case_0", "case_1"}
    assert by_case_id["case_0"].status is EvaluationCaseStatus.PASSED
    assert by_case_id["case_1"].status is EvaluationCaseStatus.ERROR
    assert "Invalid runner output" in by_case_id["case_1"].message


_PARTIAL_RUNNER_PASSED_CASE_0 = (
    '[{"case_id": "case_0", "status": "passed", "message": ""}]'
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
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(stdout=_PARTIAL_RUNNER_PASSED_CASE_0)

    with patch("dr_code.humaneval.task.subprocess.run", fake_run):
        result = score_humaneval_submission(
            raw_submission="def add_one(x):\n    return x + 1\n",
            task=_task(),
            parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
            timeout_seconds=2.0,
        )

    assert isinstance(result, CompletedScore)
    assert result.outcome is SubmissionOutcome.EVALUATION_INCOMPLETE
    assert result.score == 0.0
    assert result.evaluation is not None
    assert result.evaluation.failures == []
    assert result.evaluation.coverage_complete is False


def test_score_humaneval_submission_returns_harness_failure() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(stdout="not-json")

    with patch("dr_code.humaneval.task.subprocess.run", fake_run):
        result = score_humaneval_submission(
            raw_submission="def add_one(x):\n    return x + 1\n",
            task=_task(),
            parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
            timeout_seconds=2.0,
        )

    assert isinstance(result, HarnessFailure)
    assert result.kind == "harness_failure"
    assert result.failure_class == "unknown"
    assert result.cause.exception_type == "JSONDecodeError"
    assert result.evaluation is not None
    assert result.evaluation.results[0].elapsed_seconds is not None


def test_score_humaneval_submission_reports_empty_submission() -> None:
    result = score_humaneval_submission(
        raw_submission=" \n\t ",
        task=_task(),
        parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
        timeout_seconds=2.0,
    )

    assert isinstance(result, CompletedScore)
    assert result.kind == "completed"
    assert result.raw_submission == " \n\t "
    assert result.extraction.raw_submission == " \n\t "
    assert result.outcome is SubmissionOutcome.EMPTY_SUBMISSION
    assert result.evaluation is None


def test_evaluation_incomplete_when_runner_returns_partial_results() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(stdout=_PARTIAL_RUNNER_PASSED_CASE_0)

    with patch("dr_code.humaneval.task.subprocess.run", fake_run):
        result = evaluate_human_eval_code(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            timeout_seconds=2.0,
        )

    assert result.passed is False
    assert result.coverage_complete is False
    assert result.failures == []
    assert result.status_counts == {"passed": 1}


def test_compression_metrics_are_stable_for_methods_and_ratios() -> None:
    metrics = compression_metrics(
        ground_truth_code="def f():\n    return 1\n",
        representation_text="return 1",
    )

    assert set(metrics) == set(CompressionMethod)
    raw = metrics[CompressionMethod.RAW]
    assert raw.ground_truth_bytes == len(b"def f():\n    return 1\n")
    assert raw.representation_bytes == len(b"return 1")
    assert raw.compressed_bytes == raw.representation_bytes
    assert raw.ratio_to_ground_truth == pytest.approx(
        raw.representation_bytes / raw.ground_truth_bytes
    )


def test_compression_metrics_keep_empty_ground_truth_ratio_null() -> None:
    metrics = compression_metrics(
        ground_truth_code="",
        representation_text="return 1",
    )

    assert all(
        metric.ratio_to_ground_truth is None
        for metric in metrics.values()
    )
    assert all(
        metric.percent_reduction_vs_ground_truth is None
        for metric in metrics.values()
    )


def test_apply_cleaning_returns_empty_for_blank_input() -> None:
    assert apply_cleaning("") == []
    assert apply_cleaning("   \n\t  ") == []


def test_apply_cleaning_supports_tilde_fences() -> None:
    source = "~~~python\ndef add_one(x):\n    return x + 1\n~~~"
    candidates = apply_cleaning(source, apply_dedent=True)

    assert candidates
    assert "def add_one" in candidates[0]


def test_validate_python_source_reports_syntax_errors() -> None:
    validation = validate_python_source("def bad(x)\n  pass")

    assert validation.parse_ok is False
    assert validation.compile_ok is False
    assert validation.parse_error is not None
    assert validation.compile_error is not None


def test_score_humaneval_submission_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="raw_submission must be str"):
        score_humaneval_submission(
            raw_submission={"code": "def add_one(x):\n    return x + 1\n"},  # type: ignore[arg-type]
            task=_task(),
            parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
            timeout_seconds=2.0,
        )


@pytest.mark.parametrize(
    ("test_source", "match"),
    [
        ("def helper():\n    pass\n", "Could not find check"),
        (
            "def check(a, b):\n    pass\n",
            "one positional argument",
        ),
        (
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            "    results = [1, 2]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n",
            "does not match",
        ),
        (
            "def check(candidate):\n"
            "    inputs = range(3)\n"
            "    results = [0, 1, 2]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n",
            "not a literal",
        ),
    ],
)
def test_parse_human_eval_tests_rejects_invalid_formats(
    test_source: str,
    match: str,
) -> None:
    with pytest.raises(UnsupportedTestFormatError, match=match):
        parse_human_eval_tests(test_source)


def test_run_subprocess_batch_raises_for_nonzero_returncode() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(
            stdout="",
            stderr="runner crashed",
            returncode=1,
        )

    with (
        patch("dr_code.humaneval.task.subprocess.run", fake_run),
        pytest.raises(EvaluationHarnessError) as exc_info,
    ):
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
        )

    results = exc_info.value.case_results
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "runner crashed" in results[0].message


def test_run_subprocess_batch_raises_for_invalid_json() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(stdout="not-json")

    with (
        patch("dr_code.humaneval.task.subprocess.run", fake_run),
        pytest.raises(EvaluationHarnessError) as exc_info,
    ):
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
        )

    results = exc_info.value.case_results
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "Could not decode runner output" in results[0].message


def test_run_subprocess_batch_raises_for_non_list_json() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(stdout='{"not": "a list"}')

    with (
        patch("dr_code.humaneval.task.subprocess.run", fake_run),
        pytest.raises(EvaluationHarnessError) as exc_info,
    ):
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
        )

    results = exc_info.value.case_results
    assert "expected a JSON list" in results[0].message


def test_run_subprocess_batch_fallback_case_id_is_harness_detail() -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _CompletedProcessStub:
        return _CompletedProcessStub(
            stdout='[{"status": "passed", "message": ""}]',
        )

    with (
        patch("dr_code.humaneval.task.subprocess.run", fake_run),
        pytest.raises(EvaluationHarnessError) as exc_info,
    ):
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
        )

    results = exc_info.value.case_results
    assert results[0].case_id == "case_0"


def test_apply_human_eval_override_passthrough() -> None:
    row = _row("HumanEval/99", 1)
    assert apply_human_eval_override(row, {}) == dict(row)

    updated = apply_human_eval_override(
        row,
        {
            "HumanEval/99": HumanEvalOverride(
                canonical_solution="    return x + 99\n",
            ),
        },
    )
    assert updated["canonical_solution"] == "    return x + 99\n"

    with pytest.raises(ValueError, match="replacement text not found"):
        apply_human_eval_override(
            row,
            {
                "HumanEval/99": HumanEvalOverride(
                    test_replacements={"missing": "text"},
                ),
            },
        )


def test_parse_human_eval_dataset_builds_tasks() -> None:
    tasks = parse_human_eval_dataset([_row("HumanEval/0", 0)])

    assert len(tasks) == 1
    assert tasks[0].task_id == "HumanEval/0"
    assert tasks[0].parsed_tests is not None


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

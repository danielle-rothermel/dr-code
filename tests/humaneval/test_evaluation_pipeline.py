"""End-to-end coverage of the HumanEval evaluation pipeline.

One file rather than per-module files because the stages compose into a
single pipeline and are asserted against each other: ``task`` (tasks,
overrides, dataset parsing), ``sampling`` (row loading and task sampling),
``parsed_tests`` and ``parsed_code`` (structured test/code parsing),
``code_parsing`` (acceptance policy), ``batch_runner`` (subprocess
batch execution), and ``scoring`` (submission outcomes). It also pins the
``dr_code.humaneval`` package ``__all__``, which spans every stage above.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import dr_code.humaneval as humaneval
from dr_code.code_analysis import validate_python_source
from dr_code.humaneval import (
    EvaluationCaseStatus,
    HumanEvalTask,
    parse_humaneval_dataset,
)
from dr_code.humaneval.batch_runner import (
    evaluate_humaneval_code,
    require_parsed_tests,
    run_subprocess_batch,
    runner_script,
)
from dr_code.humaneval.code_parsing import extract_humaneval_code
from dr_code.preprocessing import PreprocessingFailureCode
from dr_code.humaneval.parsed_code import ParsedCode, parse_code
from dr_code.humaneval.parsed_tests import (
    HumanEvalTestCaseKind,
    UnsupportedTestFormatError,
    parse_humaneval_tests,
)
from dr_code.humaneval.sampling import (
    HumanEvalRawRowsSnapshot,
    validate_snapshot_header,
    load_humaneval_rows,
    sample_humaneval_tasks_from_rows,
)
from dr_code.humaneval.scoring import (
    CompletedScore,
    HarnessFailure,
    SubmissionOutcome,
    evaluation_outcome,
    score_humaneval_submission,
)
from dr_code.humaneval.task import (
    HUMANEVAL_OVERRIDE_SET,
    EvaluationCaseResult,
    EvaluationHarnessError,
    EvaluationTaskResult,
    HumanEvalOverride,
    HumanEvalTestReplacement,
    _apply_humaneval_override,
)
from dr_code.humaneval.sandbox import (
    SandboxCompletedProcess,
    SandboxError,
    SandboxOutputLimitError,
    SandboxRunner,
    SandboxTimeoutError,
)

#: The repository's tracked offline HumanEvalPlus snapshot.
SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "corpus"
    / "humanevalplus_snapshot.json"
)


@pytest.fixture(scope="module")
def raw_snapshot() -> HumanEvalRawRowsSnapshot:
    return HumanEvalRawRowsSnapshot.model_validate_json(
        SNAPSHOT_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def snapshot_tasks() -> list[HumanEvalTask]:
    return parse_humaneval_dataset(
        load_humaneval_rows(snapshot_path=SNAPSHOT_PATH)
    )


EXPECTED_HUMANEVAL_PUBLIC_API = {
    "CodeExtractionResult",
    "CompletedScore",
    "DEFAULT_HUMANEVAL_SCORING_PROFILE",
    "DEFAULT_HUMANEVAL_TIMEOUT_SECONDS",
    "EvaluationCaseStatus",
    "EvaluationCaseSummary",
    "EvaluationTaskSummary",
    "HUMANEVAL_SCORING_PROFILE_ID",
    "HUMANEVAL_SCORING_PROFILE_VERSION",
    "HUMANEVAL_METRICS_PROFILE",
    "HUMANEVAL_OVERRIDE_SET",
    "HUMANEVAL_OVERRIDE_SET_ID",
    "HUMANEVAL_OVERRIDE_SET_VERSION",
    "HarnessFailure",
    "HarnessFailureCause",
    "HumanEvalScoringProfile",
    "HumanEvalMetricsProfile",
    "HumanEvalOverrideEntry",
    "HumanEvalOverrideSetCoordinate",
    "HumanEvalSubmissionScore",
    "HumanEvalTask",
    "HumanEvalTestCaseKind",
    "PreprocessingDefinitionReference",
    "SampledHumanEvalTask",
    "SubmissionOutcome",
    "accept_first_surviving",
    "extract_humaneval_code",
    "humaneval_runner",
    "load_humaneval_rows",
    "parse_humaneval_dataset",
    "resolve_humaneval_scoring_profile",
    "sample_humaneval_tasks",
    "sample_humaneval_tasks_from_rows",
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


def _check_payload_bytes(task: HumanEvalTask) -> list[bytes]:
    parsed_tests = require_parsed_tests(task)
    return [
        case.as_check(
            candidate_name="candidate",
            assertion_name=parsed_tests.assertion_name,
        )
        .model_dump_json()
        .encode("utf-8")
        for case in parsed_tests.cases
    ]


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


@pytest.fixture
def local_runner() -> SandboxRunner:
    """A real, injectable runner that keeps primitive tests fast.

    It runs the candidate under the host interpreter instead of the OCI
    sandbox; the container contract has its own probes in ``test_sandbox``.
    Injected via ``run_in_sandbox=`` rather than patched, so the seam is a
    real function argument.
    """

    def run_local_python(
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", source],
                input=input_json,
                capture_output=True,
                check=False,
                encoding="utf-8",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(str(exc)) from exc
        return SandboxCompletedProcess(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return run_local_python


def _stub_runner(
    *,
    stdout: str,
    stderr: str = "",
    returncode: int = 0,
) -> SandboxRunner:
    """Build a runner that returns a fixed completed process."""

    def run(
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        return SandboxCompletedProcess(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return run


def test_sampling_from_rows_is_deterministic_and_indexed() -> None:
    rows = [_row(f"HumanEval/{index}", index) for index in range(5)]

    first = sample_humaneval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
    )
    second = sample_humaneval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
    )

    assert [sample.sample_index for sample in first] == [0, 1, 2]
    assert [sample.task.task_id for sample in first] == [
        sample.task.task_id for sample in second
    ]


def test_raw_row_snapshot_rehydrates_byte_equal_checks(
    raw_snapshot: HumanEvalRawRowsSnapshot,
    snapshot_tasks: list[HumanEvalTask],
) -> None:
    assert raw_snapshot.header.override_set == HUMANEVAL_OVERRIDE_SET
    assert [task.task_id for task in snapshot_tasks] == [
        row.task_id for row in raw_snapshot.rows
    ]

    fresh_tasks = parse_humaneval_dataset(
        [row.model_dump(mode="json") for row in raw_snapshot.rows]
    )
    for fresh_task, snapshot_task in zip(
        fresh_tasks,
        snapshot_tasks,
        strict=True,
    ):
        assert _check_payload_bytes(snapshot_task) == _check_payload_bytes(
            fresh_task
        )


def test_raw_row_snapshot_rejects_structural_override_set_mismatch(
    raw_snapshot: HumanEvalRawRowsSnapshot,
) -> None:
    mismatched_header = raw_snapshot.header.model_copy(
        update={
            "override_set": raw_snapshot.header.override_set.model_copy(
                update={"entries": ()}
            )
        }
    )

    with pytest.raises(ValueError, match="override-set mismatch"):
        validate_snapshot_header(
            mismatched_header,
            dataset_name=raw_snapshot.header.dataset_id,
            hf_revision=raw_snapshot.header.hf_revision,
        )


def test_parse_input_result_tests_have_stable_case_ids() -> None:
    parsed = parse_humaneval_tests(_input_result_test())

    assert parsed.test_type is HumanEvalTestCaseKind.INPUT_RESULT
    assert [case.case_id for case in parsed.cases] == ["case_0", "case_1"]
    assert [case.kind for case in parsed.cases] == [
        HumanEvalTestCaseKind.INPUT_RESULT,
        HumanEvalTestCaseKind.INPUT_RESULT,
    ]
    checks = list(parsed.iter_checks(candidate_name="candidate"))
    assert checks[0].input_repr == "[1]"
    assert "candidate(*[1])" in checks[0].code


def test_parse_oracle_tests_have_expected_expression_metadata() -> None:
    parsed = parse_humaneval_tests(
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
    parsed = parse_humaneval_tests(
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
            'def add_one(x: int) -> int:\n    """doc"""\n    return x + 1\n'
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
            "def add_one(x):\n    return x + 1\nprint('trailing')\n",
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
def test_extraction_accepts_known_submission_shapes(
    source: str,
    expected_fragment: str,
) -> None:
    result = extract_humaneval_code(source)

    assert result.succeeded
    assert expected_fragment in result.accepted_code
    assert validate_python_source(result.accepted_code).compile_ok
    # The name guard is split away during cleaning, so it never reaches an
    # accepted candidate.
    assert "if __name__" not in result.accepted_code


def test_trailing_statements_survive_alongside_their_salvage() -> None:
    # Truncating at the last return is additive: the candidate as written
    # is accepted, and its truncation is also present in the set rather
    # than having replaced it.
    result = extract_humaneval_code(
        "def add_one(x):\n    return x + 1\nprint('trailing')\n"
    )
    assert result.succeeded
    assert "print('trailing')" in result.accepted_code
    assert result.candidate_count == 2


def test_salvaged_candidate_still_gets_its_inferred_imports() -> None:
    # A submission whose only defect is trailing prose is unparseable until
    # the last-return salvage truncates it. Import inference is parse-driven
    # and no-ops on unparseable source, so it must run after the salvage --
    # otherwise the truncated candidate is accepted still referencing `np`
    # with no import, and fails at runtime with NameError.
    result = extract_humaneval_code(
        "def f(x):\n    return np.array(x)\nThis is trailing prose.\n"
    )

    assert result.succeeded
    assert "import numpy as np" in result.accepted_code
    assert validate_python_source(result.accepted_code).compile_ok


def test_marked_code_field_wins_over_code_in_another_marked_field() -> None:
    # A response that declares which part is its answer is answering
    # directly; scraping code out of arbitrary text is inference. When a
    # preceding field carries a fenced starter or reference function, the
    # scrape must not shadow the marked answer under an acceptance policy
    # that takes the lowest surviving ordinal.
    result = extract_humaneval_code(
        "[[ ## prompt ## ]]\n"
        "```python\n"
        "def add_one(x):\n"
        '    """Reference/starter."""\n'
        "    raise NotImplementedError\n"
        "```\n\n"
        "[[ ## code ## ]]\n"
        "def add_one(x):\n"
        "    return x + 1\n"
    )

    assert result.succeeded
    assert result.accepted_code == "def add_one(x):\n    return x + 1"
    # The starter is still extracted -- readings are ordered, not exclusive.
    assert result.candidate_count == 2


def test_json_code_field_wins_over_code_quoted_in_other_json_fields() -> None:
    result = extract_humaneval_code(
        '{"reasoning": "first I tried:\\n```python\\n'
        'def f(x):\\n    raise NotImplementedError\\n```", '
        '"code": "def f(x):\\n    return x + 1\\n"}'
    )

    assert result.succeeded
    assert result.accepted_code == "def f(x):\n    return x + 1"


def test_evaluation_passes_when_best_function_passes(
    local_runner: SandboxRunner,
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
        run_in_sandbox=local_runner,
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
    local_runner: SandboxRunner,
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
        run_in_sandbox=local_runner,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is True


def test_evaluation_fails_when_best_function_does_not_pass_all_cases(
    local_runner: SandboxRunner,
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
        run_in_sandbox=local_runner,
    )

    assert result.best_function_name == "add_one"
    assert result.passed is False
    assert result.status_counts == {"passed": 1, "failed": 1}


def test_evaluation_uses_highest_pass_count(
    local_runner: SandboxRunner,
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
        run_in_sandbox=local_runner,
    )

    assert result.best_function_name == "helper"
    assert result.passed is True
    assert result.status_counts == {"passed": 2}


def test_evaluate_humaneval_code_reports_timeout_per_case() -> None:
    candidate_code = "def add_one(x):\n    return x + 1\n"
    timeout_seconds = 0.2
    forwarded_inputs: list[str] = []
    forwarded_timeouts: list[float] = []
    timeout_cause = SandboxTimeoutError("controlled sandbox timeout")

    def timeout_runner(
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        forwarded_inputs.append(input_json)
        forwarded_timeouts.append(timeout_seconds)
        raise timeout_cause

    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=candidate_code,
        timeout_seconds=timeout_seconds,
        run_in_sandbox=timeout_runner,
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
    assert len(forwarded_inputs) == 1
    assert json.loads(forwarded_inputs[0]) == {
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
    assert forwarded_timeouts == [timeout_seconds]
    assert evaluation_outcome(result) is SubmissionOutcome.TIMED_OUT


def test_run_subprocess_batch_raises_for_malformed_runner_output() -> None:
    runner = _stub_runner(
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
            run_in_sandbox=runner,
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
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        run_in_sandbox=_stub_runner(stdout=_PARTIAL_RUNNER_PASSED_CASE_0),
    )

    assert isinstance(result, CompletedScore)
    assert result.outcome is SubmissionOutcome.EVALUATION_INCOMPLETE
    assert result.score == 0.0
    assert result.evaluation is not None
    assert result.evaluation.failures == []
    assert result.evaluation.coverage_complete is False


def test_score_humaneval_submission_returns_harness_failure() -> None:
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        run_in_sandbox=_stub_runner(stdout="not-json"),
    )

    assert isinstance(result, HarnessFailure)
    assert result.kind == "harness_failure"
    assert result.failure_class == "unknown"
    assert result.cause.exception_type == "JSONDecodeError"
    assert result.evaluation is not None
    assert result.evaluation.results[0].elapsed_seconds is not None


def test_score_humaneval_submission_reports_generic_sandbox_breakage() -> None:
    """A broken sandbox is a harness failure, never a scored result.

    A candidate must not benefit from generic sandbox breakage: the base
    ``SandboxError`` surfaces as a ``HarnessFailure`` rather than a
    ``CompletedScore`` with a zero score.
    """

    def broken_sandbox(
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        raise SandboxError("sandbox runtime is unavailable")

    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        run_in_sandbox=broken_sandbox,
    )

    assert isinstance(result, HarnessFailure)
    assert result.kind == "harness_failure"
    assert result.cause.exception_type == "SandboxError"


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


def test_evaluation_incomplete_when_runner_returns_partial_results() -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        timeout_seconds=2.0,
        run_in_sandbox=_stub_runner(stdout=_PARTIAL_RUNNER_PASSED_CASE_0),
    )

    assert result.passed is False
    assert result.coverage_complete is False
    assert result.failures == []
    assert result.status_counts == {"passed": 1}


def test_extraction_reports_blank_input_as_its_own_failure() -> None:
    for blank in ("", "   \n\t  "):
        result = extract_humaneval_code(blank)
        assert not result.succeeded
        assert result.failure_code == PreprocessingFailureCode.BLANK_INPUT


def test_extraction_supports_tilde_fences() -> None:
    source = "~~~python\ndef add_one(x):\n    return x + 1\n~~~"
    result = extract_humaneval_code(source)

    assert result.succeeded
    assert "def add_one" in result.accepted_code


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
def test_parse_humaneval_tests_rejects_invalid_formats(
    test_source: str,
    match: str,
) -> None:
    with pytest.raises(UnsupportedTestFormatError, match=match):
        parse_humaneval_tests(test_source)


def test_run_subprocess_batch_scores_candidate_kill_returncode() -> None:
    results = run_subprocess_batch(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        function_name="add_one",
        timeout_seconds=2.0,
        run_in_sandbox=_stub_runner(stdout="", stderr="", returncode=137),
    )

    assert len(results) == 2
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "sandbox killed candidate execution" in results[0].message


def test_run_subprocess_batch_scores_output_limit_as_candidate_error() -> None:
    def overflowing_sandbox(
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        raise SandboxOutputLimitError("sandbox output exceeded limit")

    results = run_subprocess_batch(
        task=_task(),
        candidate_code="def add_one(x):\n    return x + 1\n",
        function_name="add_one",
        timeout_seconds=2.0,
        run_in_sandbox=overflowing_sandbox,
    )

    assert len(results) == 2
    assert all(
        result.status is EvaluationCaseStatus.ERROR for result in results
    )
    assert "SandboxOutputLimitError" in results[0].message


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
            run_in_sandbox=_stub_runner(stdout=runner_stdout),
        )


def test_candidate_module_level_sys_exit_is_scored(
    local_runner: SandboxRunner,
) -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=(
            "import sys\nsys.exit(5)\ndef add_one(x):\n    return x + 1\n"
        ),
        timeout_seconds=2.0,
        run_in_sandbox=local_runner,
    )

    assert result.passed is False
    assert result.status_counts == {"error": 2}


def test_run_subprocess_batch_raises_for_nonzero_returncode() -> None:
    runner = _stub_runner(stdout="", stderr="runner crashed", returncode=1)

    with pytest.raises(EvaluationHarnessError) as exc_info:
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
            run_in_sandbox=runner,
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
            run_in_sandbox=_stub_runner(stdout="not-json"),
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
            run_in_sandbox=_stub_runner(stdout='{"not": "a list"}'),
        )

    results = exc_info.value.case_results
    assert "expected a JSON list" in results[0].message


def test_run_subprocess_batch_fallback_case_id_is_harness_detail() -> None:
    runner = _stub_runner(stdout='[{"status": "passed", "message": ""}]')

    with pytest.raises(EvaluationHarnessError) as exc_info:
        run_subprocess_batch(
            task=_task(),
            candidate_code="def add_one(x):\n    return x + 1\n",
            function_name="add_one",
            timeout_seconds=2.0,
            run_in_sandbox=runner,
        )

    results = exc_info.value.case_results
    assert results[0].case_id == "case_0"


def test_apply_humaneval_override_passthrough() -> None:
    row = _row("HumanEval/99", 1)
    assert _apply_humaneval_override(row, {}) == dict(row)

    updated = _apply_humaneval_override(
        row,
        {
            "HumanEval/99": HumanEvalOverride(
                canonical_solution="    return x + 99\n",
            ),
        },
    )
    assert updated["canonical_solution"] == "    return x + 99\n"

    with pytest.raises(ValueError, match="replacement text not found"):
        _apply_humaneval_override(
            row,
            {
                "HumanEval/99": HumanEvalOverride(
                    test_replacements=(
                        HumanEvalTestReplacement(
                            old="missing",
                            replacement="text",
                        ),
                    ),
                ),
            },
        )


def test_parse_humaneval_dataset_builds_tasks() -> None:
    tasks = parse_humaneval_dataset([_row("HumanEval/0", 0)])

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


def test_runner_script_source_compiles() -> None:
    compile(runner_script(), "<runner>", "exec")


def test_runner_script_source_is_dependency_free() -> None:
    tree = ast.parse(runner_script())
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")

    assert not any(
        module == "dr_code" or module.startswith("dr_code.")
        for module in imported_modules
    )

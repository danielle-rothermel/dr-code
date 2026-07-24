"""Behavioral contracts for all registered metric operators.

Operators are engine-managed classes reached through ``extract_metrics``.
Compression questions pin codec levels explicitly, and HumanEval execution
is checked against its owning batch-runner behavior.
"""

from __future__ import annotations

import gzip
from typing import get_args, get_type_hints

import pytest
import zstandard
from pydantic import ValidationError

from dr_code.trace import CodeArtifact, TextArtifact

from metrics.helpers import (
    code_test_trace,
    evaluate_oracle,
    evaluation_procedure,
    procedure_trace,
    external_trace,
)

# Golden compressed sizes use gzip level 9 and zstd level 3.
_ZSTD_DEFAULT_LEVEL = 3

SAMPLE_TEXT = (
    "Here is some text with a `code` fence:\n"
    "```python\ndef foo(x):\n    return x + 1\n```\n"
    "It has keywords like def and return, plus + - * operators.\n"
)
SAMPLE_CODE = (
    "def add(a, b):\n"
    '    """Sum two numbers."""\n'
    "    total = a + b\n"
    "    if total > 0:\n"
    "        return total\n"
    "    return 0\n"
)


def test_registry_fact_units_exactly_match_compute_result_models() -> None:
    from dr_code.metrics.operators.base import OperatorResult
    from dr_code.metrics.registry import REGISTRY

    diagnostics: list[str] = []
    for registry_name, operator in sorted(REGISTRY.items()):
        return_annotation = get_type_hints(operator.compute).get("return")
        result_models = get_args(return_annotation) or (return_annotation,)
        expected_fields: set[str] = set()
        for result_model in result_models:
            if not isinstance(result_model, type) or not issubclass(
                result_model, OperatorResult
            ):
                diagnostics.append(
                    f"{registry_name}: compute return annotation "
                    f"{return_annotation!r} is not an OperatorResult model"
                )
                continue
            expected_fields.update(result_model.model_fields)

        actual_fields = set(operator.FACT_UNITS)
        missing = expected_fields - actual_fields
        unexpected = actual_fields - expected_fields
        blank_units = sorted(
            name
            for name, unit in operator.FACT_UNITS.items()
            if not isinstance(unit, str) or not unit
        )
        if missing or unexpected or blank_units:
            diagnostics.append(
                f"{registry_name}: missing units={sorted(missing)!r}, "
                f"unexpected units={sorted(unexpected)!r}, "
                f"blank units={blank_units!r}"
            )

    assert not diagnostics, "\n".join(diagnostics)


def test_metric_registry_is_immutable_after_builtin_registration() -> None:
    from dr_code.metrics.registry import REGISTRY

    with pytest.raises(TypeError):
        REGISTRY["replacement"] = next(iter(REGISTRY.values()))  # type: ignore[index]


def _definition(questions) -> object:
    from dr_code.eval import MetricExtractionDefinition

    return MetricExtractionDefinition(
        definition_id="parity", version="1", questions=tuple(questions)
    )


def _question(metric_name: str, on: str = "input", **settings) -> object:
    from dr_code.eval import MetricQuestionBinding
    from dr_code.metrics import MetricName

    return MetricQuestionBinding(
        metric=MetricName(metric_name), on=on, settings=settings
    )


def _text_trace(text: str):
    return external_trace(
        {"input": TextArtifact(text=text), "output": TextArtifact(text=text)}
    )


def _code_trace(source: str):
    return external_trace(
        {
            "input": CodeArtifact(source=source),
            "output": CodeArtifact(source=source),
        }
    )


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    metric_extraction = definition.materialize()
    procedure = evaluation_procedure(definition, metric_extraction)
    return extract_metrics(
        procedure_trace(trace, procedure),
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
        **kwargs,
    )


def _value(record, key):
    assert record.status.value == "measured", record
    return record.fact_values()[key]


# ===========================================================================
# text_stats
# ===========================================================================

# Stable text-stat values for the fixed sample.
_TEXT_STATS_GOLDEN = {
    "character_count": 141,
    "byte_count": 141,
    "line_count": 7,
    "nonempty_line_count": 6,
    "word_count": 24,
    "average_word_length": 3.7916666666666665,
    "punctuation_count": 18,
    "symbol_count": 6,
}


def test_text_stats_match_golden_values_field_for_field() -> None:
    record = _extract(
        _definition([_question("text_stats")]), _text_trace(SAMPLE_TEXT)
    )[0]

    for field, expected in _TEXT_STATS_GOLDEN.items():
        assert _value(record, field) == expected, field


def test_text_stats_empty_text_has_zero_counts() -> None:
    record = _extract(_definition([_question("text_stats")]), _text_trace(""))[
        0
    ]
    assert _value(record, "character_count") == 0
    assert _value(record, "line_count") == 0


# ===========================================================================
# code_leakage with the task_names setting
# ===========================================================================

# Stable code-leakage values for task_names=("foo", "HumanEval/x").
_CODE_LEAKAGE_GOLDEN = {
    "keyword_count": 7,
    "code_marker_count": 4,
    "fenced_code_block_count": 1,
    "code_like_line_count": 2,
    "operator_count": 6,
    "punctuation_density": 0.1276595744680851,
    "task_name_hit_count": 1,
}


def test_code_leakage_matches_golden_values_field_for_field() -> None:
    task_names = ("foo", "HumanEval/x")
    record = _extract(
        _definition([_question("code_leakage", task_names=list(task_names))]),
        _text_trace(SAMPLE_TEXT),
    )[0]

    for field, expected in _CODE_LEAKAGE_GOLDEN.items():
        assert _value(record, field) == expected, field


def test_code_leakage_task_names_are_part_of_identity() -> None:
    """task_names is a setting and therefore part of metric identity."""
    text = "def foo(x):\n    return foo(x)\n"
    none_rec = _extract(
        _definition([_question("code_leakage", task_names=[])]),
        _text_trace(text),
    )[0]
    named_rec = _extract(
        _definition([_question("code_leakage", task_names=["foo"])]),
        _text_trace(text),
    )[0]
    assert none_rec.fact_values()["task_name_hit_count"] == 0
    assert named_rec.fact_values()["task_name_hit_count"] >= 1


# This sample exercises indented lines, comments, augmented assignment,
# keyword-like prose, and whole-line code fences in the shared heuristics.
_SHARED_HEURISTIC_SAMPLE = (
    "Explanation text before any code.\n"
    "```python\n"
    "def solve(x):\n"
    "    # walk through the input\n"
    "    total = 0\n"
    "    for item in x:\n"
    "        total += item\n"
    "    return total\n"
    "```\n"
    "More prose mentioning pass, continue, and break as English words.\n"
)
_SHARED_HEURISTIC_GOLDEN = {
    "keyword_count": 9,
    "code_marker_count": 2,
    "fenced_code_block_count": 1,
    "code_like_line_count": 6,
    "operator_count": 5,
    "punctuation_density": 0.07860262008733625,
    "task_name_hit_count": 0,
}


def test_code_leakage_pins_shared_heuristic_values() -> None:
    """Pin code_leakage's use of the shared text-analysis regexes."""
    record = _extract(
        _definition([_question("code_leakage", task_names=[])]),
        _text_trace(_SHARED_HEURISTIC_SAMPLE),
    )[0]

    for field, expected in _SHARED_HEURISTIC_GOLDEN.items():
        assert _value(record, field) == expected, field


# ===========================================================================
# parse_outcome and ast_stats
# ===========================================================================


def test_parse_outcome_reports_parse_ok_for_valid_code() -> None:
    record = _extract(
        _definition([_question("parse_outcome")]), _code_trace(SAMPLE_CODE)
    )[0]
    assert _value(record, "parse_ok") is True


def test_parse_outcome_reports_parse_error_for_invalid_code() -> None:
    invalid = "def f(:\n    pass\n"
    record = _extract(
        _definition([_question("parse_outcome")]), _code_trace(invalid)
    )[0]
    assert _value(record, "parse_ok") is False


def test_parse_outcome_accepts_text_artifacts() -> None:
    """parse_outcome accepts raw text as well as code artifacts."""
    record = _extract(
        _definition([_question("parse_outcome")]), _text_trace("x = 1 + 2")
    )[0]
    assert _value(record, "parse_ok") is True


# Stable structure counts for SAMPLE_CODE.
_AST_STATS_GOLDEN = {
    "top_level_function_count": 1,
    "nested_function_count": 0,
    "async_function_count": 0,
    "lambda_count": 0,
    "class_count": 0,
    "import_count": 0,
    "ast_node_count": 27,
    "statement_count": 6,
    "branch_count": 1,
    "return_count": 2,
    "yield_count": 0,
    "call_count": 0,
    "assignment_count": 1,
    "comprehension_count": 0,
    "literal_count": 3,
    "max_branch_depth": 1,
    "function_count": 1,
    "total_argument_count": 2,
    "positional_only_argument_count": 0,
    "keyword_only_argument_count": 0,
    "vararg_count": 0,
    "kwarg_count": 0,
    "decorated_function_count": 0,
    "annotated_return_count": 0,
    "docstring_function_count": 1,
    "total_function_body_statement_count": 4,
    "max_function_body_statement_count": 4,
    "max_function_line_span": 6,
}


def test_ast_stats_match_golden_structure_counts() -> None:
    record = _extract(
        _definition([_question("ast_stats")]), _code_trace(SAMPLE_CODE)
    )[0]

    for field, expected in _AST_STATS_GOLDEN.items():
        assert _value(record, field) == expected, field


# ===========================================================================
# compressed_length with pinned gzip 9 and zstd 3
# ===========================================================================


def _reference_trace(text: str, reference: str):
    return external_trace(
        {
            "input": TextArtifact(text=text),
            "output": TextArtifact(text=text),
            "reference": CodeArtifact(source=reference),
        }
    )


def test_compressed_length_gzip_level_9_reproduces_default(task) -> None:
    """Pinned gzip level 9 determines the compressed-length result."""
    reference = task.ground_truth_code
    # The question pins gzip level 9 and divides by ground-truth byte length.
    expected_compressed = len(
        gzip.compress(SAMPLE_TEXT.encode("utf-8"), compresslevel=9)
    )
    ground_truth_bytes = len(reference.encode("utf-8"))
    expected_ratio = expected_compressed / ground_truth_bytes
    expected_percent_reduction = (1.0 - expected_ratio) * 100.0

    record = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "gzip", "level": 9},
                    reference_key="reference",
                )
            ]
        ),
        _reference_trace(SAMPLE_TEXT, reference),
    )[0]
    assert _value(record, "compressed_bytes") == expected_compressed
    assert _value(record, "representation_bytes") == len(
        SAMPLE_TEXT.encode("utf-8")
    )
    assert _value(record, "ratio_to_reference") == expected_ratio
    assert _value(record, "percent_reduction") == expected_percent_reduction


def test_compressed_length_zstd_level_3_reproduces_default(task) -> None:
    """Pinned zstd level 3 determines the compressed-length result."""
    reference = task.ground_truth_code

    record = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "zstd", "level": 3},
                    reference_key="reference",
                )
            ]
        ),
        _reference_trace(SAMPLE_TEXT, reference),
    )[0]
    # The question pins zstd level 3.
    assert record.fact_values()["compressed_bytes"] == len(
        zstandard.ZstdCompressor(level=_ZSTD_DEFAULT_LEVEL).compress(
            SAMPLE_TEXT.encode("utf-8")
        )
    )


def test_compressed_length_without_reference_has_no_ratio() -> None:
    """No reference_key ⇒ ratio stays None (empty-reference behaviour)."""
    record = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "gzip", "level": 9},
                )
            ]
        ),
        _text_trace(SAMPLE_TEXT),
    )[0]
    assert record.fact_values()["compressed_bytes"] == len(
        gzip.compress(SAMPLE_TEXT.encode("utf-8"), compresslevel=9)
    )
    assert "representation_bytes" in record.fact_values()
    assert record.fact_values()["ratio_to_reference"] is None
    ratio = next(
        fact for fact in record.facts if fact.name == "ratio_to_reference"
    )
    assert ratio.applicability.value == "not_applicable"
    assert ratio.reason == "no compression reference was configured"


def test_compressed_level_is_part_of_identity() -> None:
    """A different level is a different question."""
    trace = _text_trace(SAMPLE_TEXT)
    r6 = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "gzip", "level": 1},
                )
            ]
        ),
        trace,
    )[0]
    r9 = _extract(
        _definition(
            [
                _question(
                    "compressed_length",
                    compression={"method": "gzip", "level": 9},
                )
            ]
        ),
        trace,
    )[0]
    assert (
        r9.fact_values()["compressed_bytes"]
        <= r6.fact_values()["compressed_bytes"]
    )


# ===========================================================================
# code_test counts and attribution
# ===========================================================================


@pytest.mark.parametrize(
    "updates",
    [
        {"passed_count": 3},
        {"passed_count": 1, "coverage_complete": True},
        {"coverage_complete": False},
        {
            "function_count": 0,
            "best_function_name": None,
            "coverage_complete": False,
        },
        {
            "function_count": 0,
            "passed_count": 0,
            "best_function_name": "f",
            "coverage_complete": False,
        },
        {
            "function_count": 0,
            "passed_count": 0,
            "best_function_name": None,
            "coverage_complete": True,
        },
    ],
)
def test_code_test_result_rejects_impossible_relations(
    updates: dict[str, object],
) -> None:
    from dr_code.metrics.operators.code_test import CodeTestResult

    values = {
        "total_cases": 2,
        "passed_count": 2,
        "failed_count": 0,
        "error_count": 0,
        "timeout_count": 0,
        "coverage_complete": True,
        "function_count": 1,
        "best_function_name": "f",
    }
    with pytest.raises(ValidationError):
        CodeTestResult.model_validate({**values, **updates}, strict=True)


def _code_test_question(timeout_seconds: float = 5.0) -> object:
    return _question("code_test", on="input", timeout_seconds=timeout_seconds)


def test_direct_and_metrics_execution_share_humaneval_request_builder(
    task,
    good_submission,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dr_code.humaneval.batch_runner as batch_runner
    from dr_code.execution.subprocess import SubprocessCompletedProcess
    from dr_code.metrics.operators.code_test import CodeTest, CodeTestSettings

    from metrics.helpers import task_json_artifact

    original_builder = batch_runner.build_human_eval_batch_request
    builder_calls: list[tuple[str, str, str, float, object, object]] = []

    def recording_builder(
        *,
        task,
        candidate_code,
        function_name,
        timeout_seconds,
        checks=None,
        runner_source=None,
    ):
        builder_calls.append(
            (
                task.task_id,
                candidate_code,
                function_name,
                timeout_seconds,
                checks,
                runner_source,
            )
        )
        return original_builder(
            task=task,
            candidate_code=candidate_code,
            function_name=function_name,
            timeout_seconds=timeout_seconds,
            checks=checks,
            runner_source=runner_source,
        )

    monkeypatch.setattr(
        batch_runner,
        "build_human_eval_batch_request",
        recording_builder,
    )

    direct_calls: list[tuple[str, str, float]] = []

    def direct_runner(*, source, input_text, timeout_seconds):
        direct_calls.append((source, input_text, timeout_seconds))
        return SubprocessCompletedProcess(returncode=0, stdout="[]", stderr="")

    timeout_seconds = 2.5
    batch_runner.run_subprocess_batch(
        task=task,
        candidate_code=good_submission,
        function_name=task.entry_point,
        timeout_seconds=timeout_seconds,
        run_in_subprocess=direct_runner,
    )
    operator = CodeTest(CodeTestSettings(timeout_seconds=timeout_seconds))
    (metrics_request,) = operator.execution_requests(
        CodeArtifact(source=good_submission),
        {"task": task_json_artifact(task)},
    )

    assert len(builder_calls) == 2
    assert builder_calls[0] == builder_calls[1]
    assert direct_calls == [
        (
            metrics_request.source,
            metrics_request.input_text,
            metrics_request.timeout_seconds,
        )
    ]


def test_code_test_passing_counts_match_oracle(
    task, good_submission, local_runner
) -> None:
    oracle = evaluate_oracle(
        task,
        good_submission,
        timeout_seconds=5.0,
        run_in_subprocess=local_runner,
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_subprocess=local_runner,
    )[0]
    assert record.fact_values()["total_cases"] == oracle.total_cases
    assert record.fact_values()["passed_count"] == oracle.status_counts.get(
        "passed", 0
    )
    assert record.fact_values()["failed_count"] == oracle.status_counts.get(
        "failed", 0
    )
    assert record.fact_values()["error_count"] == oracle.status_counts.get(
        "error", 0
    )
    assert record.fact_values()["timeout_count"] == oracle.status_counts.get(
        "timeout", 0
    )
    assert (
        record.fact_values()["coverage_complete"] == oracle.coverage_complete
    )
    assert record.fact_values()["function_count"] == len(oracle.function_names)
    assert (
        record.fact_values()["best_function_name"] == oracle.best_function_name
    )


def test_code_test_failing_counts_match_oracle(
    task, failing_submission, local_runner
) -> None:
    oracle = evaluate_oracle(
        task,
        failing_submission,
        timeout_seconds=5.0,
        run_in_subprocess=local_runner,
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(failing_submission, task),
        run_in_subprocess=local_runner,
    )[0]
    assert record.fact_values()["passed_count"] == oracle.status_counts.get(
        "passed", 0
    )
    assert record.fact_values()["failed_count"] == oracle.status_counts.get(
        "failed", 0
    )


def test_code_test_kill_returncode_attributed_to_candidate(
    task, good_submission
) -> None:
    """A subprocess kill (returncode 137) is candidate data: all cases error."""
    from dr_code.execution.subprocess import SubprocessCompletedProcess

    def kill_runner(*, source, input_text, timeout_seconds):  # noqa: ANN001
        return SubprocessCompletedProcess(
            returncode=137, stdout="", stderr="killed"
        )

    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_subprocess=kill_runner,
    )[0]
    assert (
        record.fact_values()["error_count"]
        == record.fact_values()["total_cases"]
    )
    assert record.fact_values()["passed_count"] == 0


def test_code_test_nonzero_exit_attributed_to_candidate(
    task, good_submission
) -> None:
    """An unexpected nonzero returncode (e.g. ``os._exit(5)``) is
    candidate-controlled data: it becomes all-ERROR case statuses in a measured
    record, not an ``EvaluationHarnessError`` that aborts the batch."""
    from metrics.helpers import scripted_runner

    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_subprocess=scripted_runner(
            returncode=5, stdout="", stderr="boom"
        ),
    )[0]
    assert record.status.value == "measured"
    assert (
        record.fact_values()["error_count"]
        == record.fact_values()["total_cases"]
    )
    assert record.fact_values()["passed_count"] == 0


def test_code_test_malformed_stdout_attributed_to_candidate(
    task, good_submission
) -> None:
    """Malformed runner stdout (candidate shares the runner's stdout) is
    candidate data: it becomes all-ERROR case statuses in a measured record,
    not a batch-aborting error. Covers the JSON-decode / shape / case-id
    validation branches."""
    from metrics.helpers import scripted_runner

    for bad_stdout in (
        "this is not json{",  # JSON decode failure
        '{"not": "a list"}',  # wrong shape (object, not list)
        '[{"case_id": "case_0"}]',  # result schema validation failure
        '[{"case_id": "ghost", "status": "passed"}]',  # unknown case id
    ):
        record = _extract(
            _definition([_code_test_question()]),
            code_test_trace(good_submission, task),
            run_in_subprocess=scripted_runner(returncode=0, stdout=bad_stdout),
        )[0]
        assert record.status.value == "measured", bad_stdout
        assert (
            record.fact_values()["error_count"]
            == record.fact_values()["total_cases"]
        ), bad_stdout
        assert record.fact_values()["passed_count"] == 0, bad_stdout


def test_code_test_subprocess_error_still_propagates(
    task, good_submission
) -> None:
    """``SubprocessError`` is raised at the subprocess boundary before candidate code
    runs, so it remains the only propagating infrastructure path and still
    aborts the batch loudly -- it is not reclassified to case statuses."""
    import pytest

    from dr_code.execution.subprocess import SubprocessError

    from metrics.helpers import raising_runner

    with pytest.raises(SubprocessError):
        _extract(
            _definition([_code_test_question()]),
            code_test_trace(good_submission, task),
            run_in_subprocess=raising_runner(
                SubprocessError("boundary broke")
            ),
        )


def test_code_test_best_function_is_mechanical_max_passes(
    task, local_runner
) -> None:
    """best_function_name is an observation (max-passes), not a verdict — it
    stays in values; score/outcome stay in the consumer."""
    candidate = (
        "def add_one(x):\n    return x + 1\ndef decoy(x):\n    return x - 1\n"
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(candidate, task),
        run_in_subprocess=local_runner,
    )[0]
    assert record.fact_values()["best_function_name"] == task.entry_point
    assert record.fact_values()["function_count"] == 2
    assert "score" not in record.fact_values()
    assert "outcome" not in record.fact_values()


def test_code_test_partial_coverage_is_measured(task, good_submission) -> None:
    """Genuinely incomplete runner output (fewer results than cases, no
    failures) is a measured record, not an error, and coverage_complete is the
    fact "did every case produce a result" (False here) — a fact, not a
    verdict.

    coverage_complete is fact-shaped, matching the live oracle
    ``EvaluationTaskResult.coverage_complete`` (``result_count ==
    total_cases``); pass/fail thresholds belong in the policy consumer.
    """
    from metrics.helpers import partial_pass_runner_output, scripted_runner

    # Only case_0 is reported for a two-case task: genuine incomplete coverage.
    incomplete_output = partial_pass_runner_output(
        passed=("case_0",), case_ids=("case_0",)
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_subprocess=scripted_runner(stdout=incomplete_output),
    )[0]
    assert record.status.value == "measured"
    assert record.fact_values()["passed_count"] == 1
    assert record.fact_values()["failed_count"] == 0
    assert record.fact_values()["coverage_complete"] is False


def test_code_test_complete_coverage_with_failure_is_covered(
    task, good_submission
) -> None:
    """Complete coverage with a failing case: coverage_complete is True (every
    case produced a result) even though a case failed — the fact/verdict split.

    Matches the live oracle: for the same input (case_0 passed, case_1 failed,
    both reported) ``EvaluationTaskResult.coverage_complete`` is True and its
    outcome is ``tests_failed`` — the pass/fail threshold lives in the policy
    consumer, not in coverage_complete.
    """
    from metrics.helpers import partial_pass_runner_output, scripted_runner

    # Both cases reported, one failing: complete coverage, one failure.
    complete_with_failure = partial_pass_runner_output()
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_subprocess=scripted_runner(stdout=complete_with_failure),
    )[0]
    assert record.status.value == "measured"
    assert record.fact_values()["passed_count"] == 1
    assert record.fact_values()["failed_count"] == 1
    assert record.fact_values()["coverage_complete"] is True

"""Operator parity contracts (plan section 2 stubs + the existing-code map).

Each of the six operators ports the behaviour of a former
``dr_code.humaneval`` implementation:

* ``metrics.text_metrics``          → ``text_stats``
* ``metrics.python_leakage_metrics``→ ``code_leakage`` (task_names setting)
* ``metrics.ast_metrics``           → ``parse_outcome`` (parse_ok/error)
  + ``ast_stats`` (structure counts)
* ``humaneval/compression.py``      → ``compressed_length``
  (one codec+level per question, pinned levels)
* ``batch_runner.evaluate_human_eval_code`` → ``code_test`` (counts + attribution)

The old ``metrics.py`` / ``compression.py`` modules were the parity oracles
while the operators were being ported. Plan step 5 retires that path in one
coordinated break, so the pure-function oracles no longer exist. Per the plan
("moves ... tests, then deletes the old path — no aliases or shims"), parity
against those deleted oracles is now locked in as golden-value assertions on
fixed sample inputs; ``code_test`` still parity-checks against the live
``batch_runner`` oracle, which is kept.

Operators are engine-managed classes registered in ``REGISTRY`` and reached
only through ``extract_metrics`` (plan X-M4) — never called as bare functions.

``dr_code.metrics`` is imported lazily inside each test. Compression levels
are pinned explicitly (gzip 9, zstd 3) to reproduce today's implicit defaults
exactly (X-M2).
"""

from __future__ import annotations

import gzip

import zstandard

from dr_code.trace import CodeArtifact, TextArtifact, external_trace

from metrics.helpers import code_test_trace, evaluate_oracle

# gzip.compress and ZstdCompressor()'s implicit defaults were gzip level 9 and
# zstd level 3; the pinned questions must reproduce those exact sizes (X-M2).
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


def _definition(questions) -> object:
    from dr_code.metrics import MetricsDefinition

    return MetricsDefinition(
        definition_id="parity", version="1", questions=tuple(questions)
    )


def _question(metric_name: str, on: str = "input", **settings) -> object:
    from dr_code.metrics import MetricName, MetricQuestion

    return MetricQuestion(metric=MetricName(metric_name), on=on, settings=settings)


def _text_trace(text: str):
    return external_trace(
        {"input": TextArtifact(text=text), "output": TextArtifact(text=text)}
    )


def _code_trace(source: str):
    return external_trace(
        {"input": CodeArtifact(source=source), "output": CodeArtifact(source=source)}
    )


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    return extract_metrics(definition, trace, **kwargs)


def _value(record, key):
    assert record.status.value == "measured", record
    return record.values[key]


# ===========================================================================
# text_stats  ←  metrics.text_metrics
# ===========================================================================

# Golden values locked from the retired ``metrics.text_metrics`` oracle.
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


def test_text_stats_matches_text_metrics_field_for_field() -> None:
    record = _extract(_definition([_question("text_stats")]), _text_trace(SAMPLE_TEXT))[0]

    for field, expected in _TEXT_STATS_GOLDEN.items():
        assert _value(record, field) == expected, field


def test_text_stats_empty_text_matches_oracle() -> None:
    record = _extract(_definition([_question("text_stats")]), _text_trace(""))[0]
    assert _value(record, "character_count") == 0
    assert _value(record, "line_count") == 0


# ===========================================================================
# code_leakage  ←  metrics.python_leakage_metrics (task_names setting)
# ===========================================================================

# Golden values locked from the retired ``metrics.python_leakage_metrics``
# oracle, for task_names=("foo", "HumanEval/x").
_CODE_LEAKAGE_GOLDEN = {
    "keyword_count": 7,
    "code_marker_count": 4,
    "fenced_code_block_count": 1,
    "code_like_line_count": 2,
    "operator_count": 6,
    "punctuation_density": 0.1276595744680851,
    "task_name_hit_count": 1,
}


def test_code_leakage_matches_python_leakage_metrics_field_for_field() -> None:
    task_names = ("foo", "HumanEval/x")
    record = _extract(
        _definition([_question("code_leakage", task_names=list(task_names))]),
        _text_trace(SAMPLE_TEXT),
    )[0]

    for field, expected in _CODE_LEAKAGE_GOLDEN.items():
        assert _value(record, field) == expected, field


def test_code_leakage_task_names_are_part_of_identity() -> None:
    """task_names is a setting, hence part of metric identity (design L2)."""
    text = "def foo(x):\n    return foo(x)\n"
    none_rec = _extract(
        _definition([_question("code_leakage", task_names=[])]), _text_trace(text)
    )[0]
    named_rec = _extract(
        _definition([_question("code_leakage", task_names=["foo"])]),
        _text_trace(text),
    )[0]
    assert none_rec.values["task_name_hit_count"] == 0
    assert named_rec.values["task_name_hit_count"] >= 1


# ===========================================================================
# parse_outcome + ast_stats  ←  metrics.ast_metrics
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
    """parse_outcome is askable of raw text, not just CODE (plan: INPUT accepts
    TEXT or CODE)."""
    record = _extract(
        _definition([_question("parse_outcome")]), _text_trace("x = 1 + 2")
    )[0]
    assert _value(record, "parse_ok") is True


# Golden structure counts locked from the retired ``metrics.ast_metrics``
# oracle on SAMPLE_CODE.
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


def test_ast_stats_matches_ast_metrics_structure_counts() -> None:
    record = _extract(
        _definition([_question("ast_stats")]), _code_trace(SAMPLE_CODE)
    )[0]

    for field, expected in _AST_STATS_GOLDEN.items():
        assert _value(record, field) == expected, field


# ===========================================================================
# compressed_length  ←  humaneval/compression.py (pinned gzip 9 / zstd 3)
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
    """Pinned gzip level 9 reproduces today's implicit gzip.compress default."""
    reference = task.ground_truth_code
    # The retired oracle used gzip.compress()'s implicit default (level 9) and
    # divided the compressed size by the ground-truth byte length.
    expected_compressed = len(
        gzip.compress(SAMPLE_TEXT.encode("utf-8"), compresslevel=9)
    )
    ground_truth_bytes = len(reference.encode("utf-8"))
    expected_ratio = expected_compressed / ground_truth_bytes
    expected_percent_reduction = (1.0 - expected_ratio) * 100.0

    record = _extract(
        _definition(
            [_question("compressed_length", method="gzip", level=9, reference_key="reference")]
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
    """Pinned zstd level 3 reproduces today's ZstdCompressor() singleton."""
    reference = task.ground_truth_code

    record = _extract(
        _definition(
            [_question("compressed_length", method="zstd", level=3, reference_key="reference")]
        ),
        _reference_trace(SAMPLE_TEXT, reference),
    )[0]
    # The retired oracle used ZstdCompressor() with its implicit default level.
    assert record.values["compressed_bytes"] == len(
        zstandard.ZstdCompressor(level=_ZSTD_DEFAULT_LEVEL).compress(
            SAMPLE_TEXT.encode("utf-8")
        )
    )


def test_compressed_length_without_reference_has_no_ratio() -> None:
    """No reference_key ⇒ ratio stays None (empty-reference behaviour)."""
    record = _extract(
        _definition([_question("compressed_length", method="gzip", level=9)]),
        _text_trace(SAMPLE_TEXT),
    )[0]
    assert record.values["compressed_bytes"] == len(
        gzip.compress(SAMPLE_TEXT.encode("utf-8"), compresslevel=9)
    )
    assert "representation_bytes" in record.values
    assert record.values.get("ratio_to_reference") is None


def test_compressed_level_is_part_of_identity() -> None:
    """A different level is a different question (X-M2)."""
    trace = _text_trace(SAMPLE_TEXT)
    r6 = _extract(
        _definition([_question("compressed_length", method="gzip", level=1)]), trace
    )[0]
    r9 = _extract(
        _definition([_question("compressed_length", method="gzip", level=9)]), trace
    )[0]
    assert r9.values["compressed_bytes"] <= r6.values["compressed_bytes"]


# ===========================================================================
# code_test  ←  batch_runner.evaluate_human_eval_code (counts + attribution)
# ===========================================================================

def _code_test_question(timeout_seconds: float = 5.0) -> object:
    return _question("code_test", on="input", timeout_seconds=timeout_seconds)


def test_code_test_passing_counts_match_oracle(
    task, good_submission, local_runner
) -> None:
    oracle = evaluate_oracle(
        task, good_submission, timeout_seconds=5.0, run_in_sandbox=local_runner
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_sandbox=local_runner,
    )[0]
    assert record.values["total_cases"] == oracle.total_cases
    assert record.values["passed_count"] == oracle.status_counts.get("passed", 0)
    assert record.values["failed_count"] == oracle.status_counts.get("failed", 0)
    assert record.values["error_count"] == oracle.status_counts.get("error", 0)
    assert record.values["timeout_count"] == oracle.status_counts.get("timeout", 0)
    assert record.values["coverage_complete"] == oracle.coverage_complete
    assert record.values["function_count"] == len(oracle.function_names)
    assert record.values["best_function_name"] == oracle.best_function_name


def test_code_test_failing_counts_match_oracle(
    task, failing_submission, local_runner
) -> None:
    oracle = evaluate_oracle(
        task, failing_submission, timeout_seconds=5.0, run_in_sandbox=local_runner
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(failing_submission, task),
        run_in_sandbox=local_runner,
    )[0]
    assert record.values["passed_count"] == oracle.status_counts.get("passed", 0)
    assert record.values["failed_count"] == oracle.status_counts.get("failed", 0)


def test_code_test_kill_returncode_attributed_to_candidate(
    task, good_submission
) -> None:
    """A sandbox kill (returncode 137) is candidate data: all cases error."""
    from dr_code.humaneval.sandbox import SandboxCompletedProcess

    def kill_runner(*, source, input_json, timeout_seconds):  # noqa: ANN001
        return SandboxCompletedProcess(returncode=137, stdout="", stderr="killed")

    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_sandbox=kill_runner,
    )[0]
    assert record.values["error_count"] == record.values["total_cases"]
    assert record.values["passed_count"] == 0


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
        run_in_sandbox=scripted_runner(returncode=5, stdout="", stderr="boom"),
    )[0]
    assert record.status.value == "measured"
    assert record.values["error_count"] == record.values["total_cases"]
    assert record.values["passed_count"] == 0


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
        '[{"case_id": "ghost", "status": "passed"}]',  # unknown case id
    ):
        record = _extract(
            _definition([_code_test_question()]),
            code_test_trace(good_submission, task),
            run_in_sandbox=scripted_runner(returncode=0, stdout=bad_stdout),
        )[0]
        assert record.status.value == "measured", bad_stdout
        assert (
            record.values["error_count"] == record.values["total_cases"]
        ), bad_stdout
        assert record.values["passed_count"] == 0, bad_stdout


def test_code_test_sandbox_error_still_propagates(
    task, good_submission
) -> None:
    """``SandboxError`` is raised at the sandbox boundary before candidate code
    runs, so it remains the only propagating infrastructure path and still
    aborts the batch loudly -- it is not reclassified to case statuses."""
    import pytest

    from dr_code.humaneval.sandbox import SandboxError

    from metrics.helpers import raising_runner

    with pytest.raises(SandboxError):
        _extract(
            _definition([_code_test_question()]),
            code_test_trace(good_submission, task),
            run_in_sandbox=raising_runner(SandboxError("boundary broke")),
        )


def test_code_test_selector_parity_with_task_selector() -> None:
    """Parity guard for the duplicated best-function truth: the operator's
    ``_best_function_name`` and ``humaneval.task.select_best_function_name``
    agree over the same synthetic status sets. When the old scoring path
    retires, both copies and this guard go together."""
    from dr_code.humaneval.parsed_tests import HumanEvalTestCaseKind
    from dr_code.humaneval.task import (
        EvaluationCaseResult,
        EvaluationCaseStatus,
        select_best_function_name,
    )
    from dr_code.metrics.operators.code_test import (
        _best_function_name,
        _passed_counts,
    )

    P = EvaluationCaseStatus.PASSED
    F = EvaluationCaseStatus.FAILED

    scenarios = [
        # (function_names, entry_point, statuses_by_name)
        (["add_one"], "add_one", {"add_one": [P, P]}),
        (["add_one", "decoy"], "add_one", {"add_one": [P, P], "decoy": [F, F]}),
        # Decoy passes more cases: mechanical max ignores the entry point.
        (["add_one", "decoy"], "add_one", {"add_one": [P, F], "decoy": [P, P]}),
        # Tie on passes: entry-point tiebreak wins.
        (["add_one", "decoy"], "add_one", {"add_one": [P], "decoy": [P]}),
        # Tie, neither is the entry point: earliest index wins.
        (["a", "b"], "entry", {"a": [P], "b": [P]}),
        # No statuses recorded at all.
        (["a", "b"], "a", {}),
        ([], "a", {}),
    ]

    for function_names, entry_point, statuses_by_name in scenarios:
        results = [
            EvaluationCaseResult(
                task_id="t",
                case_id=f"{name}-{index}",
                function_name=name,
                status=status,
                test_type=HumanEvalTestCaseKind.INPUT_RESULT,
            )
            for name, statuses in statuses_by_name.items()
            for index, status in enumerate(statuses)
        ]
        operator_pick = _best_function_name(
            function_names=function_names,
            entry_point=entry_point,
            passed_counts=_passed_counts(statuses_by_name),
        )
        task_pick = select_best_function_name(
            function_names=function_names,
            entry_point=entry_point,
            results=results,
        )
        assert operator_pick == task_pick, (
            function_names,
            entry_point,
            statuses_by_name,
        )


def test_code_test_best_function_is_mechanical_max_passes(
    task, local_runner
) -> None:
    """best_function_name is an observation (max-passes), not a verdict — it
    stays in values; score/outcome stay in the consumer (X-M3)."""
    candidate = (
        "def add_one(x):\n    return x + 1\n"
        "def decoy(x):\n    return x - 1\n"
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(candidate, task),
        run_in_sandbox=local_runner,
    )[0]
    assert record.values["best_function_name"] == task.entry_point
    assert record.values["function_count"] == 2
    assert "score" not in record.values
    assert "outcome" not in record.values


def test_code_test_partial_coverage_is_measured(task, good_submission) -> None:
    """Genuinely incomplete runner output (fewer results than cases, no
    failures) is a measured record, not an error, and coverage_complete is the
    fact "did every case produce a result" (False here) — a fact, not a
    verdict (X-M3).

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
        run_in_sandbox=scripted_runner(stdout=incomplete_output),
    )[0]
    assert record.status.value == "measured"
    assert record.values["passed_count"] == 1
    assert record.values["failed_count"] == 0
    assert record.values["coverage_complete"] is False


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
        run_in_sandbox=scripted_runner(stdout=complete_with_failure),
    )[0]
    assert record.status.value == "measured"
    assert record.values["passed_count"] == 1
    assert record.values["failed_count"] == 1
    assert record.values["coverage_complete"] is True

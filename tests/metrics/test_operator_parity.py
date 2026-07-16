"""Operator parity contracts (plan section 2 stubs + the existing-code map).

Each of the six operators must reproduce the behaviour of the existing
``dr_code.humaneval`` implementation it ports. Those modules are the oracles:

* ``metrics.text_metrics``          → ``text_stats``
* ``metrics.python_leakage_metrics``→ ``code_leakage`` (task_names setting)
* ``metrics.ast_metrics``           → ``parse_outcome`` (parse_ok/error)
  + ``ast_stats`` (structure counts)
* ``humaneval/compression.py``      → ``compressed_length``
  (one codec+level per question, pinned levels)
* ``batch_runner.evaluate_human_eval_code`` → ``code_test`` (counts + attribution)

Operators are engine-managed classes registered in ``REGISTRY`` and reached
only through ``extract_metrics`` (plan X-M4) — never called as bare functions.

``dr_code.metrics`` is imported lazily inside each test. Compression levels
are pinned explicitly (gzip 9, zstd 3) to reproduce today's implicit defaults
exactly (X-M2).
"""

from __future__ import annotations

import gzip

from dr_code.humaneval.compression import (
    CompressionMethod,
    ZSTD_COMPRESSOR,
    compression_metrics,
)
from dr_code.humaneval.metrics import (
    ast_metrics,
    python_leakage_metrics,
    text_metrics,
)
from dr_code.trace import CodeArtifact, TextArtifact, external_trace

from metrics.helpers import code_test_trace, evaluate_oracle

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

def test_text_stats_matches_text_metrics_field_for_field() -> None:
    oracle = text_metrics(SAMPLE_TEXT)
    record = _extract(_definition([_question("text_stats")]), _text_trace(SAMPLE_TEXT))[0]

    for field in (
        "character_count",
        "byte_count",
        "line_count",
        "nonempty_line_count",
        "word_count",
        "average_word_length",
        "punctuation_count",
        "symbol_count",
    ):
        assert _value(record, field) == getattr(oracle, field), field


def test_text_stats_empty_text_matches_oracle() -> None:
    oracle = text_metrics("")
    record = _extract(_definition([_question("text_stats")]), _text_trace(""))[0]
    assert _value(record, "character_count") == oracle.character_count == 0
    assert _value(record, "line_count") == oracle.line_count == 0


# ===========================================================================
# code_leakage  ←  metrics.python_leakage_metrics (task_names setting)
# ===========================================================================

def test_code_leakage_matches_python_leakage_metrics_field_for_field() -> None:
    task_names = ("foo", "HumanEval/x")
    oracle = python_leakage_metrics(SAMPLE_TEXT, task_names=task_names)
    record = _extract(
        _definition([_question("code_leakage", task_names=list(task_names))]),
        _text_trace(SAMPLE_TEXT),
    )[0]

    for field in (
        "keyword_count",
        "code_marker_count",
        "fenced_code_block_count",
        "code_like_line_count",
        "operator_count",
        "punctuation_density",
        "task_name_hit_count",
    ):
        assert _value(record, field) == getattr(oracle, field), field


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
    oracle = ast_metrics(SAMPLE_CODE)
    assert oracle.parse_ok is True
    record = _extract(
        _definition([_question("parse_outcome")]), _code_trace(SAMPLE_CODE)
    )[0]
    assert _value(record, "parse_ok") is True


def test_parse_outcome_reports_parse_error_for_invalid_code() -> None:
    invalid = "def f(:\n    pass\n"
    assert ast_metrics(invalid).parse_ok is False
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


def test_ast_stats_matches_ast_metrics_structure_counts() -> None:
    oracle = ast_metrics(SAMPLE_CODE)
    assert oracle.parse_ok is True
    record = _extract(
        _definition([_question("ast_stats")]), _code_trace(SAMPLE_CODE)
    )[0]

    fields = (
        "top_level_function_count",
        "nested_function_count",
        "async_function_count",
        "lambda_count",
        "class_count",
        "import_count",
        "ast_node_count",
        "statement_count",
        "branch_count",
        "return_count",
        "yield_count",
        "call_count",
        "assignment_count",
        "comprehension_count",
        "literal_count",
        "max_branch_depth",
        "function_count",
        "total_argument_count",
        "positional_only_argument_count",
        "keyword_only_argument_count",
        "vararg_count",
        "kwarg_count",
        "decorated_function_count",
        "annotated_return_count",
        "docstring_function_count",
        "total_function_body_statement_count",
        "max_function_body_statement_count",
        "max_function_line_span",
    )
    for field in fields:
        assert _value(record, field) == getattr(oracle, field), field


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
    oracle = compression_metrics(
        ground_truth_code=reference, representation_text=SAMPLE_TEXT
    )[CompressionMethod.GZIP]

    record = _extract(
        _definition(
            [_question("compressed_length", method="gzip", level=9, reference_key="reference")]
        ),
        _reference_trace(SAMPLE_TEXT, reference),
    )[0]
    assert _value(record, "compressed_bytes") == oracle.compressed_bytes
    assert _value(record, "representation_bytes") == oracle.representation_bytes
    assert _value(record, "ratio_to_reference") == oracle.ratio_to_ground_truth
    assert _value(record, "percent_reduction") == (
        oracle.percent_reduction_vs_ground_truth
    )
    # and it matches raw gzip at level 9
    assert record.values["compressed_bytes"] == len(
        gzip.compress(SAMPLE_TEXT.encode("utf-8"), compresslevel=9)
    )


def test_compressed_length_zstd_level_3_reproduces_default(task) -> None:
    """Pinned zstd level 3 reproduces today's ZstdCompressor() singleton."""
    reference = task.ground_truth_code
    oracle = compression_metrics(
        ground_truth_code=reference, representation_text=SAMPLE_TEXT
    )[CompressionMethod.ZSTD]

    record = _extract(
        _definition(
            [_question("compressed_length", method="zstd", level=3, reference_key="reference")]
        ),
        _reference_trace(SAMPLE_TEXT, reference),
    )[0]
    assert _value(record, "compressed_bytes") == oracle.compressed_bytes
    assert record.values["compressed_bytes"] == len(
        ZSTD_COMPRESSOR.compress(SAMPLE_TEXT.encode("utf-8"))
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
    """Partial runner output (incomplete coverage, no failures) is a measured
    record, not an error — coverage_complete is the fact (X-M3)."""
    from metrics.helpers import partial_pass_runner_output, scripted_runner

    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_sandbox=scripted_runner(stdout=partial_pass_runner_output()),
    )[0]
    assert record.status.value == "measured"
    assert record.values["passed_count"] == 1
    assert record.values["coverage_complete"] is False

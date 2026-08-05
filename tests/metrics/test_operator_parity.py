"""Metric-operator output contracts.

The six registered operators cover text statistics, code leakage, parse
outcomes, AST statistics, compressed length, and HumanEval execution.

Pure operators are pinned to golden values on fixed sample inputs.
``code_test`` is cross-checked against
``runner.evaluate_humaneval_code`` for counts and attribution.

Operators are engine-managed classes registered in ``REGISTRY`` and reached
only through ``extract_metrics`` — never called as bare functions.

Compression levels are explicit: gzip level 9 and zstd level 3.
"""

from __future__ import annotations

import gzip

import pytest
import zstandard
from pydantic import ValidationError

from dr_code.trace import CodeArtifact, TextArtifact, external_trace

# Golden compressed sizes use gzip level 9 and zstd level 3.
_ZSTD_GOLDEN_LEVEL = 3

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

    return MetricQuestion(
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

    return extract_metrics(definition, trace, **kwargs)


def _facts(record):
    """The measured record's facts as a name-to-value mapping."""
    assert record.status.value == "measured", record
    return {fact.name: fact.value for fact in record.facts}


def _value(record, key):
    return _facts(record)[key]


# ===========================================================================
# text_stats
# ===========================================================================

# Golden text-statistic values for ``SAMPLE_TEXT``.
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


def test_text_stats_distinguishes_unicode_characters_from_utf8_bytes() -> None:
    record = _extract(
        _definition([_question("text_stats")]), _text_trace("é🙂")
    )[0]

    assert _value(record, "character_count") == 2
    assert _value(record, "byte_count") == 6


# ===========================================================================
# code_leakage with the task_names setting
# ===========================================================================

# Golden values for task_names=("foo", "HumanEval/x"). ``code_leakage`` uses
# the shared fence and code-like-line matching in ``dr_code.core.source.text_analysis``.
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
    assert _value(none_rec, "task_name_hit_count") == 0
    assert _value(named_rec, "task_name_hit_count") >= 1


# The shared ``dr_code.core.source.text_analysis`` regexes count indented lines, comments,
# and Python keywords as code-like, and match fences as whole lines. These
# values pin that shared heuristic contract for ``code_leakage``.
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
    """Pins code_leakage's use of the shared text_analysis regexes.

    ``CODE_LIKE_LINE_RE`` counts six code-like lines, including the comment
    and augmented assignment, and the whole-line fence matcher counts one
    fenced block.
    """
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


# Golden structure counts for ``SAMPLE_CODE``.
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


def test_ast_stats_positive_category_and_nested_depth_witnesses() -> None:
    source = (
        "import os\n"
        "class C:\n"
        "    @staticmethod\n"
        "    async def outer(a, /, *args, flag=True, **kwargs) -> int:\n"
        "        transform = lambda x: x + 1\n"
        "        async def nested():\n"
        "            yield 1\n"
        "        values = [transform(x) for x in args]\n"
        "        if flag:\n"
        "            for item in values:\n"
        "                while item:\n"
        "                    item -= 1\n"
        "        return sum(values)\n"
    )
    record = _extract(
        _definition([_question("ast_stats")]), _code_trace(source)
    )[0]
    facts = _facts(record)

    for name in (
        "nested_function_count",
        "async_function_count",
        "lambda_count",
        "class_count",
        "import_count",
        "yield_count",
        "call_count",
        "comprehension_count",
        "positional_only_argument_count",
        "keyword_only_argument_count",
        "vararg_count",
        "kwarg_count",
        "decorated_function_count",
        "annotated_return_count",
    ):
        assert facts[name] > 0, name
    assert facts["max_branch_depth"] == 3


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


def test_compressed_length_gzip_level_9_matches_golden(task) -> None:
    """Pinned gzip level 9 determines the compressed-length result."""
    reference = task.ground_truth_code
    # The ratio divides gzip-compressed bytes by ground-truth source bytes.
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


def test_compressed_length_zstd_level_3_matches_golden(task) -> None:
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
    assert _value(record, "compressed_bytes") == len(
        zstandard.ZstdCompressor(level=_ZSTD_GOLDEN_LEVEL).compress(
            SAMPLE_TEXT.encode("utf-8")
        )
    )


def test_compressed_length_without_reference_has_no_ratio() -> None:
    """Without a reference_key the reference columns are absent from the
    record entirely — no ``ratio_to_reference`` key is emitted."""
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
    assert _value(record, "compressed_bytes") == len(
        gzip.compress(SAMPLE_TEXT.encode("utf-8"), compresslevel=9)
    )
    assert "representation_bytes" in _facts(record)
    assert "ratio_to_reference" not in _facts(record)


@pytest.mark.parametrize(
    ("method", "level"),
    [
        pytest.param("gzip", 0, id="gzip-min"),
        pytest.param("gzip", 9, id="gzip-max"),
        pytest.param("zstd", -1, id="zstd-negative"),
        pytest.param("zstd", 1, id="zstd-min-positive"),
        pytest.param("zstd", 22, id="zstd-max"),
    ],
)
def test_compressed_length_accepts_valid_level_edges(
    method: str,
    level: int,
) -> None:
    question = _question(
        "compressed_length",
        compression={"method": method, "level": level},
    )

    record = _extract(_definition([question]), _text_trace(SAMPLE_TEXT))[0]

    assert record.status.value == "measured"
    assert _value(record, "compressed_bytes") > 0


@pytest.mark.parametrize(
    ("settings", "error_type", "error_loc"),
    [
        pytest.param(
            {"compression": {"method": "gzip", "level": -1}},
            "value_error",
            ("compression", "gzip"),
            id="gzip-below-min",
        ),
        pytest.param(
            {"compression": {"method": "gzip", "level": 10}},
            "value_error",
            ("compression", "gzip"),
            id="gzip-above-max",
        ),
        pytest.param(
            {"compression": {"method": "zstd", "level": 0}},
            "value_error",
            ("compression", "zstd"),
            id="zstd-zero",
        ),
        pytest.param(
            {"compression": {"method": "zstd", "level": 23}},
            "value_error",
            ("compression", "zstd"),
            id="zstd-above-max",
        ),
        pytest.param(
            {"compression": {"method": "brotli", "level": 1}},
            "union_tag_invalid",
            ("compression",),
            id="unknown-method",
        ),
        pytest.param(
            {
                "compression": {"method": "gzip", "level": 1},
                "reference_key": "",
            },
            "value_error",
            (),
            id="empty-reference-key",
        ),
    ],
)
def test_compressed_length_rejects_invalid_settings(
    settings: dict[str, object],
    error_type: str,
    error_loc: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _question("compressed_length", **settings)

    assert [
        (error["type"], error["loc"]) for error in exc_info.value.errors()
    ] == [(error_type, error_loc)]


def test_compressed_level_is_part_of_identity() -> None:
    """Compression level is persisted as part of metric identity."""
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    trace = _text_trace(SAMPLE_TEXT)
    level_1 = _extract(
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
    level_9 = _extract(
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
    restored_1 = METRIC_RECORD_ADAPTER.validate_json(level_1.model_dump_json())
    restored_9 = METRIC_RECORD_ADAPTER.validate_json(level_9.model_dump_json())

    assert restored_1.identity != restored_9.identity
    assert {
        setting.name: setting.value
        for setting in restored_1.identity.question.settings
    } == {
        "compression.method": "gzip",
        "compression.level": 1,
        "reference_key": None,
    }
    assert {
        setting.name: setting.value
        for setting in restored_9.identity.question.settings
    } == {
        "compression.method": "gzip",
        "compression.level": 9,
        "reference_key": None,
    }
    assert (
        restored_1.identity.question.settings
        == restored_1.identity.metrics_definition.questions[0].settings
    )
    assert (
        restored_9.identity.question.settings
        == restored_9.identity.metrics_definition.questions[0].settings
    )


# ===========================================================================
# code_test counts and attribution
# ===========================================================================


def _code_test_question(timeout_seconds: float = 5.0) -> object:
    return _question("code_test", on="input", timeout_seconds=timeout_seconds)


def test_code_test_passing_counts_match_oracle(
    task, good_submission, local_runner, code_test_trace, evaluate_oracle
) -> None:
    oracle = evaluate_oracle(
        task, good_submission, timeout_seconds=5.0, run_in_sandbox=local_runner
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_sandbox=local_runner,
    )[0]
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
    task, failing_submission, local_runner, code_test_trace, evaluate_oracle
) -> None:
    oracle = evaluate_oracle(
        task,
        failing_submission,
        timeout_seconds=5.0,
        run_in_sandbox=local_runner,
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(failing_submission, task),
        run_in_sandbox=local_runner,
    )[0]
    assert _value(record, "passed_count") == oracle.status_counts.get(
        "passed", 0
    )
    assert _value(record, "failed_count") == oracle.status_counts.get(
        "failed", 0
    )


def test_code_test_kill_returncode_attributed_to_candidate(
    task, good_submission, code_test_trace
) -> None:
    """A sandbox kill (returncode 137) is candidate data: all cases error."""
    from dr_code.core.execution.sandbox import SandboxCompletedProcess

    def kill_runner(*, source, input_json, timeout_seconds):  # noqa: ANN001
        return SandboxCompletedProcess(
            returncode=137, stdout="", stderr="killed"
        )

    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_sandbox=kill_runner,
    )[0]
    assert _value(record, "error_count") == _value(record, "total_cases")
    assert _value(record, "passed_count") == 0


def test_code_test_nonzero_exit_attributed_to_candidate(
    task, good_submission, code_test_trace, scripted_runner
) -> None:
    """An unexpected nonzero returncode (e.g. ``os._exit(5)``) is
    candidate-controlled data: it becomes all-ERROR case statuses in a measured
    record, not an ``EvaluationHarnessError`` that aborts the batch."""
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_sandbox=scripted_runner(returncode=5, stdout="", stderr="boom"),
    )[0]
    assert record.status.value == "measured"
    assert _value(record, "error_count") == _value(record, "total_cases")
    assert _value(record, "passed_count") == 0


def test_code_test_malformed_stdout_attributed_to_candidate(
    task, good_submission, code_test_trace, scripted_runner
) -> None:
    """Malformed runner stdout is candidate data: it becomes all-ERROR case
    statuses in a measured record, not a batch-aborting error. Covers the
    JSON-decode / shape / case-id validation branches.

    The runner reserves its results channel, so well-formed stdout is trusted
    output; malformed stdout still means a broken batch, and metrics attributes
    a broken batch to the candidate rather than aborting."""
    for bad_stdout in (
        "this is not json{",  # JSON decode failure
        '{"not": "a list"}',  # wrong shape (object, not list)
        '[{"case_id": "case_0"}]',  # result schema validation failure
        '[{"case_id": "ghost", "status": "passed"}]',  # unknown case id
    ):
        record = _extract(
            _definition([_code_test_question()]),
            code_test_trace(good_submission, task),
            run_in_sandbox=scripted_runner(returncode=0, stdout=bad_stdout),
        )[0]
        assert record.status.value == "measured", bad_stdout
        assert _value(record, "error_count") == _value(
            record, "total_cases"
        ), bad_stdout
        assert _value(record, "passed_count") == 0, bad_stdout


def test_code_test_sandbox_error_still_propagates(
    task, good_submission, code_test_trace, raising_runner
) -> None:
    """``SandboxError`` is raised at the sandbox boundary before candidate code
    runs, so it remains the only propagating infrastructure path and still
    aborts the batch loudly -- it is not reclassified to case statuses."""
    import pytest

    from dr_code.core.execution.sandbox import SandboxError

    with pytest.raises(SandboxError):
        _extract(
            _definition([_code_test_question()]),
            code_test_trace(good_submission, task),
            run_in_sandbox=raising_runner(SandboxError("boundary broke")),
        )


def test_code_test_requests_are_the_canonical_batch_request(task) -> None:
    """The operator does not build its own runner payload.

    Both scored paths reach the sandbox through
    ``runner.build_humaneval_batch_request``, so for one task, candidate,
    and function name the request the operator submits is byte-identical to the
    one the direct batch path submits. Equality on ``input_json`` is the guard
    that matters: it is the payload the runner parses, so any divergence in
    check construction, support code, or field naming shows up here.
    """
    from dr_code.humaneval.runner import build_humaneval_batch_request
    from dr_code.humaneval.metric_operator import CodeTest, CodeTestSettings
    from dr_code.trace import CodeArtifact, JsonArtifact

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


def test_code_test_function_names_come_from_the_shared_rule(task) -> None:
    """One top-level-function rule feeds both paths.

    The operator submits one request per name that
    ``runner.top_level_function_names`` returns, so the two paths cannot
    disagree about which functions get evaluated.
    """
    from dr_code.humaneval.runner import top_level_function_names
    from dr_code.humaneval.metric_operator import CodeTest, CodeTestSettings
    from dr_code.trace import CodeArtifact, JsonArtifact

    # Async and duplicate top-level names are both in scope of the rule.
    candidate = (
        "def add_one(x):\n    return x + 1\n"
        "async def fetch(x):\n    return x\n"
        "def add_one(x):\n    return x + 2\n"
        "class Ignored:\n    def method(self):\n        return 0\n"
    )
    operator = CodeTest(CodeTestSettings())
    requests = operator.execution_requests(
        CodeArtifact(source=candidate),
        {"task": JsonArtifact(payload=task.model_dump(mode="json"))},
    )

    names = top_level_function_names(candidate)
    assert names == ["add_one", "fetch", "add_one"]
    assert len(requests) == len(names)


def test_code_test_selection_is_the_task_selection_rule(
    task, local_runner, code_test_trace, evaluate_oracle
) -> None:
    """The operator's best-function fact is the task selector's answer.

    The operator builds an ``EvaluationTaskResult`` and reads
    ``best_function_name`` off it, so ``select_best_function_name`` is the only
    selection rule in the codebase. This pins the observable end of that: a
    decoy that passes more cases wins over the entry point, mechanically.
    """
    from dr_code.humaneval.task import select_best_function_name

    # ``decoy`` matches the task's expectations; the entry point does not.
    candidate = (
        "def add_one(x):\n    return x - 1\ndef decoy(x):\n    return x + 1\n"
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(candidate, task),
        run_in_sandbox=local_runner,
    )[0]
    oracle = evaluate_oracle(
        task, candidate, timeout_seconds=5.0, run_in_sandbox=local_runner
    )

    assert _value(record, "best_function_name") == "decoy"
    assert _value(record, "best_function_name") == oracle.best_function_name
    assert oracle.best_function_name == select_best_function_name(
        function_names=oracle.function_names,
        entry_point=task.entry_point,
        results=oracle.results,
    )


def test_code_test_best_function_is_mechanical_max_passes(
    task, local_runner, code_test_trace
) -> None:
    """best_function_name is an observation (max-passes), not a verdict — it
    stays in values; score/outcome stay in the consumer."""
    candidate = (
        "def add_one(x):\n    return x + 1\ndef decoy(x):\n    return x - 1\n"
    )
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(candidate, task),
        run_in_sandbox=local_runner,
    )[0]
    assert _value(record, "best_function_name") == task.entry_point
    assert _value(record, "function_count") == 2
    assert "score" not in _facts(record)
    assert "outcome" not in _facts(record)


def test_code_test_partial_coverage_is_measured(
    task,
    good_submission,
    code_test_trace,
    scripted_runner,
    partial_pass_runner_output,
) -> None:
    """Genuinely incomplete runner output (fewer results than cases, no
    failures) is a measured record, not an error, and coverage_complete is the
    fact "did every case produce a result" (False here) — a fact, not a
    verdict.

    coverage_complete is fact-shaped, matching the live oracle
    ``EvaluationTaskResult.coverage_complete`` (``result_count ==
    total_cases``); pass/fail thresholds belong in the policy consumer.
    """
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
    assert _value(record, "passed_count") == 1
    assert _value(record, "failed_count") == 0
    assert _value(record, "coverage_complete") is False


def test_code_test_complete_coverage_with_failure_is_covered(
    task,
    good_submission,
    code_test_trace,
    scripted_runner,
    partial_pass_runner_output,
) -> None:
    """Complete coverage with a failing case: coverage_complete is True (every
    case produced a result) even though a case failed — the fact/verdict split.

    Matches the live oracle: for the same input (case_0 passed, case_1 failed,
    both reported) ``EvaluationTaskResult.coverage_complete`` is True and its
    outcome is ``tests_failed`` — the pass/fail threshold lives in the policy
    consumer, not in coverage_complete.
    """
    # Both cases reported, one failing: complete coverage, one failure.
    complete_with_failure = partial_pass_runner_output()
    record = _extract(
        _definition([_code_test_question()]),
        code_test_trace(good_submission, task),
        run_in_sandbox=scripted_runner(stdout=complete_with_failure),
    )[0]
    assert record.status.value == "measured"
    assert _value(record, "passed_count") == 1
    assert _value(record, "failed_count") == 1
    assert _value(record, "coverage_complete") is True

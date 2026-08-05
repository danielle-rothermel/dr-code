"""Domain-neutral metric-operator output contracts.

The five domain-neutral operators cover text statistics, code leakage, parse
outcomes, AST statistics, and compressed length. They are pinned to golden
values on fixed sample inputs.

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

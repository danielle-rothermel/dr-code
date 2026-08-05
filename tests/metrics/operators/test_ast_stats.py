"""AST-statistics operator contracts."""

from __future__ import annotations

from dr_code.trace import CodeArtifact, external_trace

from ._helpers import (
    SAMPLE_CODE,
    _code_trace,
    _definition,
    _extract,
    _facts,
    _question,
    _value,
)

_q = _question


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


def test_ast_stats_raises_on_unparseable_code_instead_of_fabricating_zeros() -> (
    None
):
    """CodeArtifact documents "passed a compile check upstream", so unparseable
    CODE is a producer contract violation -- ast_stats must not mask it as an
    all-zero (indistinguishable from empty) measurement. It becomes an
    OPERATOR_FAILURE record, consistent with code_test's SyntaxError-on-parse
    behavior; parse facts stay the job of parse_outcome."""
    from dr_code.metrics import MetricName

    invalid = "def f(:\n    pass\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=invalid),
            "output": CodeArtifact(source=invalid),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    record = _extract(definition, trace)[0]
    assert record.status.value == "operator_failure"
    assert record.identity.question.metric is MetricName.AST_STATS
    assert not hasattr(record, "facts")

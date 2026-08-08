from __future__ import annotations

import pytest
from pydantic import ValidationError

from _humaneval_builders import _input_result_test
from dr_code.humaneval.parsed_tests import (
    HumanEvalTestCaseKind,
    ParsedTests,
    UnsupportedTestFormatError,
    parse_humaneval_tests,
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


def test_parsed_tests_are_structurally_immutable() -> None:
    parsed = parse_humaneval_tests(
        "def check(candidate):\n"
        "    inputs = [([1, 2],)]\n"
        "    results = [[2, 3]]\n"
        "    for inp, expected in zip(inputs, results):\n"
        "        assertion(candidate(*inp), expected)\n"
    )
    case = parsed.cases[0]

    assert isinstance(parsed.cases, tuple)
    with pytest.raises(ValidationError):
        parsed.cases = ()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        case.case_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        case.args.source = "[]"  # type: ignore[union-attr,misc]
    with pytest.raises(AttributeError):
        getattr(parsed.cases, "clear")()

    check = next(parsed.iter_checks(candidate_name="candidate"))
    assert check.code == "assertion(candidate(*[[1, 2]]), [2, 3], 0.0)"


def test_parsed_tests_round_trip_without_changing_generated_checks() -> None:
    parsed = parse_humaneval_tests(
        "def check(candidate):\n"
        "    inputs = [([1, 2],)]\n"
        "    results = [[2, 3]]\n"
        "    for inp, expected in zip(inputs, results):\n"
        "        assertion(candidate(*inp), expected)\n"
    )

    restored = ParsedTests.model_validate_json(parsed.model_dump_json())

    assert restored == parsed
    assert list(restored.iter_checks()) == list(parsed.iter_checks())


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

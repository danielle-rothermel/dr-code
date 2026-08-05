from __future__ import annotations

from ._helpers import (
    SAMPLE_TEXT,
    _definition,
    _extract,
    _question,
    _text_trace,
    _value,
)


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
    record = _extract(
        _definition([_question("code_leakage", task_names=[])]),
        _text_trace(_SHARED_HEURISTIC_SAMPLE),
    )[0]

    for field, expected in _SHARED_HEURISTIC_GOLDEN.items():
        assert _value(record, field) == expected, field

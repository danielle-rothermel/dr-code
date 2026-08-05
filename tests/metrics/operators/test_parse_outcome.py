"""Parse-outcome operator contracts."""

from __future__ import annotations

from ._helpers import (
    SAMPLE_CODE,
    _code_trace,
    _definition,
    _extract,
    _question,
    _text_trace,
    _value,
)


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

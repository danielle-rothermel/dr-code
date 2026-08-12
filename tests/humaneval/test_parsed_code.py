from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.core.source.python_analysis import validate_python_source
from dr_code.humaneval.parsed_code import ParsedCode, parse_code


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


def test_parsed_code_is_structurally_immutable() -> None:
    parsed = parse_code(
        display_title="fixture",
        code_str="def add_one(x: int) -> int:\n    return x + 1\n",
    )

    assert isinstance(parsed.signatures, tuple)
    assert isinstance(parsed.signatures[0].function_args, tuple)
    with pytest.raises(ValidationError):
        parsed.display_title = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        parsed.signatures[0].function_name = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        parsed.signatures[0].function_args[0].name = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        getattr(parsed.signatures, "append")(parsed.signatures[0])


def test_validate_python_source_reports_syntax_errors() -> None:
    validation = validate_python_source("def bad(x)\n  pass")

    assert validation.parse_ok is False
    assert validation.compile_ok is False
    assert validation.parse_error is not None
    assert validation.compile_error is not None

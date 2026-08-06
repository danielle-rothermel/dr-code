from __future__ import annotations

import pytest

from dr_code.core.source.text_transforms import (
    collapse_blank_runs,
    drop_after_last_return,
    drop_if_name,
    normalize_line_endings,
    normalize_text,
    recover_escaped_python,
    strip_code_fences,
    strip_markdown_wrappers,
    strip_trailing_whitespace,
    wrap_code_fence,
)

GARBAGE_INPUTS = (
    "",
    "def broken(:\n",
    "```\nunterminated fence",
    "plain prose, no code at all",
    "smart ‘quotes’ and “doubles”\r\nCRLF\ttabs  \n\n\n\n",
)

TOTAL_TRANSFORMS = (
    collapse_blank_runs,
    normalize_line_endings,
    normalize_text,
    strip_code_fences,
    strip_markdown_wrappers,
    strip_trailing_whitespace,
    wrap_code_fence,
)


@pytest.mark.parametrize(
    "transform", TOTAL_TRANSFORMS, ids=lambda fn: fn.__name__
)
@pytest.mark.parametrize("source", GARBAGE_INPUTS)
def test_text_transforms_are_total(transform, source: str) -> None:
    assert isinstance(transform(source), str)


def test_normalize_line_endings_converts_crlf_and_cr() -> None:
    assert normalize_line_endings("a\r\nb\rc\n") == "a\nb\nc\n"


def test_strip_trailing_whitespace_per_line() -> None:
    assert strip_trailing_whitespace("x = 1  \ny = 2\t\n") == "x = 1\ny = 2\n"


def test_collapse_blank_runs_to_one_blank_line() -> None:
    assert collapse_blank_runs("a\n\n\n\nb") == "a\n\nb"


def test_normalize_text_folds_crlf_tabs_unicode_and_blanks() -> None:
    raw = "ｄｅｆ f():\r\n\treturn 1  \r\n\r\n\r\n\r\nx = 2\r\n"
    out = normalize_text(raw)
    assert out == "def f():\n    return 1\n\nx = 2"


def test_wrap_then_strip_code_fences_round_trips() -> None:
    source = "def f():\n    return 1\n"
    assert strip_code_fences(wrap_code_fence(source)) == source


def test_wrap_code_fence_untagged_when_lang_empty() -> None:
    assert wrap_code_fence("x = 1", "") == "```\nx = 1\n```\n"


def test_strip_code_fences_leaves_unfenced_text_alone() -> None:
    assert strip_code_fences("x = 1\ny = 2") == "x = 1\ny = 2"


def test_strip_markdown_wrappers_removes_one_marker_per_line() -> None:
    text = "> quoted\n1. numbered\n- bullet\nplain"
    assert strip_markdown_wrappers(text) == "quoted\nnumbered\nbullet\nplain"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"prose\ndef f():\n\treturn 1", "def f():\n\treturn 1"),
        (r"prose\rdef f():\r\treturn 1", "def f():\n\treturn 1"),
        (
            r"prose\r\ndef f():\r\n\treturn 1",
            "def f():\n\treturn 1",
        ),
        (r'"def f():\n    return \"ok\""', 'def f():\n    return "ok"'),
    ],
)
def test_recover_escaped_python_decodes_supported_shapes(
    source: str,
    expected: str,
) -> None:
    assert recover_escaped_python(source) == expected


def test_recover_escaped_python_requires_python_anchor() -> None:
    assert recover_escaped_python(r"line one\nline two") is None
    assert recover_escaped_python(r"value\\n") is None
    assert recover_escaped_python("no escapes") is None


def test_recover_escaped_python_preserves_string_literal_escapes() -> None:
    source = (
        r"prose\ndef separators(values):\n"
        r'\treturn "\n".join(values), "\t", "\r"'
    )

    assert recover_escaped_python(source) == (
        'def separators(values):\n\treturn "\\n".join(values), "\\t", "\\r"'
    )


def test_drop_if_name_splits_before_main_guard() -> None:
    assert drop_if_name(
        "def f():\n    return 1\nif __name__ == '__main__':"
    ) == ["def f():\n    return 1\n"]


def test_drop_after_last_return_truncates_trailing_lines() -> None:
    assert drop_after_last_return("def f():\n    return 1\nprint('x')") == (
        "def f():\n    return 1\n"
    )


def test_drop_after_last_return_keeps_a_bracketed_return_whole() -> None:
    source = (
        "def f(x):\n"
        "    return (\n"
        "        x +\n"
        "        1\n"
        "    )\n"
        "Trailing prose.\n"
    )

    truncated = drop_after_last_return(source)

    assert truncated == (
        "def f(x):\n    return (\n        x +\n        1\n    )\n"
    )
    compile(truncated, "<salvaged>", "exec")


def test_drop_after_last_return_ignores_a_return_inside_a_docstring() -> None:
    source = 'def f():\n    """return 5"""\n    return 1\nProse.'

    assert drop_after_last_return(source) == (
        'def f():\n    """return 5"""\n    return 1\n'
    )


def test_drop_after_last_return_ignores_a_return_inside_a_comment() -> None:
    source = "def f():\n    # return 5\n    return 1\nProse."

    assert drop_after_last_return(source) == (
        "def f():\n    # return 5\n    return 1\n"
    )


def test_drop_after_last_return_ends_at_a_semicolon_terminated_return() -> (
    None
):
    assert drop_after_last_return("def f():\n    return 1; x = 2\nProse.") == (
        "def f():\n    return 1;"
    )


@pytest.mark.parametrize(
    ("reason", "source"),
    [
        ("no return at all", "def f():\n    pass\nProse."),
        ("return outside a function body", "return 1\nProse."),
        ("return never completes", "def f():\n    return (\n        1,\n"),
        ("not code at all", "plain prose, no code at all"),
        ("nothing to tokenize", ""),
        ("text the tokenizer cannot encode", "def f():\n    return '\ud800'"),
    ],
)
def test_drop_after_last_return_fails_closed(reason: str, source: str) -> None:
    assert drop_after_last_return(source) is None, reason


@pytest.mark.parametrize("source", GARBAGE_INPUTS)
def test_drop_after_last_return_never_raises(source: str) -> None:
    assert drop_after_last_return(source) in (None, source)

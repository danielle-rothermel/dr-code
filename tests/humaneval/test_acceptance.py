"""Tests for HumanEval candidate acceptance."""

from __future__ import annotations

import pytest

from dr_code.core.source.python_analysis import validate_python_source
from dr_code.humaneval.acceptance import extract_humaneval_code
from dr_code.preprocessing import PreprocessingFailureCode


@pytest.mark.parametrize(
    ("source", "expected_fragment"),
    [
        ("```python\ndef add_one(x):\n    return x + 1\n```", "def add_one"),
        ("> def add_one(x):\n>     return x + 1", "def add_one"),
        (
            "    def add_one(x):\n        return x + 1\n",
            "def add_one",
        ),
        (
            "def add_one(x):\n    return x + 1\nprint('trailing')\n",
            "return x + 1",
        ),
        (
            "def add_one(x):\n"
            "    return x + 1\n"
            "if __name__ == '__main__':\n"
            "    print(add_one(1))\n",
            "def add_one",
        ),
    ],
)
def test_extraction_accepts_known_submission_shapes(
    source: str,
    expected_fragment: str,
) -> None:
    result = extract_humaneval_code(source)

    assert result.succeeded
    assert expected_fragment in result.accepted_code
    assert validate_python_source(result.accepted_code).compile_ok
    # The name guard is split away during cleaning, so it never reaches an
    # accepted candidate.
    assert "if __name__" not in result.accepted_code


def test_trailing_statements_survive_alongside_their_salvage() -> None:
    # Truncating at the last return is additive: the candidate as written
    # is accepted, and its truncation is also present in the set rather
    # than having replaced it.
    result = extract_humaneval_code(
        "def add_one(x):\n    return x + 1\nprint('trailing')\n"
    )
    assert result.succeeded
    assert "print('trailing')" in result.accepted_code
    assert result.candidate_count == 2


def test_salvaged_candidate_still_gets_its_inferred_imports() -> None:
    # A submission whose only defect is trailing prose is unparseable until
    # the last-return salvage truncates it. Import inference is parse-driven
    # and no-ops on unparseable source, so it must run after the salvage --
    # otherwise the truncated candidate is accepted still referencing `np`
    # with no import, and fails at runtime with NameError.
    result = extract_humaneval_code(
        "def f(x):\n    return np.array(x)\nThis is trailing prose.\n"
    )

    assert result.succeeded
    assert "import numpy as np" in result.accepted_code
    assert validate_python_source(result.accepted_code).compile_ok


def test_marked_code_field_wins_over_code_in_another_marked_field() -> None:
    # A response that declares which part is its answer is answering
    # directly; scraping code out of arbitrary text is inference. When a
    # preceding field carries a fenced starter or reference function, the
    # scrape must not shadow the marked answer under an acceptance policy
    # that takes the lowest surviving ordinal.
    result = extract_humaneval_code(
        "[[ ## prompt ## ]]\n"
        "```python\n"
        "def add_one(x):\n"
        '    """Reference/starter."""\n'
        "    raise NotImplementedError\n"
        "```\n\n"
        "[[ ## code ## ]]\n"
        "def add_one(x):\n"
        "    return x + 1\n"
    )

    assert result.succeeded
    assert result.accepted_code == "def add_one(x):\n    return x + 1"
    # The starter is still extracted -- readings are ordered, not exclusive.
    assert result.candidate_count == 2


def test_marked_code_field_wins_even_when_it_wraps_the_answer() -> None:
    # A declared code field does not always hold bare source: the answer
    # may still be wrapped in prose or fences inside the field. The value
    # is contributed as written *and* segmented, so a wrapped declaration
    # still yields a parseable candidate ahead of any general scrape --
    # otherwise the wrapped value survives no filter and an earlier
    # field's starter takes ordinal 0.
    result = extract_humaneval_code(
        "[[ ## prompt ## ]]\n"
        "```python\n"
        "def add_one(x):\n"
        '    """Reference/starter."""\n'
        "    raise NotImplementedError\n"
        "```\n\n"
        "[[ ## code ## ]]\n"
        "Here is code:\n"
        "```python\n"
        "def add_one(x):\n"
        "    return x + 1\n"
        "```\n"
    )

    assert result.succeeded
    assert result.accepted_code == "def add_one(x):\n    return x + 1"


@pytest.mark.parametrize(
    "marked_value",
    (
        "> def add_one(x):\n>     return x + 1",
        "- def add_one(x):\n-     return x + 1",
    ),
    ids=("blockquote", "bullet"),
)
def test_marked_code_field_wins_when_wrapped_in_markdown(
    marked_value: str,
) -> None:
    # A declared value may carry any wrapper the general readings handle,
    # not just fences: a blockquoted or bulleted answer must still beat an
    # earlier field's starter.
    result = extract_humaneval_code(
        "[[ ## prompt ## ]]\n"
        "```python\n"
        "def add_one(x):\n"
        "    raise NotImplementedError\n"
        "```\n\n"
        f"[[ ## code ## ]]\n{marked_value}\n"
    )

    assert result.succeeded
    assert result.accepted_code == "def add_one(x):\n    return x + 1"


def test_json_code_field_wins_over_code_quoted_in_other_json_fields() -> None:
    result = extract_humaneval_code(
        '{"reasoning": "first I tried:\\n```python\\n'
        'def f(x):\\n    raise NotImplementedError\\n```", '
        '"code": "def f(x):\\n    return x + 1\\n"}'
    )

    assert result.succeeded
    assert result.accepted_code == "def f(x):\n    return x + 1"


def test_extraction_reports_blank_input_as_its_own_failure() -> None:
    for blank in ("", "   \n\t  "):
        result = extract_humaneval_code(blank)
        assert not result.succeeded
        assert result.failure_code == PreprocessingFailureCode.BLANK_INPUT


def test_extraction_supports_tilde_fences() -> None:
    source = "~~~python\ndef add_one(x):\n    return x + 1\n~~~"
    result = extract_humaneval_code(source)

    assert result.succeeded
    assert "def add_one" in result.accepted_code

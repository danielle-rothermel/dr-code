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

    assert "if __name__" not in result.accepted_code


def test_trailing_statements_survive_alongside_their_salvage() -> None:
    result = extract_humaneval_code(
        "def add_one(x):\n    return x + 1\nprint('trailing')\n"
    )
    assert result.succeeded
    assert "print('trailing')" in result.accepted_code
    assert result.candidate_count == 2


def test_salvaged_candidate_still_gets_its_inferred_imports() -> None:
    result = extract_humaneval_code(
        "def f(x):\n    return np.array(x)\nThis is trailing prose.\n"
    )

    assert result.succeeded
    assert "import numpy as np" in result.accepted_code
    assert validate_python_source(result.accepted_code).compile_ok


def test_extraction_preserves_same_local_import_in_sibling_functions() -> None:
    result = extract_humaneval_code(
        "def floor_value(x):\n"
        "    import math\n"
        "    return math.floor(x)\n\n"
        "def ceil_value(x):\n"
        "    import math\n"
        "    return math.ceil(x)\n"
    )

    assert result.succeeded
    assert result.accepted_code is not None
    namespace: dict[str, object] = {}
    exec(result.accepted_code, namespace)
    ceil_value = namespace["ceil_value"]
    assert callable(ceil_value)
    assert ceil_value(1.5) == 2


def test_marked_code_field_wins_over_code_in_another_marked_field() -> None:
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

    assert result.candidate_count == 2


def test_marked_code_field_wins_even_when_it_wraps_the_answer() -> None:
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

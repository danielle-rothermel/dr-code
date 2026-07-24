"""Profile selection agrees with the pipeline it selects over.

``extract_code_with_profile`` is not a second parsing implementation: every
profile routes the raw submission through the registered
``humaneval-function-candidates`` definition and then applies its
candidate-selection policy. So this file does not compare two parsers. It
pins the two properties that relationship must have:

* **Cross-API agreement** — whatever a profile returns is one of the
  candidates the pipeline produced for that same input, at the index the
  result reports. A profile may narrow, never invent or rewrite.
* **Profile divergence is selection-only** — the strict field-marker
  profile differs from best-effort exactly by which candidates it admits,
  over identical pipeline output.

Robustness across the corruption recipes is asserted here too, because it
is a property of the shared pipeline that both profiles inherit.
"""

from __future__ import annotations

import json
import random

import pytest

from dr_code.humaneval.code_parsing import (
    FIELD_MARKER_REPRESENTATION,
    NO_FIELD_MARKER_ERROR,
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    STRICT_FIELD_MARKER_PARSER_PROFILE,
    CandidateSelection,
    CodeParserProfile,
    extract_code_with_profile,
)
from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    run_preprocessing,
)
from dr_code.preprocessing.extraction import RESPONSE_REPRESENTATION_OPERATION
from dr_code.synthetic.corruption_recipes import RECIPES_BY_NAME, apply_recipe
from dr_code.trace import (
    OUTPUT_KEY,
    CodeCandidateSetArtifact,
    TextArtifact,
    is_absent,
)

SEED = 0

#: A HumanEval-shaped solution: an import (so import recipes are
#: meaningful), a loop, and local variables.
CLEAN = (
    "import numpy as np\n"
    "\n"
    "def make_array(values):\n"
    "    total = 0\n"
    "    for v in values:\n"
    "        total += v\n"
    "    return np.array([total, total])\n"
)


def _corrupt(recipe_name: str) -> str:
    recipe = RECIPES_BY_NAME[recipe_name]
    return apply_recipe(recipe, CLEAN, random.Random(SEED)).corrupted_source


#: Direct-shaped inputs exercising each extraction path plus cases where
#: no candidate survives.
_DIRECT_INPUTS: dict[str, str] = {
    "clean": CLEAN,
    "fenced": "```python\n" + CLEAN + "```\n",
    "fenced_untagged": "```\n" + CLEAN + "```\n",
    "prose_wrapped": "Here is the solution:\n\n```python\n"
    + CLEAN
    + "```\nDone.\n",
    "indented": "    " + CLEAN.replace("\n", "\n    ").rstrip() + "\n",
    "name_guard": CLEAN
    + "\nif __name__ == '__main__':\n    print(make_array([1]))\n",
    "escaped_json": '"import numpy as np\\n\\ndef f():\\n    return 1\\n"',
    # The four escaped-newline recovery shapes. Their behavioral home is
    # ``test_escaped_pipeline``; here they only have to survive selection.
    "escaped_newline_fenced": (
        r"Intro\n```python\ndef f():\n    return 1\n```"
    ),
    "escaped_newline_unfenced": r"Explanation:\ndef f():\n\treturn 1",
    "escaped_newline_mixed": "Intro\n"
    + r"```python\ndef f():\n    return 1\n```",
    "escaped_newline_json_string": (
        r'"Intro\n```python\ndef f():\n    return 1\n```"'
    ),
    "json_wrapped_markdown": json.dumps(
        "- def add(a, b):\n-     return a + b"
    ),
    "smart_quotes": "def greet():\n    return “hello”",
    "field_marker": "[[ ## code ## ]]\ndef f():\n    return 1\n",
    "field_marker_literal": "[[ ## code ## ]]\n{1: 2, 3: 4}\n",
    "field_marker_empty": "[[ ## code ## ]]\n\n",
    # Field-marker shapes that make the strict branch of the
    # selection-divergence test actually execute over several candidates.
    "field_marker_after_prose": (
        "Sure, here you go.\n[[ ## code ## ]]\ndef g(a):\n    return a * 2\n"
    ),
    "field_marker_fenced": ("[[ ## code ## ]]\n```python\n" + CLEAN + "```\n"),
    "field_marker_trailing_statement": (
        "[[ ## code ## ]]\ndef f():\n    return 1\nprint(f())\n"
    ),
    "field_marker_multi_function": (
        "[[ ## code ## ]]\n"
        "def first():\n    return 1\n"
        "\n"
        "def second():\n    return 2\n"
    ),
    # no-candidate cases
    "empty": "",
    "whitespace_only": "   \n\n  \t\n",
    "prose_only": "This is an explanation with no code whatsoever.\n",
    "no_field_marker": "def f():\n    return 1\n",
    "plain_literal_only": "{1: 2, 3: 4}\n",
    "code_repr_only": 'code = "def f(): pass"\n',
}
_CORRUPTION_INPUTS: dict[str, str] = {
    f"corrupt_{name}": _corrupt(name) for name in RECIPES_BY_NAME
}
_ALL_INPUTS: dict[str, str] = {**_DIRECT_INPUTS, **_CORRUPTION_INPUTS}

_PROFILES = [
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    STRICT_FIELD_MARKER_PARSER_PROFILE,
]


def _pipeline_candidates(raw: str) -> tuple[str, ...]:
    """The candidates the shared pipeline produces, before any selection."""
    output = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION, TextArtifact(text=raw)
    ).value(OUTPUT_KEY)
    if is_absent(output):
        return ()
    assert isinstance(output, CodeCandidateSetArtifact)
    return output.candidates


# --- cross-API agreement: selection never invents or rewrites -------------


@pytest.mark.parametrize(
    "profile",
    _PROFILES,
    ids=["best-effort", "field-marker"],
)
@pytest.mark.parametrize("input_name", sorted(_ALL_INPUTS))
def test_profile_result_is_a_pipeline_candidate(
    profile: CodeParserProfile, input_name: str
) -> None:
    raw = _ALL_INPUTS[input_name]
    candidates = _pipeline_candidates(raw)
    result = extract_code_with_profile(raw, profile=profile)

    assert result.candidate_count == len(candidates)
    if result.extracted_code is None:
        assert result.selected_candidate_index is None
        assert result.extraction_error is not None
        return
    assert result.selected_candidate_index is not None
    assert candidates[result.selected_candidate_index] == result.extracted_code


@pytest.mark.parametrize(
    "profile",
    _PROFILES,
    ids=["best-effort", "field-marker"],
)
def test_no_candidate_means_no_extraction(
    profile: CodeParserProfile,
) -> None:
    assert _pipeline_candidates("") == ()
    result = extract_code_with_profile("", profile=profile)
    assert result.extracted_code is None
    assert result.candidate_count == 0


# --- profiles differ by selection only -----------------------------------


def _field_marker_indexes(raw: str) -> tuple[int, ...]:
    """Candidate indexes whose lineage starts at the ``code`` field marker.

    Recomputes ``code_parsing._select_candidate``'s
    ``FIRST_FIELD_MARKER`` admissibility rule from the trace, independently
    of the selection code under test.
    """
    output = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION, TextArtifact(text=raw)
    ).value(OUTPUT_KEY)
    if is_absent(output):
        return ()
    assert isinstance(output, CodeCandidateSetArtifact)
    return tuple(
        index
        for index, lineage in enumerate(output.lineage)
        if any(
            origin.path[0].kind == RESPONSE_REPRESENTATION_OPERATION
            and origin.path[0].details.get("name")
            == FIELD_MARKER_REPRESENTATION
            for origin in lineage.origins
        )
    )


@pytest.mark.parametrize("input_name", sorted(_ALL_INPUTS))
def test_profiles_share_pipeline_output_and_differ_only_by_selection(
    input_name: str,
) -> None:
    raw = _ALL_INPUTS[input_name]
    candidates = _pipeline_candidates(raw)
    best = extract_code_with_profile(
        raw, profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE
    )
    strict = extract_code_with_profile(
        raw, profile=STRICT_FIELD_MARKER_PARSER_PROFILE
    )

    # Same pipeline, so the candidate pool is identical for both profiles.
    assert best.candidate_count == strict.candidate_count == len(candidates)

    # Best-effort takes the pipeline's first candidate, always.
    expected_best = candidates[0] if candidates else None
    assert best.extracted_code == expected_best
    assert best.selected_candidate_index == (0 if candidates else None)

    # Strict takes the FIRST candidate whose lineage starts at the ``code``
    # field marker, and nothing else. Both the chosen index and the absence
    # of any admissible candidate are pinned, so a selection policy that
    # returns any other candidate — or the first candidate unconditionally —
    # fails here.
    admissible = _field_marker_indexes(raw)
    expected_index = admissible[0] if admissible else None
    assert strict.selected_candidate_index == expected_index
    if expected_index is None:
        assert strict.extracted_code is None
        assert strict.extraction_error is not None
    else:
        assert strict.extracted_code == candidates[expected_index]
        assert strict.extraction_error is None
        # Divergence is selection-only: the strict pick is drawn from the
        # very same pool best-effort chose from.
        assert strict.extracted_code in candidates


def test_strict_profile_requires_the_field_marker() -> None:
    raw = "def f():\n    return 1\n"
    assert _pipeline_candidates(raw)  # the pipeline does find a candidate
    strict = extract_code_with_profile(
        raw, profile=STRICT_FIELD_MARKER_PARSER_PROFILE
    )
    assert strict.extracted_code is None
    assert strict.extraction_error == NO_FIELD_MARKER_ERROR
    best = extract_code_with_profile(
        raw, profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE
    )
    assert best.extracted_code == "def f():\n    return 1"


def test_field_marker_input_satisfies_both_profiles() -> None:
    raw = "[[ ## code ## ]]\ndef f():\n    return 1\n"
    expected = "def f():\n    return 1"
    for profile in _PROFILES:
        result = extract_code_with_profile(raw, profile=profile)
        assert result.extracted_code == expected, profile.profile_id


def test_registered_profiles_carry_distinct_selection_policies() -> None:
    assert BEST_EFFORT_HUMANEVAL_PARSER_PROFILE.selection is (
        CandidateSelection.FIRST
    )
    assert STRICT_FIELD_MARKER_PARSER_PROFILE.selection is (
        CandidateSelection.FIRST_FIELD_MARKER
    )


# --- pipeline robustness both profiles inherit ---------------------------


@pytest.mark.parametrize("recipe_name", sorted(RECIPES_BY_NAME))
def test_best_effort_recovers_corrupted_submissions(
    recipe_name: str,
) -> None:
    # ``inline_backticks`` interpolates backticks into the source itself;
    # no candidate compiles, and reporting extraction failure is correct.
    raw = _corrupt(recipe_name)
    result = extract_code_with_profile(
        raw, profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE
    )
    if recipe_name == "inline_backticks":
        assert result.extracted_code is None
        return
    assert result.extracted_code is not None, result.extraction_error
    assert "def make_array" in result.extracted_code


def test_smart_quote_delimiters_are_recovered() -> None:
    raw = "def greet():\n    return “hello”"
    result = extract_code_with_profile(
        raw, profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE
    )
    assert result.extracted_code == 'def greet():\n    return "hello"'


def test_smart_quotes_inside_literal_are_preserved() -> None:
    raw = 'def f():\n    return "don’t “quote” me"'
    result = extract_code_with_profile(
        raw, profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE
    )
    # Contents inside the ASCII-quoted literal are untouched, so the
    # smart-quote step is scoped to delimiters.
    assert result.extracted_code == raw


def test_shared_import_inference_preserves_bound_names() -> None:
    # Import inference must not inject a bogus import for a shadowed name.
    raw = "def solve(F):\n    return F + 1\n"
    result = extract_code_with_profile(
        raw, profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE
    )
    assert result.extracted_code is not None
    assert "import torch.nn.functional as F" not in result.extracted_code


def test_json_wrapped_markdown_is_recovered() -> None:
    raw = json.dumps("- def add(a, b):\n-     return a + b")
    result = extract_code_with_profile(
        raw, profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE
    )
    assert result.extracted_code == "def add(a, b):\n    return a + b"

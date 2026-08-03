"""Output parity between preprocessing and ``extract_code_with_profile``.

The named v2 definitions reproduce ``extract_code_with_profile`` output on
their shared cases, including every corruption recipe and both-absent cases.
The comparison covers extracted strings (parser ``None`` corresponds to a
preprocessing ``Absent``), not trace shapes, because the APIs record
provenance differently.

A small set of inputs has deliberately different behavior (string-aware
smart-quote recovery, the empty-fence drop, the field-marker code-repr
rejection). Those live in the divergence section below. The fourth extraction
strategy preserves parity on JSON-wrapped markdown, and both APIs share import
inference, so shadowed mapped names remain unchanged in each path.
"""

from __future__ import annotations

import json
import random

import pytest

from dr_code.humaneval.code_parsing import (
    extract_code_with_profile,
    resolve_parser_profile,
)
from dr_code.preprocessing import (
    PreprocessingDefinition,
    resolve_preprocessing_definition,
    run_preprocessing,
)
from dr_code.synthetic.corruption_recipes import RECIPES_BY_NAME, apply_recipe
from dr_code.trace import CodeArtifact, TextArtifact, is_absent

BEST_EFFORT_ID = "humaneval-best-effort"
FIELD_MARKER_ID = "humaneval-field-marker"
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


#: Direct-shaped inputs exercising each extraction path plus both-absent
#: cases. Inputs with intentionally distinct preprocessing behavior are
#: NOT here — they live in the divergence section.
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
    "json_wrapped_markdown": json.dumps(
        "- def add(a, b):\n-     return a + b"
    ),
    "field_marker": "[[ ## code ## ]]\ndef f():\n    return 1\n",
    "field_marker_literal": "[[ ## code ## ]]\n{1: 2, 3: 4}\n",
    "field_marker_empty": "[[ ## code ## ]]\n\n",
    # both-absent cases
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


def _preprocessing_output(
    definition: PreprocessingDefinition, raw: str
) -> str | None:
    output = run_preprocessing(definition, TextArtifact(text=raw)).value(
        "output"
    )
    if is_absent(output):
        return None
    assert isinstance(output, CodeArtifact)
    return output.source


def _parser_output(profile_id: str, version: str, raw: str) -> str | None:
    profile = resolve_parser_profile(
        parser_profile_id=profile_id, parser_version=version
    )
    return extract_code_with_profile(raw, profile=profile).extracted_code


_PROFILES = [
    (BEST_EFFORT_ID, "v2"),
    (FIELD_MARKER_ID, "v2"),
]


# --- intended-identical parity across v2 coordinates -----------------


@pytest.mark.parametrize(
    "profile_id, version",
    _PROFILES,
    ids=["best-effort-v2", "field-marker-v2"],
)
@pytest.mark.parametrize("input_name", sorted(_ALL_INPUTS))
def test_preprocessing_output_matches_parser_output(
    profile_id: str, version: str, input_name: str
) -> None:
    raw = _ALL_INPUTS[input_name]
    definition = resolve_preprocessing_definition(
        definition_id=profile_id, version=version
    )
    expected = _parser_output(profile_id, version, raw)
    actual = _preprocessing_output(definition, raw)
    assert actual == expected, (
        f"{profile_id}@{version} / {input_name}: "
        f"parser={expected!r} preprocessing={actual!r}"
    )


@pytest.mark.parametrize(
    "profile_id, version",
    _PROFILES,
    ids=["best-effort-v2", "field-marker-v2"],
)
def test_both_pipelines_absent_on_empty_input(
    profile_id: str, version: str
) -> None:
    definition = resolve_preprocessing_definition(
        definition_id=profile_id, version=version
    )
    assert _parser_output(profile_id, version, "") is None
    output = run_preprocessing(definition, TextArtifact(text="")).value(
        "output"
    )
    assert is_absent(output)


def test_fourth_rung_restores_parity_on_json_wrapped_markdown() -> None:
    # The escaped_markdown_wrapper strategy makes both APIs recover the
    # JSON-wrapped, markdown-list-wrapped code.
    raw = json.dumps("- def add(a, b):\n-     return a + b")
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    expected = "def add(a, b):\n    return a + b"
    assert _parser_output(BEST_EFFORT_ID, "v2", raw) == expected
    assert _preprocessing_output(definition, raw) == expected


def test_shared_import_inference_preserves_bound_names() -> None:
    # Both APIs use the same import-inference implementation, so neither
    # injects a bogus import for a shadowed name.
    raw = "def solve(F):\n    return F + 1\n"
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    parser = _parser_output(BEST_EFFORT_ID, "v2", raw)
    preprocessing = _preprocessing_output(definition, raw)
    assert parser == preprocessing
    assert preprocessing is not None
    assert "import torch.nn.functional as F" not in preprocessing


# --- intentional differences between the two APIs -------------------


def test_smart_quote_delimiters_recovered_only_by_preprocessing() -> None:
    raw = "def greet():\n    return “hello”"
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    preprocessing = _preprocessing_output(definition, raw)
    assert preprocessing == 'def greet():\n    return "hello"'
    # The parser does not normalize smart delimiters and cannot compile this
    # candidate.
    assert _parser_output(BEST_EFFORT_ID, "v2", raw) is None


def test_smart_quotes_inside_literal_preserved_by_preprocessing() -> None:
    raw = 'def f():\n    return "don’t “quote” me"'
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    preprocessing = _preprocessing_output(definition, raw)
    # Contents inside the ASCII-quoted literal are untouched; both pipelines
    # extract, so this is parity on contents but proves the step is scoped.
    assert preprocessing == raw
    assert _parser_output(BEST_EFFORT_ID, "v2", raw) == raw


def test_empty_fence_is_absent_only_in_preprocessing() -> None:
    raw = "```\n\n```"
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    output = run_preprocessing(definition, TextArtifact(text=raw)).value(
        "output"
    )
    assert is_absent(output)
    # The parser selects the empty string as extracted code.
    assert _parser_output(BEST_EFFORT_ID, "v2", raw) == ""


def test_field_marker_code_repr_rejected_only_in_preprocessing() -> None:
    raw = '[[ ## code ## ]]\ncode = "def f(): pass"\n'
    definition = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID, version="v2"
    )
    preprocessing = _preprocessing_output(definition, raw)
    assert preprocessing is None
    # The parser's field-marker path accepts this representation.
    assert (
        _parser_output(FIELD_MARKER_ID, "v2", raw) == 'code = "def f(): pass"'
    )

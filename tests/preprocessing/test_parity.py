"""Canonical preprocessing behavior over representative submission shapes."""

from __future__ import annotations

import json
import random

import pytest

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


#: Direct-shaped inputs exercising each extraction path plus both-absent cases.
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
    output = run_preprocessing(
        definition.materialize(), TextArtifact(text=raw)
    ).value("output")
    if is_absent(output):
        return None
    assert isinstance(output, CodeArtifact)
    return output.source


_PROFILES = [
    (BEST_EFFORT_ID, "v2"),
    (FIELD_MARKER_ID, "v2"),
]


# --- deterministic behavior across v2 coordinates -------------------


@pytest.mark.parametrize(
    "profile_id, version",
    _PROFILES,
    ids=["best-effort-v2", "field-marker-v2"],
)
@pytest.mark.parametrize("input_name", sorted(_ALL_INPUTS))
def test_canonical_output_is_deterministic(
    profile_id: str, version: str, input_name: str
) -> None:
    raw = _ALL_INPUTS[input_name]
    definition = resolve_preprocessing_definition(
        definition_id=profile_id, version=version
    )
    output = _preprocessing_output(definition, raw)
    assert output == _preprocessing_output(definition, raw)
    if output is not None:
        compile(output, f"<{input_name}>", "exec")


@pytest.mark.parametrize(
    "profile_id, version",
    _PROFILES,
    ids=["best-effort-v2", "field-marker-v2"],
)
def test_canonical_pipeline_is_absent_on_empty_input(
    profile_id: str, version: str
) -> None:
    definition = resolve_preprocessing_definition(
        definition_id=profile_id, version=version
    )
    output = run_preprocessing(
        definition.materialize(), TextArtifact(text="")
    ).value("output")
    assert is_absent(output)


def test_fourth_rung_recovers_json_wrapped_markdown() -> None:
    raw = json.dumps("- def add(a, b):\n-     return a + b")
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    expected = "def add(a, b):\n    return a + b"
    assert _preprocessing_output(definition, raw) == expected


def test_import_inference_does_not_import_shadowed_name() -> None:
    raw = "def solve(F):\n    return F + 1\n"
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    preprocessing = _preprocessing_output(definition, raw)
    assert preprocessing is not None
    assert "import torch.nn.functional as F" not in preprocessing


def test_smart_quote_delimiters_are_recovered() -> None:
    raw = "def greet():\n    return “hello”"
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    preprocessing = _preprocessing_output(definition, raw)
    assert preprocessing == 'def greet():\n    return "hello"'


def test_smart_quotes_inside_literal_are_preserved() -> None:
    raw = 'def f():\n    return "don’t “quote” me"'
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    preprocessing = _preprocessing_output(definition, raw)
    # Contents inside the ASCII-quoted literal are untouched.
    assert preprocessing == raw


def test_empty_fence_is_absent() -> None:
    raw = "```\n\n```"
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    output = run_preprocessing(
        definition.materialize(), TextArtifact(text=raw)
    ).value("output")
    assert is_absent(output)


def test_field_marker_code_repr_is_rejected() -> None:
    raw = '[[ ## code ## ]]\ncode = "def f(): pass"\n'
    definition = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID, version="v2"
    )
    preprocessing = _preprocessing_output(definition, raw)
    assert preprocessing is None

"""Output parity: named definitions vs. the old ``extract_code_with_profile``.

For every profile (best-effort v2, best-effort v1, field-marker v2/v1) and a
diverse input battery — including every corruption recipe and both-absent
cases — the *extracted output* of ``run_preprocessing`` over the named
definition must equal the *extracted output* of the old
``extract_code_with_profile``. We compare extracted strings only (``None``
old <-> ``Absent`` new), never trace shapes: the two pipelines record
provenance differently by design, and only the extracted code is a stable
contract.
"""

from __future__ import annotations

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
#: cases (empty / whitespace / prose-only / no-marker) where both pipelines
#: must produce nothing.
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
    "field_marker": "[[ ## code ## ]]\ndef f():\n    return 1\n",
    "field_marker_literal": "[[ ## code ## ]]\n{1: 2, 3: 4}\n",
    "field_marker_repr": '[[ ## code ## ]]\ncode = "def f(): pass"\n',
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


def _new_output(definition: PreprocessingDefinition, raw: str) -> str | None:
    output = run_preprocessing(
        definition, TextArtifact(text=raw)
    ).value("output")
    if is_absent(output):
        return None
    assert isinstance(output, CodeArtifact)
    return output.source


def _old_output(profile_id: str, version: str, raw: str) -> str | None:
    profile = resolve_parser_profile(
        parser_profile_id=profile_id, parser_version=version
    )
    return extract_code_with_profile(raw, profile=profile).extracted_code


_PROFILES = [
    (BEST_EFFORT_ID, "v2"),
    (BEST_EFFORT_ID, "v1"),
    (FIELD_MARKER_ID, "v2"),
    (FIELD_MARKER_ID, "v1"),
]


@pytest.mark.parametrize(
    "profile_id, version",
    _PROFILES,
    ids=["best-effort-v2", "best-effort-v1", "field-marker-v2", "field-marker-v1"],
)
@pytest.mark.parametrize("input_name", sorted(_ALL_INPUTS))
def test_output_parity_with_old_pipeline(
    profile_id: str, version: str, input_name: str
) -> None:
    raw = _ALL_INPUTS[input_name]
    definition = resolve_preprocessing_definition(
        definition_id=profile_id, version=version
    )
    expected = _old_output(profile_id, version, raw)
    actual = _new_output(definition, raw)
    assert actual == expected, (
        f"{profile_id}@{version} / {input_name}: "
        f"old={expected!r} new={actual!r}"
    )


@pytest.mark.parametrize(
    "profile_id, version",
    _PROFILES,
    ids=["best-effort-v2", "best-effort-v1", "field-marker-v2", "field-marker-v1"],
)
def test_both_pipelines_absent_on_empty_input(
    profile_id: str, version: str
) -> None:
    # The both-absent case is a real parity point: empty raw yields None
    # from the old pipeline and Absent from the new one.
    definition = resolve_preprocessing_definition(
        definition_id=profile_id, version=version
    )
    assert _old_output(profile_id, version, "") is None
    output = run_preprocessing(definition, TextArtifact(text="")).value(
        "output"
    )
    assert is_absent(output)

"""Output parity (and intended divergence) vs. ``extract_code_with_profile``.

The named v2 definitions reproduce the old ``extract_code_with_profile``
output on every input where the new behaviour is intended-identical — a
diverse battery including every corruption recipe and both-absent cases. We
compare extracted strings only (``None`` old <-> ``Absent`` new), never trace
shapes: the two pipelines record provenance differently by design.

A small set of inputs is *deliberately* different now (string-aware
smart-quote recovery, the empty-fence drop, the field-marker code-repr
rejection). Those live in the divergence section below, which asserts the new
behaviour and, where cheap, that the old pipeline differs. Two changes that
might look divergent are not: the fourth extraction rung *restores* parity on
the JSON-wrapped markdown case, and the import-inference fix flows into the
old pipeline via delegation — so both hold parity.
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
#: cases. Inputs whose behaviour the new pipeline intentionally changes are
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


def _new_output(definition: PreprocessingDefinition, raw: str) -> str | None:
    output = run_preprocessing(definition, TextArtifact(text=raw)).value(
        "output"
    )
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
    (FIELD_MARKER_ID, "v2"),
]


# --- intended-identical parity across v2 coordinates -----------------


@pytest.mark.parametrize(
    "profile_id, version",
    _PROFILES,
    ids=["best-effort-v2", "field-marker-v2"],
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
    ids=["best-effort-v2", "field-marker-v2"],
)
def test_both_pipelines_absent_on_empty_input(
    profile_id: str, version: str
) -> None:
    definition = resolve_preprocessing_definition(
        definition_id=profile_id, version=version
    )
    assert _old_output(profile_id, version, "") is None
    output = run_preprocessing(definition, TextArtifact(text="")).value(
        "output"
    )
    assert is_absent(output)


def test_fourth_rung_restores_parity_on_json_wrapped_markdown() -> None:
    # Item 1: the escaped_markdown_wrapper rung makes both pipelines recover
    # the JSON-wrapped, markdown-list-wrapped code — parity, not divergence.
    raw = json.dumps("- def add(a, b):\n-     return a + b")
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    expected = "def add(a, b):\n    return a + b"
    assert _old_output(BEST_EFFORT_ID, "v2", raw) == expected
    assert _new_output(definition, raw) == expected


def test_import_inference_fix_holds_parity_via_delegation() -> None:
    # Item 7: the bound-name fix lives in the shared module the old pipeline
    # delegates to, so neither injects a bogus import for a shadowed name.
    raw = "def solve(F):\n    return F + 1\n"
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    old = _old_output(BEST_EFFORT_ID, "v2", raw)
    new = _new_output(definition, raw)
    assert old == new
    assert new is not None
    assert "import torch.nn.functional as F" not in new


# --- intended divergences: new behaviour, old pipeline differs -------


def test_smart_quote_delimiters_recovered_only_by_new_pipeline() -> None:
    raw = "def greet():\n    return “hello”"
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    new = _new_output(definition, raw)
    assert new == 'def greet():\n    return "hello"'
    # The old pipeline never normalizes smart delimiters, so it can't compile
    # this candidate and gives up.
    assert _old_output(BEST_EFFORT_ID, "v2", raw) is None


def test_smart_quotes_inside_literal_preserved_by_new_pipeline() -> None:
    raw = 'def f():\n    return "don’t “quote” me"'
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    new = _new_output(definition, raw)
    # Contents inside the ASCII-quoted literal are untouched; both pipelines
    # extract, so this is parity on contents but proves the step is scoped.
    assert new == raw
    assert _old_output(BEST_EFFORT_ID, "v2", raw) == raw


def test_empty_fence_is_absent_only_in_new_pipeline() -> None:
    raw = "```\n\n```"
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )
    output = run_preprocessing(definition, TextArtifact(text=raw)).value(
        "output"
    )
    assert is_absent(output)
    # The old pipeline selects the empty string as the extracted code.
    assert _old_output(BEST_EFFORT_ID, "v2", raw) == ""


def test_field_marker_code_repr_rejected_only_in_new_pipeline() -> None:
    raw = '[[ ## code ## ]]\ncode = "def f(): pass"\n'
    definition = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID, version="v2"
    )
    new = _new_output(definition, raw)
    assert new is None
    # The old field-marker path lacks the code-repr filter, so it accepts it.
    assert _old_output(FIELD_MARKER_ID, "v2", raw) == 'code = "def f(): pass"'

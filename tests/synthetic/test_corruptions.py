"""Functional invariants for every synthetic corruption."""

from __future__ import annotations

import ast
import random
import unicodedata

import pytest

from dr_code.synthetic.corruptions import REGISTRY
from dr_code.synthetic.corruptions.add_code_fences import (
    AddCodeFences,
    AddCodeFencesSettings,
)
from dr_code.synthetic.corruptions.base import CorruptionSettings
from dr_code.synthetic.names import CorruptionName, FenceLangTag

SOURCE = (
    "import math\n"
    "from collections import Counter\n"
    "\n"
    "def greet(name):\n"
    '    message = f"hello {name}"\n'
    "    total = len(message)\n"
    "    return message\n"
)
SEED = 42


def _apply(
    name: CorruptionName,
    source: str = SOURCE,
    *,
    seed: int = SEED,
) -> str:
    transform = REGISTRY[name.value]()
    return transform.apply(source, random.Random(seed)).corrupted_source


@pytest.mark.parametrize("name", CorruptionName)
def test_corruption_is_deterministic_for_seed(name: CorruptionName) -> None:
    assert _apply(name) == _apply(name)


@pytest.mark.parametrize(
    "tag, opening",
    [
        (FenceLangTag.PYTHON, "```python\n"),
        (FenceLangTag.PY, "```py\n"),
        (FenceLangTag.PYTHON3, "```python3\n"),
        (FenceLangTag.NONE, "```\n"),
    ],
)
def test_add_code_fences_emits_its_configured_tag(
    tag: FenceLangTag, opening: str
) -> None:
    transform = AddCodeFences(AddCodeFencesSettings(language_tag=tag))
    corrupted = transform.apply(SOURCE, random.Random(SEED)).corrupted_source

    assert corrupted.count("```") == 2
    assert corrupted.startswith(opening)
    assert SOURCE in corrupted
    assert corrupted.endswith("```\n")


def test_add_code_fences_ignores_rng_state() -> None:
    transform = AddCodeFences(
        AddCodeFencesSettings(language_tag=FenceLangTag.PYTHON)
    )

    assert transform.apply(SOURCE, random.Random(1)).corrupted_source == (
        transform.apply(SOURCE, random.Random(2)).corrupted_source
    )


def test_add_prose_wrapper_preserves_source_between_prose() -> None:
    corrupted = _apply(CorruptionName.ADD_PROSE_WRAPPER)

    assert SOURCE.rstrip() in corrupted
    assert not corrupted.startswith(SOURCE)
    assert corrupted.endswith("\n")


def test_add_smart_quotes_replaces_ascii_delimiters() -> None:
    corrupted = _apply(
        CorruptionName.ADD_SMART_QUOTES,
        'answer = "yes"\n',
    )

    assert corrupted == "answer = “yes”\n"


def test_add_indentation_applies_uniform_left_margin() -> None:
    corrupted = _apply(CorruptionName.ADD_INDENTATION)
    source_lines = SOURCE.splitlines()
    corrupted_lines = corrupted.splitlines()
    added_indents = {
        len(after) - len(before)
        for before, after in zip(source_lines, corrupted_lines, strict=True)
        if before
    }

    assert added_indents == {8}
    assert [line.lstrip() for line in corrupted_lines] == [
        line.lstrip() for line in source_lines
    ]


def test_add_tabs_replaces_leading_space_groups() -> None:
    corrupted = _apply(CorruptionName.ADD_TABS)

    assert "\n\tmessage =" in corrupted
    assert "\n    message =" not in corrupted
    assert corrupted.expandtabs(4) == SOURCE


def test_add_trailing_whitespace_changes_only_line_suffixes() -> None:
    corrupted = _apply(CorruptionName.ADD_TRAILING_WHITESPACE)

    assert all(
        not line or line.endswith("   ") for line in corrupted.splitlines()
    )
    assert (
        "\n".join(line.rstrip() for line in corrupted.splitlines())
        == SOURCE.rstrip()
    )


def test_add_crlf_replaces_every_line_ending() -> None:
    corrupted = _apply(CorruptionName.ADD_CRLF)

    assert corrupted.replace("\r\n", "\n") == SOURCE
    assert "\n" not in corrupted.replace("\r\n", "")


def test_add_unicode_noise_decomposes_unicode_without_changing_text() -> None:
    source = 'label = "café"\n'
    corrupted = _apply(CorruptionName.ADD_UNICODE_NOISE, source)

    assert corrupted != source
    assert unicodedata.normalize("NFC", corrupted) == source
    assert unicodedata.is_normalized("NFD", corrupted)


def test_add_blank_lines_preserves_nonblank_lines_in_order() -> None:
    corrupted = _apply(CorruptionName.ADD_BLANK_LINES)

    def nonblank(text: str) -> list[str]:
        return [line for line in text.splitlines() if line.strip()]

    assert nonblank(corrupted) == nonblank(SOURCE)
    assert len(corrupted.splitlines()) > len(SOURCE.splitlines())


def test_add_markdown_wrappers_prefixes_every_nonblank_line() -> None:
    corrupted = _apply(CorruptionName.ADD_MARKDOWN_WRAPPERS)

    assert all(
        not line or line.startswith("- ") for line in corrupted.splitlines()
    )
    assert corrupted.replace("- ", "") == SOURCE


def test_add_inline_backticks_wraps_complete_source() -> None:
    assert _apply(CorruptionName.ADD_INLINE_BACKTICKS) == (
        f"`{SOURCE.rstrip()}`"
    )


def test_truncate_produces_incomplete_source() -> None:
    corrupted = _apply(CorruptionName.TRUNCATE)

    assert corrupted != SOURCE
    with pytest.raises(SyntaxError):
        ast.parse(corrupted)


def test_remove_imports_keeps_program_without_top_level_imports() -> None:
    corrupted = _apply(CorruptionName.REMOVE_IMPORTS)
    tree = ast.parse(corrupted)

    assert not any(
        isinstance(node, ast.Import | ast.ImportFrom) for node in tree.body
    )
    assert "def greet(name):" in corrupted


def test_mangle_import_lines_breaks_an_import_not_the_function() -> None:
    corrupted = _apply(CorruptionName.MANGLE_IMPORT_LINES)

    assert "def greet(name):" in corrupted
    with pytest.raises(SyntaxError):
        ast.parse(corrupted)


def test_duplicate_imports_duplicates_each_top_level_import() -> None:
    corrupted = _apply(CorruptionName.DUPLICATE_IMPORTS)
    imports = [
        node
        for node in ast.parse(corrupted).body
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]

    assert len(imports) == 4
    assert corrupted.count("import math\n") == 2
    assert corrupted.count("from collections import Counter\n") == 2


def test_add_multiple_solutions_preserves_both_fenced_options() -> None:
    corrupted = _apply(CorruptionName.ADD_MULTIPLE_SOLUTIONS)

    assert corrupted.count("```") == 4
    assert SOURCE in corrupted
    assert "def _alt_solution():" in corrupted


def test_add_comments_noise_preserves_python_ast() -> None:
    corrupted = _apply(CorruptionName.ADD_COMMENTS_NOISE)

    assert "# Step-by-step solution" in corrupted
    assert ast.dump(ast.parse(corrupted)) == ast.dump(ast.parse(SOURCE))


def test_add_dead_code_prepends_unused_import() -> None:
    corrupted = _apply(CorruptionName.ADD_DEAD_CODE)

    assert corrupted.startswith("import os as _unused_module")
    assert corrupted.endswith(SOURCE)


def test_change_quote_style_preserves_string_value() -> None:
    source = 'answer = "yes"\n'
    corrupted = _apply(CorruptionName.CHANGE_QUOTE_STYLE, source)

    assert "'yes'" in corrupted
    assert ast.dump(ast.parse(corrupted)) == ast.dump(ast.parse(source))


def test_change_string_form_preserves_runtime_result() -> None:
    corrupted = _apply(CorruptionName.CHANGE_STRING_FORM)
    original_namespace: dict[str, object] = {}
    corrupted_namespace: dict[str, object] = {}

    exec(SOURCE, original_namespace)
    exec(corrupted, corrupted_namespace)

    assert '".format(name)' in corrupted
    assert original_namespace["greet"]("Ada") == corrupted_namespace["greet"](
        "Ada"
    )


def test_add_type_annotations_prepends_annotated_assignment() -> None:
    corrupted = _apply(CorruptionName.ADD_TYPE_ANNOTATIONS)

    assert isinstance(ast.parse(corrupted).body[0], ast.AnnAssign)
    assert corrupted.endswith(SOURCE)


def test_rename_locals_preserves_signature_and_runtime_result() -> None:
    corrupted = _apply(CorruptionName.RENAME_LOCALS)
    function = next(
        node
        for node in ast.parse(corrupted).body
        if isinstance(node, ast.FunctionDef)
    )
    original_namespace: dict[str, object] = {}
    corrupted_namespace: dict[str, object] = {}

    exec(SOURCE, original_namespace)
    exec(corrupted, corrupted_namespace)

    assert [argument.arg for argument in function.args.args] == ["name"]
    assert "_v0" in corrupted
    assert original_namespace["greet"]("Ada") == corrupted_namespace["greet"](
        "Ada"
    )


def test_rename_locals_preserves_fresh_module_binding_runtime_result() -> None:
    source = (
        "_v0: int = 10\ndef run():\n    value = 3\n    return value + _v0\n"
    )
    corrupted = _apply(CorruptionName.RENAME_LOCALS, source)
    original_namespace: dict[str, object] = {}
    corrupted_namespace: dict[str, object] = {}

    exec(
        compile(source, "<rename-locals-original>", "exec"), original_namespace
    )
    exec(
        compile(corrupted, "<rename-locals-corrupted>", "exec"),
        corrupted_namespace,
    )
    original = original_namespace["run"]
    transformed = corrupted_namespace["run"]
    assert callable(original)
    assert callable(transformed)
    assert original() == transformed()


def test_registry_covers_all_named_corruptions() -> None:
    assert set(REGISTRY) == {corruption.value for corruption in CorruptionName}


def test_every_registered_corruption_declares_its_own_version() -> None:
    assert {name: corruption.VERSION for name, corruption in REGISTRY.items()}
    assert {corruption.VERSION for corruption in REGISTRY.values()} == {"0"}
    assert all(
        "VERSION" in corruption.__dict__ for corruption in REGISTRY.values()
    )


@pytest.mark.parametrize("name", CorruptionName)
def test_registered_corruption_settings_model_is_frozen(
    name: CorruptionName,
) -> None:
    settings_model = REGISTRY[name.value].Settings

    assert issubclass(settings_model, CorruptionSettings)
    assert settings_model.model_config["frozen"] is True

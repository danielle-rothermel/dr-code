"""Optional composed identifier rename for a mutant program.

A deliberately optional (flagged) transformation: rename the task's entry-point
function to a neutral target name with all-occurrence coverage (definition,
recursive self-calls, and any other references). This composes with a
behavioral
mutation to additionally break surface-level name memorization without changing
behavior. Kept behavior-preserving (a pure rename) so it never affects the
oracle outcomes; the mutant's expected outputs are unchanged by the rename.

Publication-hardening TODO: broaden to argument/local renames and to
docstring/spec regeneration for direct-generation arms (not built here).
"""

from __future__ import annotations

import ast

DEFAULT_TARGET_NAME = "target_fxn"


class RenameError(ValueError):
    """The entry point was not found or the source did not parse."""


def rename_entry_point(
    source: str, *, entry_point: str, target_name: str = DEFAULT_TARGET_NAME
) -> str:
    """Rename ``entry_point`` to ``target_name`` with all-occurrence coverage.

    Renames the top-level function definition and every ``Name`` reference to
    it (including recursive calls). Behavior-preserving. Raises
    :class:`RenameError` if no top-level function named ``entry_point`` exists.
    """

    if entry_point == target_name:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RenameError(f"source does not parse: {exc}") from exc

    if not _has_top_level_function(tree, entry_point):
        raise RenameError(f"no top-level function named {entry_point!r}")

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
        ):
            node.name = target_name
        elif isinstance(node, ast.Name) and node.id == entry_point:
            node.id = target_name
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _has_top_level_function(tree: ast.Module, name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
        for node in tree.body
    )


__all__ = ["DEFAULT_TARGET_NAME", "RenameError", "rename_entry_point"]

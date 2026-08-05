"""Module-shape classifications for candidate sources.

``CandidateInspection`` records structure only — whether a source parses and
compiles, and what it defines at module level. Whether a module that merely
holds a literal, or assigns a code string to a ``code`` name, counts as an
extracted solution is a policy question, and policy lives in filters rather
than in the trace layer.

Answering those two questions needs the module body, which the inspection
does not carry, so this module parses. It parses at most once per distinct
source: both classifications come from one cached tree, so adding the second
filter to a definition costs no additional parse. Parsing here is not a
second opinion on whether a source parses — a source that does not parse is
classified as neither shape, and the compilability filter is what rejects it.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from typing import Final

#: The name a code-representation assignment binds. A module whose whole
#: body is ``code = "..."`` is a description of code, not code.
CODE_REPRESENTATION_NAME: Final[str] = "code"

#: Parsed-tree cache bound to the process, keyed on exact source text.
#: Sized to hold a full candidate set many times over; candidate sources are
#: short and the pipeline is single-pass, so eviction is not load-bearing.
_TREE_CACHE_SIZE: Final[int] = 512


@lru_cache(maxsize=_TREE_CACHE_SIZE)
def _tree(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _sole_statement(source: str) -> ast.stmt | None:
    """``source``'s only module-level statement, when it has exactly one."""
    tree = _tree(source)
    if tree is None or len(tree.body) != 1:
        return None
    return tree.body[0]


def is_plain_literal_module(source: str) -> bool:
    """True when ``source`` is one bare dict, list, set, or tuple literal.

    A response that returns the *answer* to a task rather than a program
    computing it lands here — data where code was asked for.
    """
    statement = _sole_statement(source)
    if not isinstance(statement, ast.Expr):
        return False
    return isinstance(
        statement.value, ast.Dict | ast.List | ast.Set | ast.Tuple
    )


def is_code_representation_assignment(source: str) -> bool:
    """True when ``source`` is exactly ``code = "<some string>"``.

    A field-marker payload that quotes its program instead of emitting it
    lands here: the module is a string binding, and running it defines
    nothing the task asked for.
    """
    statement = _sole_statement(source)
    if not isinstance(statement, ast.Assign):
        return False
    if len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    if (
        not isinstance(target, ast.Name)
        or target.id != CODE_REPRESENTATION_NAME
    ):
        return False
    return isinstance(statement.value, ast.Constant) and isinstance(
        statement.value.value, str
    )


__all__ = [
    "CODE_REPRESENTATION_NAME",
    "is_code_representation_assignment",
    "is_plain_literal_module",
]

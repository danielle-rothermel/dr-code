from __future__ import annotations

import ast
from functools import lru_cache
from typing import Final

CODE_REPRESENTATION_NAME: Final[str] = "code"

# Preprocessing runs one worker process per core and each worker carries its
# own copy of this cache, so the resident cost is this size times the worker
# count rather than once per run.
_TREE_CACHE_SIZE: Final[int] = 2048


@lru_cache(maxsize=_TREE_CACHE_SIZE)
def _tree(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _sole_statement(source: str) -> ast.stmt | None:
    tree = _tree(source)
    if tree is None or len(tree.body) != 1:
        return None
    return tree.body[0]


def is_plain_literal_module(source: str) -> bool:
    statement = _sole_statement(source)
    if not isinstance(statement, ast.Expr):
        return False
    return isinstance(
        statement.value, ast.Dict | ast.List | ast.Set | ast.Tuple
    )


def is_code_representation_assignment(source: str) -> bool:
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

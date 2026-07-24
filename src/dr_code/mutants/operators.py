"""Deterministic, one-site behavioral mutations over Python syntax trees."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class OperatorFamily(StrEnum):
    """The complete behavioral-mutant operator vocabulary."""

    COMPARISON_FLIP = "comparison_flip"
    BOUNDARY_SHIFT = "boundary_shift"
    AGGREGATION_SWAP = "aggregation_swap"
    BRANCH_SWAP = "branch_swap"
    RANGE_INCLUSIVITY = "range_inclusivity"


ALL_FAMILIES: Final[tuple[OperatorFamily, ...]] = tuple(OperatorFamily)


@dataclass(frozen=True, slots=True)
class MutationSite:
    """A stable address for one concrete AST edit."""

    family: OperatorFamily
    node_path: int
    target_index: int
    description: str


class MutationError(ValueError):
    """Mutation source or a requested site is invalid."""


_COMPARISON_FLIPS: Final[dict[type[ast.cmpop], type[ast.cmpop]]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
}
_AGGREGATION_SWAPS: Final[dict[str, str]] = {
    "min": "max",
    "max": "min",
}


def iter_sites(
    source: str,
    family: OperatorFamily,
) -> tuple[MutationSite, ...]:
    """Return applicable sites in stable source pre-order."""

    tree = _parse(source)
    sites: list[MutationSite] = []
    for node_path, node in enumerate(_preorder(tree)):
        sites.extend(_sites_for_node(node, node_path, family))
    return tuple(sites)


def apply_site(source: str, site: MutationSite) -> str:
    """Apply exactly the edit identified by ``site`` to a fresh parse."""

    tree = _parse(source)
    nodes = tuple(_preorder(tree))
    if not 0 <= site.node_path < len(nodes):
        raise MutationError(
            f"site node_path {site.node_path} is outside the source tree"
        )
    target = nodes[site.node_path]
    current_sites = _sites_for_node(
        target,
        site.node_path,
        site.family,
    )
    matching = [
        current
        for current in current_sites
        if current.target_index == site.target_index
    ]
    if len(matching) != 1:
        raise MutationError(
            f"site {site.node_path}:{site.target_index} is not applicable "
            f"for {site.family.value}"
        )
    _apply(target, site)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _sites_for_node(
    node: ast.AST,
    node_path: int,
    family: OperatorFamily,
) -> tuple[MutationSite, ...]:
    line = getattr(node, "lineno", "?")
    if family is OperatorFamily.COMPARISON_FLIP:
        if not isinstance(node, ast.Compare):
            return ()
        return tuple(
            MutationSite(
                family=family,
                node_path=node_path,
                target_index=index,
                description=(
                    f"line {line}: comparison operand {index} "
                    f"{_comparison_symbol(op)}"
                ),
            )
            for index, op in enumerate(node.ops)
            if type(op) in _COMPARISON_FLIPS
        )
    if family is OperatorFamily.BOUNDARY_SHIFT:
        candidates = _boundary_candidates(node)
        return tuple(
            MutationSite(
                family=family,
                node_path=node_path,
                target_index=index,
                description=f"line {line}: boundary literal {value}",
            )
            for index, _expression, value in candidates
        )
    if family is OperatorFamily.AGGREGATION_SWAP:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _AGGREGATION_SWAPS
        ):
            return (
                MutationSite(
                    family=family,
                    node_path=node_path,
                    target_index=0,
                    description=f"line {line}: call {node.func.id}()",
                ),
            )
        return ()
    if family is OperatorFamily.BRANCH_SWAP:
        if (
            isinstance(node, ast.If)
            and node.body
            and node.orelse
            and not (
                len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)
            )
        ):
            return (
                MutationSite(
                    family=family,
                    node_path=node_path,
                    target_index=0,
                    description=f"line {line}: if/else arms",
                ),
            )
        return ()
    if (
        isinstance(node, ast.Call)
        and _is_range_call(node)
        and _range_stop_index(node) is not None
    ):
        return (
            MutationSite(
                family=family,
                node_path=node_path,
                target_index=0,
                description=f"line {line}: range() stop",
            ),
        )
    return ()


def _apply(node: ast.AST, site: MutationSite) -> None:
    if site.family is OperatorFamily.COMPARISON_FLIP:
        assert isinstance(node, ast.Compare)
        old = node.ops[site.target_index]
        replacement = _COMPARISON_FLIPS.get(type(old))
        if replacement is None:
            raise MutationError("comparison operator is no longer mutable")
        node.ops[site.target_index] = replacement()
        return
    if site.family is OperatorFamily.BOUNDARY_SHIFT:
        candidates = _boundary_candidates(node)
        try:
            _, value = {
                index: (expression, value)
                for index, expression, value in candidates
            }[site.target_index]
        except KeyError as exc:
            raise MutationError(
                "boundary target is no longer mutable"
            ) from exc
        _replace_boundary_expression(
            node,
            site.target_index,
            ast.Constant(value=value + 1),
        )
        return
    if site.family is OperatorFamily.AGGREGATION_SWAP:
        assert isinstance(node, ast.Call)
        assert isinstance(node.func, ast.Name)
        node.func.id = _AGGREGATION_SWAPS[node.func.id]
        return
    if site.family is OperatorFamily.BRANCH_SWAP:
        assert isinstance(node, ast.If)
        node.body, node.orelse = node.orelse, node.body
        return
    assert isinstance(node, ast.Call)
    stop_index = _range_stop_index(node)
    assert stop_index is not None
    node.args[stop_index] = ast.BinOp(
        left=node.args[stop_index],
        op=ast.Sub(),
        right=ast.Constant(value=1),
    )


def _boundary_candidates(
    node: ast.AST,
) -> tuple[tuple[int, ast.expr, int], ...]:
    values = _boundary_expressions(node)
    return tuple(
        (index, value, integer)
        for index, value in enumerate(values)
        if value is not None
        and (integer := _integer_literal_value(value)) is not None
    )


def _boundary_expressions(
    node: ast.AST,
) -> tuple[ast.expr | None, ...]:
    values: tuple[ast.expr | None, ...]
    if isinstance(node, ast.Compare):
        values = (node.left, *node.comparators)
    elif isinstance(node, ast.Slice):
        values = (node.lower, node.upper, node.step)
    elif isinstance(node, ast.Call) and _is_range_call(node):
        values = tuple(node.args)
    else:
        return ()
    return values


def _integer_literal_value(expression: ast.expr) -> int | None:
    if (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
    ):
        return expression.value
    if (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, ast.USub)
        and isinstance(expression.operand, ast.Constant)
        and isinstance(expression.operand.value, int)
        and not isinstance(expression.operand.value, bool)
    ):
        return -expression.operand.value
    return None


def _replace_boundary_expression(
    node: ast.AST,
    target_index: int,
    replacement: ast.expr,
) -> None:
    if isinstance(node, ast.Compare):
        if target_index == 0:
            node.left = replacement
        else:
            node.comparators[target_index - 1] = replacement
        return
    if isinstance(node, ast.Slice):
        if target_index == 0:
            node.lower = replacement
        elif target_index == 1:
            node.upper = replacement
        else:
            node.step = replacement
        return
    assert isinstance(node, ast.Call)
    node.args[target_index] = replacement


def _is_range_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "range"


def _range_stop_index(node: ast.Call) -> int | None:
    if any(isinstance(argument, ast.Starred) for argument in node.args):
        return None
    if len(node.args) == 1:
        return 0
    if len(node.args) >= 2:
        return 1
    return None


def _comparison_symbol(operator: ast.cmpop) -> str:
    return {
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    }.get(type(operator), type(operator).__name__)


def _preorder(root: ast.AST) -> Iterator[ast.AST]:
    yield root
    for child in ast.iter_child_nodes(root):
        yield from _preorder(child)


def _parse(source: str) -> ast.Module:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise MutationError(f"source does not parse: {exc}") from exc


__all__ = (
    "ALL_FAMILIES",
    "MutationError",
    "MutationSite",
    "OperatorFamily",
    "apply_site",
    "iter_sites",
)

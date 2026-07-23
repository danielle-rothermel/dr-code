"""Rule-based, seeded behavioral mutation operators over a Python AST.

Each operator family is a pure function ``AST -> [applicable site]`` plus a
site-level transformation. A *site* is one location in the tree where the
family can apply; applying a family means rewriting exactly one chosen site.
Choice is deterministic and seeded (see :mod:`dr_code.mutants.generate`): the
generator enumerates sites in a stable pre-order and selects among them with
an explicit seeded index, so ``mutant = f(task_id, family, seed)``.

The families are deliberately *behavior-altering* (they change what the
function computes), not cosmetic. Behavioral distinctness from the canonical
solution is not guaranteed by construction (e.g. flipping ``<`` to ``<=`` on
values that never tie); the execution oracle validates distinctness and the
generator searches over sites/seeds when a mutant is behaviorally silent.

Publication-hardening TODOs (preliminary-results scope): broaden operator
coverage (arithmetic-operator swaps, logical-connective swaps, return-constant
replacement), calibrate mutant difficulty, and add spec/docstring regeneration
for direct-generation arms (not built here; enc-dec needs only the code body).
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class OperatorFamily(StrEnum):
    """The behavioral mutation operator families."""

    COMPARISON_FLIP = "comparison_flip"
    BOUNDARY_SHIFT = "boundary_shift"
    AGGREGATION_SWAP = "aggregation_swap"
    BRANCH_SWAP = "branch_swap"
    RANGE_INCLUSIVITY = "range_inclusivity"


@dataclass(frozen=True, slots=True)
class MutationSite:
    """One applicable location for a family, addressed by a stable path.

    ``node_path`` is the pre-order index of the target node within the module
    (assigned by :func:`iter_sites`), making site selection reproducible
    without holding live AST references across a re-parse. ``description``
    summarizes the concrete edit for the diff/dry-run listing.
    """

    family: OperatorFamily
    node_path: int
    description: str


# --------------------------------------------------------------------------
# Comparison flips: < <-> <=, > <-> >= (strictness flips that change boundary
# behavior). We deliberately do NOT flip == <-> != here (that is a different
# behavioral family and often trivially distinct); boundary strictness is the
# off-by-one-adjacent behavior we target.
# --------------------------------------------------------------------------

_COMPARISON_FLIP: Final[dict[type[ast.cmpop], type[ast.cmpop]]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
}


def _comparison_sites(node: ast.AST) -> bool:
    return isinstance(node, ast.Compare) and any(
        type(op) in _COMPARISON_FLIP for op in node.ops
    )


def _apply_comparison(node: ast.AST) -> str:
    assert isinstance(node, ast.Compare)
    parts: list[str] = []
    for index, op in enumerate(node.ops):
        flipped = _COMPARISON_FLIP.get(type(op))
        if flipped is not None:
            node.ops[index] = flipped()
            parts.append(f"{_op_symbol(op)}->{_op_symbol(node.ops[index])}")
    return "comparison " + ", ".join(parts)


# --------------------------------------------------------------------------
# Boundary / off-by-one constant shifts: +/-1 on integer literals that sit in
# a behaviorally meaningful position (a comparison operand, a slice bound, or a
# range() argument). We shift by +1 (deterministic direction; the seed selects
# the site, not the direction, to keep the search space small and legible).
# --------------------------------------------------------------------------

_BOUNDARY_SHIFT_DELTA: Final = 1


def _is_int_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def _boundary_sites(node: ast.AST) -> bool:
    # A site is an int-constant-bearing "boundary" context: a comparison, a
    # slice, or a range()/len()-style call. We record the *container* node and
    # shift its first eligible int literal, so the path addresses one edit.
    if isinstance(node, ast.Compare):
        return any(_is_int_constant(operand) for operand in node.comparators)
    if isinstance(node, ast.Slice):
        return any(
            bound is not None and _is_int_constant(bound)
            for bound in (node.lower, node.upper, node.step)
        )
    if isinstance(node, ast.Call) and _is_range_call(node):
        return any(_is_int_constant(arg) for arg in node.args)
    return False


def _apply_boundary(node: ast.AST) -> str:
    target = _first_boundary_int_constant(node)
    assert target is not None
    old = target.value
    assert isinstance(old, int)
    new_value = old + _BOUNDARY_SHIFT_DELTA
    target.value = new_value
    return f"boundary literal {old}->{new_value}"


def _first_boundary_int_constant(node: ast.AST) -> ast.Constant | None:
    candidates: Sequence[ast.expr | None]
    if isinstance(node, ast.Compare):
        candidates = node.comparators
    elif isinstance(node, ast.Slice):
        candidates = (node.lower, node.upper, node.step)
    elif isinstance(node, ast.Call):
        candidates = node.args
    else:
        return None
    for candidate in candidates:
        if candidate is not None and _is_int_constant(candidate):
            assert isinstance(candidate, ast.Constant)
            return candidate
    return None


def _is_range_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "range"


# --------------------------------------------------------------------------
# Aggregation swaps: min <-> max (always type-safe as a name swap). We include
# the pair only; sum->len is intentionally excluded from preliminary scope
# because its type-safety is context-dependent and needs a value analysis to
# avoid trivially non-distinct or crashing mutants (hardening TODO).
# --------------------------------------------------------------------------

_AGGREGATION_SWAP: Final[dict[str, str]] = {"min": "max", "max": "min"}


def _aggregation_sites(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _AGGREGATION_SWAP
    )


def _apply_aggregation(node: ast.AST) -> str:
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    old = node.func.id
    node.func.id = _AGGREGATION_SWAP[old]
    return f"aggregation {old}->{node.func.id}"


# --------------------------------------------------------------------------
# Branch-outcome swaps: swap the two arms of an if-statement whose body and
# orelse are both non-empty (type-compatible by construction: both are
# statement lists executed for effect). This changes which arm runs for a
# given condition without touching the condition itself.
# --------------------------------------------------------------------------


def _branch_sites(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and len(node.body) > 0
        and len(node.orelse) > 0
        # Exclude elif chains (orelse is a single If): swapping there is
        # confusing and often reshapes control flow non-locally.
        and not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If))
    )


def _apply_branch(node: ast.AST) -> str:
    assert isinstance(node, ast.If)
    node.body, node.orelse = node.orelse, node.body
    return "branch arms swapped"


# --------------------------------------------------------------------------
# Range inclusivity flips: range(n) <-> range(n + 1) style, expressed as a
# +/-1 on the *stop* argument of a range() call. Distinct from boundary_shift
# in intent (loop-extent change) though mechanically adjacent; we target the
# stop argument specifically and shift it down by 1 (drop the last iteration),
# which is the classic inclusive/exclusive bug.
# --------------------------------------------------------------------------


def _range_inclusivity_sites(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Call) and _is_range_call(node)):
        return False
    stop = _range_stop_arg(node)
    return stop is not None


def _apply_range_inclusivity(node: ast.AST) -> str:
    assert isinstance(node, ast.Call)
    stop_index = 1 if len(node.args) >= 2 else 0
    stop = node.args[stop_index]
    node.args[stop_index] = ast.BinOp(
        left=stop, op=ast.Sub(), right=ast.Constant(value=1)
    )
    return "range stop reduced by 1 (inclusive<->exclusive)"


def _range_stop_arg(node: ast.Call) -> ast.expr | None:
    # range(stop) or range(start, stop[, step]); the stop is arg 0 or arg 1.
    if len(node.args) == 1:
        return node.args[0]
    if len(node.args) >= 2:
        return node.args[1]
    return None


@dataclass(frozen=True, slots=True)
class _FamilyImpl:
    is_site: Callable[[ast.AST], bool]
    apply: Callable[[ast.AST], str]


_FAMILIES: Final[dict[OperatorFamily, _FamilyImpl]] = {
    OperatorFamily.COMPARISON_FLIP: _FamilyImpl(
        _comparison_sites, _apply_comparison
    ),
    OperatorFamily.BOUNDARY_SHIFT: _FamilyImpl(
        _boundary_sites, _apply_boundary
    ),
    OperatorFamily.AGGREGATION_SWAP: _FamilyImpl(
        _aggregation_sites, _apply_aggregation
    ),
    OperatorFamily.BRANCH_SWAP: _FamilyImpl(_branch_sites, _apply_branch),
    OperatorFamily.RANGE_INCLUSIVITY: _FamilyImpl(
        _range_inclusivity_sites, _apply_range_inclusivity
    ),
}

ALL_FAMILIES: Final[tuple[OperatorFamily, ...]] = tuple(OperatorFamily)


class MutationError(ValueError):
    """The source could not be parsed or a site index is out of range."""


def iter_sites(
    source: str, family: OperatorFamily
) -> tuple[MutationSite, ...]:
    """Enumerate applicable sites for ``family`` in stable pre-order.

    Pre-order (``ast.walk`` is breadth-first, so we use an explicit pre-order
    traversal) gives a deterministic, source-order-stable site index that is
    reproducible across parses.
    """

    tree = _parse(source)
    impl = _FAMILIES[family]
    sites: list[MutationSite] = []
    for index, node in enumerate(_preorder(tree)):
        if impl.is_site(node):
            sites.append(
                MutationSite(
                    family=family,
                    node_path=index,
                    description=_describe_site(node, family),
                )
            )
    return tuple(sites)


def apply_site(source: str, site: MutationSite) -> str:
    """Return ``source`` with exactly the one edit at ``site`` applied.

    Re-parses and re-walks so the transformation is a pure function of
    ``(source, site)`` with no shared mutable state between calls.
    """

    tree = _parse(source)
    impl = _FAMILIES[site.family]
    nodes = list(_preorder(tree))
    if site.node_path < 0 or site.node_path >= len(nodes):
        raise MutationError(
            f"site node_path {site.node_path} out of range for source"
        )
    target = nodes[site.node_path]
    if not impl.is_site(target):
        raise MutationError(
            f"node at path {site.node_path} is not a {site.family} site"
        )
    impl.apply(target)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _describe_site(node: ast.AST, family: OperatorFamily) -> str:
    line = getattr(node, "lineno", None)
    prefix = f"line {line}: " if line is not None else ""
    if family is OperatorFamily.COMPARISON_FLIP:
        assert isinstance(node, ast.Compare)
        ops = ", ".join(_op_symbol(op) for op in node.ops)
        return f"{prefix}compare ({ops})"
    if family is OperatorFamily.BOUNDARY_SHIFT:
        target = _first_boundary_int_constant(node)
        value = target.value if target is not None else "?"
        return f"{prefix}int literal {value}"
    if family is OperatorFamily.AGGREGATION_SWAP:
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        return f"{prefix}call {node.func.id}()"
    if family is OperatorFamily.BRANCH_SWAP:
        return f"{prefix}if-statement with both arms"
    return f"{prefix}range() call"


def _op_symbol(op: ast.cmpop) -> str:
    return {
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Is: "is",
        ast.IsNot: "is not",
        ast.In: "in",
        ast.NotIn: "not in",
    }.get(type(op), type(op).__name__)


def _preorder(tree: ast.AST) -> list[ast.AST]:
    ordered: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        ordered.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return ordered


def _parse(source: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise MutationError(f"source does not parse: {exc}") from exc


__all__ = [
    "ALL_FAMILIES",
    "MutationError",
    "MutationSite",
    "OperatorFamily",
    "apply_site",
    "iter_sites",
]

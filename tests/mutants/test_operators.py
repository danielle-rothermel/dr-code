"""Operator applicability + transformation-correctness unit tests."""

from __future__ import annotations

import ast

import pytest

from dr_code.mutants.operators import (
    ALL_FAMILIES,
    MutationError,
    OperatorFamily,
    apply_site,
    iter_sites,
)


def _only_site(source: str, family: OperatorFamily) -> str:
    sites = iter_sites(source, family)
    assert len(sites) == 1, f"expected exactly one {family} site"
    return apply_site(source, sites[0])


def _normalized(source: str) -> str:
    return ast.unparse(ast.parse(source))


# --- comparison_flip -------------------------------------------------------


def test_comparison_flip_lt_to_lte() -> None:
    out = _only_site(
        "def f(a, b):\n    return a < b\n", OperatorFamily.COMPARISON_FLIP
    )
    assert "a <= b" in out


def test_comparison_flip_ge_to_gt() -> None:
    out = _only_site(
        "def f(a, b):\n    return a >= b\n", OperatorFamily.COMPARISON_FLIP
    )
    assert "a > b" in out


def test_comparison_flip_ignores_eq() -> None:
    sites = iter_sites(
        "def f(a, b):\n    return a == b\n", OperatorFamily.COMPARISON_FLIP
    )
    assert sites == ()


# --- boundary_shift --------------------------------------------------------


def test_boundary_shift_on_comparison_literal() -> None:
    out = _only_site(
        "def f(x):\n    return x < 10\n", OperatorFamily.BOUNDARY_SHIFT
    )
    assert "x < 11" in out


def test_boundary_shift_on_slice_bound() -> None:
    out = _only_site(
        "def f(xs):\n    return xs[2:]\n", OperatorFamily.BOUNDARY_SHIFT
    )
    assert _normalized(out) == _normalized("def f(xs):\n    return xs[3:]\n")


def test_boundary_shift_skips_boolean_literal() -> None:
    # True/False are int subclasses; they must not be treated as int constants.
    sites = iter_sites(
        "def f(x):\n    return x < True\n", OperatorFamily.BOUNDARY_SHIFT
    )
    assert sites == ()


def test_boundary_shift_no_site_when_no_int_literal() -> None:
    sites = iter_sites(
        "def f(a, b):\n    return a < b\n", OperatorFamily.BOUNDARY_SHIFT
    )
    assert sites == ()


# --- aggregation_swap ------------------------------------------------------


def test_aggregation_swap_min_to_max() -> None:
    out = _only_site(
        "def f(xs):\n    return min(xs)\n", OperatorFamily.AGGREGATION_SWAP
    )
    assert "max(xs)" in out


def test_aggregation_swap_max_to_min() -> None:
    out = _only_site(
        "def f(xs):\n    return max(xs)\n", OperatorFamily.AGGREGATION_SWAP
    )
    assert "min(xs)" in out


# --- branch_swap -----------------------------------------------------------


def test_branch_swap_exchanges_arms() -> None:
    source = (
        "def f(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    else:\n"
        "        return 2\n"
    )
    out = _only_site(source, OperatorFamily.BRANCH_SWAP)
    tree = ast.parse(out)
    if_node = tree.body[0].body[0]  # type: ignore[attr-defined]
    assert isinstance(if_node, ast.If)
    assert ast.unparse(if_node.body[0]) == "return 2"
    assert ast.unparse(if_node.orelse[0]) == "return 1"


def test_branch_swap_skips_missing_else() -> None:
    source = "def f(x):\n    if x:\n        return 1\n    return 2\n"
    assert iter_sites(source, OperatorFamily.BRANCH_SWAP) == ()


def test_branch_swap_skips_elif_chain() -> None:
    source = (
        "def f(x):\n"
        "    if x == 1:\n"
        "        return 1\n"
        "    elif x == 2:\n"
        "        return 2\n"
        "    else:\n"
        "        return 3\n"
    )
    # The outer if's orelse is a single If (elif); it is excluded.
    sites = iter_sites(source, OperatorFamily.BRANCH_SWAP)
    # Only the inner elif (which itself has body+else) is a valid swap site.
    assert len(sites) == 1


# --- range_inclusivity -----------------------------------------------------


def test_range_inclusivity_reduces_stop_single_arg() -> None:
    out = _only_site(
        "def f(n):\n    return list(range(n))\n",
        OperatorFamily.RANGE_INCLUSIVITY,
    )
    assert _normalized(out) == _normalized(
        "def f(n):\n    return list(range(n - 1))\n"
    )


def test_range_inclusivity_targets_stop_of_two_arg_range() -> None:
    out = _only_site(
        "def f(n):\n    return list(range(1, n))\n",
        OperatorFamily.RANGE_INCLUSIVITY,
    )
    assert _normalized(out) == _normalized(
        "def f(n):\n    return list(range(1, n - 1))\n"
    )


# --- general contracts -----------------------------------------------------


def test_all_families_enumerate_deterministic_site_order() -> None:
    source = (
        "def f(xs):\n"
        "    for i in range(len(xs) - 1):\n"
        "        if xs[i] < xs[i + 1]:\n"
        "            return min(xs)\n"
        "    return max(xs)\n"
    )
    for family in ALL_FAMILIES:
        first = iter_sites(source, family)
        second = iter_sites(source, family)
        assert [s.node_path for s in first] == [s.node_path for s in second]


def test_apply_site_is_pure_and_repeatable() -> None:
    source = "def f(a, b):\n    return a < b\n"
    site = iter_sites(source, OperatorFamily.COMPARISON_FLIP)[0]
    assert apply_site(source, site) == apply_site(source, site)


def test_apply_site_rejects_out_of_range_path() -> None:
    source = "def f(a, b):\n    return a < b\n"
    site = iter_sites(source, OperatorFamily.COMPARISON_FLIP)[0]
    bad = site.model_copy(update={"node_path": 9999}) if hasattr(
        site, "model_copy"
    ) else None
    _ = bad
    from dataclasses import replace

    with pytest.raises(MutationError):
        apply_site(source, replace(site, node_path=9999))


def test_unparseable_source_raises() -> None:
    with pytest.raises(MutationError):
        iter_sites("def f(:\n", OperatorFamily.COMPARISON_FLIP)

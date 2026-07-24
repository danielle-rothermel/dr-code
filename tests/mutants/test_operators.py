"""Exact-site contracts for all behavioral mutation families."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dr_code.mutants.generate import seeded_site_order
from dr_code.mutants.operators import (
    ALL_FAMILIES,
    MutationError,
    OperatorFamily,
    apply_site,
    iter_sites,
)


def test_comparison_chain_exposes_one_site_per_operator() -> None:
    source = "def f(x):\n    return 0 < x <= 5\n"
    sites = iter_sites(source, OperatorFamily.COMPARISON_FLIP)

    assert [(site.node_path, site.target_index) for site in sites] == [
        (5, 0),
        (5, 1),
    ]
    assert apply_site(source, sites[0]) == (
        "def f(x):\n    return 0 <= x <= 5"
    )
    assert apply_site(source, sites[1]) == ("def f(x):\n    return 0 < x < 5")


def test_boundary_shift_changes_only_selected_literal() -> None:
    source = "def f(x):\n    return x[1:3]\n"
    sites = iter_sites(source, OperatorFamily.BOUNDARY_SHIFT)

    assert len(sites) == 2
    assert apply_site(source, sites[0]) == "def f(x):\n    return x[2:3]"
    assert apply_site(source, sites[1]) == "def f(x):\n    return x[1:4]"


def test_boundary_shift_addresses_compare_left_and_each_comparator() -> None:
    source = "def f(x):\n    return 1 < x < 3\n"
    sites = iter_sites(source, OperatorFamily.BOUNDARY_SHIFT)

    assert [(site.node_path, site.target_index) for site in sites] == [
        (5, 0),
        (5, 2),
    ]
    assert apply_site(source, sites[0]) == ("def f(x):\n    return 2 < x < 3")
    assert apply_site(source, sites[1]) == ("def f(x):\n    return 1 < x < 4")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def f(x):\n    return x >= -1\n", "def f(x):\n    return x >= 0"),
        ("def f(xs):\n    return xs[:-1]\n", "def f(xs):\n    return xs[:0]"),
        (
            "def f():\n    return list(range(-5, 0))\n",
            "def f():\n    return list(range(-4, 0))",
        ),
    ],
)
def test_boundary_shift_treats_negative_literal_as_one_expression(
    source: str,
    expected: str,
) -> None:
    sites = iter_sites(source, OperatorFamily.BOUNDARY_SHIFT)

    negative = next(
        site
        for site in sites
        if "-1" in site.description or "-5" in site.description
    )
    assert apply_site(source, negative) == expected


def test_aggregation_swap_changes_one_call() -> None:
    source = "def f(xs):\n    return min(xs) + max(xs)\n"
    sites = iter_sites(source, OperatorFamily.AGGREGATION_SWAP)

    assert len(sites) == 2
    assert apply_site(source, sites[0]) == (
        "def f(xs):\n    return max(xs) + max(xs)"
    )


def test_branch_swap_changes_one_if_else() -> None:
    source = (
        "def f(flag):\n"
        "    if flag:\n"
        "        return 1\n"
        "    else:\n"
        "        return 2\n"
    )
    site = iter_sites(source, OperatorFamily.BRANCH_SWAP)[0]

    assert apply_site(source, site) == (
        "def f(flag):\n"
        "    if flag:\n"
        "        return 2\n"
        "    else:\n"
        "        return 1"
    )


def test_range_inclusivity_changes_only_stop_expression() -> None:
    source = "def f(n):\n    return list(range(2, n, 3))\n"
    site = iter_sites(source, OperatorFamily.RANGE_INCLUSIVITY)[0]

    assert apply_site(source, site) == (
        "def f(n):\n    return list(range(2, n - 1, 3))"
    )


@pytest.mark.parametrize(
    "source",
    [
        "def f(args):\n    return list(range(*args))\n",
        "def f(args):\n    return list(range(1, *args))\n",
    ],
)
def test_range_inclusivity_rejects_starred_arguments(source: str) -> None:
    assert iter_sites(source, OperatorFamily.RANGE_INCLUSIVITY) == ()


def test_all_families_have_stable_site_addresses() -> None:
    source = (
        "def f(xs, flag):\n"
        "    if flag:\n"
        "        return min(xs[:2])\n"
        "    else:\n"
        "        return max(xs[:3]) if len(xs) < 4 else sum(range(5))\n"
    )
    for family in ALL_FAMILIES:
        first = iter_sites(source, family)
        second = iter_sites(source, family)
        assert first == second
        assert len(
            {(site.node_path, site.target_index) for site in first}
        ) == len(first)


def test_seeded_search_order_is_stable_and_seed_sensitive() -> None:
    first = seeded_site_order(
        task_id="HumanEval/7",
        family=OperatorFamily.COMPARISON_FLIP,
        seed=0,
        site_count=8,
    )
    assert first == seeded_site_order(
        task_id="HumanEval/7",
        family=OperatorFamily.COMPARISON_FLIP,
        seed=0,
        site_count=8,
    )
    assert first != seeded_site_order(
        task_id="HumanEval/7",
        family=OperatorFamily.COMPARISON_FLIP,
        seed=1,
        site_count=8,
    )
    assert sorted(first) == list(range(8))


def test_malformed_source_and_stale_addresses_are_rejected() -> None:
    with pytest.raises(MutationError, match="does not parse"):
        iter_sites("def f(:\n", OperatorFamily.COMPARISON_FLIP)

    source = "def f(x):\n    return x < 1\n"
    site = iter_sites(source, OperatorFamily.COMPARISON_FLIP)[0]
    with pytest.raises(MutationError, match="outside"):
        apply_site(source, replace(site, node_path=999))
    with pytest.raises(MutationError, match="not applicable"):
        apply_site(source, replace(site, target_index=2))

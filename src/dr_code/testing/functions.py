"""Discover top-level candidate functions in generated code."""

from __future__ import annotations

import ast

from dr_code.models.outcomes import CandidateFunction


def discover_candidate_functions(source: str) -> tuple[CandidateFunction, ...]:
    """Return top-level functions in source order."""
    module = ast.parse(source)
    candidates: list[CandidateFunction] = []
    for index, node in enumerate(module.body):
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args
        candidates.append(
            CandidateFunction(
                name=node.name,
                positional_arity=len(args.posonlyargs) + len(args.args),
                source_order=index,
                has_varargs=args.vararg is not None,
            )
        )
    return tuple(candidates)


def arity_matching_functions(
    candidates: tuple[CandidateFunction, ...],
    *,
    expected_arity: int,
) -> tuple[CandidateFunction, ...]:
    """Return candidates eligible for functional-recovery execution."""
    return tuple(
        candidate
        for candidate in candidates
        if candidate.positional_arity == expected_arity
        and not candidate.has_varargs
    )

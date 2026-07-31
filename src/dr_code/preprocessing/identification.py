"""Parse-once Python candidate identification and bounded canonicalization."""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass

from pydantic import JsonValue

from dr_code.preprocessing.candidate_identity import candidate_id_for_source
from dr_code.preprocessing.import_inference import (
    infer_missing_imports_from_tree,
)
from dr_code.trace import (
    CandidateInspection,
    CandidateLineage,
    CandidateOrigin,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    IdentifiedCandidate,
    IdentifiedCandidateSetArtifact,
)

CODE_REPR_VARIABLE_NAME = "code"
BARE_LAMBDA_FUNCTION_NAME = "candidate"


@dataclass(frozen=True, slots=True)
class _IdentifiedDraft:
    source: str
    lineage: CandidateLineage
    inspected: _SourceInspection


@dataclass(frozen=True, slots=True)
class _SourceInspection:
    tree: ast.Module | None
    parse_ok: bool
    parse_error: str | None
    compile_ok: bool
    compile_error: str | None
    compile_warnings: tuple[str, ...]
    parser_stack_overflow: bool
    parser_recursion_overflow: bool


def validate_python_source_with_ast(source: str) -> _SourceInspection:
    """Parse and compile one exact source at most once each."""
    parser_stack_overflow = False
    parser_recursion_overflow = False
    with warnings.catch_warnings(record=True) as parse_warnings:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            tree = None
        except RecursionError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            tree = None
            parser_recursion_overflow = True
        except MemoryError as exc:
            if not str(exc).startswith("Parser stack overflowed"):
                raise
            parse_error = f"{type(exc).__name__}: {exc}"
            tree = None
            parser_stack_overflow = True
        else:
            parse_error = None

    compile_warnings: list[warnings.WarningMessage] = []
    if tree is None:
        compile_error = parse_error
    else:
        with warnings.catch_warnings(record=True) as compile_warnings:
            warnings.simplefilter("always", SyntaxWarning)
            try:
                compile(tree, "<candidate>", "exec")
            except (SyntaxError, ValueError, RecursionError) as exc:
                compile_error = f"{type(exc).__name__}: {exc}"
            except MemoryError as exc:
                if not str(exc).startswith("Parser stack overflowed"):
                    raise
                compile_error = f"{type(exc).__name__}: {exc}"
                parser_stack_overflow = True
            else:
                compile_error = None

    warning_messages = tuple(
        f"{warning.category.__name__}: {warning.message} "
        f"(line {warning.lineno})"
        for warning in (*parse_warnings, *compile_warnings)
    )
    return _SourceInspection(
        tree=tree,
        parse_ok=tree is not None,
        parse_error=parse_error,
        compile_ok=tree is not None and compile_error is None,
        compile_error=compile_error,
        compile_warnings=warning_messages,
        parser_stack_overflow=parser_stack_overflow,
        parser_recursion_overflow=parser_recursion_overflow,
    )


def identify_candidates(
    value: CodeCandidateSetArtifact,
) -> tuple[IdentifiedCandidateSetArtifact, dict[str, JsonValue]]:
    """Identify unique candidates with one inspection per exact source.

    Missing imports are derived from the inspection tree rather than reparsing
    in a separate cleaning step. Lambda-to-function rendering is additive and
    bounded: primary candidates retain their order and rendered alternatives
    follow them in parent order.
    """
    cache: dict[str, _SourceInspection] = {}
    input_drafts = _dedupe_input_sources(value)
    primary: list[_IdentifiedDraft] = []
    rendered: list[_IdentifiedDraft] = []
    transformations: list[dict[str, JsonValue]] = []

    for source, lineage in input_drafts:
        canonical = _canonicalize_imports(source, lineage, cache)
        primary.append(canonical)
        if canonical.source != source:
            transformations.append(
                {
                    "kind": "infer_missing_imports",
                    "input_source": source,
                    "output_source": canonical.source,
                }
            )

    for canonical in primary:
        rendered_source = _render_lambda_function(
            canonical.source, canonical.inspected.tree
        )
        if rendered_source is None or rendered_source == canonical.source:
            continue
        rendered_lineage = _append_lineage_operation(
            canonical.lineage,
            ExtractionOperation(
                kind="lambda_to_function",
                details={
                    "function_name": _rendered_function_name(
                        canonical.inspected.tree
                    )
                },
            ),
        )
        rendered_candidate = _canonicalize_imports(
            rendered_source, rendered_lineage, cache
        )
        rendered.append(rendered_candidate)
        transformations.append(
            {
                "kind": "lambda_to_function",
                "input_source": canonical.source,
                "output_source": rendered_candidate.source,
            }
        )

    merged = _merge_identified((*primary, *rendered))
    identified = tuple(
        IdentifiedCandidate(
            source=draft.source,
            lineage=draft.lineage.model_copy(
                update={"candidate_id": candidate_id_for_source(draft.source)}
            ),
            inspection=_inspection(draft.inspected),
        )
        for draft in merged
    )
    facts: dict[str, JsonValue] = {
        "input_candidate_count": len(value.candidates),
        "unique_input_source_count": len(input_drafts),
        "identified_candidate_count": len(identified),
        "inspection_count": len(cache),
        "transformations": transformations,
        "inspections": [
            {
                "candidate_id": candidate.lineage.candidate_id,
                **_inspection_facts(candidate.inspection),
            }
            for candidate in identified
        ],
    }
    return IdentifiedCandidateSetArtifact(candidates=identified), facts


def _dedupe_input_sources(
    value: CodeCandidateSetArtifact,
) -> list[tuple[str, CandidateLineage]]:
    ordered: list[tuple[str, CandidateLineage]] = []
    positions: dict[str, int] = {}
    for index, source in enumerate(value.candidates):
        lineage = value.lineage[index].model_copy(
            update={"candidate_id": None}
        )
        position = positions.get(source)
        if position is None:
            positions[source] = len(ordered)
            ordered.append((source, lineage))
            continue
        prior_source, prior_lineage = ordered[position]
        ordered[position] = (
            prior_source,
            _merge_lineage(prior_lineage, lineage),
        )
    return ordered


def _inspect(
    source: str, cache: dict[str, _SourceInspection]
) -> _SourceInspection:
    inspected = cache.get(source)
    if inspected is None:
        inspected = validate_python_source_with_ast(source)
        cache[source] = inspected
    return inspected


def _canonicalize_imports(
    source: str,
    lineage: CandidateLineage,
    cache: dict[str, _SourceInspection],
) -> _IdentifiedDraft:
    inspected = _inspect(source, cache)
    if inspected.tree is None:
        return _IdentifiedDraft(source, lineage, inspected)
    inferred = infer_missing_imports_from_tree(source, inspected.tree)
    if inferred == source:
        return _IdentifiedDraft(source, lineage, inspected)
    inferred_lineage = _append_lineage_operation(
        lineage, ExtractionOperation(kind="infer_missing_imports")
    )
    return _IdentifiedDraft(
        inferred, inferred_lineage, _inspect(inferred, cache)
    )


def _merge_identified(
    drafts: tuple[_IdentifiedDraft, ...],
) -> tuple[_IdentifiedDraft, ...]:
    merged: list[_IdentifiedDraft] = []
    positions: dict[str, int] = {}
    for draft in drafts:
        position = positions.get(draft.source)
        if position is None:
            positions[draft.source] = len(merged)
            merged.append(draft)
            continue
        prior = merged[position]
        merged[position] = _IdentifiedDraft(
            source=prior.source,
            lineage=_merge_lineage(prior.lineage, draft.lineage),
            inspected=prior.inspected,
        )
    return tuple(merged)


def _merge_lineage(
    first: CandidateLineage, second: CandidateLineage
) -> CandidateLineage:
    origins = list(first.origins)
    for origin in second.origins:
        if origin not in origins:
            origins.append(origin)
    return CandidateLineage(origins=tuple(origins))


def _append_lineage_operation(
    lineage: CandidateLineage, operation: ExtractionOperation
) -> CandidateLineage:
    return CandidateLineage(
        origins=tuple(
            CandidateOrigin(path=(*origin.path, operation))
            for origin in lineage.origins
        )
    )


def _inspection(inspected: _SourceInspection) -> CandidateInspection:
    tree = inspected.tree
    functions = (
        tuple(
            statement
            for statement in tree.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        )
        if tree is not None
        else ()
    )
    return CandidateInspection(
        parse_ok=inspected.parse_ok,
        parse_error=inspected.parse_error,
        compile_ok=inspected.compile_ok,
        compile_error=inspected.compile_error,
        compile_warnings=inspected.compile_warnings,
        parser_stack_overflow=inspected.parser_stack_overflow,
        parser_recursion_overflow=inspected.parser_recursion_overflow,
        is_plain_literal_module=(
            tree is not None and _is_plain_literal_module(tree)
        ),
        is_code_repr_assignment=(
            tree is not None and _is_code_repr_assignment(tree)
        ),
        top_level_function_names=tuple(node.name for node in functions),
        top_level_async_function_names=tuple(
            node.name
            for node in functions
            if isinstance(node, ast.AsyncFunctionDef)
        ),
    )


def _inspection_facts(
    inspection: CandidateInspection,
) -> dict[str, JsonValue]:
    return {
        "parse_ok": inspection.parse_ok,
        "parse_error": inspection.parse_error,
        "compile_ok": inspection.compile_ok,
        "compile_error": inspection.compile_error,
        "compile_warnings": list(inspection.compile_warnings),
        "top_level_function_count": len(inspection.top_level_function_names),
        "top_level_function_names": list(inspection.top_level_function_names),
        "top_level_async_function_names": list(
            inspection.top_level_async_function_names
        ),
    }


def _is_plain_literal_module(tree: ast.Module) -> bool:
    if len(tree.body) != 1:
        return False
    statement = tree.body[0]
    return isinstance(statement, ast.Expr) and isinstance(
        statement.value, ast.Dict | ast.List | ast.Set | ast.Tuple
    )


def _is_code_repr_assignment(tree: ast.Module) -> bool:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return False
    statement = tree.body[0]
    return (
        len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == CODE_REPR_VARIABLE_NAME
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _lambda_statement(tree: ast.Module | None) -> ast.Expr | ast.Assign | None:
    if tree is None:
        return None
    non_imports = [
        statement
        for statement in tree.body
        if not isinstance(statement, ast.Import | ast.ImportFrom)
    ]
    if len(non_imports) != 1:
        return None
    statement = non_imports[0]
    if isinstance(statement, ast.Expr) and isinstance(
        statement.value, ast.Lambda
    ):
        return statement
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Lambda)
    ):
        return statement
    return None


def _rendered_function_name(tree: ast.Module | None) -> str:
    statement = _lambda_statement(tree)
    if isinstance(statement, ast.Assign):
        target = statement.targets[0]
        assert isinstance(target, ast.Name)
        return target.id
    return BARE_LAMBDA_FUNCTION_NAME


def _render_lambda_function(
    source: str, tree: ast.Module | None
) -> str | None:
    statement = _lambda_statement(tree)
    if statement is None or statement.col_offset != 0:
        return None
    lambda_node = statement.value
    assert isinstance(lambda_node, ast.Lambda)
    function_name = _rendered_function_name(tree)
    try:
        arguments = ast.unparse(lambda_node.args)
        body = ast.unparse(lambda_node.body)
    except RecursionError:
        # Deep but valid expressions can exceed ``ast.unparse``'s recursive
        # renderer. Lambda conversion is only an additive recovery path, so
        # retaining the already-inspected raw candidate is the safe fallback.
        return None
    replacement = f"def {function_name}({arguments}):\n    return {body}"

    lines = source.splitlines(keepends=True)
    start = statement.lineno - 1
    end = statement.end_lineno or statement.lineno
    replaced_had_newline = bool(lines[end - 1 : end]) and lines[
        end - 1
    ].endswith(("\n", "\r"))
    if replaced_had_newline or end < len(lines):
        replacement += "\n"
    return "".join((*lines[:start], replacement, *lines[end:]))


__all__ = [
    "BARE_LAMBDA_FUNCTION_NAME",
    "identify_candidates",
]

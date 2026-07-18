"""Python AST structure statistics."""

from __future__ import annotations

import ast
from collections.abc import Mapping

from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorResult,
    OperatorSettings,
    artifact_text,
)
from dr_code.trace import Artifact, ArtifactKind

_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.IfExp,
    ast.BoolOp,
    ast.Match,
)
_ASSIGNMENT_NODES = (ast.Assign, ast.AnnAssign, ast.AugAssign)
_COMPREHENSION_NODES = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
_LITERAL_NODES = (ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict)
_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


class AstStatsResult(OperatorResult):
    top_level_function_count: int
    nested_function_count: int
    async_function_count: int
    lambda_count: int
    class_count: int
    import_count: int
    ast_node_count: int
    statement_count: int
    branch_count: int
    return_count: int
    yield_count: int
    call_count: int
    assignment_count: int
    comprehension_count: int
    literal_count: int
    max_branch_depth: int
    function_count: int
    total_argument_count: int
    positional_only_argument_count: int
    keyword_only_argument_count: int
    vararg_count: int
    kwarg_count: int
    decorated_function_count: int
    annotated_return_count: int
    docstring_function_count: int
    total_function_body_statement_count: int
    max_function_body_statement_count: int
    max_function_line_span: int


class AstStats(MetricOperator[OperatorSettings]):
    NAME = MetricName.AST_STATS
    VERSION = "1"
    INPUT = ArtifactKind.CODE

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> AstStatsResult:
        _ = aux
        source = artifact_text(value)
        tree = ctx.views.parsed_module(source)
        if tree is None:
            # CodeArtifact documents "passed a compile check upstream", so an
            # unparseable CODE artifact is a producer contract violation, not
            # an empty module. Raise (→ operator_failure) instead of
            # fabricating all-zero fields that would be indistinguishable
            # from a genuinely empty module. Parse facts remain the job of
            # ``parse_outcome``.
            raise ValueError(
                f"ast_stats requires parseable code: {ctx.views.parse_error(source)}"
            )

        nodes = list(ast.walk(tree))
        top_level_functions = [
            node for node in tree.body if isinstance(node, _FUNCTION_NODES)
        ]
        all_functions = [
            node for node in nodes if isinstance(node, _FUNCTION_NODES)
        ]
        return AstStatsResult(
            top_level_function_count=len(top_level_functions),
            nested_function_count=(
                len(all_functions) - len(top_level_functions)
            ),
            async_function_count=sum(
                isinstance(node, ast.AsyncFunctionDef)
                for node in all_functions
            ),
            lambda_count=sum(
                isinstance(node, ast.Lambda) for node in nodes
            ),
            class_count=sum(
                isinstance(node, ast.ClassDef) for node in nodes
            ),
            import_count=sum(
                isinstance(node, ast.Import | ast.ImportFrom)
                for node in nodes
            ),
            ast_node_count=len(nodes),
            statement_count=sum(
                isinstance(node, ast.stmt) for node in nodes
            ),
            branch_count=sum(
                isinstance(node, _BRANCH_NODES) for node in nodes
            ),
            return_count=sum(
                isinstance(node, ast.Return) for node in nodes
            ),
            yield_count=sum(
                isinstance(node, ast.Yield | ast.YieldFrom)
                for node in nodes
            ),
            call_count=sum(isinstance(node, ast.Call) for node in nodes),
            assignment_count=sum(
                isinstance(node, _ASSIGNMENT_NODES) for node in nodes
            ),
            comprehension_count=sum(
                isinstance(node, _COMPREHENSION_NODES) for node in nodes
            ),
            literal_count=sum(
                isinstance(node, _LITERAL_NODES) for node in nodes
            ),
            max_branch_depth=_max_branch_depth(tree),
            function_count=len(all_functions),
            total_argument_count=sum(
                _function_argument_count(node) for node in all_functions
            ),
            positional_only_argument_count=sum(
                len(node.args.posonlyargs) for node in all_functions
            ),
            keyword_only_argument_count=sum(
                len(node.args.kwonlyargs) for node in all_functions
            ),
            vararg_count=sum(
                node.args.vararg is not None for node in all_functions
            ),
            kwarg_count=sum(
                node.args.kwarg is not None for node in all_functions
            ),
            decorated_function_count=sum(
                bool(node.decorator_list) for node in all_functions
            ),
            annotated_return_count=sum(
                node.returns is not None for node in all_functions
            ),
            docstring_function_count=sum(
                ast.get_docstring(node) is not None
                for node in all_functions
            ),
            total_function_body_statement_count=sum(
                len(node.body) for node in all_functions
            ),
            max_function_body_statement_count=max(
                (len(node.body) for node in all_functions),
                default=0,
            ),
            max_function_line_span=max(
                (_function_line_span(node) for node in all_functions),
                default=0,
            ),
        )


def _function_argument_count(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    return (
        len(node.args.posonlyargs)
        + len(node.args.args)
        + len(node.args.kwonlyargs)
        + int(node.args.vararg is not None)
        + int(node.args.kwarg is not None)
    )


def _function_line_span(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    if end_lineno is None:
        return 0
    return max(0, end_lineno - node.lineno + 1)


def _max_branch_depth(node: ast.AST, *, current_depth: int = 0) -> int:
    next_depth = (
        current_depth + 1
        if isinstance(node, _BRANCH_NODES)
        else current_depth
    )
    child_depth = max(
        (
            _max_branch_depth(child, current_depth=next_depth)
            for child in ast.iter_child_nodes(node)
        ),
        default=next_depth,
    )
    return max(next_depth, child_depth)

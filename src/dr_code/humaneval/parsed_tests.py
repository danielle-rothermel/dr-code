from __future__ import annotations

import ast
from collections.abc import Iterator
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "HumanEvalTestCaseKind",
    "InputExpressionTestCase",
    "InputOracleTestCase",
    "InputResultTestCase",
    "ParsedTests",
    "SingleCaseCheck",
    "TestCase",
    "UnsupportedTestFormatError",
    "find_check_function",
    "parse_humaneval_tests",
    "support_code_without_check",
]

EXPECTED_ARG_INDEX = 1
TOLERANCE_ARG_INDEX = 2
PAIR_TARGET_SIZE = 2


class UnsupportedTestFormatError(ValueError):
    pass


class HumanEvalTestCaseKind(StrEnum):
    INPUT_RESULT = "input_result"
    INPUT_ORACLE = "input_oracle"
    INPUT_EXPRESSION = "input_expression"


class SingleCaseCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    code: str
    input_repr: str = ""
    expected_output_repr: str = ""
    actual_output_expr: str = ""
    expected_output_expr: str | None = None


class InputResultTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[HumanEvalTestCaseKind.INPUT_RESULT] = (
        HumanEvalTestCaseKind.INPUT_RESULT
    )
    case_id: str
    args: list[Any]
    expected: Any
    atol: float = 0

    def as_check(
        self,
        *,
        candidate_name: str,
        assertion_name: str,
    ) -> SingleCaseCheck:
        return SingleCaseCheck(
            case_id=self.case_id,
            input_repr=repr(self.args),
            expected_output_repr=repr(self.expected),
            actual_output_expr=f"{candidate_name}(*{self.args!r})",
            code=(
                f"{assertion_name}("
                f"{candidate_name}(*{self.args!r}), "
                f"{self.expected!r}, {self.atol!r})"
            ),
        )


class InputOracleTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[HumanEvalTestCaseKind.INPUT_ORACLE] = (
        HumanEvalTestCaseKind.INPUT_ORACLE
    )
    case_id: str
    args: list[Any]
    oracle_name: str
    atol: float = 0

    def as_check(
        self,
        *,
        candidate_name: str,
        assertion_name: str,
    ) -> SingleCaseCheck:
        expected_expr = f"{self.oracle_name}(*{self.args!r})"
        return SingleCaseCheck(
            case_id=self.case_id,
            input_repr=repr(self.args),
            expected_output_repr=expected_expr,
            actual_output_expr=f"{candidate_name}(*{self.args!r})",
            expected_output_expr=expected_expr,
            code=(
                f"{assertion_name}("
                f"{candidate_name}(*{self.args!r}), "
                f"{expected_expr}, {self.atol!r})"
            ),
        )


class InputExpressionTestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[HumanEvalTestCaseKind.INPUT_EXPRESSION] = (
        HumanEvalTestCaseKind.INPUT_EXPRESSION
    )
    case_id: str
    args: list[Any]
    expected: Any
    expression: str
    input_name: str
    expected_name: str
    index_name: str | None = None

    def as_check(
        self,
        *,
        candidate_name: str,
        assertion_name: str,
    ) -> SingleCaseCheck:
        _ = assertion_name
        lines = []
        if self.index_name is not None:
            index = int(self.case_id.rsplit("_", maxsplit=1)[-1])
            lines.append(f"{self.index_name} = {index!r}")
        lines.extend(
            [
                f"{self.input_name} = {self.args!r}",
                f"{self.expected_name} = {self.expected!r}",
                f"candidate = {candidate_name}",
                self.expression,
            ]
        )
        return SingleCaseCheck(
            case_id=self.case_id,
            input_repr=repr(self.args),
            expected_output_repr=repr(self.expected),
            actual_output_expr=f"{candidate_name}(*{self.args!r})",
            code="\n".join(lines),
        )


TestCase = Annotated[
    InputResultTestCase | InputOracleTestCase | InputExpressionTestCase,
    Field(discriminator="kind"),
]


class ParsedTests(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_type: HumanEvalTestCaseKind
    support_code: str
    check_name: str
    candidate_arg_name: str
    assertion_name: str
    cases: list[TestCase]
    original_test: str

    def iter_checks(
        self,
        *,
        candidate_name: str = "candidate",
    ) -> Iterator[SingleCaseCheck]:
        for case in self.cases:
            yield case.as_check(
                candidate_name=candidate_name,
                assertion_name=self.assertion_name,
            )


def _literal_assignment(function_node: ast.FunctionDef, name: str) -> Any:
    value = _find_assignment_value(function_node, name)
    if value is None:
        raise UnsupportedTestFormatError(
            f"Could not find assignment for {name!r}"
        )
    try:
        return ast.literal_eval(value)
    except ValueError as exc:
        raise UnsupportedTestFormatError(
            f"Assignment for {name!r} is not a literal"
        ) from exc


def _find_assignment_value(
    function_node: ast.FunctionDef,
    name: str,
) -> ast.expr | None:
    for stmt in function_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return stmt.value
    return None


def _find_for_loop(function_node: ast.FunctionDef) -> ast.For:
    for stmt in function_node.body:
        if isinstance(stmt, ast.For):
            return stmt
    raise UnsupportedTestFormatError(
        f"{function_node.name} does not contain a for loop"
    )


def _find_assertion_call(function_node: ast.FunctionDef) -> ast.Call | None:
    for node in ast.walk(function_node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assertion"
        ):
            return node
    return None


def _find_assert_statement(function_node: ast.FunctionDef) -> ast.Assert:
    for node in ast.walk(function_node):
        if isinstance(node, ast.Assert):
            return node
    raise UnsupportedTestFormatError(
        f"{function_node.name} does not contain assert ..."
    )


def _find_oracle_name(assertion_call: ast.Call) -> str | None:
    if len(assertion_call.args) <= EXPECTED_ARG_INDEX:
        return None
    expected_expr = assertion_call.args[EXPECTED_ARG_INDEX]
    if isinstance(expected_expr, ast.Call) and isinstance(
        expected_expr.func, ast.Name
    ):
        return expected_expr.func.id
    return None


def _assertion_tolerance(assertion_call: ast.Call) -> float:
    if len(assertion_call.args) <= TOLERANCE_ARG_INDEX:
        return 0
    value = assertion_call.args[TOLERANCE_ARG_INDEX]
    try:
        tolerance = ast.literal_eval(value)
    except ValueError as exc:
        raise UnsupportedTestFormatError(
            "Assertion tolerance is not a literal"
        ) from exc
    if isinstance(tolerance, int | float):
        return float(tolerance)
    raise UnsupportedTestFormatError("Assertion tolerance must be numeric")


def _for_loop_names(loop_node: ast.For) -> tuple[str | None, str, str]:
    target = loop_node.target
    if (
        isinstance(target, ast.Tuple)
        and len(target.elts) == PAIR_TARGET_SIZE
        and isinstance(target.elts[0], ast.Name)
        and isinstance(target.elts[1], ast.Tuple)
        and len(target.elts[1].elts) == PAIR_TARGET_SIZE
        and isinstance(target.elts[1].elts[0], ast.Name)
        and isinstance(target.elts[1].elts[1], ast.Name)
    ):
        return (
            target.elts[0].id,
            target.elts[1].elts[0].id,
            target.elts[1].elts[1].id,
        )
    if (
        isinstance(target, ast.Tuple)
        and len(target.elts) == PAIR_TARGET_SIZE
        and isinstance(target.elts[0], ast.Name)
        and isinstance(target.elts[1], ast.Name)
    ):
        return (None, target.elts[0].id, target.elts[1].id)
    raise UnsupportedTestFormatError("Unsupported for-loop target shape")


def find_check_function(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            return node
    raise UnsupportedTestFormatError(
        "Could not find check(candidate) function"
    )


def support_code_without_check(tree: ast.Module) -> str:
    support_nodes = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "check")
    ]
    module = ast.Module(body=support_nodes, type_ignores=[])
    return ast.unparse(module)


def parse_humaneval_tests(test_str: str) -> ParsedTests:
    tree = ast.parse(test_str)
    check_node = find_check_function(tree)
    if len(check_node.args.args) != 1:
        raise UnsupportedTestFormatError(
            "Expected check(candidate) with one positional argument"
        )

    inputs = _literal_assignment(check_node, "inputs")
    results_value = _find_assignment_value(check_node, "results")
    assertion_call = _find_assertion_call(check_node)
    tolerance = _assertion_tolerance(assertion_call) if assertion_call else 0
    support_code = support_code_without_check(tree)
    candidate_arg_name = check_node.args.args[0].arg

    cases: list[TestCase]
    if results_value is not None:
        results = _literal_assignment(check_node, "results")
        if len(inputs) != len(results):
            raise UnsupportedTestFormatError(
                f"len(inputs)={len(inputs)} does not match "
                f"len(results)={len(results)}"
            )
        if assertion_call is None:
            loop_node = _find_for_loop(check_node)
            index_name, input_name, expected_name = _for_loop_names(loop_node)
            assert_statement = _find_assert_statement(check_node)
            cases = [
                InputExpressionTestCase(
                    case_id=f"case_{index}",
                    args=args,
                    expected=expected,
                    expression=ast.unparse(assert_statement),
                    input_name=input_name,
                    expected_name=expected_name,
                    index_name=index_name,
                )
                for index, (args, expected) in enumerate(
                    zip(inputs, results, strict=True)
                )
            ]
            test_type = HumanEvalTestCaseKind.INPUT_EXPRESSION
        else:
            cases = [
                InputResultTestCase(
                    case_id=f"case_{index}",
                    args=args,
                    expected=expected,
                    atol=tolerance,
                )
                for index, (args, expected) in enumerate(
                    zip(inputs, results, strict=True)
                )
            ]
            test_type = HumanEvalTestCaseKind.INPUT_RESULT
    else:
        _ = _find_for_loop(check_node)
        if assertion_call is None:
            raise UnsupportedTestFormatError(
                "Expected assertion(..., ref_func(*inp), ...) for oracle tests"
            )
        oracle_name = _find_oracle_name(assertion_call)
        if oracle_name is None:
            raise UnsupportedTestFormatError(
                "Expected assertion(..., ref_func(*inp), ...) for oracle tests"
            )
        cases = [
            InputOracleTestCase(
                case_id=f"case_{index}",
                args=args,
                oracle_name=oracle_name,
                atol=tolerance,
            )
            for index, args in enumerate(inputs)
        ]
        test_type = HumanEvalTestCaseKind.INPUT_ORACLE

    return ParsedTests(
        test_type=test_type,
        support_code=support_code,
        check_name=check_node.name,
        candidate_arg_name=candidate_arg_name,
        assertion_name="assertion",
        cases=cases,
        original_test=test_str,
    )

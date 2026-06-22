"""Bridge dr-code HumanEval+ tasks to local execution inputs."""

from __future__ import annotations

import ast
from typing import Any, Literal

from dr_code.models.base import FrozenModel
from dr_code.models.humaneval import HumanEvalPlusTask

RUN_SINGLE_TEST_CASE_FUNCTION = "run_single_test_case"

HumanEvalTestShape = Literal["inputs_results", "inputs_ref_func"]


class TestCase(FrozenModel):
    """A single function-call test case."""

    __test__ = False

    input_value: Any
    expected_output: Any


class HumanEvalTestSuite(FrozenModel):
    """Parsed HumanEval+ test suite shape used by the test stage."""

    shape: HumanEvalTestShape
    inputs: tuple[Any, ...]
    results: tuple[Any, ...] | None = None


def supports_function_call_tests(task: HumanEvalPlusTask) -> bool:
    """Return True when per-case expected outputs are available."""
    return parse_humaneval_test(task).shape == "inputs_results"


def load_test_cases(task: HumanEvalPlusTask) -> list[TestCase]:
    """Load direct input/result TestCase rows for a dr-code task."""
    suite = parse_humaneval_test(task)
    if suite.shape != "inputs_results" or suite.results is None:
        msg = "sample does not provide expected test results"
        raise ValueError(msg)
    return [
        TestCase(input_value=input_value, expected_output=expected_output)
        for input_value, expected_output in zip(
            suite.inputs,
            suite.results,
            strict=True,
        )
    ]


def parse_humaneval_test(task: HumanEvalPlusTask) -> HumanEvalTestSuite:
    """Parse the HumanEval+ check() inputs/results literal assignments."""
    check_fn = _find_named_function(task.test, "check")
    _inputs_assign, inputs = _literal_list_assignment_in_body(
        check_fn.body,
        "inputs",
    )
    results_assign = _find_named_assignment_in_body(check_fn.body, "results")
    if results_assign is None:
        return HumanEvalTestSuite(
            shape="inputs_ref_func",
            inputs=tuple(inputs),
            results=None,
        )
    results = _literal_list_assignment_value(results_assign, "results")
    if len(inputs) != len(results):
        msg = "test inputs and results must have the same length"
        raise ValueError(msg)
    return HumanEvalTestSuite(
        shape="inputs_results",
        inputs=tuple(inputs),
        results=tuple(results),
    )


def build_eval_code(extracted_code: str, entry_point: str) -> str:
    """Wrap extracted solution body for per-case execution."""
    return "\n".join(
        [
            extracted_code.rstrip(),
            "",
            "",
            f"def {RUN_SINGLE_TEST_CASE_FUNCTION}(input_value):",
            f"    return {entry_point}(*input_value)",
            "",
        ],
    )


def run_function_name() -> str:
    """Function name used by the per-case execution helper."""
    return RUN_SINGLE_TEST_CASE_FUNCTION


def _find_named_function(source: str, function_name: str) -> ast.FunctionDef:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    msg = f"no function named {function_name!r} found"
    raise ValueError(msg)


def _literal_list_assignment_in_body(
    body: list[ast.stmt],
    name: str,
) -> tuple[ast.Assign, list[Any]]:
    assign = _find_named_assignment_in_body(body, name)
    if assign is None:
        msg = f"no `{name} = ...` assignment found"
        raise ValueError(msg)
    return assign, _literal_list_assignment_value(assign, name)


def _find_named_assignment_in_body(
    body: list[ast.stmt],
    name: str,
) -> ast.Assign | None:
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
        ):
            return stmt
    return None


def _literal_list_assignment_value(assign: ast.Assign, name: str) -> list[Any]:
    if not isinstance(assign.value, ast.List):
        msg = f"`{name}` assignment must be a list literal"
        raise TypeError(msg)
    value = ast.literal_eval(assign.value)
    if not isinstance(value, list):
        msg = f"`{name}` assignment must evaluate to a list"
        raise TypeError(msg)
    return value

"""Bridge dr-code HumanEval+ tasks to nl-code execution inputs."""

from __future__ import annotations

from nl_code.code_execution.models import TestCase
from nl_code.datasets.humaneval_task import HumanEvalSource, RawHumanEvalTask
from nl_code.optim.humaneval_dspy_eval import (
    RUN_SINGLE_TEST_CASE_FUNCTION,
    build_single_test_case_solution,
)
from nl_code.optim.humaneval_dspy_sample import has_function_call_tests, test_cases

from dr_code.models.humaneval import HumanEvalPlusTask


def task_to_raw(task: HumanEvalPlusTask) -> RawHumanEvalTask:
    """Convert a dr-code task row to nl-code's raw HumanEval model."""
    return RawHumanEvalTask(
        task_id=task.task_id,
        entry_point=task.entry_point,
        source=HumanEvalSource(
            prompt=task.prompt,
            canonical_solution=task.canonical_solution,
            test=task.test,
        ),
    )


def supports_function_call_tests(task: HumanEvalPlusTask) -> bool:
    """Return True when per-case expected outputs are available."""
    return has_function_call_tests(task_to_raw(task))


def load_test_cases(task: HumanEvalPlusTask) -> list[TestCase]:
    """Load nl-code TestCase rows for a dr-code task."""
    return test_cases(task_to_raw(task))


def build_eval_code(extracted_code: str, entry_point: str) -> str:
    """Wrap extracted solution body for nl-code per-case execution."""
    return build_single_test_case_solution(extracted_code, entry_point)


def run_function_name() -> str:
    """Function name passed to nl-code run_test_cases."""
    return RUN_SINGLE_TEST_CASE_FUNCTION

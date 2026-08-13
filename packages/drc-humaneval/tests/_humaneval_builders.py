from __future__ import annotations

from dr_exec import FakeExecutor

from _executor_stubs import scripted_executor
from drc_humaneval import HumanEvalTask


def _task(*, test: str | None = None) -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=test or _input_result_test(),
    )


def _row(task_id: str, offset: int) -> dict[str, str]:
    return {
        "task_id": task_id,
        "prompt": f"def f_{offset}(x):\n",
        "canonical_solution": f"    return x + {offset}\n",
        "entry_point": f"f_{offset}",
        "test": (
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            f"    results = [{1 + offset}]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    }


def _input_result_test() -> str:
    return (
        "def check(candidate):\n"
        "    inputs = [(1,), (2,)]\n"
        "    results = [2, 3]\n"
        "    for inp, expected in zip(inputs, results):\n"
        "        assertion(candidate(*inp), expected)\n"
    )


def _stub_executor(
    *,
    stdout: str,
    stderr: str = "",
    returncode: int = 0,
) -> FakeExecutor:
    return scripted_executor(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


_PARTIAL_RUNNER_PASSED_CASE_0 = (
    '{"kind": "case_results", "results": '
    '[{"case_id": "case_0", "status": "passed", "message": ""}]}'
)

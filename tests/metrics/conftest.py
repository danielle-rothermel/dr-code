from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from _executor_stubs import (
    CountingExecutor,
    local_python_executor,
    raising_executor,
)
from dr_exec import FakeExecutor
from dr_code.humaneval.task import (
    HumanEvalTask,
)
from dr_code.trace import (
    CodeArtifact,
    JsonArtifact,
    TextArtifact,
    Trace,
    external_trace,
)


_PROMPT = "def add_one(x):\n"
_CANONICAL = "    return x + 1\n"
_ENTRY_POINT = "add_one"
_INPUT_RESULT_TEST = (
    "def check(candidate):\n"
    "    inputs = [(1,), (2,)]\n"
    "    results = [2, 3]\n"
    "    for inp, expected in zip(inputs, results):\n"
    "        assertion(candidate(*inp), expected)\n"
)


def _make_task(
    *,
    task_id: str = "HumanEval/fixture",
    prompt: str = _PROMPT,
    canonical_solution: str = _CANONICAL,
    entry_point: str = _ENTRY_POINT,
    test: str | None = None,
) -> HumanEvalTask:
    return HumanEvalTask(
        task_id=task_id,
        prompt=prompt,
        canonical_solution=canonical_solution,
        entry_point=entry_point,
        test=test or _INPUT_RESULT_TEST,
    )


@pytest.fixture
def task() -> HumanEvalTask:
    return _make_task()


def _text_trace(
    text: str, namespace: Mapping[str, object] | None = None
) -> Trace:
    values: dict[str, object] = {
        "input": TextArtifact(text=text),
        "output": TextArtifact(text=text),
    }
    if namespace:
        values.update(namespace)
    return external_trace(values)


def _task_json_artifact(task: HumanEvalTask) -> JsonArtifact:
    return JsonArtifact(payload=task.model_dump(mode="json"))


def _code_test_trace(
    candidate_code: str,
    task: HumanEvalTask,
    *,
    code_key: str = "input",
    task_key: str = "task",
) -> Trace:
    code = CodeArtifact(source=candidate_code)
    return external_trace(
        {
            "input": code,
            "output": code,
            code_key: code,
            task_key: _task_json_artifact(task),
        }
    )


@pytest.fixture
def text_trace() -> Callable[..., Trace]:
    return _text_trace


@pytest.fixture
def code_test_trace() -> Callable[..., Trace]:
    return _code_test_trace


@pytest.fixture
def local_executor() -> FakeExecutor:
    return local_python_executor()


@pytest.fixture
def counting_executor() -> CountingExecutor:
    return CountingExecutor(local_python_executor())


@pytest.fixture
def raising() -> Callable[[BaseException], FakeExecutor]:
    return raising_executor

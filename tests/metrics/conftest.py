from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping

import pytest

from dr_code.core.execution.sandbox import (
    SandboxCompletedProcess,
    SandboxRunner,
    SandboxTimeoutError,
)
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


def _local_runner() -> SandboxRunner:
    def run_local_python(
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", source],
                input=input_json,
                capture_output=True,
                check=False,
                encoding="utf-8",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(str(exc)) from exc
        return SandboxCompletedProcess(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return run_local_python


class CountingRunner:
    def __init__(self, inner: SandboxRunner) -> None:
        self._inner = inner
        self.calls: list[tuple[str, str, float]] = []

    def __call__(
        self,
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        self.calls.append((source, input_json, timeout_seconds))
        return self._inner(
            source=source,
            input_json=input_json,
            timeout_seconds=timeout_seconds,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _raising_runner(exc: BaseException) -> SandboxRunner:
    def run(
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        raise exc

    return run


@pytest.fixture
def local_runner() -> SandboxRunner:
    return _local_runner()


@pytest.fixture
def counting_runner(local_runner: SandboxRunner) -> CountingRunner:
    return CountingRunner(local_runner)


@pytest.fixture
def raising_runner() -> Callable[[BaseException], SandboxRunner]:
    return _raising_runner

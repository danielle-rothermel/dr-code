"""Pytest fixtures for the ``dr_code.metrics`` contract tests.

Builders and runner fakes are exposed as fixtures discovered from this
directory. Metrics symbols are imported by the tests that exercise them;
shared fixtures depend only on ``dr_code.trace`` and
``dr_code.humaneval.*``.

``dr_code.humaneval`` supplies comparison implementations, and
``dr_code.trace`` supplies the input contract. Execution stays behind the
injectable ``SandboxRunner`` seam, so nothing here touches a real
container runtime.
"""

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


# ---------------------------------------------------------------------------
# HumanEval task fixtures / builders.
# ---------------------------------------------------------------------------

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
    """Build a two-case input/result HumanEval task (the metrics oracle)."""
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


# ---------------------------------------------------------------------------
# Trace builders: fresh, deserialized, and external traces produce equal
# records.
# ---------------------------------------------------------------------------


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
    """A JsonArtifact carrying a serialised HumanEvalTask payload.

    ``code_test`` revalidates this back to ``HumanEvalTask`` at bind time.
    """
    return JsonArtifact(payload=task.model_dump(mode="json"))


def _code_test_trace(
    candidate_code: str,
    task: HumanEvalTask,
    *,
    code_key: str = "input",
    task_key: str = "task",
) -> Trace:
    """A trace carrying candidate code + task for the ``code_test`` operator."""
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
    """Build a text trace whose ``input``/``output`` carry the same text."""
    return _text_trace


@pytest.fixture
def code_test_trace() -> Callable[..., Trace]:
    """Build a candidate-code + task trace for the ``code_test`` operator."""
    return _code_test_trace


# ---------------------------------------------------------------------------
# Injectable SandboxRunner fakes.
# ---------------------------------------------------------------------------


def _local_runner() -> SandboxRunner:
    """An injectable runner that runs the trusted program under the host
    interpreter — fast and container-free. The OCI boundary has its own
    probes elsewhere (``tests/humaneval/test_sandbox.py``)."""

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
    """A runner that records every call and delegates to an inner runner.

    The engine deduplicates equivalent requests, so identical requests
    execute once per cache lifetime.
    """

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
    """A runner that always raises (infra breakage or candidate timeout)."""

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
    """Injectable host-interpreter runner (no OCI container)."""
    return _local_runner()


@pytest.fixture
def counting_runner(local_runner: SandboxRunner) -> CountingRunner:
    """A counting wrapper around the local runner (observes at-most-once)."""
    return CountingRunner(local_runner)


@pytest.fixture
def raising_runner() -> Callable[[BaseException], SandboxRunner]:
    """Build a runner that always raises the supplied exception."""
    return _raising_runner

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

import json
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
    EvaluationCaseStatus,
    EvaluationTaskResult,
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


@pytest.fixture
def good_submission() -> str:
    """A submission that passes every case of the ``task`` fixture."""
    return "def add_one(x):\n    return x + 1\n"


@pytest.fixture
def failing_submission() -> str:
    """A submission that compiles and runs but fails assertions."""
    return "def add_one(x):\n    return x - 1\n"


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


def _scripted_runner(
    *,
    stdout: str = "[]",
    stderr: str = "",
    returncode: int = 0,
) -> SandboxRunner:
    """Build a runner that returns a fixed completed process."""

    def run(
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        return SandboxCompletedProcess(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return run


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
def scripted_runner() -> Callable[..., SandboxRunner]:
    """Build a runner returning a fixed completed process."""
    return _scripted_runner


@pytest.fixture
def raising_runner() -> Callable[[BaseException], SandboxRunner]:
    """Build a runner that always raises the supplied exception."""
    return _raising_runner


# ---------------------------------------------------------------------------
# Scripted runner-JSON builders for deterministic code_test parity.
# These script the runner's stdout so parity does not need a subprocess.
# ---------------------------------------------------------------------------


def _partial_pass_runner_output(
    *,
    passed: tuple[str, ...] = ("case_0",),
    case_ids: tuple[str, ...] = ("case_0", "case_1"),
) -> str:
    """Runner JSON that passes ``passed`` and fails the rest of ``case_ids``."""
    payload = []
    for case_id in case_ids:
        status = (
            EvaluationCaseStatus.PASSED.value
            if case_id in passed
            else EvaluationCaseStatus.FAILED.value
        )
        payload.append(
            {
                "case_id": case_id,
                "status": status,
                "message": "",
                "input_repr": "[1]",
                "expected_output_repr": "2",
                "actual_output_repr": "2",
                "elapsed_seconds": 0.0,
                "timeout_seconds": None,
            }
        )
    return json.dumps(payload)


@pytest.fixture
def partial_pass_runner_output() -> Callable[..., str]:
    """Build runner JSON passing some case ids and failing the rest."""
    return _partial_pass_runner_output


# ---------------------------------------------------------------------------
# Oracle runner: delegate to the existing runner for parity comparisons.
# ---------------------------------------------------------------------------


def _evaluate_oracle(
    task: HumanEvalTask,
    candidate_code: str,
    *,
    timeout_seconds: float,
    run_in_sandbox: SandboxRunner,
) -> EvaluationTaskResult:
    """Run the existing runner to get the oracle EvaluationTaskResult."""
    from dr_code.humaneval.runner import evaluate_humaneval_code

    return evaluate_humaneval_code(
        task=task,
        candidate_code=candidate_code,
        timeout_seconds=timeout_seconds,
        run_in_sandbox=run_in_sandbox,
    )


@pytest.fixture
def evaluate_oracle() -> Callable[..., EvaluationTaskResult]:
    """Evaluate a candidate through ``runner`` for parity comparison."""
    return _evaluate_oracle

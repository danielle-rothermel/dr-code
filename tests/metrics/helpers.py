"""Pure builders and fakes for the ``dr_code.metrics`` acceptance suite.

Pure helpers (no pytest fixtures) so test modules import them directly. Pytest
fixtures live in ``conftest.py``. Import as ``from metrics.helpers import ...``.

The existing ``dr_code.humaneval`` modules are used as **oracles**;
``dr_code.trace`` is the input contract. Execution stays behind the injectable
``PythonSubprocessRunner`` seam.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

from dr_code.execution.subprocess import (
    PythonSubprocessRunner,
    SubprocessCompletedProcess,
    SubprocessTimeoutError,
)
from dr_code.humaneval.batch_runner import CANDIDATE_KILL_RETURNCODES
from dr_code.humaneval.task import EvaluationCaseStatus, HumanEvalTask
from dr_code.trace import (
    Absent,
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


def make_task(
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


# ---------------------------------------------------------------------------
# Trace builders (X-S2: fresh / deserialized / external all produce equal
# records).
# ---------------------------------------------------------------------------

def text_trace(text: str, namespace: Mapping[str, object] | None = None) -> Trace:
    values: dict[str, object] = {
        "input": TextArtifact(text=text),
        "output": TextArtifact(text=text),
    }
    if namespace:
        values.update(namespace)
    return external_trace(values)


def code_trace(source: str, namespace: Mapping[str, object] | None = None) -> Trace:
    code = CodeArtifact(source=source)
    values: dict[str, object] = {
        "input": code,
        "output": code,
    }
    if namespace:
        values.update(namespace)
    return external_trace(values)


def task_json_artifact(task: HumanEvalTask) -> JsonArtifact:
    """A JsonArtifact carrying a serialised HumanEvalTask payload.

    ``code_test`` revalidates this back to ``HumanEvalTask`` at bind time.
    """
    return JsonArtifact(payload=task.model_dump(mode="json"))


def code_test_trace(
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
            task_key: task_json_artifact(task),
        }
    )


def absent_trace(
    *,
    key: str = "input",
    failed_step: str = "extract",
    cause: str = "no code extracted",
) -> Trace:
    """A trace whose ``key`` is Absent with causal lineage."""
    return external_trace(
        {
            "input": Absent(failed_step=failed_step, cause=cause),
            "output": Absent(failed_step=failed_step, cause=cause),
            key: Absent(failed_step=failed_step, cause=cause),
        }
    )


# ---------------------------------------------------------------------------
# Injectable runner fakes.
# ---------------------------------------------------------------------------

def local_runner() -> PythonSubprocessRunner:
    """An injectable runner that runs the trusted program under the host
    interpreter."""

    def run_local_python(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", source],
                input=input_text,
                capture_output=True,
                check=False,
                encoding="utf-8",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise SubprocessTimeoutError(str(exc)) from exc
        return SubprocessCompletedProcess(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return run_local_python


class CountingRunner:
    """A runner that records every call and delegates to an inner runner.

    Observes at-most-once execution (X-S4): the engine dedupes by content
    hash so identical requests execute once per cache lifetime.
    """

    def __init__(self, inner: PythonSubprocessRunner) -> None:
        self._inner = inner
        self.calls: list[tuple[str, str, float]] = []

    def __call__(
        self,
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        self.calls.append((source, input_text, timeout_seconds))
        return self._inner(
            source=source,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


def scripted_runner(
    *,
    stdout: str = "[]",
    stderr: str = "",
    returncode: int = 0,
) -> PythonSubprocessRunner:
    """Build a runner that returns a fixed completed process."""

    def run(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        return SubprocessCompletedProcess(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return run


def raising_runner(exc: BaseException) -> PythonSubprocessRunner:
    """A runner that always raises (infra breakage or candidate timeout)."""

    def run(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        raise exc

    return run


# ---------------------------------------------------------------------------
# Scripted runner-JSON builders for deterministic code_test parity.
# These script the runner's stdout so parity does not need a subprocess.
# ---------------------------------------------------------------------------

def case_result(
    *,
    case_id: str,
    status: str,
    message: str = "",
    input_repr: str = "[1]",
    expected_output_repr: str = "2",
    actual_output_repr: str = "2",
    elapsed_seconds: float | None = 0.0,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": status,
        "message": message,
        "input_repr": input_repr,
        "expected_output_repr": expected_output_repr,
        "actual_output_repr": actual_output_repr,
        "elapsed_seconds": elapsed_seconds,
        "timeout_seconds": timeout_seconds,
    }


def full_pass_runner_output(
    case_ids: tuple[str, ...] = ("case_0", "case_1"),
) -> str:
    """Runner JSON that passes every supplied case id."""
    return json.dumps(
        [
            case_result(case_id=case_id, status=EvaluationCaseStatus.PASSED.value)
            for case_id in case_ids
        ]
    )


def partial_pass_runner_output(
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
        payload.append(case_result(case_id=case_id, status=status))
    return json.dumps(payload)


def kill_runner_process(
    returncode: int = next(iter(CANDIDATE_KILL_RETURNCODES)),
) -> SubprocessCompletedProcess:
    """A completed-process shape for the candidate-kill attribution path."""
    return SubprocessCompletedProcess(returncode=returncode, stdout="", stderr="killed")


def json_runner(
    results: list[dict[str, Any]] | None = None,
) -> tuple[PythonSubprocessRunner, list[tuple[str, str, float]]]:
    """A deterministic fake runner plus its call log (source, input, timeout)."""
    calls: list[tuple[str, str, float]] = []
    payload = json.dumps(results or [])

    def run(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        calls.append((source, input_text, timeout_seconds))
        return SubprocessCompletedProcess(returncode=0, stdout=payload, stderr="")

    return run, calls


# ---------------------------------------------------------------------------
# Oracle runner: delegate to the existing batch_runner for parity comparisons.
# ---------------------------------------------------------------------------

def evaluate_oracle(
    task: HumanEvalTask,
    candidate_code: str,
    *,
    timeout_seconds: float,
    run_in_subprocess: PythonSubprocessRunner,
):
    """Run the existing batch_runner to get the oracle EvaluationTaskResult."""
    from dr_code.humaneval.batch_runner import evaluate_human_eval_code

    return evaluate_human_eval_code(
        task=task,
        candidate_code=candidate_code,
        timeout_seconds=timeout_seconds,
        run_in_subprocess=run_in_subprocess,
    )

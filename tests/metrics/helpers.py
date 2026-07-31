"""Pure builders and executor doubles for the ``dr_code.metrics`` suite.

Pure helpers (no pytest fixtures) so test modules import them directly. Pytest
fixtures live in ``conftest.py``. Import as ``from metrics.helpers import ...``.

Logic tests drive a ``FakeExecutor`` scripted with dr-exec outcomes; parity and
oracle tests drive the real batch executor with ``Records.none()`` — the
contract's sanctioned way to genuinely execute a payload in a test. The
existing ``dr_code.humaneval`` modules are the oracle; ``dr_code.trace`` is the
input contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dr_exec import (
    Attribution,
    BatchRequest,
    BatchResult,
    BudgetAxis,
    ContainmentProfile,
    Budgets,
    EnvironmentGrant,
    ExitVerdict,
    FakeExecutor,
    ItemResult,
    Measurements,
    Outcome,
    OutputBudget,
    OverflowPolicy,
    PythonRuntime,
    Records,
    RunResult,
    ScriptedBatch,
    TruncationMark,
)

from dr_code.humaneval.batch_runner import (
    PRODUCTION_EXECUTOR,
    CANDIDATE_KILL_RETURNCODES,
)
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
# dr-exec RunResult / ScriptedBatch builders.
# ---------------------------------------------------------------------------

def _measurements(*, stdout: str = "", stderr: str = "") -> Measurements:
    return Measurements(
        duration_seconds=0.0,
        teardown_seconds=0.0,
        stdout_bytes_produced=len(stdout.encode("utf-8")),
        stderr_bytes_produced=len(stderr.encode("utf-8")),
        input_bytes=0,
    )


def clean_run(*, stderr: str = "") -> RunResult:
    """A payload run that exited 0 cleanly (the healthy batch child)."""
    return RunResult(
        returncode=0,
        stdout="",
        stderr=stderr,
        truncation=TruncationMark(),
        measurements=_measurements(stderr=stderr),
        outcome=Outcome(
            attribution=Attribution.PAYLOAD,
            exit_verdict=ExitVerdict.REPORT_ONLY,
        ),
    )


def killed_run(returncode: int, *, stderr: str = "killed") -> RunResult:
    """A payload run whose child died on a signal (candidate crash)."""
    return RunResult(
        returncode=returncode,
        stdout="",
        stderr=stderr,
        truncation=TruncationMark(),
        measurements=_measurements(stderr=stderr),
        outcome=Outcome(
            attribution=Attribution.PAYLOAD,
            exit_verdict=ExitVerdict.REPORT_ONLY,
        ),
    )


def wall_clock_run(output_budget: OutputBudget | None = None) -> RunResult:
    """A run killed on its wall-clock budget."""
    return RunResult(
        returncode=-9,
        stdout="",
        stderr="",
        truncation=TruncationMark(),
        measurements=_measurements(),
        outcome=Outcome(
            attribution=Attribution.BUDGET,
            violated_axis=BudgetAxis.WALL_CLOCK,
        ),
    )


def output_budget_run() -> RunResult:
    """A run killed on its output budget."""
    return RunResult(
        returncode=-9,
        stdout="",
        stderr="x",
        truncation=TruncationMark(stderr_bytes_dropped=1024),
        measurements=Measurements(
            duration_seconds=0.0,
            teardown_seconds=0.0,
            stdout_bytes_produced=0,
            stderr_bytes_produced=1025,
            input_bytes=0,
        ),
        outcome=Outcome(
            attribution=Attribution.BUDGET,
            violated_axis=BudgetAxis.OUTPUT,
        ),
    )


def passed_payload(
    *,
    input_repr: str = "[1]",
    expected_output_repr: str = "2",
) -> dict[str, Any]:
    return {
        "status": EvaluationCaseStatus.PASSED.value,
        "message": "",
        "input_repr": input_repr,
        "expected_output_repr": expected_output_repr,
        "actual_output_repr": "",
        "elapsed_seconds": 0.0,
    }


def failed_payload(*, message: str = "assertion failed") -> dict[str, Any]:
    return {
        "status": EvaluationCaseStatus.FAILED.value,
        "message": message,
        "input_repr": "[1]",
        "expected_output_repr": "2",
        "actual_output_repr": "0",
        "elapsed_seconds": 0.0,
    }


def scripted_batch(
    *,
    case_payloads: Mapping[str, Any],
    run: RunResult | None = None,
    completion_seen: bool = True,
) -> ScriptedBatch:
    """A ScriptedBatch delivering ``case_payloads`` (item_id -> payload)."""
    results = tuple(
        ItemResult(item_id=item_id, payload=payload)
        for item_id, payload in case_payloads.items()
    )
    return ScriptedBatch(
        run=run if run is not None else clean_run(),
        results=results,
        completion_seen=completion_seen,
    )


def full_pass_batch(
    case_ids: Sequence[str] = ("case_0", "case_1"),
) -> ScriptedBatch:
    return scripted_batch(
        case_payloads={case_id: passed_payload() for case_id in case_ids}
    )


def partial_pass_batch(
    *,
    passed: Sequence[str] = ("case_0",),
    case_ids: Sequence[str] = ("case_0", "case_1"),
) -> ScriptedBatch:
    payloads = {
        case_id: (passed_payload() if case_id in passed else failed_payload())
        for case_id in case_ids
    }
    return scripted_batch(case_payloads=payloads)


# ---------------------------------------------------------------------------
# Executor doubles.
# ---------------------------------------------------------------------------

def fake_executor_scripted(*batches: ScriptedBatch) -> FakeExecutor:
    """A FakeExecutor answering each run_batch call with the next batch, FIFO."""
    fake = FakeExecutor()
    for batch in batches:
        fake.enqueue_batch(batch)
    return fake


def fake_executor_always(batch_for: Any) -> FakeExecutor:
    """A FakeExecutor answering every run_batch via a callable over the call."""
    fake = FakeExecutor()
    fake.script_batches_with(batch_for)
    return fake


class CountingExecutor:
    """Wraps an executor and counts run_batch calls (observes at-most-once)."""

    def __init__(self, inner: Any = PRODUCTION_EXECUTOR) -> None:
        self._inner = inner
        self.calls: list[BatchRequest] = []

    def run_batch(
        self,
        request: BatchRequest,
        *,
        profile: ContainmentProfile,
        budgets: Budgets,
        records: Records,
        runtime: PythonRuntime,
        environment: EnvironmentGrant,
    ) -> BatchResult:
        self.calls.append(request)
        return self._inner.run_batch(
            request,
            profile=profile,
            budgets=budgets,
            records=records,
            runtime=runtime,
            environment=environment,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


# ---------------------------------------------------------------------------
# Oracle: delegate to the real batch_runner for parity comparisons.
# ---------------------------------------------------------------------------

def evaluate_oracle(
    task: HumanEvalTask,
    candidate_code: str,
    *,
    timeout_seconds: float,
    executor: Any = PRODUCTION_EXECUTOR,
):
    """Run the real batch_runner to get the oracle EvaluationTaskResult."""
    from dr_code.humaneval.batch_runner import evaluate_human_eval_code

    return evaluate_human_eval_code(
        task=task,
        candidate_code=candidate_code,
        timeout_seconds=timeout_seconds,
        executor=executor,
        records=Records.none(),
    )


__all__ = [
    "CANDIDATE_KILL_RETURNCODES",
    "PRODUCTION_EXECUTOR",
    "CountingExecutor",
    "OutputBudget",
    "OverflowPolicy",
    "absent_trace",
    "clean_run",
    "code_test_trace",
    "code_trace",
    "evaluate_oracle",
    "failed_payload",
    "fake_executor_always",
    "fake_executor_scripted",
    "full_pass_batch",
    "killed_run",
    "make_task",
    "output_budget_run",
    "partial_pass_batch",
    "passed_payload",
    "scripted_batch",
    "task_json_artifact",
    "text_trace",
    "wall_clock_run",
]

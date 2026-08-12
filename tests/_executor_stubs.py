"""Shared dr-exec executor stubs for dr-code tests.

Every stub is a dr-exec ``FakeExecutor`` so scripted completions still pass
declaration validation. ``local_python_executor`` really runs the driver in
a local subprocess through a minimal stand-in bootstrap: dr-exec's
production engine is platform-restricted, so local runs substitute for it
while preserving the driver contract (request on stdin, results on stdout).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from uuid import uuid4

from dr_exec import (
    AttemptId,
    BudgetAxis,
    BudgetExceededOutcome,
    CancelledOutcome,
    CompletedExecution,
    ExecutionAttribution,
    ExecutionId,
    ExecutionJob,
    ExecutionMeasurements,
    ExecutionOutcome,
    ExecutionResult,
    ExitedOutcome,
    FailureOwner,
    FakeExecutor,
    FakeRecordReceipt,
    FiniteDurationLimit,
    PayloadOutputs,
    ProtocolFailedOutcome,
    ProtocolFailureCode,
    RetainedPayloadStream,
    SignaledOutcome,
    SpawnAbsentOutcome,
    SpawnFailedOutcome,
    UntrustedPythonTarget,
)
from dr_serialize import (
    IdentityDocument,
    build_identity_document,
    validate_identity_document,
)

_LOCAL_BOOTSTRAP = (
    "\n"
    "import json as _stub_json\n"
    "import sys as _stub_sys\n"
    "_stub_results = _stub_sys.stdout\n"
    "_stub_sys.stdout = _stub_sys.stderr\n"
    "def _stub_emit(_document):\n"
    "    _stub_results.write(_stub_json.dumps(_document) + '\\n')\n"
    "    _stub_results.flush()\n"
    "dr_exec_main(\n"
    "    _stub_json.loads(_stub_sys.stdin.buffer.read().decode('utf-8')),\n"
    "    _stub_emit,\n"
    ")\n"
)


def _attribute(outcome: ExecutionOutcome) -> ExecutionAttribution:
    """Mirror dr-exec's engine attribution for scripted completions."""

    match outcome:
        case ExitedOutcome():
            owner = (
                FailureOwner.NONE
                if outcome.exit_code == 0
                else FailureOwner.PAYLOAD
            )
        case SignaledOutcome():
            owner = FailureOwner.PAYLOAD
        case BudgetExceededOutcome():
            owner = (
                FailureOwner.EXECUTOR
                if outcome.axis is BudgetAxis.WALL_TIME
                else FailureOwner.PAYLOAD
            )
        case ProtocolFailedOutcome():
            owner = (
                FailureOwner.EXECUTOR
                if outcome.failure_code is ProtocolFailureCode.OVERSIZED_FRAME
                else FailureOwner.PAYLOAD
            )
        case SpawnAbsentOutcome():
            owner = FailureOwner.EXECUTOR
        case SpawnFailedOutcome():
            owner = FailureOwner.MACHINE
        case CancelledOutcome():
            owner = FailureOwner.NONE
    return ExecutionAttribution(owner=owner)


def _stream(data: bytes) -> RetainedPayloadStream:
    return RetainedPayloadStream(
        head=data,
        tail=b"",
        produced_bytes=len(data),
        dropped_bytes=0,
    )


def completed_execution(
    job: ExecutionJob,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    outcome: ExecutionOutcome | None = None,
    protocol_outputs: tuple[IdentityDocument, ...] = (),
) -> CompletedExecution:
    if outcome is None:
        outcome = (
            ExitedOutcome(exit_code=returncode)
            if returncode >= 0
            else SignaledOutcome(signal_number=-returncode)
        )
    execution_id = ExecutionId(
        job_id=job.job_id,
        attempt_id=AttemptId(uuid4()),
    )
    now = datetime.now(UTC)
    return CompletedExecution(
        result=ExecutionResult(
            execution_id=execution_id,
            outcome=outcome,
            attribution=_attribute(outcome),
            protocol_outputs=protocol_outputs,
            payload_outputs=PayloadOutputs(
                stdout=_stream(stdout.encode("utf-8")),
                stderr=_stream(stderr.encode("utf-8")),
            ),
            measurements=ExecutionMeasurements(
                started_at=now,
                finished_at=now,
                duration_ns=0,
                teardown_duration_ns=0,
                input_bytes=0,
                protocol_bytes_received=0,
            ),
        ),
        record_receipt=FakeRecordReceipt(execution_id=execution_id),
    )


def scripted_executor(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    outcome: ExecutionOutcome | None = None,
) -> FakeExecutor:
    """Return the same scripted completion for every job."""

    def respond(job: ExecutionJob, cancellation: object) -> CompletedExecution:
        del cancellation
        return completed_execution(
            job,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            outcome=outcome,
        )

    return FakeExecutor(responder=respond)


def timeout_executor() -> FakeExecutor:
    return scripted_executor(
        outcome=BudgetExceededOutcome(axis=BudgetAxis.WALL_TIME)
    )


def output_limit_executor() -> FakeExecutor:
    return scripted_executor(
        outcome=BudgetExceededOutcome(axis=BudgetAxis.PAYLOAD_OUTPUT)
    )


def raising_executor(exc: BaseException) -> FakeExecutor:
    def respond(job: ExecutionJob, cancellation: object) -> CompletedExecution:
        del job, cancellation
        raise exc

    return FakeExecutor(responder=respond)


def local_python_executor() -> FakeExecutor:
    """Really run the declared driver in a local subprocess."""

    def respond(job: ExecutionJob, cancellation: object) -> CompletedExecution:
        del cancellation
        target = job.target
        assert isinstance(target, UntrustedPythonTarget)
        wall_time = job.budgets.wall_time
        timeout = (
            wall_time.max_ns / 1e9
            if isinstance(wall_time, FiniteDurationLimit)
            else None
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    target.driver_source + _LOCAL_BOOTSTRAP,
                ],
                input=json.dumps(target.request.to_json_dict()),
                capture_output=True,
                check=False,
                encoding="utf-8",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return completed_execution(
                job,
                outcome=BudgetExceededOutcome(axis=BudgetAxis.WALL_TIME),
            )
        return completed_execution(
            job,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            protocol_outputs=tuple(
                validate_identity_document(json.loads(line))
                for line in completed.stdout.splitlines()
                if line
            ),
        )

    return FakeExecutor(responder=respond)


def importable_json_executor() -> FakeExecutor:
    """Run the installed HumanEval entry point without a real-time subprocess."""

    def respond(job: ExecutionJob, cancellation: object) -> CompletedExecution:
        del cancellation
        from dr_code.humaneval.job import evaluate_humaneval_candidate_job

        target = job.target
        assert isinstance(target, UntrustedPythonTarget)
        result = evaluate_humaneval_candidate_job(target.request.payload)
        output = build_identity_document(
            schema="dr_exec.importable_json",
            schema_version=1,
            payload=result,
        )
        return completed_execution(job, protocol_outputs=(output,))

    return FakeExecutor(responder=respond)


class CountingExecutor:
    """Executor wrapper recording every job it forwards."""

    def __init__(self, inner: FakeExecutor) -> None:
        self._inner = inner
        self.calls: list[ExecutionJob] = []

    def run_blocking(
        self,
        job: ExecutionJob,
        /,
        *,
        cancellation: object = None,
    ) -> CompletedExecution:
        self.calls.append(job)
        return self._inner.run_blocking(job, cancellation=None)

    def run(
        self,
        job: ExecutionJob,
        /,
        *,
        cancellation: object = None,
    ) -> CompletedExecution:
        return self.run_blocking(job, cancellation=cancellation)

    @property
    def call_count(self) -> int:
        return len(self.calls)


class ConcurrencyTrackingExecutor:
    """Executor wrapper recording peak concurrent run() calls."""

    def __init__(
        self,
        inner: FakeExecutor,
        *,
        delay_seconds: float = 0.01,
    ) -> None:
        import threading

        self._inner = inner
        self._delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run_blocking(
        self,
        job: ExecutionJob,
        /,
        *,
        cancellation: object = None,
    ) -> CompletedExecution:
        import time

        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self._delay_seconds:
                time.sleep(self._delay_seconds)
            return self._inner.run_blocking(job, cancellation=None)
        finally:
            with self._lock:
                self.active -= 1

    def run(
        self,
        job: ExecutionJob,
        /,
        *,
        cancellation: object = None,
    ) -> CompletedExecution:
        return self.run_blocking(job, cancellation=cancellation)

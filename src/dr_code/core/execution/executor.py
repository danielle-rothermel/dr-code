"""Python execution through dr-exec executors.

dr-code declares one execution shape: an untrusted Python driver evaluated
against a JSON request under finite wall-clock, input, and payload-output
budgets. Jobs run through any dr-exec ``Executor``; dr-exec's typed outcome
and attribution taxonomy is interpreted back into this repository's
candidate-versus-harness semantics. Exited, signaled, wall-time, and
payload-output outcomes have fixed mappings; payload-owned protocol failures
are candidate-attributable kills; every other outcome fails closed as
dr-exec's ``ExecutorFailure``.

Under subprocess execution, submitted programs are not contained: the process
boundary retains the invoking worker's permissions, and worker isolation is
the deployment boundary.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Never

from dr_serialize import build_identity_document

from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    Budgets,
    CompletedExecution,
    ContainmentProfile,
    DeclarationError,
    DirectoryRunStore,
    EnvGrant,
    ExecutionJob,
    Executor,
    ExecutorFailure,
    ExitedOutcome,
    FailureOwner,
    FiniteByteLimit,
    FiniteDurationLimit,
    FiniteOutput,
    IsolatedHostPythonRuntime,
    JobId,
    OutputOverflowPolicy,
    PayloadRetentionBudget,
    ProcessExecutor,
    ProtocolFailedOutcome,
    RetainedPayloadStream,
    SignaledOutcome,
    StreamRetentionBudget,
    UntrustedPythonTarget,
)

EXECUTION_REQUEST_SCHEMA: Final[str] = "dr-code/python-execution-request"
EXECUTION_REQUEST_SCHEMA_VERSION: Final[int] = 1
MAX_EXECUTION_INPUT_BYTES: Final[int] = 1_048_576
MAX_EXECUTION_STREAM_BYTES: Final[int] = 1_048_576
_NANOSECONDS_PER_SECOND: Final[int] = 1_000_000_000
# The child inherits nothing from the operator environment; hashing stays
# deterministic and bytecode caches stay off.
_EXECUTION_ENVIRONMENT: Final[dict[str, str]] = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}


class ExecutionTimeoutError(RuntimeError):
    """The payload exhausted its wall-clock budget (candidate-owned)."""


class ExecutionOutputLimitError(RuntimeError):
    """The payload exhausted an output budget (candidate-owned)."""


class ExecutionKilledError(RuntimeError):
    """The payload died mid-run without completing (candidate-owned).

    Covers signals (resource-exhaustion SIGKILL, interpreter-crash SIGSEGV)
    and payload-owned protocol breakage such as an incomplete protected
    stream after a forced exit.
    """


@dataclass(frozen=True, slots=True)
class CompletedPythonProcess:
    returncode: int
    stdout: str
    stderr: str


def _reject_json_constant(constant: str) -> Never:
    raise ValueError(f"invalid JSON constant: {constant}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def build_python_execution_job(
    *,
    driver_source: str,
    input_json: str,
    timeout_seconds: float,
) -> ExecutionJob:
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise DeclarationError("execution timeout must be finite and positive")
    try:
        payload = json.loads(
            input_json,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except ValueError as exc:
        raise DeclarationError(
            "execution input must be strict JSON text"
        ) from exc
    timeout_nanoseconds = timeout_seconds * _NANOSECONDS_PER_SECOND
    if not math.isfinite(timeout_nanoseconds):
        raise DeclarationError(
            "execution timeout is too large to represent in nanoseconds"
        )
    return ExecutionJob(
        job_id=JobId(uuid.uuid4()),
        target=UntrustedPythonTarget(
            driver_source=driver_source,
            request=build_identity_document(
                schema=EXECUTION_REQUEST_SCHEMA,
                schema_version=EXECUTION_REQUEST_SCHEMA_VERSION,
                payload=payload,
            ),
            containment_profile=ContainmentProfile.PROCESS_BOUNDARY_ONLY,
        ),
        env=EnvGrant.fixed(_EXECUTION_ENVIRONMENT),
        budgets=Budgets(
            wall_time=FiniteDurationLimit(
                max_ns=math.ceil(timeout_nanoseconds)
            ),
            input_bytes=FiniteByteLimit(max_bytes=MAX_EXECUTION_INPUT_BYTES),
            payload_output=FiniteOutput(
                max_bytes=2 * MAX_EXECUTION_STREAM_BYTES,
                overflow_policy=OutputOverflowPolicy.FAIL,
                retention=PayloadRetentionBudget(
                    stdout=StreamRetentionBudget(
                        head_bytes=MAX_EXECUTION_STREAM_BYTES,
                        tail_bytes=0,
                    ),
                    stderr=StreamRetentionBudget(
                        head_bytes=MAX_EXECUTION_STREAM_BYTES,
                        tail_bytes=0,
                    ),
                ),
            ),
        ),
    )


def run_python_source(
    executor: Executor | None,
    *,
    source: str,
    input_json: str,
    timeout_seconds: float,
) -> CompletedPythonProcess:
    """Run one Python driver job and interpret its typed completion."""

    if executor is None:
        raise ExecutorFailure("no executor was provided for execution")
    job = build_python_execution_job(
        driver_source=source,
        input_json=input_json,
        timeout_seconds=timeout_seconds,
    )
    return interpret_completed_execution(executor.run(job))


def interpret_completed_execution(
    execution: CompletedExecution,
) -> CompletedPythonProcess:
    """Map dr-exec's outcome and attribution onto dr-code semantics."""

    result = execution.result
    outcome = result.outcome
    attribution = result.attribution
    match outcome:
        case ExitedOutcome():
            return CompletedPythonProcess(
                returncode=outcome.exit_code,
                stdout=_stream_text(result.payload_outputs.stdout, "stdout"),
                stderr=_stream_text(result.payload_outputs.stderr, "stderr"),
            )
        case SignaledOutcome():
            raise ExecutionKilledError(
                "execution died on signal "
                f"{outcome.signal_number} (resource exhaustion or "
                "interpreter crash): "
                + _kill_detail(result.payload_outputs.stderr)
            )
        case BudgetExceededOutcome() if outcome.axis is BudgetAxis.WALL_TIME:
            raise ExecutionTimeoutError(
                "execution exceeded its wall-clock budget"
            )
        case BudgetExceededOutcome() if (
            outcome.axis is BudgetAxis.PAYLOAD_OUTPUT
        ):
            raise ExecutionOutputLimitError(
                "execution exceeded "
                f"{2 * MAX_EXECUTION_STREAM_BYTES} payload output bytes"
            )
        case ProtocolFailedOutcome() if (
            attribution.owner is FailureOwner.PAYLOAD
        ):
            # A payload that dies before the protected protocol completes
            # (forced exit, uncatchable signal, stream tampering) is
            # candidate evidence, not harness breakage.
            raise ExecutionKilledError(
                "execution ended before completing its protected protocol "
                f"({outcome.failure_code}): "
                + _kill_detail(result.payload_outputs.stderr)
            )
        case _:
            raise ExecutorFailure(
                "execution produced no payload-owned outcome: "
                f"{outcome.kind} attributed to {attribution.owner}"
                + (f" ({attribution.detail})" if attribution.detail else "")
            )


def host_process_executor(
    record_root: Path,
    *,
    runtime_executable: Path,
) -> ProcessExecutor:
    """The production executor: selected Python, records under ``record_root``.

    Construction probes the selected interpreter anywhere; running jobs is
    restricted to the platforms dr-exec supports. Callers own provisioning
    the interpreter and its required packages.
    """

    return ProcessExecutor(
        runtime=IsolatedHostPythonRuntime(runtime_executable),
        run_store=DirectoryRunStore(root=record_root),
    )


def _stream_text(stream: RetainedPayloadStream, name: str) -> str:
    if stream.dropped_bytes > 0:
        raise ExecutionOutputLimitError(
            f"execution exceeded {MAX_EXECUTION_STREAM_BYTES} retained "
            f"{name} bytes"
        )
    return (stream.head + stream.tail).decode("utf-8", errors="replace")


def _kill_detail(stderr: RetainedPayloadStream) -> str:
    detail = (
        (stderr.head + stderr.tail).decode("utf-8", errors="replace").strip()
    )
    return detail if detail else "no stderr was captured"

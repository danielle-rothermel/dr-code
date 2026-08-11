"""Execution helpers for the task-difficulty verification workflow."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final
from uuid import uuid4

from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    Budgets,
    CompletedExecution,
    ContainmentProfile,
    EnvGrant,
    ExecutionJob,
    Executor,
    ExitedOutcome,
    FailureOwner,
    FiniteByteLimit,
    FiniteDurationLimit,
    FiniteOutput,
    JobId,
    OutputOverflowPolicy,
    PayloadRetentionBudget,
    ProtocolFailedOutcome,
    SignaledOutcome,
    StreamRetentionBudget,
    UntrustedPythonTarget,
)
from dr_serialize import Sha256Digest, build_identity_document

from dr_code.evaluation.batch import CandidateJobBudget
from dr_code.evaluation.execution import execute_candidate_job
from dr_code.evaluation.identity import (
    EvaluationCandidateIdentity,
    EvaluationRuntimeIdentity,
    EvaluationSampleIdentity,
    MaterializedEvaluationCandidate,
)
from dr_code.evaluation.records import (
    ExecutorExecutionFailure,
    HarnessExecutionFailure,
)
from dr_code.humaneval.job import (
    HumanEvalCandidateJobRequest,
    HumanEvalEvaluatorSuite,
)
from dr_code.humaneval.metric_operator import CodeTestSettings
from dr_code.humaneval.runner import evaluation_from_candidate_execution
from dr_code.humaneval.task import EvaluationCaseStatus, HumanEvalTask
from dr_code.metrics import MetricName, MetricQuestionCoordinate
from dr_code.metrics.coordinates import question_settings
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
)
from dr_code.trace import CodeArtifact, PreprocessingDefinitionCoordinate

_EXECUTION_REQUEST_SCHEMA: Final = "dr-code/python-execution-request"
_EXECUTION_REQUEST_SCHEMA_VERSION: Final = 1
_EXECUTION_ENVIRONMENT: Final[dict[str, str]] = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}
_PROBE_TIMEOUT_SECONDS: Final = 10.0
_NANOSECONDS_PER_SECOND: Final = 1_000_000_000
_MAX_INPUT_BYTES: Final = 2_097_152
_MAX_STREAM_BYTES: Final = 536_870_912
_WORKFLOW_METRICS_DEFINITION_ID: Final = (
    "directional-humaneval-task-difficulty"
)
_WORKFLOW_METRICS_DEFINITION_VERSION: Final = "0"
_WORKFLOW_CACHE_NAMESPACE: Final = "directional-humaneval-task-difficulty"


@dataclass(frozen=True, slots=True)
class CompletedPythonProcess:
    returncode: int
    stdout: str
    stderr: str


def candidate_job_budget(timeout_seconds: float) -> CandidateJobBudget:
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite and positive")
    timeout_nanoseconds = timeout_seconds * _NANOSECONDS_PER_SECOND
    if not math.isfinite(timeout_nanoseconds):
        raise ValueError("timeout_seconds is too large to represent")
    return CandidateJobBudget(
        wall_time_ns=math.ceil(timeout_nanoseconds),
        input_bytes=_MAX_INPUT_BYTES,
        payload_output_bytes=2 * _MAX_STREAM_BYTES,
        stdout_head_bytes=_MAX_STREAM_BYTES,
        stderr_head_bytes=_MAX_STREAM_BYTES,
    )


def _build_python_execution_job(
    *,
    driver_source: str,
    input_json: str,
    timeout_seconds: float,
) -> ExecutionJob:
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("execution timeout must be finite and positive")
    payload = json.loads(input_json)
    timeout_nanoseconds = timeout_seconds * _NANOSECONDS_PER_SECOND
    if not math.isfinite(timeout_nanoseconds):
        raise ValueError("execution timeout is too large to represent")
    return ExecutionJob(
        job_id=JobId(uuid4()),
        target=UntrustedPythonTarget(
            driver_source=driver_source,
            request=build_identity_document(
                schema=_EXECUTION_REQUEST_SCHEMA,
                schema_version=_EXECUTION_REQUEST_SCHEMA_VERSION,
                payload=payload,
            ),
            containment_profile=ContainmentProfile.PROCESS_BOUNDARY_ONLY,
        ),
        env=EnvGrant.fixed(_EXECUTION_ENVIRONMENT),
        budgets=Budgets(
            wall_time=FiniteDurationLimit(
                max_ns=math.ceil(timeout_nanoseconds)
            ),
            input_bytes=FiniteByteLimit(max_bytes=_MAX_INPUT_BYTES),
            payload_output=FiniteOutput(
                max_bytes=2 * _MAX_STREAM_BYTES,
                overflow_policy=OutputOverflowPolicy.FAIL,
                retention=PayloadRetentionBudget(
                    stdout=StreamRetentionBudget(
                        head_bytes=_MAX_STREAM_BYTES,
                        tail_bytes=0,
                    ),
                    stderr=StreamRetentionBudget(
                        head_bytes=_MAX_STREAM_BYTES,
                        tail_bytes=0,
                    ),
                ),
            ),
        ),
    )


def _stream_text(payload_stream: object, label: str) -> str:
    head = getattr(payload_stream, "head", b"")
    if not isinstance(head, (bytes, bytearray)):
        raise TypeError(f"{label} head must be bytes")
    return bytes(head).decode("utf-8", errors="replace")


def interpret_completed_execution(
    execution: CompletedExecution,
) -> CompletedPythonProcess:
    result = execution.result
    outcome = result.outcome
    attribution = result.attribution
    if isinstance(outcome, ExitedOutcome):
        return CompletedPythonProcess(
            returncode=outcome.exit_code,
            stdout=_stream_text(result.payload_outputs.stdout, "stdout"),
            stderr=_stream_text(result.payload_outputs.stderr, "stderr"),
        )
    if isinstance(outcome, SignaledOutcome):
        raise RuntimeError(
            "execution died on signal "
            f"{outcome.signal_number}: "
            + _stream_text(result.payload_outputs.stderr, "stderr")
        )
    if isinstance(outcome, BudgetExceededOutcome):
        if outcome.axis is BudgetAxis.WALL_TIME:
            raise RuntimeError("execution exceeded its wall-clock budget")
        raise RuntimeError("execution exceeded its payload output budget")
    if isinstance(outcome, ProtocolFailedOutcome):
        if attribution.owner is FailureOwner.PAYLOAD:
            raise RuntimeError(
                "execution ended before completing its protected protocol "
                f"({outcome.failure_code}): "
                + _stream_text(result.payload_outputs.stderr, "stderr")
            )
    raise RuntimeError(
        "execution produced no payload-owned outcome: "
        f"{outcome.kind} attributed to {attribution.owner}"
        + (f" ({attribution.detail})" if attribution.detail else "")
    )


def run_python_source(
    executor: Executor,
    *,
    source: str,
    input_json: str,
    timeout_seconds: float,
) -> CompletedPythonProcess:
    completed = executor.run(
        _build_python_execution_job(
            driver_source=source,
            input_json=input_json,
            timeout_seconds=timeout_seconds,
        )
    )
    return interpret_completed_execution(completed)


def runtime_identity(executor: Executor) -> EvaluationRuntimeIdentity:
    runtime = getattr(executor, "runtime", None)
    if runtime is not None:
        return EvaluationRuntimeIdentity(document=runtime.describe().id_doc)
    return EvaluationRuntimeIdentity(
        document=build_identity_document(
            schema="dr-code/task-difficulty-runtime",
            schema_version=1,
            payload={"executor": type(executor).__name__},
        )
    )


def _code_test_suite(task: HumanEvalTask) -> HumanEvalEvaluatorSuite:
    settings = CodeTestSettings()
    return HumanEvalEvaluatorSuite(
        question=MetricQuestionCoordinate(
            metric=MetricName.CODE_TEST,
            on_key="output",
            settings=question_settings(settings),
        ),
        task=task,
        settings=settings,
    )


def _materialized_candidate(
    *,
    sample_id: str,
    candidate_index: int,
    source: str,
) -> MaterializedEvaluationCandidate:
    return MaterializedEvaluationCandidate(
        identity=EvaluationCandidateIdentity(
            sample=EvaluationSampleIdentity(sample_id=sample_id),
            preprocessing=PreprocessingDefinitionCoordinate(
                definition_id=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
                version=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
                steps=(),
            ),
            candidate_ordinal=candidate_index,
        ),
        source=CodeArtifact(source=source),
        source_sha256=Sha256Digest(
            hashlib.sha256(source.encode("utf-8")).hexdigest()
        ),
    )


def evaluate_code_test(
    *,
    source: str,
    task: HumanEvalTask,
    sample_id: str,
    candidate_index: int,
    timeout_seconds: float,
    executor: Executor,
    runtime: EvaluationRuntimeIdentity,
) -> dict[str, object]:
    suite = _code_test_suite(task)
    request = HumanEvalCandidateJobRequest(
        candidate=_materialized_candidate(
            sample_id=sample_id,
            candidate_index=candidate_index,
            source=source,
        ),
        suites=(suite,),
    )
    record = execute_candidate_job(
        request,
        job_id=JobId(uuid4()),
        budget=candidate_job_budget(timeout_seconds),
        runtime=runtime,
        cache_namespace=_WORKFLOW_CACHE_NAMESPACE,
        executor=executor,
    )
    identity_values = {
        "metric_schema_version": 0,
        "metric_name": str(MetricName.CODE_TEST),
        "metric_version": "0",
        "metrics_definition_id": _WORKFLOW_METRICS_DEFINITION_ID,
        "metrics_definition_version": _WORKFLOW_METRICS_DEFINITION_VERSION,
    }
    outcome = record.outcome
    if isinstance(outcome, HarnessExecutionFailure | ExecutorExecutionFailure):
        return {
            **identity_values,
            "metric_status": "operator_failure",
            "candidate_passed": None,
            "failure_type": outcome.failure_type,
            "failure_message": outcome.message,
        }
    try:
        evaluation = evaluation_from_candidate_execution(
            task=task,
            candidate_source=source,
            question=suite.question,
            outcome=outcome,
        )
    except Exception as exc:
        return {
            **identity_values,
            "metric_status": "operator_failure",
            "candidate_passed": None,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }

    counts = evaluation.status_counts
    passed_count = counts.get(EvaluationCaseStatus.PASSED.value, 0)
    failed_count = counts.get(EvaluationCaseStatus.FAILED.value, 0)
    error_count = counts.get(EvaluationCaseStatus.ERROR.value, 0)
    timeout_count = counts.get(EvaluationCaseStatus.TIMEOUT.value, 0)
    return {
        **identity_values,
        "metric_status": "measured",
        "total_cases": evaluation.total_cases,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "error_count": error_count,
        "timeout_count": timeout_count,
        "coverage_complete": evaluation.coverage_complete,
        "function_count": len(evaluation.function_names),
        "best_function_name": evaluation.best_function_name,
        "candidate_passed": (
            evaluation.coverage_complete
            and passed_count == evaluation.total_cases
        ),
        "failure_type": None,
        "failure_message": None,
    }


def probe_runtime_packages(
    executor: Executor,
    *,
    probe_source: str,
) -> dict[str, object]:
    completed = run_python_source(
        executor,
        source=probe_source,
        input_json="{}",
        timeout_seconds=_PROBE_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "evaluation runtime dependency probe failed: "
            + completed.stderr.strip()
        )
    try:
        package_identity = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "evaluation runtime dependency probe returned invalid JSON"
        ) from exc
    if not isinstance(package_identity, dict):
        raise RuntimeError(
            "evaluation runtime dependency probe returned non-object JSON"
        )
    return package_identity

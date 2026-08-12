from __future__ import annotations

import asyncio
from typing import Final, cast

from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    Budgets,
    CancelledOutcome,
    CompletedExecution,
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
    build_untrusted_importable_json_job,
    parse_importable_json_result,
)
from dr_serialize import (
    Jsonable,
    Sha256Digest,
    build_identity_document,
    identity_document_hash,
)
from dr_store import derive_cache_key
from pydantic import ValidationError

from dr_code.evaluation.batch import CandidateJobBudget, RunGrade
from dr_code.evaluation.identity import EvaluationRuntimeIdentity
from dr_code.evaluation.records import (
    CandidateExecutionOutcome,
    CandidateExecutionRecord,
    CandidateJobCompleted,
    CandidateJobTerminated,
    CandidateTerminationReason,
    ExecutedCandidateProvenance,
    ExecutorExecutionFailure,
    HarnessExecutionFailure,
    ReusedCandidateProvenance,
)
from dr_code.evaluation.references import EvidenceReference
from dr_code.humaneval.job import (
    HUMANEVAL_CANDIDATE_ENTRY_POINT,
    HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION,
    HumanEvalCandidateJobRequest,
    HumanEvalCandidateJobResult,
)

_CANDIDATE_REQUEST_SCHEMA: Final = "dr-code/humaneval-candidate-job-v1"
_EXECUTION_ENVIRONMENT: Final[dict[str, str]] = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}


def build_candidate_execution_job(
    job_id: JobId,
    request: HumanEvalCandidateJobRequest,
    budget: CandidateJobBudget,
    /,
) -> ExecutionJob:
    """Build the one bounded process job for a materialized candidate."""

    return build_untrusted_importable_json_job(
        job_id,
        HUMANEVAL_CANDIDATE_ENTRY_POINT,
        _request_payload(request),
        env=EnvGrant.fixed(_EXECUTION_ENVIRONMENT),
        budgets=Budgets(
            wall_time=FiniteDurationLimit(max_ns=budget.wall_time_ns),
            input_bytes=FiniteByteLimit(max_bytes=budget.input_bytes),
            payload_output=FiniteOutput(
                max_bytes=budget.payload_output_bytes,
                overflow_policy=OutputOverflowPolicy.FAIL,
                retention=PayloadRetentionBudget(
                    stdout=StreamRetentionBudget(
                        head_bytes=budget.stdout_head_bytes,
                        tail_bytes=0,
                    ),
                    stderr=StreamRetentionBudget(
                        head_bytes=budget.stderr_head_bytes,
                        tail_bytes=0,
                    ),
                ),
            ),
        ),
    )


def candidate_execution_request_identity(
    request: HumanEvalCandidateJobRequest,
    /,
) -> Sha256Digest:
    return identity_document_hash(
        build_identity_document(
            schema=_CANDIDATE_REQUEST_SCHEMA,
            schema_version=HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION,
            payload=_request_payload(request),
        )
    )


def candidate_execution_cache_key(
    request: HumanEvalCandidateJobRequest,
    budget: CandidateJobBudget,
    cache_namespace: str,
    /,
    *,
    run_grade: RunGrade,
) -> str:
    # Persisted key payload literals are a wire contract. Never derive them
    # from field names, and never build this payload by iterating an enum.
    return derive_cache_key(
        cache_namespace,
        {
            "request_identity": str(
                candidate_execution_request_identity(request)
            ),
            "run_grade": run_grade.value,
            "wall_time_ns": budget.wall_time_ns,
            "input_bytes": budget.input_bytes,
            "payload_output_bytes": budget.payload_output_bytes,
            "stdout_head_bytes": budget.stdout_head_bytes,
            "stderr_head_bytes": budget.stderr_head_bytes,
        },
    )


def execute_candidate_job(
    request: HumanEvalCandidateJobRequest,
    /,
    *,
    job_id: JobId,
    budget: CandidateJobBudget,
    runtime: EvaluationRuntimeIdentity,
    cache_namespace: str,
    run_grade: RunGrade,
    executor: Executor,
) -> CandidateExecutionRecord:
    completed = executor.run_blocking(
        build_candidate_execution_job(job_id, request, budget)
    )
    return executed_candidate_record(
        request,
        completed,
        budget=budget,
        runtime=runtime,
        cache_namespace=cache_namespace,
        run_grade=run_grade,
    )


def executed_candidate_record(
    request: HumanEvalCandidateJobRequest,
    completed: CompletedExecution,
    /,
    *,
    budget: CandidateJobBudget,
    runtime: EvaluationRuntimeIdentity,
    cache_namespace: str,
    run_grade: RunGrade,
) -> CandidateExecutionRecord:
    return CandidateExecutionRecord(
        candidate=request.candidate.identity,
        request_identity=candidate_execution_request_identity(request),
        runtime=runtime,
        cache_namespace=cache_namespace,
        cache_key=candidate_execution_cache_key(
            request,
            budget,
            cache_namespace,
            run_grade=run_grade,
        ),
        provenance=ExecutedCandidateProvenance(
            record_receipt=completed.record_receipt
        ),
        outcome=interpret_candidate_execution(request, completed),
    )


def reused_candidate_record(
    request: HumanEvalCandidateJobRequest,
    source_record: EvidenceReference,
    outcome: CandidateExecutionOutcome,
    /,
    *,
    budget: CandidateJobBudget,
    runtime: EvaluationRuntimeIdentity,
    cache_namespace: str,
    run_grade: RunGrade,
) -> CandidateExecutionRecord:
    return CandidateExecutionRecord(
        candidate=request.candidate.identity,
        request_identity=candidate_execution_request_identity(request),
        runtime=runtime,
        cache_namespace=cache_namespace,
        cache_key=candidate_execution_cache_key(
            request,
            budget,
            cache_namespace,
            run_grade=run_grade,
        ),
        provenance=ReusedCandidateProvenance(source_record=source_record),
        outcome=outcome,
    )


def interpret_candidate_execution(
    request: HumanEvalCandidateJobRequest,
    completed: CompletedExecution,
    /,
) -> CandidateExecutionOutcome:
    result = completed.result
    outcome = result.outcome
    attribution = result.attribution
    measurements = result.measurements
    if isinstance(outcome, CancelledOutcome):
        raise asyncio.CancelledError
    if isinstance(outcome, ExitedOutcome):
        if outcome.exit_code != 0:
            if attribution.owner is FailureOwner.PAYLOAD:
                return CandidateJobTerminated(
                    reason=CandidateTerminationReason.NONZERO_EXIT,
                    execution_outcome=outcome,
                    attribution=attribution,
                    measurements=measurements,
                )
            return ExecutorExecutionFailure(
                failure_type="ExitedOutcome",
                message=(
                    f"nonzero exit {outcome.exit_code} was attributed to "
                    f"{attribution.owner}"
                ),
                execution_outcome=outcome,
                attribution=attribution,
                measurements=measurements,
            )
        try:
            parsed = parse_importable_json_result(completed)
            candidate_result = HumanEvalCandidateJobResult.model_validate(
                parsed,
            )
            _validate_result_matches_request(request, candidate_result)
        except (ValueError, TypeError, ValidationError) as error:
            return HarnessExecutionFailure(
                failure_type=type(error).__name__,
                message=str(error),
                execution_outcome=outcome,
                attribution=attribution,
                measurements=measurements,
            )
        return CandidateJobCompleted(
            result=candidate_result,
            execution_outcome=outcome,
            attribution=attribution,
            measurements=measurements,
        )

    if attribution.owner is FailureOwner.PAYLOAD:
        reason: CandidateTerminationReason | None = None
        if isinstance(outcome, SignaledOutcome):
            reason = CandidateTerminationReason.SIGNALED
        elif isinstance(outcome, BudgetExceededOutcome):
            if outcome.axis is BudgetAxis.PAYLOAD_OUTPUT:
                reason = CandidateTerminationReason.PAYLOAD_OUTPUT
        elif isinstance(outcome, ProtocolFailedOutcome):
            reason = CandidateTerminationReason.PAYLOAD_PROTOCOL
        if reason is not None:
            return CandidateJobTerminated(
                reason=reason,
                execution_outcome=outcome,
                attribution=attribution,
                measurements=measurements,
            )

    return ExecutorExecutionFailure(
        failure_type=type(outcome).__name__,
        message=(
            f"execution outcome {outcome.kind} was attributed to "
            f"{attribution.owner}"
            + (f": {attribution.detail}" if attribution.detail else "")
        ),
        execution_outcome=outcome,
        attribution=attribution,
        measurements=measurements,
    )


def _validate_result_matches_request(
    request: HumanEvalCandidateJobRequest,
    result: HumanEvalCandidateJobResult,
) -> None:
    if result.candidate != request.candidate.identity:
        raise ValueError(
            "candidate job result identity does not match request"
        )
    if result.namespace.kind == "candidate_failure":
        if result.suites:
            raise ValueError(
                "candidate namespace failure returned suite results"
            )
        return
    questions = tuple(suite.question for suite in request.suites)
    result_questions = tuple(suite.question for suite in result.suites)
    if result_questions != questions:
        raise ValueError(
            "candidate job suite results do not match request order"
        )


def _request_payload(request: HumanEvalCandidateJobRequest) -> Jsonable:
    return cast(
        Jsonable,
        request.model_dump(mode="json", exclude_computed_fields=True),
    )


__all__ = [
    "build_candidate_execution_job",
    "candidate_execution_cache_key",
    "candidate_execution_request_identity",
    "execute_candidate_job",
    "executed_candidate_record",
    "interpret_candidate_execution",
    "reused_candidate_record",
]

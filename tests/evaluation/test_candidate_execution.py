from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from dr_exec import (
    BudgetAxis,
    BudgetExceededOutcome,
    CancelledOutcome,
    CompleteRecordReceipt,
    CompletedExecution,
    DegradedRecordReceipt,
    EnvGrantKind,
    FailureOwner,
    FiniteByteLimit,
    FiniteDurationLimit,
    FiniteOutput,
    ExitedOutcome,
    JobId,
    ProtocolFailedOutcome,
    ProtocolFailureCode,
    RecordState,
    RecordingFailure,
    RunRecordReference,
    SignaledOutcome,
    SpawnAbsentOutcome,
)
from dr_serialize import build_identity_document

from _executor_stubs import completed_execution
from dr_code.evaluation import (
    BundleRecordReference,
    CandidateJobBudget,
    CandidateJobTerminated,
    CandidateTerminationReason,
    EvaluationRuntimeIdentity,
    ExecutedCandidateProvenance,
    ExecutorExecutionFailure,
    ReusedCandidateProvenance,
)
from dr_code.evaluation.execution import (
    build_candidate_execution_job,
    executed_candidate_record,
    interpret_candidate_execution,
    reused_candidate_record,
)
from _candidate_job_builders import candidate_job_budget, candidate_job_request


def _runtime() -> EvaluationRuntimeIdentity:
    return EvaluationRuntimeIdentity(
        document=build_identity_document(
            schema="tests/runtime",
            schema_version=1,
            payload={"name": "candidate-execution"},
        )
    )


def _job():
    return build_candidate_execution_job(
        JobId(UUID("00000000-0000-0000-0000-000000000002")),
        candidate_job_request("def observed_load_count(_x):\n    return 1\n"),
        candidate_job_budget(),
    )


def _source_record() -> BundleRecordReference:
    return BundleRecordReference(
        artifact_name="sample-records-00000000.jsonl",
        record_index=0,
        record_sha256="a" * 64,
        schema="dr-code/sample-evaluation-record-v1",
        schema_version=1,
    )


def test_candidate_job_budget_and_declaration_wire_shape_are_fixed() -> None:
    budget = candidate_job_budget()
    assert budget.model_dump(mode="json") == {
        "wall_time_ns": 5_000_000_000,
        "input_bytes": 2_097_152,
        "payload_output_bytes": 2_097_152,
        "stdout_head_bytes": 1_048_576,
        "stderr_head_bytes": 1_048_576,
    }
    job = _job()
    assert job.env.kind is EnvGrantKind.FIXED
    assert {item.name: item.value for item in job.env.variables} == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    assert job.budgets.wall_time == FiniteDurationLimit(
        max_ns=budget.wall_time_ns
    )
    assert job.budgets.input_bytes == FiniteByteLimit(
        max_bytes=budget.input_bytes
    )
    assert isinstance(job.budgets.payload_output, FiniteOutput)
    assert job.budgets.payload_output.max_bytes == budget.payload_output_bytes
    assert job.target.request.payload.keys() == {
        "schema_version",
        "candidate",
        "suites",
    }


def test_candidate_job_budget_rejects_mismatched_retention() -> None:
    with pytest.raises(ValueError, match="must equal payload_output_bytes"):
        CandidateJobBudget(
            wall_time_ns=1,
            input_bytes=1,
            payload_output_bytes=3,
            stdout_head_bytes=1,
            stderr_head_bytes=1,
        )


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (
            ExitedOutcome(exit_code=7),
            CandidateTerminationReason.NONZERO_EXIT,
        ),
        (
            SignaledOutcome(signal_number=9),
            CandidateTerminationReason.SIGNALED,
        ),
        (
            BudgetExceededOutcome(axis=BudgetAxis.WALL_TIME),
            CandidateTerminationReason.WALL_TIME,
        ),
        (
            BudgetExceededOutcome(axis=BudgetAxis.PAYLOAD_OUTPUT),
            CandidateTerminationReason.PAYLOAD_OUTPUT,
        ),
        (
            ProtocolFailedOutcome(
                failure_code=ProtocolFailureCode.INCOMPLETE_STREAM,
                failure_detail="incomplete",
                accepted_output_count=0,
            ),
            CandidateTerminationReason.PAYLOAD_PROTOCOL,
        ),
    ],
)
def test_payload_outcomes_have_closed_candidate_termination_mapping(
    outcome: object,
    reason: CandidateTerminationReason,
) -> None:
    completed = completed_execution(_job(), outcome=outcome)  # type: ignore[arg-type]
    interpreted = interpret_candidate_execution(
        candidate_job_request("def observed_load_count(_x):\n    return 1\n"),
        completed,
    )
    assert isinstance(interpreted, CandidateJobTerminated)
    assert interpreted.reason is reason


def test_non_payload_outcome_fails_closed_as_executor_failure() -> None:
    completed = completed_execution(
        _job(),
        outcome=SpawnAbsentOutcome(executable="missing-python"),
    )
    interpreted = interpret_candidate_execution(
        candidate_job_request("def observed_load_count(_x):\n    return 1\n"),
        completed,
    )
    assert isinstance(interpreted, ExecutorExecutionFailure)
    assert interpreted.attribution is not None
    assert interpreted.attribution.owner is FailureOwner.EXECUTOR


def test_cancellation_raises_and_creates_no_outcome() -> None:
    completed = completed_execution(_job(), outcome=CancelledOutcome())
    with pytest.raises(asyncio.CancelledError):
        interpret_candidate_execution(
            candidate_job_request(
                "def observed_load_count(_x):\n    return 1\n"
            ),
            completed,
        )


@pytest.mark.parametrize("receipt_kind", ["complete", "degraded"])
def test_real_receipt_reference_is_preserved(receipt_kind: str) -> None:
    request = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n"
    )
    fake = completed_execution(
        _job(), outcome=SignaledOutcome(signal_number=9)
    )
    reference = RunRecordReference(record_id=uuid4())
    if receipt_kind == "complete":
        receipt = CompleteRecordReceipt(
            execution_id=fake.result.execution_id,
            reference=reference,
        )
    else:
        receipt = DegradedRecordReceipt(
            execution_id=fake.result.execution_id,
            reference=reference,
            latest_state=RecordState.RUNNING,
            failures=(
                RecordingFailure(
                    operation="finalize",
                    errno=None,
                    detail="test failure",
                ),
            ),
        )
    completed = CompletedExecution(result=fake.result, record_receipt=receipt)

    record = executed_candidate_record(
        request,
        completed,
        budget=candidate_job_budget(),
        runtime=_runtime(),
        cache_namespace="tests/execution",
    )

    assert isinstance(record.provenance, ExecutedCandidateProvenance)
    assert record.provenance.record_receipt == receipt
    assert record.provenance.record_receipt.reference == reference


def test_fake_receipt_has_no_run_record_reference() -> None:
    request = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n"
    )
    completed = completed_execution(
        _job(), outcome=SignaledOutcome(signal_number=9)
    )
    record = executed_candidate_record(
        request,
        completed,
        budget=candidate_job_budget(),
        runtime=_runtime(),
        cache_namespace="tests/execution",
    )
    assert isinstance(record.provenance, ExecutedCandidateProvenance)
    assert not hasattr(record.provenance.record_receipt, "reference")


def test_reused_record_points_to_source_and_claims_no_receipt() -> None:
    request = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n"
    )
    executed = executed_candidate_record(
        request,
        completed_execution(_job(), outcome=SignaledOutcome(signal_number=9)),
        budget=candidate_job_budget(),
        runtime=_runtime(),
        cache_namespace="tests/execution",
    )
    source = _source_record()
    reused = reused_candidate_record(
        request,
        source,
        executed.outcome,
        budget=candidate_job_budget(),
        runtime=_runtime(),
        cache_namespace="tests/execution",
    )

    assert isinstance(reused.provenance, ReusedCandidateProvenance)
    assert reused.provenance.source_record == source
    assert not hasattr(reused.provenance, "record_receipt")

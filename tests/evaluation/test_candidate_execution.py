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
from dr_store import derive_cache_key

from _executor_stubs import completed_execution
from pydantic import ValidationError

from dr_code.evaluation import (
    BundleRecordReference,
    CANDIDATE_PAYLOAD_OUTPUT_BYTES,
    CANDIDATE_STREAM_HEAD_BYTES,
    CandidateJobBudget,
    CandidateJobCompleted,
    CandidateJobTerminated,
    CandidateTerminationReason,
    EvalRuntimeId,
    ExecutedCandidateProvenance,
    ExecutorExecutionFailure,
    FailureClass,
    HarnessExecutionFailure,
    ReusedCandidateProvenance,
    RunGrade,
    failure_class_of,
)
from dr_code.evaluation.execution import (
    build_candidate_execution_job,
    candidate_execution_cache_key,
    candidate_execution_request_identity,
    executed_candidate_record,
    interpret_candidate_execution,
    reused_candidate_record,
)
from _candidate_job_builders import (
    candidate_job_budget,
    candidate_job_request,
    candidate_job_suite,
)
from dr_code.humaneval.job import (
    DEFAULT_FIELD_LIMIT,
    evaluate_humaneval_candidate_job,
)


def _runtime() -> EvalRuntimeId:
    return EvalRuntimeId(
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
        "field_limit",
    }
    assert job.target.request.payload["field_limit"] == DEFAULT_FIELD_LIMIT


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
                    detail="OSError",
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
        run_grade=RunGrade.TRIAL,
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
        run_grade=RunGrade.TRIAL,
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
        run_grade=RunGrade.TRIAL,
    )
    source = _source_record()
    reused = reused_candidate_record(
        request,
        source,
        executed.outcome,
        budget=candidate_job_budget(),
        runtime=_runtime(),
        cache_namespace="tests/execution",
        run_grade=RunGrade.TRIAL,
    )

    assert isinstance(reused.provenance, ReusedCandidateProvenance)
    assert reused.provenance.source_record == source
    assert not hasattr(reused.provenance, "record_receipt")


def _completed_candidate_execution() -> CompletedExecution:
    request = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n",
        candidate_job_suite("output"),
    )
    payload = evaluate_humaneval_candidate_job(
        request.model_dump(mode="json", exclude_computed_fields=True)
    )
    return completed_execution(
        build_candidate_execution_job(
            JobId(UUID("00000000-0000-0000-0000-000000000003")),
            request,
            candidate_job_budget(),
        ),
        outcome=ExitedOutcome(exit_code=0),
        protocol_outputs=(
            build_identity_document(
                schema="dr_exec.importable_json",
                schema_version=1,
                payload=payload,
            ),
        ),
    )


def test_failure_class_literals_are_pinned() -> None:
    assert FailureClass.HARNESS.value == "harness"
    assert FailureClass.CANDIDATE.value == "candidate"
    assert FailureClass.INFRASTRUCTURE.value == "infrastructure"
    assert len(FailureClass) == 3


def test_completed_candidate_execution_attributes_no_failure() -> None:
    request = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n",
        candidate_job_suite("output"),
    )
    interpreted = interpret_candidate_execution(
        request, _completed_candidate_execution()
    )
    assert isinstance(interpreted, CandidateJobCompleted)
    assert failure_class_of(interpreted) is None


def test_terminated_candidate_execution_attributes_the_candidate() -> None:
    interpreted = interpret_candidate_execution(
        candidate_job_request("def observed_load_count(_x):\n    return 1\n"),
        completed_execution(_job(), outcome=SignaledOutcome(signal_number=9)),
    )
    assert isinstance(interpreted, CandidateJobTerminated)
    assert failure_class_of(interpreted) is FailureClass.CANDIDATE


def test_wall_time_exhaustion_is_infrastructure_owned() -> None:
    interpreted = interpret_candidate_execution(
        candidate_job_request("def observed_load_count(_x):\n    return 1\n"),
        completed_execution(
            _job(),
            outcome=BudgetExceededOutcome(axis=BudgetAxis.WALL_TIME),
        ),
    )
    assert isinstance(interpreted, ExecutorExecutionFailure)
    assert interpreted.attribution is not None
    assert interpreted.attribution.owner is FailureOwner.EXECUTOR
    assert failure_class_of(interpreted) is FailureClass.INFRASTRUCTURE


def test_harness_execution_failure_attributes_the_harness() -> None:
    request = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n",
        candidate_job_suite("output"),
    )
    unmatched = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n",
        candidate_job_suite("other-output"),
    )
    interpreted = interpret_candidate_execution(
        unmatched, _completed_candidate_execution()
    )
    assert request.suites != unmatched.suites
    assert isinstance(interpreted, HarnessExecutionFailure)
    assert failure_class_of(interpreted) is FailureClass.HARNESS


def test_executor_execution_failure_attributes_infrastructure() -> None:
    interpreted = interpret_candidate_execution(
        candidate_job_request("def observed_load_count(_x):\n    return 1\n"),
        completed_execution(
            _job(), outcome=SpawnAbsentOutcome(executable="missing-python")
        ),
    )
    assert isinstance(interpreted, ExecutorExecutionFailure)
    assert failure_class_of(interpreted) is FailureClass.INFRASTRUCTURE


# Literal payload keys pin the persisted cache key; deriving them from field
# names would hide silent drift of stored identity.
_GOLDEN_CACHE_KEY_PAYLOAD_KEYS = [
    "request_identity",
    "run_grade",
    "wall_time_ns",
    "input_bytes",
    "payload_output_bytes",
    "stdout_head_bytes",
    "stderr_head_bytes",
]


def test_run_grade_literals_are_pinned() -> None:
    assert RunGrade.TRIAL.value == "trial"
    assert RunGrade.SELECTION.value == "selection"
    assert len(RunGrade) == 2


def test_cache_key_hashes_the_golden_payload_including_grade() -> None:
    request = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n"
    )
    budget = candidate_job_budget()
    payload = {
        "request_identity": str(candidate_execution_request_identity(request)),
        "run_grade": "selection",
        "wall_time_ns": budget.wall_time_ns,
        "input_bytes": budget.input_bytes,
        "payload_output_bytes": budget.payload_output_bytes,
        "stdout_head_bytes": budget.stdout_head_bytes,
        "stderr_head_bytes": budget.stderr_head_bytes,
    }
    assert list(payload) == _GOLDEN_CACHE_KEY_PAYLOAD_KEYS
    assert candidate_execution_cache_key(
        request,
        budget,
        "tests/execution",
        run_grade=RunGrade.SELECTION,
    ) == derive_cache_key("tests/execution", payload)


def test_trial_and_selection_grades_never_share_a_cache_key() -> None:
    request = candidate_job_request(
        "def observed_load_count(_x):\n    return 1\n"
    )
    budget = candidate_job_budget()
    trial = candidate_execution_cache_key(
        request, budget, "tests/execution", run_grade=RunGrade.TRIAL
    )
    selection = candidate_execution_cache_key(
        request, budget, "tests/execution", run_grade=RunGrade.SELECTION
    )
    assert trial != selection


def test_candidate_retention_defaults_are_the_library_values() -> None:
    budget = CandidateJobBudget(
        wall_time_ns=5_000_000_000,
        input_bytes=2_097_152,
    )

    assert budget.stdout_head_bytes == 536_870_912
    assert budget.stderr_head_bytes == 536_870_912
    assert budget.payload_output_bytes == 1_073_741_824
    assert budget.stdout_head_bytes == CANDIDATE_STREAM_HEAD_BYTES
    assert budget.payload_output_bytes == CANDIDATE_PAYLOAD_OUTPUT_BYTES


def test_wall_time_and_input_bytes_carry_no_default() -> None:
    with pytest.raises(ValidationError):
        CandidateJobBudget(input_bytes=2_097_152)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        CandidateJobBudget(wall_time_ns=5_000_000_000)  # type: ignore[call-arg]

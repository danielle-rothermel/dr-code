from __future__ import annotations

import hashlib
from uuid import UUID

import pytest
from dr_serialize import IdentityDocument, Sha256Digest
from pydantic import ValidationError

from _builders import (
    candidate,
    evaluation_slot,
    measured,
    policy,
    procedure,
    record_identity,
    sampling_plan,
    sample_identity,
    task_set,
)
from dr_code.evaluation import (
    AttemptCompleteness,
    AttemptLimitExhaustion,
    AttemptLimitKind,
    AttemptValidity,
    BundleRecordReference,
    CandidateExecutionRecord,
    EvalAttemptIdentity,
    EvalAttemptRecord,
    EvalMemberRecord,
    EvalPlan,
    EvalRuntimeIdentity,
    EvalSampleMetadata,
    EvaluatedSampleRecord,
    ExecutedCandidateProvenance,
    ExecutorExecutionFailure,
    GeneratedSampleProvenance,
    HarnessExecutionFailure,
    MaterializedEvalCandidate,
    NoCandidatesSampleRecord,
    PreprocessingAbsentSampleRecord,
    ReplayMode,
    ReplaySource,
    ReusedCandidateProvenance,
    SAMPLE_EVAL_RECORD_ADAPTER,
)
from dr_code.trace import Absent, CodeArtifact, SerializedTrace, TextArtifact
from dr_exec import AttemptId, ExecutionId, FakeRecordReceipt, JobId

_DIGEST = Sha256Digest("a" * 64)
_CANDIDATE_SOURCE = "def f(): return 1"
_CANDIDATE_SOURCE_DIGEST = Sha256Digest(
    hashlib.sha256(_CANDIDATE_SOURCE.encode("utf-8")).hexdigest()
)


def reference(index: int = 0) -> BundleRecordReference:
    return BundleRecordReference(
        artifact_name="sample-records-00000000.jsonl",
        record_index=index,
        record_sha256=_DIGEST,
        schema="dr-code/sample-evaluation-record-v1",
        schema_version=1,
    )


def runtime() -> EvalRuntimeIdentity:
    return EvalRuntimeIdentity(
        document=IdentityDocument(
            schema="dr-code/runtime",
            schema_version=1,
            payload={"runtime": "test"},
        )
    )


def metadata(**overrides: object) -> EvalSampleMetadata:
    return EvalSampleMetadata(
        **{
            "identity": sample_identity(),
            "task_id": "t0",
            "provenance": GeneratedSampleProvenance(
                source_identity={"namespace": "generator", "value": "run-1"},
                source_reference=reference(),
                generation_id="generation-1",
            ),
            **overrides,
        }
    )


def trace() -> SerializedTrace:
    return SerializedTrace(
        schema_version=3,
        producer=record_identity().producer,
        values={
            "input": TextArtifact(text="raw input"),
            "output": CodeArtifact(source="def f(): return 1"),
        },
    )


def materialized(**overrides: object) -> MaterializedEvalCandidate:
    return MaterializedEvalCandidate(
        **{
            "identity": candidate(),
            "source": CodeArtifact(source=_CANDIDATE_SOURCE),
            "source_sha256": _CANDIDATE_SOURCE_DIGEST,
            **overrides,
        }
    )


def execution(**overrides: object) -> CandidateExecutionRecord:
    return CandidateExecutionRecord(
        **{
            "candidate": candidate(),
            "request_identity": _DIGEST,
            "runtime": runtime(),
            "cache_namespace": "evaluation-v1",
            "cache_key": "request-1",
            "provenance": ReusedCandidateProvenance(source_record=reference()),
            "outcome": HarnessExecutionFailure(
                failure_type="HarnessError",
                message="bad evaluator",
                execution_outcome=None,
                attribution=None,
                measurements=None,
            ),
            **overrides,
        }
    )


def evaluated(**overrides: object) -> EvaluatedSampleRecord:
    return EvaluatedSampleRecord(
        **{
            "slot": evaluation_slot(),
            "sample": metadata(),
            "trace": trace(),
            "candidates": (materialized(),),
            "executions": (execution(),),
            "metrics": (measured(),),
            **overrides,
        }
    )


def evaluation_plan() -> EvalPlan:
    return EvalPlan(
        plan_id="plan",
        version="1",
        task_set=task_set(),
        sampling_plan=sampling_plan(),
        procedure=procedure(),
        aggregation=policy(),
    )


def test_sample_terminal_variants_round_trip_by_status() -> None:
    absence = Absent(
        failed_step="normalize",
        failure_code="blank_input",
        cause="blank",
    )
    records = (
        PreprocessingAbsentSampleRecord(
            slot=evaluation_slot(),
            sample=metadata(),
            trace=trace(),
            absence=absence,
        ),
        NoCandidatesSampleRecord(
            slot=evaluation_slot(), sample=metadata(), trace=trace()
        ),
        evaluated(),
    )
    assert [record.status for record in records] == [
        "preprocessing_absent",
        "no_candidates",
        "evaluated",
    ]
    for record in records:
        assert (
            SAMPLE_EVAL_RECORD_ADAPTER.validate_json(record.model_dump_json())
            == record
        )


def test_persisted_sample_records_do_not_repeat_raw_input() -> None:
    record = evaluated()
    assert "raw_input" not in record.model_dump_json()
    assert record.trace.values["input"] == TextArtifact(text="raw input")
    assert set(EvaluatedSampleRecord.model_fields) == {
        "schema_version",
        "status",
        "slot",
        "sample",
        "trace",
        "candidates",
        "executions",
        "metrics",
    }


@pytest.mark.parametrize(
    ("record_type", "fields"),
    (
        (
            PreprocessingAbsentSampleRecord,
            {
                "trace": trace(),
                "absence": Absent(
                    failed_step="normalize",
                    failure_code="blank_input",
                    cause="blank",
                ),
            },
        ),
        (NoCandidatesSampleRecord, {"trace": trace()}),
        (
            EvaluatedSampleRecord,
            {
                "trace": trace(),
                "candidates": (materialized(),),
                "executions": (execution(),),
                "metrics": (measured(),),
            },
        ),
    ),
)
def test_sample_records_require_slot_and_sample_task_ids_to_match(
    record_type: type,
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="sample task_id must match"):
        record_type(
            slot=evaluation_slot(),
            sample=metadata(task_id="other-task"),
            **fields,
        )


def test_evaluated_record_requires_candidate_sample_identity_to_match() -> (
    None
):
    other_candidate = materialized(
        identity=candidate(
            sample=sample_identity(sample_id="different-sample")
        )
    )
    with pytest.raises(ValidationError, match="sample identities must match"):
        evaluated(candidates=(other_candidate,))


def test_evaluated_record_requires_execution_order_to_match_candidates() -> (
    None
):
    other = execution(candidate=candidate(candidate_ordinal=1))
    with pytest.raises(ValidationError, match="match materialized candidates"):
        evaluated(executions=(other,))


def test_execution_provenance_variants_have_exact_discriminators() -> None:
    assert (
        ReusedCandidateProvenance(source_record=reference()).kind == "reused"
    )
    executed = ExecutedCandidateProvenance(
        record_receipt=FakeRecordReceipt(
            execution_id=ExecutionId(
                job_id=JobId(UUID(int=1)), attempt_id=AttemptId(UUID(int=2))
            )
        )
    )
    assert executed.kind == "executed"


def test_execution_failure_variants_have_exact_discriminators() -> None:
    common = {
        "failure_type": "Failure",
        "message": "message",
        "execution_outcome": None,
        "attribution": None,
        "measurements": None,
    }
    assert HarnessExecutionFailure(**common).kind == "harness_failure"
    assert ExecutorExecutionFailure(**common).kind == "executor_failure"


def test_attempt_limit_kind_is_closed() -> None:
    assert {kind.value for kind in AttemptLimitKind} == {
        "slots",
        "materialized_candidates",
        "admitted_jobs",
        "retained_evidence_bytes",
        "projection_rows",
    }


@pytest.mark.parametrize("observed", (1, 2))
def test_attempt_limit_exhaustion_requires_observed_to_exceed_configured(
    observed: int,
) -> None:
    with pytest.raises(ValidationError, match="must exceed configured"):
        AttemptLimitExhaustion(
            limit=AttemptLimitKind.SLOTS,
            configured=2,
            observed=observed,
        )


def attempt(**overrides: object) -> EvalAttemptRecord:
    return EvalAttemptRecord(
        **{
            "identity": EvalAttemptIdentity(attempt_id=UUID(int=2)),
            "plan": evaluation_plan(),
            "runtime": runtime(),
            "cache_namespace": "evaluation-v1",
            "members": (
                EvalMemberRecord(
                    slot=evaluation_slot(),
                    sample=sample_identity(),
                    record=reference(),
                ),
            ),
            "completeness": AttemptCompleteness.COMPLETE,
            "validity": AttemptValidity.VALID,
            "limit_exhaustion": None,
            "replay": None,
            **overrides,
        }
    )


def test_complete_attempt_requires_every_member_record() -> None:
    member = EvalMemberRecord(
        slot=evaluation_slot(), sample=sample_identity(), record=None
    )
    with pytest.raises(ValidationError, match="complete.*missing record"):
        attempt(members=(member,))


def test_partial_attempt_is_invalid_and_names_missing_members() -> None:
    member = EvalMemberRecord(
        slot=evaluation_slot(), sample=sample_identity(), record=None
    )
    built = attempt(
        members=(member,),
        completeness=AttemptCompleteness.PARTIAL,
        validity=AttemptValidity.INVALID,
        limit_exhaustion=AttemptLimitExhaustion(
            limit=AttemptLimitKind.ADMITTED_JOBS,
            configured=1,
            observed=2,
        ),
    )
    assert built.members[0].record is None


def test_limit_exhaustion_requires_a_partial_invalid_attempt() -> None:
    exhaustion = AttemptLimitExhaustion(
        limit=AttemptLimitKind.ADMITTED_JOBS,
        configured=1,
        observed=2,
    )
    with pytest.raises(ValidationError, match="partial invalid"):
        attempt(
            completeness=AttemptCompleteness.COMPLETE,
            validity=AttemptValidity.INVALID,
            limit_exhaustion=exhaustion,
        )


def test_attempt_rejects_duplicate_slot_or_sample_identity() -> None:
    duplicate = EvalMemberRecord(
        slot=evaluation_slot(), sample=sample_identity(), record=reference(1)
    )
    with pytest.raises(ValidationError, match="slots must be unique"):
        attempt(members=(attempt().members[0], duplicate))


def test_replay_source_nests_attempt_identity_and_closed_mode() -> None:
    replay = ReplaySource(
        attempt=EvalAttemptIdentity(attempt_id=UUID(int=3)),
        mode=ReplayMode.MATERIALIZED_CANDIDATES,
    )
    assert replay.mode.value == "materialized_candidates"

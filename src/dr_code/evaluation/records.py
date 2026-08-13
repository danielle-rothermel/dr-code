from __future__ import annotations

from enum import StrEnum, UNIQUE, verify
from typing import Annotated, Final, Literal, Self, TypeAlias

from dr_exec import (
    ExecutionAttribution,
    ExecutionMeasurements,
    ExecutionOutcome,
    RecordReceipt,
)
from dr_serialize import Sha256Digest
from pydantic import Field, PositiveInt, TypeAdapter, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.id import (
    EvalAttemptId,
    EvalCandidateId,
    EvalRuntimeId,
    EvalSampleId,
    EvalSampleMetadata,
    EvalSlotId,
    MaterializedEvalCandidate,
)
from dr_code.evaluation.plan import EvalPlan
from dr_code.evaluation.references import EvidenceReference
from dr_code.humaneval.job import HumanEvalCandidateJobResult
from dr_code.metrics import MetricRecord
from dr_code.trace import Absent, SerializedTrace

EVAL_ATTEMPT_SCHEMA_VERSION: Final = 3
SAMPLE_EVAL_RECORD_SCHEMA_VERSION: Final = 2
CANDIDATE_EXECUTION_RECORD_SCHEMA_VERSION: Final = 2


class ExecutedCandidateProvenance(FrozenModel):
    kind: Literal["executed"] = "executed"
    record_receipt: RecordReceipt


class ReusedCandidateProvenance(FrozenModel):
    kind: Literal["reused"] = "reused"
    source_record: EvidenceReference


CandidateExecutionProvenance: TypeAlias = Annotated[
    ExecutedCandidateProvenance | ReusedCandidateProvenance,
    Field(discriminator="kind"),
]


@verify(UNIQUE)
class CandidateTerminationReason(StrEnum):
    # Never build payloads by iterating this closed persisted vocabulary.

    NONZERO_EXIT = "nonzero_exit"
    SIGNALED = "signaled"
    WALL_TIME = "wall_time"
    PAYLOAD_OUTPUT = "payload_output"
    PAYLOAD_PROTOCOL = "payload_protocol"


class CandidateJobCompleted(FrozenModel):
    kind: Literal["completed"] = "completed"
    result: HumanEvalCandidateJobResult
    execution_outcome: ExecutionOutcome
    attribution: ExecutionAttribution
    measurements: ExecutionMeasurements


class CandidateJobTerminated(FrozenModel):
    kind: Literal["candidate_terminated"] = "candidate_terminated"
    reason: CandidateTerminationReason
    execution_outcome: ExecutionOutcome
    attribution: ExecutionAttribution
    measurements: ExecutionMeasurements


class HarnessExecutionFailure(FrozenModel):
    kind: Literal["harness_failure"] = "harness_failure"
    failure_type: str
    message: str
    execution_outcome: ExecutionOutcome | None
    attribution: ExecutionAttribution | None
    measurements: ExecutionMeasurements | None


class ExecutorExecutionFailure(FrozenModel):
    kind: Literal["executor_failure"] = "executor_failure"
    failure_type: str
    message: str
    execution_outcome: ExecutionOutcome | None
    attribution: ExecutionAttribution | None
    measurements: ExecutionMeasurements | None


CandidateExecutionOutcome: TypeAlias = Annotated[
    CandidateJobCompleted
    | CandidateJobTerminated
    | HarnessExecutionFailure
    | ExecutorExecutionFailure,
    Field(discriminator="kind"),
]


@verify(UNIQUE)
class FailureClass(StrEnum):
    # Never build payloads by iterating this closed persisted vocabulary.

    HARNESS = "harness"
    CANDIDATE = "candidate"
    INFRASTRUCTURE = "infrastructure"


def failure_class_of(
    outcome: CandidateExecutionOutcome,
    /,
) -> FailureClass | None:
    """Name which party owns a candidate execution outcome's failure.

    A completed job has no failure to attribute and returns ``None``. Wall-time
    budget exhaustion is infrastructure-owned at dr-exec 0.1.10, which
    attributes it to the executor.
    """

    match outcome:
        case CandidateJobCompleted():
            return None
        case CandidateJobTerminated():
            return FailureClass.CANDIDATE
        case HarnessExecutionFailure():
            return FailureClass.HARNESS
        case ExecutorExecutionFailure():
            return FailureClass.INFRASTRUCTURE


def outcome_is_cacheable(outcome: CandidateExecutionOutcome, /) -> bool:
    """Say whether an outcome describes the candidate rather than the run.

    Only candidate-owned outcomes are reusable evidence: a completed job and a
    candidate-owned termination both describe the candidate itself, so a later
    identical request may reuse them. Harness and infrastructure failures
    describe the run that observed the candidate, so caching them would replay
    an environment fault as if it were the candidate's behavior.
    """

    return failure_class_of(outcome) in {None, FailureClass.CANDIDATE}


class CandidateExecutionRecord(FrozenModel):
    schema_version: Literal[2] = CANDIDATE_EXECUTION_RECORD_SCHEMA_VERSION
    candidate: EvalCandidateId
    request_identity: Sha256Digest
    runtime: EvalRuntimeId
    cache_namespace: str
    cache_key: str
    provenance: CandidateExecutionProvenance
    outcome: CandidateExecutionOutcome


class PreprocessingAbsentSampleRecord(FrozenModel):
    schema_version: Literal[2] = SAMPLE_EVAL_RECORD_SCHEMA_VERSION
    status: Literal["preprocessing_absent"] = "preprocessing_absent"
    slot: EvalSlotId
    sample: EvalSampleMetadata
    trace: SerializedTrace
    absence: Absent

    @model_validator(mode="after")
    def validate_slot_sample_task(self) -> Self:
        _validate_slot_sample_task(self.slot, self.sample)
        return self


class NoCandidatesSampleRecord(FrozenModel):
    schema_version: Literal[2] = SAMPLE_EVAL_RECORD_SCHEMA_VERSION
    status: Literal["no_candidates"] = "no_candidates"
    slot: EvalSlotId
    sample: EvalSampleMetadata
    trace: SerializedTrace

    @model_validator(mode="after")
    def validate_slot_sample_task(self) -> Self:
        _validate_slot_sample_task(self.slot, self.sample)
        return self


class EvaluatedSampleRecord(FrozenModel):
    schema_version: Literal[2] = SAMPLE_EVAL_RECORD_SCHEMA_VERSION
    status: Literal["evaluated"] = "evaluated"
    slot: EvalSlotId
    sample: EvalSampleMetadata
    trace: SerializedTrace
    candidates: tuple[MaterializedEvalCandidate, ...]
    executions: tuple[CandidateExecutionRecord, ...]
    metrics: tuple[MetricRecord, ...]

    @model_validator(mode="after")
    def validate_candidates_and_executions(self) -> Self:
        _validate_slot_sample_task(self.slot, self.sample)
        identities = tuple(candidate.identity for candidate in self.candidates)
        if any(
            identity.sample != self.sample.identity for identity in identities
        ):
            raise ValueError(
                "materialized candidate sample identities must match the sample record"
            )
        if len(set(identities)) != len(identities):
            raise ValueError(
                "materialized candidate identities must be unique"
            )
        execution_candidates = tuple(
            execution.candidate for execution in self.executions
        )
        if len(set(execution_candidates)) != len(execution_candidates):
            raise ValueError("candidate execution records must be unique")
        if execution_candidates != identities:
            raise ValueError(
                "candidate execution records must match materialized candidates in order"
            )
        return self


def _validate_slot_sample_task(
    slot: EvalSlotId,
    sample: EvalSampleMetadata,
) -> None:
    if slot.task_id != sample.task_id:
        raise ValueError(
            "sample task_id must match the evaluation slot task_id"
        )


SampleEvalRecord: TypeAlias = Annotated[
    PreprocessingAbsentSampleRecord
    | NoCandidatesSampleRecord
    | EvaluatedSampleRecord,
    Field(discriminator="status"),
]
SAMPLE_EVAL_RECORD_ADAPTER: Final = TypeAdapter[SampleEvalRecord](
    SampleEvalRecord
)


@verify(UNIQUE)
class AttemptCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


@verify(UNIQUE)
class AttemptValidity(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


@verify(UNIQUE)
class AttemptLimitKind(StrEnum):
    SLOTS = "slots"
    MATERIALIZED_CANDIDATES = "materialized_candidates"
    ADMITTED_JOBS = "admitted_jobs"
    RETAINED_EVIDENCE_BYTES = "retained_evidence_bytes"
    PROJECTION_ROWS = "projection_rows"


class AttemptLimitExhaustion(FrozenModel):
    limit: AttemptLimitKind
    configured: PositiveInt
    observed: PositiveInt

    @model_validator(mode="after")
    def validate_observed_exceeds_configured(self) -> Self:
        if self.observed <= self.configured:
            raise ValueError(
                "limit exhaustion observed count must exceed configured count"
            )
        return self


class EvalMemberRecord(FrozenModel):
    slot: EvalSlotId
    sample: EvalSampleId
    record: EvidenceReference | None


@verify(UNIQUE)
class ReplayMode(StrEnum):
    SAMPLES = "samples"
    MATERIALIZED_CANDIDATES = "materialized_candidates"


class ReplaySource(FrozenModel):
    attempt: EvalAttemptId
    mode: ReplayMode


class EvalAttemptRecord(FrozenModel):
    schema_version: Literal[3] = EVAL_ATTEMPT_SCHEMA_VERSION
    identity: EvalAttemptId
    plan: EvalPlan
    runtime: EvalRuntimeId
    cache_namespace: str
    members: tuple[EvalMemberRecord, ...]
    completeness: AttemptCompleteness
    validity: AttemptValidity
    limit_exhaustion: AttemptLimitExhaustion | None
    replay: ReplaySource | None

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        slots = tuple(member.slot for member in self.members)
        samples = tuple(member.sample for member in self.members)
        if len(set(slots)) != len(slots):
            raise ValueError("evaluation member slots must be unique")
        if len(set(samples)) != len(samples):
            raise ValueError("evaluation member samples must be unique")
        has_missing = any(member.record is None for member in self.members)
        if self.limit_exhaustion is not None and (
            self.completeness is not AttemptCompleteness.PARTIAL
            or self.validity is not AttemptValidity.INVALID
        ):
            raise ValueError(
                "limit exhaustion requires a partial invalid evaluation attempt"
            )
        if self.completeness is AttemptCompleteness.COMPLETE and has_missing:
            raise ValueError(
                "a complete evaluation attempt cannot have a missing record"
            )
        if self.completeness is AttemptCompleteness.PARTIAL:
            if self.validity is not AttemptValidity.INVALID:
                raise ValueError(
                    "a partial evaluation attempt must be invalid"
                )
            if not has_missing:
                raise ValueError(
                    "a partial evaluation attempt must have a missing record"
                )
        from dr_code.evaluation.validation import validate_attempt_membership

        validate_attempt_membership(self)
        return self


__all__ = [
    "AttemptCompleteness",
    "AttemptLimitExhaustion",
    "AttemptLimitKind",
    "AttemptValidity",
    "CANDIDATE_EXECUTION_RECORD_SCHEMA_VERSION",
    "CandidateExecutionOutcome",
    "CandidateExecutionProvenance",
    "CandidateExecutionRecord",
    "CandidateJobCompleted",
    "CandidateJobTerminated",
    "CandidateTerminationReason",
    "EVAL_ATTEMPT_SCHEMA_VERSION",
    "EvalAttemptRecord",
    "EvalMemberRecord",
    "EvaluatedSampleRecord",
    "ExecutedCandidateProvenance",
    "ExecutorExecutionFailure",
    "FailureClass",
    "HarnessExecutionFailure",
    "NoCandidatesSampleRecord",
    "PreprocessingAbsentSampleRecord",
    "ReplayMode",
    "ReplaySource",
    "ReusedCandidateProvenance",
    "SAMPLE_EVAL_RECORD_ADAPTER",
    "SAMPLE_EVAL_RECORD_SCHEMA_VERSION",
    "SampleEvalRecord",
    "failure_class_of",
    "outcome_is_cacheable",
]

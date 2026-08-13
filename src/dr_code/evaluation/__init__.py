from __future__ import annotations

from typing import TYPE_CHECKING

from dr_code.evaluation.aggregation import (
    AggregationEmptyDenominator,
    AggregationInput,
    AggregationMissing,
    AggregationNonFinite,
    AggregationNotApplicable,
    AggregationOk,
    AggregationResult,
    AggregationSlot,
    AggregationStatus,
    aggregate,
)
from dr_code.evaluation.batch import (
    AttemptLimits,
    CANDIDATE_PAYLOAD_OUTPUT_BYTES,
    CANDIDATE_STREAM_HEAD_BYTES,
    CandidateJobBudget,
    EvalBatchRequest,
    EvalBatchResult,
    EvalProjectionReference,
    ProjectionKind,
    ProjectionRequest,
    RecordPlacement,
    RunGrade,
    SampleData,
    SampleWithCandidatesData,
    ShardLimits,
    SlotData,
    SlotPayload,
    WindowLimits,
    evaluate_batch,
    evaluate_durable_partition,
)
from dr_code.evaluation.coordinates import (
    DatasetCoordinate,
    SamplingPlan,
    SamplingPlanCoordinate,
    TaskSet,
    TaskSetCoordinate,
)
from dr_code.evaluation.id import (
    CorpusSampleProvenance,
    EvalAttemptId,
    EvalCandidateId,
    EvalRuntimeId,
    EvalSample,
    EvalSampleAuxiliaryArtifact,
    EvalSampleId,
    EvalSampleMetadata,
    EvalSampleProvenance,
    EvalSlotId,
    EvalSourceId,
    GeneratedSampleProvenance,
    MaterializedEvalCandidate,
    SyntheticSampleProvenance,
)
from dr_code.evaluation.plan import (
    AggregationPolicy,
    AggregationStatistic,
    EvalPlan,
    EvalProcedure,
    NotApplicablePolicy,
)
from dr_code.evaluation.score import EvalCoordinate, Score
from dr_code.evaluation.projections import (
    AggregationResultProjectionRow,
    EvalSampleProjectionRow,
    MaterializedCandidateProjectionRow,
    MetricRecordProjectionRow,
    ProjectionRow,
    ScoreProjectionRow,
)
from dr_code.evaluation.references import (
    BundleRecordReference,
    EvidenceReference,
    StoredRecordReference,
)
from dr_code.evaluation.work_key import (
    WORK_KEY_SCHEMA,
    WORK_KEY_SCHEMA_VERSION,
    derive_work_key,
)

if TYPE_CHECKING:
    from dr_code.evaluation.bundle import (
        EVAL_BUNDLE_FORMAT,
        EVAL_BUNDLE_SCHEMA_VERSION,
        EVAL_PROJECTION_FORMAT,
        EVAL_PROJECTION_SCHEMA_VERSION,
        SAMPLE_RECORD_OBJECT_SCHEMA,
        EvalBundleAudit,
        EvalBundlePayload,
        EvalReadLimits,
        ProjectionArtifactHeader,
        RestoredEvalAttempt,
        audit_eval_bundle,
        read_eval_projection,
        restore_eval_attempt,
    )
    from dr_code.evaluation.comparison import (
        ComparableProjectionComparison,
        ComparisonStatus,
        EvalEvidenceResolver,
        ProjectionComparison,
        ProjectionNotComparable,
        StructuralEvalComparison,
        StructuralMemberId,
        StructuralRecordComparison,
        compare_eval_attempts,
    )
    from dr_code.evaluation.evidence import (
        ATTEMPT_RECORD_OBJECT_SCHEMA,
        EnlistedObjectStore,
        OUTPUT_REFERENCE_BINDING_PREFIX,
        commit_eval_evidence,
        output_reference_binding_key,
        sample_record_binding_key,
    )
    from dr_code.evaluation.records import (
        AttemptCompleteness,
        AttemptLimitExhaustion,
        AttemptLimitKind,
        AttemptValidity,
        CANDIDATE_EXECUTION_RECORD_SCHEMA_VERSION,
        CandidateExecutionOutcome,
        CandidateExecutionProvenance,
        CandidateExecutionRecord,
        CandidateJobCompleted,
        CandidateJobTerminated,
        CandidateTerminationReason,
        EVAL_ATTEMPT_SCHEMA_VERSION,
        EvalAttemptRecord,
        EvalMemberRecord,
        EvaluatedSampleRecord,
        ExecutedCandidateProvenance,
        ExecutorExecutionFailure,
        FailureClass,
        HarnessExecutionFailure,
        NoCandidatesSampleRecord,
        PreprocessingAbsentSampleRecord,
        ReplayMode,
        ReplaySource,
        ReusedCandidateProvenance,
        SAMPLE_EVAL_RECORD_ADAPTER,
        SAMPLE_EVAL_RECORD_SCHEMA_VERSION,
        SampleEvalRecord,
        failure_class_of,
        outcome_is_cacheable,
    )
    from dr_code.evaluation.flows import (
        PreprocessingCoverage,
        PreprocessingValidation,
        TestingValidation,
        validate_preprocessing,
        validate_testing,
    )
    from dr_code.evaluation.replay import (
        ReplayPreflight,
        ReplayReady,
        ReplayUnavailable,
        preflight_replay,
        replay_eval_attempt,
    )

_RECORD_EXPORTS = frozenset(
    {
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
    }
)

_BUNDLE_EXPORTS = frozenset(
    {
        "EVAL_BUNDLE_FORMAT",
        "EVAL_BUNDLE_SCHEMA_VERSION",
        "EVAL_PROJECTION_FORMAT",
        "EVAL_PROJECTION_SCHEMA_VERSION",
        "SAMPLE_RECORD_OBJECT_SCHEMA",
        "EvalBundleAudit",
        "EvalBundlePayload",
        "EvalReadLimits",
        "ProjectionArtifactHeader",
        "RestoredEvalAttempt",
        "audit_eval_bundle",
        "read_eval_projection",
        "restore_eval_attempt",
    }
)

_EVIDENCE_EXPORTS = frozenset(
    {
        "ATTEMPT_RECORD_OBJECT_SCHEMA",
        "EnlistedObjectStore",
        "OUTPUT_REFERENCE_BINDING_PREFIX",
        "commit_eval_evidence",
        "output_reference_binding_key",
        "sample_record_binding_key",
    }
)

_COMPARISON_EXPORTS = frozenset(
    {
        "ComparableProjectionComparison",
        "ComparisonStatus",
        "EvalEvidenceResolver",
        "ProjectionComparison",
        "ProjectionNotComparable",
        "StructuralEvalComparison",
        "StructuralMemberId",
        "StructuralRecordComparison",
        "compare_eval_attempts",
    }
)

_FLOW_EXPORTS = frozenset(
    {
        "PreprocessingCoverage",
        "PreprocessingValidation",
        "TestingValidation",
        "validate_preprocessing",
        "validate_testing",
    }
)

_REPLAY_EXPORTS = frozenset(
    {
        "ReplayPreflight",
        "ReplayReady",
        "ReplayUnavailable",
        "preflight_replay",
        "replay_eval_attempt",
    }
)


def __getattr__(name: str) -> object:
    if name in _RECORD_EXPORTS:
        from dr_code.evaluation import records

        value = getattr(records, name)
    elif name in _BUNDLE_EXPORTS:
        from dr_code.evaluation import bundle

        value = getattr(bundle, name)
    elif name in _EVIDENCE_EXPORTS:
        from dr_code.evaluation import evidence

        value = getattr(evidence, name)
    elif name in _COMPARISON_EXPORTS:
        from dr_code.evaluation import comparison

        value = getattr(comparison, name)
    elif name in _FLOW_EXPORTS:
        from dr_code.evaluation import flows

        value = getattr(flows, name)
    elif name in _REPLAY_EXPORTS:
        from dr_code.evaluation import replay

        value = getattr(replay, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | _RECORD_EXPORTS
        | _BUNDLE_EXPORTS
        | _EVIDENCE_EXPORTS
        | _COMPARISON_EXPORTS
        | _FLOW_EXPORTS
        | _REPLAY_EXPORTS
    )


__all__ = [
    "AggregationEmptyDenominator",
    "AggregationInput",
    "AggregationMissing",
    "AggregationNonFinite",
    "AggregationNotApplicable",
    "AggregationOk",
    "AggregationPolicy",
    "AggregationResult",
    "AggregationSlot",
    "AggregationStatistic",
    "AggregationStatus",
    "AggregationResultProjectionRow",
    "AttemptLimits",
    "AttemptCompleteness",
    "AttemptLimitExhaustion",
    "AttemptLimitKind",
    "AttemptValidity",
    "BundleRecordReference",
    "CANDIDATE_EXECUTION_RECORD_SCHEMA_VERSION",
    "CANDIDATE_PAYLOAD_OUTPUT_BYTES",
    "CANDIDATE_STREAM_HEAD_BYTES",
    "CandidateExecutionOutcome",
    "CandidateExecutionProvenance",
    "CandidateExecutionRecord",
    "CandidateJobCompleted",
    "CandidateJobBudget",
    "CandidateJobTerminated",
    "CandidateTerminationReason",
    "ComparableProjectionComparison",
    "ComparisonStatus",
    "CorpusSampleProvenance",
    "DatasetCoordinate",
    "EvalCoordinate",
    "EvalEvidenceResolver",
    "EVAL_ATTEMPT_SCHEMA_VERSION",
    "EvalAttemptId",
    "EvalAttemptRecord",
    "EvalBatchRequest",
    "EvalBatchResult",
    "EvalBundleAudit",
    "EvalBundlePayload",
    "EvalCandidateId",
    "EvalMemberRecord",
    "EvalPlan",
    "EvalProcedure",
    "EvalRuntimeId",
    "EvalSample",
    "EvalSampleAuxiliaryArtifact",
    "EvalSampleId",
    "EvalSampleMetadata",
    "EvalSampleProjectionRow",
    "EvalSampleProvenance",
    "EvalSlotId",
    "EvalSourceId",
    "EvaluatedSampleRecord",
    "EvalProjectionReference",
    "EvalReadLimits",
    "EvidenceReference",
    "ExecutedCandidateProvenance",
    "ExecutorExecutionFailure",
    "FailureClass",
    "GeneratedSampleProvenance",
    "HarnessExecutionFailure",
    "MaterializedEvalCandidate",
    "MaterializedCandidateProjectionRow",
    "MetricRecordProjectionRow",
    "NoCandidatesSampleRecord",
    "NotApplicablePolicy",
    "PreprocessingAbsentSampleRecord",
    "PreprocessingCoverage",
    "PreprocessingValidation",
    "ProjectionKind",
    "ProjectionArtifactHeader",
    "ProjectionComparison",
    "ProjectionRequest",
    "ProjectionNotComparable",
    "ProjectionRow",
    "ReplayMode",
    "ReplayPreflight",
    "ReplayReady",
    "ReplaySource",
    "ReplayUnavailable",
    "ReusedCandidateProvenance",
    "SAMPLE_EVAL_RECORD_ADAPTER",
    "SAMPLE_EVAL_RECORD_SCHEMA_VERSION",
    "SAMPLE_RECORD_OBJECT_SCHEMA",
    "SampleData",
    "SampleEvalRecord",
    "SampleWithCandidatesData",
    "Score",
    "ScoreProjectionRow",
    "RecordPlacement",
    "RunGrade",
    "SlotData",
    "SlotPayload",
    "SamplingPlan",
    "SamplingPlanCoordinate",
    "ShardLimits",
    "StoredRecordReference",
    "StructuralEvalComparison",
    "StructuralMemberId",
    "StructuralRecordComparison",
    "SyntheticSampleProvenance",
    "TaskSet",
    "TaskSetCoordinate",
    "TestingValidation",
    "WindowLimits",
    "WORK_KEY_SCHEMA",
    "WORK_KEY_SCHEMA_VERSION",
    "aggregate",
    "derive_work_key",
    "failure_class_of",
    "outcome_is_cacheable",
    "audit_eval_bundle",
    "compare_eval_attempts",
    "evaluate_batch",
    "evaluate_durable_partition",
    "read_eval_projection",
    "preflight_replay",
    "replay_eval_attempt",
    "restore_eval_attempt",
    "validate_preprocessing",
    "validate_testing",
    "RestoredEvalAttempt",
    "EVAL_BUNDLE_FORMAT",
    "EVAL_BUNDLE_SCHEMA_VERSION",
    "EVAL_PROJECTION_FORMAT",
    "EVAL_PROJECTION_SCHEMA_VERSION",
    "ATTEMPT_RECORD_OBJECT_SCHEMA",
    "OUTPUT_REFERENCE_BINDING_PREFIX",
    "EnlistedObjectStore",
    "commit_eval_evidence",
    "output_reference_binding_key",
    "sample_record_binding_key",
]

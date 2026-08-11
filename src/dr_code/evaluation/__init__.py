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
    CandidateJobBudget,
    EvaluationBatchRequest,
    EvaluationBatchResult,
    EvaluationInput,
    EvaluationProjectionReference,
    FrozenCandidateEvaluationInput,
    ProjectionKind,
    ProjectionRequest,
    RecordPlacement,
    SampleEvaluationInput,
    ShardLimits,
    WindowLimits,
    evaluate_batch,
    evaluate_durable_partition,
)
from dr_code.evaluation.coordinates import (
    DatasetCoordinate,
    RepeatPlan,
    RepeatPlanCoordinate,
    TaskSet,
    TaskSetCoordinate,
)
from dr_code.evaluation.identity import (
    CorpusSampleProvenance,
    EvaluationAttemptIdentity,
    EvaluationCandidateIdentity,
    EvaluationRuntimeIdentity,
    EvaluationSample,
    EvaluationSampleAuxiliaryArtifact,
    EvaluationSampleIdentity,
    EvaluationSampleMetadata,
    EvaluationSampleProvenance,
    EvaluationSlotIdentity,
    EvaluationSourceIdentity,
    GeneratedSampleProvenance,
    MaterializedEvaluationCandidate,
    SyntheticSampleProvenance,
)
from dr_code.evaluation.plan import (
    AggregationPolicy,
    AggregationStatistic,
    EvaluationPlan,
    EvaluationProcedure,
    NotApplicablePolicy,
)
from dr_code.evaluation.score import EvaluationCoordinate, Score
from dr_code.evaluation.projections import (
    AggregationResultProjectionRow,
    EvaluationSampleProjectionRow,
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

if TYPE_CHECKING:
    from dr_code.evaluation.bundle import (
        EVALUATION_BUNDLE_FORMAT,
        EVALUATION_BUNDLE_SCHEMA_VERSION,
        EVALUATION_PROJECTION_FORMAT,
        EVALUATION_PROJECTION_SCHEMA_VERSION,
        SAMPLE_RECORD_OBJECT_SCHEMA,
        EvaluationBundleAudit,
        EvaluationBundlePayload,
        EvaluationReadLimits,
        ProjectionArtifactHeader,
        RestoredEvaluationAttempt,
        audit_evaluation_bundle,
        read_evaluation_projection,
        restore_evaluation_attempt,
    )
    from dr_code.evaluation.comparison import (
        ComparableProjectionComparison,
        ComparisonStatus,
        EvaluationEvidenceResolver,
        ProjectionComparison,
        ProjectionNotComparable,
        StructuralEvaluationComparison,
        StructuralMemberIdentity,
        StructuralRecordComparison,
        compare_evaluation_attempts,
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
        EVALUATION_ATTEMPT_SCHEMA_VERSION,
        EvaluationAttemptRecord,
        EvaluationMemberRecord,
        EvaluatedSampleRecord,
        ExecutedCandidateProvenance,
        ExecutorExecutionFailure,
        HarnessExecutionFailure,
        NoCandidatesSampleRecord,
        PreprocessingAbsentSampleRecord,
        ReplayMode,
        ReplaySource,
        ReusedCandidateProvenance,
        SAMPLE_EVALUATION_RECORD_ADAPTER,
        SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION,
        SampleEvaluationRecord,
    )
    from dr_code.evaluation.replay import (
        ReplayPreflight,
        ReplayReady,
        ReplayUnavailable,
        preflight_replay,
        replay_evaluation_attempt,
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
        "EVALUATION_ATTEMPT_SCHEMA_VERSION",
        "EvaluationAttemptRecord",
        "EvaluationMemberRecord",
        "EvaluatedSampleRecord",
        "ExecutedCandidateProvenance",
        "ExecutorExecutionFailure",
        "HarnessExecutionFailure",
        "NoCandidatesSampleRecord",
        "PreprocessingAbsentSampleRecord",
        "ReplayMode",
        "ReplaySource",
        "ReusedCandidateProvenance",
        "SAMPLE_EVALUATION_RECORD_ADAPTER",
        "SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION",
        "SampleEvaluationRecord",
    }
)

_BUNDLE_EXPORTS = frozenset(
    {
        "EVALUATION_BUNDLE_FORMAT",
        "EVALUATION_BUNDLE_SCHEMA_VERSION",
        "EVALUATION_PROJECTION_FORMAT",
        "EVALUATION_PROJECTION_SCHEMA_VERSION",
        "SAMPLE_RECORD_OBJECT_SCHEMA",
        "EvaluationBundleAudit",
        "EvaluationBundlePayload",
        "EvaluationReadLimits",
        "ProjectionArtifactHeader",
        "RestoredEvaluationAttempt",
        "audit_evaluation_bundle",
        "read_evaluation_projection",
        "restore_evaluation_attempt",
    }
)

_COMPARISON_EXPORTS = frozenset(
    {
        "ComparableProjectionComparison",
        "ComparisonStatus",
        "EvaluationEvidenceResolver",
        "ProjectionComparison",
        "ProjectionNotComparable",
        "StructuralEvaluationComparison",
        "StructuralMemberIdentity",
        "StructuralRecordComparison",
        "compare_evaluation_attempts",
    }
)

_REPLAY_EXPORTS = frozenset(
    {
        "ReplayPreflight",
        "ReplayReady",
        "ReplayUnavailable",
        "preflight_replay",
        "replay_evaluation_attempt",
    }
)


def __getattr__(name: str) -> object:
    if name in _RECORD_EXPORTS:
        from dr_code.evaluation import records

        value = getattr(records, name)
    elif name in _BUNDLE_EXPORTS:
        from dr_code.evaluation import bundle

        value = getattr(bundle, name)
    elif name in _COMPARISON_EXPORTS:
        from dr_code.evaluation import comparison

        value = getattr(comparison, name)
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
        | _COMPARISON_EXPORTS
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
    "EvaluationCoordinate",
    "EvaluationEvidenceResolver",
    "EVALUATION_ATTEMPT_SCHEMA_VERSION",
    "EvaluationAttemptIdentity",
    "EvaluationAttemptRecord",
    "EvaluationBatchRequest",
    "EvaluationBatchResult",
    "EvaluationBundleAudit",
    "EvaluationBundlePayload",
    "EvaluationCandidateIdentity",
    "EvaluationMemberRecord",
    "EvaluationPlan",
    "EvaluationProcedure",
    "EvaluationRuntimeIdentity",
    "EvaluationSample",
    "EvaluationSampleAuxiliaryArtifact",
    "EvaluationSampleIdentity",
    "EvaluationSampleMetadata",
    "EvaluationSampleProjectionRow",
    "EvaluationSampleProvenance",
    "EvaluationSlotIdentity",
    "EvaluationSourceIdentity",
    "EvaluatedSampleRecord",
    "EvaluationInput",
    "EvaluationProjectionReference",
    "EvaluationReadLimits",
    "EvidenceReference",
    "ExecutedCandidateProvenance",
    "ExecutorExecutionFailure",
    "GeneratedSampleProvenance",
    "HarnessExecutionFailure",
    "FrozenCandidateEvaluationInput",
    "MaterializedEvaluationCandidate",
    "MaterializedCandidateProjectionRow",
    "MetricRecordProjectionRow",
    "NoCandidatesSampleRecord",
    "NotApplicablePolicy",
    "RepeatPlan",
    "RepeatPlanCoordinate",
    "PreprocessingAbsentSampleRecord",
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
    "SAMPLE_EVALUATION_RECORD_ADAPTER",
    "SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION",
    "SAMPLE_RECORD_OBJECT_SCHEMA",
    "SampleEvaluationRecord",
    "Score",
    "ScoreProjectionRow",
    "RecordPlacement",
    "SampleEvaluationInput",
    "ShardLimits",
    "StoredRecordReference",
    "StructuralEvaluationComparison",
    "StructuralMemberIdentity",
    "StructuralRecordComparison",
    "SyntheticSampleProvenance",
    "TaskSet",
    "TaskSetCoordinate",
    "WindowLimits",
    "aggregate",
    "audit_evaluation_bundle",
    "compare_evaluation_attempts",
    "evaluate_batch",
    "evaluate_durable_partition",
    "read_evaluation_projection",
    "preflight_replay",
    "replay_evaluation_attempt",
    "restore_evaluation_attempt",
    "RestoredEvaluationAttempt",
    "EVALUATION_BUNDLE_FORMAT",
    "EVALUATION_BUNDLE_SCHEMA_VERSION",
    "EVALUATION_PROJECTION_FORMAT",
    "EVALUATION_PROJECTION_SCHEMA_VERSION",
]

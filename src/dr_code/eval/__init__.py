"""dr-code evaluation kernel: Definition/Config lifecycle, typed code
artifacts, Metric Facts/Scores, compression references, and pure
aggregation.

Experiment policy, authority, optimization, and orchestration stay in
Whetstone; this package is the reusable, provenance-free kernel.
"""

from dr_code.eval.aggregation import (
    AggregationInput,
    AggregationOutput,
    AggregationStatus,
    aggregate,
)
from dr_code.eval.code import (
    CodeArtifact,
    CodeCandidate,
    CodeCandidateSet,
    CodeCompilationError,
    PythonSource,
)
from dr_code.eval.compression_reference import (
    ZERO_DENOMINATOR,
    CompressionReferenceArtifact,
    CompressionReferenceKey,
    CompressionReferenceResolver,
    ReferenceResolutionError,
    compression_ratio,
)
from dr_code.eval.facts import (
    AbsenceMode,
    Applicability,
    MetricFact,
    MetricRecord,
    OperatorLineage,
    Score,
)
from dr_code.eval.identity import identity_hash_for
from dr_code.eval.lifecycle import (
    AggregationConfig,
    AggregationDefinition,
    DefinitionRef,
    EvalConfig,
    EvalDefinition,
    EvaluationProcedureConfig,
    EvaluationProcedureDefinition,
    MetricExtractionConfig,
    MetricExtractionDefinition,
    MetricQuestionBinding,
    PreprocessingConfig,
    PreprocessingDefinition,
    PreprocessingStepBinding,
    SamplingConfig,
    SamplingDefinition,
)
from dr_code.eval.resolved_versions import (
    resolved_operator_version,
    resolved_step_version,
)
from dr_code.eval.tasks import (
    Repeat,
    RepeatId,
    RepeatPlan,
    RepeatProvenanceRow,
    SelectionRule,
    TaskSet,
    humaneval_task_identity,
    humaneval_task_identity_payload,
    repeat_plan_from_provenance,
)
from dr_code.eval.variables import (
    VariableError,
    VariableSpec,
    resolve_assignment,
)

__all__ = [
    "ZERO_DENOMINATOR",
    "AbsenceMode",
    "AggregationConfig",
    "AggregationDefinition",
    "AggregationInput",
    "AggregationOutput",
    "AggregationStatus",
    "Applicability",
    "CodeArtifact",
    "CodeCandidate",
    "CodeCandidateSet",
    "CodeCompilationError",
    "CompressionReferenceArtifact",
    "CompressionReferenceKey",
    "CompressionReferenceResolver",
    "DefinitionRef",
    "EvalConfig",
    "EvalDefinition",
    "EvaluationProcedureConfig",
    "EvaluationProcedureDefinition",
    "MetricExtractionConfig",
    "MetricExtractionDefinition",
    "MetricFact",
    "MetricQuestionBinding",
    "MetricRecord",
    "OperatorLineage",
    "PreprocessingConfig",
    "PreprocessingDefinition",
    "PreprocessingStepBinding",
    "PythonSource",
    "ReferenceResolutionError",
    "Repeat",
    "RepeatId",
    "RepeatPlan",
    "RepeatProvenanceRow",
    "SamplingConfig",
    "SamplingDefinition",
    "Score",
    "SelectionRule",
    "TaskSet",
    "VariableError",
    "VariableSpec",
    "aggregate",
    "compression_ratio",
    "humaneval_task_identity",
    "humaneval_task_identity_payload",
    "identity_hash_for",
    "repeat_plan_from_provenance",
    "resolve_assignment",
    "resolved_operator_version",
    "resolved_step_version",
]

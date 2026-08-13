from __future__ import annotations

import ast
import importlib
from importlib.util import resolve_name
from pathlib import Path

DR_CODE_PACKAGE = Path(__file__).parents[2] / "src" / "dr_code"
CORE_PACKAGE = DR_CODE_PACKAGE / "core"
FUNCTIONAL_PACKAGES = frozenset(
    {
        "caching",
        "evaluation",
        "generation_corpus",
        "humaneval",
        "metrics",
        "preprocessing",
        "synthetic",
        "trace",
    }
)

EVAL_CORE_EXPORTS = {
    "dr_code.caching": frozenset(
        {
            "CACHED_EXECUTION_OBSERVATION_SCHEMA_VERSION",
            "EXECUTION_CACHE_NAMESPACE",
            "EXECUTION_CACHE_RECORD_SCHEMA",
            "CachedExecutionObservation",
            "ExecutionCacheStats",
            "WindowedExecutionCache",
        }
    ),
    "dr_code.evaluation": frozenset(
        {
            "AggregationPolicy",
            "AttemptCompleteness",
            "AttemptLimitExhaustion",
            "AttemptLimitKind",
            "AttemptLimits",
            "AttemptValidity",
            "BundleRecordReference",
            "CANDIDATE_EXECUTION_RECORD_SCHEMA_VERSION",
            "CandidateExecutionOutcome",
            "CandidateExecutionProvenance",
            "CandidateExecutionRecord",
            "CandidateJobBudget",
            "CandidateJobCompleted",
            "CandidateJobTerminated",
            "CandidateTerminationReason",
            "ComparableProjectionComparison",
            "ComparisonStatus",
            "CorpusSampleProvenance",
            "EVAL_ATTEMPT_SCHEMA_VERSION",
            "EVAL_BUNDLE_FORMAT",
            "EVAL_BUNDLE_SCHEMA_VERSION",
            "EVAL_PROJECTION_FORMAT",
            "EVAL_PROJECTION_SCHEMA_VERSION",
            "EvalAttemptId",
            "EvalAttemptRecord",
            "EvalBatchRequest",
            "EvalBatchResult",
            "EvalBundleAudit",
            "EvalBundlePayload",
            "EvalCandidateId",
            "EvalEvidenceResolver",
            "EvalMemberRecord",
            "EvalProjectionReference",
            "EvalReadLimits",
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
            "EvidenceReference",
            "ExecutedCandidateProvenance",
            "ExecutorExecutionFailure",
            "GeneratedSampleProvenance",
            "HarnessExecutionFailure",
            "MaterializedCandidateProjectionRow",
            "MaterializedEvalCandidate",
            "MetricRecordProjectionRow",
            "NoCandidatesSampleRecord",
            "PreprocessingAbsentSampleRecord",
            "ProjectionArtifactHeader",
            "ProjectionComparison",
            "ProjectionKind",
            "ProjectionNotComparable",
            "ProjectionRequest",
            "ProjectionRow",
            "RecordPlacement",
            "ReplayMode",
            "ReplayPreflight",
            "ReplayReady",
            "ReplaySource",
            "ReplayUnavailable",
            "RestoredEvalAttempt",
            "ReusedCandidateProvenance",
            "SAMPLE_EVAL_RECORD_SCHEMA_VERSION",
            "SAMPLE_RECORD_OBJECT_SCHEMA",
            "SampleData",
            "SampleEvalRecord",
            "SampleWithCandidatesData",
            "Score",
            "ScoreProjectionRow",
            "ShardLimits",
            "SlotData",
            "StoredRecordReference",
            "StructuralEvalComparison",
            "StructuralMemberId",
            "StructuralRecordComparison",
            "SyntheticSampleProvenance",
            "WindowLimits",
            "AggregationResultProjectionRow",
            "audit_eval_bundle",
            "compare_eval_attempts",
            "evaluate_batch",
            "evaluate_durable_partition",
            "preflight_replay",
            "read_eval_projection",
            "replay_eval_attempt",
            "restore_eval_attempt",
        }
    ),
    "dr_code.humaneval": frozenset(
        {
            "CandidateNamespaceFailure",
            "CandidateNamespaceLoaded",
            "CandidateNamespaceOutcome",
            "CompletedSubmissionResult",
            "HUMANEVAL_CANDIDATE_ENTRY_POINT",
            "HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION",
            "HarnessFailure",
            "HarnessFailureCause",
            "HumanEvalCandidateJobRequest",
            "HumanEvalCandidateJobResult",
            "HumanEvalEvaluatorSuite",
            "HumanEvalFunctionGroupResult",
            "HumanEvalSubmissionRequest",
            "HumanEvalSubmissionResult",
            "HumanEvalSuiteCompleted",
            "HumanEvalSuiteHarnessFailure",
            "HumanEvalSuiteResult",
            "SubmissionOutcome",
            "evaluate_humaneval_candidate_job",
            "project_humaneval_submission",
            "project_humaneval_submissions_batch",
            "score_humaneval_submission",
            "score_humaneval_submissions_batch",
        }
    ),
    "dr_code.metrics": frozenset(
        {
            "MeasuredRecord",
            "MetricRecord",
            "MetricValue",
            "MetricValueCoordinate",
            "MetricValueUnit",
            "NotApplicableRecord",
            "OperatorFailureRecord",
        }
    ),
}


def _internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path)
    relative_parent = path.relative_to(DR_CODE_PACKAGE).parent
    current_package = ".".join(("dr_code", *relative_parent.parts))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("dr_code.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                module = resolve_name(relative_name, current_package)
            else:
                module = node.module or ""
            if module.startswith("dr_code"):
                imports.add(module)
                imports.update(
                    f"{module}.{alias.name}" for alias in node.names
                )
    return imports


def test_python_root_contains_only_core_and_functional_packages() -> None:
    assert {
        path.name
        for path in DR_CODE_PACKAGE.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    } == {"core", *FUNCTIONAL_PACKAGES}
    assert {path.name for path in DR_CODE_PACKAGE.glob("*.py")} == {
        "__init__.py"
    }


def test_core_does_not_import_functional_packages() -> None:
    forbidden = tuple(f"dr_code.{name}" for name in FUNCTIONAL_PACKAGES)
    violations = {
        path.relative_to(DR_CODE_PACKAGE): sorted(
            target
            for target in _internal_imports(path)
            if any(
                target == prefix or target.startswith(f"{prefix}.")
                for prefix in forbidden
            )
        )
        for path in sorted(CORE_PACKAGE.rglob("*.py"))
    }
    assert not {
        path: imports for path, imports in violations.items() if imports
    }


def test_eval_core_public_exports_are_complete() -> None:
    for module_name, expected in EVAL_CORE_EXPORTS.items():
        module = importlib.import_module(module_name)
        public_names = tuple(module.__all__)
        assert len(public_names) == len(set(public_names))
        exported = frozenset(public_names)
        assert expected <= exported, (
            f"{module_name} is missing exports: {sorted(expected - exported)}"
        )
        assert not {
            name
            for name in public_names
            if getattr(module, name, None) is None
        }

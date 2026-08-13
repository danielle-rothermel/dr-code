#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_SMOKE_PROGRAM = r"""
from importlib.metadata import entry_points, version
import hashlib
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import dr_code.caching
import dr_code.evaluation
import dr_code.metrics
import dr_code.preprocessing
import dr_code.synthetic
import dr_code.trace
from dr_exec import ProcessExecutor
from dr_serialize import Sha256Digest
from dr_code.core.execution.executor import host_process_executor
from dr_code.evaluation import (
    EVAL_BUNDLE_FORMAT,
    EVAL_PROJECTION_FORMAT,
    EvalCandidateIdentity,
    EvalBundlePayload,
    EvalAttemptIdentity,
    EvalSampleIdentity,
    MaterializedEvalCandidate,
    ProjectionArtifactHeader,
    ProjectionKind,
    ReplayReady,
    StructuralEvalComparison,
    audit_eval_bundle,
    compare_eval_attempts,
    evaluate_batch,
    evaluate_durable_partition,
    preflight_replay,
    read_eval_projection,
    replay_eval_attempt,
    restore_eval_attempt,
)
from dr_code.humaneval import (
    HumanEvalCandidateJobRequest,
    HumanEvalCandidateJobResult,
    HumanEvalEvaluatorSuite,
    HumanEvalTask,
    evaluate_humaneval_candidate_job,
)
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.metrics import MetricName, MetricQuestionCoordinate
from dr_code.metrics.coordinates import question_settings
from dr_code.trace import CodeArtifact, PreprocessingDefinitionCoordinate

required_exports = {
    "dr_code.caching": {
        "CACHED_EXECUTION_OBSERVATION_SCHEMA_VERSION",
        "EXECUTION_CACHE_NAMESPACE",
        "EXECUTION_CACHE_RECORD_SCHEMA",
        "CachedExecutionObservation",
        "ExecutionCacheStats",
        "WindowedExecutionCache",
    },
    "dr_code.evaluation": {
        "AggregationPolicy",
        "AggregationResultProjectionRow",
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
        "EvalAttemptIdentity",
        "EvalAttemptRecord",
        "EvalBatchRequest",
        "EvalBatchResult",
        "EvalBundleAudit",
        "EvalBundlePayload",
        "EvalCandidateIdentity",
        "EvalEvidenceResolver",
        "EvalInput",
        "EvalMemberRecord",
        "EvalProjectionReference",
        "EvalReadLimits",
        "EvalRuntimeIdentity",
        "EvalSample",
        "EvalSampleAuxiliaryArtifact",
        "EvalSampleIdentity",
        "EvalSampleMetadata",
        "EvalSampleProjectionRow",
        "EvalSampleProvenance",
        "EvalSlotIdentity",
        "EvalSourceIdentity",
        "EvaluatedSampleRecord",
        "EvidenceReference",
        "ExecutedCandidateProvenance",
        "ExecutorExecutionFailure",
        "FrozenCandidateEvalInput",
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
        "SampleEvalInput",
        "SampleEvalRecord",
        "Score",
        "ScoreProjectionRow",
        "ShardLimits",
        "StoredRecordReference",
        "StructuralEvalComparison",
        "StructuralRecordComparison",
        "SyntheticSampleProvenance",
        "WindowLimits",
        "audit_eval_bundle",
        "compare_eval_attempts",
        "evaluate_batch",
        "evaluate_durable_partition",
        "preflight_replay",
        "read_eval_projection",
        "replay_eval_attempt",
        "restore_eval_attempt",
    },
    "dr_code.humaneval": {
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
    },
    "dr_code.metrics": {
        "MeasuredRecord",
        "MetricRecord",
        "MetricValue",
        "MetricValueCoordinate",
        "MetricValueUnit",
        "NotApplicableRecord",
        "OperatorFailureRecord",
    },
}

for module_name, expected in required_exports.items():
    module = importlib.import_module(module_name)
    exported = set(module.__all__)
    missing = expected - exported
    if missing:
        raise SystemExit(
            f"installed wheel {module_name} is missing exports: "
            f"{sorted(missing)}"
        )
    missing_attributes = {
        name for name in exported if getattr(module, name, None) is None
    }
    if missing_attributes:
        raise SystemExit(
            f"installed wheel {module_name} has unresolved exports: "
            f"{sorted(missing_attributes)}"
        )

expected_console_scripts = {
    "dr-code-humaneval-schema": "dr_code.humaneval.schema_cli:app",
    "dr-code-synthetic": "dr_code.synthetic.cli:app",
    "dr-code-validate-preprocessing": (
        "dr_code.evaluation.cli:validate_preprocessing_app"
    ),
    "dr-code-validate-testing": "dr_code.evaluation.cli:validate_testing_app",
}
installed_console_scripts = {
    entry.name: entry.value
    for entry in entry_points(group="console_scripts")
    if entry.name in expected_console_scripts
}
if installed_console_scripts != expected_console_scripts:
    raise SystemExit(
        "installed wheel console scripts do not match the declared verbs: "
        f"{sorted(installed_console_scripts.items())}"
    )
for script_name, target in expected_console_scripts.items():
    target_module, _, target_attribute = target.partition(":")
    resolved = getattr(
        importlib.import_module(target_module), target_attribute, None
    )
    if resolved is None:
        raise SystemExit(
            f"installed wheel console script {script_name} does not resolve "
            f"{target}"
        )

expected_version = sys.argv[1]
installed_version = version("dr-code")
if installed_version != expected_version:
    raise SystemExit(
        f"installed dr-code version {installed_version!r} does not match "
        f"{expected_version!r}"
    )
if not callable(evaluate_humaneval_candidate_job):
    raise SystemExit("installed wheel is missing the HumanEval entry point")
if not all(
    callable(value)
    for value in (
        evaluate_batch,
        evaluate_durable_partition,
        read_eval_projection,
        restore_eval_attempt,
        audit_eval_bundle,
        preflight_replay,
        replay_eval_attempt,
        compare_eval_attempts,
    )
):
    raise SystemExit("installed wheel is missing the evaluation bundle API")
if ReplayReady is None or StructuralEvalComparison is None:
    raise SystemExit("installed wheel is missing replay or comparison models")
bundle_attempt = EvalAttemptIdentity(
    attempt_id="00000000-0000-0000-0000-000000000001"
)
if (
    EvalBundlePayload(attempt=bundle_attempt, projections=()).format
    != EVAL_BUNDLE_FORMAT
    or ProjectionArtifactHeader(
        source_attempt=bundle_attempt,
        kind=ProjectionKind.SCORES,
    ).format
    != EVAL_PROJECTION_FORMAT
):
    raise SystemExit("installed wheel evaluation wire constants disagree")

with TemporaryDirectory(prefix="dr-code-wheel-records-") as record_root:
    executor = host_process_executor(
        Path(record_root),
        runtime_executable=Path(sys.executable),
    )
    if not isinstance(executor, ProcessExecutor):
        raise SystemExit("production executor is not a ProcessExecutor")
    task = HumanEvalTask(
        task_id="wheel-smoke",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=(
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            "    results = [2]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    )
    source = CodeArtifact(source="def add_one(x):\n    return x + 1\n")
    settings = CodeTestSettings()
    request = HumanEvalCandidateJobRequest(
        candidate=MaterializedEvalCandidate(
            identity=EvalCandidateIdentity(
                sample=EvalSampleIdentity(sample_id="wheel-smoke"),
                preprocessing=PreprocessingDefinitionCoordinate(
                    definition_id="wheel-smoke",
                    version="1",
                    steps=(),
                ),
                candidate_ordinal=0,
            ),
            source=source,
            source_sha256=Sha256Digest(
                hashlib.sha256(source.source.encode()).hexdigest()
            ),
        ),
        suites=(
            HumanEvalEvaluatorSuite(
                question=MetricQuestionCoordinate(
                    metric=MetricName.CODE_TEST,
                    on_key="output",
                    settings=question_settings(settings),
                ),
                task=task,
                settings=settings,
            ),
        ),
    )
    result = HumanEvalCandidateJobResult.model_validate(
        evaluate_humaneval_candidate_job(
            request.model_dump(mode="json", exclude_computed_fields=True)
        )
    )
    if result.namespace.kind != "loaded" or len(result.suites) != 1:
        raise SystemExit(f"installed-wheel execution failed: {result!r}")

print(f"installed wheel smoke passed for dr-code {installed_version}")
"""


def _project_version() -> str:
    with (_ROOT / "pyproject.toml").open("rb") as file:
        document = tomllib.load(file)
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml has no project table")
    version = project.get("version")
    if not isinstance(version, str):
        raise ValueError("pyproject.toml project version must be a string")
    return version


def _built_wheel(version: str) -> Path:
    wheels = tuple((_ROOT / "dist").glob(f"dr_code-{version}-*.whl"))
    if len(wheels) != 1:
        raise ValueError(
            f"expected exactly one built wheel for dr-code {version}, "
            f"found {len(wheels)}"
        )
    return wheels[0]


def main() -> int:
    version = _project_version()
    wheel = _built_wheel(version)
    with tempfile.TemporaryDirectory(
        prefix="dr-code-installed-wheel-"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        environment = temporary_root / "venv"
        subprocess.run(
            [
                "uv",
                "venv",
                "--python",
                sys.executable,
                str(environment),
            ],
            check=True,
        )
        environment_python = environment / "bin" / "python"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment_python),
                "--no-cache",
                str(wheel),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(environment_python),
                "-I",
                "-c",
                _SMOKE_PROGRAM,
                version,
            ],
            cwd=temporary_root,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

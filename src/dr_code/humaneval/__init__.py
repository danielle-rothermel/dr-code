from __future__ import annotations

from typing import TYPE_CHECKING

from dr_code.humaneval.acceptance import (
    CodeExtractionResult,
    accept_first_surviving,
    extract_humaneval_code,
    humaneval_runner,
)
from dr_code.humaneval.parsed_tests import HumanEvalTestCaseKind
from dr_code.humaneval.plus_dataset import (
    HumanEvalPlusTask,
    load_humaneval_plus,
)
from dr_code.humaneval.profiles import (
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    HUMANEVAL_METRICS_PROFILE,
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
    HumanEvalMetricsProfile,
    resolve_humaneval_scoring_profile,
)
from dr_code.humaneval.sampling import (
    SampledHumanEvalTask,
    load_humaneval_rows,
    sample_humaneval_tasks,
    sample_humaneval_tasks_from_rows,
)
from dr_code.humaneval.task import (
    EvaluationCaseStatus,
    EvaluationCaseSummary,
    EvaluationTaskSummary,
    HUMANEVAL_OVERRIDE_SET,
    HUMANEVAL_OVERRIDE_SET_ID,
    HUMANEVAL_OVERRIDE_SET_VERSION,
    HumanEvalOverrideEntry,
    HumanEvalOverrideSetCoordinate,
    HumanEvalTask,
    parse_humaneval_dataset,
)

_JOB_EXPORTS = frozenset(
    {
        "CandidateNamespaceFailure",
        "CandidateNamespaceLoaded",
        "CandidateNamespaceOutcome",
        "HUMANEVAL_CANDIDATE_ENTRY_POINT",
        "HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION",
        "HumanEvalCandidateJobRequest",
        "HumanEvalCandidateJobResult",
        "HumanEvalEvaluatorSuite",
        "HumanEvalFunctionGroupResult",
        "HumanEvalSuiteCompleted",
        "HumanEvalSuiteHarnessFailure",
        "HumanEvalSuiteResult",
        "evaluate_humaneval_candidate_job",
    }
)

if TYPE_CHECKING:
    from dr_code.humaneval.projection import (
        CompletedSubmissionResult,
        HarnessFailure,
        HarnessFailureCause,
        HumanEvalSubmissionRequest,
        HumanEvalSubmissionResult,
        SubmissionOutcome,
        project_humaneval_submission,
        project_humaneval_submissions_batch,
        score_humaneval_submission,
        score_humaneval_submissions_batch,
    )

_PROJECTION_EXPORTS = frozenset(
    {
        "CompletedSubmissionResult",
        "HarnessFailure",
        "HarnessFailureCause",
        "HumanEvalSubmissionRequest",
        "HumanEvalSubmissionResult",
        "SubmissionOutcome",
        "project_humaneval_submission",
        "project_humaneval_submissions_batch",
        "score_humaneval_submission",
        "score_humaneval_submissions_batch",
    }
)


def __getattr__(name: str) -> object:
    if name in _JOB_EXPORTS:
        from dr_code.humaneval import job as module
    elif name in _PROJECTION_EXPORTS:
        from dr_code.humaneval import projection as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _JOB_EXPORTS | _PROJECTION_EXPORTS)


__all__ = (
    "CodeExtractionResult",
    "CandidateNamespaceFailure",
    "CandidateNamespaceLoaded",
    "CandidateNamespaceOutcome",
    "CompletedSubmissionResult",
    "DEFAULT_HUMANEVAL_SCORING_PROFILE",
    "EvaluationCaseStatus",
    "EvaluationCaseSummary",
    "EvaluationTaskSummary",
    "HUMANEVAL_SCORING_PROFILE_ID",
    "HUMANEVAL_SCORING_PROFILE_VERSION",
    "HUMANEVAL_METRICS_PROFILE",
    "HUMANEVAL_CANDIDATE_ENTRY_POINT",
    "HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION",
    "HUMANEVAL_OVERRIDE_SET",
    "HUMANEVAL_OVERRIDE_SET_ID",
    "HUMANEVAL_OVERRIDE_SET_VERSION",
    "HarnessFailure",
    "HarnessFailureCause",
    "HumanEvalScoringProfile",
    "HumanEvalMetricsProfile",
    "HumanEvalCandidateJobRequest",
    "HumanEvalCandidateJobResult",
    "HumanEvalEvaluatorSuite",
    "HumanEvalFunctionGroupResult",
    "HumanEvalOverrideEntry",
    "HumanEvalOverrideSetCoordinate",
    "HumanEvalPlusTask",
    "HumanEvalSubmissionResult",
    "HumanEvalSubmissionRequest",
    "HumanEvalTask",
    "HumanEvalSuiteCompleted",
    "HumanEvalSuiteHarnessFailure",
    "HumanEvalSuiteResult",
    "HumanEvalTestCaseKind",
    "SampledHumanEvalTask",
    "SubmissionOutcome",
    "accept_first_surviving",
    "extract_humaneval_code",
    "humaneval_runner",
    "load_humaneval_rows",
    "load_humaneval_plus",
    "parse_humaneval_dataset",
    "project_humaneval_submission",
    "project_humaneval_submissions_batch",
    "resolve_humaneval_scoring_profile",
    "sample_humaneval_tasks",
    "sample_humaneval_tasks_from_rows",
    "score_humaneval_submission",
    "score_humaneval_submissions_batch",
    "evaluate_humaneval_candidate_job",
)

from __future__ import annotations

from typing import TYPE_CHECKING

from drc_humaneval.acceptance import (
    CodeExtractionResult,
    accept_first_surviving,
    extract_humaneval_code,
    humaneval_runner,
)
from drc_humaneval.parsed_tests import HumanEvalTestCaseKind
from drc_humaneval.plus_dataset import (
    HumanEvalPlusTask,
    load_humaneval_plus,
)
from drc_humaneval.profiles import (
    ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE,
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    HUMANEVAL_ANY_CANDIDATE_SCORING_PROFILE_ID,
    HUMANEVAL_ANY_CANDIDATE_SCORING_PROFILE_VERSION,
    HUMANEVAL_METRICS_PROFILE,
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    CandidateReduction,
    HumanEvalScoringProfile,
    HumanEvalMetricsProfile,
    resolve_humaneval_scoring_profile,
)
from drc_humaneval.sampling import (
    SampledHumanEvalTask,
    load_humaneval_rows,
    sample_humaneval_tasks,
    sample_humaneval_tasks_from_rows,
)
from drc_humaneval.task import (
    EvalCaseStatus,
    EvalCaseSummary,
    EvalTaskSummary,
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
    from drc_humaneval.projection import (
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
        from drc_humaneval import job as module
    elif name in _PROJECTION_EXPORTS:
        from drc_humaneval import projection as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _JOB_EXPORTS | _PROJECTION_EXPORTS)


__all__ = (
    "ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE",
    "CandidateReduction",
    "CodeExtractionResult",
    "CandidateNamespaceFailure",
    "CandidateNamespaceLoaded",
    "CandidateNamespaceOutcome",
    "CompletedSubmissionResult",
    "DEFAULT_HUMANEVAL_SCORING_PROFILE",
    "EvalCaseStatus",
    "EvalCaseSummary",
    "EvalTaskSummary",
    "HUMANEVAL_ANY_CANDIDATE_SCORING_PROFILE_ID",
    "HUMANEVAL_ANY_CANDIDATE_SCORING_PROFILE_VERSION",
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

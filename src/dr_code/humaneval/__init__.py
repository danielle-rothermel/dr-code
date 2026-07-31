from importlib import import_module
from typing import TYPE_CHECKING

from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    STRICT_FIELD_MARKER_PARSER_PROFILE,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    CodeExtractionResult,
    CodeParserProfile,
    resolve_parser_profile,
)
from dr_code.humaneval.parsed_tests import HumanEvalTestCaseKind
from dr_code.humaneval.profiles import (
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
    resolve_humaneval_scoring_profile,
)
from dr_code.humaneval.sampling import (
    DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256,
    SampledHumanEvalTask,
    load_human_eval_rows,
    run_human_eval_sampling,
    sample_human_eval_tasks,
    sample_human_eval_tasks_from_rows,
)
from dr_code.humaneval.task import (
    EvaluationCaseStatus,
    EvaluationCaseSummary,
    EvaluationTaskSummary,
    HumanEvalTask,
    parse_human_eval_dataset,
)

if TYPE_CHECKING:
    from dr_code.humaneval.scoring import (
        CompletedScore,
        HarnessFailure,
        HarnessFailureCause,
        HumanEvalSubmissionScore,
        SubmissionOutcome,
        evaluation_aggregate_metrics,
        score_humaneval_submission,
    )


_SCORING_EXPORTS = frozenset(
    {
        "CompletedScore",
        "HarnessFailure",
        "HarnessFailureCause",
        "HumanEvalSubmissionScore",
        "SubmissionOutcome",
        "evaluation_aggregate_metrics",
        "score_humaneval_submission",
    }
)


def __getattr__(name: str) -> object:
    """Load scoring exports only when the public facade needs them.

    Preprocessing definitions use parsing coordinates, so importing scoring
    while this package initializes would re-enter the preprocessing package.
    """
    if name not in _SCORING_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("dr_code.humaneval.scoring"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Report the complete public facade without importing scoring."""

    return sorted(set(globals()) | set(__all__))


__all__ = (
    "BEST_EFFORT_HUMANEVAL_PARSER_PROFILE",
    "BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID",
    "CodeExtractionResult",
    "CodeParserProfile",
    "CompletedScore",
    "DEFAULT_HUMANEVAL_SCORING_PROFILE",
    "DEFAULT_HUMANEVAL_TIMEOUT_SECONDS",
    "DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256",
    "EvaluationCaseStatus",
    "EvaluationCaseSummary",
    "EvaluationTaskSummary",
    "HUMANEVAL_SCORING_PROFILE_ID",
    "HUMANEVAL_SCORING_PROFILE_VERSION",
    "HarnessFailure",
    "HarnessFailureCause",
    "HumanEvalScoringProfile",
    "HumanEvalSubmissionScore",
    "HumanEvalTask",
    "HumanEvalTestCaseKind",
    "PARSER_PROFILE_VERSION",
    "STRICT_FIELD_MARKER_PARSER_PROFILE",
    "STRICT_FIELD_MARKER_PARSER_PROFILE_ID",
    "SampledHumanEvalTask",
    "SubmissionOutcome",
    "evaluation_aggregate_metrics",
    "load_human_eval_rows",
    "parse_human_eval_dataset",
    "resolve_humaneval_scoring_profile",
    "resolve_parser_profile",
    "run_human_eval_sampling",
    "sample_human_eval_tasks",
    "sample_human_eval_tasks_from_rows",
    "score_humaneval_submission",
)

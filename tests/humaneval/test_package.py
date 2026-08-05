"""Tests for the public HumanEval package facade."""

from __future__ import annotations

import dr_code.humaneval as humaneval


EXPECTED_HUMANEVAL_PUBLIC_API = {
    "CodeExtractionResult",
    "CompletedScore",
    "DEFAULT_HUMANEVAL_SCORING_PROFILE",
    "DEFAULT_HUMANEVAL_TIMEOUT_SECONDS",
    "EvaluationCaseStatus",
    "EvaluationCaseSummary",
    "EvaluationTaskSummary",
    "HUMANEVAL_SCORING_PROFILE_ID",
    "HUMANEVAL_SCORING_PROFILE_VERSION",
    "HUMANEVAL_METRICS_PROFILE",
    "HUMANEVAL_OVERRIDE_SET",
    "HUMANEVAL_OVERRIDE_SET_ID",
    "HUMANEVAL_OVERRIDE_SET_VERSION",
    "HarnessFailure",
    "HarnessFailureCause",
    "HumanEvalScoringProfile",
    "HumanEvalMetricsProfile",
    "HumanEvalOverrideEntry",
    "HumanEvalOverrideSetCoordinate",
    "HumanEvalPlusTask",
    "HumanEvalSubmissionScore",
    "HumanEvalTask",
    "HumanEvalTestCaseKind",
    "PreprocessingDefinitionReference",
    "SampledHumanEvalTask",
    "SubmissionOutcome",
    "accept_first_surviving",
    "extract_humaneval_code",
    "humaneval_runner",
    "load_humaneval_plus",
    "load_humaneval_rows",
    "parse_humaneval_dataset",
    "resolve_humaneval_scoring_profile",
    "sample_humaneval_tasks",
    "sample_humaneval_tasks_from_rows",
    "score_humaneval_submission",
}


def test_humaneval_public_api_is_curated() -> None:
    assert set(humaneval.__all__) == EXPECTED_HUMANEVAL_PUBLIC_API

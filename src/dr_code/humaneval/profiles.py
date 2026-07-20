from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictStr

from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)

HUMANEVAL_METRICS_PROFILE_ID = "humaneval-metrics"
HUMANEVAL_METRICS_PROFILE_VERSION = "v1"
HUMANEVAL_SCORING_PROFILE_ID = "humaneval"
HUMANEVAL_SCORING_PROFILE_VERSION = "v3"
DEFAULT_HUMANEVAL_TIMEOUT_SECONDS = 2.0


class HumanEvalScoringProfile(BaseModel):
    """Stable scoring settings and the declared preprocessing coordinates."""

    model_config = ConfigDict(extra="forbid")

    profile_id: StrictStr
    version: StrictStr
    preprocessing_definition_id: StrictStr
    preprocessing_definition_version: StrictStr
    timeout_seconds: StrictFloat
    metrics_profile_id: StrictStr
    metrics_profile_version: StrictStr


DEFAULT_HUMANEVAL_SCORING_PROFILE = HumanEvalScoringProfile(
    profile_id=HUMANEVAL_SCORING_PROFILE_ID,
    version=HUMANEVAL_SCORING_PROFILE_VERSION,
    preprocessing_definition_id=HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
    preprocessing_definition_version=(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.version
    ),
    timeout_seconds=DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    metrics_profile_id=HUMANEVAL_METRICS_PROFILE_ID,
    metrics_profile_version=HUMANEVAL_METRICS_PROFILE_VERSION,
)


def resolve_humaneval_scoring_profile(
    *,
    scoring_profile_id: str,
    scoring_profile_version: str,
) -> HumanEvalScoringProfile:
    if scoring_profile_id == HUMANEVAL_SCORING_PROFILE_ID:
        if scoring_profile_version == HUMANEVAL_SCORING_PROFILE_VERSION:
            return DEFAULT_HUMANEVAL_SCORING_PROFILE
    raise ValueError(
        "unsupported HumanEval scoring profile: "
        f"{scoring_profile_id}@{scoring_profile_version}"
    )

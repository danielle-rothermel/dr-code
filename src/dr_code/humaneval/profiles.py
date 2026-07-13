from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictStr

from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    LEGACY_PARSER_PROFILE_VERSION,
    PARSER_PROFILE_VERSION,
    CodeParserProfile,
    resolve_parser_profile,
)
from dr_code.humaneval.metrics import (
    HUMANEVAL_METRICS_PROFILE_ID,
    HUMANEVAL_METRICS_PROFILE_VERSION,
)

HUMANEVAL_SCORING_PROFILE_ID = "humaneval"
LEGACY_HUMANEVAL_SCORING_PROFILE_VERSION = "v1"
HUMANEVAL_SCORING_PROFILE_VERSION = "v2"
DEFAULT_HUMANEVAL_TIMEOUT_SECONDS = 2.0


class HumanEvalScoringProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: StrictStr
    version: StrictStr
    parser_profile: CodeParserProfile
    timeout_seconds: StrictFloat
    metrics_profile_id: StrictStr
    metrics_profile_version: StrictStr


LEGACY_HUMANEVAL_SCORING_PROFILE = HumanEvalScoringProfile(
    profile_id=HUMANEVAL_SCORING_PROFILE_ID,
    version=LEGACY_HUMANEVAL_SCORING_PROFILE_VERSION,
    parser_profile=resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=LEGACY_PARSER_PROFILE_VERSION,
    ),
    timeout_seconds=DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    metrics_profile_id=HUMANEVAL_METRICS_PROFILE_ID,
    metrics_profile_version=HUMANEVAL_METRICS_PROFILE_VERSION,
)

DEFAULT_HUMANEVAL_SCORING_PROFILE = HumanEvalScoringProfile(
    profile_id=HUMANEVAL_SCORING_PROFILE_ID,
    version=HUMANEVAL_SCORING_PROFILE_VERSION,
    parser_profile=resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
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
        if scoring_profile_version == LEGACY_HUMANEVAL_SCORING_PROFILE_VERSION:
            return LEGACY_HUMANEVAL_SCORING_PROFILE
    raise ValueError(
        "unsupported HumanEval scoring profile: "
        f"{scoring_profile_id}@{scoring_profile_version}"
    )

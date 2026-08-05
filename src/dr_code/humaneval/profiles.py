from __future__ import annotations

from types import MappingProxyType

from pydantic import StrictFloat, StrictStr

from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
)
from dr_code.core.models import FrozenModel

HUMANEVAL_METRICS_PROFILE_ID = "humaneval-metrics"
HUMANEVAL_METRICS_PROFILE_VERSION = "0"
HUMANEVAL_SCORING_PROFILE_ID = "humaneval"
HUMANEVAL_SCORING_PROFILE_VERSION = "0"
DEFAULT_HUMANEVAL_TIMEOUT_SECONDS = 2.0


class PreprocessingDefinitionReference(FrozenModel):
    definition_id: StrictStr
    version: StrictStr


class HumanEvalScoringProfile(FrozenModel):
    profile_id: StrictStr
    version: StrictStr
    preprocessing_definition: PreprocessingDefinitionReference
    timeout_seconds: StrictFloat
    metrics_profile: HumanEvalMetricsProfile


class HumanEvalMetricsProfile(FrozenModel):
    profile_id: StrictStr
    version: StrictStr
    passed_score: StrictFloat
    failed_score: StrictFloat


HUMANEVAL_METRICS_PROFILE = HumanEvalMetricsProfile(
    profile_id=HUMANEVAL_METRICS_PROFILE_ID,
    version=HUMANEVAL_METRICS_PROFILE_VERSION,
    passed_score=1.0,
    failed_score=0.0,
)


DEFAULT_HUMANEVAL_SCORING_PROFILE = HumanEvalScoringProfile(
    profile_id=HUMANEVAL_SCORING_PROFILE_ID,
    version=HUMANEVAL_SCORING_PROFILE_VERSION,
    preprocessing_definition=PreprocessingDefinitionReference(
        definition_id=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
        version=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
    ),
    timeout_seconds=DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    metrics_profile=HUMANEVAL_METRICS_PROFILE,
)

_SCORING_PROFILES = MappingProxyType(
    {
        (
            DEFAULT_HUMANEVAL_SCORING_PROFILE.profile_id,
            DEFAULT_HUMANEVAL_SCORING_PROFILE.version,
        ): DEFAULT_HUMANEVAL_SCORING_PROFILE,
    }
)


def resolve_humaneval_scoring_profile(
    *,
    scoring_profile_id: str,
    scoring_profile_version: str,
) -> HumanEvalScoringProfile:
    profile = _SCORING_PROFILES.get(
        (scoring_profile_id, scoring_profile_version)
    )
    if profile is not None:
        return profile
    raise ValueError(
        "unsupported HumanEval scoring profile: "
        f"{scoring_profile_id}@{scoring_profile_version}"
    )

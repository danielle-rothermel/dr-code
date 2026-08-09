from __future__ import annotations

from types import MappingProxyType

from pydantic import StrictFloat, StrictStr

from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    bind_external_preprocessing,
)
from dr_code.core.models import FrozenModel
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.metrics import MetricName, MetricQuestionCoordinate
from dr_code.metrics.coordinates import question_settings
from dr_code.trace import (
    ExternalPreprocessingTraceProducer,
    PreprocessingDefinitionCoordinate,
)

HUMANEVAL_METRICS_PROFILE_ID = "humaneval-metrics"
HUMANEVAL_METRICS_PROFILE_VERSION = "0"
HUMANEVAL_SCORING_PROFILE_ID = "humaneval"
HUMANEVAL_SCORING_PROFILE_VERSION = "0"


class HumanEvalScoringProfile(FrozenModel):
    profile_id: StrictStr
    version: StrictStr
    preprocessing_definition: PreprocessingDefinitionCoordinate
    question: MetricQuestionCoordinate
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

_DEFAULT_PREPROCESSING_PRODUCER = bind_external_preprocessing(
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
).producer
if not isinstance(
    _DEFAULT_PREPROCESSING_PRODUCER,
    ExternalPreprocessingTraceProducer,
):
    raise AssertionError("external preprocessing must preserve its coordinate")


DEFAULT_HUMANEVAL_SCORING_PROFILE = HumanEvalScoringProfile(
    profile_id=HUMANEVAL_SCORING_PROFILE_ID,
    version=HUMANEVAL_SCORING_PROFILE_VERSION,
    preprocessing_definition=_DEFAULT_PREPROCESSING_PRODUCER.definition,
    question=MetricQuestionCoordinate(
        metric=MetricName.CODE_TEST,
        on_key="output",
        settings=question_settings(CodeTestSettings()),
    ),
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

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.humaneval.profiles import (
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    HUMANEVAL_METRICS_PROFILE_VERSION,
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    resolve_humaneval_scoring_profile,
)
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
    bind_external_preprocessing,
    resolve_preprocessing_definition,
)
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.metrics import MetricName


def test_default_scoring_profile_uses_declared_component_versions() -> None:
    profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
    )

    assert profile is DEFAULT_HUMANEVAL_SCORING_PROFILE
    assert profile.version == HUMANEVAL_SCORING_PROFILE_VERSION
    assert profile.preprocessing_definition.version == (
        EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION
    )
    assert profile.metrics_profile.version == HUMANEVAL_METRICS_PROFILE_VERSION
    assert (
        profile.preprocessing_definition
        == bind_external_preprocessing(
            EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
        ).producer.definition
    )
    assert profile.question.metric is MetricName.CODE_TEST
    assert profile.question.on_key == "output"
    assert {setting.name for setting in profile.question.settings} == {
        "task_key"
    }
    assert "timeout_seconds" not in profile.model_dump(mode="json")


def test_code_test_settings_reject_a_false_per_suite_timeout_claim() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        CodeTestSettings.model_validate({"timeout_seconds": 5.0})


def test_scoring_profile_names_a_resolvable_preprocessing_definition() -> None:
    reference = DEFAULT_HUMANEVAL_SCORING_PROFILE.preprocessing_definition
    assert reference.definition_id == (
        EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID
    )
    definition = resolve_preprocessing_definition(
        definition_id=reference.definition_id,
        version=reference.version,
    )
    assert definition.definition_id == reference.definition_id


def test_scoring_profile_resolver_rejects_unregistered_coordinates() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported HumanEval scoring profile",
    ):
        resolve_humaneval_scoring_profile(
            scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
            scoring_profile_version="1",
        )


def test_profiles_are_immutable() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        DEFAULT_HUMANEVAL_SCORING_PROFILE.version = "1"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        DEFAULT_HUMANEVAL_SCORING_PROFILE.preprocessing_definition.version = (  # type: ignore[misc]
            "1"
        )

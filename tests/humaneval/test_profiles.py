from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_VERSION,
    CodeParserProfile,
    extract_code_with_profile,
    resolve_parser_profile,
)
from dr_code.humaneval.profiles import (
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    HUMANEVAL_METRICS_PROFILE_VERSION,
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    resolve_humaneval_scoring_profile,
)


def test_default_scoring_profile_uses_declared_component_versions() -> None:
    profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
    )

    assert profile is DEFAULT_HUMANEVAL_SCORING_PROFILE
    assert profile.version == HUMANEVAL_SCORING_PROFILE_VERSION
    assert (
        profile.parser_profile.version
        == BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_VERSION
    )
    assert profile.metrics_profile.version == HUMANEVAL_METRICS_PROFILE_VERSION


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
        BEST_EFFORT_HUMANEVAL_PARSER_PROFILE.version = "1"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        DEFAULT_HUMANEVAL_SCORING_PROFILE.version = "1"  # type: ignore[misc]


def test_profile_resolvers_are_stable() -> None:
    first_parser = resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_VERSION,
    )
    second_parser = resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_VERSION,
    )
    first_scoring = resolve_humaneval_scoring_profile(
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
    )
    second_scoring = resolve_humaneval_scoring_profile(
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
    )

    assert first_parser is second_parser
    assert first_scoring is second_scoring


@pytest.mark.parametrize(
    "profile",
    [
        CodeParserProfile(
            profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
            version="stale",
        ),
        CodeParserProfile(profile_id="unregistered", version="0"),
    ],
)
def test_execution_rejects_unregistered_or_stale_profiles(
    profile: CodeParserProfile,
) -> None:
    with pytest.raises(ValueError, match="unsupported parser profile"):
        extract_code_with_profile("def f():\n    return 1\n", profile=profile)

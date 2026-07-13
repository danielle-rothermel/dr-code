from __future__ import annotations

from dr_code.humaneval.code_parsing import (
    LEGACY_PARSER_PROFILE_VERSION,
    PARSER_PROFILE_VERSION,
)
from dr_code.humaneval.profiles import (
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    LEGACY_HUMANEVAL_SCORING_PROFILE_VERSION,
    resolve_humaneval_scoring_profile,
)


def test_persisted_v1_scoring_profile_keeps_parser_v1() -> None:
    profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=LEGACY_HUMANEVAL_SCORING_PROFILE_VERSION,
    )

    assert profile.version == "v1"
    assert profile.parser_profile.version == LEGACY_PARSER_PROFILE_VERSION


def test_default_v2_scoring_profile_uses_parser_v2() -> None:
    profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
    )

    assert profile is DEFAULT_HUMANEVAL_SCORING_PROFILE
    assert profile.version == "v2"
    assert profile.parser_profile.version == PARSER_PROFILE_VERSION

from __future__ import annotations

import pytest

from dr_code.humaneval.profiles import (
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    resolve_humaneval_scoring_profile,
)


def test_default_v3_scoring_profile_uses_official_preprocessing() -> None:
    profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
    )

    assert profile is DEFAULT_HUMANEVAL_SCORING_PROFILE
    assert profile.version == "v3"
    assert profile.preprocessing_definition_id == "humaneval-function-candidates"
    assert profile.preprocessing_definition_version == "v1"


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_legacy_scoring_profiles_are_not_resolvable(version: str) -> None:
    with pytest.raises(ValueError, match="unsupported HumanEval scoring profile"):
        resolve_humaneval_scoring_profile(
            scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
            scoring_profile_version=version,
        )

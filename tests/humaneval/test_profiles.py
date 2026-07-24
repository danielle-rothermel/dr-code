from __future__ import annotations

import pytest

from dr_code.humaneval.code_parsing import (
    PARSER_PROFILE_VERSION,
)
from dr_code.humaneval.profiles import (
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    resolve_humaneval_scoring_profile,
)


def test_legacy_scoring_profile_is_not_resolvable() -> None:
    with pytest.raises(ValueError, match="unsupported HumanEval scoring"):
        resolve_humaneval_scoring_profile(
            scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
            scoring_profile_version="v1",
        )


def test_default_v2_scoring_profile_uses_parser_v2() -> None:
    profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=HUMANEVAL_SCORING_PROFILE_ID,
        scoring_profile_version=HUMANEVAL_SCORING_PROFILE_VERSION,
    )

    assert profile is DEFAULT_HUMANEVAL_SCORING_PROFILE
    assert profile.version == "v2"
    assert profile.parser_profile.version == PARSER_PROFILE_VERSION

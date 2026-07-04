"""Golden identity tests for the ported v1 parser/scoring profiles.

The fixture was captured in whetstone-ai before extraction (its Stage 0
golden baselines). The port must reproduce every extraction and scoring
output byte-for-byte under the existing v1 profile IDs. If one of these
fails, fix the port — never the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    STRICT_FIELD_MARKER_PARSER_PROFILE,
    CodeParserProfile,
    extract_code_with_profile,
)
from dr_code.humaneval.profiles import (
    resolve_humaneval_scoring_profile,
)
from dr_code.humaneval.scoring import score_humaneval_generation
from dr_code.humaneval.task import HumanEvalTask

GOLDEN_FIXTURE = (
    Path(__file__).parent / "fixtures" / "parser_scoring_golden.json"
)

PROFILES: dict[str, CodeParserProfile] = {
    "best_effort": BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    "field_marker": STRICT_FIELD_MARKER_PARSER_PROFILE,
}


def golden() -> dict[str, Any]:
    return json.loads(GOLDEN_FIXTURE.read_text())


def golden_task() -> HumanEvalTask:
    return HumanEvalTask.model_validate(golden()["task"])


def extraction_case_ids() -> list[str]:
    return sorted(golden()["extraction"])


def scoring_case_ids() -> list[str]:
    return sorted(golden()["scoring"])


@pytest.mark.parametrize("case_name", extraction_case_ids())
@pytest.mark.parametrize("profile_name", sorted(PROFILES))
def test_golden_extraction_reproduces(
    case_name: str, profile_name: str
) -> None:
    case = golden()["extraction"][case_name]
    result = extract_code_with_profile(
        case["raw_generation"],
        profile=PROFILES[profile_name],
    )
    assert result.model_dump(mode="json") == case["profiles"][profile_name]


@pytest.mark.parametrize("case_name", scoring_case_ids())
def test_golden_scoring_reproduces(case_name: str) -> None:
    stored = golden()
    scoring_profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=stored["scoring_profile"]["profile_id"],
        scoring_profile_version=stored["scoring_profile"]["version"],
    )
    assert (
        scoring_profile.model_dump(mode="json")
        == stored["scoring_profile"]
    )
    case = stored["scoring"][case_name]
    # The fixture's raw generations are all strings, for which whetstone's
    # injected recordable_text is a passthrough.
    score = score_humaneval_generation(
        raw_generation=case["raw_generation"],
        task=golden_task(),
        parser_profile=scoring_profile.parser_profile,
        timeout_seconds=scoring_profile.timeout_seconds,
        recordable_text=str,
    )
    assert score.model_dump(mode="json") == case["score"]

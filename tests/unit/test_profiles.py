"""Unit tests for OpenRouter profile resolution."""

from __future__ import annotations

import pytest
from dr_providers import ReasoningSpec, SamplingControls
from dr_providers.names import EffortLevel

from dr_code.generation.profiles import (
    default_profiles_path,
    list_profile_ids,
    resolve_profile,
)


def test_list_profile_ids_includes_all_demo_profiles() -> None:
    profile_ids = list_profile_ids(default_profiles_path())
    assert len(profile_ids) == 8
    assert "openrouter/google/gemini-3.1-flash-lite/off/v1" in profile_ids
    assert "openrouter/openai/gpt-oss-20b/low/v1" in profile_ids


@pytest.mark.parametrize(
    ("profile_id", "model", "reasoning", "sampling"),
    [
        (
            "openrouter/google/gemini-3.1-flash-lite/off/v1",
            "google/gemini-3.1-flash-lite",
            ReasoningSpec(enabled=False),
            SamplingControls(temperature=0.7, top_p=0.95),
        ),
        (
            "openrouter/openai/gpt-oss-20b/low/v1",
            "openai/gpt-oss-20b",
            ReasoningSpec(effort=EffortLevel.LOW),
            SamplingControls(temperature=0.7, top_p=0.95),
        ),
        (
            "openrouter/xiaomi/mimo-v2.5/off/v1",
            "xiaomi/mimo-v2.5",
            ReasoningSpec(enabled=False),
            SamplingControls(temperature=0.7, top_p=0.95),
        ),
    ],
)
def test_resolve_profile_maps_yaml_to_dr_providers(
    profile_id: str,
    model: str,
    reasoning: ReasoningSpec,
    sampling: SamplingControls,
) -> None:
    resolved = resolve_profile(profile_id, profiles_path=default_profiles_path())
    assert resolved.profile_id == profile_id
    assert resolved.model == model
    assert resolved.reasoning == reasoning
    assert resolved.sampling == sampling


def test_resolve_profile_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        resolve_profile(
            "openrouter/does/not/exist/off/v1",
            profiles_path=default_profiles_path(),
        )

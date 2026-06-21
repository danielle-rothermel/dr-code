"""OpenRouter profile registry for stage 1b generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml
from dr_providers import ReasoningSpec, SamplingControls
from dr_providers.names import EffortLevel

from dr_code.models.base import FrozenModel

DEFAULT_PROFILES_REL_PATH: Final[str] = "configs/openrouter_profiles.yaml"


class ResolvedProfile(FrozenModel):
    """Resolved OpenRouter call settings from a profile id."""

    profile_id: str
    model: str
    reasoning: ReasoningSpec | None = None
    sampling: SamplingControls | None = None


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def default_profiles_path() -> Path:
    """Return the default OpenRouter profiles YAML path."""
    return _repo_root() / DEFAULT_PROFILES_REL_PATH


def load_profiles(path: Path | str) -> dict[str, Any]:
    """Load raw profiles YAML."""
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        msg = f"Invalid profiles file (expected mapping): {path}"
        raise ValueError(msg)
    return loaded


def list_profile_ids(path: Path | str | None = None) -> list[str]:
    """Return sorted profile ids from the registry."""
    profiles_path = Path(path) if path is not None else default_profiles_path()
    config = load_profiles(profiles_path)
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        msg = f"Invalid profiles section in {profiles_path}"
        raise ValueError(msg)
    return sorted(str(profile_id) for profile_id in profiles)


def resolve_profile(
    profile_id: str,
    *,
    profiles_path: Path | str | None = None,
) -> ResolvedProfile:
    """Resolve a profile id to dr-providers call settings."""
    resolved_path = (
        Path(profiles_path) if profiles_path is not None else default_profiles_path()
    )
    config = load_profiles(resolved_path)
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        msg = f"Invalid profiles section in {resolved_path}"
        raise ValueError(msg)
    if profile_id not in profiles:
        allowed = ", ".join(sorted(str(key) for key in profiles))
        msg = f"Unknown profile: {profile_id}. Expected one of: {allowed}"
        raise ValueError(msg)

    profile_config = profiles[profile_id]
    if not isinstance(profile_config, dict):
        msg = f"Invalid profile config for {profile_id}"
        raise ValueError(msg)

    model = profile_config.get("model")
    if not isinstance(model, str) or not model:
        msg = f"Profile {profile_id} is missing model"
        raise ValueError(msg)

    reasoning = _reasoning_from_profile(profile_config)
    sampling = _sampling_from_config(config)
    return ResolvedProfile(
        profile_id=profile_id,
        model=model,
        reasoning=reasoning,
        sampling=sampling,
    )


def _reasoning_from_profile(profile_config: dict[str, Any]) -> ReasoningSpec | None:
    if profile_config.get("reasoning_disabled"):
        return ReasoningSpec(enabled=False)
    effort = profile_config.get("effort")
    if effort is not None:
        return ReasoningSpec(effort=EffortLevel(str(effort).lower()))
    return None


def _sampling_from_config(config: dict[str, Any]) -> SamplingControls | None:
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        return None
    temperature = defaults.get("temperature")
    top_p = defaults.get("top_p")
    if temperature is None and top_p is None:
        return None
    return SamplingControls(
        temperature=float(temperature) if temperature is not None else None,
        top_p=float(top_p) if top_p is not None else None,
    )

"""Run configuration for stage 1b fresh generation."""

from __future__ import annotations

from pathlib import Path

from dr_code.generation.profiles import default_profiles_path
from dr_code.models.base import FrozenModel


class GenerationRunConfig(FrozenModel):
    """Configuration for a fresh decoder generation batch."""

    run_id: str
    profile_id: str
    profiles_path: Path = default_profiles_path()
    task_ids: list[str] | None = None
    limit: int | None = None
    max_tokens: int | None = None
    prefer_snapshot: bool = True

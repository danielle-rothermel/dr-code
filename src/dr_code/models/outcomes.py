"""Stage 2–3 outcome skeletons (projection logic deferred)."""

from __future__ import annotations

from dr_code.models.base import FrozenModel


class ParseOutcome(FrozenModel):
    """Parse-stage result projection (stage 2)."""

    sample_id: str
    run_id: str | None
    task_id: str
    parse_success: bool
    extracted_code: str | None = None
    skip_reason: str | None = None


class TestOutcome(FrozenModel):
    """Test-stage result projection (stage 3)."""

    sample_id: str
    run_id: str | None
    task_id: str
    parse_success: bool
    skipped: bool = False
    skip_reason: str | None = None
    test_pass_rate: float | None = None
    all_tests_passed: bool | None = None

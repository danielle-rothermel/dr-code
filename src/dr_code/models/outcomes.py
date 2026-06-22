"""Stage 2–3 outcome models."""

from __future__ import annotations

from dr_code.models.base import FrozenModel


class CodeEvalProvenance(FrozenModel):
    """Slim code-eval provenance projected from ValidationResult."""

    config_fingerprint: str
    extractor_path: tuple[str, ...] | None = None
    repairs_applied: tuple[str, ...] | None = None
    extraction_log_summary: tuple[str, ...] | None = None


class ParseOutcome(FrozenModel):
    """Parse-stage result projection (stage 2)."""

    sample_id: str
    run_id: str | None
    task_id: str
    parse_success: bool
    extracted_code: str | None = None
    candidate_count: int = 0
    valid_count: int = 0
    code_eval: CodeEvalProvenance | None = None
    skip_reason: str | None = None
    latency_ms: float | None = None


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

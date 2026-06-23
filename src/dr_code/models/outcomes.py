"""Stage 2–3 outcome models."""

from __future__ import annotations

from typing import Any, Literal

from dr_code.models.base import FrozenModel

TestOutcomeKind = Literal[
    "skipped",
    "tested",
    "infra_error",
    "internal_error",
]


class CodeEvalProvenance(FrozenModel):
    """Slim code-eval provenance projected from ValidationResult."""

    config_fingerprint: str
    selected_candidate_id: str | None = None
    selected_attempt_id: str | None = None
    recovery_attempt_count: int = 0
    extractor_path: tuple[str, ...] | None = None
    repairs_applied: tuple[str, ...] | None = None
    extraction_log_summary: tuple[str, ...] | None = None


class CandidateFunction(FrozenModel):
    """Top-level function discovered in generated code."""

    name: str
    positional_arity: int
    source_order: int
    has_varargs: bool = False


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


class InfraErrorProjection(FrozenModel):
    """Structured infrastructure failure from test execution."""

    stage: str
    execution_mode: str
    detail: str


class TestCaseResultProjection(FrozenModel):
    """Per-case HumanEval+ test result."""

    __test__ = False

    input_value: Any
    expected_output: Any
    actual_output: Any | None = None
    passed: bool = False
    error: str | None = None
    compile_success: bool | None = None
    compile_error: str | None = None


class TestOutcome(FrozenModel):
    """Test-stage result projection (stage 3)."""

    __test__ = False

    sample_id: str
    run_id: str | None
    task_id: str
    parse_success: bool
    outcome_kind: TestOutcomeKind
    skipped: bool = False
    skip_reason: str | None = None
    tests_ran: bool = False
    entry_point: str | None = None
    selected_function_name: str | None = None
    candidate_functions: tuple[CandidateFunction, ...] = ()
    expected_entry_point_present: bool = False
    extracted_code: str | None = None
    test_pass_rate: float | None = None
    all_tests_passed: bool | None = None
    test_case_results: tuple[TestCaseResultProjection, ...] = ()
    infra_error: InfraErrorProjection | None = None
    internal_error: str | None = None
    latency_ms: float | None = None

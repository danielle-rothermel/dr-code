"""Display helpers for test-stage walkthroughs."""

from __future__ import annotations

import json

from dr_code.models.attempts import AttemptRecord
from dr_code.models.outcomes import ParseOutcome, TestOutcome


def format_outcome_banner(outcome: TestOutcome) -> str:
    """Return a single-line outcome summary for manual inspection."""
    if outcome.outcome_kind == "skipped":
        reason = outcome.skip_reason or "skipped"
        return f"[SKIPPED: {reason}]"
    if outcome.outcome_kind == "infra_error":
        stage = outcome.infra_error.stage if outcome.infra_error else "unknown"
        return f"[INFRA ERROR: {stage}]"
    if outcome.outcome_kind == "internal_error":
        detail = outcome.internal_error or "unknown"
        first_line = detail.splitlines()[0]
        return f"[INTERNAL ERROR: {first_line}]"
    if outcome.all_tests_passed:
        return "[ALL TESTS PASSED]"
    passed = sum(1 for case in outcome.test_case_results if case.passed)
    total = len(outcome.test_case_results)
    return f"[TEST FAIL: {passed}/{total} passed]"


def format_test_walkthrough(
    record: AttemptRecord,
    parse_outcome: ParseOutcome,
    outcome: TestOutcome,
    *,
    preview_chars: int = 600,
) -> str:
    """Format attempt + parse + test outcome for manual review."""
    lines = [
        format_outcome_banner(outcome),
        "",
        f"=== Test walkthrough: {record.task_id} ===",
        "",
        "Attempt metadata:",
        f"  sample_id: {record.sample_id}",
        f"  run_id: {outcome.run_id}",
        f"  task_id: {record.task_id}",
        f"  entry_point: {record.entry_point}",
        "",
        "Parse summary:",
        f"  parse_success: {parse_outcome.parse_success}",
        "  extracted_code preview:",
        _indent_block(
            _truncate(outcome.extracted_code or "", preview_chars),
            4,
        ),
        "",
        "Test summary:",
        f"  outcome_kind: {outcome.outcome_kind}",
        f"  tests_ran: {outcome.tests_ran}",
        f"  test_pass_rate: {outcome.test_pass_rate}",
        f"  all_tests_passed: {outcome.all_tests_passed}",
    ]
    if outcome.latency_ms is not None:
        lines.append(f"  latency_ms: {outcome.latency_ms:.2f}")

    if outcome.infra_error is not None:
        lines.extend(
            [
                "",
                "Infrastructure failure (per-case results not available):",
                f"  stage: {outcome.infra_error.stage}",
                f"  execution_mode: {outcome.infra_error.execution_mode}",
                f"  detail: {outcome.infra_error.detail}",
            ],
        )

    if outcome.internal_error is not None:
        lines.extend(
            [
                "",
                "Internal error (per-case results not available):",
                _indent_block(outcome.internal_error, 2),
            ],
        )

    if outcome.tests_ran and outcome.test_case_results:
        lines.extend(["", "Per-case results:"])
        for index, case in enumerate(outcome.test_case_results, start=1):
            status = "PASS" if case.passed else "FAIL"
            lines.append(f"  [{index}] {status}")
            lines.append(f"    input: {case.input_value!r}")
            lines.append(f"    expected: {case.expected_output!r}")
            lines.append(f"    actual: {case.actual_output!r}")
            if case.error is not None:
                lines.append(f"    error: {case.error}")
            if case.compile_error is not None:
                lines.append(f"    compile_error: {case.compile_error}")

    return "\n".join(lines)


def format_eval_result_reference(outcome: TestOutcome) -> str:
    """JSON reference document for future eval_results Mongo rows."""
    payload = outcome.model_dump(mode="json")
    return json.dumps(payload, indent=2, sort_keys=True)


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."

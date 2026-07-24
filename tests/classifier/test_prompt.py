from __future__ import annotations

import json
import hashlib

import pytest

import dr_code.classifier.prompt as prompt_module
from dr_code.classifier.prompt import (
    correction_prompt,
    prompt_template_identity,
    render_parse_prompt,
    render_test_prompt,
)


def _evaluation_evidence(
    *,
    failed_count: int,
    error_count: int,
    timeout_count: int,
) -> dict[str, object]:
    prompt = render_test_prompt(
        cleaned_source="def candidate(value):\n    return value\n",
        outcome="tests_failed",
        function_count=1,
        best_function_name="candidate",
        total_cases=3,
        passed_count=2,
        failed_count=failed_count,
        error_count=error_count,
        timeout_count=timeout_count,
        coverage_complete=True,
        task_context={"task_id": "Task/1"},
    )
    marker = "Evidence follows as canonical JSON:\n"
    payload = prompt.split(marker, 1)[1].splitlines()[0]
    evidence = json.loads(payload)
    return evidence["evaluation"]


def test_measured_test_evidence_distinguishes_failure_mechanisms() -> None:
    assertion = _evaluation_evidence(
        failed_count=1,
        error_count=0,
        timeout_count=0,
    )
    error = _evaluation_evidence(
        failed_count=0,
        error_count=1,
        timeout_count=0,
    )
    timeout = _evaluation_evidence(
        failed_count=0,
        error_count=0,
        timeout_count=1,
    )

    assert (
        len(
            {
                json.dumps(item, sort_keys=True)
                for item in (
                    assertion,
                    error,
                    timeout,
                )
            }
        )
        == 3
    )
    assert set(assertion) == {
        "best_function_name",
        "coverage_complete",
        "error_count",
        "failed_count",
        "function_count",
        "outcome",
        "passed_count",
        "timeout_count",
        "total_cases",
    }


def test_large_task_context_has_deterministic_total_and_field_budget() -> None:
    context = {"blob": "x" * 100_000, "task_id": "Task/large"}
    prompt = render_parse_prompt(
        decoder_output="y" * 100_000,
        failure_code="failure",
        failed_step="step",
        cause=None,
        task_context=context,
    )
    marker = "Evidence follows as canonical JSON:\n"
    encoded = prompt.split(marker, 1)[1].splitlines()[0]
    evidence = json.loads(encoded)
    bounded_context = evidence["task_context"]

    assert len(encoded) <= prompt_module.MAX_EVIDENCE_CHARS
    assert len(evidence["decoder_output"]) <= prompt_module.MAX_SOURCE_CHARS
    assert len(json.dumps(bounded_context, separators=(",", ":"))) <= (
        prompt_module.MAX_TASK_CONTEXT_CHARS
    )
    canonical_context = json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert (
        bounded_context["sha256"]
        == hashlib.sha256(canonical_context.encode()).hexdigest()
    )
    assert bounded_context["truncated"] is True


def test_escaped_source_is_truncated_by_serialized_delivery_size() -> None:
    prompt = render_parse_prompt(
        decoder_output=("\n\\\x00") * 100_000,
        failure_code=("\n\\\x00") * 10_000,
        failed_step=("\n\\\x00") * 10_000,
        cause=("\n\\\x00") * 10_000,
        task_context={"blob": ("\n\\\x00") * 100_000},
    )
    marker = "Evidence follows as canonical JSON:\n"
    encoded = prompt.split(marker, 1)[1].splitlines()[0]

    assert len(prompt) <= prompt_module.MAX_PRIMARY_PROMPT_CHARS
    assert len(encoded) <= prompt_module.MAX_EVIDENCE_CHARS
    assert json.loads(encoded)["decoder_output"].endswith(
        "...[input truncated]"
    )


def test_correction_prompt_bounds_original_and_parser_error() -> None:
    prompt = correction_prompt(
        ("\n\\\x00") * 100_000,
        ("\n\\\x00") * 100_000,
    )

    assert len(prompt) <= prompt_module.MAX_DELIVERED_PROMPT_CHARS
    assert "Correct it once" in prompt


def test_prompt_budget_is_part_of_template_identity(monkeypatch) -> None:
    before = prompt_template_identity()
    monkeypatch.setattr(
        prompt_module,
        "MAX_EVIDENCE_CHARS",
        prompt_module.MAX_EVIDENCE_CHARS + 1,
    )
    assert prompt_template_identity() != before


@pytest.mark.parametrize(
    ("template_name", "render"),
    [
        (
            "PRIMARY_PROMPT_TEMPLATE",
            lambda: render_parse_prompt(
                decoder_output="text",
                failure_code="no_code",
                failed_step="parse",
                cause=None,
                task_context={},
            ),
        ),
        (
            "CORRECTION_PROMPT_TEMPLATE",
            lambda: correction_prompt("prompt", "invalid"),
        ),
    ],
)
def test_template_identity_hashes_exact_rendering_templates(
    monkeypatch: pytest.MonkeyPatch,
    template_name: str,
    render,
) -> None:
    before_identity = prompt_template_identity()
    before_render = render()
    template = getattr(prompt_module, template_name)
    monkeypatch.setattr(prompt_module, template_name, template + "\nchanged")

    assert prompt_template_identity() != before_identity
    assert render() != before_render

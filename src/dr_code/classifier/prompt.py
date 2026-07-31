"""Deterministic prompt rendering for failure classification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Final

from dr_code.classifier.lane import MAX_RESPONSE_ERROR_CHARS
from dr_code.classifier.taxonomy import (
    FailureFamily,
    definitions_block,
    label_names,
)
from dr_code.eval.identity import identity_hash_for

PROMPT_VERSION: Final = "failure-prompt-v4"
PROMPT_TEMPLATE_VERSION: Final = "failure-prompt-template-v4"
MAX_EVIDENCE_CHARS: Final = 16_000
MAX_SOURCE_CHARS: Final = 10_000
MAX_TASK_CONTEXT_CHARS: Final = 2_000
MAX_METADATA_CHARS: Final = 512
MAX_PRIMARY_PROMPT_CHARS: Final = 15_500
MAX_DELIVERED_PROMPT_CHARS: Final = 16_384
CORRECTION_ATTEMPTS: Final = 1
PRIMARY_PROMPT_TEMPLATE: Final = (
    "Classifier prompt version: {prompt_version}\n"
    "Classify one {family} failure.\n\n"
    "Choose exactly one closed-taxonomy label:\n"
    "{definitions}\n\n"
    "Valid labels: {labels}\n\n"
    "Evidence follows as canonical JSON:\n"
    "{evidence_json}\n\n"
    'Return only {{"label":"<valid label>",'
    '"rationale":"<one concise line>"}} with exactly those fields.'
)
CORRECTION_PROMPT_TEMPLATE: Final = (
    "{original_prompt}\n\n"
    "Your response was invalid. Correct it once.\n"
    "Validation error: {error}\n"
    'Return only {{"label":"<valid label>",'
    '"rationale":"<one concise line>"}} with exactly those fields.'
)


def render_parse_prompt(
    *,
    decoder_output: str,
    failure_code: str,
    failed_step: str,
    cause: str | None,
    task_context: Mapping[str, object],
) -> str:
    evidence = {
        "cause": None
        if cause is None
        else _bounded_json_string(cause, MAX_METADATA_CHARS),
        "decoder_output": _bounded_json_string(
            decoder_output, MAX_SOURCE_CHARS
        ),
        "failed_step": _bounded_json_string(failed_step, MAX_METADATA_CHARS),
        "failure_code": _bounded_json_string(failure_code, MAX_METADATA_CHARS),
        "task_context": _bounded_task_context(task_context),
    }
    return _render(FailureFamily.PARSE, evidence)


def render_test_prompt(
    *,
    cleaned_source: str,
    outcome: str,
    function_count: int,
    best_function_name: str | None,
    total_cases: int,
    passed_count: int,
    failed_count: int,
    error_count: int,
    timeout_count: int,
    coverage_complete: bool,
    task_context: Mapping[str, object],
) -> str:
    evidence = {
        "candidate_source": _bounded_json_string(
            cleaned_source, MAX_SOURCE_CHARS
        ),
        "evaluation": {
            "best_function_name": (
                None
                if best_function_name is None
                else _bounded_json_string(
                    best_function_name, MAX_METADATA_CHARS
                )
            ),
            "coverage_complete": coverage_complete,
            "error_count": error_count,
            "failed_count": failed_count,
            "function_count": function_count,
            "outcome": _bounded_json_string(outcome, MAX_METADATA_CHARS),
            "passed_count": passed_count,
            "timeout_count": timeout_count,
            "total_cases": total_cases,
        },
        "task_context": _bounded_task_context(task_context),
    }
    return _render(FailureFamily.TEST, evidence)


def correction_prompt(original_prompt: str, error: str) -> str:
    bounded_error = _bounded_text(error, MAX_RESPONSE_ERROR_CHARS)
    fixed = CORRECTION_PROMPT_TEMPLATE.format(
        original_prompt="",
        error=bounded_error,
    )
    bounded_original = _bounded_text(
        original_prompt,
        MAX_DELIVERED_PROMPT_CHARS - len(fixed),
    )
    prompt = CORRECTION_PROMPT_TEMPLATE.format(
        original_prompt=bounded_original,
        error=bounded_error,
    )
    if len(prompt) > MAX_DELIVERED_PROMPT_CHARS:
        raise AssertionError("correction prompt exceeded its delivery budget")
    return prompt


def prompt_template_identity() -> str:
    """Hash the versioned prompt-rendering and response policy."""
    return identity_hash_for(
        schema="dr_code.failure_classifier.prompt_template",
        payload={
            "prompt_version": PROMPT_VERSION,
            "template_version": PROMPT_TEMPLATE_VERSION,
            "primary_template": PRIMARY_PROMPT_TEMPLATE,
            "correction_template": CORRECTION_PROMPT_TEMPLATE,
            "max_evidence_chars": MAX_EVIDENCE_CHARS,
            "max_source_chars": MAX_SOURCE_CHARS,
            "max_task_context_chars": MAX_TASK_CONTEXT_CHARS,
            "max_metadata_chars": MAX_METADATA_CHARS,
            "max_primary_prompt_chars": MAX_PRIMARY_PROMPT_CHARS,
            "max_delivered_prompt_chars": MAX_DELIVERED_PROMPT_CHARS,
            "max_response_error_chars": MAX_RESPONSE_ERROR_CHARS,
            "correction_attempts": CORRECTION_ATTEMPTS,
            "response_fields": ["label", "rationale"],
            "rationale_max_chars": 280,
        },
    )


def _render(family: FailureFamily, evidence: object) -> str:
    labels = ", ".join(label_names(family))
    evidence_json = json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(evidence_json) > MAX_EVIDENCE_CHARS:
        raise ValueError(
            "classification evidence exceeds the deterministic prompt budget"
        )
    prompt = PRIMARY_PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION,
        family=family.value,
        definitions=definitions_block(family),
        labels=labels,
        evidence_json=evidence_json,
    )
    if len(prompt) > MAX_PRIMARY_PROMPT_CHARS:
        raise ValueError(
            "classification prompt exceeds the deterministic delivery budget"
        )
    return prompt


def _bounded_json_string(value: str, limit: int) -> str:
    """Bound the exact canonical JSON representation, including escaping."""
    encoded = _canonical_json(value)
    if len(encoded) <= limit:
        return value
    marker = "\n...[input truncated]"
    if len(_canonical_json(marker)) > limit:
        raise ValueError("JSON string budget cannot contain truncation marker")
    lower = 0
    upper = len(value)
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        if len(_canonical_json(value[:candidate] + marker)) <= limit:
            lower = candidate
        else:
            upper = candidate - 1
    return value[:lower] + marker


def _bounded_text(value: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("text budget must be positive")
    if len(value) <= limit:
        return value
    marker = "...[input truncated]"
    if len(marker) > limit:
        return marker[:limit]
    return value[: limit - len(marker)] + marker


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _bounded_task_context(
    task_context: Mapping[str, object],
) -> dict[str, object]:
    value = dict(task_context)
    encoded = _canonical_json(value)
    if len(encoded) <= MAX_TASK_CONTEXT_CHARS:
        return value
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    marker = {
        "canonical_json_prefix": "",
        "sha256": digest,
        "truncated": True,
    }
    lower = 0
    upper = len(encoded)
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        marker["canonical_json_prefix"] = encoded[:candidate]
        bounded = _canonical_json(marker)
        if len(bounded) <= MAX_TASK_CONTEXT_CHARS:
            lower = candidate
        else:
            upper = candidate - 1
    marker["canonical_json_prefix"] = encoded[:lower]
    return marker


__all__ = (
    "CORRECTION_ATTEMPTS",
    "CORRECTION_PROMPT_TEMPLATE",
    "MAX_EVIDENCE_CHARS",
    "MAX_DELIVERED_PROMPT_CHARS",
    "MAX_METADATA_CHARS",
    "MAX_PRIMARY_PROMPT_CHARS",
    "MAX_SOURCE_CHARS",
    "MAX_TASK_CONTEXT_CHARS",
    "PRIMARY_PROMPT_TEMPLATE",
    "PROMPT_TEMPLATE_VERSION",
    "PROMPT_VERSION",
    "correction_prompt",
    "prompt_template_identity",
    "render_parse_prompt",
    "render_test_prompt",
)

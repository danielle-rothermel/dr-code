"""Display helpers for parse-stage walkthroughs."""

from __future__ import annotations

import json

from dr_code.models.attempts import AttemptRecord, AttemptSource
from dr_code.models.outcomes import ParseOutcome


def format_parse_walkthrough(
    record: AttemptRecord,
    outcome: ParseOutcome,
    *,
    preview_chars: int = 600,
) -> str:
    """Format attempt + parse outcome for manual review."""
    lines = [
        f"=== Parse walkthrough: {record.task_id} ===",
        "",
        "Attempt metadata:",
        f"  sample_id: {record.sample_id}",
        f"  run_id: {record.run_id}",
        f"  task_id: {record.task_id}",
        f"  entry_point: {record.entry_point}",
        f"  source: {record.provenance.source.value}",
    ]
    if record.provenance.source is AttemptSource.POOL:
        lines.append(
            f"  occurrence_count: {record.provenance.occurrence_count}"
        )
    lines.extend(
        [
            "",
            "raw_output (input to code-eval):",
            _indent_block(_truncate(record.raw_output, preview_chars), 2),
            "",
            "Parse summary:",
            f"  parse_success: {outcome.parse_success}",
            f"  candidate_count: {outcome.candidate_count}",
            f"  valid_count: {outcome.valid_count}",
        ]
    )
    if outcome.latency_ms is not None:
        lines.append(f"  latency_ms: {outcome.latency_ms:.2f}")
    if outcome.skip_reason is not None:
        lines.append(f"  skip_reason: {outcome.skip_reason}")

    if outcome.parse_success and outcome.extracted_code is not None:
        lines.extend(
            [
                "",
                "Before / after:",
                "  [raw_output]",
                _indent_block(_truncate(record.raw_output, preview_chars), 4),
                "  [extracted_code]",
                _indent_block(
                    _truncate(outcome.extracted_code, preview_chars),
                    4,
                ),
            ]
        )
    elif outcome.extracted_code is not None:
        lines.extend(
            [
                "",
                "extracted_code:",
                _indent_block(
                    _truncate(outcome.extracted_code, preview_chars),
                    2,
                ),
            ]
        )

    if outcome.code_eval is not None:
        prov = outcome.code_eval
        lines.extend(
            [
                "",
                "code_eval provenance:",
                f"  config_fingerprint: {prov.config_fingerprint}",
                f"  selected_candidate_id: {prov.selected_candidate_id}",
                f"  selected_attempt_id: {prov.selected_attempt_id}",
                f"  recovery_attempt_count: {prov.recovery_attempt_count}",
            ]
        )
        if prov.extractor_path is not None:
            lines.append(f"  extractor_path: {list(prov.extractor_path)}")
        if prov.repairs_applied is not None:
            lines.append(f"  repairs_applied: {list(prov.repairs_applied)}")
        if prov.extraction_log_summary is not None:
            lines.append("  extraction_log_summary:")
            for entry in prov.extraction_log_summary:
                lines.append(f"    - {entry}")

    return "\n".join(lines)


def format_eval_result_reference(outcome: ParseOutcome) -> str:
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

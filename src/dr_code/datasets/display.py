"""Display helpers for AttemptRecord spot checks."""

from __future__ import annotations

from dr_code.models.attempts import AttemptRecord, AttemptSource


def format_spot_check(
    pool_records: list[AttemptRecord],
    fresh_records: list[AttemptRecord],
    *,
    task_id: str,
    preview_chars: int = 400,
) -> str:
    """Format side-by-side pool vs fresh_stub rows for manual review."""
    pool_matches = [
        record for record in pool_records if record.task_id == task_id
    ]
    fresh_matches = [
        record for record in fresh_records if record.task_id == task_id
    ]
    if not pool_matches:
        msg = f"No pool records found for task_id={task_id!r}"
        raise ValueError(msg)
    if not fresh_matches:
        msg = f"No fresh_stub records found for task_id={task_id!r}"
        raise ValueError(msg)

    lines = [
        f"=== Spot check: {task_id} ===",
        (
            "Note: pool decoder_input is encoder-compressed text; "
            "fresh_stub uses the official HumanEval stub. Text will differ "
            "by design — compare schema, provenance, and plausible outputs."
        ),
        "",
    ]
    lines.extend(
        _format_record_block(
            label="pool",
            record=pool_matches[0],
            preview_chars=preview_chars,
        )
    )
    lines.append("")
    lines.extend(
        _format_record_block(
            label="fresh_stub",
            record=fresh_matches[0],
            preview_chars=preview_chars,
        )
    )
    return "\n".join(lines)


def _format_record_block(
    *,
    label: str,
    record: AttemptRecord,
    preview_chars: int,
) -> list[str]:
    provenance = record.provenance
    lines = [
        f"[{label}]",
        f"  sample_id: {record.sample_id}",
        f"  source: {provenance.source.value}",
        f"  run_id: {record.run_id}",
        f"  model: {provenance.model}",
    ]
    if provenance.source == AttemptSource.POOL:
        lines.append(f"  occurrence_count: {provenance.occurrence_count}")
        if provenance.pool_attempt_id is not None:
            lines.append(f"  pool_attempt_id: {provenance.pool_attempt_id}")
    if provenance.dec_llm_config_id is not None:
        lines.append(f"  profile: {provenance.dec_llm_config_id}")
    lines.extend(
        [
            "  decoder_input:",
            _indent_block(_truncate(record.decoder_input, preview_chars), 4),
            "  raw_output:",
            _indent_block(_truncate(record.raw_output, preview_chars), 4),
        ]
    )
    return lines


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."

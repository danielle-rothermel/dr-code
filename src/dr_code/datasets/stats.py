"""Summary statistics for AttemptRecord collections."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from dr_code.models.attempts import AttemptRecord


def summarize_attempts(
    records: list[AttemptRecord],
    *,
    sample_count: int = 3,
    preview_chars: int = 120,
) -> dict[str, Any]:
    """Return summary stats and sample previews for demo/reporting."""
    if not records:
        return {
            "record_count": 0,
            "weighted_count": 0,
            "unique_tasks": 0,
            "source_breakdown": {},
            "decoder_input_length_p50": 0,
            "decoder_input_length_p90": 0,
            "raw_output_length_p50": 0,
            "raw_output_length_p90": 0,
            "top_tasks": [],
            "samples": [],
        }

    source_breakdown = Counter(
        record.provenance.source.value for record in records
    )
    task_counts = Counter(record.task_id for record in records)
    decoder_lengths = [len(record.decoder_input) for record in records]
    raw_lengths = [len(record.raw_output) for record in records]
    weighted_count = sum(
        record.provenance.occurrence_count for record in records
    )

    return {
        "record_count": len(records),
        "weighted_count": weighted_count,
        "unique_tasks": len(task_counts),
        "source_breakdown": dict(source_breakdown),
        "decoder_input_length_p50": _percentile(decoder_lengths, 50),
        "decoder_input_length_p90": _percentile(decoder_lengths, 90),
        "raw_output_length_p50": _percentile(raw_lengths, 50),
        "raw_output_length_p90": _percentile(raw_lengths, 90),
        "top_tasks": task_counts.most_common(5),
        "samples": [
            _preview_record(record, preview_chars=preview_chars)
            for record in records[:sample_count]
        ],
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Format a summary dict as plain text for CLI output."""
    lines = [
        f"record_count: {summary['record_count']}",
        f"weighted_count: {summary['weighted_count']}",
        f"unique_tasks: {summary['unique_tasks']}",
        f"source_breakdown: {summary['source_breakdown']}",
        (
            "decoder_input lengths p50/p90: "
            f"{summary['decoder_input_length_p50']}/"
            f"{summary['decoder_input_length_p90']}"
        ),
        (
            "raw_output lengths p50/p90: "
            f"{summary['raw_output_length_p50']}/"
            f"{summary['raw_output_length_p90']}"
        ),
        f"top_tasks: {summary['top_tasks']}",
    ]
    for index, sample in enumerate(summary["samples"], start=1):
        lines.append(f"sample {index}:")
        lines.append(f"  sample_id: {sample['sample_id']}")
        lines.append(f"  task_id: {sample['task_id']}")
        lines.append(f"  source: {sample['source']}")
        lines.append(f"  occurrence_count: {sample['occurrence_count']}")
        lines.append(f"  decoder_input: {sample['decoder_input_preview']}")
        lines.append(f"  raw_output: {sample['raw_output_preview']}")
    return "\n".join(lines)


def _preview_record(
    record: AttemptRecord,
    *,
    preview_chars: int,
) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "task_id": record.task_id,
        "source": record.provenance.source.value,
        "occurrence_count": record.provenance.occurrence_count,
        "decoder_input_preview": _truncate(record.decoder_input, preview_chars),
        "raw_output_preview": _truncate(record.raw_output, preview_chars),
    }


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _percentile(values: list[int], pct: int) -> int:
    if not values:
        return 0
    if pct == 50:
        return int(median(values))
    sorted_values = sorted(values)
    index = int(round((pct / 100) * (len(sorted_values) - 1)))
    return sorted_values[index]

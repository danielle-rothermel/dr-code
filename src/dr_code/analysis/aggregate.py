"""Aggregate enriched analysis rows into summary and slice tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from dr_code.analysis.join import EnrichedRow, JoinReport

_JOINT_OBJECTIVE_FORMULA = "pass - lambda * decoder_input_len_zstd22"
_JOINT_OBJECTIVE_LAMBDA_DEFAULT = 0.001
_JOINT_OBJECTIVE_NOTE = (
    "Preview only for future DSPy optimization; not computed per row in v1."
)


def build_summary(
    rows: list[EnrichedRow], join_report: JoinReport
) -> dict[str, Any]:
    """Build headline summary JSON for a run."""
    outcome_kind_counts = Counter(
        row.outcome_kind for row in rows if row.has_test_outcome
    )
    outcome_kind_counts_weighted: Counter[str] = Counter()
    for row in rows:
        if not row.has_test_outcome or row.outcome_kind is None:
            continue
        outcome_kind_counts_weighted[row.outcome_kind] += (
            row.provenance_occurrence_count
        )

    tested_rows = [
        row
        for row in rows
        if row.has_test_outcome and row.outcome_kind == "tested"
    ]
    tested_pass = sum(1 for row in tested_rows if row.all_tests_passed is True)
    tested_fail = sum(
        1 for row in tested_rows if row.all_tests_passed is not True
    )
    weighted_tested = sum(
        row.provenance_occurrence_count for row in tested_rows
    )
    weighted_pass = sum(
        row.provenance_occurrence_count
        for row in tested_rows
        if row.all_tests_passed is True
    )

    correctness_pass_rate = (
        tested_pass / len(tested_rows) if tested_rows else None
    )
    correctness_pass_rate_weighted = (
        weighted_pass / weighted_tested if weighted_tested else None
    )

    parse_funnel = _parse_funnel(rows)
    return {
        "attempt_count": join_report.attempt_count,
        "outcome_kind_counts": dict(outcome_kind_counts),
        "outcome_kind_counts_weighted": dict(outcome_kind_counts_weighted),
        "tested_pass_count": tested_pass,
        "tested_fail_count": tested_fail,
        "correctness_pass_rate": correctness_pass_rate,
        "correctness_pass_rate_weighted": correctness_pass_rate_weighted,
        "parse_funnel": parse_funnel,
        "join_failures": {
            "missing_parse_count": len(join_report.missing_parse_sample_ids),
            "missing_parse_sample_ids": list(
                join_report.missing_parse_sample_ids
            ),
            "missing_test_count": len(join_report.missing_test_sample_ids),
            "missing_test_sample_ids": list(
                join_report.missing_test_sample_ids
            ),
        },
        "joint_objective_preview": {
            "formula": _JOINT_OBJECTIVE_FORMULA,
            "lambda_default": _JOINT_OBJECTIVE_LAMBDA_DEFAULT,
            "note": _JOINT_OBJECTIVE_NOTE,
        },
        "comparison_runs_note": (
            "Single-run v1. Future multi-run comparison should load "
            "multiple enriched.parquet files keyed by run_id."
        ),
    }


def build_aggregates(
    rows: list[EnrichedRow],
) -> dict[str, list[dict[str, Any]]]:
    """Build all slice aggregate tables."""
    return {
        "by_source": aggregate_by_source(rows),
        "by_model": aggregate_by_model(rows),
        "by_task": aggregate_by_task(rows),
        "by_compression_quartile": aggregate_by_compression_quartile(rows),
    }


def aggregate_by_source(rows: list[EnrichedRow]) -> list[dict[str, Any]]:
    return _aggregate_slice(rows, lambda row: row.provenance_source)


def aggregate_by_model(rows: list[EnrichedRow]) -> list[dict[str, Any]]:
    return _aggregate_slice(
        rows,
        lambda row: (
            row.provenance_model if row.provenance_model else "(unknown)"
        ),
    )


def aggregate_by_task(rows: list[EnrichedRow]) -> list[dict[str, Any]]:
    return _aggregate_slice(rows, lambda row: row.task_id)


def aggregate_by_compression_quartile(
    rows: list[EnrichedRow],
) -> list[dict[str, Any]]:
    return _aggregate_slice(
        rows,
        lambda row: row.compression_quartile or "(unknown)",
    )


def _aggregate_slice(
    rows: list[EnrichedRow],
    key_fn: Any,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[EnrichedRow]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)

    results: list[dict[str, Any]] = []
    for key in sorted(grouped, key=str):
        group_rows = grouped[key]
        tested = [
            row
            for row in group_rows
            if row.has_test_outcome and row.outcome_kind == "tested"
        ]
        row_count = len(group_rows)
        weighted_count = sum(
            row.provenance_occurrence_count for row in group_rows
        )
        tested_count = len(tested)
        tested_weighted = sum(
            row.provenance_occurrence_count for row in tested
        )
        pass_count = sum(1 for row in tested if row.all_tests_passed is True)
        weighted_pass = sum(
            row.provenance_occurrence_count
            for row in tested
            if row.all_tests_passed is True
        )
        results.append(
            {
                "slice_key": str(key),
                "row_count": row_count,
                "weighted_count": weighted_count,
                "tested_count": tested_count,
                "tested_weighted_count": tested_weighted,
                "pass_count": pass_count,
                "weighted_pass_count": weighted_pass,
                "pass_rate": (
                    pass_count / tested_count if tested_count else None
                ),
                "weighted_pass_rate": (
                    weighted_pass / tested_weighted
                    if tested_weighted
                    else None
                ),
            }
        )
    return results


def _parse_funnel(rows: list[EnrichedRow]) -> dict[str, int | float]:
    raw = len(rows)
    parse_success = sum(1 for row in rows if row.parse_success is True)
    tested = sum(
        1
        for row in rows
        if row.has_test_outcome and row.outcome_kind == "tested"
    )
    all_passed = sum(
        1
        for row in rows
        if row.has_test_outcome
        and row.outcome_kind == "tested"
        and row.all_tests_passed is True
    )

    weighted_raw = sum(row.provenance_occurrence_count for row in rows)
    weighted_parse_success = sum(
        row.provenance_occurrence_count
        for row in rows
        if row.parse_success is True
    )
    weighted_tested = sum(
        row.provenance_occurrence_count
        for row in rows
        if row.has_test_outcome and row.outcome_kind == "tested"
    )
    weighted_all_passed = sum(
        row.provenance_occurrence_count
        for row in rows
        if row.has_test_outcome
        and row.outcome_kind == "tested"
        and row.all_tests_passed is True
    )

    return {
        "raw": raw,
        "parse_success": parse_success,
        "tested": tested,
        "all_tests_passed": all_passed,
        "weighted_raw": weighted_raw,
        "weighted_parse_success": weighted_parse_success,
        "weighted_tested": weighted_tested,
        "weighted_all_tests_passed": weighted_all_passed,
    }

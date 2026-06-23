"""Unit tests for analysis aggregates."""

from __future__ import annotations

from dr_code.analysis.aggregate import (
    aggregate_by_source,
    build_aggregates,
    build_summary,
)
from dr_code.analysis.join import EnrichedRow, JoinReport


def _row(
    *,
    sample_id: str,
    source: str = "pool",
    model: str | None = "model-a",
    task_id: str = "HumanEval/0",
    occurrence_count: int = 1,
    parse_success: bool | None = True,
    has_test_outcome: bool = True,
    outcome_kind: str | None = "tested",
    all_tests_passed: bool | None = True,
    test_pass_rate: float | None = None,
    compression_quartile: str = "Q2",
) -> EnrichedRow:
    resolved_pass_rate = (
        test_pass_rate
        if test_pass_rate is not None
        else (1.0 if all_tests_passed else 0.0)
    )
    return EnrichedRow(
        sample_id=sample_id,
        run_id=None,
        task_id=task_id,
        decoder_input_len_raw=100,
        decoder_input_len_zstd22=50,
        compression_quartile=compression_quartile,
        provenance_source=source,
        provenance_model=model,
        provenance_occurrence_count=occurrence_count,
        parse_success=parse_success,
        has_test_outcome=has_test_outcome,
        outcome_kind=outcome_kind,
        tests_ran=True if outcome_kind == "tested" else None,
        all_tests_passed=all_tests_passed,
        test_pass_rate=resolved_pass_rate,
    )


def test_build_summary_excludes_infra_from_correctness_pass_rate() -> None:
    rows = [
        _row(sample_id="pass", all_tests_passed=True),
        _row(
            sample_id="fail",
            all_tests_passed=False,
            test_pass_rate=0.0,
        ),
        _row(
            sample_id="infra",
            outcome_kind="infra_error",
            all_tests_passed=None,
            test_pass_rate=None,
        ),
    ]
    report = JoinReport(
        attempt_count=3,
        matched_parse_count=3,
        matched_test_count=3,
        missing_parse_sample_ids=(),
        missing_test_sample_ids=(),
    )

    summary = build_summary(rows, report)

    assert summary["tested_pass_count"] == 1
    assert summary["tested_fail_count"] == 1
    assert summary["correctness_pass_rate"] == 0.5
    assert summary["outcome_kind_counts"]["infra_error"] == 1


def test_build_summary_weighted_pass_rate_uses_occurrence_count() -> None:
    rows = [
        _row(sample_id="pass", occurrence_count=10, all_tests_passed=True),
        _row(
            sample_id="fail",
            occurrence_count=1,
            all_tests_passed=False,
            test_pass_rate=0.0,
        ),
    ]
    report = JoinReport(
        attempt_count=2,
        matched_parse_count=2,
        matched_test_count=2,
        missing_parse_sample_ids=(),
        missing_test_sample_ids=(),
    )

    summary = build_summary(rows, report)

    assert summary["correctness_pass_rate_weighted"] == 10 / 11


def test_aggregate_by_source_keeps_pool_and_fresh_stub_separate() -> None:
    rows = [
        _row(sample_id="pool-row", source="pool", all_tests_passed=True),
        _row(
            sample_id="fresh-row",
            source="fresh_stub",
            all_tests_passed=False,
            test_pass_rate=0.0,
        ),
    ]

    by_source = aggregate_by_source(rows)
    keys = {row["slice_key"] for row in by_source}

    assert keys == {"fresh_stub", "pool"}
    pool_row = next(row for row in by_source if row["slice_key"] == "pool")
    fresh_row = next(
        row for row in by_source if row["slice_key"] == "fresh_stub"
    )
    assert pool_row["pass_rate"] == 1.0
    assert fresh_row["pass_rate"] == 0.0


def test_build_aggregates_includes_all_slice_tables() -> None:
    rows = [_row(sample_id="s1")]

    aggregates = build_aggregates(rows)

    assert set(aggregates) == {
        "by_source",
        "by_model",
        "by_task",
        "by_compression_quartile",
    }


def test_parse_funnel_counts_weighted_and_unweighted() -> None:
    rows = [
        _row(sample_id="s1", occurrence_count=3, all_tests_passed=True),
        _row(
            sample_id="s2",
            occurrence_count=1,
            parse_success=False,
            has_test_outcome=False,
            outcome_kind=None,
            all_tests_passed=None,
        ),
    ]
    report = JoinReport(
        attempt_count=2,
        matched_parse_count=2,
        matched_test_count=1,
        missing_parse_sample_ids=(),
        missing_test_sample_ids=("s2",),
    )

    summary = build_summary(rows, report)
    funnel = summary["parse_funnel"]

    assert funnel["raw"] == 2
    assert funnel["parse_success"] == 1
    assert funnel["tested"] == 1
    assert funnel["all_tests_passed"] == 1
    assert funnel["weighted_raw"] == 4
    assert funnel["weighted_all_tests_passed"] == 3

"""Unit tests for analysis join logic."""

from __future__ import annotations

from dr_code.analysis.join import enrich_eval_run
from dr_code.models.attempts import AttemptProvenance, AttemptRecord, AttemptSource
from dr_code.models.outcomes import ParseOutcome, TestOutcome


def _attempt(
    sample_id: str,
    *,
    source: AttemptSource = AttemptSource.POOL,
    occurrence_count: int = 1,
    decoder_input: str = "def foo(): pass",
) -> AttemptRecord:
    return AttemptRecord(
        sample_id=sample_id,
        run_id=None,
        task_id="HumanEval/0",
        entry_point="has_close_elements",
        decoder_input=decoder_input,
        raw_output="out",
        provenance=AttemptProvenance(
            source=source,
            model="model-a",
            occurrence_count=occurrence_count,
        ),
    )


def test_enrich_eval_run_joins_parse_and_test_outcomes() -> None:
    attempts = [_attempt("s1")]
    parse_by_sample_id = {
        "s1": ParseOutcome(
            sample_id="s1",
            run_id=None,
            task_id="HumanEval/0",
            parse_success=True,
        )
    }
    test_by_sample_id = {
        "s1": TestOutcome(
            sample_id="s1",
            run_id=None,
            task_id="HumanEval/0",
            parse_success=True,
            outcome_kind="tested",
            tests_ran=True,
            all_tests_passed=True,
            test_pass_rate=1.0,
        )
    }

    rows, report = enrich_eval_run(
        attempts,
        parse_by_sample_id,
        test_by_sample_id,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.has_test_outcome is True
    assert row.parse_success is True
    assert row.outcome_kind == "tested"
    assert row.all_tests_passed is True
    assert row.decoder_input_len_raw > 0
    assert row.decoder_input_len_zstd22 > 0
    assert row.compression_quartile in {"Q1", "Q2", "Q3", "Q4"}
    assert report.matched_test_count == 1
    assert report.missing_test_sample_ids == ()


def test_enrich_eval_run_reports_missing_test_outcome() -> None:
    attempts = [_attempt("s1"), _attempt("s2")]
    parse_by_sample_id = {
        "s1": ParseOutcome(
            sample_id="s1",
            run_id=None,
            task_id="HumanEval/0",
            parse_success=True,
        ),
        "s2": ParseOutcome(
            sample_id="s2",
            run_id=None,
            task_id="HumanEval/0",
            parse_success=False,
            skip_reason="no_valid_candidate",
        ),
    }
    test_by_sample_id = {
        "s1": TestOutcome(
            sample_id="s1",
            run_id=None,
            task_id="HumanEval/0",
            parse_success=True,
            outcome_kind="tested",
            tests_ran=True,
            all_tests_passed=False,
            test_pass_rate=0.0,
        )
    }

    rows, report = enrich_eval_run(
        attempts,
        parse_by_sample_id,
        test_by_sample_id,
    )

    missing = [row for row in rows if not row.has_test_outcome]
    assert len(missing) == 1
    assert missing[0].sample_id == "s2"
    assert report.missing_test_sample_ids == ("s2",)


def test_enrich_eval_run_preserves_occurrence_count() -> None:
    attempts = [_attempt("s1", occurrence_count=42)]
    parse_by_sample_id = {
        "s1": ParseOutcome(
            sample_id="s1",
            run_id=None,
            task_id="HumanEval/0",
            parse_success=True,
        )
    }
    test_by_sample_id = {
        "s1": TestOutcome(
            sample_id="s1",
            run_id=None,
            task_id="HumanEval/0",
            parse_success=True,
            outcome_kind="tested",
            tests_ran=True,
            all_tests_passed=True,
            test_pass_rate=1.0,
        )
    }

    rows, _report = enrich_eval_run(
        attempts,
        parse_by_sample_id,
        test_by_sample_id,
    )

    assert rows[0].provenance_occurrence_count == 42

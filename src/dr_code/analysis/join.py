"""Join attempt, parse, and test exports into enriched analysis rows."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import TypeVar

from dr_code.analysis.compress import decoder_input_compression
from dr_code.datasets.export import read_attempts
from dr_code.models.attempts import AttemptRecord
from dr_code.models.base import FrozenModel
from dr_code.models.outcomes import ParseOutcome, TestOutcome

_TOutcome = TypeVar("_TOutcome", ParseOutcome, TestOutcome)


class JoinReport(FrozenModel):
    """Counts and ids for join coverage diagnostics."""

    attempt_count: int
    matched_parse_count: int
    matched_test_count: int
    missing_parse_sample_ids: tuple[str, ...]
    missing_test_sample_ids: tuple[str, ...]


class EnrichedRow(FrozenModel):
    """Row-level analysis record for Parquet export."""

    sample_id: str
    run_id: str | None
    task_id: str
    entry_point: str
    decoder_input_len_raw: int
    decoder_input_len_zstd22: int
    compression_quartile: str | None = None
    provenance_source: str
    provenance_model: str | None = None
    provenance_pool_name: str | None = None
    provenance_prompt_template_id: str | None = None
    provenance_enc_llm_config_id: str | None = None
    provenance_dec_llm_config_id: str | None = None
    provenance_occurrence_count: int = 1
    provenance_pool_attempt_id: str | None = None
    parse_success: bool | None = None
    parse_skip_reason: str | None = None
    has_test_outcome: bool = False
    outcome_kind: str | None = None
    tests_ran: bool | None = None
    all_tests_passed: bool | None = None
    test_pass_rate: float | None = None
    skipped: bool | None = None
    skip_reason: str | None = None


def load_parse_outcomes(path: Path) -> dict[str, ParseOutcome]:
    """Load ParseOutcome rows keyed by sample_id."""
    return _load_jsonl_outcomes(path, ParseOutcome)


def load_test_outcomes(path: Path) -> dict[str, TestOutcome]:
    """Load TestOutcome rows keyed by sample_id."""
    return _load_jsonl_outcomes(path, TestOutcome)


def load_attempts(path: Path) -> list[AttemptRecord]:
    """Load AttemptRecord rows from Parquet or JSONL."""
    return read_attempts(path)


def enrich_eval_run(
    attempts: list[AttemptRecord],
    parse_by_sample_id: dict[str, ParseOutcome],
    test_by_sample_id: dict[str, TestOutcome],
) -> tuple[list[EnrichedRow], JoinReport]:
    """Join exports and attach compression metrics."""
    rows: list[EnrichedRow] = []
    missing_parse: list[str] = []
    missing_test: list[str] = []
    matched_parse = 0
    matched_test = 0

    for record in attempts:
        parse_outcome = _match_parse(record, parse_by_sample_id)
        test_outcome = _match_test(record, test_by_sample_id)

        if parse_outcome is None:
            missing_parse.append(record.sample_id)
        else:
            matched_parse += 1

        if test_outcome is None:
            missing_test.append(record.sample_id)
        else:
            matched_test += 1

        raw_len, zstd_len = decoder_input_compression(record.decoder_input)
        provenance = record.provenance
        rows.append(
            EnrichedRow(
                sample_id=record.sample_id,
                run_id=record.run_id,
                task_id=record.task_id,
                entry_point=record.entry_point,
                decoder_input_len_raw=raw_len,
                decoder_input_len_zstd22=zstd_len,
                provenance_source=provenance.source.value,
                provenance_model=provenance.model,
                provenance_pool_name=provenance.pool_name,
                provenance_prompt_template_id=provenance.prompt_template_id,
                provenance_enc_llm_config_id=provenance.enc_llm_config_id,
                provenance_dec_llm_config_id=provenance.dec_llm_config_id,
                provenance_occurrence_count=provenance.occurrence_count,
                provenance_pool_attempt_id=provenance.pool_attempt_id,
                parse_success=(
                    parse_outcome.parse_success if parse_outcome else None
                ),
                parse_skip_reason=(
                    parse_outcome.skip_reason if parse_outcome else None
                ),
                has_test_outcome=test_outcome is not None,
                outcome_kind=(
                    test_outcome.outcome_kind if test_outcome else None
                ),
                tests_ran=test_outcome.tests_ran if test_outcome else None,
                all_tests_passed=(
                    test_outcome.all_tests_passed if test_outcome else None
                ),
                test_pass_rate=(
                    test_outcome.test_pass_rate if test_outcome else None
                ),
                skipped=test_outcome.skipped if test_outcome else None,
                skip_reason=test_outcome.skip_reason if test_outcome else None,
            )
        )

    rows = _assign_compression_quartiles(rows)
    report = JoinReport(
        attempt_count=len(attempts),
        matched_parse_count=matched_parse,
        matched_test_count=matched_test,
        missing_parse_sample_ids=tuple(missing_parse),
        missing_test_sample_ids=tuple(missing_test),
    )
    return rows, report


def _load_jsonl_outcomes(
    path: Path, model: type[_TOutcome]
) -> dict[str, _TOutcome]:
    by_sample_id: dict[str, _TOutcome] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        outcome = model.model_validate(json.loads(line))
        by_sample_id[outcome.sample_id] = outcome
    return by_sample_id


def _match_parse(
    record: AttemptRecord,
    parse_by_sample_id: dict[str, ParseOutcome],
) -> ParseOutcome | None:
    outcome = parse_by_sample_id.get(record.sample_id)
    if outcome is None:
        return None
    if not _run_ids_compatible(record.run_id, outcome.run_id):
        return None
    return outcome


def _match_test(
    record: AttemptRecord,
    test_by_sample_id: dict[str, TestOutcome],
) -> TestOutcome | None:
    outcome = test_by_sample_id.get(record.sample_id)
    if outcome is None:
        return None
    if not _run_ids_compatible(record.run_id, outcome.run_id):
        return None
    return outcome


def _run_ids_compatible(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return True
    return left == right


def _assign_compression_quartiles(
    rows: list[EnrichedRow],
) -> list[EnrichedRow]:
    if not rows:
        return rows
    values = [row.decoder_input_len_zstd22 for row in rows]
    q1 = _percentile(values, 25)
    q2 = _percentile(values, 50)
    q3 = _percentile(values, 75)
    labeled: list[EnrichedRow] = []
    for row in rows:
        value = row.decoder_input_len_zstd22
        if value <= q1:
            quartile = "Q1"
        elif value <= q2:
            quartile = "Q2"
        elif value <= q3:
            quartile = "Q3"
        else:
            quartile = "Q4"
        labeled.append(
            row.model_copy(update={"compression_quartile": quartile})
        )
    return labeled


def _percentile(values: list[int], pct: int) -> int:
    if not values:
        return 0
    if pct == 50:
        return int(median(values))
    sorted_values = sorted(values)
    index = int(round((pct / 100) * (len(sorted_values) - 1)))
    return sorted_values[index]

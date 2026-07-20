"""Deterministic, compact analysis of preprocessing corpus artifacts.

The preprocessing run remains authoritative.  This module only validates its
relational projections, joins them to the source corpus by ``sample_id``, and
writes small derived summaries suitable for review and a static viewer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.parquet as pq


ANALYSIS_SCHEMA_VERSION: Final = 1
_MISSING: Final = "<null>"
_BLANK: Final = "<blank>"
_OTHER: Final = "<other>"
_SUCCESS: Final = "function_candidates_extracted"
_RATE_DENOMINATORS: Final = ("all", "present", "nonblank")
_DIMENSIONS: Final = (
    "source_kind",
    "source_database_table",
    "model",
    "encoder_model",
    "decoder_model",
    "prompt_fidelity",
    "retry_partial",
    "task_id",
    "date_month",
    "date_day",
)
_COMPACT_TABLES: Final = (
    "outcome_by_dimension",
    "candidate_multiplicity",
    "origin_contribution",
    "failure_modes",
    "compile_warnings",
)
_EXAMPLE_LIMIT: Final = 30
_TEXT_LIMIT: Final = 1_200
_CANDIDATE_LIMIT: Final = 1_200
_FACTS_LIMIT: Final = 1_200
_REJECTIONS_LIMIT: Final = 8


class PreprocessingAnalysisError(ValueError):
    """The input corpus and preprocessing projections cannot be reconciled."""


@dataclass(frozen=True, slots=True)
class Result:
    outcome: str
    failure_code: str | None
    failed_step: str | None
    final_candidate_count: int


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    decoder_present: bool
    decoder_nonblank: bool
    dimensions: Mapping[str, str]


def analyze_preprocessing_corpus(
    *,
    corpus_path: Path | str,
    run_dir: Path | str,
    output_dir: Path | str,
    candidate_membership_path: Path | str | None = None,
    candidate_results_path: Path | str | None = None,
) -> Path:
    """Validate a run and emit deterministic compact analysis deliverables.

    Existing optional candidate-evaluation relations are recorded as
    provenance-only inputs.  They intentionally do not affect preprocessing
    metrics until a later analysis contract defines their join semantics.
    """
    corpus_file = Path(corpus_path).expanduser().resolve(strict=True)
    run_root = Path(run_dir).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"analysis output already exists: {destination}")
    _validate_run(run_root)

    results = _read_results(run_root / "results.parquet")
    candidate_stats, origin_final, origin_converged, warning_stats = (
        _read_candidates(run_root / "candidates.parquet", results)
    )
    initial_origins = _read_initial_origins(run_root / "step_facts.parquet")
    rejections, rejected_sample_ids = _read_rejections(
        run_root / "rejections.parquet", results
    )
    _validate_candidate_invariants(results, candidate_stats)

    samples, dimension_counts = _read_and_join_corpus(corpus_file, results)
    summary = _build_summary(
        results=results,
        samples=samples,
        candidate_stats=candidate_stats,
        dimension_counts=dimension_counts,
        origin_final=origin_final,
        origin_converged=origin_converged,
        initial_origins=initial_origins,
        rejections=rejections,
        warning_stats=warning_stats,
        rejected_sample_ids=rejected_sample_ids,
        corpus_file=corpus_file,
        run_root=run_root,
        optional_inputs={
            "candidate_membership": candidate_membership_path,
            "candidate_results": candidate_results_path,
        },
    )
    examples = _build_examples(
        corpus_file=corpus_file,
        run_root=run_root,
        results=results,
        samples=samples,
        candidate_stats=candidate_stats,
        rejected_sample_ids=rejected_sample_ids,
    )
    _write_deliverables(destination, summary, examples)
    return destination


def _validate_run(run_root: Path) -> None:
    required = (
        "manifest.json",
        "results.parquet",
        "candidates.parquet",
        "step_facts.parquet",
        "rejections.parquet",
    )
    missing = [name for name in required if not (run_root / name).is_file()]
    if missing:
        raise PreprocessingAnalysisError(
            "preprocessing run is missing: " + ", ".join(missing)
        )
    manifest = _read_json(run_root / "manifest.json")
    if manifest.get("complete") is not True:
        raise PreprocessingAnalysisError(
            "preprocessing run manifest is incomplete"
        )


def _read_results(path: Path) -> dict[str, Result]:
    required = {
        "sample_id",
        "outcome",
        "failure_code",
        "failed_step",
        "final_candidate_count",
    }
    _require_columns(path, required)
    rows: dict[str, Result] = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536):
        for row in batch.to_pylist():
            sample_id = _required_str(row, "sample_id", path)
            if sample_id in rows:
                raise PreprocessingAnalysisError(
                    f"results sample_id is not unique: {sample_id}"
                )
            final_count = row["final_candidate_count"]
            if not isinstance(final_count, int) or final_count < 0:
                raise PreprocessingAnalysisError(
                    f"invalid final_candidate_count for {sample_id}"
                )
            rows[sample_id] = Result(
                outcome=_required_str(row, "outcome", path),
                failure_code=_str_or_none(row.get("failure_code")),
                failed_step=_str_or_none(row.get("failed_step")),
                final_candidate_count=final_count,
            )
    if not rows:
        raise PreprocessingAnalysisError("results relation is empty")
    return rows


def _read_candidates(
    path: Path, results: Mapping[str, Result]
) -> tuple[
    dict[str, dict[str, object]],
    Counter[tuple[str, str]],
    Counter[tuple[str, str]],
    Counter[str],
]:
    required = {
        "sample_id",
        "candidate_index",
        "candidate_id",
        "origins",
        "parse_ok",
        "compile_ok",
        "top_level_function_count",
        "compile_warnings",
    }
    _require_columns(path, required)
    stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "count": 0,
            "indices": set(),
            "ids": set(),
            "bad": [],
            "origin_count": 0,
            "converged": 0,
        }
    )
    origin_final: Counter[tuple[str, str]] = Counter()
    origin_converged: Counter[tuple[str, str]] = Counter()
    warnings: Counter[str] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32_768):
        for row in batch.to_pylist():
            sample_id = _required_str(row, "sample_id", path)
            if sample_id not in results:
                raise PreprocessingAnalysisError(
                    f"candidate references unknown sample_id: {sample_id}"
                )
            index = row["candidate_index"]
            candidate_id = row["candidate_id"]
            if (
                not isinstance(index, int)
                or index < 0
                or not isinstance(candidate_id, str)
                or not candidate_id
            ):
                raise PreprocessingAnalysisError(
                    f"invalid candidate identity for {sample_id}"
                )
            sample_stats = stats[sample_id]
            indices = sample_stats["indices"]
            ids = sample_stats["ids"]
            assert isinstance(indices, set) and isinstance(ids, set)
            indices.add(index)
            ids.add(candidate_id)
            sample_stats["count"] = _int_at(sample_stats, "count") + 1
            origins = row["origins"] or []
            if not origins:
                _append_bad(sample_stats, "missing_origins")
            sample_stats["origin_count"] = _int_at(
                sample_stats, "origin_count"
            ) + len(origins)
            if len(origins) > 1:
                sample_stats["converged"] = (
                    _int_at(sample_stats, "converged") + 1
                )
            for origin in origins:
                variant = (
                    _str_or_none(origin.get("variant"))
                    if isinstance(origin, dict)
                    else None
                )
                strategy = (
                    _str_or_none(origin.get("strategy"))
                    if isinstance(origin, dict)
                    else None
                )
                if variant is None or strategy is None:
                    _append_bad(sample_stats, "invalid_origin")
                else:
                    origin_final[(variant, strategy)] += 1
                    if len(origins) > 1:
                        origin_converged[(variant, strategy)] += 1
            if row["parse_ok"] is not True or row["compile_ok"] is not True:
                _append_bad(sample_stats, "not_parse_compile_ok")
            function_count = row["top_level_function_count"]
            if not isinstance(function_count, int) or function_count < 1:
                _append_bad(sample_stats, "missing_top_level_function")
            compile_warnings = row["compile_warnings"] or []
            for warning in compile_warnings:
                if isinstance(warning, str):
                    warnings[warning] += 1
    return stats, origin_final, origin_converged, warnings


def _read_initial_origins(path: Path) -> Counter[tuple[str, str]]:
    _require_columns(path, {"step_name", "facts_json"})
    origins: Counter[tuple[str, str]] = Counter()
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        batch_size=65_536, columns=["step_name", "facts_json"]
    ):
        for row in batch.to_pylist():
            if row["step_name"] != "extract_candidates":
                continue
            facts = _parse_json_object(row["facts_json"], path)
            values = facts.get("origins")
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict):
                    continue
                variant = _str_or_none(value.get("variant"))
                strategy = _str_or_none(value.get("strategy"))
                count = value.get("candidate_count")
                if (
                    variant is not None
                    and strategy is not None
                    and isinstance(count, int)
                ):
                    origins[(variant, strategy)] += count
    return origins


def _read_rejections(
    path: Path, results: Mapping[str, Result]
) -> tuple[dict[tuple[str, str, str], dict[str, object]], set[str]]:
    _require_columns(path, {"sample_id", "step_name", "reason_code"})
    stats: dict[tuple[str, str, str], dict[str, object]] = {}
    rejected_samples: set[str] = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536):
        for row in batch.to_pylist():
            sample_id = _required_str(row, "sample_id", path)
            if sample_id not in results:
                raise PreprocessingAnalysisError(
                    f"rejection references unknown sample_id: {sample_id}"
                )
            step = _display(row.get("step_name"))
            reason = _display(row.get("reason_code"))
            key = ("candidate_rejection", step, reason)
            value = stats.setdefault(key, {"count": 0, "sample_ids": set()})
            value["count"] = _int_at(value, "count") + 1
            sample_ids = _string_set_at(value, "sample_ids")
            sample_ids.add(sample_id)
            rejected_samples.add(sample_id)
    for sample_id, result in results.items():
        if result.failure_code is None:
            continue
        key = (
            "sample_outcome",
            _display(result.failed_step),
            result.failure_code,
        )
        value = stats.setdefault(key, {"count": 0, "sample_ids": set()})
        value["count"] = _int_at(value, "count") + 1
        sample_ids = _string_set_at(value, "sample_ids")
        sample_ids.add(sample_id)
    return stats, rejected_samples


def _validate_candidate_invariants(
    results: Mapping[str, Result],
    candidate_stats: Mapping[str, Mapping[str, object]],
) -> None:
    for sample_id, result in results.items():
        stats = candidate_stats.get(sample_id)
        count = 0 if stats is None else _int_at(stats, "count")
        if count != result.final_candidate_count:
            raise PreprocessingAnalysisError(
                f"candidate count mismatch for {sample_id}: {count} != {result.final_candidate_count}"
            )
        if count == 0:
            continue
        assert stats is not None
        indices = _int_set_at(stats, "indices")
        candidate_ids = _string_set_at(stats, "ids")
        bad = _string_list_at(stats, "bad")
        if indices != set(range(count)) or len(candidate_ids) != count or bad:
            raise PreprocessingAnalysisError(
                f"candidate invariants failed for {sample_id}: {sorted(bad)}"
            )
        if result.outcome != _SUCCESS:
            raise PreprocessingAnalysisError(
                f"non-success result has final candidates: {sample_id}"
            )


def _read_and_join_corpus(
    path: Path, results: Mapping[str, Result]
) -> tuple[dict[str, Sample], Counter[tuple[str, str]]]:
    required = {
        "sample_id",
        "decoder_output",
        "source_kind",
        "source_database",
        "source_table",
        "model",
        "encoder_model",
        "decoder_model",
        "prompt_fidelity",
        "is_retry",
        "is_partial",
        "task_id",
        "date",
    }
    _require_columns(path, required)
    samples: dict[str, Sample] = {}
    dimension_counts: Counter[tuple[str, str]] = Counter()
    columns = sorted(required)
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=32_768, columns=columns
    ):
        for row in batch.to_pylist():
            sample_id = _required_str(row, "sample_id", path)
            if sample_id in samples:
                raise PreprocessingAnalysisError(
                    f"corpus sample_id is not unique: {sample_id}"
                )
            if sample_id not in results:
                raise PreprocessingAnalysisError(
                    f"corpus sample_id is absent from results: {sample_id}"
                )
            raw = row["decoder_output"]
            if raw is not None and not isinstance(raw, str):
                raise PreprocessingAnalysisError(
                    f"decoder_output is not string: {sample_id}"
                )
            dimensions = _dimensions(row)
            for dimension, value in dimensions.items():
                dimension_counts[(dimension, value)] += 1
            samples[sample_id] = Sample(
                sample_id=sample_id,
                decoder_present=raw is not None,
                decoder_nonblank=raw is not None and bool(raw.strip()),
                dimensions=dimensions,
            )
    missing = set(results).difference(samples)
    if missing:
        raise PreprocessingAnalysisError(
            f"results sample_id is absent from corpus: {min(missing)}"
        )
    return samples, dimension_counts


def _dimensions(row: Mapping[str, object]) -> dict[str, str]:
    date = row.get("date")
    if date is not None and not isinstance(date, datetime):
        raise PreprocessingAnalysisError("corpus date must be a timestamp")
    database = _display(row.get("source_database"))
    table = _display(row.get("source_table"))
    return {
        "source_kind": _display(row.get("source_kind")),
        "source_database_table": f"{database}.{table}",
        "model": _display(row.get("model")),
        "encoder_model": _display(row.get("encoder_model")),
        "decoder_model": _display(row.get("decoder_model")),
        "prompt_fidelity": _display(row.get("prompt_fidelity")),
        "retry_partial": f"retry={_display(row.get('is_retry'))};partial={_display(row.get('is_partial'))}",
        "task_id": _display(row.get("task_id")),
        "date_month": date.strftime("%Y-%m") if date is not None else _MISSING,
        "date_day": date.strftime("%Y-%m-%d")
        if date is not None
        else _MISSING,
    }


def _build_summary(
    *,
    results: Mapping[str, Result],
    samples: Mapping[str, Sample],
    candidate_stats: Mapping[str, Mapping[str, object]],
    dimension_counts: Mapping[tuple[str, str], int],
    origin_final: Mapping[tuple[str, str], int],
    origin_converged: Mapping[tuple[str, str], int],
    initial_origins: Mapping[tuple[str, str], int],
    rejections: Mapping[tuple[str, str, str], Mapping[str, object]],
    warning_stats: Mapping[str, int],
    rejected_sample_ids: set[str],
    corpus_file: Path,
    run_root: Path,
    optional_inputs: Mapping[str, Path | str | None],
) -> dict[str, object]:
    denominator_counts = _denominators(samples.values())
    funnel = _funnel(results, samples, candidate_stats)
    outcomes = _outcome_summary(results, samples, denominator_counts)
    outcome_rows = _outcome_rows(results, samples)
    multiplicity = _multiplicity_rows(results, samples)
    origin_rows = _origin_rows(initial_origins, origin_final, origin_converged)
    failures = _failure_rows(rejections)
    warnings = _warning_rows(warning_stats, candidate_stats)
    source_kind_reconciliation = _source_kind_reconciliation(samples)
    if sum(row["sample_count"] for row in source_kind_reconciliation) != len(
        samples
    ):
        raise PreprocessingAnalysisError(
            "source_kind reconciliation is incomplete"
        )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "provenance": _provenance(corpus_file, run_root, optional_inputs),
        "limitations": [
            "This is a preprocessing-only analysis; it does not claim task correctness or execution success.",
            "Candidate-evaluation membership/results may be recorded as optional provenance, but are not joined until their contract is defined.",
            f"Viewer examples cap raw decoder and candidate text at {_TEXT_LIMIT} and {_CANDIDATE_LIMIT} characters.",
        ],
        "denominators": denominator_counts,
        "outcomes": outcomes,
        "funnel": funnel,
        "source_kind_reconciliation": source_kind_reconciliation,
        "candidate_invariants": {
            "validated_samples": len(results),
            "final_candidate_rows": sum(
                result.final_candidate_count for result in results.values()
            ),
            "successful_candidate_rows": sum(
                result.final_candidate_count
                for result in results.values()
                if result.outcome == _SUCCESS
            ),
        },
        "tables": {
            "outcome_by_dimension": outcome_rows,
            "candidate_multiplicity": multiplicity,
            "origin_contribution": origin_rows,
            "failure_modes": failures,
            "compile_warnings": warnings,
        },
        "rejection_sample_count": len(rejected_sample_ids),
    }


def _denominators(samples: Iterable[Sample]) -> dict[str, int]:
    values = list(samples)
    return {
        "all": len(values),
        "present": sum(sample.decoder_present for sample in values),
        "nonblank": sum(sample.decoder_nonblank for sample in values),
    }


def _source_kind_reconciliation(
    samples: Mapping[str, Sample],
) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str]] = Counter()
    for sample in samples.values():
        kind = sample.dimensions["source_kind"]
        counts[(kind, "all")] += 1
        if sample.decoder_present:
            counts[(kind, "present")] += 1
        if sample.decoder_nonblank:
            counts[(kind, "nonblank")] += 1
    rows: list[dict[str, object]] = []
    total = len(samples)
    for kind in sorted({kind for kind, _ in counts}):
        sample_count = counts[(kind, "all")]
        rows.append(
            {
                "source_kind": kind,
                "source_kind_value_state": (
                    "null"
                    if kind == _MISSING
                    else "blank"
                    if kind == _BLANK
                    else "nonblank"
                ),
                "sample_count": sample_count,
                "sample_rate_of_all": _rate(sample_count, total),
                "decoder_output_missing_count": sample_count
                - counts[(kind, "present")],
                "decoder_output_present_count": counts[(kind, "present")],
                "decoder_output_blank_count": counts[(kind, "present")]
                - counts[(kind, "nonblank")],
                "decoder_output_nonblank_count": counts[(kind, "nonblank")],
            }
        )
    return rows


def _funnel(
    results: Mapping[str, Result],
    samples: Mapping[str, Sample],
    candidate_stats: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    total = len(results)
    present = sum(sample.decoder_present for sample in samples.values())
    nonblank = sum(sample.decoder_nonblank for sample in samples.values())
    success = sum(result.outcome == _SUCCESS for result in results.values())
    candidates = sum(
        result.final_candidate_count for result in results.values()
    )
    converged = sum(
        _int_at(stats, "converged") for stats in candidate_stats.values()
    )
    return [
        _sample_funnel_row("source samples", total, total),
        _sample_funnel_row("decoder output present", present, total),
        _sample_funnel_row("decoder output nonblank", nonblank, total),
        _sample_funnel_row("function candidates extracted", success, total),
        {
            "stage": "final candidates",
            "unit": "candidate_row",
            "count": candidates,
            "rate_label": "candidates per extracted sample",
            "rate": _rate(candidates, success),
        },
        {
            "stage": "final candidates with converged origins",
            "unit": "candidate_row",
            "count": converged,
            "rate_label": "share of final candidate rows",
            "rate": _rate(converged, candidates),
        },
    ]


def _sample_funnel_row(
    stage: str, count: int, total: int
) -> dict[str, object]:
    return {
        "stage": stage,
        "unit": "sample",
        "count": count,
        "rate_label": "share of all samples",
        "rate": _rate(count, total),
    }


def _outcome_summary(
    results: Mapping[str, Result],
    samples: Mapping[str, Sample],
    denominators: Mapping[str, int],
) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str]] = Counter()
    outcomes = sorted({result.outcome for result in results.values()})
    for sample_id, sample in samples.items():
        outcome = results[sample_id].outcome
        for denominator in _memberships(sample):
            counts[(outcome, denominator)] += 1
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        row: dict[str, object] = {"outcome": outcome}
        for denominator in _RATE_DENOMINATORS:
            count = counts[(outcome, denominator)]
            row[f"count_{denominator}"] = count
            row[f"rate_of_{denominator}"] = _rate(
                count, denominators[denominator]
            )
        rows.append(row)
    return rows


def _outcome_rows(
    results: Mapping[str, Result], samples: Mapping[str, Sample]
) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    denoms: Counter[tuple[str, str, str]] = Counter()
    for sample_id, sample in samples.items():
        memberships = _memberships(sample)
        outcome = results[sample_id].outcome
        for dimension, value in sample.dimensions.items():
            for denominator in memberships:
                denoms[(dimension, value, denominator)] += 1
                counts[(dimension, value, denominator, outcome)] += 1
    rows: list[dict[str, object]] = []
    for (dimension, value, denominator, outcome), count in sorted(
        counts.items()
    ):
        total = denoms[(dimension, value, denominator)]
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "denominator": denominator,
                "outcome": outcome,
                "count": count,
                "denominator_count": total,
                "rate": _rate(count, total),
            }
        )
    return rows


def _multiplicity_rows(
    results: Mapping[str, Result], samples: Mapping[str, Sample]
) -> list[dict[str, object]]:
    counts: Counter[tuple[str, int]] = Counter()
    for sample_id, sample in samples.items():
        for denominator in _memberships(sample):
            counts[
                (denominator, results[sample_id].final_candidate_count)
            ] += 1
    rows: list[dict[str, object]] = []
    for (denominator, count), samples_count in sorted(counts.items()):
        denominator_count = sum(
            value for (name, _), value in counts.items() if name == denominator
        )
        rows.append(
            {
                "denominator": denominator,
                "final_candidate_count": count,
                "sample_count": samples_count,
                "denominator_count": denominator_count,
                "rate": _rate(samples_count, denominator_count),
            }
        )
    return rows


def _origin_rows(
    initial: Mapping[tuple[str, str], int],
    final: Mapping[tuple[str, str], int],
    converged: Mapping[tuple[str, str], int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(set(initial) | set(final)):
        extracted = initial.get(key, 0)
        retained = final.get(key, 0)
        converged_count = converged.get(key, 0)
        rows.append(
            {
                "variant": key[0],
                "strategy": key[1],
                "extracted_candidate_count": extracted,
                "final_candidate_origin_count": retained,
                "recovery_rate": _rate(retained, extracted),
                "converged_final_candidate_count": converged_count,
                "convergence_rate": _rate(converged_count, retained),
            }
        )
    return rows


def _failure_rows(
    rejections: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (scope, step, reason), value in sorted(rejections.items()):
        sample_ids = _string_set_at(value, "sample_ids")
        rows.append(
            {
                "scope": scope,
                "failed_step": step,
                "reason": reason,
                "count": _int_at(value, "count"),
                "sample_count": len(sample_ids),
            }
        )
    return rows


def _warning_rows(
    warning_stats: Mapping[str, int],
    candidate_stats: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    total_candidates = sum(
        _int_at(stats, "count") for stats in candidate_stats.values()
    )
    return [
        {
            "warning": warning,
            "candidate_count": count,
            "candidate_rate": _rate(count, total_candidates),
        }
        for warning, count in sorted(warning_stats.items())
    ]


def _build_examples(
    *,
    corpus_file: Path,
    run_root: Path,
    results: Mapping[str, Result],
    samples: Mapping[str, Sample],
    candidate_stats: Mapping[str, Mapping[str, object]],
    rejected_sample_ids: set[str],
) -> list[dict[str, object]]:
    chosen: dict[str, str] = {}
    for sample_id, sample in samples.items():
        result = results[sample_id]
        categories = [
            f"outcome:{result.outcome}",
            f"source_kind:{sample.dimensions['source_kind']}",
        ]
        categories.append(
            "multiplicity:multiple"
            if result.final_candidate_count > 1
            else f"multiplicity:{result.final_candidate_count}"
        )
        if sample_id in rejected_sample_ids:
            categories.append("has_rejection")
        for category in categories:
            if _is_better_example(sample_id, chosen.get(category)):
                chosen[category] = sample_id
    selected = sorted(set(chosen.values()), key=_example_rank)[:_EXAMPLE_LIMIT]
    raw_rows = _selected_corpus_rows(corpus_file, set(selected))
    candidates = _selected_relation_rows(
        run_root / "candidates.parquet",
        set(selected),
        [
            "sample_id",
            "candidate_index",
            "candidate_id",
            "cleaned_source",
            "origins",
            "compile_warnings",
            "top_level_function_names",
        ],
    )
    facts = _selected_relation_rows(
        run_root / "step_facts.parquet",
        set(selected),
        ["sample_id", "step_name", "facts_json"],
    )
    rejections = _selected_relation_rows(
        run_root / "rejections.parquet",
        set(selected),
        ["sample_id", "step_name", "reason_code", "details_json"],
    )
    by_candidate = _group_rows(candidates)
    by_fact = _group_rows(facts)
    by_rejection = _group_rows(rejections)
    examples: list[dict[str, object]] = []
    category_by_id: dict[str, list[str]] = defaultdict(list)
    for category, sample_id in chosen.items():
        if sample_id in selected:
            category_by_id[sample_id].append(category)
    for sample_id in selected:
        row = raw_rows[sample_id]
        raw = row.get("decoder_output")
        examples.append(
            {
                "sample_id": sample_id,
                "categories": sorted(category_by_id[sample_id]),
                "outcome": results[sample_id].outcome,
                "final_candidate_count": results[
                    sample_id
                ].final_candidate_count,
                "context": {
                    key: samples[sample_id].dimensions[key]
                    for key in (
                        "source_kind",
                        "source_database_table",
                        "model",
                        "prompt_fidelity",
                        "task_id",
                        "date_day",
                    )
                },
                "raw_decoder_output": _truncate(
                    raw if isinstance(raw, str) else None, _TEXT_LIMIT
                ),
                "candidates": [
                    {
                        "candidate_index": value.get("candidate_index"),
                        "candidate_id": value.get("candidate_id"),
                        "origins": value.get("origins"),
                        "top_level_function_names": value.get(
                            "top_level_function_names"
                        ),
                        "compile_warnings": value.get("compile_warnings"),
                        "cleaned_source": _truncate(
                            _str_or_none(value.get("cleaned_source")),
                            _CANDIDATE_LIMIT,
                        ),
                    }
                    for value in sorted(
                        by_candidate.get(sample_id, []),
                        key=lambda value: int(value["candidate_index"]),
                    )
                ],
                "facts": [
                    {
                        "step_name": value.get("step_name"),
                        "facts_json": _truncate(
                            _str_or_none(value.get("facts_json")), _FACTS_LIMIT
                        ),
                    }
                    for value in sorted(
                        by_fact.get(sample_id, []),
                        key=lambda value: _display(value.get("step_name")),
                    )
                ],
                "rejections": [
                    {
                        "step_name": value.get("step_name"),
                        "reason_code": value.get("reason_code"),
                        "details_json": _truncate(
                            _str_or_none(value.get("details_json")),
                            _FACTS_LIMIT,
                        ),
                    }
                    for value in sorted(
                        by_rejection.get(sample_id, []),
                        key=lambda value: (
                            _display(value.get("step_name")),
                            _display(value.get("reason_code")),
                        ),
                    )[:_REJECTIONS_LIMIT]
                ],
            }
        )
    return examples


def _selected_corpus_rows(
    path: Path, selected: set[str]
) -> dict[str, dict[str, object]]:
    columns = ["sample_id", "decoder_output"]
    rows: dict[str, dict[str, object]] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=32_768, columns=columns
    ):
        for row in batch.to_pylist():
            sample_id = row["sample_id"]
            if sample_id in selected:
                rows[sample_id] = row
    if set(rows) != selected:
        raise PreprocessingAnalysisError(
            "selected examples could not be reread"
        )
    return rows


def _selected_relation_rows(
    path: Path, selected: set[str], columns: list[str]
) -> list[dict[str, object]]:
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=65_536, columns=columns
    ):
        rows.extend(
            row for row in batch.to_pylist() if row["sample_id"] in selected
        )
    return rows


def _group_rows(
    rows: Iterable[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        sample_id = row["sample_id"]
        assert isinstance(sample_id, str)
        grouped[sample_id].append(row)
    return grouped


def _is_better_example(candidate: str, current: str | None) -> bool:
    return current is None or _example_rank(candidate) < _example_rank(current)


def _example_rank(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode()).hexdigest()


def _memberships(sample: Sample) -> tuple[str, ...]:
    values = ["all"]
    if sample.decoder_present:
        values.append("present")
    if sample.decoder_nonblank:
        values.append("nonblank")
    return tuple(values)


def _rate_row(
    row: dict[str, object], count: int, denominators: Mapping[str, int]
) -> dict[str, object]:
    return {
        **row,
        **{
            f"rate_of_{name}": _rate(count, total)
            for name, total in denominators.items()
        },
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 8)


def _write_deliverables(
    destination: Path,
    summary: Mapping[str, object],
    examples: list[dict[str, object]],
) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        tables = cast(dict[str, list[dict[str, object]]], summary["tables"])
        summary_payload = dict(summary)
        summary_payload["tables"] = {
            name: {"path": f"tables/{name}.parquet", "row_count": len(rows)}
            for name, rows in tables.items()
        }
        _write_json(temporary / "summary.json", summary_payload)
        _write_json(
            temporary / "viewer-data.json",
            _viewer_payload(summary, tables, examples),
            compact=True,
        )
        table_dir = temporary / "tables"
        table_dir.mkdir()
        for name in _COMPACT_TABLES:
            rows = tables[name]
            assert isinstance(rows, list)
            _write_table(table_dir / f"{name}.parquet", rows)
        (temporary / "report.md").write_text(
            _report(summary, examples), encoding="utf-8"
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_table(path: Path, rows: Iterable[dict[str, object]]) -> None:
    normalized = [row for row in rows if isinstance(row, dict)]
    if normalized:
        table = pa.Table.from_pylist(normalized)
    else:
        table = pa.table({"empty": pa.array([], type=pa.string())})
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _viewer_payload(
    summary: Mapping[str, object],
    tables: Mapping[str, list[dict[str, object]]],
    examples: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "headline": {
            "denominators": summary["denominators"],
            "funnel": summary["funnel"],
            "outcomes": summary["outcomes"],
            "candidate_invariants": summary["candidate_invariants"],
            "source_kind_reconciliation": summary[
                "source_kind_reconciliation"
            ],
        },
        "failure_modes": tables["failure_modes"],
        "origin_contribution": tables["origin_contribution"],
        "candidate_multiplicity": tables["candidate_multiplicity"],
        "outcome_by_dimension": tables["outcome_by_dimension"],
        "compile_warnings": tables["compile_warnings"],
        "examples": examples,
    }


def _report(
    summary: Mapping[str, object], examples: list[dict[str, object]]
) -> str:
    denominators = cast(Mapping[str, object], summary["denominators"])
    outcomes = cast(list[dict[str, object]], summary["outcomes"])
    funnel = cast(list[dict[str, object]], summary["funnel"])
    lines = [
        "# Preprocessing corpus analysis",
        "",
        "## Scope",
        "",
        "This report is derived from the authoritative preprocessing Parquets. It validates the corpus-to-results join by unique `sample_id`; detailed cross-tabs and failure/origin tables are in `tables/`.",
        "",
        "## Denominators",
        "",
        "| denominator | count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {name} | {count} |" for name, count in denominators.items()
    )
    lines.extend(
        [
            "",
            "## Funnel",
            "",
            "| stage | unit | count | metric | rate |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    lines.extend(
        f"| {row['stage']} | {row['unit']} | {row['count']} | {row['rate_label']} | {row['rate'] if row['rate'] is not None else 'n/a'} |"
        for row in funnel
    )
    lines.extend(
        [
            "",
            "## Outcomes",
            "",
            "| outcome | all count (rate) | present count (rate) | nonblank count (rate) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {row['outcome']} | {_count_rate(row, 'all')} | "
        f"{_count_rate(row, 'present')} | "
        f"{_count_rate(row, 'nonblank')} |"
        for row in outcomes
    )
    success = next(row for row in outcomes if row["outcome"] == _SUCCESS)
    failure_modes = cast(
        list[dict[str, object]],
        cast(Mapping[str, object], summary["tables"])["failure_modes"],
    )
    leading_failure = max(failure_modes, key=lambda row: int(row["count"]))
    tables = cast(Mapping[str, object], summary["tables"])
    multiplicity = cast(
        list[dict[str, object]], tables["candidate_multiplicity"]
    )
    multiple_samples = sum(
        _row_int(row, "sample_count")
        for row in multiplicity
        if row["denominator"] == "all"
        and _row_int(row, "final_candidate_count") > 1
    )
    multiple_rate = _rate(multiple_samples, _mapping_int(denominators, "all"))
    origins = cast(list[dict[str, object]], tables["origin_contribution"])
    leading_origin = max(
        origins, key=lambda row: int(row["final_candidate_origin_count"])
    )
    origin_attributions = sum(
        _row_int(row, "final_candidate_origin_count") for row in origins
    )
    source_kinds = cast(
        list[dict[str, object]], summary["source_kind_reconciliation"]
    )
    leading_source_kind = max(
        source_kinds, key=lambda row: int(row["sample_count"])
    )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"- {success['rate_of_present']:.2%} of present decoder outputs produced at least one final top-level-function candidate.",
            f"- The most frequent candidate rejection was `{leading_failure['reason']}` at `{leading_failure['failed_step']}` ({leading_failure['count']} rejection rows across {leading_failure['sample_count']} samples).",
            f"- {multiple_samples} samples ({multiple_rate:.2%} of all samples) retained multiple final candidates; candidate rows and sample outcomes are therefore reported separately.",
            f"- `{leading_origin['variant']}` / `{leading_origin['strategy']}` supplied {leading_origin['final_candidate_origin_count']} of {origin_attributions} final-origin attributions; its recovery rate is {leading_origin['recovery_rate']:.2%} from extracted candidates.",
            f"- `{leading_source_kind['source_kind']}` is the largest source kind with {leading_source_kind['sample_count']} samples ({leading_source_kind['sample_rate_of_all']:.2%} of all samples).",
        ]
    )
    lines.extend(
        [
            "",
            "## Viewer data",
            "",
            f"`viewer-data.json` contains {len(examples)} deterministic, bounded examples. Raw text is intentionally truncated; the preprocessing run is authoritative for complete sources.",
            "",
            "## Limitations",
            "",
        ]
    )
    limitations = summary["limitations"]
    lines.extend(f"- {item}" for item in cast(list[object], limitations))
    return "\n".join(lines) + "\n"


def _provenance(
    corpus_file: Path,
    run_root: Path,
    optional_inputs: Mapping[str, Path | str | None],
) -> dict[str, object]:
    optional = {}
    for name, value in optional_inputs.items():
        if value is None:
            optional[name] = {"provided": False}
            continue
        path = Path(value).expanduser().resolve()
        optional[name] = {
            "provided": True,
            "exists": path.is_file(),
            "label": path.name,
            "sha256": _file_sha256(path) if path.is_file() else None,
        }
    return {
        "corpus": {
            "label": corpus_file.name,
            "sha256": _file_sha256(corpus_file),
        },
        "run": {"run_id": run_root.name},
        "run_manifest_sha256": _file_sha256(run_root / "manifest.json"),
        "relations": {
            name: _file_sha256(run_root / f"{name}.parquet")
            for name in ("results", "candidates", "step_facts", "rejections")
        },
        "analysis_module_sha256": _file_sha256(Path(__file__)),
        "optional_inputs": optional,
    }


def _require_columns(path: Path, required: set[str]) -> None:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(required.difference(names))
    if missing:
        raise PreprocessingAnalysisError(
            f"{path} is missing columns: {', '.join(missing)}"
        )


def _required_str(row: Mapping[str, object], key: str, path: Path) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise PreprocessingAnalysisError(f"{path} has invalid {key!r}")
    return value


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _display(value: object) -> str:
    if value is None:
        return _MISSING
    if isinstance(value, str):
        return _BLANK if not value.strip() else value
    return str(value).lower() if isinstance(value, bool) else str(value)


def _append_bad(stats: Mapping[str, object], value: str) -> None:
    _string_list_at(stats, "bad").append(value)


def _int_at(values: Mapping[str, object], key: str) -> int:
    value = values[key]
    assert isinstance(value, int)
    return value


def _int_set_at(values: Mapping[str, object], key: str) -> set[int]:
    value = values[key]
    assert isinstance(value, set) and all(
        isinstance(item, int) for item in value
    )
    return cast(set[int], value)


def _string_set_at(values: Mapping[str, object], key: str) -> set[str]:
    value = values[key]
    assert isinstance(value, set) and all(
        isinstance(item, str) for item in value
    )
    return cast(set[str], value)


def _string_list_at(values: Mapping[str, object], key: str) -> list[str]:
    value = values[key]
    assert isinstance(value, list) and all(
        isinstance(item, str) for item in value
    )
    return cast(list[str], value)


def _parse_json_object(value: object, path: Path) -> dict[str, object]:
    if not isinstance(value, str):
        raise PreprocessingAnalysisError(f"{path} facts_json is not a string")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise PreprocessingAnalysisError(f"{path} facts_json is not an object")
    return parsed


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None or len(value) <= limit:
        return value
    return value[:limit] + "… [truncated]"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreprocessingAnalysisError(f"JSON object expected: {path}")
    return value


def _write_json(path: Path, value: object, *, compact: bool = False) -> None:
    serialized = (
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        if compact
        else json.dumps(value, sort_keys=True, indent=2)
    )
    path.write_text(serialized + "\n", encoding="utf-8")


def _count_rate(row: Mapping[str, object], denominator: str) -> str:
    count = row[f"count_{denominator}"]
    rate = row[f"rate_of_{denominator}"]
    return f"{count} ({rate if rate is not None else 'n/a'})"


def _row_int(row: Mapping[str, object], key: str) -> int:
    return _mapping_int(row, key)


def _mapping_int(values: Mapping[str, object], key: str) -> int:
    value = values[key]
    assert isinstance(value, int)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

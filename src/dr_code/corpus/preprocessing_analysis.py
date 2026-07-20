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
from pydantic import ValidationError

from dr_code.metrics.definition import MetricsDefinition, metrics_definition_hash
from dr_code.metrics.names import MetricName
from dr_code.metrics.policy_example import derive_outcome
from dr_code.metrics.records import MetricRecord, RecordStatus


ANALYSIS_SCHEMA_VERSION: Final = 2
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
    "evaluation_funnel",
    "candidate_test_outcomes",
    "sample_best_test_outcomes",
    "test_success_by_origin",
    "test_success_by_multiplicity",
    "test_success_by_preprocessing_outcome",
    "test_success_by_dimension",
)
_EXAMPLE_LIMIT: Final = 30
_TEXT_LIMIT: Final = 1_200
_CANDIDATE_LIMIT: Final = 1_200
_FACTS_LIMIT: Final = 1_200
_REJECTIONS_LIMIT: Final = 8
_FAILURE_EXAMPLE_SHARD_LIMIT: Final = 100
_EVALUATION_EXAMPLE_LIMIT: Final = 24
_EVALUATION_CATEGORIES: Final = (
    "passed",
    "failed",
    "timed_out",
    "infrastructure_failure",
)
_CANDIDATE_EVALUATION_SCHEMA_VERSION: Final = 1
_EVALUATION_SEMANTIC_FIELDS: Final = (
    "snapshot_sha256",
    "sandbox_image",
    "runner_identity",
    "execution_fingerprint",
    "trusted_source_sha256",
    "metrics_definition",
    "metrics_definition_hash",
    "operator",
    "operator_settings",
    "metrics_profile",
    "python",
    "python_implementation",
    "completed_at",
)


class PreprocessingAnalysisError(ValueError):
    """The input corpus and preprocessing projections cannot be reconciled."""


@dataclass(frozen=True, slots=True)
class PreprocessingAnalysisArtifacts:
    """Paths emitted by one completed preprocessing analysis."""

    output_dir: Path
    summary_path: Path
    viewer_data_path: Path
    failure_examples_path: Path
    report_path: Path
    table_paths: Mapping[str, Path]


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


CandidateKey = tuple[str, int, str]


@dataclass(frozen=True, slots=True)
class CandidateTestResult:
    evaluation_key: str
    task_id: str
    cleaned_source: str
    source_sha256: str
    task_fingerprint: str
    metrics_profile: str
    operator: str
    record_status: str
    failure_type: str | None
    failure_message: str | None
    official_outcome: str | None
    category: str
    function_count: int | None
    best_function_name: str | None
    total_cases: int | None
    passed_count: int | None
    failed_count: int | None
    error_count: int | None
    timeout_count: int | None
    coverage_complete: bool | None


@dataclass(frozen=True, slots=True)
class CandidateMembership:
    sample_id: str
    candidate_id: str
    candidate_index: int
    task_id: str
    source_kind: str
    source_sha256: str
    task_fingerprint: str
    evaluation_key: str
    metrics_profile: str
    operator: str


@dataclass(slots=True)
class CandidateEvaluation:
    memberships: dict[CandidateKey, CandidateMembership]
    results: dict[str, CandidateTestResult]
    seen_candidates: set[CandidateKey]
    candidate_origins: dict[CandidateKey, tuple[tuple[str, str], ...]]
    provenance: Mapping[str, object]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FailureExampleGroup:
    failure_code: str
    failed_step: str
    examples: list[dict[str, object]]
    raw_character_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _MaterializedExamples:
    examples: list[dict[str, object]]
    raw_character_counts: Mapping[str, int]


def analyze_preprocessing_corpus(
    *,
    corpus_path: Path | str,
    run_dir: Path | str,
    output_dir: Path | str,
    candidate_membership_path: Path | str | None = None,
    candidate_results_path: Path | str | None = None,
    candidate_evaluation_manifest_path: Path | str | None = None,
) -> PreprocessingAnalysisArtifacts:
    """Validate a run and emit deterministic compact analysis deliverables.

    Candidate evaluation is optional, but its manifest, membership, and result
    relations must be supplied together.  When present, they are validated and
    joined to the authoritative final candidates before test metrics are
    derived.
    """
    corpus_file = Path(corpus_path).expanduser().resolve(strict=True)
    run_root = Path(run_dir).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"analysis output already exists: {destination}")
    _validate_run(run_root)

    results = _read_results(run_root / "results.parquet")
    samples, _dimension_counts = _read_and_join_corpus(corpus_file, results)
    evaluation = _read_candidate_evaluation(
        candidate_membership_path=candidate_membership_path,
        candidate_results_path=candidate_results_path,
        candidate_evaluation_manifest_path=(
            candidate_evaluation_manifest_path
        ),
        samples=samples,
        corpus_file=corpus_file,
        run_root=run_root,
    )
    candidate_stats, origin_final, origin_converged, warning_stats = (
        _read_candidates(
            run_root / "candidates.parquet", results, evaluation=evaluation
        )
    )
    initial_origins = _read_initial_origins(run_root / "step_facts.parquet")
    rejections, rejected_sample_ids = _read_rejections(
        run_root / "rejections.parquet", results
    )
    _validate_candidate_invariants(results, candidate_stats)
    _validate_evaluation_coverage(evaluation)
    summary = _build_summary(
        results=results,
        samples=samples,
        candidate_stats=candidate_stats,
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
            "candidate_evaluation_manifest": (
                candidate_evaluation_manifest_path
            ),
        },
        evaluation=evaluation,
    )
    examples = _build_examples(
        corpus_file=corpus_file,
        run_root=run_root,
        results=results,
        samples=samples,
        candidate_stats=candidate_stats,
        rejected_sample_ids=rejected_sample_ids,
    )
    failure_examples = _build_failure_examples(
        corpus_file=corpus_file,
        run_root=run_root,
        results=results,
        samples=samples,
    )
    evaluation_examples = _build_evaluation_examples(evaluation, samples)
    return _write_deliverables(
        destination,
        summary,
        examples,
        failure_examples,
        evaluation_examples,
    )


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


def _read_candidate_evaluation(
    *,
    candidate_membership_path: Path | str | None,
    candidate_results_path: Path | str | None,
    candidate_evaluation_manifest_path: Path | str | None,
    samples: Mapping[str, Sample],
    corpus_file: Path,
    run_root: Path,
) -> CandidateEvaluation | None:
    optional_paths = (
        candidate_membership_path,
        candidate_results_path,
        candidate_evaluation_manifest_path,
    )
    provided_count = sum(value is not None for value in optional_paths)
    if provided_count not in {0, 3}:
        raise PreprocessingAnalysisError(
            "candidate evaluation manifest, membership, and results paths "
            "must be supplied together"
        )
    if provided_count == 0:
        return None
    assert candidate_membership_path is not None
    assert candidate_results_path is not None
    assert candidate_evaluation_manifest_path is not None
    membership_path = _required_optional_file(
        candidate_membership_path, "candidate membership"
    )
    results_path = _required_optional_file(
        candidate_results_path, "candidate results"
    )
    manifest_path = _required_optional_file(
        candidate_evaluation_manifest_path,
        "candidate evaluation manifest",
    )
    manifest, provenance, limitations = _read_evaluation_manifest(
        manifest_path=manifest_path,
        membership_path=membership_path,
        results_path=results_path,
        corpus_file=corpus_file,
        run_root=run_root,
    )
    evaluation_results = _read_candidate_test_results(results_path)
    memberships = _read_candidate_memberships(
        membership_path, evaluation_results, samples
    )
    referenced_keys = {
        membership.evaluation_key for membership in memberships.values()
    }
    unreferenced = set(evaluation_results).difference(referenced_keys)
    if unreferenced:
        raise PreprocessingAnalysisError(
            "candidate results contain unreferenced evaluation_key: "
            + min(unreferenced)
        )
    _validate_evaluation_coordinates(
        manifest, memberships, evaluation_results
    )
    return CandidateEvaluation(
        memberships=memberships,
        results=evaluation_results,
        seen_candidates=set(),
        candidate_origins={},
        provenance=provenance,
        limitations=limitations,
    )


def _required_optional_file(value: Path | str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise PreprocessingAnalysisError(f"{label} file does not exist: {path}")
    return path


def _read_evaluation_manifest(
    *,
    manifest_path: Path,
    membership_path: Path,
    results_path: Path,
    corpus_file: Path,
    run_root: Path,
) -> tuple[dict[str, object], dict[str, object], tuple[str, ...]]:
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreprocessingAnalysisError(
            "candidate evaluation manifest is invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PreprocessingAnalysisError(
            "candidate evaluation manifest must be a JSON object"
        )
    manifest = cast(dict[str, object], value)
    if manifest.get("schema_version") != _CANDIDATE_EVALUATION_SCHEMA_VERSION:
        raise PreprocessingAnalysisError(
            "unsupported candidate evaluation manifest schema_version"
        )
    if manifest.get("complete") is not True:
        raise PreprocessingAnalysisError(
            "candidate evaluation manifest is incomplete"
        )
    expected_counts = {
        "membership_rows": pq.ParquetFile(membership_path).metadata.num_rows,
        "result_rows": pq.ParquetFile(results_path).metadata.num_rows,
    }
    for field, expected in expected_counts.items():
        actual = manifest.get(field)
        if (
            not isinstance(actual, int)
            or isinstance(actual, bool)
            or actual < 0
            or actual != expected
        ):
            raise PreprocessingAnalysisError(
                f"candidate evaluation manifest {field} mismatch: "
                f"{actual!r} != {expected}"
            )
    expected_hashes = {
        "preprocessing_manifest_sha256": _file_sha256(
            run_root / "manifest.json"
        ),
        "preprocessing_candidates_sha256": _file_sha256(
            run_root / "candidates.parquet"
        ),
        "preprocessing_results_sha256": _file_sha256(
            run_root / "results.parquet"
        ),
        "corpus_sha256": _file_sha256(corpus_file),
    }
    for field, expected in expected_hashes.items():
        if manifest.get(field) != expected:
            raise PreprocessingAnalysisError(
                f"candidate evaluation manifest {field} mismatch"
            )
    _manifest_required_str(manifest, "metrics_profile")
    _manifest_required_str(manifest, "operator")
    missing_semantic = [
        field for field in _EVALUATION_SEMANTIC_FIELDS if field not in manifest
    ]
    if missing_semantic:
        raise PreprocessingAnalysisError(
            "candidate evaluation schema v1 manifest is missing field(s): "
            + ", ".join(missing_semantic)
        )
    _validate_v1_manifest_semantics(manifest)
    limitations = [
        "The candidate evaluation manifest does not contain membership/results file hashes; same-run linkage is validated through row counts, full relational joins, singleton coordinates, and deterministic evaluation keys."
    ]
    semantic_coordinates = {
        field: manifest[field]
        for field in _EVALUATION_SEMANTIC_FIELDS
        if field in manifest
    }
    provenance = {
        "manifest": {
            "label": manifest_path.name,
            "sha256": _file_sha256(manifest_path),
            "schema_version": manifest["schema_version"],
            "complete": True,
            "membership_rows": manifest["membership_rows"],
            "result_rows": manifest["result_rows"],
        },
        "semantic_coordinates": semantic_coordinates,
    }
    return manifest, provenance, tuple(limitations)


def _validate_v1_manifest_semantics(
    manifest: Mapping[str, object],
) -> None:
    for field in (
        "snapshot_sha256",
        "runner_identity",
        "execution_fingerprint",
        "metrics_definition_hash",
        "python",
        "python_implementation",
        "completed_at",
    ):
        _manifest_required_str(manifest, field)
    for field in (
        "trusted_source_sha256",
        "metrics_definition",
        "operator_settings",
    ):
        if not isinstance(manifest.get(field), Mapping):
            raise PreprocessingAnalysisError(
                "candidate evaluation manifest field must be an object: "
                + field
            )
    sandbox_image = manifest.get("sandbox_image")
    if sandbox_image is not None and not isinstance(sandbox_image, str):
        raise PreprocessingAnalysisError(
            "candidate evaluation manifest sandbox_image must be string or null"
        )
    try:
        definition = MetricsDefinition.model_validate(
            manifest["metrics_definition"]
        )
    except ValidationError as exc:
        raise PreprocessingAnalysisError(
            "candidate evaluation manifest metrics_definition is invalid"
        ) from exc
    if (
        metrics_definition_hash(definition)
        != manifest["metrics_definition_hash"]
    ):
        raise PreprocessingAnalysisError(
            "candidate evaluation manifest metrics_definition_hash mismatch"
        )
    if (
        len(definition.questions) != 1
        or definition.questions[0].metric is not MetricName.CODE_TEST
        or definition.questions[0].settings != manifest["operator_settings"]
    ):
        raise PreprocessingAnalysisError(
            "candidate evaluation manifest operator settings differ from "
            "metrics_definition"
        )


def _validate_evaluation_coordinates(
    manifest: Mapping[str, object],
    memberships: Mapping[CandidateKey, CandidateMembership],
    results: Mapping[str, CandidateTestResult],
) -> None:
    expected_profile = _manifest_required_str(manifest, "metrics_profile")
    expected_operator = _manifest_required_str(manifest, "operator")
    profiles = {result.metrics_profile for result in results.values()}
    profiles.update(
        membership.metrics_profile for membership in memberships.values()
    )
    operators = {result.operator for result in results.values()}
    operators.update(
        membership.operator for membership in memberships.values()
    )
    if profiles and profiles != {expected_profile}:
        raise PreprocessingAnalysisError(
            "candidate evaluation rows have mixed or manifest-mismatched "
            "metrics_profile"
        )
    if operators and operators != {expected_operator}:
        raise PreprocessingAnalysisError(
            "candidate evaluation rows have mixed or manifest-mismatched operator"
        )
    definition_hash = _manifest_required_str(
        manifest, "metrics_definition_hash"
    )
    execution_fingerprint = _manifest_required_str(
        manifest, "execution_fingerprint"
    )
    operator_settings = manifest["operator_settings"]
    if not isinstance(operator_settings, Mapping):
        raise PreprocessingAnalysisError(
            "candidate evaluation manifest operator_settings must be an object"
        )
    for result in results.values():
        payload = {
            "task_id": result.task_id,
            "task_fingerprint": result.task_fingerprint,
            "candidate_source_sha256": result.source_sha256,
            "metrics_definition_hash": definition_hash,
            "metrics_profile": expected_profile,
            "operator": expected_operator,
            "settings": dict(operator_settings),
            "execution_fingerprint": execution_fingerprint,
        }
        expected_key = _text_sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        if result.evaluation_key != expected_key:
            raise PreprocessingAnalysisError(
                "candidate evaluation_key does not match manifest coordinates: "
                + result.evaluation_key
            )


def _manifest_required_str(
    manifest: Mapping[str, object], field: str
) -> str:
    value = manifest.get(field)
    if not isinstance(value, str) or not value:
        raise PreprocessingAnalysisError(
            f"candidate evaluation manifest has invalid {field!r}"
        )
    return value


def _read_candidate_test_results(
    path: Path,
) -> dict[str, CandidateTestResult]:
    required = {
        "evaluation_key",
        "task_id",
        "cleaned_source",
        "source_sha256",
        "task_fingerprint",
        "metrics_profile",
        "operator",
        "record_status",
        "failure_type",
        "failure_message",
        "outcome",
        "function_count",
        "best_function_name",
        "total_cases",
        "passed_count",
        "failed_count",
        "error_count",
        "timeout_count",
        "coverage_complete",
    }
    _require_columns(path, required)
    results: dict[str, CandidateTestResult] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=32_768, columns=sorted(required)
    ):
        for row in batch.to_pylist():
            evaluation_key = _required_str(row, "evaluation_key", path)
            if evaluation_key in results:
                raise PreprocessingAnalysisError(
                    "candidate result evaluation_key is not unique: "
                    + evaluation_key
                )
            cleaned_source = _required_str(row, "cleaned_source", path)
            source_sha256 = _required_str(row, "source_sha256", path)
            if _text_sha256(cleaned_source) != source_sha256:
                raise PreprocessingAnalysisError(
                    f"candidate result source_sha256 mismatch: {evaluation_key}"
                )
            record_status = _required_str(row, "record_status", path)
            official_outcome = _str_or_none(row.get("outcome"))
            measured = record_status == "measured"
            integer_facts = {
                name: _optional_nonnegative_int(row.get(name), name, path)
                for name in (
                    "function_count",
                    "total_cases",
                    "passed_count",
                    "failed_count",
                    "error_count",
                    "timeout_count",
                )
            }
            coverage_complete = row.get("coverage_complete")
            failure_type = _str_or_none(row.get("failure_type"))
            failure_message = _str_or_none(row.get("failure_message"))
            best_function_name = _str_or_none(row.get("best_function_name"))
            if measured and (
                any(value is None for value in integer_facts.values())
                or not isinstance(coverage_complete, bool)
            ):
                raise PreprocessingAnalysisError(
                    f"measured candidate result lacks test facts: {evaluation_key}"
                )
            category = _validate_candidate_test_result(
                evaluation_key=evaluation_key,
                record_status=record_status,
                official_outcome=official_outcome,
                integer_facts=integer_facts,
                best_function_name=best_function_name,
                coverage_complete=coverage_complete,
                failure_type=failure_type,
                failure_message=failure_message,
            )
            results[evaluation_key] = CandidateTestResult(
                evaluation_key=evaluation_key,
                task_id=_required_str(row, "task_id", path),
                cleaned_source=cleaned_source,
                source_sha256=source_sha256,
                task_fingerprint=_required_str(row, "task_fingerprint", path),
                metrics_profile=_required_str(row, "metrics_profile", path),
                operator=_required_str(row, "operator", path),
                record_status=record_status,
                failure_type=failure_type,
                failure_message=failure_message,
                official_outcome=official_outcome,
                category=category,
                function_count=integer_facts["function_count"],
                best_function_name=best_function_name,
                total_cases=integer_facts["total_cases"],
                passed_count=integer_facts["passed_count"],
                failed_count=integer_facts["failed_count"],
                error_count=integer_facts["error_count"],
                timeout_count=integer_facts["timeout_count"],
                coverage_complete=(
                    coverage_complete
                    if isinstance(coverage_complete, bool)
                    else None
                ),
            )
    return results


def _validate_candidate_test_result(
    *,
    evaluation_key: str,
    record_status: str,
    official_outcome: str | None,
    integer_facts: Mapping[str, int | None],
    best_function_name: str | None,
    coverage_complete: object,
    failure_type: str | None,
    failure_message: str | None,
) -> str:
    if record_status == "infrastructure_failure":
        if (
            official_outcome is not None
            or best_function_name is not None
            or coverage_complete is not None
            or any(value is not None for value in integer_facts.values())
        ):
            raise PreprocessingAnalysisError(
                "infrastructure result must not contain measured facts: "
                + evaluation_key
            )
        if failure_type is None or failure_message is None:
            raise PreprocessingAnalysisError(
                "infrastructure result requires failure diagnostics: "
                + evaluation_key
            )
        return "infrastructure_failure"
    if record_status != "measured" or official_outcome is None:
        raise PreprocessingAnalysisError(
            f"invalid candidate result status/outcome: {evaluation_key}"
        )
    if failure_type is not None or failure_message is not None:
        raise PreprocessingAnalysisError(
            "measured candidate result must not have failure diagnostics: "
            + evaluation_key
        )
    facts = {
        key: value
        for key, value in integer_facts.items()
        if value is not None
    }
    if len(facts) != len(integer_facts) or not isinstance(
        coverage_complete, bool
    ):
        raise PreprocessingAnalysisError(
            f"measured candidate result lacks test facts: {evaluation_key}"
        )
    _validate_measured_test_facts(
        facts,
        best_function_name=best_function_name,
        coverage_complete=coverage_complete,
        evaluation_key=evaluation_key,
    )
    derived_outcome = _derive_measured_outcome(facts, coverage_complete)
    if official_outcome != derived_outcome:
        raise PreprocessingAnalysisError(
            "candidate result outcome contradicts measured facts: "
            f"{evaluation_key} ({official_outcome!r} != {derived_outcome!r})"
        )
    if derived_outcome == "passed":
        return "passed"
    if derived_outcome == "timed_out":
        return "timed_out"
    if derived_outcome in {
        "tests_failed",
        "no_top_level_functions",
        "evaluation_incomplete",
    }:
        return "failed"
    raise PreprocessingAnalysisError(
        f"unknown candidate test outcome {derived_outcome!r}: {evaluation_key}"
    )


def _validate_measured_test_facts(
    facts: Mapping[str, int],
    *,
    best_function_name: str | None,
    coverage_complete: bool,
    evaluation_key: str,
) -> None:
    function_count = facts["function_count"]
    observed_cases = sum(
        facts[name]
        for name in (
            "passed_count",
            "failed_count",
            "error_count",
            "timeout_count",
        )
    )
    total_cases = facts["total_cases"]
    # Duplicate top-level function names intentionally stack status rows in the
    # producer.  That can make observed_cases exceed total_cases, in which case
    # coverage is incomplete; equality, rather than <=, defines coverage.
    if function_count == 0 and observed_cases != 0:
        raise PreprocessingAnalysisError(
            "candidate result has case statuses without a selected function: "
            + evaluation_key
        )
    if (function_count == 0) != (best_function_name is None):
        raise PreprocessingAnalysisError(
            "candidate result function_count/best_function_name mismatch: "
            + evaluation_key
        )
    expected_coverage = (
        best_function_name is not None and observed_cases == total_cases
    )
    if coverage_complete != expected_coverage:
        raise PreprocessingAnalysisError(
            f"candidate result coverage contradicts case counts: {evaluation_key}"
        )


def _derive_measured_outcome(
    facts: Mapping[str, int], coverage_complete: bool
) -> str:
    record = MetricRecord(
        metric=MetricName.CODE_TEST,
        metric_version="1",
        settings={},
        on_key="candidate",
        producer_id="candidate-evaluation",
        producer_version=None,
        producer_definition_hash=None,
        metrics_definition_id="candidate-evaluation",
        metrics_definition_version="1",
        status=RecordStatus.MEASURED,
        values={**facts, "coverage_complete": coverage_complete},
    )
    return str(derive_outcome(record))


def _read_candidate_memberships(
    path: Path,
    evaluation_results: Mapping[str, CandidateTestResult],
    samples: Mapping[str, Sample],
) -> dict[CandidateKey, CandidateMembership]:
    required = {
        "sample_id",
        "candidate_id",
        "candidate_index",
        "task_id",
        "source_kind",
        "source_sha256",
        "task_fingerprint",
        "evaluation_key",
        "metrics_profile",
        "operator",
    }
    _require_columns(path, required)
    memberships: dict[CandidateKey, CandidateMembership] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=32_768, columns=sorted(required)
    ):
        for row in batch.to_pylist():
            sample_id = _required_str(row, "sample_id", path)
            sample = samples.get(sample_id)
            if sample is None:
                raise PreprocessingAnalysisError(
                    f"candidate membership references unknown sample_id: {sample_id}"
                )
            candidate_id = _required_str(row, "candidate_id", path)
            candidate_index = row.get("candidate_index")
            if not isinstance(candidate_index, int) or candidate_index < 0:
                raise PreprocessingAnalysisError(
                    f"invalid candidate membership index: {sample_id}"
                )
            key = (sample_id, candidate_index, candidate_id)
            if key in memberships:
                raise PreprocessingAnalysisError(
                    f"candidate membership is not unique: {key}"
                )
            evaluation_key = _required_str(row, "evaluation_key", path)
            result = evaluation_results.get(evaluation_key)
            if result is None:
                raise PreprocessingAnalysisError(
                    "candidate membership references unknown evaluation_key: "
                    + evaluation_key
                )
            membership = CandidateMembership(
                sample_id=sample_id,
                candidate_id=candidate_id,
                candidate_index=candidate_index,
                task_id=_required_str(row, "task_id", path),
                source_kind=_display(row.get("source_kind")),
                source_sha256=_required_str(row, "source_sha256", path),
                task_fingerprint=_required_str(row, "task_fingerprint", path),
                evaluation_key=evaluation_key,
                metrics_profile=_required_str(row, "metrics_profile", path),
                operator=_required_str(row, "operator", path),
            )
            _validate_membership_coordinates(membership, result, sample)
            memberships[key] = membership
    return memberships


def _validate_membership_coordinates(
    membership: CandidateMembership,
    result: CandidateTestResult,
    sample: Sample,
) -> None:
    expected = (
        result.task_id,
        result.source_sha256,
        result.task_fingerprint,
        result.metrics_profile,
        result.operator,
    )
    actual = (
        membership.task_id,
        membership.source_sha256,
        membership.task_fingerprint,
        membership.metrics_profile,
        membership.operator,
    )
    if actual != expected:
        raise PreprocessingAnalysisError(
            "candidate membership/result coordinates differ: "
            + membership.evaluation_key
        )
    if membership.task_id != sample.dimensions["task_id"]:
        raise PreprocessingAnalysisError(
            f"candidate membership task_id differs from corpus: {membership.sample_id}"
        )
    if membership.source_kind != sample.dimensions["source_kind"]:
        raise PreprocessingAnalysisError(
            "candidate membership source_kind differs from corpus: "
            + membership.sample_id
        )


def _read_candidates(
    path: Path,
    results: Mapping[str, Result],
    *,
    evaluation: CandidateEvaluation | None,
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
        "cleaned_source",
        "source_sha256",
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
            candidate_key = (sample_id, index, candidate_id)
            if evaluation is not None:
                _join_candidate_evaluation(
                    evaluation, candidate_key, row, path
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


def _join_candidate_evaluation(
    evaluation: CandidateEvaluation,
    candidate_key: CandidateKey,
    row: Mapping[str, object],
    path: Path,
) -> None:
    membership = evaluation.memberships.get(candidate_key)
    if membership is None:
        raise PreprocessingAnalysisError(
            f"final candidate lacks evaluation membership: {candidate_key}"
        )
    cleaned_source = _required_str(row, "cleaned_source", path)
    source_sha256 = _required_str(row, "source_sha256", path)
    result = evaluation.results[membership.evaluation_key]
    if (
        source_sha256 != membership.source_sha256
        or source_sha256 != result.source_sha256
        or cleaned_source != result.cleaned_source
        or _text_sha256(cleaned_source) != source_sha256
    ):
        raise PreprocessingAnalysisError(
            f"candidate evaluation source differs from preprocessing: {candidate_key}"
        )
    if candidate_key in evaluation.seen_candidates:
        raise PreprocessingAnalysisError(
            f"duplicate final candidate during evaluation join: {candidate_key}"
        )
    origins_value = row.get("origins")
    origins = origins_value if isinstance(origins_value, list) else []
    normalized_origins: list[tuple[str, str]] = []
    for origin in origins:
        if not isinstance(origin, Mapping):
            continue
        variant = _str_or_none(origin.get("variant"))
        strategy = _str_or_none(origin.get("strategy"))
        if variant is not None and strategy is not None:
            normalized_origins.append((variant, strategy))
    evaluation.seen_candidates.add(candidate_key)
    evaluation.candidate_origins[candidate_key] = tuple(normalized_origins)


def _validate_evaluation_coverage(
    evaluation: CandidateEvaluation | None,
) -> None:
    if evaluation is None:
        return
    missing = set(evaluation.memberships).difference(
        evaluation.seen_candidates
    )
    if missing:
        raise PreprocessingAnalysisError(
            f"evaluation membership is not a final candidate: {min(missing)}"
        )


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
    origin_final: Mapping[tuple[str, str], int],
    origin_converged: Mapping[tuple[str, str], int],
    initial_origins: Mapping[tuple[str, str], int],
    rejections: Mapping[tuple[str, str, str], Mapping[str, object]],
    warning_stats: Mapping[str, int],
    rejected_sample_ids: set[str],
    corpus_file: Path,
    run_root: Path,
    optional_inputs: Mapping[str, Path | str | None],
    evaluation: CandidateEvaluation | None,
) -> dict[str, object]:
    denominator_counts = _denominators(samples.values())
    funnel = _funnel(results, samples, candidate_stats)
    outcomes = _outcome_summary(results, samples, denominator_counts)
    outcome_rows = _outcome_rows(results, samples)
    multiplicity = _multiplicity_rows(results, samples)
    origin_rows = _origin_rows(initial_origins, origin_final, origin_converged)
    failures = _failure_rows(rejections)
    warnings = _warning_rows(warning_stats, candidate_stats)
    evaluation_summary, evaluation_tables = _evaluation_analysis(
        evaluation, results, samples
    )
    source_kind_reconciliation = _source_kind_reconciliation(samples)
    if sum(row["sample_count"] for row in source_kind_reconciliation) != len(
        samples
    ):
        raise PreprocessingAnalysisError(
            "source_kind reconciliation is incomplete"
        )
    tables = {
        "outcome_by_dimension": outcome_rows,
        "candidate_multiplicity": multiplicity,
        "origin_contribution": origin_rows,
        "failure_modes": failures,
        "compile_warnings": warnings,
        **evaluation_tables,
    }
    limitations = [
        "Preprocessing metrics alone do not claim task correctness or execution success.",
        f"Viewer examples cap raw decoder and candidate text at {_TEXT_LIMIT} and {_CANDIDATE_LIMIT} characters.",
    ]
    if evaluation is None:
        limitations.append(
            "Candidate evaluation was not supplied; test outcomes and execution comparisons are unavailable."
        )
    else:
        limitations.append(
            "Candidate test rates describe HumanEval+ execution under the supplied evaluation profile; they do not generalize to non-HumanEval tasks."
        )
        limitations.extend(evaluation.limitations)
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "provenance": _provenance(corpus_file, run_root, optional_inputs),
        "limitations": limitations,
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
        "candidate_evaluation": evaluation_summary,
        "tables": tables,
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


def _evaluation_analysis(
    evaluation: CandidateEvaluation | None,
    preprocessing_results: Mapping[str, Result],
    samples: Mapping[str, Sample],
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    if evaluation is None:
        return {"available": False}, {}
    sample_outcomes = _sample_best_test_outcomes(evaluation)
    extracted_samples = {
        sample_id
        for sample_id, result in preprocessing_results.items()
        if result.final_candidate_count > 0
    }
    if set(sample_outcomes) != extracted_samples:
        missing = extracted_samples.symmetric_difference(sample_outcomes)
        raise PreprocessingAnalysisError(
            f"evaluation sample coverage differs from extracted samples: {min(missing)}"
        )

    category_counts = Counter(
        evaluation.results[membership.evaluation_key].category
        for membership in evaluation.memberships.values()
    )
    sample_counts = Counter(sample_outcomes.values())
    candidate_total = len(evaluation.memberships)
    measured_total = sum(
        category_counts[category]
        for category in ("passed", "failed", "timed_out")
    )
    evaluation_funnel = [
        _evaluation_funnel_row(
            "extracted final candidates", candidate_total, candidate_total
        ),
        _evaluation_funnel_row(
            "evaluation attempted candidates", candidate_total, candidate_total
        ),
        _evaluation_funnel_row(
            "tested candidates", measured_total, candidate_total
        ),
    ]
    evaluation_funnel.extend(
        _evaluation_funnel_row(
            f"candidate {category}", category_counts[category], candidate_total
        )
        for category in _EVALUATION_CATEGORIES
    )
    candidate_outcomes = _candidate_test_outcome_rows(evaluation)
    sample_best: list[dict[str, object]] = [
        {
            "best_test_outcome": category,
            "sample_count": sample_counts[category],
            "extracted_sample_count": len(extracted_samples),
            "rate_of_extracted_samples": _rate(
                sample_counts[category], len(extracted_samples)
            ),
        }
        for category in _EVALUATION_CATEGORIES
    ]
    tables: dict[str, list[dict[str, object]]] = {
        "evaluation_funnel": evaluation_funnel,
        "candidate_test_outcomes": candidate_outcomes,
        "sample_best_test_outcomes": sample_best,
        "test_success_by_origin": _test_success_by_origin(evaluation),
        "test_success_by_multiplicity": _test_success_by_multiplicity(
            sample_outcomes, preprocessing_results
        ),
        "test_success_by_preprocessing_outcome": (
            _test_success_by_preprocessing_outcome(
                sample_outcomes, preprocessing_results
            )
        ),
        "test_success_by_dimension": _test_success_by_dimension(
            sample_outcomes, preprocessing_results, samples
        ),
    }
    summary = {
        "available": True,
        "provenance": evaluation.provenance,
        "limitations": list(evaluation.limitations),
        "candidate_membership_count": candidate_total,
        "deduplicated_evaluation_count": len(evaluation.results),
        "extracted_sample_count": len(extracted_samples),
        "funnel": evaluation_funnel,
        "candidate_outcomes": candidate_outcomes,
        "sample_best_outcomes": sample_best,
    }
    return summary, tables


def _evaluation_funnel_row(
    stage: str, count: int, extracted_candidate_count: int
) -> dict[str, object]:
    return {
        "stage": stage,
        "unit": "candidate_row",
        "count": count,
        "denominator": "extracted_final_candidates",
        "denominator_count": extracted_candidate_count,
        "rate": _rate(count, extracted_candidate_count),
    }


def _candidate_test_outcome_rows(
    evaluation: CandidateEvaluation,
) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    total = len(evaluation.memberships)
    for membership in evaluation.memberships.values():
        result = evaluation.results[membership.evaluation_key]
        counts[
            (
                result.category,
                _display(result.official_outcome),
                result.record_status,
                _display(result.failure_type),
            )
        ] += 1
    return [
        {
            "test_outcome": category,
            "official_outcome": official_outcome,
            "record_status": record_status,
            "failure_type": failure_type,
            "candidate_count": count,
            "candidate_denominator_count": total,
            "candidate_rate": _rate(count, total),
        }
        for (
            category,
            official_outcome,
            record_status,
            failure_type,
        ), count in sorted(counts.items())
    ]


def _sample_best_test_outcomes(
    evaluation: CandidateEvaluation,
) -> dict[str, str]:
    categories: dict[str, list[str]] = defaultdict(list)
    for membership in evaluation.memberships.values():
        categories[membership.sample_id].append(
            evaluation.results[membership.evaluation_key].category
        )
    priority = {category: index for index, category in enumerate(_EVALUATION_CATEGORIES)}
    return {
        sample_id: min(values, key=lambda value: priority[value])
        for sample_id, values in categories.items()
    }


def _test_success_by_origin(
    evaluation: CandidateEvaluation,
) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for candidate_key, membership in evaluation.memberships.items():
        category = evaluation.results[membership.evaluation_key].category
        for variant, strategy in evaluation.candidate_origins[candidate_key]:
            counts[(variant, strategy, category)] += 1
    rows: list[dict[str, object]] = []
    origins = sorted({(variant, strategy) for variant, strategy, _ in counts})
    for variant, strategy in origins:
        outcome_counts = {
            category: counts[(variant, strategy, category)]
            for category in _EVALUATION_CATEGORIES
        }
        total = sum(outcome_counts.values())
        rows.append(
            {
                "variant": variant,
                "strategy": strategy,
                "unit": "final_candidate_origin_attribution",
                "candidate_origin_count": total,
                **{
                    f"{category}_count": count
                    for category, count in outcome_counts.items()
                },
                "pass_rate": _rate(outcome_counts["passed"], total),
            }
        )
    return rows


def _test_success_by_multiplicity(
    sample_outcomes: Mapping[str, str],
    preprocessing_results: Mapping[str, Result],
) -> list[dict[str, object]]:
    counts: Counter[tuple[int, str]] = Counter()
    for sample_id, category in sample_outcomes.items():
        counts[(preprocessing_results[sample_id].final_candidate_count, category)] += 1
    rows: list[dict[str, object]] = []
    for multiplicity in sorted({value for value, _ in counts}):
        rows.append(
            _sample_comparison_row(
                "final_candidate_count",
                str(multiplicity),
                {
                    category: counts[(multiplicity, category)]
                    for category in _EVALUATION_CATEGORIES
                },
            )
        )
    return rows


def _test_success_by_preprocessing_outcome(
    sample_outcomes: Mapping[str, str],
    preprocessing_results: Mapping[str, Result],
) -> list[dict[str, object]]:
    categories = (*_EVALUATION_CATEGORIES, "not_extracted")
    counts: Counter[tuple[str, str]] = Counter()
    for sample_id, result in preprocessing_results.items():
        category = sample_outcomes.get(sample_id, "not_extracted")
        counts[(result.outcome, category)] += 1
    rows: list[dict[str, object]] = []
    for outcome in sorted({outcome for outcome, _ in counts}):
        outcome_counts = {
            category: counts[(outcome, category)] for category in categories
        }
        rows.append(
            _sample_comparison_row(
                "preprocessing_outcome", outcome, outcome_counts
            )
        )
    return rows


def _test_success_by_dimension(
    sample_outcomes: Mapping[str, str],
    preprocessing_results: Mapping[str, Result],
    samples: Mapping[str, Sample],
) -> list[dict[str, object]]:
    categories = (*_EVALUATION_CATEGORIES, "not_extracted")
    counts: Counter[tuple[str, str, str]] = Counter()
    for sample_id, sample in samples.items():
        category = sample_outcomes.get(sample_id, "not_extracted")
        if preprocessing_results[sample_id].final_candidate_count > 0:
            assert category != "not_extracted"
        for dimension, value in sample.dimensions.items():
            counts[(dimension, value, category)] += 1
    rows: list[dict[str, object]] = []
    groups = sorted({(dimension, value) for dimension, value, _ in counts})
    for dimension, value in groups:
        rows.append(
            _sample_comparison_row(
                dimension,
                value,
                {
                    category: counts[(dimension, value, category)]
                    for category in categories
                },
            )
        )
    return rows


def _sample_comparison_row(
    dimension: str,
    value: str,
    outcome_counts: Mapping[str, int],
) -> dict[str, object]:
    sample_count = sum(outcome_counts.values())
    evaluated_count = sample_count - outcome_counts.get("not_extracted", 0)
    passed = outcome_counts.get("passed", 0)
    return {
        "dimension": dimension,
        "value": value,
        "sample_count": sample_count,
        "evaluated_sample_count": evaluated_count,
        **{
            f"{category}_count": count
            for category, count in outcome_counts.items()
        },
        "pass_rate_of_all_samples": _rate(passed, sample_count),
        "pass_rate_of_evaluated_samples": _rate(passed, evaluated_count),
    }


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
    categories_by_id: dict[str, list[str]] = defaultdict(list)
    for category, sample_id in chosen.items():
        if sample_id in selected:
            categories_by_id[sample_id].append(category)
    return _materialize_examples(
        corpus_file=corpus_file,
        run_root=run_root,
        selected=selected,
        results=results,
        samples=samples,
        categories_by_id=categories_by_id,
        raw_text_limit=_TEXT_LIMIT,
    ).examples


def _build_failure_examples(
    *,
    corpus_file: Path,
    run_root: Path,
    results: Mapping[str, Result],
    samples: Mapping[str, Sample],
) -> dict[tuple[str, str], _FailureExampleGroup]:
    grouped_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sample_id, sample in samples.items():
        result = results[sample_id]
        if not sample.decoder_nonblank or result.final_candidate_count != 0:
            continue
        if not result.failure_code or not result.failed_step:
            raise PreprocessingAnalysisError(
                "nonblank zero-candidate result has no terminal failure: "
                + sample_id
            )
        grouped_ids[(result.failure_code, result.failed_step)].append(sample_id)

    selected = sorted(
        sample_id
        for group_ids in grouped_ids.values()
        for sample_id in group_ids
    )
    categories_by_id = {
        sample_id: [
            f"outcome:{results[sample_id].outcome}",
            f"failure_code:{results[sample_id].failure_code}",
        ]
        for sample_id in selected
    }
    # Failure shards intentionally carry only the terminal step's facts.
    # Full fact histories would duplicate a large relation in static JSON.
    materialized = _materialize_examples(
        corpus_file=corpus_file,
        run_root=run_root,
        selected=selected,
        results=results,
        samples=samples,
        categories_by_id=categories_by_id,
        raw_text_limit=_TEXT_LIMIT,
        fact_step_by_id={
            sample_id: cast(str, results[sample_id].failed_step)
            for sample_id in selected
        },
    )
    examples_by_id = {
        cast(str, example["sample_id"]): example
        for example in materialized.examples
    }
    groups: dict[tuple[str, str], _FailureExampleGroup] = {}
    for failure_code, failed_step in sorted(grouped_ids):
        group_ids = sorted(grouped_ids[(failure_code, failed_step)])
        groups[(failure_code, failed_step)] = _FailureExampleGroup(
            failure_code=failure_code,
            failed_step=failed_step,
            examples=[examples_by_id[sample_id] for sample_id in group_ids],
            raw_character_counts={
                sample_id: materialized.raw_character_counts[sample_id]
                for sample_id in group_ids
            },
        )
    return groups


def _materialize_examples(
    *,
    corpus_file: Path,
    run_root: Path,
    selected: list[str],
    results: Mapping[str, Result],
    samples: Mapping[str, Sample],
    categories_by_id: Mapping[str, list[str]],
    raw_text_limit: int | None,
    fact_step_by_id: Mapping[str, str] | None = None,
) -> _MaterializedExamples:
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
    raw_character_counts: dict[str, int] = {}
    for sample_id in selected:
        row = raw_rows[sample_id]
        raw = row.get("decoder_output")
        raw_text = raw if isinstance(raw, str) else None
        raw_character_counts[sample_id] = len(raw_text or "")
        sample_facts = by_fact.get(sample_id, [])
        if fact_step_by_id is not None:
            sample_facts = [
                value
                for value in sample_facts
                if value.get("step_name") == fact_step_by_id[sample_id]
            ]
        examples.append(
            {
                "sample_id": sample_id,
                "categories": sorted(categories_by_id[sample_id]),
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
                "raw_decoder_output": (
                    raw_text
                    if raw_text_limit is None
                    else _truncate(raw_text, raw_text_limit)
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
                        sample_facts,
                        key=lambda value: _display(value.get("step_name")),
                    )
                ],
                "rejections": [
                    {
                        "step_name": value.get("step_name"),
                        "reason_code": _display(value.get("reason_code")),
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
    return _MaterializedExamples(
        examples=examples,
        raw_character_counts=raw_character_counts,
    )


def _build_evaluation_examples(
    evaluation: CandidateEvaluation | None,
    samples: Mapping[str, Sample],
) -> list[dict[str, object]]:
    if evaluation is None:
        return []
    chosen: dict[str, CandidateKey] = {}
    for candidate_key, membership in evaluation.memberships.items():
        result = evaluation.results[membership.evaluation_key]
        source_kind = samples[membership.sample_id].dimensions["source_kind"]
        categories = (
            f"test_outcome:{result.category}",
            f"test_outcome:{result.category};source_kind:{source_kind}",
        )
        for category in categories:
            current = chosen.get(category)
            if current is None or _candidate_example_rank(
                candidate_key
            ) < _candidate_example_rank(current):
                chosen[category] = candidate_key
    required = {
        chosen[f"test_outcome:{category}"]
        for category in _EVALUATION_CATEGORIES
        if f"test_outcome:{category}" in chosen
    }
    diverse = sorted(
        set(chosen.values()).difference(required),
        key=_candidate_example_rank,
    )
    selected = sorted(required, key=_candidate_example_rank)
    selected.extend(
        diverse[: _EVALUATION_EXAMPLE_LIMIT - len(selected)]
    )
    categories_by_key: dict[CandidateKey, list[str]] = defaultdict(list)
    for category, candidate_key in chosen.items():
        if candidate_key in selected:
            categories_by_key[candidate_key].append(category)
    examples: list[dict[str, object]] = []
    for candidate_key in selected:
        membership = evaluation.memberships[candidate_key]
        result = evaluation.results[membership.evaluation_key]
        sample = samples[membership.sample_id]
        examples.append(
            {
                "sample_id": membership.sample_id,
                "candidate_id": membership.candidate_id,
                "candidate_index": membership.candidate_index,
                "evaluation_key": membership.evaluation_key,
                "categories": sorted(categories_by_key[candidate_key]),
                "test_outcome": result.category,
                "official_outcome": result.official_outcome,
                "record_status": result.record_status,
                "task_id": result.task_id,
                "context": {
                    key: sample.dimensions[key]
                    for key in (
                        "source_kind",
                        "source_database_table",
                        "model",
                        "encoder_model",
                        "decoder_model",
                        "prompt_fidelity",
                        "retry_partial",
                        "date_day",
                    )
                },
                "origins": [
                    {"variant": variant, "strategy": strategy}
                    for variant, strategy in evaluation.candidate_origins[
                        candidate_key
                    ]
                ],
                "cleaned_source": _truncate(
                    result.cleaned_source, _CANDIDATE_LIMIT
                ),
                "diagnostics": {
                    "failure_type": result.failure_type,
                    "failure_message": _truncate(
                        result.failure_message, _FACTS_LIMIT
                    ),
                    "function_count": result.function_count,
                    "best_function_name": result.best_function_name,
                    "total_cases": result.total_cases,
                    "passed_count": result.passed_count,
                    "failed_count": result.failed_count,
                    "error_count": result.error_count,
                    "timeout_count": result.timeout_count,
                    "coverage_complete": result.coverage_complete,
                },
            }
        )
    return examples


def _candidate_example_rank(candidate_key: CandidateKey) -> str:
    value = "\0".join(
        (candidate_key[0], str(candidate_key[1]), candidate_key[2])
    )
    return hashlib.sha256(value.encode()).hexdigest()


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
    failure_examples: Mapping[tuple[str, str], _FailureExampleGroup],
    evaluation_examples: list[dict[str, object]],
) -> PreprocessingAnalysisArtifacts:
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
        failure_browser = _write_failure_examples(
            temporary / "failure-examples", failure_examples
        )
        _write_json(
            temporary / "viewer-data.json",
            _viewer_payload(
                summary,
                tables,
                examples,
                failure_browser,
                evaluation_examples,
            ),
            compact=True,
        )
        table_dir = temporary / "tables"
        table_dir.mkdir()
        for name in _COMPACT_TABLES:
            if name not in tables:
                continue
            rows = tables[name]
            assert isinstance(rows, list)
            _write_table(table_dir / f"{name}.parquet", rows)
        (temporary / "report.md").write_text(
            _report(
                summary,
                examples,
                failure_browser,
                evaluation_examples,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return PreprocessingAnalysisArtifacts(
        output_dir=destination,
        summary_path=destination / "summary.json",
        viewer_data_path=destination / "viewer-data.json",
        failure_examples_path=destination / "failure-examples",
        report_path=destination / "report.md",
        table_paths={
            name: destination / "tables" / f"{name}.parquet"
            for name in _COMPACT_TABLES
            if name in tables
        },
    )


def _write_failure_examples(
    destination: Path,
    failure_examples: Mapping[tuple[str, str], _FailureExampleGroup],
) -> dict[str, object]:
    destination.mkdir()
    artifact_id = _failure_examples_artifact_id(failure_examples)
    artifact_root = destination / artifact_id
    artifact_root.mkdir()
    groups: list[dict[str, object]] = []
    total_count = 0
    seen_group_paths: set[str] = set()
    for group_identity in sorted(failure_examples):
        group = failure_examples[group_identity]
        failure_code = group.failure_code
        failed_step = group.failed_step
        examples = group.examples
        group_path = "group-" + hashlib.sha256(
            "\0".join(group_identity).encode()
        ).hexdigest()[:20]
        if group_path in seen_group_paths:
            raise PreprocessingAnalysisError(
                "failure browser group artifact path collision"
            )
        seen_group_paths.add(group_path)
        group_dir = artifact_root / group_path
        group_dir.mkdir()
        entries: list[dict[str, object]] = []
        for shard_index, start in enumerate(
            range(0, len(examples), _FAILURE_EXAMPLE_SHARD_LIMIT)
        ):
            shard_examples = examples[
                start : start + _FAILURE_EXAMPLE_SHARD_LIMIT
            ]
            shard_name = f"examples-{shard_index:04d}.json"
            shard_path = f"{artifact_id}/{group_path}/{shard_name}"
            _write_json(
                group_dir / shard_name,
                {
                    "schema_version": 1,
                    "failure_code": failure_code,
                    "examples": shard_examples,
                },
                compact=True,
            )
            for example in shard_examples:
                raw = example["raw_decoder_output"]
                assert isinstance(raw, str)
                rejections = cast(
                    list[Mapping[str, object]], example["rejections"]
                )
                entries.append(
                    {
                        "sample_id": example["sample_id"],
                        "outcome": example["outcome"],
                        "failed_step": failed_step,
                        "context": example["context"],
                        "rejection_reasons": sorted(
                            {
                                reason
                                for rejection in rejections
                                if isinstance(
                                    reason := rejection.get("reason_code"),
                                    str,
                                )
                            }
                        ),
                        "raw_character_count": group.raw_character_counts[
                            cast(str, example["sample_id"])
                        ],
                        "detail_shard": shard_path,
                    }
                )
        count = len(entries)
        index_path = f"{artifact_id}/{group_path}/index.json"
        _write_json(
            destination / index_path,
            {
                "schema_version": 1,
                "failure_code": failure_code,
                "failed_step": failed_step,
                "count": count,
                "entries": entries,
            },
            compact=True,
        )
        groups.append(
            {
                "failure_code": failure_code,
                "failed_step": failed_step,
                "count": count,
                "index_path": index_path,
            }
        )
        total_count += count
    browser = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "total_count": total_count,
        "groups": groups,
    }
    _write_json(destination / "manifest.json", browser, compact=True)
    return browser


def _failure_examples_artifact_id(
    failure_examples: Mapping[tuple[str, str], _FailureExampleGroup],
) -> str:
    content = {
        "schema_version": 1,
        "shard_limit": _FAILURE_EXAMPLE_SHARD_LIMIT,
        "groups": [
            {
                "failure_code": failure_examples[identity].failure_code,
                "failed_step": failure_examples[identity].failed_step,
                "examples": failure_examples[identity].examples,
                "raw_character_counts": failure_examples[
                    identity
                ].raw_character_counts,
            }
            for identity in sorted(failure_examples)
        ],
    }
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:20]


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
    failure_browser: Mapping[str, object],
    evaluation_examples: list[dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
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
    if failure_browser["total_count"]:
        payload["failure_browser"] = failure_browser
    evaluation = cast(Mapping[str, object], summary["candidate_evaluation"])
    if evaluation.get("available") is True:
        payload["candidate_evaluation"] = {
            "summary": evaluation,
            "test_success_by_origin": tables["test_success_by_origin"],
            "test_success_by_multiplicity": tables[
                "test_success_by_multiplicity"
            ],
            "test_success_by_preprocessing_outcome": tables[
                "test_success_by_preprocessing_outcome"
            ],
            "test_success_by_dimension": tables[
                "test_success_by_dimension"
            ],
            "examples": evaluation_examples,
        }
    return payload


def _report(
    summary: Mapping[str, object],
    examples: list[dict[str, object]],
    failure_browser: Mapping[str, object],
    evaluation_examples: list[dict[str, object]],
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
    evaluation_summary = cast(
        Mapping[str, object], summary["candidate_evaluation"]
    )
    if evaluation_summary.get("available") is True:
        evaluation_funnel = cast(
            list[dict[str, object]], evaluation_summary["funnel"]
        )
        lines.extend(
            [
                "",
                "## Candidate evaluation funnel",
                "",
                "| stage | count | extracted-candidate rate |",
                "| --- | ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {row['stage']} | {row['count']} | "
            f"{row['rate'] if row['rate'] is not None else 'n/a'} |"
            for row in evaluation_funnel
        )
        evaluation_provenance = cast(
            Mapping[str, object], evaluation_summary["provenance"]
        )
        semantic_coordinates = cast(
            Mapping[str, object],
            evaluation_provenance["semantic_coordinates"],
        )
        lines.extend(["", "### Evaluation provenance", ""])
        evaluation_manifest = cast(
            Mapping[str, object], evaluation_provenance["manifest"]
        )
        lines.append(
            "- `candidate_evaluation_manifest_sha256`: "
            f"`{evaluation_manifest['sha256']}`"
        )
        for field in (
            "metrics_profile",
            "operator",
            "snapshot_sha256",
            "runner_identity",
            "sandbox_image",
            "execution_fingerprint",
            "metrics_definition_hash",
            "trusted_source_sha256",
            "operator_settings",
            "python",
            "python_implementation",
        ):
            if field in semantic_coordinates:
                lines.append(
                    f"- `{field}`: "
                    f"`{_report_coordinate(semantic_coordinates[field])}`"
                )
        evaluation_limitations = cast(
            list[object], evaluation_summary["limitations"]
        )
        lines.extend(
            f"- Limitation: {limitation}"
            for limitation in evaluation_limitations
        )
    success = next(
        (row for row in outcomes if row["outcome"] == _SUCCESS), None
    )
    failure_modes = cast(
        list[dict[str, object]],
        cast(Mapping[str, object], summary["tables"])["failure_modes"],
    )
    leading_failure = max(
        failure_modes, key=lambda row: int(row["count"]), default=None
    )
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
        origins,
        key=lambda row: int(row["final_candidate_origin_count"]),
        default=None,
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
    conclusions: list[str] = []
    if success is not None:
        conclusions.append(
            f"- {success['rate_of_present']:.2%} of present decoder outputs produced at least one final top-level-function candidate."
        )
    if leading_failure is not None:
        conclusions.append(
            f"- The most frequent candidate rejection was `{leading_failure['reason']}` at `{leading_failure['failed_step']}` ({leading_failure['count']} rejection rows across {leading_failure['sample_count']} samples)."
        )
    conclusions.append(
        f"- {multiple_samples} samples ({multiple_rate:.2%} of all samples) retained multiple final candidates; candidate rows and sample outcomes are therefore reported separately."
    )
    if leading_origin is not None:
        conclusions.append(
            f"- `{leading_origin['variant']}` / `{leading_origin['strategy']}` supplied {leading_origin['final_candidate_origin_count']} of {origin_attributions} final-origin attributions; its recovery rate is {leading_origin['recovery_rate']:.2%} from extracted candidates."
        )
    conclusions.append(
        f"- `{leading_source_kind['source_kind']}` is the largest source kind with {leading_source_kind['sample_count']} samples ({leading_source_kind['sample_rate_of_all']:.2%} of all samples)."
    )
    if evaluation_summary.get("available") is True:
        sample_best = cast(
            list[dict[str, object]],
            evaluation_summary["sample_best_outcomes"],
        )
        passed = next(
            row for row in sample_best if row["best_test_outcome"] == "passed"
        )
        passed_rate = passed["rate_of_extracted_samples"]
        if passed_rate is None:
            conclusions.append(
                "- No extracted samples were available for candidate testing."
            )
        else:
            conclusions.append(
                f"- {passed['sample_count']} extracted samples "
                f"({passed_rate:.2%}) had at least one passing candidate."
            )
    lines.extend(["", "## Conclusions", "", *conclusions])
    lines.extend(
        [
            "",
            "## Viewer data",
            "",
            f"`viewer-data.json` contains {len(examples)} preprocessing examples and {len(evaluation_examples)} candidate-test examples. Selection is deterministic and raw text is intentionally bounded; the authoritative Parquets retain complete sources.",
            f"`failure-examples/` contains {_mapping_int(failure_browser, 'total_count')} nonblank, zero-final-candidate examples across {len(cast(list[object], failure_browser['groups']))} terminal failure groups. Its indexes and bounded details are loaded lazily by the viewer.",
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


def _report_coordinate(value: object) -> str:
    if value is None or isinstance(value, Mapping | list):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _row_int(row: Mapping[str, object], key: str) -> int:
    return _mapping_int(row, key)


def _mapping_int(values: Mapping[str, object], key: str) -> int:
    value = values[key]
    assert isinstance(value, int)
    return value


def _optional_nonnegative_int(
    value: object, field: str, path: Path
) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PreprocessingAnalysisError(
            f"{path} has invalid non-negative integer {field!r}"
        )
    return value


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

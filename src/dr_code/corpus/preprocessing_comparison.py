"""Deterministic identity-level comparison of immutable preprocessing runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.preprocessing_artifacts import normalize_persisted_origins
from dr_code.viewer.domain import RunDescriptor, RunValidationError


COMPARISON_SCHEMA_VERSION: Final = 1
SUMMARY_FILENAME: Final = "comparison_summary.json"
MANIFEST_FILENAME: Final = "comparison_manifest.json"
_SEMANTIC_RESULT_FIELDS: Final = (
    "raw_output_sha256",
    "decoder_output_presence",
    "outcome",
    "outcome_code",
    "failure_code",
    "failed_step",
    "cause",
    "propagated_through",
    "final_candidate_count",
)

SAMPLE_TRANSITIONS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("before_output_sha256", pa.string()),
        pa.field("after_output_sha256", pa.string()),
        pa.field("before_output_presence", pa.string(), nullable=False),
        pa.field("after_output_presence", pa.string(), nullable=False),
        pa.field("before_outcome", pa.string(), nullable=False),
        pa.field("after_outcome", pa.string(), nullable=False),
        pa.field("before_outcome_code", pa.string()),
        pa.field("after_outcome_code", pa.string()),
        pa.field("before_failure_code", pa.string()),
        pa.field("after_failure_code", pa.string()),
        pa.field("before_failed_step", pa.string()),
        pa.field("after_failed_step", pa.string()),
        pa.field("before_cause", pa.string()),
        pa.field("after_cause", pa.string()),
        pa.field("before_propagated_through", pa.list_(pa.string())),
        pa.field("after_propagated_through", pa.list_(pa.string())),
        pa.field("before_final_candidate_count", pa.int64(), nullable=False),
        pa.field("after_final_candidate_count", pa.int64(), nullable=False),
        pa.field("output_identity_changed", pa.bool_(), nullable=False),
        pa.field("outcome_changed", pa.bool_(), nullable=False),
        pa.field("semantic_result_changed", pa.bool_(), nullable=False),
        pa.field("changed_fields", pa.list_(pa.string()), nullable=False),
        pa.field("change", pa.string(), nullable=False),
    ]
)

CANDIDATE_CHANGES_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("before_present", pa.bool_(), nullable=False),
        pa.field("after_present", pa.bool_(), nullable=False),
        pa.field("before_candidate_index", pa.int64()),
        pa.field("after_candidate_index", pa.int64()),
        pa.field("before_source_sha256", pa.string()),
        pa.field("after_source_sha256", pa.string()),
        pa.field("membership_changed", pa.bool_(), nullable=False),
        pa.field("source_changed", pa.bool_(), nullable=False),
        pa.field("change", pa.string(), nullable=False),
    ]
)

PROVENANCE_PATH_DELTAS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("path_json", pa.string(), nullable=False),
        pa.field("before_count", pa.int64(), nullable=False),
        pa.field("after_count", pa.int64(), nullable=False),
        pa.field("count_delta", pa.int64(), nullable=False),
        pa.field("change", pa.string(), nullable=False),
    ]
)

EVALUATION_MEMBERSHIP_CHANGES_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("before_present", pa.bool_(), nullable=False),
        pa.field("after_present", pa.bool_(), nullable=False),
        pa.field("before_evaluation_key", pa.string()),
        pa.field("after_evaluation_key", pa.string()),
        pa.field("before_membership_json", pa.string()),
        pa.field("after_membership_json", pa.string()),
        pa.field("change", pa.string(), nullable=False),
    ]
)

EVALUATION_RESULT_CHANGES_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("before_evaluation_key", pa.string()),
        pa.field("after_evaluation_key", pa.string()),
        pa.field("before_result_json", pa.string()),
        pa.field("after_result_json", pa.string()),
        pa.field("change", pa.string(), nullable=False),
    ]
)

_RELATION_SCHEMAS: Final[Mapping[str, pa.Schema]] = {
    "sample_outcome_transitions": SAMPLE_TRANSITIONS_SCHEMA,
    "candidate_changes": CANDIDATE_CHANGES_SCHEMA,
    "provenance_path_deltas": PROVENANCE_PATH_DELTAS_SCHEMA,
    "evaluation_membership_changes": EVALUATION_MEMBERSHIP_CHANGES_SCHEMA,
    "evaluation_result_changes": EVALUATION_RESULT_CHANGES_SCHEMA,
}


class PreprocessingComparisonError(ValueError):
    """The compared bundles cannot produce a trustworthy identity audit."""


@dataclass(frozen=True, slots=True)
class PreprocessingComparisonArtifacts:
    """The append-only files emitted for one before/after comparison."""

    output_dir: Path
    summary_path: Path
    manifest_path: Path
    relation_paths: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class _RunRows:
    descriptor: RunDescriptor
    results: Mapping[str, Mapping[str, object]]
    candidates: Mapping[tuple[str, str], Mapping[str, object]]
    provenance: Mapping[tuple[str, str, str], int]
    memberships: Mapping[tuple[str, str], Mapping[str, object]]
    evaluation_results: Mapping[str, Mapping[str, object]]


def compare_preprocessing_runs(
    *,
    corpus_path: Path | str,
    before_run: Path | str,
    after_run: Path | str,
    output_dir: Path | str,
    before_evaluation: Path | str | None = None,
    after_evaluation: Path | str | None = None,
) -> PreprocessingComparisonArtifacts:
    """Write a deterministic audit of two immutable runs over one corpus."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"comparison output already exists: {destination}"
        )
    if (before_evaluation is None) != (after_evaluation is None):
        raise PreprocessingComparisonError(
            "before and after evaluation bundles must be supplied together"
        )
    corpus = Path(corpus_path).expanduser().resolve(strict=True)
    before_descriptor = _load_descriptor(
        label="before",
        corpus=corpus,
        preprocessing=before_run,
        evaluation=before_evaluation,
    )
    after_descriptor = _load_descriptor(
        label="after",
        corpus=corpus,
        preprocessing=after_run,
        evaluation=after_evaluation,
    )
    if before_descriptor.corpus_sha256 != after_descriptor.corpus_sha256:
        raise PreprocessingComparisonError(
            "before and after preprocessing runs reference different corpora"
        )

    sample_ids = _read_corpus_sample_ids(corpus)
    before = _read_run(before_descriptor, sample_ids)
    after = _read_run(after_descriptor, sample_ids)
    relations = _comparison_relations(before, after, sample_ids)
    summary = _comparison_summary(before, after, relations, len(sample_ids))

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    relation_paths: dict[str, Path] = {}
    relation_manifest: dict[str, object] = {}
    for name, schema in _RELATION_SCHEMAS.items():
        path = destination / f"{name}.parquet"
        rows = relations[name]
        _write_relation(path, rows, schema)
        relation_paths[name] = path
        relation_manifest[name] = {
            "filename": path.name,
            "row_count": len(rows),
            "sha256": _sha256_file(path),
            "schema": schema.serialize().to_pybytes().hex(),
        }

    summary_path = destination / SUMMARY_FILENAME
    _write_json(summary_path, summary)
    manifest_path = destination / MANIFEST_FILENAME
    _write_json(
        manifest_path,
        {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "complete": True,
            "corpus_sha256": before.descriptor.corpus_sha256,
            "corpus_rows": len(sample_ids),
            "before": _run_manifest_value(before.descriptor),
            "after": _run_manifest_value(after.descriptor),
            "relations": relation_manifest,
            "summary": {
                "filename": summary_path.name,
                "sha256": _sha256_file(summary_path),
            },
        },
    )
    return PreprocessingComparisonArtifacts(
        output_dir=destination,
        summary_path=summary_path,
        manifest_path=manifest_path,
        relation_paths=relation_paths,
    )


def _load_descriptor(
    *,
    label: str,
    corpus: Path,
    preprocessing: Path | str,
    evaluation: Path | str | None,
) -> RunDescriptor:
    try:
        return RunDescriptor.from_paths(
            label=label,
            corpus_path=corpus,
            preprocessing=preprocessing,
            candidate_evaluation=evaluation,
        )
    except (OSError, RunValidationError) as exc:
        raise PreprocessingComparisonError(
            f"{label} immutable bundle is invalid: {exc}"
        ) from exc


def _read_corpus_sample_ids(path: Path) -> tuple[str, ...]:
    parquet = pq.ParquetFile(path)
    if "sample_id" not in parquet.schema_arrow.names:
        raise PreprocessingComparisonError(
            "comparison corpus is missing sample_id"
        )
    values: list[str] = []
    seen: set[str] = set()
    for batch in parquet.iter_batches(columns=["sample_id"]):
        for raw in batch.column(0).to_pylist():
            if not isinstance(raw, str) or not raw:
                raise PreprocessingComparisonError(
                    "comparison corpus contains an invalid sample_id"
                )
            if raw in seen:
                raise PreprocessingComparisonError(
                    f"comparison corpus contains duplicate sample_id: {raw}"
                )
            seen.add(raw)
            values.append(raw)
    return tuple(sorted(values))


def _read_run(
    descriptor: RunDescriptor, sample_ids: tuple[str, ...]
) -> _RunRows:
    expected_samples = set(sample_ids)
    results = _read_unique_rows(
        descriptor.results_path, ("sample_id",), "preprocessing result"
    )
    result_samples = {key[0] for key in results}
    if result_samples != expected_samples:
        raise PreprocessingComparisonError(
            f"{descriptor.label} results sample identities do not match corpus"
        )

    candidates, provenance = _read_candidates(
        descriptor, expected_samples, results, sample_ids
    )

    memberships: Mapping[tuple[str, str], Mapping[str, object]] = {}
    evaluation_results: Mapping[str, Mapping[str, object]] = {}
    if descriptor.has_evaluation:
        memberships, evaluation_results = _read_evaluation(
            descriptor, candidates
        )
    return _RunRows(
        descriptor=descriptor,
        results={key[0]: row for key, row in results.items()},
        candidates=candidates,
        provenance=provenance,
        memberships=memberships,
        evaluation_results=evaluation_results,
    )


def _read_candidates(
    descriptor: RunDescriptor,
    expected_samples: set[str],
    results: Mapping[tuple[str, ...], Mapping[str, object]],
    sample_ids: tuple[str, ...],
) -> tuple[
    Mapping[tuple[str, str], Mapping[str, object]],
    Mapping[tuple[str, str, str], int],
]:
    candidates: dict[tuple[str, str], Mapping[str, object]] = {}
    provenance: Counter[tuple[str, str, str]] = Counter()
    indices: dict[str, list[int]] = {sample_id: [] for sample_id in sample_ids}
    columns = [
        "sample_id",
        "candidate_id",
        "candidate_index",
        "cleaned_source",
        "source_sha256",
        "origins",
    ]
    for batch in pq.ParquetFile(descriptor.candidates_path).iter_batches(
        batch_size=32_768, columns=columns
    ):
        for row in batch.to_pylist():
            sample_id = row.get("sample_id")
            candidate_id = row.get("candidate_id")
            if (
                not isinstance(sample_id, str)
                or sample_id not in expected_samples
                or not isinstance(candidate_id, str)
                or not candidate_id
            ):
                raise PreprocessingComparisonError(
                    f"{descriptor.label} candidate has invalid identity"
                )
            key = (sample_id, candidate_id)
            if key in candidates:
                raise PreprocessingComparisonError(
                    f"duplicate preprocessing candidate identity: {key}"
                )
            index = row.get("candidate_index")
            source = row.get("cleaned_source")
            source_sha256 = row.get("source_sha256")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
            ):
                raise PreprocessingComparisonError(
                    f"{descriptor.label} candidate has invalid index: "
                    f"{sample_id}/{candidate_id}"
                )
            if (
                not isinstance(source, str)
                or not isinstance(source_sha256, str)
                or hashlib.sha256(source.encode("utf-8")).hexdigest()
                != source_sha256
            ):
                raise PreprocessingComparisonError(
                    f"{descriptor.label} candidate source identity mismatch: "
                    f"{sample_id}/{candidate_id}"
                )
            candidates[key] = {
                "candidate_index": index,
                "source_sha256": source_sha256,
            }
            indices[sample_id].append(index)
            origins = normalize_persisted_origins(
                row.get("origins"), descriptor.preprocessing_schema_version
            )
            for origin in origins:
                provenance[
                    (sample_id, candidate_id, _canonical_json(origin))
                ] += 1
    for sample_id in sample_ids:
        expected_count = results[(sample_id,)].get("final_candidate_count")
        actual_indices = sorted(indices[sample_id])
        if expected_count != len(actual_indices):
            raise PreprocessingComparisonError(
                f"{descriptor.label} final candidate count mismatch: {sample_id}"
            )
        if actual_indices != list(range(len(actual_indices))):
            raise PreprocessingComparisonError(
                f"{descriptor.label} candidate indices are not contiguous: "
                f"{sample_id}"
            )
    return candidates, provenance


def _read_unique_rows(
    path: Path,
    identity_fields: tuple[str, ...],
    label: str,
    *,
    exclude_fields: frozenset[str] = frozenset(),
) -> dict[tuple[str, ...], Mapping[str, object]]:
    rows: dict[tuple[str, ...], Mapping[str, object]] = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65_536):
        for row in batch.to_pylist():
            identity: list[str] = []
            for field in identity_fields:
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    raise PreprocessingComparisonError(
                        f"{label} contains invalid {field}"
                    )
                identity.append(value)
            key = tuple(identity)
            if key in rows:
                raise PreprocessingComparisonError(
                    f"duplicate {label} identity: {key}"
                )
            rows[key] = {
                field: value
                for field, value in row.items()
                if field not in exclude_fields
            }
    return rows


def _read_evaluation(
    descriptor: RunDescriptor,
    candidates: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[
    Mapping[tuple[str, str], Mapping[str, object]],
    Mapping[str, Mapping[str, object]],
]:
    assert descriptor.candidate_membership_path is not None
    assert descriptor.candidate_results_path is not None
    memberships = _read_unique_rows(
        descriptor.candidate_membership_path,
        ("sample_id", "candidate_id"),
        "evaluation membership",
    )
    results_by_key = _read_unique_rows(
        descriptor.candidate_results_path,
        ("evaluation_key",),
        "evaluation result",
        exclude_fields=frozenset({"cleaned_source"}),
    )
    if set(memberships) != set(candidates):
        raise PreprocessingComparisonError(
            f"{descriptor.label} evaluation membership identities do not "
            "match preprocessing candidates"
        )
    assert descriptor.evaluation_coordinates is not None
    expected_profile = descriptor.evaluation_coordinates["metrics_profile"]
    expected_operator = descriptor.evaluation_coordinates["operator"]
    referenced: set[str] = set()
    for raw_candidate_key, membership in memberships.items():
        candidate_key = cast(tuple[str, str], raw_candidate_key)
        evaluation_key = cast(str, membership["evaluation_key"])
        result = results_by_key.get((evaluation_key,))
        if result is None:
            raise PreprocessingComparisonError(
                f"{descriptor.label} evaluation membership references "
                f"missing result: {evaluation_key}"
            )
        referenced.add(evaluation_key)
        candidate = candidates[candidate_key]
        if membership.get("candidate_index") != candidate.get(
            "candidate_index"
        ):
            raise PreprocessingComparisonError(
                f"{descriptor.label} evaluation membership index does not "
                f"match candidate: {candidate_key}"
            )
        if membership.get("source_sha256") != candidate.get("source_sha256"):
            raise PreprocessingComparisonError(
                f"{descriptor.label} evaluation membership source does not "
                f"match candidate: {candidate_key}"
            )
        if result.get("source_sha256") != membership.get("source_sha256"):
            raise PreprocessingComparisonError(
                f"{descriptor.label} evaluation result source does not match "
                f"membership: {candidate_key}"
            )
        for field, expected in (
            ("metrics_profile", expected_profile),
            ("operator", expected_operator),
        ):
            if (
                membership.get(field) != expected
                or result.get(field) != expected
            ):
                raise PreprocessingComparisonError(
                    f"{descriptor.label} evaluation {field} does not match "
                    "its manifest"
                )
        for field in ("task_id", "task_fingerprint"):
            if result.get(field) != membership.get(field):
                raise PreprocessingComparisonError(
                    f"{descriptor.label} evaluation result {field} does not "
                    f"match membership: {candidate_key}"
                )
    if referenced != {key[0] for key in results_by_key}:
        raise PreprocessingComparisonError(
            f"{descriptor.label} evaluation contains unreferenced results"
        )
    return (
        {cast(tuple[str, str], key): row for key, row in memberships.items()},
        {key[0]: row for key, row in results_by_key.items()},
    )


def _comparison_relations(
    before: _RunRows,
    after: _RunRows,
    sample_ids: tuple[str, ...],
) -> dict[str, list[dict[str, object]]]:
    return {
        "sample_outcome_transitions": _sample_transition_rows(
            before.results, after.results, sample_ids
        ),
        "candidate_changes": _candidate_change_rows(
            before.candidates, after.candidates
        ),
        "provenance_path_deltas": _provenance_delta_rows(
            before.provenance, after.provenance
        ),
        "evaluation_membership_changes": _evaluation_membership_rows(
            before.memberships, after.memberships
        ),
        "evaluation_result_changes": _evaluation_result_rows(before, after),
    }


def _sample_transition_rows(
    before: Mapping[str, Mapping[str, object]],
    after: Mapping[str, Mapping[str, object]],
    sample_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        old = before[sample_id]
        new = after[sample_id]
        output_changed = (
            old["raw_output_sha256"] != new["raw_output_sha256"]
            or old["decoder_output_presence"] != new["decoder_output_presence"]
        )
        outcome_changed = old["outcome"] != new["outcome"]
        changed_fields = [
            field
            for field in _SEMANTIC_RESULT_FIELDS
            if old[field] != new[field]
        ]
        semantic_result_changed = bool(changed_fields)
        rows.append(
            {
                "sample_id": sample_id,
                "before_output_sha256": old["raw_output_sha256"],
                "after_output_sha256": new["raw_output_sha256"],
                "before_output_presence": old["decoder_output_presence"],
                "after_output_presence": new["decoder_output_presence"],
                "before_outcome": old["outcome"],
                "after_outcome": new["outcome"],
                "before_outcome_code": old["outcome_code"],
                "after_outcome_code": new["outcome_code"],
                "before_failure_code": old["failure_code"],
                "after_failure_code": new["failure_code"],
                "before_failed_step": old["failed_step"],
                "after_failed_step": new["failed_step"],
                "before_cause": old["cause"],
                "after_cause": new["cause"],
                "before_propagated_through": old["propagated_through"],
                "after_propagated_through": new["propagated_through"],
                "before_final_candidate_count": old["final_candidate_count"],
                "after_final_candidate_count": new["final_candidate_count"],
                "output_identity_changed": output_changed,
                "outcome_changed": outcome_changed,
                "semantic_result_changed": semantic_result_changed,
                "changed_fields": changed_fields,
                "change": (
                    "semantic_result_changed"
                    if semantic_result_changed
                    else "unchanged"
                ),
            }
        )
    return rows


def _candidate_change_rows(
    before: Mapping[tuple[str, str], Mapping[str, object]],
    after: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id, candidate_id in sorted(set(before) | set(after)):
        old = before.get((sample_id, candidate_id))
        new = after.get((sample_id, candidate_id))
        before_present = old is not None
        after_present = new is not None
        membership_changed = before_present != after_present or (
            old is not None
            and new is not None
            and old["candidate_index"] != new["candidate_index"]
        )
        source_changed = (
            old is not None
            and new is not None
            and old["source_sha256"] != new["source_sha256"]
        )
        if old is None:
            change = "added"
        elif new is None:
            change = "removed"
        elif membership_changed or source_changed:
            change = "modified"
        else:
            change = "unchanged"
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "before_present": before_present,
                "after_present": after_present,
                "before_candidate_index": (
                    old["candidate_index"] if old is not None else None
                ),
                "after_candidate_index": (
                    new["candidate_index"] if new is not None else None
                ),
                "before_source_sha256": (
                    old["source_sha256"] if old is not None else None
                ),
                "after_source_sha256": (
                    new["source_sha256"] if new is not None else None
                ),
                "membership_changed": membership_changed,
                "source_changed": source_changed,
                "change": change,
            }
        )
    return rows


def _provenance_delta_rows(
    before: Mapping[tuple[str, str, str], int],
    after: Mapping[tuple[str, str, str], int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id, candidate_id, path_json in sorted(set(before) | set(after)):
        old = before.get((sample_id, candidate_id, path_json), 0)
        new = after.get((sample_id, candidate_id, path_json), 0)
        change = "unchanged"
        if old == 0:
            change = "added"
        elif new == 0:
            change = "removed"
        elif old != new:
            change = "count_changed"
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "path_json": path_json,
                "before_count": old,
                "after_count": new,
                "count_delta": new - old,
                "change": change,
            }
        )
    return rows


def _evaluation_membership_rows(
    before: Mapping[tuple[str, str], Mapping[str, object]],
    after: Mapping[tuple[str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_id, candidate_id in sorted(set(before) | set(after)):
        old = before.get((sample_id, candidate_id))
        new = after.get((sample_id, candidate_id))
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "before_present": old is not None,
                "after_present": new is not None,
                "before_evaluation_key": (
                    old["evaluation_key"] if old is not None else None
                ),
                "after_evaluation_key": (
                    new["evaluation_key"] if new is not None else None
                ),
                "before_membership_json": (
                    _canonical_json(old) if old is not None else None
                ),
                "after_membership_json": (
                    _canonical_json(new) if new is not None else None
                ),
                "change": _optional_mapping_change(old, new),
            }
        )
    return rows


def _evaluation_result_rows(
    before: _RunRows, after: _RunRows
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = sorted(set(before.memberships) | set(after.memberships))
    for sample_id, candidate_id in keys:
        old_membership = before.memberships.get((sample_id, candidate_id))
        new_membership = after.memberships.get((sample_id, candidate_id))
        old_key = (
            cast(str, old_membership["evaluation_key"])
            if old_membership is not None
            else None
        )
        new_key = (
            cast(str, new_membership["evaluation_key"])
            if new_membership is not None
            else None
        )
        old = before.evaluation_results.get(old_key) if old_key else None
        new = after.evaluation_results.get(new_key) if new_key else None
        old_value = _compact_evaluation_result(old)
        new_value = _compact_evaluation_result(new)
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "before_evaluation_key": old_key,
                "after_evaluation_key": new_key,
                "before_result_json": (
                    _canonical_json(old_value)
                    if old_value is not None
                    else None
                ),
                "after_result_json": (
                    _canonical_json(new_value)
                    if new_value is not None
                    else None
                ),
                "change": _optional_mapping_change(old_value, new_value),
            }
        )
    return rows


def _compact_evaluation_result(
    value: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    return {
        key: item for key, item in value.items() if key != "cleaned_source"
    }


def _optional_mapping_change(
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
) -> str:
    if before is None:
        return "added"
    if after is None:
        return "removed"
    return "unchanged" if before == after else "modified"


def _comparison_summary(
    before: _RunRows,
    after: _RunRows,
    relations: Mapping[str, list[dict[str, object]]],
    corpus_rows: int,
) -> dict[str, object]:
    samples = relations["sample_outcome_transitions"]
    candidates = relations["candidate_changes"]
    provenance = relations["provenance_path_deltas"]
    memberships = relations["evaluation_membership_changes"]
    results = relations["evaluation_result_changes"]
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "corpus_rows": corpus_rows,
        "sample_outcome_transitions": {
            **_change_counts(samples),
            "output_identity_changed_count": sum(
                row["output_identity_changed"] is True for row in samples
            ),
            "outcome_changed_count": sum(
                row["outcome_changed"] is True for row in samples
            ),
            "semantic_result_changed_count": sum(
                row["semantic_result_changed"] is True for row in samples
            ),
            "transitions": _transition_counts(samples),
        },
        "candidate_changes": _presence_counts(candidates),
        "provenance_path_deltas": {
            **_change_counts(provenance),
            "before_count": sum(
                cast(int, row["before_count"]) for row in provenance
            ),
            "after_count": sum(
                cast(int, row["after_count"]) for row in provenance
            ),
            "net_count_delta": sum(
                cast(int, row["count_delta"]) for row in provenance
            ),
        },
        "evaluation": {
            "included": before.descriptor.has_evaluation,
            "membership_changes": _presence_counts(memberships),
            "result_changes": _change_counts(results),
            "coordinates": _evaluation_coordinate_summary(before, after),
        },
        "reconciliation": {
            "sample_identity_rows": len(samples),
            "sample_rows_match_corpus": len(samples) == corpus_rows,
            "candidate_before_count": sum(
                row["before_present"] is True for row in candidates
            ),
            "candidate_after_count": sum(
                row["after_present"] is True for row in candidates
            ),
            "provenance_before_count": sum(
                cast(int, row["before_count"]) for row in provenance
            ),
            "provenance_after_count": sum(
                cast(int, row["after_count"]) for row in provenance
            ),
            "evaluation_membership_before_count": sum(
                row["before_present"] is True for row in memberships
            ),
            "evaluation_membership_after_count": sum(
                row["after_present"] is True for row in memberships
            ),
            "evaluation_result_identity_rows": len(results),
        },
    }


def _change_counts(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    values = list(rows)
    counts = Counter(cast(str, row["change"]) for row in values)
    return {
        "identity_rows": len(values),
        "changed_identity_rows": len(values) - counts.get("unchanged", 0),
        "by_change": dict(sorted(counts.items())),
    }


def _presence_counts(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    values = list(rows)
    return {
        **_change_counts(values),
        "before_count": sum(row["before_present"] is True for row in values),
        "after_count": sum(row["after_present"] is True for row in values),
        "count_delta": sum(row["after_present"] is True for row in values)
        - sum(row["before_present"] is True for row in values),
    }


def _transition_counts(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    counts = Counter(
        (cast(str, row["before_outcome"]), cast(str, row["after_outcome"]))
        for row in rows
    )
    return [
        {"before_outcome": old, "after_outcome": new, "count": count}
        for (old, new), count in sorted(counts.items())
    ]


def _evaluation_coordinate_summary(
    before: _RunRows, after: _RunRows
) -> dict[str, object] | None:
    if not before.descriptor.has_evaluation:
        return None
    old = _semantic_coordinates(before.descriptor)
    new = _semantic_coordinates(after.descriptor)
    fields = sorted(set(old) | set(new))
    return {
        "before": old,
        "after": new,
        "changed_fields": [
            field for field in fields if old.get(field) != new.get(field)
        ],
    }


def _semantic_coordinates(descriptor: RunDescriptor) -> Mapping[str, object]:
    assert descriptor.evaluation_coordinates is not None
    return {
        key: value
        for key, value in sorted(descriptor.evaluation_coordinates.items())
        if key != "manifest_name"
    }


def _run_manifest_value(descriptor: RunDescriptor) -> dict[str, object]:
    return {
        "run_id": descriptor.run_id,
        "preprocessing_schema_version": descriptor.preprocessing_schema_version,
        "preprocessing_manifest_sha256": (
            descriptor.preprocessing_manifest_sha256
        ),
        "artifact_sha256": dict(sorted(descriptor.artifact_sha256.items())),
        "evaluation_manifest_sha256": descriptor.evaluation_manifest_sha256,
    }


def _write_relation(
    path: Path, rows: list[dict[str, object]], schema: pa.Schema
) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "COMPARISON_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "SUMMARY_FILENAME",
    "PreprocessingComparisonArtifacts",
    "PreprocessingComparisonError",
    "compare_preprocessing_runs",
)

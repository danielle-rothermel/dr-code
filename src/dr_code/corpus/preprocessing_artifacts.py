"""Thin, deterministic Parquet projections of preprocessing traces.

This module deliberately does not parse candidate source or infer labels.  It
only serializes the preprocessing runner's output and mechanically joins the
filter facts already recorded in a trace.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.trace import Absent, CodeCandidateSetArtifact, Trace

RESULTS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("decoder_output_presence", pa.string(), nullable=False),
        pa.field("raw_output_sha256", pa.string()),
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("outcome_code", pa.string()),
        pa.field("failure_code", pa.string()),
        pa.field("failed_step", pa.string()),
        pa.field("cause", pa.string()),
        pa.field("propagated_through", pa.list_(pa.string())),
        pa.field("final_candidate_count", pa.int64(), nullable=False),
    ]
)

_ORIGIN_SCHEMA: Final = pa.struct(
    [
        pa.field("variant", pa.string(), nullable=False),
        pa.field("strategy", pa.string(), nullable=False),
    ]
)

CANDIDATES_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_index", pa.int64(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("cleaned_source", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("origins", pa.list_(_ORIGIN_SCHEMA), nullable=False),
        pa.field("parse_ok", pa.bool_()),
        pa.field("parse_error", pa.string()),
        pa.field("compile_ok", pa.bool_()),
        pa.field("compile_error", pa.string()),
        pa.field("compile_warnings", pa.list_(pa.string())),
        pa.field("top_level_function_count", pa.int64()),
        pa.field("top_level_function_names", pa.list_(pa.string())),
        pa.field("top_level_async_function_names", pa.list_(pa.string())),
    ]
)

STEP_FACTS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("step_name", pa.string(), nullable=False),
        pa.field("facts_json", pa.string(), nullable=False),
    ]
)

REJECTIONS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("step_name", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string()),
        pa.field("input_index", pa.int64()),
        pa.field("reason_code", pa.string()),
        pa.field("details_json", pa.string(), nullable=False),
    ]
)

PROJECTED_ARTIFACT_SCHEMAS: Final[Mapping[str, pa.Schema]] = {
    "results": RESULTS_SCHEMA,
    "candidates": CANDIDATES_SCHEMA,
    "step_facts": STEP_FACTS_SCHEMA,
    "rejections": REJECTIONS_SCHEMA,
}

_RELATION_NAMES: Final = tuple(PROJECTED_ARTIFACT_SCHEMAS)
_CANDIDATE_FACT_FIELDS: Final = (
    "parse_ok",
    "parse_error",
    "compile_ok",
    "compile_error",
    "compile_warnings",
    "top_level_function_count",
    "top_level_function_names",
    "top_level_async_function_names",
)
_REJECTION_COLUMNS: Final = frozenset(
    {
        "candidate_id",
        "input_index",
        "index",
        "reason_code",
        "reason",
    }
)
_ROW_GROUP_SIZE: Final = 65_536


@dataclass(slots=True)
class ProjectedArtifacts:
    """Rows for each relation produced from one source sample."""

    results: list[dict[str, object]]
    candidates: list[dict[str, object]]
    step_facts: list[dict[str, object]]
    rejections: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class ProjectedPart:
    """Locations and counts for one explicitly written artifact part."""

    part_id: str
    relation_paths: Mapping[str, Path]
    row_counts: Mapping[str, int]


class AtomicProjectedPartWriter:
    """Append bounded projections into one atomically published shard.

    The writers are opened eagerly so relations with no rows still become
    schema-correct Parquet files.  A shard becomes visible only after all four
    writers close and the temporary directory is renamed into place.
    """

    def __init__(self, output_dir: str | Path, part_id: str) -> None:
        self.part_id = _validated_part_id(part_id)
        self._parts_dir = Path(output_dir) / "parts"
        self._part_dir = self._parts_dir / self.part_id
        self._temporary_part_dir = self._parts_dir / f".{self.part_id}.tmp"
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._row_counts = {
            relation_name: 0 for relation_name in _RELATION_NAMES
        }
        self._part: ProjectedPart | None = None
        self._aborted = False

    def __enter__(self) -> AtomicProjectedPartWriter:
        self._open()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> bool:
        if exception_type is None:
            self.finish()
        else:
            self.abort()
        return False

    @property
    def part(self) -> ProjectedPart:
        """Return the published part after :meth:`finish` succeeds."""
        if self._part is None:
            raise RuntimeError("projected part has not been published")
        return self._part

    def append(self, projected: ProjectedArtifacts) -> None:
        """Write one bounded projection without retaining prior projections."""
        if not self._writers:
            raise RuntimeError("projected part writer is not open")
        for relation_name, rows in _relation_rows(projected):
            if not rows:
                continue
            table = pa.Table.from_pylist(
                rows,
                schema=PROJECTED_ARTIFACT_SCHEMAS[relation_name],
            )
            self._writers[relation_name].write_table(
                table,
                row_group_size=_ROW_GROUP_SIZE,
            )
            self._row_counts[relation_name] += len(rows)

    def finish(self) -> ProjectedPart:
        """Close all relation files and atomically publish the shard."""
        if self._part is not None:
            return self._part
        if self._aborted:
            raise RuntimeError("cannot finish an aborted projected part")
        if not self._writers:
            raise RuntimeError("projected part writer is not open")

        self._close_writers()
        os.replace(self._temporary_part_dir, self._part_dir)
        relation_paths = {
            relation_name: self._part_dir / f"{relation_name}.parquet"
            for relation_name in _RELATION_NAMES
        }
        self._part = ProjectedPart(
            part_id=self.part_id,
            relation_paths=relation_paths,
            row_counts=dict(self._row_counts),
        )
        return self._part

    def abort(self) -> None:
        """Discard an unpublished shard after a failed batch."""
        if self._part is not None or self._aborted:
            return
        self._close_writers()
        if self._temporary_part_dir.exists():
            shutil.rmtree(self._temporary_part_dir)
        self._aborted = True

    def _open(self) -> None:
        if self._part is not None or self._aborted or self._writers:
            raise RuntimeError("projected part writer cannot be reopened")
        if self._part_dir.exists():
            raise FileExistsError(
                f"projected part already exists: {self._part_dir}"
            )
        self._parts_dir.mkdir(parents=True, exist_ok=True)
        if self._temporary_part_dir.exists():
            # Temporary shards are never checkpointed; recover stale debris.
            shutil.rmtree(self._temporary_part_dir)
        self._temporary_part_dir.mkdir()
        try:
            for relation_name, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
                self._writers[relation_name] = pq.ParquetWriter(
                    self._temporary_part_dir / f"{relation_name}.parquet",
                    schema,
                    compression="zstd",
                    use_dictionary=False,
                    write_statistics=True,
                    version="2.6",
                )
        except BaseException:
            self.abort()
            raise

    def _close_writers(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


def project_preprocessing_result(
    sample_id: str,
    decoder_output: str | None,
    trace: Trace | None,
) -> ProjectedArtifacts:
    """Project one source row without interpreting source text.

    ``None`` decoder output is distinct from a present empty or blank string:
    it has no trace-derived relations.  A present string must have its official
    preprocessing trace, whose output is either ``Absent`` or a candidate set.
    """
    if decoder_output is None:
        if trace is not None:
            raise ValueError("missing decoder output must not have a trace")
        return ProjectedArtifacts(
            results=[
                {
                    "sample_id": sample_id,
                    "decoder_output_presence": "missing",
                    "raw_output_sha256": None,
                    "outcome": "decoder_output_missing",
                    "outcome_code": None,
                    "failure_code": None,
                    "failed_step": None,
                    "cause": None,
                    "propagated_through": None,
                    "final_candidate_count": 0,
                }
            ],
            candidates=[],
            step_facts=[],
            rejections=[],
        )

    if trace is None:
        raise ValueError(
            "present decoder output requires a preprocessing trace"
        )

    output = trace.value("output")
    if not isinstance(output, Absent | CodeCandidateSetArtifact):
        raise ValueError(
            "preprocessing trace output must be Absent or "
            "CodeCandidateSetArtifact"
        )

    result = _result_row(sample_id, decoder_output, output, trace)
    step_fact_rows = _step_fact_rows(sample_id, trace)
    rejection_rows = _rejection_rows(sample_id, trace)
    candidate_rows = (
        _candidate_rows(sample_id, output, trace)
        if isinstance(output, CodeCandidateSetArtifact)
        else []
    )
    return ProjectedArtifacts(
        results=[result],
        candidates=candidate_rows,
        step_facts=step_fact_rows,
        rejections=rejection_rows,
    )


def write_projected_part(
    output_dir: str | Path,
    part_id: str,
    projected: ProjectedArtifacts,
) -> ProjectedPart:
    """Convenience wrapper for writing one already-materialized projection."""
    with AtomicProjectedPartWriter(output_dir, part_id) as writer:
        writer.append(projected)
    return writer.part


def combine_projected_parts(
    output_dir: str | Path,
    part_ids: Iterable[str],
) -> Mapping[str, Path]:
    """Stream existing parts into root-level final relations.

    Individual parts remain checkpointable under ``parts/<part_id>`` while
    these final files are directly consumable analysis deliverables.
    """
    normalized_ids = tuple(_validated_part_id(part_id) for part_id in part_ids)
    root = Path(output_dir)
    paths: dict[str, Path] = {}
    for relation_name, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
        destination = root / f"{relation_name}.parquet"
        _combine_relation(
            root,
            normalized_ids,
            relation_name,
            schema,
            destination,
        )
        paths[relation_name] = destination
    return paths


def _result_row(
    sample_id: str,
    decoder_output: str,
    output: Absent | CodeCandidateSetArtifact,
    trace: Trace,
) -> dict[str, object]:
    if isinstance(output, Absent):
        outcome_code: str | None = None
        failure_code: str | None = output.failure_code
        failed_step: str | None = output.failed_step
        cause: str | None = output.cause
        propagated_through: list[str] | None = list(output.propagated_through)
        final_candidate_count = 0
        outcome = output.failure_code
    else:
        outcome_code = _outcome_code(trace)
        if outcome_code is None:
            raise ValueError(
                "successful preprocessing trace has no outcome_code"
            )
        failure_code = None
        failed_step = None
        cause = None
        propagated_through = None
        final_candidate_count = len(output.candidates)
        outcome = outcome_code
    return {
        "sample_id": sample_id,
        "decoder_output_presence": "present",
        "raw_output_sha256": _sha256(decoder_output),
        "outcome": outcome,
        "outcome_code": outcome_code,
        "failure_code": failure_code,
        "failed_step": failed_step,
        "cause": cause,
        "propagated_through": propagated_through,
        "final_candidate_count": final_candidate_count,
    }


def _outcome_code(trace: Trace) -> str | None:
    outcome_code: str | None = None
    for facts in trace.step_facts.values():
        value = facts.get("outcome_code")
        if isinstance(value, str):
            outcome_code = value
    return outcome_code


def _step_fact_rows(sample_id: str, trace: Trace) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample_id,
            "step_name": step_name,
            "facts_json": _canonical_json(facts),
        }
        for step_name, facts in trace.step_facts.items()
    ]


def _rejection_rows(sample_id: str, trace: Trace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step_name, facts in trace.step_facts.items():
        rejections = facts.get("rejections")
        if not isinstance(rejections, list):
            continue
        for rejection in rejections:
            if not isinstance(rejection, Mapping):
                continue
            input_index = rejection.get("input_index", rejection.get("index"))
            reason_code = rejection.get("reason_code", rejection.get("reason"))
            rows.append(
                {
                    "sample_id": sample_id,
                    "step_name": step_name,
                    "candidate_id": _string_or_none(
                        rejection.get("candidate_id")
                    ),
                    "input_index": _int_or_none(input_index),
                    "reason_code": _string_or_none(reason_code),
                    "details_json": _canonical_json(
                        {
                            key: value
                            for key, value in rejection.items()
                            if key not in _REJECTION_COLUMNS
                        }
                    ),
                }
            )
    return rows


def _candidate_rows(
    sample_id: str,
    output: CodeCandidateSetArtifact,
    trace: Trace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, source in enumerate(output.candidates):
        lineage = output.lineage_at(index)
        candidate_id = lineage.candidate_id
        if candidate_id is None:
            raise ValueError(
                "final preprocessing candidate is missing candidate_id"
            )
        diagnostics = _candidate_diagnostics(candidate_id, trace)
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_index": index,
                "candidate_id": candidate_id,
                "cleaned_source": source,
                "source_sha256": _sha256(source),
                "origins": [
                    {
                        "variant": origin.variant,
                        "strategy": origin.strategy,
                    }
                    for origin in lineage.origins
                ],
                **diagnostics,
            }
        )
    return rows


def _candidate_diagnostics(
    candidate_id: str | None,
    trace: Trace,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "parse_ok": None,
        "parse_error": None,
        "compile_ok": None,
        "compile_error": None,
        "compile_warnings": None,
        "top_level_function_count": None,
        "top_level_function_names": None,
        "top_level_async_function_names": None,
    }
    if candidate_id is None:
        return diagnostics

    for facts in trace.step_facts.values():
        survivors = facts.get("survivors")
        if not isinstance(survivors, list):
            continue
        for survivor in survivors:
            if not isinstance(survivor, Mapping):
                continue
            if survivor.get("candidate_id") != candidate_id:
                continue
            for field_name in _CANDIDATE_FACT_FIELDS:
                if field_name in survivor:
                    diagnostics[field_name] = survivor[field_name]
    return diagnostics


def _relation_rows(
    projected: ProjectedArtifacts,
) -> tuple[tuple[str, list[dict[str, object]]], ...]:
    return (
        ("results", projected.results),
        ("candidates", projected.candidates),
        ("step_facts", projected.step_facts),
        ("rejections", projected.rejections),
    )


def _combine_relation(
    root: Path,
    part_ids: tuple[str, ...],
    relation_name: str,
    schema: pa.Schema,
    destination: Path,
) -> int:
    count = 0
    temporary_destination = destination.with_name(f".{destination.name}.tmp")
    with pq.ParquetWriter(
        temporary_destination,
        schema,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
    ) as writer:
        for part_id in part_ids:
            source = root / "parts" / part_id / f"{relation_name}.parquet"
            if not source.is_file():
                raise FileNotFoundError(
                    f"projected relation part not found: {source}"
                )
            parquet_file = pq.ParquetFile(source)
            if not parquet_file.schema_arrow.equals(schema):
                raise ValueError(
                    f"unexpected schema in projected part: {source}"
                )
            for batch in parquet_file.iter_batches(
                batch_size=_ROW_GROUP_SIZE,
                columns=schema.names,
            ):
                table = pa.Table.from_batches([batch], schema=schema)
                writer.write_table(table, row_group_size=_ROW_GROUP_SIZE)
                count += batch.num_rows
    os.replace(temporary_destination, destination)
    return count


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _validated_part_id(part_id: str) -> str:
    if not part_id or part_id in {".", ".."} or Path(part_id).name != part_id:
        raise ValueError("part_id must be a single non-empty path component")
    return part_id


__all__ = (
    "CANDIDATES_SCHEMA",
    "AtomicProjectedPartWriter",
    "PROJECTED_ARTIFACT_SCHEMAS",
    "REJECTIONS_SCHEMA",
    "RESULTS_SCHEMA",
    "STEP_FACTS_SCHEMA",
    "ProjectedArtifacts",
    "ProjectedPart",
    "combine_projected_parts",
    "project_preprocessing_result",
    "write_projected_part",
)
